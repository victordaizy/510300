"""冻结后唯一一次运行2015起点敏感性的事件门与条件式组合回测。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from research.macro_stress_avoidance_2015_v2 import (  # noqa: E402
    CONFIG_PATH,
    EVIDENCE_CLASS,
    ROOT,
    evaluate_protocol,
    json_default,
    load_config,
    markdown_report,
    sha256_file,
)


MANIFEST_PATH = ROOT / "config" / "510300_macro_stress_avoidance_2015_v2_manifest.json"


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2, default=json_default)
            file.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as file:
            file.write(content)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f"{path.stem}.", suffix=".tmp.parquet", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify_manifest(manifest: dict[str, Any]) -> None:
    """验证冻结清单自身及所有实现、输入哈希。"""

    recorded = manifest.get("manifest_content_sha256")
    payload = dict(manifest)
    payload.pop("manifest_content_sha256", None)
    if recorded != canonical_hash(payload):
        raise ValueError("冻结清单内容哈希不匹配")
    if manifest.get("status") != "FROZEN_BEFORE_FIRST_EVENT_RUN":
        raise ValueError("冻结清单状态不允许首次运行")
    if bool(manifest.get("historical_run_completed")):
        raise RuntimeError("历史事件研究已执行，禁止第二次运行")
    if manifest.get("evidence_class") != EVIDENCE_CLASS:
        raise ValueError("冻结清单证据等级不匹配")
    if manifest.get("original_2021_rejection_overridden") is not False:
        raise ValueError("冻结清单不得覆盖原拒绝")
    if sha256_file(CONFIG_PATH) != manifest["protocol_sha256"]:
        raise ValueError("冻结协议发生漂移")
    for relative, expected in manifest["source_artifacts"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"冻结实现发生漂移：{relative}，实际{actual}")
    for relative, expected in manifest["input_artifacts"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"冻结输入发生漂移：{relative}，实际{actual}")
    predecessor = manifest["predecessor"]
    if sha256_file(ROOT / predecessor["result_file"]) != predecessor["result_sha256"]:
        raise ValueError("前序拒绝结果发生漂移")
    if sha256_file(ROOT / predecessor["manifest_file"]) != predecessor["manifest_sha256"]:
        raise ValueError("前序冻结清单发生漂移")


def _rewrite_manifest(manifest: dict[str, Any]) -> None:
    manifest.pop("manifest_content_sha256", None)
    manifest["manifest_content_sha256"] = canonical_hash(manifest)
    _atomic_json(MANIFEST_PATH, manifest)


def _output_paths(config: dict[str, Any]) -> dict[str, Path]:
    return {
        key: ROOT / relative
        for key, relative in config["artifacts"].items()
        if key != "freeze_manifest"
    }


def main() -> int:
    """仅读取一次真实未来收益；事件门失败时不运行组合层。"""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("缺少冻结清单，拒绝读取真实未来收益")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    verify_manifest(manifest)
    config = load_config()
    output_paths = _output_paths(config)
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(f"历史输出已存在，拒绝覆盖：{path}")

    started_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    try:
        report, artifacts = evaluate_protocol(config)
        report["generated_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        written: dict[str, str] = {}
        for key in ("point_in_time_panel", "event_table", "control_table"):
            path = output_paths[key]
            _atomic_parquet(path, artifacts[key])
            written[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        conditional_keys = (
            "five_year_ledger",
            "five_year_trades",
            "three_year_ledger",
            "three_year_trades",
            "double_cost_ledger",
            "double_cost_trades",
            "buy_hold_ledger",
            "buy_hold_trades",
        )
        for key in conditional_keys:
            if key not in artifacts:
                continue
            path = output_paths[key]
            _atomic_parquet(path, artifacts[key])
            written[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        _atomic_json(output_paths["report_json"], report)
        written[output_paths["report_json"].relative_to(ROOT).as_posix()] = sha256_file(
            output_paths["report_json"]
        )
        _atomic_text(output_paths["report_markdown"], markdown_report(report))
        written[
            output_paths["report_markdown"].relative_to(ROOT).as_posix()
        ] = sha256_file(output_paths["report_markdown"])
    except Exception as error:
        manifest.update(
            {
                "status": "PROGRAM_FAILED_AFTER_FROZEN_RETURN_READ",
                "historical_run_completed": True,
                "historical_run_started_at": started_at,
                "historical_run_completed_at": datetime.now(
                    ZoneInfo("Asia/Shanghai")
                ).isoformat(),
                "return_outcomes_read_after_freeze": True,
                "program_error_type": type(error).__name__,
                "program_error": str(error),
                "goal_achieved": False,
                "position_mapping_enabled": False,
                "order_generation_enabled": False,
                "broker_connection_enabled": False,
                "live_trading_authorized": False,
            }
        )
        _rewrite_manifest(manifest)
        raise

    manifest.update(
        {
            "status": "HISTORICAL_EVENT_RUN_COMPLETED",
            "historical_run_completed": True,
            "historical_run_started_at": started_at,
            "historical_run_completed_at": datetime.now(
                ZoneInfo("Asia/Shanghai")
            ).isoformat(),
            "return_outcomes_read_after_freeze": True,
            "event_decision": report["event_study"]["decision"],
            "event_gate_passed": bool(report["event_study"]["passed"]),
            "portfolio_backtest_status": report["portfolio_backtest_status"],
            "return_evaluation": report["return_evaluation"],
            "mechanism_decision_before_evidence_class": report[
                "mechanism_decision_before_evidence_class"
            ],
            "historical_result_status": report["decision"],
            "result_files": written,
            "goal_achieved": False,
            "original_2021_rejection_overridden": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        }
    )
    _rewrite_manifest(manifest)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "evidence_class": EVIDENCE_CLASS,
                "event_decision": report["event_study"]["decision"],
                "independent_complete_primary_events": report["event_study"][
                    "independent_complete_primary_events"
                ],
                "portfolio_backtest_status": report["portfolio_backtest_status"],
                "return_evaluation": report["return_evaluation"],
                "first_model_eligible_dates": report["data_scope"][
                    "first_model_eligible_dates"
                ],
                "report": output_paths["report_markdown"].relative_to(ROOT).as_posix(),
                "goal_achieved": False,
                "live_trading_authorized": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
