"""运行行业预期差独立原点与结果输入监控。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.industry_expectation_gap_forward_evaluation import (  # noqa: E402
    verify_hash_manifest,
)
from research.industry_expectation_gap_forward_operations import (  # noqa: E402
    ForwardOperationsError,
    assess_next_origin_gate,
    assess_outcome_input_gate,
    validate_operations_config,
    validate_origin_ledgers,
)


CONFIG_PATH = ROOT / "config" / "industry_expectation_gap_forward_operations_v1.yaml"
LOCAL_TIMEZONE = ZoneInfo("Asia/Shanghai")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ForwardOperationsError(f"YAML顶层必须是对象：{path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ForwardOperationsError(f"JSON顶层必须是对象：{path}")
    return value


def _path(relative: str) -> Path:
    root = ROOT.resolve()
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ForwardOperationsError(f"路径越出工作区：{relative}") from exc
    return result


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _verify_start_audit(path: Path) -> dict[str, Any]:
    audit = _load_json(path)
    if audit.get("decision") != "ELIGIBLE_FOR_STRICT_FORWARD_COLLECTION_FROM_2026-08-19":
        raise ForwardOperationsError("前瞻启动审计未授权严格前瞻收集")
    for entry in audit.get("frozen_evidence", []):
        evidence_path = _path(str(entry.get("path")))
        if not evidence_path.exists():
            raise ForwardOperationsError(f"冻结证据缺失：{evidence_path}")
        if _sha256(evidence_path) != str(entry.get("sha256", "")).lower():
            raise ForwardOperationsError(f"冻结证据哈希变化：{evidence_path}")
    authorization = audit.get("authorization", {})
    for field in (
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_enabled",
    ):
        if authorization.get(field) is not False:
            raise ForwardOperationsError(f"启动审计安全开关异常：{field}")
    return audit


def _latest_parquet_date(path: Path, column: str = "date") -> str:
    if not path.exists():
        raise ForwardOperationsError(f"输入文件缺失：{path}")
    frame = pd.read_parquet(path, columns=[column])
    values = pd.to_datetime(frame[column], errors="coerce").dropna()
    if values.empty:
        raise ForwardOperationsError(f"输入日期列为空：{path}:{column}")
    return values.max().date().isoformat()


def _parse_as_of(value: str | None) -> datetime:
    if value is None:
        return datetime.now(LOCAL_TIMEZONE)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TIMEZONE)
    return parsed.astimezone(LOCAL_TIMEZONE)


def _expected_closed_trading_date(
    calendar_path: Path,
    as_of: datetime,
    market_close_confirmation_time: str,
) -> str:
    calendar = pd.read_csv(calendar_path)
    if "trade_date" not in calendar.columns:
        raise ForwardOperationsError("交易日历缺少trade_date列")
    dates = pd.to_datetime(calendar["trade_date"], errors="coerce").dropna().dt.normalize()
    cutoff = time.fromisoformat(market_close_confirmation_time)
    today = pd.Timestamp(as_of.date())
    if as_of.timetz().replace(tzinfo=None) < cutoff:
        eligible = dates[dates < today]
    else:
        eligible = dates[dates <= today]
    if eligible.empty:
        raise ForwardOperationsError("交易日历没有已收盘日期")
    return eligible.max().date().isoformat()


def _run_python_capture(relative_script: str) -> dict[str, Any]:
    script = _path(relative_script)
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    record = {
        "script": relative_script,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    return record


def _run_python(relative_script: str) -> dict[str, Any]:
    record = _run_python_capture(relative_script)
    if record["exit_code"] != 0:
        stderr = record["stderr"]
        stdout = record["stdout"]
        detail = stderr or stdout or "无错误输出"
        raise ForwardOperationsError(
            f"运行失败：{relative_script}，退出码{record['exit_code']}，{detail}"
        )
    return record


def _markdown(report: dict[str, Any]) -> str:
    origin = report["origin_collection"]
    outcome = report["outcome_collection"]
    maturity = report["maturity"]
    lines = [
        "# 行业预期差前瞻运行状态",
        "",
        f"- 总状态：`{report['status']}`",
        f"- 运行健康：`{report['operations_health']}`",
        f"- 生成时间：`{report['generated_at']}`",
        f"- 独立预测原点：{maturity['origin_cluster_count']} 个",
        f"- 成熟原点：{maturity['mature_origin_cluster_count']} 个",
        f"- 非重叠60日块：{maturity['non_overlapping_60d_block_count']} 个",
        f"- 下一原点闸门：`{origin['status']}`",
        f"- 结果输入闸门：`{outcome['status']}`",
        f"- 共同结果数据截止：`{outcome['common_latest_date']}`",
        f"- 最近应覆盖交易日：`{outcome['expected_latest_closed_trading_date']}`",
        "",
        "## 当前阻断",
        "",
    ]
    blockers = origin["blockers"] + outcome["blockers"]
    if report["evaluation_run"].get("reason") == "EVALUATION_FAILED_CLOSED":
        blockers.append("EVALUATION_FAILED_CLOSED")
    if blockers:
        lines.extend(f"- `{item}`" for item in blockers)
    else:
        lines.append("- 无运行阻断；仅允许刷新追加式评价。")
    lines.extend(
        [
            "",
            "## 治理边界",
            "",
            "- `prediction_date` 才是独立时间样本；同日行业行不重复计数。",
            "- 新原点必须进入新月份，并同时取得更新后的来源注册表、行业点时面板和至少两个新官方发布。",
            "- 新原点仍需人工证据包；程序不会复制旧判断或自动生成行业预测。",
            "- 未成熟期不输出部分收益，原 `NO_VIEW` 不得改写。",
            "- 仓位映射、订单生成、券商连接和实盘均关闭。",
            "",
        ]
    )
    return "\n".join(lines)


def build_status(config: dict[str, Any], as_of: datetime, evaluate_if_ready: bool) -> dict[str, Any]:
    validate_operations_config(config)
    parent = config["frozen_parent"]
    start_audit_path = _path(parent["start_audit_manifest"])
    start_audit = _verify_start_audit(start_audit_path)
    verify_hash_manifest(ROOT, parent["evaluation_manifest"])
    verify_hash_manifest(ROOT, parent["recovery_execution_manifest"])

    source_registry_path = _path(config["inputs"]["source_registry"])
    source_registry = _load_yaml(source_registry_path)
    ledger_paths = sorted(ROOT.glob(config["inputs"]["origin_ledger_glob"]))
    ledgers = [_load_json(path) for path in ledger_paths]
    origin_settings = config["origin_collection"]
    origins = validate_origin_ledgers(
        ledgers,
        source_registry,
        minimum_sources_per_industry=int(origin_settings["minimum_sources_per_industry"]),
    )

    sector_path = _path(config["inputs"]["sector_point_in_time_panel"])
    constituent_path = _path(config["inputs"]["constituent_total_return_daily"])
    etf_path = _path(config["inputs"]["etf_total_return_daily"])
    sector_latest = _latest_parquet_date(sector_path)
    constituent_latest = _latest_parquet_date(constituent_path)
    etf_latest = _latest_parquet_date(etf_path)
    expected_latest = _expected_closed_trading_date(
        _path(config["inputs"]["trading_calendar"]),
        as_of,
        config["outcome_collection"]["market_close_confirmation_time"],
    )

    origin_gate = assess_next_origin_gate(
        origins,
        source_registry,
        sector_panel_latest_date=sector_latest,
        as_of=as_of.isoformat(),
        minimum_new_official_releases=int(
            origin_settings["minimum_new_official_releases"]
        ),
    )
    evaluation_config = _load_yaml(_path(parent["evaluation_config"]))
    outcome_gate = assess_outcome_input_gate(
        entry_date=evaluation_config["entry_date"],
        constituent_latest_date=constituent_latest,
        etf_latest_date=etf_latest,
        expected_latest_trading_date=expected_latest,
    )

    evaluation_run: dict[str, Any] = {
        "requested": evaluate_if_ready,
        "executed": False,
        "reason": "OUTCOME_INPUT_GATE_NOT_READY",
    }
    if evaluate_if_ready and outcome_gate["ready"]:
        evaluation_run = _run_python_capture(parent["evaluation_runner"])
        evaluation_run["requested"] = True
        evaluation_run["attempted"] = True
        evaluation_run["executed"] = evaluation_run["exit_code"] == 0
        evaluation_run["reason"] = (
            "INPUT_GATE_READY"
            if evaluation_run["exit_code"] == 0
            else "EVALUATION_FAILED_CLOSED"
        )
    maturity_run = _run_python(parent["maturity_runner"])
    maturity = _load_json(_path(config["inputs"]["maturity_status"]))

    status = str(maturity.get("status", "COLLECTING_FORWARD"))
    operations_health = (
        "FAILED_EVALUATION_CLOSED"
        if evaluation_run.get("reason") == "EVALUATION_FAILED_CLOSED"
        else "PASS"
    )
    return {
        "operations_version": config["version"],
        "generated_at": as_of.isoformat(),
        "status": status,
        "operations_health": operations_health,
        "research_mode": "DISCOVERY_ONLY",
        "view_status": "NO_VIEW",
        "origin_collection": origin_gate,
        "outcome_collection": outcome_gate,
        "evaluation_run": evaluation_run,
        "maturity_run": maturity_run,
        "maturity": maturity,
        "evidence": {
            "strict_forward_start_audit": start_audit["audit_id"],
            "strict_forward_start_decision": start_audit["decision"],
            "recovery_execution_manifest": {
                "path": parent["recovery_execution_manifest"],
                "sha256": _sha256(_path(parent["recovery_execution_manifest"])),
            },
            "source_registry": {
                "path": _relative(source_registry_path),
                "sha256": _sha256(source_registry_path),
            },
            "origin_ledgers": [
                {"path": _relative(path), "sha256": _sha256(path)} for path in ledger_paths
            ],
            "sector_point_in_time_panel": {
                "path": _relative(sector_path),
                "sha256": _sha256(sector_path),
                "latest_date": sector_latest,
            },
            "constituent_total_return_daily": {
                "path": _relative(constituent_path),
                "sha256": _sha256(constituent_path),
                "latest_date": constituent_latest,
            },
            "etf_total_return_daily": {
                "path": _relative(etf_path),
                "sha256": _sha256(etf_path),
                "latest_date": etf_latest,
            },
        },
        "automation": {
            "existing_origin_evaluation_enabled_when_gate_ready": True,
            "automatic_new_origin_generation_enabled": False,
        },
        "safety": dict(config["safety"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="行业预期差前瞻运行监控")
    parser.add_argument("--as-of", help="可复现运行时点，默认上海当前时间")
    parser.add_argument(
        "--evaluate-if-ready",
        action="store_true",
        help="仅当结果数据闸门通过时刷新冻结原点的追加式评价",
    )
    args = parser.parse_args()
    config = _load_yaml(CONFIG_PATH)
    as_of = _parse_as_of(args.as_of)
    report = build_status(config, as_of, args.evaluate_if_ready)
    json_path = _path(config["outputs"]["status_json"])
    markdown_path = _path(config["outputs"]["status_markdown"])
    _atomic_write(json_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    _atomic_write(markdown_path, _markdown(report))
    print(f"行业预期差运行状态：{report['status']}")
    print(f"下一原点：{report['origin_collection']['status']}")
    print(f"结果输入：{report['outcome_collection']['status']}")
    print(f"状态文件：{json_path}")
    return 1 if report["operations_health"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
