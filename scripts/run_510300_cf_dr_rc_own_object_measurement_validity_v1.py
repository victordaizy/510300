"""冻结并运行 CF、DR、RC 自有对象的测量有效性审计。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.cf_dr_rc_own_object_measurement_validity_v1 import (  # noqa: E402
    build_object_measurement_panel,
    summarize_measurement_validity,
)


PROGRAM_ID = "510300_CF_DR_RC_OWN_OBJECT_MEASUREMENT_VALIDITY_V1"
CONFIG_PATH = ROOT / "config/510300_cf_dr_rc_own_object_measurement_validity_v1.yaml"
RUNNER_PATH = Path(__file__).resolve()
MODULE_PATH = ROOT / "research/cf_dr_rc_own_object_measurement_validity_v1.py"
TIMEZONE = ZoneInfo("Asia/Shanghai")


def now_iso() -> str:
    return datetime.now(TIMEZONE).isoformat()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["program"]["program_id"] != PROGRAM_ID:
        raise ValueError("自有对象测量配置的 PROGRAM_ID 不匹配")
    return config


def project_path(relative: str) -> Path:
    return ROOT / Path(str(relative).replace("/", os.sep))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_bytes_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise RuntimeError(f"冻结产物已存在，禁止静默覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json_new(path: Path, value: Any) -> None:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        default=str,
        allow_nan=False,
    )
    _atomic_bytes_new(path, (payload + "\n").encode("utf-8"))


def atomic_text_new(path: Path, value: str) -> None:
    _atomic_bytes_new(path, value.encode("utf-8"))


def atomic_parquet_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise RuntimeError(f"冻结产物已存在，禁止静默覆盖：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def _registered_files(config: dict[str, Any]) -> dict[str, tuple[str, str]]:
    parent = config["parent_contract"]
    source = config["source_contract"]
    return {
        parent["ledger_manifest"]: ("", "FROZEN_COMPONENT_LEDGER_PROTOCOL"),
        parent["ledger_status"]: ("", "FROZEN_COMPONENT_LEDGER_STATUS"),
        parent["ledger_panel"]: (
            parent["expected_ledger_sha256"],
            "FROZEN_COMPONENT_LEDGER_PANEL",
        ),
        parent["ledger_coverage"]: (
            parent["expected_coverage_sha256"],
            "FROZEN_COMPONENT_LEDGER_COVERAGE",
        ),
        source["cf_state"]["path"]: (
            source["cf_state"]["expected_sha256"],
            "CF_TARGET_MEASUREMENT_STATE",
        ),
        source["present_value_state"]["path"]: (
            source["present_value_state"]["expected_sha256"],
            "DR_TARGET_MEASUREMENT_STATE",
        ),
        source["risk_capacity_daily"]["path"]: (
            source["risk_capacity_daily"]["expected_sha256"],
            "RC_TARGET_MEASUREMENT_STATE",
        ),
        source["total_return_index"]["path"]: (
            source["total_return_index"]["expected_sha256"],
            "RC_REALIZED_RISK_TARGET_SOURCE",
        ),
    }


def validate_inputs(config: dict[str, Any]) -> dict[str, Any]:
    parent = config["parent_contract"]
    manifest = json.loads(
        project_path(parent["ledger_manifest"]).read_text(encoding="utf-8")
    )
    status = json.loads(
        project_path(parent["ledger_status"]).read_text(encoding="utf-8")
    )
    if manifest.get("manifest_payload_sha256") != parent[
        "expected_manifest_payload_sha256"
    ]:
        raise ValueError("成分账本协议哈希不一致")
    if status.get("status_payload_sha256") != parent[
        "expected_status_payload_sha256"
    ]:
        raise ValueError("成分账本状态哈希不一致")
    if status.get("status") != parent["expected_status"]:
        raise ValueError("成分账本状态不允许进入自有对象测量")
    for relative, (expected, label) in _registered_files(config).items():
        path = project_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"{label} 缺失：{path}")
        if expected and sha256_file(path) != expected:
            raise ValueError(f"{label} 哈希不一致")
    return status


def _output_paths(config: dict[str, Any]) -> list[Path]:
    return [
        project_path(value)
        for key, value in config["artifacts"].items()
        if key != "protocol_manifest"
    ]


def freeze_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if manifest_path.exists():
        raise RuntimeError(f"自有对象测量冻结清单已存在：{manifest_path}")
    existing = [str(path) for path in _output_paths(config) if path.exists()]
    if existing:
        raise RuntimeError(f"冻结前发现同名自有对象测量输出：{existing}")
    parent_status = validate_inputs(config)
    registered: dict[str, Any] = {}
    for relative, (_, role) in _registered_files(config).items():
        path = project_path(relative)
        registered[Path(relative).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "role": role,
        }
    manifest = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "protocol_revision": config["program"]["protocol_revision"],
        "status": "FROZEN_BEFORE_OWN_OBJECT_TARGET_VALUE_READ",
        "frozen_at": now_iso(),
        "config_file": CONFIG_PATH.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(CONFIG_PATH),
        "config_canonical_sha256": canonical_hash(config),
        "implementation_files": {
            RUNNER_PATH.relative_to(ROOT).as_posix(): sha256_file(RUNNER_PATH),
            MODULE_PATH.relative_to(ROOT).as_posix(): sha256_file(MODULE_PATH),
        },
        "registered_inputs": registered,
        "parent_status_payload_sha256": parent_status["status_payload_sha256"],
        "objects_clocks_and_coverage_gates_frozen": True,
        "direct_total_return_target_allowed": False,
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    manifest["manifest_payload_sha256"] = canonical_hash(manifest)
    atomic_json_new(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return manifest


def verify_protocol(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if not manifest_path.is_file():
        raise FileNotFoundError("自有对象测量协议尚未冻结")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.get("manifest_payload_sha256")
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    if canonical_hash(payload) != expected_hash:
        raise ValueError("自有对象测量清单自身哈希失败")
    if manifest.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise ValueError("自有对象测量配置在冻结后发生漂移")
    if manifest.get("config_canonical_sha256") != canonical_hash(config):
        raise ValueError("自有对象测量配置语义在冻结后发生漂移")
    for relative, expected in manifest["implementation_files"].items():
        if sha256_file(project_path(relative)) != expected:
            raise ValueError(f"自有对象测量实现发生漂移：{relative}")
    for relative, metadata in manifest["registered_inputs"].items():
        path = project_path(relative)
        if not path.is_file() or sha256_file(path) != metadata["sha256"]:
            raise ValueError(f"自有对象测量输入发生漂移或缺失：{relative}")
    validate_inputs(config)
    return manifest


def _module_summary(summary: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module_id, group in summary.groupby("module_id", sort=True):
        statuses = set(group["measurement_validity_status"])
        if statuses == {"PASS_OWN_OBJECT_MEASUREMENT_VALIDITY"}:
            status = "PASS_ALL_REGISTERED_OBJECT_MEASUREMENTS"
        elif any(value.startswith("NO_VIEW") for value in statuses):
            status = "NO_VIEW_OR_MIXED_REGISTERED_OBJECT_MEASUREMENTS"
        else:
            status = "PARTIAL_REGISTERED_OBJECT_MEASUREMENTS"
        rows.append(
            {
                "module_id": module_id,
                "registered_object_horizon_count": int(len(group)),
                "measurement_status": status,
                "object_status_counts": {
                    str(key): int(value)
                    for key, value in group[
                        "measurement_validity_status"
                    ].value_counts().items()
                },
            }
        )
    return rows


def _build_report(
    manifest: dict[str, Any],
    panel: pd.DataFrame,
    summary: pd.DataFrame,
    modules: list[dict[str, Any]],
) -> str:
    lines = [
        "# 510300_CF_DR_RC_OWN_OBJECT_MEASUREMENT_VALIDITY_V1",
        "",
        "## 裁决",
        "",
        "九个预注册自有对象的历史测量有效性已经检查。本阶段不拟合模型、不预测总回报、不计算组合收益。",
        "",
        f"- 协议哈希：`{manifest['manifest_payload_sha256']}`",
        f"- 对象测量：`{len(panel)}` 行；对象×horizon 汇总：`{len(summary)}` 行。",
        f"- 模块结论：`{json.dumps(modules, ensure_ascii=False, sort_keys=True)}`",
        "- `RETURN_PREDICTION_ALLOWED=false`，`PORTFOLIO_EVALUATION_ALLOWED=false`。",
        "",
        "## 对象覆盖",
        "",
        "| 模块 | 对象 | horizon | origin | 已测量 | PASS | PARTIAL | NO_VIEW | 可用率 | 年份 | 裁决 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.module_id} | {row.object_id} | {int(row.horizon_market_days)}D | {int(row.origin_count)} | {int(row.observed_count)} | {int(row.pass_count)} | {int(row.partial_count)} | {int(row.no_view_count)} | {row.observed_share:.2%} | {int(row.distinct_calendar_years)} | {row.measurement_validity_status} |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- CF 的盈利、经营现金流和 breadth 具有足够历史覆盖时仍最多为 `PARTIAL`，因为首次披露日期虽被用作主时钟，但数值来自可能含后修订的二级聚合历史，且成分权重是市值代理。",
            "- DR 的固定 cohort 倍数变化和 ERP gap 变化同样受财务数值版本与代理权重限制，因此不得提升为完全 PIT 测量。",
            "- RC 的实现波动、下行半方差、成员相关性和 ETF Amihud 冲击来自已发生市场路径；它们可以作为风险容量对象，但仍不是未来回报标签。",
            "- 任一 `NO_VIEW` 保持空值，不用插值、当前成分或未来修订值补齐。",
            "",
            "## 下一步边界",
            "",
            "只有测量为 PASS/PARTIAL 的对象可进入另行冻结的自有对象 walk-forward 诊断；NO_VIEW 必须逐 origin 传播。直接总回报目标、组合评估、仓位映射仍被禁止。当前为 `ABSTAIN / POSITION_UNSET`。",
            "",
        ]
    )
    return "\n".join(lines)


def run_measurement(config: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    validate_inputs(config)
    parent = config["parent_contract"]
    source = config["source_contract"]
    ledger = pd.read_parquet(project_path(parent["ledger_panel"]))
    coverage = pd.read_parquet(project_path(parent["ledger_coverage"]))
    cf = pd.read_parquet(project_path(source["cf_state"]["path"]))
    pv = pd.read_parquet(project_path(source["present_value_state"]["path"]))
    risk = pd.read_parquet(project_path(source["risk_capacity_daily"]["path"]))
    total = pd.read_parquet(project_path(source["total_return_index"]["path"]))
    cf_contract = config["object_contracts"]["CF"]
    rc_contract = config["object_contracts"]["RC"]
    panel = build_object_measurement_panel(
        ledger=ledger,
        coverage=coverage,
        cf_state=cf,
        present_value_state=pv,
        risk_capacity_daily=risk,
        total_return_index=total,
        minimum_cf_weight_coverage=float(
            cf_contract["OPERATING_CASHFLOW_GROWTH"]["minimum_weight_coverage"]
        ),
        annualization_days=int(
            rc_contract["REALIZED_VOLATILITY"]["annualization_days"]
        ),
    )
    gates = config["measurement_gates"]
    summary = summarize_measurement_validity(
        panel,
        minimum_observed_share=float(gates["minimum_object_observed_share"]),
        minimum_pass_share=float(gates["minimum_object_pass_share"]),
        minimum_distinct_years=int(gates["minimum_distinct_calendar_years"]),
    )
    modules = _module_summary(summary)
    artifacts = config["artifacts"]
    paths = {name: project_path(relative) for name, relative in artifacts.items()}
    pending = [path for name, path in paths.items() if name != "protocol_manifest" and path.exists()]
    if pending:
        raise RuntimeError(f"自有对象测量输出已存在，禁止覆盖：{pending}")
    atomic_parquet_new(paths["object_measurement_panel"], panel)
    atomic_parquet_new(paths["object_summary"], summary)
    payload = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": "OWN_OBJECT_MEASUREMENT_VALIDITY_COMPLETE_NO_FORECAST",
        "created_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "module_summary": modules,
        "object_summary": dataframe_records(summary),
        "measurement_status_counts": {
            str(key): int(value)
            for key, value in panel["measurement_status"].value_counts().items()
        },
        "measurement_records": dataframe_records(panel),
        "direct_total_return_target_allowed": False,
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
    }
    payload["payload_sha256"] = canonical_hash(payload)
    atomic_json_new(paths["measurement_json"], payload)
    report = _build_report(manifest, panel, summary, modules)
    atomic_text_new(paths["final_report"], report)
    status = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "status": "OWN_OBJECT_MEASUREMENT_VALIDITY_COMPLETE_NO_FORECAST",
        "completed_at": now_iso(),
        "protocol_manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "object_measurement_panel": {
            "path": paths["object_measurement_panel"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(paths["object_measurement_panel"]),
            "rows": len(panel),
            "module_counts": {
                str(key): int(value)
                for key, value in panel["module_id"].value_counts().items()
            },
        },
        "object_summary": {
            "path": paths["object_summary"].relative_to(ROOT).as_posix(),
            "sha256": sha256_file(paths["object_summary"]),
            "rows": len(summary),
            "validity_status_counts": {
                str(key): int(value)
                for key, value in summary[
                    "measurement_validity_status"
                ].value_counts().items()
            },
        },
        "module_summary": modules,
        "next_research_action": "PREREGISTER_OWN_OBJECT_WALK_FORWARD_DIAGNOSTIC_WITH_NO_VIEW_PROPAGATION",
        "current_forecast_specification": "REJECTED_FROZEN",
        "portfolio_action": "ABSTAIN",
        "position_state": "POSITION_UNSET",
        "current_investable_strategy": "NONE",
        "direct_total_return_target_allowed": False,
        "return_prediction_allowed": False,
        "portfolio_evaluation_allowed": False,
        "model_position_target": "UNSET",
        "nav_calculated": False,
        "sharpe_calculated": False,
        "order_generation_allowed": False,
        "paper_shadow_authorized": False,
        "live_authorized": False,
    }
    status["status_payload_sha256"] = canonical_hash(status)
    atomic_json_new(paths["authoritative_status"], status)
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 CF/DR/RC 自有对象测量有效性审计")
    parser.add_argument("--phase", choices=["freeze", "measure", "all", "verify"], default="all")
    args = parser.parse_args()
    config = load_config()
    manifest_path = project_path(config["artifacts"]["protocol_manifest"])
    if args.phase == "freeze":
        freeze_protocol(config)
        return 0
    if args.phase == "all" and not manifest_path.exists():
        freeze_protocol(config)
    manifest = verify_protocol(config)
    if args.phase in {"measure", "all"}:
        run_measurement(config, manifest)
    if args.phase == "verify":
        print("自有对象测量协议与全部冻结输入校验通过。", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
