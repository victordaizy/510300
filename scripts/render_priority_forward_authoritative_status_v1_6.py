"""自动生成零付费优先前瞻研究的唯一权威状态。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "priority_forward_authoritative_status_v1_6.yaml"


def _path(relative: str) -> Path:
    root = ROOT.resolve()
    value = (root / relative).resolve()
    value.relative_to(root)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _evidence(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "path": path.relative_to(ROOT).as_posix(),
            "exists": False,
            "sha256": None,
            "bytes": None,
        }
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "exists": True,
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _json(path: Path, *, required: bool) -> dict[str, Any] | None:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"缺少权威状态输入：{path}")
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def _latest_task_status(
    current: Path,
    receipt_glob: str,
) -> tuple[Path | None, dict[str, Any] | None]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    if current.is_file():
        payload = _json(current, required=True)
        assert payload is not None
        candidates.append((current, payload))
    for path in ROOT.glob(receipt_glob):
        payload = _json(path, required=True)
        assert payload is not None
        candidates.append((path, payload))
    if not candidates:
        return None, None

    def sort_key(item: tuple[Path, dict[str, Any]]) -> tuple[str, int, str]:
        path, payload = item
        timestamp = str(
            payload.get("ended_at")
            or payload.get("generated_at")
            or payload.get("started_at")
            or ""
        )
        return (
            timestamp,
            1 if payload.get("immutable_receipt") is True else 0,
            path.as_posix(),
        )

    return max(candidates, key=sort_key)


def _registry(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("免费来源登记表为空")
    source_ids = [row["source_id"] for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("免费来源登记表 source_id 不唯一")
    nonzero = [
        row["source_id"]
        for row in rows
        if float(row.get("access_cost_cny", "0") or 0) != 0
    ]
    second = next(
        (
            row
            for row in rows
            if row["source_id"] == "SSE_ETF_DAILY_TURNOVER_OFFICIAL"
        ),
        None,
    )
    return rows, {
        **_evidence(path),
        "row_count": len(rows),
        "unique_source_id_count": len(set(source_ids)),
        "nonzero_cost_source_ids": nonzero,
        "second_official_endpoint": second,
    }


def _optional_gate(
    path: Path,
    accepted_statuses: set[str],
    fields: tuple[str, ...],
) -> dict[str, Any]:
    payload = _json(path, required=False)
    status = None
    if payload is not None:
        for field in fields:
            if payload.get(field) is not None:
                status = str(payload[field])
                break
    return {
        **_evidence(path),
        "status": status or "PENDING_EVIDENCE",
        "passed": status in accepted_statuses,
    }


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("权威状态配置顶层必须是对象")
    contract = value["contract"]
    streams = contract["active_research_streams"]
    if int(contract["data_purchase_budget_cny"]) != 0:
        raise ValueError("数据采购预算必须为 0 CNY")
    if len(streams) > int(contract["maximum_active_research_streams"]):
        raise ValueError("活动研究流超过冻结上限")
    if len({stream["id"] for stream in streams}) != len(streams):
        raise ValueError("活动研究流 ID 不唯一")
    return value


def build_report(config: dict[str, Any], generated_at: datetime) -> dict[str, Any]:
    inputs = config["inputs"]
    contract = config["contract"]
    registry_rows, registry = _registry(_path(inputs["source_registry"]))
    readiness = _json(_path(inputs["primary_market_readiness"]), required=True)
    industry = _json(_path(inputs["industry_operations_status"]), required=True)
    cash = _json(_path(inputs["cash_option_admission_audit"]), required=True)
    cash_terminal = _json(_path(inputs["final_cash_source_repair"]), required=False)
    assert readiness is not None and industry is not None and cash is not None

    primary_task_path, primary_task = _latest_task_status(
        _path(inputs["primary_market_current_task_status"]),
        inputs["primary_market_task_receipt_glob"],
    )
    industry_task_path, industry_task = _latest_task_status(
        _path(inputs["industry_current_task_status"]),
        inputs["industry_task_receipt_glob"],
    )
    second_endpoint = registry["second_official_endpoint"]
    second_endpoint_admitted = bool(
        second_endpoint
        and second_endpoint.get("admission_status")
        == "ADMITTED_FORWARD_OFFICIAL_SECOND_ENDPOINT"
        and float(second_endpoint.get("access_cost_cny", "0") or 0) == 0
    )

    maturity = industry["maturity"]
    primary_full_days = int(readiness["full_coverage_days"])
    crosscheck_rows = int(
        readiness.get("lineage", {}).get("v1_2_daily_crosscheck_rows", 0)
    )
    active_streams = [
        {
            "id": "PRIMARY_MARKET_PCF_IOPV",
            "priority": 1,
            "status": readiness["status"],
            "stage": readiness["stage"],
            "latest_observed_trade_date": readiness["last_observed_trade_date"],
            "observed_trading_days": int(readiness["observed_trading_days"]),
            "full_quality_days": primary_full_days,
            "quality_day_gate": 20,
            "feature_freeze_gate": 40,
            "first_unseen_gate": 80,
            "replication_gate": 120,
            "v1_2_official_crosscheck_rows": crosscheck_rows,
            "latest_task": {
                "path": (
                    primary_task_path.relative_to(ROOT).as_posix()
                    if primary_task_path
                    else None
                ),
                "run_status": primary_task.get("run_status") if primary_task else None,
                "collection_status": (
                    primary_task.get("collection_status") if primary_task else None
                ),
                "error": primary_task.get("error") if primary_task else None,
            },
            "next_action": "CONTINUE_STRICT_FORWARD_COLLECTION",
        },
        {
            "id": "INDUSTRY_EXPECTATION_GAP",
            "priority": 2,
            "status": industry["status"],
            "origin_collection_status": industry["origin_collection"]["status"],
            "outcome_collection_status": industry["outcome_collection"]["status"],
            "origin_cluster_count": int(maturity["origin_cluster_count"]),
            "mature_origin_cluster_count": int(
                maturity["mature_origin_cluster_count"]
            ),
            "non_overlapping_60d_block_count": int(
                maturity["non_overlapping_60d_block_count"]
            ),
            "calibration_eligible": bool(maturity["calibration"]["eligible"]),
            "model_comparison_eligible": bool(
                maturity["model_comparison"]["eligible"]
            ),
            "latest_task": {
                "path": (
                    industry_task_path.relative_to(ROOT).as_posix()
                    if industry_task_path
                    else None
                ),
                "run_status": industry_task.get("run_status") if industry_task else None,
                "collection_status": (
                    industry_task.get("outcome_refresh_collection_status")
                    if industry_task
                    else None
                ),
                "error": industry_task.get("error") if industry_task else None,
            },
            "next_action": "WAIT_FOR_NEW_OFFICIAL_PACKAGE_AND_CURRENT_OUTCOME_INPUTS",
        },
    ]

    scheduler = _optional_gate(
        _path(inputs["scheduler_logout_reboot_validation"]),
        {"PASS_LOGOUT_AND_REBOOT_PERSISTENCE_VERIFIED"},
        ("validation_status", "status", "deployment_status"),
    )
    frozen_inputs = _optional_gate(
        _path(inputs["frozen_input_audit"]),
        {"PASS_FROZEN_INPUTS_CONTENT_ADDRESSED_AND_LATEST_ISOLATED"},
        ("audit_status", "status"),
    )
    option_orderbook = _optional_gate(
        _path(inputs["option_orderbook_qualification"]),
        {
            "PASS_ZERO_COST_SOURCE_QUALIFIED_FOR_20D_TRIAL",
            "BLOCKED_ZERO_COST_SOURCE_UNAVAILABLE",
        },
        ("qualification_status", "status", "audit_status"),
    )
    cash_repair = _optional_gate(
        _path(inputs["final_cash_source_repair"]),
        {
            "PASS_DATA_GATE",
            "NO_VIEW_FREE_DATA_INSUFFICIENT",
        },
        ("final_status", "status", "audit_status"),
    )
    manifest = _optional_gate(
        _path(inputs["current_manifest"]),
        {
            "FROZEN_ZERO_PAID_FOUNDATION_ACTIVE_AWAITING_NEXT_WINDOW",
            "FROZEN_ZERO_PAID_FOUNDATION_V1_6_ACTIVE",
        },
        ("status",),
    )

    cash_summary = cash["cash_option_admission"]
    cash_terminal_decision = (
        cash_terminal.get("terminal_decision", {}) if cash_terminal is not None else {}
    )
    cash_terminal_status = (
        str(cash_terminal.get("final_status") or cash_terminal.get("status"))
        if cash_terminal is not None
        else str(cash_summary["current_status"])
    )
    foundation_gates = {
        "zero_purchase_budget": {
            "passed": int(contract["data_purchase_budget_cny"]) == 0
            and not registry["nonzero_cost_source_ids"],
            "budget_cny": int(contract["data_purchase_budget_cny"]),
            "lock_days": int(contract["zero_purchase_lock_days"]),
        },
        "active_stream_cap": {
            "passed": len(active_streams)
            <= int(contract["maximum_active_research_streams"]),
            "active_count": len(active_streams),
            "maximum": int(contract["maximum_active_research_streams"]),
        },
        "second_official_etf_endpoint": {
            "passed": second_endpoint_admitted,
            "source_id": (
                second_endpoint.get("source_id") if second_endpoint else None
            ),
            "independent_vendor": False,
            "real_forward_quality_day_observed": crosscheck_rows > 0,
        },
        "authoritative_status_generation": {"passed": True},
        "windows_logout_reboot_persistence": scheduler,
        "frozen_input_content_addressing": frozen_inputs,
        "option_orderbook_zero_cost_qualification": option_orderbook,
        "cash_option_final_free_source_repair": cash_repair,
        "v1_6_manifest_frozen": manifest,
    }
    pending_gate_ids = [
        gate_id
        for gate_id, gate in foundation_gates.items()
        if gate.get("passed") is not True
    ]

    alerts: list[dict[str, str]] = []
    if crosscheck_rows == 0:
        alerts.append(
            {
                "code": "PCF_IOPV_V1_2_REAL_DAY_PENDING",
                "message": "第二官方端点已实现，但尚无 V1.2 真实完整质量日。",
            }
        )
    if industry["outcome_collection"].get("ready") is not True:
        alerts.append(
            {
                "code": str(industry["outcome_collection"]["status"]),
                "message": "行业预期差结果输入尚未达到日期闸门。",
            }
        )
    for gate_id in pending_gate_ids:
        if gate_id in {
            "windows_logout_reboot_persistence",
            "frozen_input_content_addressing",
            "option_orderbook_zero_cost_qualification",
            "cash_option_final_free_source_repair",
            "v1_6_manifest_frozen",
        }:
            alerts.append(
                {
                    "code": gate_id.upper(),
                    "message": str(foundation_gates[gate_id]["status"]),
                }
            )

    return {
        "schema_version": "1.6.0",
        "report_version": config["version"],
        "generated_at": generated_at.isoformat(),
        "overall_research_status": contract["overall_research_status"],
        "decision": contract["decision"],
        "foundation_status": (
            "PASS_FOUNDATION_GATES_AWAITING_FORWARD_MATURITY"
            if not pending_gate_ids
            else "PARTIAL_SUCCESS_FOUNDATION_GATES_PENDING"
        ),
        "contract": {
            "data_purchase_budget_cny": int(contract["data_purchase_budget_cny"]),
            "zero_purchase_lock_days": int(contract["zero_purchase_lock_days"]),
            "maximum_active_research_streams": int(
                contract["maximum_active_research_streams"]
            ),
            "active_research_stream_count": len(active_streams),
            "only_new_frozen_test": contract["only_new_frozen_test"],
            "option_orderbook_policy": contract["option_orderbook_policy"],
        },
        "active_research_streams": active_streams,
        "maintenance_protocols": contract["maintenance_protocols"],
        "data_admission_only": {
            "test_id": contract["only_new_frozen_test"],
            "status": cash_terminal_status,
            "pre_repair_status": cash_summary["current_status"],
            "event_count": int(cash_summary["event_count"]),
            "data_gate_pass_count": int(cash_summary["data_gate_pass_count"]),
            "price_values_read": bool(
                cash_terminal_decision.get(
                    "price_values_read", cash_summary["price_values_read_by_matrix"]
                )
            ),
            "return_values_read": bool(
                cash_terminal_decision.get(
                    "return_values_read", cash_summary["return_values_read_by_matrix"]
                )
            ),
            "attempt_consumed": (
                bool(cash_terminal.get("attempt_consumed"))
                if cash_terminal is not None
                else False
            ),
            "second_repair_allowed": bool(
                cash_terminal_decision.get("second_repair_allowed", False)
            ),
            "candidate_closed_for_free_source_repair": bool(
                cash_terminal_decision.get(
                    "candidate_closed_for_free_source_repair", False
                )
            ),
            "counts_as_active_research_stream": False,
        },
        "free_source_registry": registry,
        "source_health": {
            "status": (
                "IMPLEMENTED_AWAITING_REAL_FORWARD_DAY"
                if second_endpoint_admitted and crosscheck_rows == 0
                else (
                    "PASS_REAL_FORWARD_CROSSCHECK_PRESENT"
                    if second_endpoint_admitted
                    else "BLOCKED_SECOND_OFFICIAL_ENDPOINT_NOT_ADMITTED"
                )
            ),
            "pcf_iopv_real_forward_crosscheck_rows": crosscheck_rows,
            "industry_official_release_status": industry["origin_collection"][
                "status"
            ],
            "nonzero_cost_source_ids": registry["nonzero_cost_source_ids"],
        },
        "foundation_gates": foundation_gates,
        "pending_foundation_gate_ids": pending_gate_ids,
        "alerts": alerts,
        "input_evidence": {
            "primary_market_readiness": _evidence(
                _path(inputs["primary_market_readiness"])
            ),
            "industry_operations_status": _evidence(
                _path(inputs["industry_operations_status"])
            ),
            "cash_option_admission_audit": _evidence(
                _path(inputs["cash_option_admission_audit"])
            ),
            "cash_option_final_free_source_repair": _evidence(
                _path(inputs["final_cash_source_repair"])
            ),
            "scheduler_deployment": _evidence(
                _path(inputs["scheduler_deployment"])
            ),
        },
        "execution_authorization": dict(config["execution_authorization"]),
    }


def _markdown(report: dict[str, Any]) -> str:
    primary, industry = report["active_research_streams"]
    cash = report["data_admission_only"]
    lines = [
        "# 优先前瞻研究权威状态 V1.6",
        "",
        f"- 研究结论：`{report['overall_research_status']}`",
        f"- 当前决策：`{report['decision']}`",
        f"- 基础状态：`{report['foundation_status']}`",
        f"- 生成时间：`{report['generated_at']}`",
        f"- 数据采购预算：{report['contract']['data_purchase_budget_cny']} CNY；冻结期 {report['contract']['zero_purchase_lock_days']} 天。",
        f"- 活动研究流：{report['contract']['active_research_stream_count']}/{report['contract']['maximum_active_research_streams']}。",
        "",
        "## 两条活动研究流",
        "",
        f"1. `PRIMARY_MARKET_PCF_IOPV`：{primary['full_quality_days']}/20 个完整质量日；V1.2 官方日端点真实复核记录 {primary['v1_2_official_crosscheck_rows']} 条。",
        f"2. `INDUSTRY_EXPECTATION_GAP`：原点 {industry['origin_cluster_count']}，成熟原点 {industry['mature_origin_cluster_count']}，非重叠 60 日块 {industry['non_overlapping_60d_block_count']}。",
        "",
        "## 数据准入",
        "",
        f"- 唯一新冻结测试：`{cash['test_id']}`。",
        f"- 事件：{cash['event_count']}；通过数据闸门：{cash['data_gate_pass_count']}。",
        f"- 当前状态：`{cash['status']}`；价格读取：`{cash['price_values_read']}`；收益读取：`{cash['return_values_read']}`。",
        "",
        "## 待完成基础门槛",
        "",
    ]
    if report["pending_foundation_gate_ids"]:
        lines.extend(
            f"- `{gate_id}`：`{report['foundation_gates'][gate_id].get('status', 'PENDING')}`"
            for gate_id in report["pending_foundation_gate_ids"]
        )
    else:
        lines.append("- 无；继续等待前瞻成熟度。")
    lines.extend(
        [
            "",
            "## 执行边界",
            "",
            "- 仅研究；Shadow、仓位映射、订单、券商连接和实盘均未启用。",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
    )
    encoded = content.encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 V1.6 优先前瞻研究权威状态")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    generated_at = datetime.now(ZoneInfo(str(config["timezone"])))
    report = build_report(config, generated_at)
    _atomic_text(
        _path(config["outputs"]["json"]),
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_text(_path(config["outputs"]["markdown"]), _markdown(report))
    print(f"权威状态：{report['overall_research_status']}")
    print(f"决策：{report['decision']}")
    print(f"待完成基础门槛：{len(report['pending_foundation_gate_ids'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
