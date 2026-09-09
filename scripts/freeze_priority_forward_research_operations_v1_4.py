"""冻结优先前瞻研究 V1.4 单一 Codex 编排入口。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

import freeze_priority_forward_research_operations_v1 as base
import freeze_priority_forward_research_operations_v1_3 as v1_3


ROOT = base.ROOT
CONFIG = base.CONFIG
PREVIOUS_MANIFEST = (
    ROOT / "config" / "priority_forward_research_operations_v1_3_manifest.json"
)
MANIFEST = ROOT / "config" / "priority_forward_research_operations_v1_4_manifest.json"
CODEX_AUTOMATION = (
    ROOT
    / "reports"
    / "audit"
    / "priority_forward_codex_automation_v1_4_20260819.json"
)
ORCHESTRATION_RECEIPT = (
    ROOT
    / "reports"
    / "audit"
    / "priority_forward_codex_runs"
    / "20260819T131659369938Z_13648_close.json"
)
FROZEN_FILES = sorted(
    set(
        v1_3.FROZEN_FILES
        + [
            "docs/PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_4_SINGLE_ENTRYPOINT_ADDENDUM.md",
            "reports/audit/priority_forward_codex_automation_v1_4_20260819.json",
            "scripts/freeze_priority_forward_research_operations_v1_4.py",
            "scripts/run_priority_forward_codex_automation.py",
            "tests/test_priority_forward_codex_automation.py",
        ]
    )
)


def _validate_entrypoint(automation: dict[str, Any]) -> list[dict[str, Any]]:
    external = v1_3._validate_codex_automation(automation)
    entrypoint = automation["single_tested_entrypoint"]
    path = base._path(str(entrypoint["path"]))
    if base._sha256(path) != entrypoint["sha256"]:
        raise ValueError("单一编排入口哈希不一致")
    if path.stat().st_size != int(entrypoint["bytes"]):
        raise ValueError("单一编排入口字节数不一致")
    for key, phase in (("primary_market_morning", "morning"), ("close_validation", "close")):
        item = automation["automations"][key]
        if item.get("single_entrypoint_verified") is not True:
            raise ValueError(f"自动化未声明单一入口：{key}")
        if item.get("entrypoint_phase") != phase:
            raise ValueError(f"自动化阶段不一致：{key}")
        text = Path(str(item["toml_path"])).read_text(encoding="utf-8")
        required = f"run_priority_forward_codex_automation.py --phase {phase}"
        if required not in text:
            raise ValueError(f"外部心跳没有调用单一入口：{required}")
    morning_text = Path(
        str(automation["automations"]["primary_market_morning"]["toml_path"])
    ).read_text(encoding="utf-8")
    close_text = Path(
        str(automation["automations"]["close_validation"]["toml_path"])
    ).read_text(encoding="utf-8")
    if "run_510300_primary_market_collection_task.ps1" in morning_text:
        raise ValueError("早间心跳仍绕过单一入口直接调用PCF运行器")
    if "run_industry_expectation_gap_forward_operations_task.ps1" in close_text:
        raise ValueError("收盘心跳仍绕过单一入口直接调用行业运行器")
    return external


def _validate_orchestration_receipt(payload: dict[str, Any]) -> None:
    base._assert_safety(payload, "Codex编排收据")
    if payload.get("immutable_receipt") is not True:
        raise ValueError("Codex编排收据未声明不可变")
    if payload.get("phase") != "close" or payload.get("exit_code") != 0:
        raise ValueError("Codex收盘编排收据状态不一致")
    expected = ["INDUSTRY_EXPECTATION_GAP", "PRIORITY_FORWARD_STATUS"]
    actual = [item["task_id"] for item in payload["task_results"]]
    if actual != expected:
        raise ValueError("Codex收盘编排任务集合发生变化")
    if any(item["decision"] != "ALREADY_ATTEMPTED" for item in payload["task_results"]):
        raise ValueError("真实编排收据未证明同日去重")
    if any(item["exit_code"] is not None for item in payload["task_results"]):
        raise ValueError("同日去重时意外执行了子任务")


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("优先前瞻配置顶层必须是对象")
    base._assert_safety(config, "优先前瞻配置")
    if config["inputs"].get("codex_automation_status") != CODEX_AUTOMATION.relative_to(
        ROOT
    ).as_posix():
        raise ValueError("统一配置未绑定V1.4 Codex自动化证据")
    automation = base._json(CODEX_AUTOMATION)
    external_automations = _validate_entrypoint(automation)
    orchestration = base._json(ORCHESTRATION_RECEIPT)
    _validate_orchestration_receipt(orchestration)

    priority_receipt_path, priority_receipt = base._latest_receipt(
        base._path("reports/audit/priority_forward_status_task_runs")
    )
    report_snapshot = base._path(str(priority_receipt["report_snapshot_file"]))
    markdown_snapshot = base._path(
        str(priority_receipt["report_markdown_snapshot_file"])
    )
    report = base._json(report_snapshot)
    base._assert_safety(report, "优先前瞻报告快照")
    if report["scheduler"].get("codex_heartbeats_active") is not True:
        raise ValueError("报告快照未确认Codex心跳自动化")
    if report.get("threshold_events", {}).get("total_event_count") != 0:
        raise ValueError("冻结时不应已有门槛跨越事件")
    if report["directions"]["orthogonal_low_vol_replication"]["eligible_to_start"]:
        raise ValueError("正交低波复制未被治理闸门阻断")

    frozen_records = []
    for relative in FROZEN_FILES:
        path = base._path(relative)
        if not path.is_file():
            raise ValueError(f"待冻结文件不存在：{relative}")
        frozen_records.append(base._record(path))

    now = datetime.now(ZoneInfo(str(config["timezone"])))
    manifest = {
        "manifest_version": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_4_MANIFEST",
        "status": "FROZEN_SINGLE_ENTRYPOINT_CODEX_AUTOMATION",
        "frozen_at": now.isoformat(timespec="seconds"),
        "supersedes": {
            **base._record(PREVIOUS_MANIFEST),
            "reason": "TESTED_SINGLE_ENTRYPOINT_REPLACES_NATURAL_LANGUAGE_TASK_ORCHESTRATION",
            "previous_manifest_preserved": True,
        },
        "frozen_files": frozen_records,
        "external_automation_evidence": external_automations,
        "runtime_evidence": {
            "codex_automation_readback": base._record(CODEX_AUTOMATION),
            "codex_close_orchestration_receipt": base._record(ORCHESTRATION_RECEIPT),
            "priority_status_task_receipt": base._record(priority_receipt_path),
            "priority_status_json_snapshot": base._record(report_snapshot),
            "priority_status_markdown_snapshot": base._record(markdown_snapshot),
        },
        "current_state": {
            "primary_market_full_coverage_days": int(
                report["directions"]["primary_market_pcf_iopv"]["full_coverage_days"]
            ),
            "industry_origin_cluster_count": int(
                report["directions"]["industry_expectation_gap"]["origin_cluster_count"]
            ),
            "industry_mature_origin_cluster_count": int(
                report["directions"]["industry_expectation_gap"][
                    "mature_origin_cluster_count"
                ]
            ),
            "threshold_event_count": 0,
            "governance_status": report["governance_status"],
            "orthogonal_low_vol_started": False,
        },
        "deployment": {
            "codex_morning_heartbeat_active": True,
            "codex_close_heartbeat_active": True,
            "single_tested_entrypoint": True,
            "real_close_deduplication_receipt_verified": True,
            "windows_task_scheduler_verified": False,
        },
        "authorization": {
            "research_only": True,
            **{field: False for field in base.SAFETY_FIELDS},
            "automatic_trading_authorized": False,
        },
        "git": {
            "commit": base._git_text("rev-parse", "HEAD"),
            "branch": base._git_text("branch", "--show-current"),
            "dirty_before_freeze": bool(base._git_text("status", "--porcelain")),
            "all_monitoring_files_tracked": False,
        },
    }
    state = base._write_once(
        MANIFEST,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(f"优先前瞻监控V1.4清单：{state}")
    print(f"输出：{MANIFEST}")
    print(f"冻结文件：{len(frozen_records)}")
    print("单一受测入口：启用；真实同日去重：通过；交易授权：false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
