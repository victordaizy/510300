"""冻结优先前瞻研究 V1.3 Codex 心跳自动化证据。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

import freeze_priority_forward_research_operations_v1 as base
import freeze_priority_forward_research_operations_v1_2 as v1_2


ROOT = base.ROOT
CONFIG = base.CONFIG
PREVIOUS_MANIFEST = (
    ROOT / "config" / "priority_forward_research_operations_v1_2_manifest.json"
)
MANIFEST = ROOT / "config" / "priority_forward_research_operations_v1_3_manifest.json"
CODEX_AUTOMATION = (
    ROOT / "reports" / "audit" / "priority_forward_codex_automation_20260819.json"
)
FROZEN_FILES = sorted(
    set(
        v1_2.FROZEN_FILES
        + [
            "docs/PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_3_CODEX_HEARTBEAT_ADDENDUM.md",
            "reports/audit/priority_forward_codex_automation_20260819.json",
            "scripts/freeze_priority_forward_research_operations_v1_3.py",
        ]
    )
)


def _external_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_external_automation(item: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(item["toml_path"]))
    if not path.is_file():
        raise ValueError(f"Codex自动化配置不存在：{path}")
    if path.stat().st_size != int(item["toml_bytes"]):
        raise ValueError(f"Codex自动化配置字节数不一致：{path}")
    if _external_sha256(path) != item["toml_sha256"]:
        raise ValueError(f"Codex自动化配置哈希不一致：{path}")
    text = path.read_text(encoding="utf-8")
    required = (
        f'id = "{item["id"]}"',
        'kind = "heartbeat"',
        'status = "ACTIVE"',
        f'target_thread_id = "{item["target_thread_id"]}"',
    )
    for fragment in required:
        if fragment not in text:
            raise ValueError(f"Codex自动化配置缺少字段：{fragment}")
    return {
        "id": item["id"],
        "path": str(path).replace("\\", "/"),
        "sha256": item["toml_sha256"],
        "bytes": item["toml_bytes"],
        "target_thread_id": item["target_thread_id"],
        "status": item["status"],
    }


def _validate_codex_automation(payload: dict[str, Any]) -> list[dict[str, Any]]:
    base._assert_safety(payload, "Codex心跳自动化证据")
    if payload.get("status") != "ACTIVE_CODEX_HEARTBEAT_AUTOMATION":
        raise ValueError("Codex心跳自动化状态不是ACTIVE")
    if payload.get("readback_verified") is not True:
        raise ValueError("Codex心跳自动化没有回读证据")
    morning = payload["automations"]["primary_market_morning"]
    close = payload["automations"]["close_validation"]
    if morning.get("late_backfill_authorized") is not False:
        raise ValueError("早间心跳错误授权晚到补跑")
    if close.get("paper_signal_execution_authorized") is not False:
        raise ValueError("收盘后心跳错误授权纸面信号")
    if close.get("orthogonal_low_vol_start_authorized") is not False:
        raise ValueError("收盘后心跳错误授权正交低波启动")
    return [
        _validate_external_automation(morning),
        _validate_external_automation(close),
    ]


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("优先前瞻配置顶层必须是对象")
    base._assert_safety(config, "优先前瞻配置")
    if config["inputs"].get("codex_automation_status") != CODEX_AUTOMATION.relative_to(
        ROOT
    ).as_posix():
        raise ValueError("统一配置未绑定Codex自动化证据")
    automation = base._json(CODEX_AUTOMATION)
    external_automations = _validate_codex_automation(automation)

    priority_receipt_path, priority_receipt = base._latest_receipt(
        base._path("reports/audit/priority_forward_status_task_runs")
    )
    report_snapshot = base._path(str(priority_receipt["report_snapshot_file"]))
    markdown_snapshot = base._path(
        str(priority_receipt["report_markdown_snapshot_file"])
    )
    report = base._json(report_snapshot)
    base._assert_safety(report, "优先前瞻报告快照")
    scheduler = report["scheduler"]
    if scheduler.get("codex_heartbeats_active") is not True:
        raise ValueError("报告快照未确认Codex心跳自动化")
    if scheduler.get("registered_tasks_verified") is not False:
        raise ValueError("报告快照错误声明Windows计划任务")
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
        "manifest_version": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_3_MANIFEST",
        "status": "FROZEN_CODEX_HEARTBEAT_AUTOMATION_WINDOWS_FALLBACK_BLOCKED",
        "frozen_at": now.isoformat(timespec="seconds"),
        "supersedes": {
            **base._record(PREVIOUS_MANIFEST),
            "reason": "CODEX_HEARTBEAT_AUTOMATION_REPLACES_WINDOWS_SCHEDULER_AS_PRIMARY_PATH",
            "previous_manifest_preserved": True,
        },
        "frozen_files": frozen_records,
        "external_automation_evidence": external_automations,
        "runtime_evidence": {
            "codex_automation_readback": base._record(CODEX_AUTOMATION),
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
            "windows_task_scheduler_verified": False,
            "login_autostart_verified": False,
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
    print(f"优先前瞻监控V1.3清单：{state}")
    print(f"输出：{MANIFEST}")
    print(f"冻结文件：{len(frozen_records)}")
    print("Codex心跳：ACTIVE；Windows备用：权限阻断；交易授权：false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

