"""冻结优先前瞻研究 V1.2 当前会话守护进程实现。"""

from __future__ import annotations

import json
import ctypes
from ctypes import wintypes
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

import freeze_priority_forward_research_operations_v1 as base
import freeze_priority_forward_research_operations_v1_1 as v1_1


ROOT = base.ROOT
CONFIG = base.CONFIG
SUPERVISOR_CONFIG = ROOT / "config" / "priority_forward_supervisor_v1.yaml"
PREVIOUS_MANIFEST = (
    ROOT / "config" / "priority_forward_research_operations_v1_1_manifest.json"
)
MANIFEST = ROOT / "config" / "priority_forward_research_operations_v1_2_manifest.json"
DEPLOYMENT_EVIDENCE = (
    ROOT
    / "reports"
    / "audit"
    / "priority_forward_automation_deployment_v1_2_20260819.json"
)
FROZEN_FILES = sorted(
    set(
        v1_1.FROZEN_FILES
        + [
            "config/priority_forward_supervisor_v1.yaml",
            "docs/PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_2_SESSION_SUPERVISOR_ADDENDUM.md",
            "scripts/freeze_priority_forward_research_operations_v1_2.py",
            "scripts/install_priority_forward_supervisor_autostart.ps1",
            "scripts/run_priority_forward_supervisor.py",
            "scripts/start_priority_forward_supervisor.ps1",
            "tests/test_priority_forward_supervisor.py",
        ]
    )
)


def _validate_deployment(payload: dict[str, Any]) -> None:
    base._assert_safety(payload, "会话守护进程部署证据")
    deployment = payload["deployment"]
    expected = {
        "windows_task_scheduler_verified": False,
        "login_autostart_verified": False,
        "session_supervisor_started": True,
        "session_supervisor_heartbeat_verified": True,
        "automatic_restart_after_logoff_or_reboot": False,
    }
    for field, value in expected.items():
        if deployment.get(field) is not value:
            raise ValueError(f"会话守护进程部署字段不一致：{field}")
    for group in ("runtime_evidence", "implementation_evidence"):
        for item in payload[group]:
            path = base._path(str(item["path"]))
            if base._sha256(path) != item["sha256"]:
                raise ValueError(f"部署证据哈希不一致：{path}")
            if "bytes" in item and path.stat().st_size != item["bytes"]:
                raise ValueError(f"部署证据字节数不一致：{path}")


def _validate_supervisor_config(config: dict[str, Any]) -> None:
    base._assert_safety(config, "会话守护进程配置")
    tasks = {item["id"]: item for item in config["tasks"]}
    if set(tasks) != {
        "PRIMARY_MARKET_PCF_IOPV",
        "INDUSTRY_EXPECTATION_GAP",
        "PRIORITY_FORWARD_STATUS",
    }:
        raise ValueError("会话守护进程任务集合发生变化")
    primary = tasks["PRIMARY_MARKET_PCF_IOPV"]
    if primary["schedule"] != "09:25" or primary["latest_start"] != "09:35":
        raise ValueError("PCF/IOPV严格盘中启动窗口发生变化")


def _assert_process_running(pid: int) -> None:
    process_query_limited_information = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        error = ctypes.get_last_error()
        raise ValueError(f"会话守护进程未运行：PID={pid}，WinError={error}")
    kernel32.CloseHandle(handle)


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    supervisor_config = yaml.safe_load(SUPERVISOR_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(supervisor_config, dict):
        raise ValueError("运行配置顶层必须是对象")
    base._assert_safety(config, "优先前瞻配置")
    _validate_supervisor_config(supervisor_config)
    deployment = base._json(DEPLOYMENT_EVIDENCE)
    _validate_deployment(deployment)

    for relative in (
        "scripts/install_510300_daily_collection_task.ps1",
        "scripts/run_510300_primary_market_collection_task.ps1",
        "scripts/install_industry_expectation_gap_forward_operations_task.ps1",
        "scripts/run_industry_expectation_gap_forward_operations_task.ps1",
        "scripts/install_priority_forward_research_status_task.ps1",
        "scripts/run_priority_forward_research_status_task.ps1",
        "scripts/install_priority_forward_supervisor_autostart.ps1",
        "scripts/start_priority_forward_supervisor.ps1",
    ):
        v1_1._assert_windows_powershell_parser(base._path(relative))

    supervisor_status = base._json(
        base._path("reports/audit/priority_forward_supervisor_status.json")
    )
    base._assert_safety(supervisor_status, "会话守护进程当前状态")
    if supervisor_status.get("status") != "RUNNING_SESSION_SCOPED_DEGRADED_AUTOMATION":
        raise ValueError("会话守护进程当前状态不一致")
    if supervisor_status.get("login_autostart_verified") is not False:
        raise ValueError("不得把登录自启动声明为已验证")
    heartbeat = datetime.fromisoformat(
        str(supervisor_status["heartbeat_at"]).replace("Z", "+00:00")
    )
    now = datetime.now(ZoneInfo(str(config["timezone"])))
    age = (now - heartbeat.astimezone(now.tzinfo)).total_seconds()
    if not 0 <= age <= 120:
        raise ValueError(f"会话守护进程心跳不新鲜：{age:.3f}秒")
    _assert_process_running(int(supervisor_status["pid"]))

    run_receipt = base._path(str(supervisor_status["run_receipt_file"]))
    run_payload = base._json(run_receipt)
    base._assert_safety(run_payload, "会话守护进程启动收据")
    if run_payload.get("immutable_receipt") is not True:
        raise ValueError("会话守护进程启动收据未声明不可变")
    if run_payload.get("automation_scope") != "CURRENT_SESSION_ONLY":
        raise ValueError("会话守护进程自动化范围不一致")

    priority_receipt_path, priority_receipt = base._latest_receipt(
        base._path("reports/audit/priority_forward_status_task_runs")
    )
    report_snapshot = base._path(str(priority_receipt["report_snapshot_file"]))
    markdown_snapshot = base._path(
        str(priority_receipt["report_markdown_snapshot_file"])
    )
    report = base._json(report_snapshot)
    base._assert_safety(report, "优先前瞻报告快照")
    if report["scheduler"].get("session_supervisor_heartbeat_fresh") is not True:
        raise ValueError("报告快照未核验会话守护进程心跳")
    if report["scheduler"].get("login_autostart_verified") is not False:
        raise ValueError("报告快照错误声明登录自启动")
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

    manifest = {
        "manifest_version": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_2_MANIFEST",
        "status": "FROZEN_SESSION_SCOPED_DEGRADED_AUTOMATION",
        "frozen_at": now.isoformat(timespec="seconds"),
        "supersedes": {
            **base._record(PREVIOUS_MANIFEST),
            "reason": "SESSION_SCOPED_SUPERVISOR_AFTER_SCHEDULER_AND_LOGIN_AUTOSTART_PERMISSION_DENIAL",
            "previous_manifest_preserved": True,
        },
        "frozen_files": frozen_records,
        "runtime_evidence": {
            "automation_deployment": base._record(DEPLOYMENT_EVIDENCE),
            "supervisor_run_receipt": base._record(run_receipt),
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
            "status": "RUNNING_SESSION_SCOPED_DEGRADED_AUTOMATION",
            "session_supervisor_heartbeat_verified": True,
            "windows_task_scheduler_verified": False,
            "login_autostart_verified": False,
            "automatic_restart_after_logoff_or_reboot": False,
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
    print(f"优先前瞻监控V1.2清单：{state}")
    print(f"输出：{MANIFEST}")
    print(f"冻结文件：{len(frozen_records)}")
    print("当前会话守护进程：运行；持久自启动：未授权；交易授权：false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
