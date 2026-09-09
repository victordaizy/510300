"""冻结优先前瞻研究 V1.1 兼容性与调度器披露状态。"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

import freeze_priority_forward_research_operations_v1 as base


ROOT = base.ROOT
CONFIG = base.CONFIG
PREVIOUS_MANIFEST = ROOT / "config" / "priority_forward_research_operations_v1_manifest.json"
MANIFEST = ROOT / "config" / "priority_forward_research_operations_v1_1_manifest.json"
SCHEDULER_STATUS = (
    ROOT / "reports" / "audit" / "priority_forward_scheduler_installation_20260819.json"
)
FROZEN_FILES = sorted(
    set(
        base.FROZEN_FILES
        + [
            "docs/PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_1_COMPATIBILITY_ADDENDUM.md",
            "scripts/freeze_priority_forward_research_operations_v1_1.py",
        ]
    )
)


def _assert_windows_powershell_parser(path: Path) -> None:
    escaped = str(path).replace("'", "''")
    command = (
        "$tokens=$null; $errors=$null; "
        "[System.Management.Automation.Language.Parser]::ParseFile("
        f"'{escaped}', [ref]$tokens, [ref]$errors) | Out-Null; "
        "if ($errors.Count -gt 0) { "
        "$errors | ForEach-Object { Write-Error $_.Message }; exit 1 }"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        raise ValueError(f"Windows PowerShell 5.1解析失败：{path}：{message}")


def _validate_scheduler_status(payload: dict[str, Any]) -> None:
    base._assert_safety(payload, "计划任务安装状态")
    if payload.get("status") != "BLOCKED_LOCAL_TASK_SCHEDULER_PERMISSION":
        raise ValueError("计划任务状态与本次权限证据不一致")
    if payload.get("registered_tasks_verified") is not False:
        raise ValueError("不得把未核验的计划任务声明为已安装")
    if payload.get("manual_runners_verified") is not True:
        raise ValueError("手动运行器尚未通过验证")
    if payload.get("windows_powershell_5_1_parser_verified") is not True:
        raise ValueError("Windows PowerShell 5.1解析状态未通过")


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("优先前瞻配置顶层必须是对象")
    base._assert_safety(config, "优先前瞻配置")
    if config["inputs"].get("scheduler_installation_status") != str(
        SCHEDULER_STATUS.relative_to(ROOT).as_posix()
    ):
        raise ValueError("统一配置未绑定计划任务安装状态")

    scheduler = base._json(SCHEDULER_STATUS)
    _validate_scheduler_status(scheduler)
    for relative in (
        "scripts/install_510300_daily_collection_task.ps1",
        "scripts/run_510300_primary_market_collection_task.ps1",
        "scripts/install_industry_expectation_gap_forward_operations_task.ps1",
        "scripts/run_industry_expectation_gap_forward_operations_task.ps1",
        "scripts/install_priority_forward_research_status_task.ps1",
        "scripts/run_priority_forward_research_status_task.ps1",
    ):
        _assert_windows_powershell_parser(base._path(relative))

    industry_receipt_path, _ = base._latest_receipt(
        base._path("reports/forward/industry_expectation_gap_v1_evaluation/task_runs")
    )
    priority_receipt_path, priority_receipt = base._latest_receipt(
        base._path("reports/audit/priority_forward_status_task_runs")
    )
    report_snapshot = base._path(str(priority_receipt["report_snapshot_file"]))
    markdown_snapshot = base._path(
        str(priority_receipt["report_markdown_snapshot_file"])
    )
    report = base._json(report_snapshot)
    base._assert_safety(report, "优先前瞻报告快照")
    if report.get("scheduler", {}).get("registered_tasks_verified") is not False:
        raise ValueError("报告未保留计划任务部署阻断")
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

    previous = base._record(PREVIOUS_MANIFEST)
    runtime_evidence = {
        "scheduler_installation_status": base._record(SCHEDULER_STATUS),
        "primary_market_last_failure": base._record(
            base._path("reports/data_quality/510300_primary_market_task_status_20260819.json")
        ),
        "industry_task_receipt": base._record(industry_receipt_path),
        "priority_status_task_receipt": base._record(priority_receipt_path),
        "priority_status_json_snapshot": base._record(report_snapshot),
        "priority_status_markdown_snapshot": base._record(markdown_snapshot),
    }
    now = datetime.now(ZoneInfo(str(config["timezone"])))
    manifest = {
        "manifest_version": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_1_MANIFEST",
        "status": "FROZEN_MONITORING_IMPLEMENTATION_SCHEDULER_INSTALL_BLOCKED",
        "frozen_at": now.isoformat(timespec="seconds"),
        "supersedes": {
            **previous,
            "reason": "WINDOWS_POWERSHELL_5_1_INSTALLER_COMPATIBILITY_AND_SCHEDULER_DEPLOYMENT_DISCLOSURE",
            "previous_manifest_preserved": True,
        },
        "frozen_files": frozen_records,
        "runtime_evidence": runtime_evidence,
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
            "manual_runners_verified": True,
            "windows_powershell_5_1_parser_verified": True,
            "scheduler_tasks_registered_and_verified": False,
            "scheduler_status": "BLOCKED_LOCAL_TASK_SCHEDULER_PERMISSION",
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
    print(f"优先前瞻监控V1.1清单：{state}")
    print(f"输出：{MANIFEST}")
    print(f"冻结文件：{len(frozen_records)}")
    print("计划任务：权限阻断；门槛事件：0；交易授权：false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

