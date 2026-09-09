"""Codex 心跳调用的早间与收盘后严格前瞻编排运行器。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

if __package__:
    from scripts.run_priority_forward_supervisor import (
        ROOT,
        SAFETY_FIELDS,
        TaskSpec,
        _assert_safety,
        _atomic_json,
        _path,
        decision_for_task,
        evidence_for_date,
        load_config,
        task_specs,
    )
else:
    from run_priority_forward_supervisor import (
        ROOT,
        SAFETY_FIELDS,
        TaskSpec,
        _assert_safety,
        _atomic_json,
        _path,
        decision_for_task,
        evidence_for_date,
        load_config,
        task_specs,
    )


PHASE_TASK_IDS = {
    "morning": ("PRIMARY_MARKET_PCF_IOPV",),
    "close": ("INDUSTRY_EXPECTATION_GAP", "PRIORITY_FORWARD_STATUS"),
}


def selected_tasks(
    config: dict[str, Any], phase: str, root: Path = ROOT
) -> list[TaskSpec]:
    if phase not in PHASE_TASK_IDS:
        raise ValueError(f"未知编排阶段：{phase}")
    by_id = {item.task_id: item for item in task_specs(config, root)}
    return [by_id[task_id] for task_id in PHASE_TASK_IDS[phase]]


def _wait_until_schedule(task: TaskSpec, now: datetime, timezone: ZoneInfo) -> datetime:
    target = now.replace(
        hour=task.schedule.hour,
        minute=task.schedule.minute,
        second=0,
        microsecond=0,
    )
    seconds = (target - now).total_seconds()
    if seconds < 0:
        return now
    if seconds > 10 * 60:
        raise ValueError(f"早间心跳过早，拒绝等待超过10分钟：{seconds:.3f}秒")
    while seconds > 0:
        time.sleep(min(seconds, 30))
        now = datetime.now(timezone)
        seconds = (target - now).total_seconds()
    return now


def _run_powershell(
    task: TaskSpec,
    root: Path,
    log_handle: Any,
) -> int:
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(task.launcher),
        ],
        cwd=root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return int(result.returncode)


def _run_renderer(root: Path, log_handle: Any) -> int:
    result = subprocess.run(
        [
            str(root / ".venv" / "Scripts" / "python.exe"),
            str(root / "scripts" / "render_priority_forward_research_status.py"),
        ],
        cwd=root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return int(result.returncode)


def _write_once_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def run_phase(
    config: dict[str, Any],
    phase: str,
    now: datetime,
    *,
    root: Path = ROOT,
    dry_run: bool = False,
    wait_for_window: bool = True,
) -> tuple[dict[str, Any], int]:
    _assert_safety(config, "Codex心跳编排配置")
    timezone = ZoneInfo(str(config["timezone"]))
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone)
    now = now.astimezone(timezone)
    started_at = now
    outputs = config["outputs"]
    receipt_directory = _path(root, str(outputs["codex_run_receipt_directory"]))
    log_directory = _path(root, str(outputs["codex_log_directory"]))
    current_status = _path(root, str(outputs["codex_current_status"]))
    receipt_id = (
        f"{started_at.astimezone(ZoneInfo('UTC')).strftime('%Y%m%dT%H%M%S%fZ')}"
        f"_{os.getpid()}_{phase}"
    )
    receipt_path = receipt_directory / f"{receipt_id}.json"
    log_path = log_directory / f"{receipt_id}.log"
    log_directory.mkdir(parents=True, exist_ok=True)
    tasks = selected_tasks(config, phase, root)
    task_results: list[dict[str, Any]] = []
    overall_exit = 0

    with log_path.open("ab") as log_handle:
        for task in tasks:
            before = evidence_for_date(task, now.date(), timezone, root)
            decision = decision_for_task(task, now, bool(before))
            if (
                phase == "morning"
                and decision == "WAITING_FOR_SCHEDULE"
                and not dry_run
                and wait_for_window
            ):
                now = _wait_until_schedule(task, now, timezone)
                before = evidence_for_date(task, now.date(), timezone, root)
                decision = decision_for_task(task, now, bool(before))

            result: dict[str, Any] = {
                "task_id": task.task_id,
                "decision": decision,
                "pre_evidence": [path.relative_to(root).as_posix() for path in before],
                "launcher": task.launcher.relative_to(root).as_posix(),
                "exit_code": None,
                "post_evidence": [],
            }
            if decision == "RUN_NOW" and not dry_run:
                exit_code = _run_powershell(task, root, log_handle)
                result["exit_code"] = exit_code
                if exit_code != 0:
                    overall_exit = 1
                after = evidence_for_date(task, datetime.now(timezone).date(), timezone, root)
                result["post_evidence"] = [
                    path.relative_to(root).as_posix() for path in after
                ]
                if not after:
                    result["evidence_error"] = "NO_IMMUTABLE_TASK_EVIDENCE_AFTER_RUN"
                    overall_exit = 1
            elif decision == "MISSED_START_WINDOW":
                overall_exit = max(overall_exit, 2)
            elif decision == "WAITING_FOR_SCHEDULE":
                overall_exit = max(overall_exit, 2)
            task_results.append(result)

        renderer_exit = None
        if not dry_run and phase == "morning":
            renderer_exit = _run_renderer(root, log_handle)
            if renderer_exit != 0:
                overall_exit = 1

    ended_at = now if dry_run else datetime.now(timezone)
    if overall_exit == 0:
        status = "SUCCESS_OR_ALREADY_ATTEMPTED"
    elif overall_exit == 2:
        status = "MISSED_OR_NOT_DUE_NO_BACKFILL"
    else:
        status = "FAILED"
    payload = {
        "schema_version": "1.0.0",
        "receipt_id": receipt_id,
        "immutable_receipt": True,
        "phase": phase,
        "status": status,
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "exit_code": overall_exit,
        "task_results": task_results,
        "morning_renderer_exit_code": renderer_exit if phase == "morning" else None,
        "receipt_file": receipt_path.relative_to(root).as_posix(),
        "log_file": log_path.relative_to(root).as_posix(),
        "research_only": True,
        **{field: False for field in SAFETY_FIELDS},
    }
    if not dry_run:
        _write_once_json(receipt_path, payload)
        _atomic_json(current_status, payload)
    return payload, overall_exit


def main() -> int:
    parser = argparse.ArgumentParser(description="运行Codex优先前瞻研究编排")
    parser.add_argument("--phase", choices=sorted(PHASE_TASK_IDS), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--now", help="测试用ISO时间；仅允许与--dry-run同时使用")
    args = parser.parse_args()
    if args.now and not args.dry_run:
        raise ValueError("--now仅允许与--dry-run同时使用")
    config = load_config()
    timezone = ZoneInfo(str(config["timezone"]))
    now = (
        datetime.fromisoformat(args.now.replace("Z", "+00:00"))
        if args.now
        else datetime.now(timezone)
    )
    payload, exit_code = run_phase(
        config,
        args.phase,
        now,
        dry_run=args.dry_run,
        wait_for_window=not args.dry_run,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Codex优先前瞻编排失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
