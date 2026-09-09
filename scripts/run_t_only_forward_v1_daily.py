"""T_ONLY_FORWARD_V1 无人值守日更编排与可审计运行状态。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")
AUTOMATION_MANIFEST = ROOT / "config" / "t_only_forward_v1_automation_manifest.json"
DATA_GATE = ROOT / "reports" / "data_quality" / "t_only_forward_v1_data_gate.json"
FORWARD_STATUS = ROOT / "reports" / "forward" / "t_only_forward_v1_1_status.json"
DAILY_GUIDE = ROOT / "reports" / "forward" / "t_only_forward_v1_1_daily_guide.json"
LEDGER = ROOT / "paper" / "t_only_forward_v1_1" / "daily_ledger.parquet"
RUN_STATUS = ROOT / "paper" / "t_only_forward_v1" / "daily_run_status.json"
LOCK_FILE = ROOT / "paper" / "t_only_forward_v1" / "daily_run.lock"
LOG_ROOT = ROOT / "output" / "t_only_forward_v1_logs"
REFRESH_SCRIPT = ROOT / "scripts" / "refresh_t_only_forward_v1.py"
FORWARD_SCRIPT = ROOT / "scripts" / "run_t_only_forward_v1_1.py"
ALLOWED_FORWARD_STATUSES = {
    "COLLECTING_FORWARD_NOT_STARTED",
    "COLLECTING",
    "PASS_FORWARD_GATES",
    "FAIL_FORWARD_GATES",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def verify_automation_manifest() -> dict[str, Any]:
    if not AUTOMATION_MANIFEST.exists():
        raise RuntimeError("自动运行冻结清单缺失")
    manifest = json.loads(AUTOMATION_MANIFEST.read_text(encoding="utf-8"))
    for section in ("automation_files", "research_freeze_files"):
        for relative_path, expected in manifest[section].items():
            path = ROOT / relative_path
            if not path.exists():
                raise RuntimeError(f"自动运行冻结文件缺失：{relative_path}")
            actual = sha256(path)
            if actual != expected:
                raise RuntimeError(f"自动运行冻结文件指纹变化：{relative_path}")
    return manifest


def acquire_lock(now: datetime) -> int:
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError("已有 T_ONLY 前瞻日更实例正在运行") from exc
    os.write(
        descriptor,
        json.dumps(
            {"pid": os.getpid(), "started_at": now.isoformat()}, ensure_ascii=False
        ).encode("utf-8"),
    )
    return descriptor


def release_lock(descriptor: int | None) -> None:
    if descriptor is None:
        return
    os.close(descriptor)
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()


def run_stage(
    *, name: str, script: Path, log_directory: Path, timeout_seconds: int
) -> dict[str, Any]:
    started = datetime.now(TIMEZONE)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    command = [sys.executable, str(script)]
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
        exit_code = int(completed.returncode)
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        stderr += f"\n阶段超时：{timeout_seconds}秒"
    finished = datetime.now(TIMEZONE)
    stdout_path = log_directory / f"{name}.stdout.log"
    stderr_path = log_directory / f"{name}.stderr.log"
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    return {
        "name": name,
        "command": command,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "stdout_file": stdout_path.relative_to(ROOT).as_posix(),
        "stderr_file": stderr_path.relative_to(ROOT).as_posix(),
        "stdout_sha256": sha256(stdout_path),
        "stderr_sha256": sha256(stderr_path),
    }


def audit_outputs(run_started: datetime) -> dict[str, Any]:
    if not DATA_GATE.exists() or not FORWARD_STATUS.exists() or not DAILY_GUIDE.exists():
        raise RuntimeError("数据闸门、V1.1前瞻状态或每日操作卡缺失")
    data_gate = json.loads(DATA_GATE.read_text(encoding="utf-8"))
    forward = json.loads(FORWARD_STATUS.read_text(encoding="utf-8"))
    guide = json.loads(DAILY_GUIDE.read_text(encoding="utf-8"))
    if data_gate.get("status") != "PASS":
        raise RuntimeError(f"数据闸门未通过：{data_gate.get('status')}")
    retrieved_at = datetime.fromisoformat(data_gate["retrieved_at"])
    if retrieved_at < run_started:
        raise RuntimeError("数据闸门未在本次运行中刷新")
    forward_status = forward.get("status")
    if forward_status not in ALLOWED_FORWARD_STATUSES:
        raise RuntimeError(f"前瞻状态非法：{forward_status}")
    if forward.get("as_of_market_date") != data_gate.get("actual_last_date"):
        raise RuntimeError("前瞻报告与数据闸门的行情截止日不一致")
    if guide.get("as_of_market_date") != forward.get("as_of_market_date"):
        raise RuntimeError("每日操作卡与V1.1前瞻报告的行情截止日不一致")
    if guide.get("forward_status") != forward_status:
        raise RuntimeError("每日操作卡与V1.1前瞻报告的状态不一致")
    new_days = int(forward.get("new_trading_days", 0))
    ledger_rows = 0
    duplicate_ledger_dates = 0
    if new_days > 0:
        if not LEDGER.exists():
            raise RuntimeError("已有前瞻交易日但影子日账本缺失")
        ledger = pd.read_parquet(LEDGER)
        ledger["date"] = pd.to_datetime(ledger["date"], errors="raise")
        ledger_rows = int(len(ledger))
        duplicate_ledger_dates = int(ledger["date"].duplicated().sum())
        if ledger_rows != new_days or duplicate_ledger_dates != 0:
            raise RuntimeError("影子账本行数、日期唯一性与前瞻状态不一致")
    return {
        "status": "PASS",
        "data_gate_status": data_gate["status"],
        "forward_status": forward_status,
        "as_of_market_date": forward["as_of_market_date"],
        "forward_signal_start": forward["forward_signal_start"],
        "new_trading_days": new_days,
        "closed_cycles": int(forward.get("closed_cycles", 0)),
        "current_view": forward.get("current_view"),
        "current_shadow_signal": forward.get("current_shadow_signal"),
        "daily_decision": guide.get("decision"),
        "daily_headline": guide.get("headline"),
        "ledger_rows": ledger_rows,
        "duplicate_ledger_dates": duplicate_ledger_dates,
        "data_gate_sha256": sha256(DATA_GATE),
        "forward_status_sha256": sha256(FORWARD_STATUS),
        "daily_guide_sha256": sha256(DAILY_GUIDE),
    }


def overall_success_status(forward_status: str) -> str:
    mapping = {
        "COLLECTING_FORWARD_NOT_STARTED": "SUCCESS_COLLECTING_NOT_STARTED",
        "COLLECTING": "SUCCESS_COLLECTING",
        "PASS_FORWARD_GATES": "SUCCESS_FORWARD_GATES_PASS",
        "FAIL_FORWARD_GATES": "SUCCESS_FORWARD_GATES_FAIL_EVIDENCE",
    }
    return mapping[forward_status]


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 T_ONLY 前瞻影子日更")
    parser.add_argument("--stage-timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    if args.stage_timeout_seconds < 30:
        raise ValueError("阶段超时不得少于30秒")

    started = datetime.now(TIMEZONE)
    run_id = started.strftime("%Y%m%dT%H%M%S%f%z")
    log_directory = LOG_ROOT / run_id
    log_directory.mkdir(parents=True, exist_ok=False)
    status: dict[str, Any] = {
        "project_id": "510300_T_ONLY_FORWARD_V1_1_DAILY",
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": None,
        "overall_status": "RUNNING",
        "exit_code": None,
        "log_directory": log_directory.relative_to(ROOT).as_posix(),
        "stages": [],
        "output_audit": None,
        "failure": None,
        "safety": {
            "shadow_ledger_only": True,
            "live_position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
        },
    }
    atomic_json(RUN_STATUS, status)
    lock_descriptor: int | None = None
    exit_code = 1
    try:
        verify_automation_manifest()
        lock_descriptor = acquire_lock(started)
        refresh = run_stage(
            name="refresh",
            script=REFRESH_SCRIPT,
            log_directory=log_directory,
            timeout_seconds=args.stage_timeout_seconds,
        )
        status["stages"].append(refresh)
        if refresh["exit_code"] != 0:
            raise RuntimeError(
                f"刷新阶段失败，退出码：{refresh['exit_code']}"
            )
        forward = run_stage(
            name="forward",
            script=FORWARD_SCRIPT,
            log_directory=log_directory,
            timeout_seconds=args.stage_timeout_seconds,
        )
        status["stages"].append(forward)
        if forward["exit_code"] != 0:
            raise RuntimeError(
                f"前瞻账本阶段失败，退出码：{forward['exit_code']}"
            )
        output_audit = audit_outputs(started)
        status["output_audit"] = output_audit
        status["overall_status"] = overall_success_status(
            output_audit["forward_status"]
        )
        exit_code = 0
    except Exception as exc:
        status["overall_status"] = "NO_VIEW_OPERATIONAL_FAILURE"
        status["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        exit_code = 1
    finally:
        release_lock(lock_descriptor)
        finished = datetime.now(TIMEZONE)
        status["finished_at"] = finished.isoformat()
        status["duration_seconds"] = (finished - started).total_seconds()
        status["exit_code"] = exit_code
        atomic_json(RUN_STATUS, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
