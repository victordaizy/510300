"""当前用户登录期的优先前瞻研究降级调度守护进程。"""

from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import msvcrt
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, time as clock_time
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "priority_forward_supervisor_v1.yaml"
SAFETY_FIELDS = (
    "position_mapping_enabled",
    "order_generation_enabled",
    "broker_connection_enabled",
    "live_trading_enabled",
)
DATE_FIELDS = (
    "started_at",
    "scheduled_task_started_at",
    "failure_log_ended_at",
    "audit_date",
    "target_date",
)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    schedule: clock_time
    latest_start: clock_time
    weekdays: tuple[int, ...]
    launcher: Path
    evidence_globs: tuple[str, ...]


def _path(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    path.relative_to(resolved_root)
    return path


def _parse_clock(value: str) -> clock_time:
    return datetime.strptime(value, "%H:%M").time()


def _assert_safety(payload: dict[str, Any], label: str) -> None:
    safety = payload.get("safety", payload)
    for field in SAFETY_FIELDS:
        if safety.get(field) is not False:
            raise ValueError(f"{label}未明确关闭安全字段：{field}")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("守护进程配置顶层必须是对象")
    _assert_safety(value, "守护进程配置")
    return value


def task_specs(config: dict[str, Any], root: Path = ROOT) -> list[TaskSpec]:
    result: list[TaskSpec] = []
    identifiers: set[str] = set()
    for item in config["tasks"]:
        task_id = str(item["id"])
        if task_id in identifiers:
            raise ValueError(f"任务ID重复：{task_id}")
        identifiers.add(task_id)
        launcher = _path(root, str(item["launcher"]))
        if not launcher.is_file():
            raise ValueError(f"任务运行器不存在：{launcher}")
        schedule = _parse_clock(str(item["schedule"]))
        latest_start = _parse_clock(str(item["latest_start"]))
        if latest_start < schedule:
            raise ValueError(f"任务最晚启动时间早于计划时间：{task_id}")
        result.append(
            TaskSpec(
                task_id=task_id,
                schedule=schedule,
                latest_start=latest_start,
                weekdays=tuple(int(value) for value in item["weekdays"]),
                launcher=launcher,
                evidence_globs=tuple(str(value) for value in item["evidence_globs"]),
            )
        )
    return result


def _payload_date(payload: dict[str, Any], timezone: ZoneInfo) -> date | None:
    for field in DATE_FIELDS:
        raw = payload.get(field)
        if not raw:
            continue
        text = str(raw)
        try:
            if len(text) == 10:
                return date.fromisoformat(text)
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone)
            return parsed.astimezone(timezone).date()
        except ValueError:
            continue
    return None


def evidence_for_date(
    task: TaskSpec,
    target_date: date,
    timezone: ZoneInfo,
    root: Path = ROOT,
) -> list[Path]:
    matches: list[Path] = []
    for pattern in task.evidence_globs:
        absolute_pattern = str(_path(root, pattern))
        for raw_path in glob.glob(absolute_pattern):
            path = Path(raw_path)
            try:
                payload = json.loads(path.read_text(encoding="utf-8-sig"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            _assert_safety(payload, str(path))
            if _payload_date(payload, timezone) == target_date:
                matches.append(path)
    return sorted(set(matches))


def decision_for_task(task: TaskSpec, now: datetime, attempted: bool) -> str:
    if now.weekday() not in task.weekdays:
        return "NOT_DUE_NON_WEEKDAY"
    if attempted:
        return "ALREADY_ATTEMPTED"
    local_time = now.timetz().replace(tzinfo=None)
    if local_time < task.schedule:
        return "WAITING_FOR_SCHEDULE"
    if local_time > task.latest_start:
        return "MISSED_START_WINDOW"
    return "RUN_NOW"


def _canonical_event_hash(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "event_sha256"}
    encoded = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    previous_hash: str | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("previous_event_sha256") != previous_hash:
            raise ValueError(f"守护进程事件链前序哈希不一致：第{line_number}行")
        if payload.get("event_sha256") != _canonical_event_hash(payload):
            raise ValueError(f"守护进程事件哈希不一致：第{line_number}行")
        _assert_safety(payload, f"守护进程事件第{line_number}行")
        events.append(payload)
        previous_hash = payload["event_sha256"]
    return events


def append_event_once(path: Path, event: dict[str, Any]) -> bool:
    events = _read_events(path)
    stable_id = str(event["stable_event_id"])
    existing = [item for item in events if item["stable_event_id"] == stable_id]
    if existing:
        comparable = dict(event)
        comparable["previous_event_sha256"] = existing[0]["previous_event_sha256"]
        comparable["event_sha256"] = existing[0]["event_sha256"]
        if comparable != existing[0]:
            raise ValueError(f"既有守护进程事件定义发生变化：{stable_id}")
        return False
    payload = dict(event)
    payload["previous_event_sha256"] = events[-1]["event_sha256"] if events else None
    payload["event_sha256"] = _canonical_event_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return True


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def create_run_receipt(
    directory: Path,
    config: dict[str, Any],
    started_at: datetime,
    login_autostart_verified: bool,
) -> Path:
    _assert_safety(config, "守护进程启动收据配置")
    receipt_id = f"{started_at.astimezone(ZoneInfo('UTC')).strftime('%Y%m%dT%H%M%S%fZ')}_{os.getpid()}"
    path = directory / f"{receipt_id}.json"
    if path.exists():
        raise ValueError(f"守护进程启动收据已存在，禁止覆盖：{path}")
    payload = {
        "schema_version": "1.0.0",
        "receipt_id": receipt_id,
        "immutable_receipt": True,
        "started_at": started_at.isoformat(),
        "pid": os.getpid(),
        "automation_scope": (
            "CURRENT_USER_LOGIN_AUTOSTART"
            if login_autostart_verified
            else "CURRENT_SESSION_ONLY"
        ),
        "windows_task_scheduler_verified": False,
        "login_autostart_verified": login_autostart_verified,
        "research_only": True,
        **{field: False for field in SAFETY_FIELDS},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return path


@contextlib.contextmanager
def single_instance_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("优先前瞻守护进程已有实例运行") from exc
        handle.seek(0)
        handle.write(str(os.getpid()).encode("ascii").ljust(32, b" "))
        handle.flush()
        yield
    finally:
        try:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            handle.close()


def _event(
    task: TaskSpec,
    event_type: str,
    now: datetime,
    exit_code: int | None,
    message: str,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "stable_event_id": f"{now.date().isoformat()}::{task.task_id}::{event_type}",
        "recorded_at": now.isoformat(),
        "task_id": task.task_id,
        "event_type": event_type,
        "exit_code": exit_code,
        "message": message,
        "research_only": True,
        **{field: False for field in SAFETY_FIELDS},
    }


def _write_status(
    path: Path,
    config: dict[str, Any],
    now: datetime,
    decisions: list[dict[str, Any]],
    login_autostart_verified: bool,
    run_receipt_file: str | None,
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": "1.0.0",
            "status": (
                "RUNNING_LOGIN_SCOPED_DEGRADED_AUTOMATION"
                if login_autostart_verified
                else "RUNNING_SESSION_SCOPED_DEGRADED_AUTOMATION"
            ),
            "heartbeat_at": now.isoformat(),
            "pid": os.getpid(),
            "windows_task_scheduler_verified": False,
            "login_autostart_verified": login_autostart_verified,
            "manual_runners_verified": True,
            "run_receipt_file": run_receipt_file,
            "decisions": decisions,
            "safety": dict(config["safety"]),
        },
    )


def run_cycle(
    config: dict[str, Any],
    now: datetime,
    *,
    root: Path = ROOT,
    dry_run: bool = False,
    login_autostart_verified: bool = False,
    run_receipt_file: str | None = None,
) -> list[dict[str, Any]]:
    timezone = ZoneInfo(str(config["timezone"]))
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone)
    now = now.astimezone(timezone)
    ledger = _path(root, str(config["outputs"]["event_ledger"]))
    decisions: list[dict[str, Any]] = []
    for task in task_specs(config, root):
        evidence = evidence_for_date(task, now.date(), timezone, root)
        existing_events = _read_events(ledger)
        supervisor_attempted = any(
            item["task_id"] == task.task_id
            and item["stable_event_id"].startswith(now.date().isoformat() + "::")
            and item["event_type"] in {"STARTED", "COMPLETED"}
            for item in existing_events
        )
        decision = decision_for_task(task, now, bool(evidence) or supervisor_attempted)
        item: dict[str, Any] = {
            "task_id": task.task_id,
            "decision": decision,
            "evidence_count": len(evidence),
            "evidence_paths": [path.relative_to(root).as_posix() for path in evidence],
        }
        decisions.append(item)
        if decision == "MISSED_START_WINDOW":
            append_event_once(
                ledger,
                _event(
                    task,
                    "MISSED_START_WINDOW",
                    now,
                    None,
                    "未在最晚启动时间前运行；禁止补造或回填当日盘中证据。",
                ),
            )
        elif decision == "RUN_NOW" and not dry_run:
            append_event_once(
                ledger,
                _event(task, "STARTED", now, None, "按配置调用无交易研究运行器。"),
            )
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
                check=False,
            )
            completed_at = datetime.now(timezone)
            append_event_once(
                ledger,
                _event(
                    task,
                    "COMPLETED",
                    completed_at,
                    result.returncode,
                    "运行器成功完成。" if result.returncode == 0 else "运行器失败，保留失败收据。",
                ),
            )
            item["exit_code"] = result.returncode
    _write_status(
        _path(root, str(config["outputs"]["status"])),
        config,
        now,
        decisions,
        login_autostart_verified,
        run_receipt_file,
    )
    return decisions


def _login_autostart_verified() -> bool:
    command = [
        "reg.exe",
        "QUERY",
        r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run",
        "/v",
        "CodexPriorityForwardSupervisorV1",
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    return result.returncode == 0


def main() -> int:
    parser = argparse.ArgumentParser(description="运行优先前瞻研究登录期守护进程")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--once", action="store_true", help="只执行一次调度判断")
    parser.add_argument("--dry-run", action="store_true", help="不调用任何子任务")
    parser.add_argument("--now", help="测试用ISO时间；仅允许与--once同时使用")
    args = parser.parse_args()
    if args.now and not args.once:
        raise ValueError("--now仅允许与--once同时使用")
    config = load_config(args.config)
    timezone = ZoneInfo(str(config["timezone"]))
    status_path = _path(ROOT, str(config["outputs"]["status"]))
    lock_path = _path(ROOT, str(config["outputs"]["lock_file"]))
    with single_instance_lock(lock_path):
        started_at = datetime.now(timezone)
        login_autostart_verified = _login_autostart_verified()
        run_receipt = create_run_receipt(
            _path(ROOT, str(config["outputs"]["run_receipt_directory"])),
            config,
            started_at,
            login_autostart_verified,
        )
        run_receipt_file = run_receipt.relative_to(ROOT).as_posix()
        while True:
            now = (
                datetime.fromisoformat(args.now.replace("Z", "+00:00"))
                if args.now
                else datetime.now(timezone)
            )
            decisions = run_cycle(
                config,
                now,
                dry_run=args.dry_run,
                login_autostart_verified=login_autostart_verified,
                run_receipt_file=run_receipt_file,
            )
            if args.once:
                print(json.dumps(decisions, ensure_ascii=False, indent=2))
                return 0
            time.sleep(int(config["poll_seconds"]))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"优先前瞻守护进程失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        raise
