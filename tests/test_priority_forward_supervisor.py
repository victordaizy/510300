from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from scripts.run_priority_forward_supervisor import (
    SAFETY_FIELDS,
    TaskSpec,
    _read_events,
    append_event_once,
    create_run_receipt,
    decision_for_task,
    evidence_for_date,
    load_config,
    run_cycle,
    task_specs,
)


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")


def _task(schedule: str = "09:25", latest: str = "09:35") -> TaskSpec:
    return TaskSpec(
        task_id="PRIMARY_MARKET_PCF_IOPV",
        schedule=datetime.strptime(schedule, "%H:%M").time(),
        latest_start=datetime.strptime(latest, "%H:%M").time(),
        weekdays=(0, 1, 2, 3, 4),
        launcher=ROOT / "does_not_run.ps1",
        evidence_globs=(),
    )


def test_primary_market_never_backfills_after_start_window() -> None:
    task = _task()
    now = datetime(2026, 8, 19, 20, 0, tzinfo=TIMEZONE)
    assert decision_for_task(task, now, attempted=False) == "MISSED_START_WINDOW"
    assert decision_for_task(task, now, attempted=True) == "ALREADY_ATTEMPTED"


def test_task_runs_only_inside_declared_window() -> None:
    task = _task()
    assert (
        decision_for_task(
            task, datetime(2026, 8, 20, 9, 24, tzinfo=TIMEZONE), attempted=False
        )
        == "WAITING_FOR_SCHEDULE"
    )
    assert (
        decision_for_task(
            task, datetime(2026, 8, 20, 9, 30, tzinfo=TIMEZONE), attempted=False
        )
        == "RUN_NOW"
    )


def test_existing_failure_receipt_counts_as_attempt(tmp_path: Path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "audit_date": "2026-08-19",
                "run_status": "FAILED",
                **{field: False for field in SAFETY_FIELDS},
            }
        ),
        encoding="utf-8",
    )
    task = TaskSpec(
        task_id="PRIMARY_MARKET_PCF_IOPV",
        schedule=datetime.strptime("09:25", "%H:%M").time(),
        latest_start=datetime.strptime("09:35", "%H:%M").time(),
        weekdays=(0, 1, 2, 3, 4),
        launcher=receipt,
        evidence_globs=("*.json",),
    )
    assert evidence_for_date(task, datetime(2026, 8, 19).date(), TIMEZONE, tmp_path) == [
        receipt
    ]


def test_event_ledger_is_hash_chained_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    event = {
        "schema_version": "1.0.0",
        "stable_event_id": "2026-08-19::PRIMARY::MISSED_START_WINDOW",
        "recorded_at": "2026-08-19T20:00:00+08:00",
        "task_id": "PRIMARY",
        "event_type": "MISSED_START_WINDOW",
        "exit_code": None,
        "message": "test",
        "research_only": True,
        **{field: False for field in SAFETY_FIELDS},
    }
    assert append_event_once(path, event) is True
    assert append_event_once(path, event) is False
    assert len(_read_events(path)) == 1


def test_current_config_has_three_research_only_tasks() -> None:
    config = load_config()
    specs = task_specs(config)
    assert [item.task_id for item in specs] == [
        "PRIMARY_MARKET_PCF_IOPV",
        "INDUSTRY_EXPECTATION_GAP",
        "PRIORITY_FORWARD_STATUS",
    ]
    assert config["safety"]["research_only"] is True
    assert all(config["safety"][field] is False for field in SAFETY_FIELDS)


def test_dry_run_writes_status_without_invoking_tasks(tmp_path: Path) -> None:
    launcher = tmp_path / "launcher.ps1"
    launcher.write_text("throw 'must not run'\n", encoding="utf-8")
    config = {
        "timezone": "Asia/Shanghai",
        "outputs": {
            "status": "status.json",
            "event_ledger": "events.jsonl",
            "lock_file": "lock",
            "log_file": "log",
        },
        "tasks": [
            {
                "id": "TEST",
                "schedule": "09:25",
                "latest_start": "09:35",
                "weekdays": [0, 1, 2, 3, 4],
                "launcher": "launcher.ps1",
                "evidence_globs": [],
            }
        ],
        "safety": {
            "research_only": True,
            **{field: False for field in SAFETY_FIELDS},
        },
    }
    decisions = run_cycle(
        config,
        datetime(2026, 8, 20, 9, 30, tzinfo=TIMEZONE),
        root=tmp_path,
        dry_run=True,
        login_autostart_verified=False,
    )
    assert decisions[0]["decision"] == "RUN_NOW"
    status = json.loads((tmp_path / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "RUNNING_SESSION_SCOPED_DEGRADED_AUTOMATION"
    assert status["login_autostart_verified"] is False
    assert status["run_receipt_file"] is None
    assert status["safety"]["live_trading_enabled"] is False


def test_supervisor_start_receipt_is_immutable_and_never_trades(
    tmp_path: Path,
) -> None:
    config = {
        "safety": {
            "research_only": True,
            **{field: False for field in SAFETY_FIELDS},
        }
    }
    started_at = datetime(2026, 8, 19, 20, 0, tzinfo=TIMEZONE)
    receipt = create_run_receipt(tmp_path, config, started_at, False)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["immutable_receipt"] is True
    assert payload["automation_scope"] == "CURRENT_SESSION_ONLY"
    assert payload["login_autostart_verified"] is False
    assert all(payload[field] is False for field in SAFETY_FIELDS)


def test_autostart_scripts_use_hidden_window_and_never_trade() -> None:
    launcher = (ROOT / "scripts" / "start_priority_forward_supervisor.ps1").read_text(
        encoding="utf-8-sig"
    )
    installer = (
        ROOT / "scripts" / "install_priority_forward_supervisor_autostart.ps1"
    ).read_text(encoding="utf-8-sig")
    assert "-WindowStyle Hidden" in launcher
    assert "-WindowStyle Hidden" in installer
    assert "CurrentVersion\\Run" in installer
    assert "Trading authorization: false" in installer
    assert "order" not in launcher.lower()
