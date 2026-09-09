from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import scripts.run_priority_forward_codex_automation as codex_automation
from scripts.run_priority_forward_codex_automation import run_phase, selected_tasks
from scripts.run_priority_forward_supervisor import SAFETY_FIELDS, load_config


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")


def test_phase_task_sets_are_exact() -> None:
    config = load_config()
    assert [task.task_id for task in selected_tasks(config, "morning")] == [
        "PRIMARY_MARKET_PCF_IOPV"
    ]
    assert [task.task_id for task in selected_tasks(config, "close")] == [
        "INDUSTRY_EXPECTATION_GAP",
        "PRIORITY_FORWARD_STATUS",
    ]


def test_morning_dry_run_refuses_late_backfill(monkeypatch) -> None:
    config = load_config()
    monkeypatch.setattr(codex_automation, "evidence_for_date", lambda *_args: [])
    payload, exit_code = run_phase(
        config,
        "morning",
        datetime(2026, 8, 20, 20, 0, tzinfo=TIMEZONE),
        dry_run=True,
        wait_for_window=False,
    )
    assert exit_code == 2
    assert payload["status"] == "MISSED_OR_NOT_DUE_NO_BACKFILL"
    assert payload["task_results"][0]["decision"] == "MISSED_START_WINDOW"
    assert payload["live_trading_enabled"] is False


def test_morning_dry_run_allows_only_declared_window(monkeypatch) -> None:
    config = load_config()
    monkeypatch.setattr(codex_automation, "evidence_for_date", lambda *_args: [])
    payload, exit_code = run_phase(
        config,
        "morning",
        datetime(2026, 8, 20, 9, 30, tzinfo=TIMEZONE),
        dry_run=True,
        wait_for_window=False,
    )
    assert exit_code == 0
    assert payload["task_results"][0]["decision"] == "RUN_NOW"


def test_close_dry_run_never_contains_primary_market_task() -> None:
    config = load_config()
    payload, _ = run_phase(
        config,
        "close",
        datetime(2026, 8, 20, 19, 0, tzinfo=TIMEZONE),
        dry_run=True,
        wait_for_window=False,
    )
    assert [item["task_id"] for item in payload["task_results"]] == [
        "INDUSTRY_EXPECTATION_GAP",
        "PRIORITY_FORWARD_STATUS",
    ]
    assert all(
        "primary_market" not in item["launcher"].lower()
        for item in payload["task_results"]
    )


def test_existing_failure_evidence_prevents_duplicate_morning_run() -> None:
    config = load_config()
    payload, exit_code = run_phase(
        config,
        "morning",
        datetime(2026, 8, 19, 9, 30, tzinfo=TIMEZONE),
        dry_run=True,
        wait_for_window=False,
    )
    assert exit_code == 0
    result = payload["task_results"][0]
    assert result["decision"] == "ALREADY_ATTEMPTED"
    assert result["pre_evidence"] == [
        "reports/data_quality/510300_primary_market_task_status_20260819.json"
    ]


def test_every_orchestration_receipt_disables_trading() -> None:
    config = load_config()
    payload, _ = run_phase(
        config,
        "close",
        datetime(2026, 8, 23, 19, 0, tzinfo=TIMEZONE),
        dry_run=True,
        wait_for_window=False,
    )
    assert payload["research_only"] is True
    assert all(payload[field] is False for field in SAFETY_FIELDS)


def test_config_declares_codex_receipt_and_log_outputs() -> None:
    config = load_config()
    outputs = config["outputs"]
    assert outputs["codex_run_receipt_directory"].startswith("reports/audit/")
    assert outputs["codex_current_status"].endswith(".json")
    assert outputs["codex_log_directory"].startswith("output/")
