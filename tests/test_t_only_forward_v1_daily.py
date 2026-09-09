from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import scripts.run_t_only_forward_v1_daily as daily


TIMEZONE = ZoneInfo("Asia/Shanghai")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_overall_status_distinguishes_operational_success_from_model_result() -> None:
    assert (
        daily.overall_success_status("COLLECTING") == "SUCCESS_COLLECTING"
    )
    assert (
        daily.overall_success_status("PASS_FORWARD_GATES")
        == "SUCCESS_FORWARD_GATES_PASS"
    )
    assert (
        daily.overall_success_status("FAIL_FORWARD_GATES")
        == "SUCCESS_FORWARD_GATES_FAIL_EVIDENCE"
    )


def test_lock_blocks_concurrent_instance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lock = tmp_path / "daily_run.lock"
    monkeypatch.setattr(daily, "LOCK_FILE", lock)
    descriptor = daily.acquire_lock(datetime.now(TIMEZONE))
    try:
        with pytest.raises(RuntimeError, match="已有 T_ONLY"):
            daily.acquire_lock(datetime.now(TIMEZONE))
    finally:
        daily.release_lock(descriptor)
    assert not lock.exists()


def test_output_audit_accepts_consistent_unique_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_gate = tmp_path / "data_gate.json"
    forward_status = tmp_path / "forward.json"
    ledger = tmp_path / "ledger.parquet"
    guide = tmp_path / "guide.json"
    started = datetime.now(TIMEZONE) - timedelta(seconds=1)
    _write_json(
        data_gate,
        {
            "status": "PASS",
            "retrieved_at": datetime.now(TIMEZONE).isoformat(),
            "actual_last_date": "2026-08-20",
        },
    )
    _write_json(
        forward_status,
        {
            "status": "COLLECTING",
            "as_of_market_date": "2026-08-20",
            "forward_signal_start": "2026-08-19",
            "new_trading_days": 2,
            "closed_cycles": 0,
            "current_view": "NO_VIEW_UNTIL_FORWARD_MATURITY",
            "current_shadow_signal": None,
        },
    )
    _write_json(
        guide,
        {
            "as_of_market_date": "2026-08-20",
            "forward_status": "COLLECTING",
            "decision": "NO_ACTION_NO_SIGNAL",
            "headline": "无影子动作",
        },
    )
    pd.DataFrame(
        {"date": pd.to_datetime(["2026-08-19", "2026-08-20"]), "equity": [20000, 20001]}
    ).to_parquet(ledger, index=False)
    monkeypatch.setattr(daily, "DATA_GATE", data_gate)
    monkeypatch.setattr(daily, "FORWARD_STATUS", forward_status)
    monkeypatch.setattr(daily, "LEDGER", ledger)
    monkeypatch.setattr(daily, "DAILY_GUIDE", guide)
    result = daily.audit_outputs(started)
    assert result["status"] == "PASS"
    assert result["ledger_rows"] == 2
    assert result["duplicate_ledger_dates"] == 0


def test_output_audit_rejects_old_data_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_gate = tmp_path / "data_gate.json"
    forward_status = tmp_path / "forward.json"
    guide = tmp_path / "guide.json"
    started = datetime.now(TIMEZONE)
    _write_json(
        data_gate,
        {
            "status": "PASS",
            "retrieved_at": (started - timedelta(minutes=1)).isoformat(),
            "actual_last_date": "2026-08-18",
        },
    )
    _write_json(
        forward_status,
        {
            "status": "COLLECTING_FORWARD_NOT_STARTED",
            "as_of_market_date": "2026-08-18",
            "forward_signal_start": "2026-08-19",
            "new_trading_days": 0,
            "closed_cycles": 0,
        },
    )
    _write_json(
        guide,
        {
            "as_of_market_date": "2026-08-18",
            "forward_status": "COLLECTING_FORWARD_NOT_STARTED",
            "decision": "NO_ACTION_FORWARD_NOT_STARTED",
            "headline": "尚未开始",
        },
    )
    monkeypatch.setattr(daily, "DATA_GATE", data_gate)
    monkeypatch.setattr(daily, "FORWARD_STATUS", forward_status)
    monkeypatch.setattr(daily, "DAILY_GUIDE", guide)
    with pytest.raises(RuntimeError, match="本次运行中刷新"):
        daily.audit_outputs(started)
