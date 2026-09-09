"""T_ONLY逐日前瞻事件哈希链测试。"""

from __future__ import annotations

import copy

import pandas as pd
import pytest

from research.t_only_forward_event_ledger import (
    ForwardLedgerError,
    build_daily_event_records,
    reconcile_append_only_events,
    verify_event_chain,
)


def _calendar() -> pd.DataFrame:
    return pd.DataFrame(
        {"trade_date": pd.to_datetime(["2026-08-19", "2026-08-20", "2026-08-21"])}
    )


def _ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-19", "2026-08-20"]),
            "mode": ["CASH", "TREND"],
            "tier": [0, 1],
            "shares": [0, 1000],
            "day_commission": [0.0, 5.0],
            "day_slippage": [0.0, 2.4],
            "day_raw_notional": [0.0, 4800.0],
            "executed_reason": [None, "TREND_ENTRY"],
            "next_signal_reason": ["TREND_ENTRY", None],
        }
    )


def _executions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2026-08-19"]),
            "execution_date": pd.to_datetime(["2026-08-20"]),
            "reason": ["TREND_ENTRY"],
            "side": ["BUY"],
            "trade_shares": [1000],
            "raw_notional": [4800.0],
            "commission": [5.0],
            "slippage_cost": [2.4],
            "target_exposure": [0.325],
        }
    )


def _cycles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_date": pd.to_datetime(["2026-08-19"]),
            "exit_date": pd.to_datetime(["2026-08-20"]),
            "entry_reason": ["TREND_ENTRY"],
            "exit_reason": ["TREND_EXIT_WEEKLY_BEAR"],
            "holding_calendar_days": [1],
            "net_pnl": [123.45],
        }
    )


def test_daily_event_contains_signal_effective_execution_cost_and_hash_chain() -> None:
    records = build_daily_event_records(
        daily_ledger=_ledger(),
        executions=_executions(),
        cycles=_cycles(),
        calendar=_calendar(),
        action_targets={"TREND_ENTRY": 0.325},
    )
    verify_event_chain(records)
    assert records[0]["signal"]["signal_time"] == "2026-08-19T16:30:00+08:00"
    assert records[0]["signal"]["effective_time"] == "2026-08-20T09:30:00+08:00"
    assert records[1]["executions"][0]["signal_date"] == "2026-08-19"
    assert records[1]["daily_costs"]["commission"] == 5.0
    assert records[1]["closed_cycle_count"] == 1
    assert records[1]["closed_cycles"][0]["entry_date"] == "2026-08-19"
    assert records[1]["performance_metrics_disclosed"] is False
    assert "equity" not in records[1]
    assert "net_pnl" not in records[1]


def test_existing_event_mutation_is_rejected() -> None:
    records = build_daily_event_records(
        daily_ledger=_ledger(),
        executions=_executions(),
        cycles=_cycles(),
        calendar=_calendar(),
        action_targets={"TREND_ENTRY": 0.325},
    )
    mutated = copy.deepcopy(records)
    mutated[0]["shadow_state"]["shares"] = 100
    with pytest.raises(ForwardLedgerError, match="内容哈希"):
        reconcile_append_only_events(mutated, records)


def test_calendar_gap_is_rejected_instead_of_backfilled() -> None:
    ledger = _ledger().iloc[[0]].copy()
    extra = _ledger().iloc[[1]].copy()
    extra["date"] = pd.Timestamp("2026-08-21")
    ledger = pd.concat([ledger, extra], ignore_index=True)
    with pytest.raises(ForwardLedgerError, match="交易日历不连续"):
        build_daily_event_records(
            daily_ledger=ledger,
            executions=pd.DataFrame(),
            cycles=pd.DataFrame(),
            calendar=_calendar(),
            action_targets={"TREND_ENTRY": 0.325},
        )
