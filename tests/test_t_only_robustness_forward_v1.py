from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.t_only_robustness_forward_v1 import (
    load_stress_config,
    replication_audit,
    select_missed_entry_dates,
    simulate_t_only,
)
from research.weekly_daily_technical_v1 import (
    build_features,
    load_config,
    load_inputs,
    simulate_variant,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = load_config()
STRESS_CONFIG = load_stress_config()


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype=str),
            "record_date": pd.Series(dtype="datetime64[ns]"),
            "ex_date": pd.Series(dtype="datetime64[ns]"),
            "payment_date": pd.Series(dtype="datetime64[ns]"),
            "cash_dividend_per_share": pd.Series(dtype=float),
            "source": pd.Series(dtype=str),
        }
    )


def _features() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=8)
    rows = [
        {
            "date": dates[0],
            "open": 4.0,
            "close": 4.2,
            "adjusted_close": 4.2,
            "adjustment_factor": 1.0,
            "weekly_state_at_close": "W_BULL",
            "atr14": 0.1,
            "hh20": 4.1,
            "ll10": 3.5,
        },
        {
            "date": dates[1],
            "open": 4.2,
            "close": 4.35,
            "adjusted_close": 4.35,
            "adjustment_factor": 1.0,
            "weekly_state_at_close": "W_BULL",
            "atr14": 0.1,
            "hh20": 4.3,
            "ll10": 3.5,
        },
        {
            "date": dates[2],
            "open": 4.35,
            "close": 4.50,
            "adjusted_close": 4.50,
            "adjustment_factor": 1.0,
            "weekly_state_at_close": "W_BULL",
            "atr14": 0.1,
            "hh20": 4.4,
            "ll10": 3.5,
        },
        {
            "date": dates[3],
            "open": 4.50,
            "close": 4.65,
            "adjusted_close": 4.65,
            "adjustment_factor": 1.0,
            "weekly_state_at_close": "W_BULL",
            "atr14": 0.1,
            "hh20": 4.6,
            "ll10": 3.5,
        },
        {
            "date": dates[4],
            "open": 4.65,
            "close": 4.20,
            "adjusted_close": 4.20,
            "adjustment_factor": 1.0,
            "weekly_state_at_close": "W_BEAR",
            "atr14": 0.1,
            "hh20": 4.7,
            "ll10": 4.3,
        },
        {
            "date": dates[5],
            "open": 4.15,
            "close": 4.10,
            "adjusted_close": 4.10,
            "adjustment_factor": 1.0,
            "weekly_state_at_close": "W_BEAR",
            "atr14": 0.1,
            "hh20": 4.7,
            "ll10": 4.2,
        },
        {
            "date": dates[6],
            "open": 4.05,
            "close": 4.00,
            "adjusted_close": 4.00,
            "adjustment_factor": 1.0,
            "weekly_state_at_close": "W_BEAR",
            "atr14": 0.1,
            "hh20": 4.7,
            "ll10": 4.1,
        },
        {
            "date": dates[7],
            "open": 4.00,
            "close": 4.00,
            "adjusted_close": 4.00,
            "adjustment_factor": 1.0,
            "weekly_state_at_close": "W_RANGE",
            "atr14": 0.1,
            "hh20": 4.7,
            "ll10": 4.0,
        },
    ]
    return pd.DataFrame(rows)


def test_specialized_baseline_matches_parent_on_frozen_history() -> None:
    market, _, dividends = load_inputs(ROOT, BASE_CONFIG)
    features, _ = build_features(market, dividends, BASE_CONFIG)
    parent = simulate_variant(features, dividends, BASE_CONFIG, "T_ONLY")
    replica = simulate_t_only(
        features,
        dividends,
        BASE_CONFIG,
        scenario_id="BASELINE_REPLICATION",
    )
    assert replication_audit(parent, replica)["status"] == "PASS"


def test_extra_delay_moves_every_execution_one_additional_bar() -> None:
    features = _features()
    dividends = _empty_dividends()
    baseline = simulate_t_only(
        features,
        dividends,
        BASE_CONFIG,
        scenario_id="BASELINE_REPLICATION",
    )
    delayed = simulate_t_only(
        features,
        dividends,
        BASE_CONFIG,
        scenario_id="DELAY_1_EXTRA_BAR",
        execution_delay_bars=1,
    )
    baseline_first = baseline["executions"].iloc[0]
    delayed_first = delayed["executions"].iloc[0]
    assert baseline_first["execution_date"] == features.loc[1, "date"]
    assert delayed_first["execution_date"] == features.loc[2, "date"]
    assert delayed_first["signal_date"] == features.loc[0, "date"]


def test_blocked_entry_is_recorded_and_not_executed() -> None:
    features = _features()
    blocked = [features.loc[0, "date"]]
    result = simulate_t_only(
        features,
        _empty_dividends(),
        BASE_CONFIG,
        scenario_id="MISS_10PCT_ENTRIES",
        blocked_entry_signal_dates=blocked,
    )
    assert result["missed_signals"]["signal_date"].tolist() == blocked
    assert features.loc[0, "date"] not in set(result["executions"]["signal_date"])


def test_missed_entry_selection_is_deterministic() -> None:
    dates = pd.bdate_range("2025-01-01", periods=10)
    executions = pd.DataFrame(
        {"reason": ["TREND_ENTRY"] * 10, "signal_date": dates}
    )
    first = select_missed_entry_dates(executions, fraction=0.1, random_seed=20260818)
    second = select_missed_entry_dates(executions, fraction=0.1, random_seed=20260818)
    assert first == second
    assert len(first) == 1


def test_forward_governance_and_boundary_are_frozen() -> None:
    assert STRESS_CONFIG["protocol"]["forward_signal_start"] == "2026-08-19"
    assert STRESS_CONFIG["protocol"]["earliest_forward_execution"] == "2026-08-20"
    assert STRESS_CONFIG["governance"]["shadow_ledger_only"] is True
    assert STRESS_CONFIG["governance"]["live_position_mapping_enabled"] is False
    assert STRESS_CONFIG["governance"]["order_generation_enabled"] is False
    assert STRESS_CONFIG["governance"]["broker_connection_enabled"] is False
