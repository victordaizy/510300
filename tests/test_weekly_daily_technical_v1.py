from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.weekly_daily_technical_v1 import (
    build_features,
    load_config,
    simulate_variant,
    wilder_atr,
    wilder_rsi,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config()


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


def _features(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    defaults = {
        "open": 4.0,
        "close": 4.0,
        "adjusted_close": 4.0,
        "adjustment_factor": 1.0,
        "weekly_state_at_close": "W_RANGE",
        "atr14": 0.1,
        "hh20": 5.0,
        "ll10": 3.0,
        "ma20": 4.2,
        "bias28": 0.0,
        "rsi5": 50.0,
    }
    for column, value in defaults.items():
        if column not in frame:
            frame[column] = value
    return frame


def test_wilder_indicators_have_fixed_seed() -> None:
    close = pd.Series([10.0, 11.0, 12.0, 11.0, 13.0, 14.0, 13.0])
    high = close + 1.0
    low = close - 1.0
    atr = wilder_atr(high, low, close, 3)
    rsi = wilder_rsi(close, 3)
    assert atr.first_valid_index() == 2
    assert atr.iloc[2] == pytest.approx((2.0 + 2.0 + 2.0) / 3.0)
    assert rsi.first_valid_index() == 3
    assert rsi.iloc[3] == pytest.approx(2.0 / 3.0 * 100.0)


def test_weekly_state_uses_current_week_only_on_completed_week_close() -> None:
    dates = pd.bdate_range("2025-01-06", periods=190)
    close = np.linspace(2.0, 5.0, len(dates))
    market = pd.DataFrame(
        {
            "date": dates,
            "open": close,
            "high": close + 0.01,
            "low": close - 0.01,
            "close": close,
        }
    )
    features, weekly = build_features(market, _empty_dividends(), CONFIG)
    completed = features.loc[features["is_completed_week_close"]]
    assert not completed.empty
    for row in completed.itertuples(index=False):
        expected = weekly.loc[weekly["last_date"].eq(row.date), "state"].iloc[0]
        assert row.weekly_state_at_close == expected
    non_completed = features.loc[~features["is_completed_week_close"]]
    sourced = non_completed.loc[non_completed["weekly_state_source_date"].notna()]
    assert (sourced["weekly_state_source_date"] < sourced["date"]).all()


def test_mean_reversion_uses_three_next_open_tiers_and_full_exit() -> None:
    dates = pd.bdate_range("2026-01-05", periods=8)
    rows = [
        {"date": dates[0], "bias28": -6.0, "rsi5": 20.0, "open": 4.0, "close": 4.0, "adjusted_close": 4.0},
        {"date": dates[1], "bias28": -11.0, "rsi5": 20.0, "open": 4.0, "close": 3.85, "adjusted_close": 3.85},
        {"date": dates[2], "bias28": -16.0, "rsi5": 20.0, "open": 3.85, "close": 3.70, "adjusted_close": 3.70},
        {"date": dates[3], "bias28": -16.0, "rsi5": 20.0, "open": 3.70, "close": 3.70, "adjusted_close": 3.70},
        {"date": dates[4], "bias28": -5.0, "rsi5": 61.0, "open": 3.80, "close": 4.30, "adjusted_close": 4.30, "ma20": 4.2},
        {"date": dates[5], "open": 4.30, "close": 4.30, "adjusted_close": 4.30},
        {"date": dates[6]},
        {"date": dates[7]},
    ]
    result = simulate_variant(_features(rows), _empty_dividends(), CONFIG, "M_ONLY")
    executions = result["executions"]
    assert executions["reason"].tolist() == [
        "MEAN_ENTRY",
        "MEAN_ADD_2",
        "MEAN_ADD_3",
        "MEAN_EXIT_REVERSION",
    ]
    assert executions["target_exposure"].tolist() == pytest.approx([0.325, 0.65, 0.975, 0.0])
    assert (executions["execution_date"] > executions["signal_date"]).all()
    assert executions.loc[executions["side"].eq("BUY"), "raw_notional"].ge(5000.0).all()


def test_trend_adds_only_in_weekly_bull_and_exits_on_bear() -> None:
    dates = pd.bdate_range("2026-02-02", periods=7)
    rows = [
        {"date": dates[0], "weekly_state_at_close": "W_BULL", "hh20": 4.1, "close": 4.2, "adjusted_close": 4.2},
        {"date": dates[1], "weekly_state_at_close": "W_BULL", "open": 4.2, "close": 4.35, "adjusted_close": 4.35},
        {"date": dates[2], "weekly_state_at_close": "W_RANGE", "open": 4.35, "close": 4.60, "adjusted_close": 4.60},
        {"date": dates[3], "weekly_state_at_close": "W_BULL", "open": 4.60, "close": 4.60, "adjusted_close": 4.60},
        {"date": dates[4], "weekly_state_at_close": "W_BEAR", "open": 4.60, "close": 4.55, "adjusted_close": 4.55},
        {"date": dates[5], "open": 4.55, "close": 4.55, "adjusted_close": 4.55},
        {"date": dates[6]},
    ]
    result = simulate_variant(_features(rows), _empty_dividends(), CONFIG, "T_ONLY")
    reasons = result["executions"]["reason"].tolist()
    assert reasons == [
        "TREND_ENTRY",
        "TREND_ADD_2",
        "TREND_ADD_3",
        "TREND_EXIT_WEEKLY_BEAR",
    ]
    add_signals = result["executions"].loc[
        result["executions"]["reason"].str.startswith("TREND_ADD"), "signal_date"
    ]
    assert dates[2] not in set(add_signals)


def test_minimum_notional_blocks_small_normal_trade_but_not_liquidation() -> None:
    config = load_config()
    config["price_and_execution"]["initial_capital_cny"] = 12000.0
    dates = pd.bdate_range("2026-03-02", periods=4)
    rows = [
        {"date": dates[0], "bias28": -6.0, "rsi5": 20.0},
        {"date": dates[1], "bias28": -6.0, "rsi5": 20.0},
        {"date": dates[2], "bias28": -6.0, "rsi5": 20.0},
        {"date": dates[3]},
    ]
    result = simulate_variant(_features(rows), _empty_dividends(), config, "M_ONLY")
    assert result["executions"].empty
    assert result["skipped_orders"]["rejection"].eq("BELOW_MINIMUM_NOTIONAL").all()


def test_safety_switches_are_disabled_and_forward_start_is_frozen() -> None:
    assert CONFIG["governance"] == {
        "discovery_only": True,
        "live_position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
    }
    assert CONFIG["protocol"]["true_forward_start"] == "2026-08-19"
