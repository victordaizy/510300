from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.volatility_compression_change_point import (
    build_events,
    build_features,
    load_config,
    recent_state,
    select_event_indices,
)


CONFIG = load_config()


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_date": pd.Series(dtype="datetime64[ns]"),
            "ex_date": pd.Series(dtype="datetime64[ns]"),
            "payment_date": pd.Series(dtype="datetime64[ns]"),
            "cash_dividend_per_share": pd.Series(dtype=float),
        }
    )


def test_event_selector_uses_first_true_day_and_twenty_day_cooldown() -> None:
    condition = pd.Series(
        [False] * 3
        + [True] * 5
        + [False] * 3
        + [True] * 2
        + [False] * 15
        + [True] * 2
    )
    assert select_event_indices(condition, cooldown=20) == [3, 28]


def test_literal_rule_detects_high_to_low_monthly_amplitude_transition() -> None:
    amplitudes = [0.02] * 40 + [0.008] * 30
    close = np.full(len(amplitudes), 100.0)
    market = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=len(amplitudes)),
            "open": close,
            "high": close + np.asarray(amplitudes) * close / 2.0,
            "low": close - np.asarray(amplitudes) * close / 2.0,
            "close": close,
        }
    )
    features = build_features(market, _empty_dividends(), CONFIG)
    assert features["literal_condition"].any()
    first = features.index[features["literal_condition"]][0]
    assert 0.005 <= features.loc[first, "mean_amplitude_20"] <= 0.010
    assert 0.015 <= features.loc[first, "prior_mean_amplitude_20"] <= 0.025


def test_outcomes_start_after_event_day_and_do_not_include_event_amplitude() -> None:
    rows = 30
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-02", periods=rows),
            "adjusted_close": np.linspace(100.0, 103.0, rows),
            "adjusted_high": np.linspace(100.5, 103.5, rows),
            "adjusted_low": np.linspace(99.5, 102.5, rows),
            "daily_amplitude": [0.01] * rows,
            "mean_amplitude_20": [0.01] * rows,
            "prior_mean_amplitude_20": [0.02] * rows,
            "rv5": [0.10] * rows,
            "rv20": [0.12] * rows,
            "rv60": [0.15] * rows,
            "rv20_trailing_q20": [0.13] * rows,
            "past20_return": [0.01] * rows,
            "prior20_high": [101.0] * rows,
            "prior20_low": [99.0] * rows,
            "literal_condition": [False] * 5 + [True] + [False] * 24,
            "adaptive_condition": False,
        }
    )
    frame.loc[5, "daily_amplitude"] = 0.50
    frame.loc[6:25, "daily_amplitude"] = 0.02
    events = build_events(frame, CONFIG)
    event = events.iloc[0]
    assert event["event_index"] == 5
    assert event["future_mean_amplitude_20d"] == pytest.approx(0.02)
    assert event["future_amplitude_ratio_20d"] == pytest.approx(2.0)


def test_trailing_rv_quantile_excludes_current_observation() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, 900)
    close = 100.0 * np.exp(np.cumsum(returns))
    market = pd.DataFrame(
        {
            "date": pd.bdate_range("2022-01-03", periods=len(close)),
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
        }
    )
    features = build_features(market, _empty_dividends(), CONFIG)
    index = 899
    expected = features["rv20"].iloc[index - 756 : index].quantile(0.20)
    assert features.loc[index, "rv20_trailing_q20"] == pytest.approx(expected)


def test_recent_state_and_safety_have_no_trade_mapping() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-14"]),
            "mean_amplitude_20": [0.008],
            "prior_mean_amplitude_20": [0.012],
            "rv5": [0.08],
            "rv20": [0.10],
            "rv60": [0.12],
            "rv20_trailing_q20": [0.11],
            "literal_condition": [False],
            "adaptive_condition": [True],
        }
    )
    events = pd.DataFrame(
        {
            "rule_id": ["ADAPTIVE"],
            "event_date": pd.to_datetime(["2026-08-14"]),
        }
    )
    state = recent_state(frame, events)
    assert state["adaptive_condition"] is True
    governance = CONFIG["governance"]
    assert governance["live_position_mapping_enabled"] is False
    assert governance["order_generation_enabled"] is False
    assert governance["broker_connection_enabled"] is False
