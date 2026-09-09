from __future__ import annotations

import numpy as np
import pandas as pd

from src.a_share_hs_extreme_compression_resolution_alpha_v2 import (
    classify_expansion_lead_lag,
    classify_expansion_price_state,
    classify_quintile,
    compound_excess,
    compute_market_volatility_state,
    cumulative_quintile_membership,
    event_date_equal_mean,
    first_cross_after_origins,
    holm_adjust,
    trailing_valid_midrank_percentile,
)


def test_trailing_midrank_excludes_current_and_handles_ties() -> None:
    values = np.array([1.0, 1.0, 2.0, 2.0, 2.0])
    percentile, count = trailing_valid_midrank_percentile(
        values,
        lookback_valid_values=4,
        minimum_valid_history=4,
    )
    assert np.isnan(percentile[:4]).all()
    assert percentile[4] == 0.75
    assert count.tolist() == [0, 1, 2, 3, 4]


def test_market_state_is_point_in_time_and_future_changes_do_not_rewrite_history() -> None:
    dates = pd.bdate_range("2020-01-01", periods=340)
    close = 100.0 * np.exp(np.cumsum(0.0002 + 0.01 * np.sin(np.arange(340) / 9.0)))
    original = compute_market_volatility_state(
        pd.DataFrame({"date": dates, "close": close}),
        percentile_lookback=60,
        percentile_minimum_history=40,
    )
    changed = close.copy()
    changed[-10:] *= np.linspace(1.0, 1.8, 10)
    revised = compute_market_volatility_state(
        pd.DataFrame({"date": dates, "close": changed}),
        percentile_lookback=60,
        percentile_minimum_history=40,
    )
    columns = ["market_rv20", "market_vol_percentile_756", "market_log_rv5_div_rv20"]
    pd.testing.assert_frame_equal(original.loc[:329, columns], revised.loc[:329, columns])


def test_fixed_quintiles_and_cumulative_membership() -> None:
    assert [classify_quintile(value) for value in [0.2, 0.4, 0.6, 0.8, 0.81]] == [
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
    ]
    assert cumulative_quintile_membership("Q3") == ("Q1", "Q2", "Q3")


def test_first_cross_search_is_strictly_after_origin_and_window_bounded() -> None:
    positions, offsets = first_cross_after_origins(
        [5, 10, 25],
        np.array([5, 6, 20, 30]),
        maximum_offset=10,
    )
    assert positions.tolist() == [10, 10, 25, -1]
    assert offsets.tolist() == [5, 4, 5, -1]


def test_expansion_price_and_lead_lag_states() -> None:
    assert classify_expansion_price_state(11.0, 8.0, 10.0) == "EXPANSION_UP_BREAK"
    assert classify_expansion_price_state(7.0, 8.0, 10.0) == "EXPANSION_DOWN_BREAK"
    assert classify_expansion_price_state(9.0, 8.0, 10.0) == "EXPANSION_WITHIN_RANGE"
    assert classify_expansion_lead_lag(3, 5) == "STOCK_EXPANDS_FIRST"
    assert classify_expansion_lead_lag(5, 3) == "MARKET_EXPANDS_FIRST"
    assert classify_expansion_lead_lag(4, 4) == "SAME_DAY"
    assert classify_expansion_lead_lag(np.nan, 4) == "ONE_SIDE_NOT_OBSERVED"


def test_compound_excess_is_ratio_not_simple_difference() -> None:
    actual = compound_excess(np.array([0.20]), np.array([0.10]))
    assert actual[0] == (1.20 / 1.10 - 1.0)
    assert actual[0] != 0.10


def test_holm_and_event_date_equal_weighting() -> None:
    adjusted = holm_adjust([0.01, 0.04, 0.10])
    np.testing.assert_allclose(adjusted, [0.03, 0.08, 0.10])
    frame = pd.DataFrame(
        {
            "event_date": ["2026-01-01", "2026-01-01", "2026-01-02"],
            "value": [1.0, 3.0, 8.0],
        }
    )
    assert event_date_equal_mean(frame, "value") == 5.0
