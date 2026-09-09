import numpy as np
import pandas as pd
import pytest

from research.valuation_v3_state_aware_return import (
    exact_non_overlapping_queues,
    estimate_component_annual_reversion_speed,
    estimate_walk_forward_reversion_speed,
    newey_west_slope,
    v3_expected_return,
)


def test_exact_non_overlapping_queues_never_share_horizon() -> None:
    trading_dates = pd.bdate_range("2020-01-01", periods=900)
    signal_dates = trading_dates[::21]
    sample = pd.DataFrame(
        {
            "date": signal_dates,
            "signal": np.arange(len(signal_dates), dtype=float),
            "target": np.arange(len(signal_dates), dtype=float),
        }
    )
    queues = exact_non_overlapping_queues(
        sample,
        trading_dates=pd.Series(trading_dates),
        horizon_days=242,
        signal_column="signal",
        target_column="target",
    )
    assert queues
    for queue in queues:
        positions = queue["trading_positions"]
        assert all(right - left >= 242 for left, right in zip(positions, positions[1:]))


def test_newey_west_uses_requested_overlap_lag() -> None:
    x = np.linspace(-1.0, 1.0, 40)
    y = 0.5 * x + np.sin(np.arange(40)) * 0.05
    result = newey_west_slope(x, y, max_lag=11)
    assert result["max_lag"] == 11
    assert result["slope"] == pytest.approx(0.5, abs=0.03)
    assert result["hac_standard_error"] > 0


def test_v3_growth_enters_future_eps_only_once() -> None:
    result = v3_expected_return(
        normalized_eps=10.0,
        current_index=150.0,
        current_normalized_pe=15.0,
        annual_growth=0.10,
        fair_pe=12.0,
        annual_dividend_yield=0.02,
        annual_cash_rate=0.01,
        horizon_days=242,
        annual_reversion_speed=0.30,
    )
    assert result["future_normalized_eps"] == pytest.approx(11.0)
    assert result["target_pe"] == pytest.approx(14.1)
    expected_price_return = 11.0 * 14.1 / 150.0 - 1.0
    assert result["expected_price_return"] == pytest.approx(expected_price_return)
    assert result["expected_total_return"] == pytest.approx(expected_price_return + 0.02)


def test_reversion_speed_at_signal_never_uses_future_pair() -> None:
    dates = pd.date_range("2018-01-31", periods=36, freq="ME")
    frame = pd.DataFrame(
        {
            "date": dates,
            "pe_ttm": 12.0 + np.sin(np.arange(36) / 3.0),
            "historical_fair_pe": np.repeat(12.0, 36),
        }
    )
    original = estimate_walk_forward_reversion_speed(frame, minimum_pairs=8)
    perturbed = frame.copy()
    perturbed.loc[perturbed["date"].gt(dates[24]), "pe_ttm"] = 99.0
    changed = estimate_walk_forward_reversion_speed(perturbed, minimum_pairs=8)
    cutoff = dates[24]
    left = original.loc[original["date"].le(cutoff), "annual_reversion_speed"]
    right = changed.loc[changed["date"].le(cutoff), "annual_reversion_speed"]
    pd.testing.assert_series_equal(left.reset_index(drop=True), right.reset_index(drop=True))


def test_component_annual_speed_uses_only_completed_same_basis_pairs() -> None:
    dates = pd.date_range("2020-01-31", periods=36, freq="ME")
    frame = pd.DataFrame(
        {
            "date": dates,
            "component_normalized_pe": np.linspace(16.0, 12.0, 36),
            "conditioned_fair_pe_v2": np.repeat(10.0, 36),
        }
    )
    original = estimate_component_annual_reversion_speed(frame, minimum_pairs=4)
    perturbed = frame.copy()
    perturbed.loc[perturbed["date"].gt(dates[27]), "component_normalized_pe"] = 99.0
    changed = estimate_component_annual_reversion_speed(perturbed, minimum_pairs=4)
    left = original.loc[original["date"].le(dates[27]), "annual_reversion_speed"]
    right = changed.loc[changed["date"].le(dates[27]), "annual_reversion_speed"]
    pd.testing.assert_series_equal(left.reset_index(drop=True), right.reset_index(drop=True))
    valid = original.dropna(subset=["annual_reversion_speed"])
    assert valid.iloc[0]["reversion_training_last_end_date"] <= valid.iloc[0]["date"]
