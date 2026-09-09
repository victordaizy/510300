"""核对高低价因子的数值、分红转换和时间边界。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from research.high_low_relation_v1 import adjusted_levels, build_factors, make_rule


def sample_data(n=400):
    t = np.arange(n, dtype=float)
    low = 10 + .015 * t + .15 * np.sin(t / 7)
    high = low + .3 + .07 * np.cos(t / 11)
    close = (high + low) / 2
    return pd.DataFrame({"date": pd.bdate_range("2010-01-01", periods=n),
                         "low": low, "high": high, "close": close,
                         "dividend": np.zeros(n), "wealth": close})


def test_affine_high_low_recovers_slope_and_quality_without_fabricated_score():
    data = sample_data()
    data["high"] = 2 * data.low + 3
    data["close"] = (data.high + data.low) / 2
    data["wealth"] = data.close
    factors = build_factors(data)
    np.testing.assert_allclose(factors.slope.iloc[19:], 2, atol=1e-12)
    np.testing.assert_allclose(factors.fit_quality.iloc[19:], 1, atol=1e-12)
    assert factors.H1_Z.isna().all()
    assert not (factors.factor_state == "VALUE_AVAILABLE").any()


def test_ex_dividend_high_low_and_close_share_the_same_total_return_scale():
    data = pd.DataFrame({"high": [10.], "low": [9.6], "close": [9.8],
                         "dividend": [.2], "wealth": [100.]})
    high, low = adjusted_levels(data)
    np.testing.assert_allclose(high, [102.])
    np.testing.assert_allclose(low, [98.])
    np.testing.assert_allclose((data.close + data.dividend) * data.wealth / (data.close + data.dividend), [100.])
    assert not np.isclose(high.iloc[0], data.high.iloc[0] * data.wealth.iloc[0] / data.close.iloc[0])


def test_future_extension_and_changes_cannot_change_prior_factors():
    data = sample_data(700)
    cutoff = 370
    prefix = build_factors(data.iloc[:cutoff].copy())
    changed = data.copy()
    changed.loc[cutoff:, "high"] *= 3
    changed.loc[cutoff:, "low"] *= .3
    changed.loc[cutoff:, "wealth"] *= 2
    complete = build_factors(changed)
    assert_frame_equal(prefix, complete.iloc[:cutoff])


def test_standardization_uses_complete_history_excluding_current_slope():
    data = sample_data(45)
    factors = build_factors(data, window=6, history=9)
    assert factors.H1_Z.iloc[:14].isna().all()
    assert np.isfinite(factors.H1_Z.iloc[14])
    t = 25
    past = factors.slope.iloc[t - 9:t].to_numpy()
    expected = (factors.slope.iloc[t] - past.mean()) / past.std(ddof=1)
    np.testing.assert_allclose(factors.previous_slope_mean.iloc[t], past.mean())
    np.testing.assert_allclose(factors.previous_slope_std.iloc[t], past.std(ddof=1))
    np.testing.assert_allclose(factors.H1_Z.iloc[t], expected)
    changed = data.copy()
    changed.loc[t, "high"] += 3
    altered = build_factors(changed, window=6, history=9)
    np.testing.assert_allclose(altered.previous_slope_mean.iloc[t], factors.previous_slope_mean.iloc[t])
    np.testing.assert_allclose(altered.previous_slope_std.iloc[t], factors.previous_slope_std.iloc[t])
    assert not np.isclose(altered.H1_Z.iloc[t], factors.H1_Z.iloc[t])


def test_missing_observation_requires_a_new_complete_contiguous_history():
    data = sample_data(60)
    data.loc[20, "high"] = np.nan
    factors = build_factors(data, window=6, history=9)
    assert factors.slope.iloc[20:26].isna().all()
    assert factors.H1_Z.iloc[20:35].isna().all()
    assert np.isfinite(factors.H1_Z.iloc[35])


def test_negative_slope_cannot_become_a_buy_when_two_negative_factors_multiply():
    factors = pd.DataFrame({"H3_RIGHT": [1., 1., np.nan, .7, -.7, -.8],
                            "slope": [-1., 1., np.nan, 1., 1., 1.]})
    rule = make_rule(factors, "H3_RIGHT")
    assert list(rule["entry"]) == [0, 1, 0, 0, 0, 0]
    assert list(rule["exit"][1]) == [True, False, False, False, False, True]
