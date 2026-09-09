"""核对方差估计、因果窗口及状态只选择进入模式。"""
import numpy as np
import pandas as pd
import pytest

from research.variance_ratio_entry_router_v1 import CONTROL, PRIMARY, make_rules, rolling_ratio, variance_ratio


def test_variance_ratio_matches_log_level_overlap_definition():
    returns = np.array([.02, -.01, .03, .01, -.02, .015, .004, -.013, .008, .03])
    level = np.r_[0., returns.cumsum()]
    mu = (level[-1] - level[0]) / 10
    one = sum((level[1:] - level[:-1] - mu) ** 2) / 9
    multi = sum((level[5:] - level[:-5] - 5 * mu) ** 2) / (5 * 6 * .5)
    assert variance_ratio(returns, 5) == pytest.approx(multi / one, abs=1e-12)


def test_drift_and_scale_do_not_change_ratio():
    values = np.sin(np.arange(60) / 3) * .01
    assert variance_ratio(values + .007) == pytest.approx(variance_ratio(values), abs=1e-12)
    assert variance_ratio(values * -3) == pytest.approx(variance_ratio(values), abs=1e-12)


def test_serial_runs_and_alternation_have_different_variance_scaling():
    runs = np.repeat([.01, -.01, .01, -.01, .01, -.01], 10)
    alternating = np.tile([.01, -.01], 30)
    assert variance_ratio(runs) > 1
    assert variance_ratio(alternating) < 1


def test_missing_and_zero_variance_do_not_become_neutral_state():
    assert np.isnan(variance_ratio(np.zeros(60)))
    assert np.isnan(variance_ratio(np.full(60, .01)))
    x = np.arange(60, dtype=float) / 10000
    x[20] = np.nan
    assert np.isnan(variance_ratio(x))
    result = rolling_ratio(pd.Series(np.sin(np.arange(100))), 60, 5)
    assert result.iloc[:59].isna().all()
    assert np.isfinite(result.iloc[59:]).all()


def test_future_returns_do_not_change_past_ratio():
    x = pd.Series(np.sin(np.arange(160)) / 100)
    y = x.copy()
    y.iloc[100:] = .5
    pd.testing.assert_series_equal(rolling_ratio(x).iloc[:100], rolling_ratio(y).iloc[:100])


def test_router_preserves_modes_exits_and_ungated_missing_behavior():
    n = 80
    w = pd.Series(np.arange(n, dtype=float) + 100)
    data = pd.DataFrame({"wealth": w, "sma120": .1, "efficiency20": .5, "sma60": .03, "sma20": .01,
                         "mom20": .02, "mom5": .01, "z20": 0., "close_location": .5, "rsi2": 50., "dd20": 0., "feature_valid": True})
    data.loc[50, "wealth"] = data.wealth.iloc[48] - .25
    data.loc[49, "wealth"] = data.wealth.iloc[48] - .5
    data.loc[50, "z20"] = -2.
    ratio = pd.Series(1.2, index=data.index)
    ratio.iloc[50] = .8
    ratio.iloc[60] = np.nan
    rules, trend, rebound = make_rules(data, ratio)
    assert rules[PRIMARY]["entry"][40] == 1
    assert rules[PRIMARY]["entry"][50] == 2
    assert not trend[50] and rebound[50]
    assert rules[PRIMARY]["entry"][60] == 0
    assert rules[CONTROL]["entry"][60] == 1
    for mode in [1, 2]:
        np.testing.assert_array_equal(rules[PRIMARY]["exit"][mode], rules[CONTROL]["exit"][mode])
