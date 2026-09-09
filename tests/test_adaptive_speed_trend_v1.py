"""验证递推因子的可见时点、缺失中断、方向效率边界和双收盘确认。"""
import numpy as np
import pandas as pd
from research.adaptive_speed_trend_v1 import build_factors, make_rule


def frame(values):
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(values)), "wealth": values})


def test_prefix_is_unchanged_when_future_changes():
    values = 10 + np.sin(np.arange(80) / 4) + np.arange(80) / 50
    original = build_factors(frame(values))
    altered = values.copy()
    altered[50:] *= 5
    pd.testing.assert_frame_equal(original.iloc[:50], build_factors(frame(altered)).iloc[:50])


def test_monotone_path_has_fastest_gain_and_known_geometric_lag():
    factors = build_factors(frame(1 + np.arange(40, dtype=float)))
    np.testing.assert_allclose(factors.direction_efficiency10.iloc[10:], 1.)
    np.testing.assert_allclose(factors.adaptive_gain.iloc[10:], 4 / 9)
    # 单位线性增长时，初始化滞后为零，递推滞后有封闭解。
    steps = np.arange(1, 31)
    lag = (1 - 4 / 9) / (4 / 9) * (1 - (1 - 4 / 9) ** steps)
    np.testing.assert_allclose(factors.wealth.iloc[10:] - factors.adaptive_line.iloc[10:], lag)


def test_flat_and_alternating_paths_have_zero_efficiency():
    flat = build_factors(frame(np.ones(40) * 7))
    np.testing.assert_allclose(flat.adaptive_line.iloc[10:], 7)
    np.testing.assert_allclose(flat.direction_efficiency10.iloc[10:], 0)
    alternating = build_factors(frame(10 + np.arange(40) % 2))
    np.testing.assert_allclose(alternating.direction_efficiency10.iloc[10:], 0)
    np.testing.assert_allclose(alternating.adaptive_gain.iloc[10:], (2 / 31) ** 2)


def test_missing_row_breaks_state_and_requires_complete_window():
    values = np.linspace(10, 15, 60)
    values[30] = np.nan
    factors = build_factors(frame(values))
    assert factors.factor_state.iloc[30:41].eq("NO_VIEW").all()
    assert factors.adaptive_line.iloc[30:41].isna().all()
    assert factors.factor_state.iloc[41] == "VALUE_AVAILABLE"
    for key in ["ADAPTIVE_SPEED", "FIXED_NEUTRAL_SPEED"]:
        rule = make_rule(factors, key)
        assert not rule["entry"][30:42].any()
        assert not rule["exit"][1][30:42].any()


def test_two_distinct_closes_required_and_equality_does_not_confirm():
    factors = frame([10., 11., 12., 10., 9., 8., 10., 11.])
    factors["adaptive_line"] = 10.
    factors["fixed_line"] = 10.
    factors["factor_state"] = "VALUE_AVAILABLE"
    for key in ["ADAPTIVE_SPEED", "FIXED_NEUTRAL_SPEED"]:
        rule = make_rule(factors, key)
        assert np.flatnonzero(rule["entry"]).tolist() == [2]
        assert np.flatnonzero(rule["exit"][1]).tolist() == [5]
