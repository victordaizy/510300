"""只检验会改变交易决定的启动、排名、缺失及时间方向。"""
import numpy as np
import pandas as pd

from research.composite_streak_reversal_inputs_v1 import (
    factor_frame, prior_return_rank, signed_streak, trading_rule, wilder_rsi,
)


def test_wilder_manual_seed_and_recursion():
    values = wilder_rsi([10, 11, 12, 11, 12], 3)
    assert np.isnan(values[:3]).all()
    np.testing.assert_allclose(values[3:], [200/3, 700/9])
    assert wilder_rsi([2, 2, 2, 2], 3)[-1] == 50
    assert wilder_rsi([1, 2, 3, 4], 3)[-1] == 100
    assert wilder_rsi([4, 3, 2, 1], 3)[-1] == 0


def test_streak_turn_flat_and_missing():
    expected = [0, 1, 2, 0, -1, -2, 1, np.nan, 0, -1]
    np.testing.assert_allclose(signed_streak([10, 11, 12, 12, 11, 10, 11, np.nan, 11, 10]), expected, equal_nan=True)


def test_rank_excludes_today_and_equal_values():
    values = prior_return_rank([np.nan, .01, .01, -.01, .01, .02], 3)
    assert np.isnan(values[:4]).all()
    np.testing.assert_allclose(values[4:], [100/3, 100])
    assert prior_return_rank([0., 0., 0., 0.], 3)[-1] == 0


def test_future_values_do_not_change_prefix():
    data = pd.DataFrame({"date": pd.bdate_range("2000-01-03", periods=340),
        "wealth": np.exp(np.arange(340)*.001 + np.sin(np.arange(340))*.02)})
    short = factor_frame(data.iloc[:250])
    altered = data.copy()
    altered.loc[250:, "wealth"] *= 10
    pd.testing.assert_frame_equal(short, factor_frame(altered).iloc[:250])
    assert not short.factor_valid.iloc[:199].any()
    assert short.factor_valid.iloc[199:].all()


def test_missing_value_restarts_full_warmup():
    data = pd.DataFrame({"date": pd.bdate_range("2000-01-03", periods=550), "wealth": np.linspace(1, 2, 550)})
    data.loc[220, "wealth"] = np.nan
    factors = factor_frame(data)
    assert not factors.factor_valid.iloc[220:420].any()
    assert factors.factor_valid.iloc[420:].all()
    assert factors.streak_rsi2.iloc[220:223].isna().all()
    assert np.isfinite(factors.streak_rsi2.iloc[223])


def test_entry_exit_boundary_and_no_view():
    f = pd.DataFrame({"wealth_close": [2., 2., 2., 2., 1., 1.], "wealth_sma200": [1.]*6,
        "composite_score": [9., 10., 70., 71., 9., np.nan], "factor_valid": [True]*5+[False]})
    rule = trading_rule(f)
    np.testing.assert_array_equal(rule["entry"], [1, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(rule["exit"][1], [False, False, False, True, True, True])
