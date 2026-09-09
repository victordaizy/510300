"""日历和前收盘策略状态混合的执行日、三档预算及缺失隔离。"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from research.calendar_learned_equal_blend_v1 import combine_states
from research.calendar_liquidity_timing_v1 import calendar_features, prepare


CFG = {"calendar_first_month_sessions": 3, "calendar_month_end_natural_days": 5,
       "long_break_minimum_date_gap_days": 4, "post_break_sessions": 3}


def example():
    dates = pd.bdate_range("2019-12-02", "2020-03-10")
    dates = dates[(dates != "2020-01-01") & ~dates.to_series(index=dates).between("2020-01-24", "2020-02-02").to_numpy()]
    data = pd.DataFrame({"date": dates, "feature_valid": True})
    calendar = pd.DataFrame({"trade_date": dates, "is_open": True})
    prepared, _ = prepare(data, calendar, CFG)
    decisions = pd.DataFrame({"origin": dates[:-1], "execution_date": dates[1:], "reference_weight": np.arange(len(dates)-1, dtype=float) % 2})
    return prepared, decisions


def test_execution_day_calendar_is_not_previous_close_calendar():
    p, d = example()
    states = combine_states(p, d).set_index("market_date")
    row = states.loc["2020-01-23"]
    assert row.execution_date == pd.Timestamp("2020-02-03")
    assert row.decision_time == pd.Timestamp("2020-02-03 09:00") and row.calendar_state == 1
    previous_day_calendar = calendar_features(pd.DatetimeIndex(p.date), CFG).set_index("execution_date")
    assert previous_day_calendar.loc["2020-01-23", "month_edge"] == 0
    assert states.loc["2020-02-05", "calendar_state"] == 0


def test_all_four_joint_states_map_to_exact_three_budgets():
    p, d = example()
    p.loc[:3, "month_edge"] = [0., 0., 1., 1.]
    d.loc[:3, "reference_weight"] = [0., 1., 0., 1.]
    np.testing.assert_array_equal(combine_states(p, d).target.iloc[:4], [0., .5, .5, 1.])


def test_missing_source_state_is_not_forward_filled_or_cash():
    p, d = example()
    d = d.drop(3)
    result = combine_states(p, d)
    assert not result.inputs_complete.iloc[3] and np.isnan(result.target.iloc[3])
    assert result.inputs_complete.iloc[4]


def test_wrong_execution_date_or_duplicate_origin_rejected():
    p, d = example()
    wrong = d.copy()
    wrong.loc[3, "execution_date"] = p.date.iloc[5]
    with pytest.raises(ValueError, match="错移"):
        combine_states(p, wrong)
    with pytest.raises(ValueError, match="重复"):
        combine_states(p, pd.concat([d, d.iloc[:1]], ignore_index=True))


def test_future_source_states_do_not_change_prior_targets():
    p, d = example()
    original = combine_states(p, d)
    d.loc[20:, "reference_weight"] = 1.
    p.loc[20:, "month_edge"] = 0.
    assert_frame_equal(original.iloc[:20], combine_states(p, d).iloc[:20])
