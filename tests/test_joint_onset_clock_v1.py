"""检验真实时钟归因、同日多项、未知与未来后缀隔离。"""
import numpy as np
import pandas as pd
import pytest
from research.joint_onset_clock_inputs_v1 import clocks


def market():
    dates = pd.Series(pd.date_range("2020-01-06", periods=12, freq="B"))
    previous_week = dates.dt.to_period("W-FRI").dt.start_time-pd.Timedelta(days=1)
    return pd.DataFrame({
        "date": dates, "ac": [9., 11., 11., 11., 11., 11., 9., 11., 11., 11., 9., 11.],
        "ema20": 10., "daily_dif": [-1., -1., 1., 1., 1., 1., 1., 1., -1., 1., -1., 1.],
        "weekly_hist": [-1.]*5+[1.]*7, "weekly_last_date": previous_week,
        "relative_volume": 1., "breakout20": False, "available": True, "atr20": 1.,
    })


def test_all_three_last_arrivals_and_simultaneous_arrival_are_distinct():
    result = clocks(market())
    born = result.loc[result.joint_phase_onset]
    assert born.index.to_list() == [5, 7, 9, 11]
    assert born.birth_clock.to_list() == [
        "PREVIOUS_COMPLETE_WEEK_LAST", "PRICE_LAST", "DAILY_DIF_LAST", "MULTIPLE_SAME_ORIGIN",
    ]
    assert born.simultaneous_arrivals.to_list() == [1, 1, 1, 2]
    assert born.iloc[-1].price_newly_positive and born.iloc[-1].daily_dif_newly_positive
    assert not born.iloc[-1].weekly_hist_newly_positive


def test_unknown_and_future_results_do_not_create_or_change_clock():
    d = market()
    d.loc[6, "available"] = False
    full = clocks(d)
    assert full.birth_clock.iloc[6] == "NO_VIEW"
    assert full.birth_clock.iloc[7] == "NO_JOINT_BIRTH"
    d["future_cycle_return"] = np.arange(len(d))*100.
    d["retrospective_episode_low"] = np.arange(len(d))[::-1]
    pd.testing.assert_frame_equal(clocks(d), full, check_exact=True)
    for n in range(1, len(d)+1):
        short = clocks(d.iloc[:n].reset_index(drop=True))
        pd.testing.assert_frame_equal(short, full.iloc[:n].reset_index(drop=True), check_exact=True)
        changed = d.copy()
        changed.loc[n:, ["ac", "daily_dif", "weekly_hist"]] = [9., -100., -100.]
        pd.testing.assert_frame_equal(clocks(changed).iloc[:n].reset_index(drop=True), short, check_exact=True)


def test_weekly_last_arrival_requires_an_actual_new_completed_week():
    d = market()
    d.loc[5, "weekly_last_date"] = d.weekly_last_date.iloc[4]
    with pytest.raises(ValueError, match="来源时钟不一致"):
        clocks(d)
