"""检验日周同步阶段的出生、未知状态和完整周时钟。"""

import numpy as np
import pandas as pd
import pytest

from research.trend_expansion_phase_inputs_v1 import phase_states, account_signals


def market():
    dates = pd.Series(pd.date_range("2020-01-06", periods=9, freq="B"))
    previous_week = dates.dt.to_period("W-FRI").dt.start_time-pd.Timedelta(days=1)
    return pd.DataFrame({"date": dates, "ac": [11.]*9, "ema20": [10.]*9,
                         "daily_dif": [-1., 1., 1., 1., 1., -1., 1., 1., 1.],
                         "weekly_hist": [1.]*9, "weekly_last_date": previous_week,
                         "relative_volume": [1.]*9, "breakout20": [False]*9,
                         "available": [True]*9, "atr20": [1.]*9})


def test_onset_requires_a_real_previous_known_nonpositive_state():
    d = market()
    p = phase_states(d)
    assert p.joint_phase_onset.to_list() == [False, True, False, False, False, False, True, False, False]
    first_active = d.copy()
    first_active.loc[0, "daily_dif"] = 1.
    assert not phase_states(first_active).joint_phase_onset.iloc[0]
    gap = d.copy()
    gap.loc[5, "available"] = False
    q = phase_states(gap)
    assert q.phase.iloc[5] == "NO_VIEW"
    assert not q.joint_phase_onset.iloc[6]
    request = account_signals(gap, q, "JOINT_PHASE_START")
    assert not request.rule_exit.iloc[5]


def test_friday_does_not_read_its_own_weekly_close():
    d = market()
    d.loc[4, "weekly_last_date"] = d.date.iloc[4]
    with pytest.raises(ValueError, match="当前尚未完成周"):
        phase_states(d)


def test_current_phase_is_isolated_from_future_suffix():
    d = market()
    full = phase_states(d)
    for n in range(1, len(d)+1):
        short = phase_states(d.iloc[:n].reset_index(drop=True))
        pd.testing.assert_frame_equal(short, full.iloc[:n].reset_index(drop=True), check_exact=True)
        altered = d.copy()
        altered.loc[n:, ["ac", "daily_dif", "weekly_hist"]] = [1., -100., -100.]
        other = phase_states(altered)
        pd.testing.assert_frame_equal(other.iloc[:n].reset_index(drop=True), short, check_exact=True)
