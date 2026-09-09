"""核对先后顺序、固定突破位、过期、失效和除息尺度。"""
import numpy as np
import pandas as pd

from research.breakout_retest_entry_v1 import detect_retests, low_in_wealth_units


def sample(close, low=None, first=None, valid=None, level=None, days=10):
    n = len(close)
    return detect_retests(pd.Series(pd.date_range("2025-01-01", periods=n)), close,
                          close if low is None else low, np.full(n, 10.) if level is None else level,
                          [True] + [False] * (n - 1) if first is None else first,
                          np.ones(n, dtype=bool) if valid is None else valid, days)


def test_breakout_day_low_cannot_be_used_as_later_retest():
    x = sample([10.5, 10.4, 10.6], [9.8, 10.1, 9.9])
    assert x.setup_created.iloc[0] and not x.candidate_entry.iloc[0]
    assert x.candidate_entry.tolist() == [False, False, True]
    assert x.setup_age.iloc[2] == 2


def test_tenth_day_allowed_eleventh_expired():
    close = np.arange(12) * .1 + 10.5
    low = close - .02
    low[10] = 9.9
    assert sample(close, low).candidate_entry.iloc[10]
    low[10], low[11] = close[10] - .02, 9.9
    x = sample(close, low)
    assert not x.candidate_entry.any()
    assert x.setup_status.iloc[11] == "等待期限结束"


def test_close_below_reference_cancels_without_revival():
    x = sample([10.5, 9.9, 10.3, 10.4], [10.2, 9.8, 9.9, 9.9])
    assert not x.candidate_entry.any()
    assert "取消机会" in x.setup_status.iloc[1]
    assert pd.isna(x.setup_event_index.iloc[2])


def test_new_highs_do_not_move_or_restart_active_reference():
    x = sample([10.5, 10.4, 11.5, 11.6, 11.7], [10.2, 10.1, 11.2, 10.5, 9.9],
               first=[True, False, True, False, False], level=[10., 10.5, 11., 11.5, 11.6])
    assert x.fixed_breakout_level.iloc[:5].eq(10).all()
    assert not x.candidate_entry.iloc[3]
    assert x.candidate_entry.iloc[4]
    assert x.setup_event_index.iloc[4] == 0


def test_missing_data_and_future_changes_cannot_create_past_entry():
    a = sample([10.5, 10.4, 10.6, 10.7], [10.1, 10.1, 9.9, 10.5], valid=[True, False, True, True])
    assert not a.candidate_entry.any()
    assert "NO_VIEW" in a.setup_status.iloc[1]
    b = sample([10.5, 10.4, 10.6, 10.7], [10.1, 10.1, 9.9, 10.5])
    c = sample([10.5, 10.4, 10.6, 20.7], [10.1, 10.1, 9.9, 5.5])
    pd.testing.assert_frame_equal(b.iloc[:3], c.iloc[:3])


def test_ex_dividend_low_keeps_same_economic_scale():
    adjusted = low_in_wealth_units(100., 8.8, 9., 1.)
    ordinary = low_in_wealth_units(100., 9.8, 10., 0.)
    assert np.isclose(adjusted, ordinary, atol=1e-12)
