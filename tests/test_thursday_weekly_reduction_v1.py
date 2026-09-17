"""必要日历边界：普通周、未知、节假日与未来隔离。"""
import numpy as np
import pandas as pd

from research.thursday_weekly_reduction_inputs_v1 import weekly_target


def test_normal_week_preserves_other_days():
    dates=pd.date_range('2026-09-07',periods=5,freq='D')
    np.testing.assert_array_equal(weekly_target([.2,.3,.4,.5,.6],dates),[.2,.3,.4,0.,.6])


def test_known_calendar_exit_independent_of_unknown_parent():
    dates=pd.to_datetime(['2026-09-09','2026-09-10','2026-09-11'])
    np.testing.assert_allclose(weekly_target([np.nan,np.nan,0.],dates),[np.nan,0.,0.],equal_nan=True)


def test_no_advance_for_missing_thursday_and_no_lookahead_holiday():
    dates=pd.to_datetime(['2020-09-30','2020-10-09'])
    np.testing.assert_array_equal(weekly_target([.5,.6],dates),[.5,.6])
    dates=pd.to_datetime(['2026-04-29','2026-04-30','2026-05-06'])
    np.testing.assert_array_equal(weekly_target([.4,.5,.6],dates),[.4,0.,.6])


def test_append_future_prices_or_dates_does_not_change_past():
    dates=pd.to_datetime(['2026-04-29','2026-04-30','2026-05-06'])
    prefix=weekly_target([.4,.5],dates[:2])
    np.testing.assert_array_equal(weekly_target([.4,.5,1.],dates)[:2],prefix)
