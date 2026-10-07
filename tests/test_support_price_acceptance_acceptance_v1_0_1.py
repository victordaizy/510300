"""日期存储规范必须保留值，并仍拒绝不同日历。"""
import pandas as pd
import pytest

from research.support_price_acceptance_acceptance_v1_0_1 import require_same_dates


def test_millisecond_nanosecond_storage_can_have_exactly_same_values():
    dates = pd.Series(pd.date_range("2020-01-02", periods=3), name="date")
    result = require_same_dates(dates.astype("datetime64[ms]"), dates.astype("datetime64[ns]"))
    assert result["all_date_values_and_index_exact"]


def test_different_calendar_is_not_hidden_by_storage_conversion():
    dates = pd.Series(pd.date_range("2020-01-02", periods=3), name="date")
    changed = dates.copy()
    changed.iloc[1] += pd.Timedelta(days=1)
    with pytest.raises(AssertionError):
        require_same_dates(dates.astype("datetime64[ms]"), changed.astype("datetime64[ns]"))
