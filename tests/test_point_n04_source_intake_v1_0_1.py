"""来源日期的毫秒/纳秒存储精度修复，不允许抹去真实日期或时刻差。"""
import pandas as pd
import pytest
from research.point_n04_source_intake_v1_0_1 import validate_origin_calendar


def test_equal_dates_match_across_ms_us_and_ns_storage_units():
    dates = pd.date_range("2025-01-06", periods=4)
    for left_unit in ["ms", "us", "ns"]:
        for right_unit in ["ms", "us", "ns"]:
            samples = pd.DataFrame({"origin": dates[[0, 2]].as_unit(left_unit), "origin_index": [0, 2]})
            market = pd.DataFrame({"date": dates.as_unit(right_unit)})
            validate_origin_calendar(samples, market)


def test_different_day_or_intraday_time_is_still_rejected():
    dates = pd.date_range("2025-01-06", periods=4)
    market = pd.DataFrame({"date": dates.as_unit("ms")})
    for delta in [pd.Timedelta(days=1), pd.Timedelta(hours=1), pd.Timedelta(nanoseconds=1)]:
        samples = pd.DataFrame({"origin": (dates[[0, 2]] + delta).as_unit("ns"), "origin_index": [0, 2]})
        with pytest.raises(AssertionError):
            validate_origin_calendar(samples, market)
