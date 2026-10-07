"""全空时间类型不改时间值；真实价格或日期差异仍拒绝。"""
import pandas as pd
import pytest

from research import actual_holding_information_format_v1_0_1 as repair


def sample(unit):
    d = pd.DataFrame({name: pd.Series([pd.NaT], dtype="datetime64[" + unit + "]") for name in repair.DATES})
    for name in repair.CLOCKS:
        d[name] = pd.Series([pd.NaT], dtype="datetime64[" + unit + "]")
    d["price"] = [10.2]
    return d


def test_all_nat_time_precision_preserved_and_original_not_modified():
    seconds, nanos = sample("s"), sample("ns")
    original = seconds.copy(deep=True)
    repair.require_same_prefix(nanos, seconds)
    pd.testing.assert_frame_equal(seconds, original, check_exact=True)
    result = repair.normalized(seconds)
    assert all(result[name].isna().all() for name in (*repair.DATES, *repair.CLOCKS))
    assert str(result.orders_available_at.dtype) == "datetime64[ns, Asia/Shanghai]"


def test_actual_price_date_or_naive_source_clock_difference_rejected():
    a, b = sample("ns"), sample("s")
    b.loc[0, "date"] = pd.Timestamp("2024-09-24")
    with pytest.raises(AssertionError):
        repair.require_same_prefix(a, b)
    b = sample("s")
    b.loc[0, "price"] = 10.2000000001
    with pytest.raises(AssertionError):
        repair.require_same_prefix(a, b)
    b = sample("s")
    b.loc[0, "orders_available_at"] = pd.Timestamp("2024-09-24 09:30")
    with pytest.raises(ValueError, match="无时区"):
        repair.normalized(b)
