"""验证严格均线边界、分红财富以及未来数据不改变当前比较。"""
import numpy as np
import pandas as pd
import pytest
from research.factor96_exact_price_baseline_v1 import exact_wealth_ma_relation, exact_price_long


def market(values, dividend=None):
    return pd.DataFrame({"close": values, "dividend": np.zeros(len(values)) if dividend is None else dividend})


def test_exact_equal_ma_never_becomes_long():
    data = market([3+i*.001 for i in range(1,20)]+[3.010])
    assert exact_wealth_ma_relation(data).iloc[-1] == 0
    assert not exact_price_long(data).iloc[-1]


def test_constant_prices_remain_equal_after_long_history():
    data = market([3.007]*500)
    assert exact_wealth_ma_relation(data).iloc[19:].eq(0).all()
    assert not exact_price_long(data).any()


def test_one_tick_above_and_below_are_distinguished():
    prefix = [3+i*.001 for i in range(1,20)]
    assert exact_wealth_ma_relation(market(prefix+[3.011])).iloc[-1] == 1
    assert exact_wealth_ma_relation(market(prefix+[3.009])).iloc[-1] == -1


def test_dividend_reinvestment_is_part_of_wealth_comparison():
    data = market([3.]*19+[2.99], [0.]*19+[.02])
    assert exact_price_long(data).iloc[-1]
    assert not exact_price_long(market([3.]*19+[2.99])).iloc[-1]


def test_prefix_is_unchanged_and_off_tick_source_is_rejected():
    data = market([3+i*.001 for i in range(60)])
    pd.testing.assert_series_equal(exact_price_long(data).iloc[:40], exact_price_long(data.iloc[:40]))
    with pytest.raises(ValueError, match="价位"):
        exact_price_long(market([3.0002]*20))
