"""检查联合反应方向、消息与成交时序、现金分红归属。"""
import numpy as np
import pandas as pd

from research.lpr_joint_response_v1 import classify, gross_return, response_indices


def test_joint_response_preserves_zero_and_missing_information():
    assert classify(-.01, .01) == "YIELD_DOWN_EQUITY_UP"
    assert classify(.01, -.01) == "YIELD_UP_EQUITY_DOWN"
    assert classify(0., .02) == "ZERO_RESPONSE"
    assert classify(-.02, 0.) == "ZERO_RESPONSE"
    assert classify(np.nan, .02) == "NO_VIEW"


def test_observe_first_full_trading_day_then_trade_later_open():
    dates = pd.bdate_range("2024-01-18", periods=7)
    base, observation, entry = response_indices(dates, "2024-01-19")
    assert dates[base] == pd.Timestamp("2024-01-18")
    assert dates[observation] == pd.Timestamp("2024-01-22")
    assert dates[entry] == pd.Timestamp("2024-01-23")
    base, observation, entry = response_indices(dates, "2024-01-21")
    assert dates[base] == pd.Timestamp("2024-01-19")
    assert dates[observation] == pd.Timestamp("2024-01-22")
    assert dates[entry] == pd.Timestamp("2024-01-23")


def test_dividends_exclude_entry_day_and_include_exit_ex_date():
    market = pd.DataFrame({"open": [10., 10., 9.9, 9.8], "dividend": [0., .3, .1, .2]})
    # 入场日的0.3无权享有，持有后两次除息合计0.3计入财富。
    assert np.isclose(gross_return(market, 1, 3), .01)
