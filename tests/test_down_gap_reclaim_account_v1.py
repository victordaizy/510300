"""验证收复次开、固定下沿、新完整缺口失效、未知、取消及T+1。"""
import numpy as np
import pandas as pd

from research.down_gap_reclaim_inputs_v1 import reclaim_states, account_signals, PRIMARY
from research.down_gap_reclaim_account_v1 import account
from research.point_account_nr7_inputs_v1 import PARENT_A
from research.point_account_nr7_complement_v1 import verify_account


def market():
    n = 28
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=n).astype("datetime64[ns]"),
                         "open": 10., "high": 10.01, "low": 9.99, "close": 10.,
                         "ac": 10., "cash_shift": 0., "ema20": 9.9, "dividend": 0., "atr20": .1})
    data.loc[14, ["open", "high", "low", "close", "ac"]] = [9.8, 9.9, 9.7, 9.8, 9.8]
    data.loc[15, ["open", "high", "low", "close", "ac"]] = [9.9, 10.02, 9.88, 10.01, 10.01]
    data.loc[16:, ["open", "high", "low", "close", "ac"]] = [10.041, 10.055, 10.03, 10.041, 10.041]
    return data


def execute(data, states=None):
    parent = pd.DataFrame({"origin": data.date, PARENT_A: 0.})
    risk = pd.DataFrame({"idx": np.arange(len(data)), "es95": .01})
    dividends = pd.DataFrame({key: pd.Series(dtype="datetime64[ns]") for key in ("record_date", "ex_date", "payment_date")})
    dividends["cash_dividend_per_share"] = pd.Series(dtype=float)
    states = reclaim_states(data) if states is None else states
    result = account(data, dividends, parent, risk, account_signals(data, states, PRIMARY),
                     "STRESS", data.date.iloc[1], "RECLAIM_ONLY")
    verify_account(result)
    return result


def test_first_reclaim_enters_next_open_keeps_original_floor_and_never_adds():
    data = market()
    result = execute(data)
    trade = result["trades"].iloc[0]
    assert trade.entry_origin == data.date.iloc[15] and trade.entry_date == data.date.iloc[16]
    assert trade.entry_reclaim_floor_ticks == 9900 and trade.fixed_reclaim_floor_ticks == 9900
    assert trade.status == "RIGHT_CENSORED" and result["terminal"]["open_shares"] > 0
    assert result["orders"].side.eq("BUY").sum() == 1
    held = result["decisions"].loc[result["decisions"].shares_before.gt(0)]
    assert held.known_current_reclaim_floor_ticks.eq(9900).all()
    assert held.desired_shares.le(held.shares_before).all()


def test_new_down_gap_above_fixed_floor_exits_next_legal_open():
    data = market()
    data.loc[17, ["open", "high", "low", "close", "ac"]] = [9.95, 10.02, 9.94, 9.97, 9.97]
    states = reclaim_states(data)
    assert states.new_down_gap.iloc[17] and data.close.iloc[17] > 9.9
    result = execute(data, states)
    trade = result["trades"].iloc[0]
    assert trade.exit_date == data.date.iloc[18]
    assert trade.exit_reason == "DOWN_GAP_RECLAIM_NEW_DOWN_GAP"


def test_open_exactly_at_fixed_floor_cancels_once_without_late_retry():
    data = market()
    data.loc[16, ["open", "low"]] = [9.9, 9.9]
    result = execute(data)
    assert len(result["trades"]) == 0 and not result["daily"].shares.any()
    assert len(result["rejections"]) == 1
    assert result["rejections"].reason.iloc[0] == "OPEN_AT_OR_BELOW_KNOWN_RECLAIM_FLOOR_CANCELLED"


def test_entry_day_floor_equality_has_priority_and_cannot_rewrite_morning_buy():
    before = market()
    changed = before.copy()
    changed.loc[16, ["high", "low", "close", "ac"]] = [10.041, 9.89, 9.9, 9.9]
    a, b = execute(before), execute(changed)
    for key in ("date", "origin", "quantity", "raw_open", "fill_price", "commission"):
        assert a["orders"].iloc[0][key] == b["orders"].iloc[0][key]
    trade = b["trades"].iloc[0]
    assert trade.exit_date == changed.date.iloc[17] and trade.holding_sessions == 1
    assert trade.exit_reason == "DOWN_GAP_RECLAIM_FAILED_KNOWN_CLOSE"


def test_unknown_quote_keeps_inventory_and_later_known_floor_failure_exits():
    data = market()
    data.loc[20, ["low", "close", "ac"]] = [9.89, 9.9, 9.9]
    unavailable = data.copy()
    unavailable.loc[16, "high"] = np.nan
    states = reclaim_states(unavailable)
    assert not states.current_quote_known.iloc[16]
    result = execute(data, states)
    assert result["daily"].loc[result["daily"].date.eq(data.date.iloc[16]), "shares"].iloc[0] > 0
    assert result["trades"].iloc[0].exit_date == data.date.iloc[21]
    assert not result["daily"].risk_stopped.any()
