"""验证整日反转的次开、相反事件、只抬失效低点、未知与T+1。"""
import numpy as np
import pandas as pd

from research.full_range_reversal_inputs_v1 import reversal_states, account_signals, PRIMARY
from research.full_range_reversal_account_v1 import account
from research.point_account_nr7_inputs_v1 import PARENT_A
from research.point_account_nr7_complement_v1 import verify_account


def market():
    n = 28
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=n).astype("datetime64[ns]"),
                         "open": 10., "high": 10.01, "low": 9.99, "close": 10.,
                         "ac": 10., "cash_shift": 0., "ema20": 9.9, "dividend": 0., "atr20": .1})
    data.loc[15, ["open", "high", "low", "close", "ac"]] = [10., 10.05, 9.98, 10.04, 10.04]
    data.loc[16:, ["open", "high", "low", "close", "ac"]] = [10.041, 10.055, 10.03, 10.041, 10.041]
    return data


def execute(data, states=None):
    parent = pd.DataFrame({"origin": data.date, PARENT_A: 0.})
    risk = pd.DataFrame({"idx": np.arange(len(data)), "es95": .01})
    dividends = pd.DataFrame({k: pd.Series(dtype="datetime64[ns]") for k in ("record_date", "ex_date", "payment_date")})
    dividends["cash_dividend_per_share"] = pd.Series(dtype=float)
    states = reversal_states(data) if states is None else states
    result = account(data, dividends, parent, risk, account_signals(data, states, PRIMARY),
                     "STRESS", data.date.iloc[1], "REVERSAL_ONLY")
    verify_account(result)
    return result


def test_opposing_range_reversal_exits_next_open_above_initial_floor():
    data = market()
    data.loc[17, ["open", "high", "low", "close", "ac"]] = [10.04, 10.058, 10.02, 10.025, 10.025]
    assert reversal_states(data).bearish_range_reversal.iloc[17]
    result = execute(data)
    trade = result["trades"].iloc[0]
    assert trade.entry_date == data.date.iloc[16]
    assert data.close.iloc[17] > trade.entry_stop_index
    assert trade.exit_date == data.date.iloc[18]
    assert trade.exit_reason == "FULL_RANGE_REVERSAL_OPPOSING_CLOSE"


def test_new_bullish_reversal_only_raises_floor_and_known_close_break_exits():
    data = market()
    data.loc[18, ["open", "high", "low", "close", "ac"]] = [10.04, 10.09, 10.025, 10.08, 10.08]
    data.loc[19:, ["open", "high", "low", "close", "ac"]] = [10.07, 10.09, 10.06, 10.07, 10.07]
    data.loc[20, ["open", "high", "low", "close", "ac"]] = [10.07, 10.085, 10.02, 10.023, 10.023]
    states = reversal_states(data)
    assert states.bullish_range_reversal.iloc[18]
    assert not states.bearish_range_reversal.iloc[20]
    result = execute(data, states)
    trade = result["trades"].iloc[0]
    assert trade.entry_reversal_floor_ticks == 9980
    assert trade.trailing_reversal_floor_ticks == 10025
    assert data.close.iloc[20] > trade.entry_stop_index
    assert trade.exit_date == data.date.iloc[21]
    assert trade.exit_reason == "FULL_RANGE_REVERSAL_FAILED_KNOWN_CLOSE"
    known = result["decisions"].loc[result["decisions"].origin.between(data.date.iloc[16], data.date.iloc[20]),
                                    "known_current_reversal_floor_ticks"]
    assert known.diff().dropna().ge(0).all()


def test_open_exactly_at_known_low_cancels_without_delayed_retry():
    data = market()
    data.loc[16, ["open", "low"]] = [9.98, 9.98]
    result = execute(data)
    assert len(result["trades"]) == 0
    assert not result["daily"].shares.any()
    assert result["rejections"].iloc[0].reason == "OPEN_AT_OR_BELOW_KNOWN_REVERSAL_FLOOR_CANCELLED"


def test_entry_day_known_closing_failure_cannot_rewrite_morning_buy():
    before = market()
    changed = before.copy()
    changed.loc[16, ["low", "close", "ac"]] = [9.975, 9.979, 9.979]
    a, b = execute(before), execute(changed)
    for key in ("date", "origin", "quantity", "raw_open", "fill_price", "commission"):
        assert a["orders"].iloc[0][key] == b["orders"].iloc[0][key]
    assert b["trades"].iloc[0].exit_date == changed.date.iloc[17]
    assert b["trades"].iloc[0].holding_sessions == 1


def test_unknown_quote_alone_keeps_position_and_later_known_close_exits():
    data = market()
    data.loc[20, ["low", "close", "ac"]] = [9.975, 9.979, 9.979]
    unavailable = data.copy()
    unavailable.loc[16, "high"] = np.nan
    states = reversal_states(unavailable)
    assert not states.current_quote_known.iloc[16]
    assert not states.bullish_range_reversal.iloc[17]
    result = execute(data, states)
    assert result["daily"].loc[result["daily"].date.eq(data.date.iloc[16]), "shares"].iloc[0] > 0
    assert result["trades"].iloc[0].exit_date == data.date.iloc[21]
    assert not result["daily"].risk_stopped.any()
