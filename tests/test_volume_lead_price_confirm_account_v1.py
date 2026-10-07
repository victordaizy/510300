"""验证量价顺序的真实次开、固定参考、共同失效、T+1及未知处理。"""
import numpy as np
import pandas as pd

from research.volume_lead_price_confirm_inputs_v1 import volume_lead_states, account_signals, PRIMARY
from research.volume_lead_price_confirm_account_v1 import account
from research.point_account_nr7_inputs_v1 import PARENT_A
from research.point_account_nr7_complement_v1 import verify_account


def market():
    close = np.array([10, 11, 13, 12, 11, 10, 11, 12, 12, 14]+[14.041]*18, dtype=float)
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=len(close)).astype("datetime64[ns]"),
                         "open": close, "high": close+.01, "low": close-.01, "close": close,
                         "ac": close, "ema20": close-.1, "cash_shift": 0., "dividend": 0., "atr20": .1,
                         "volume": [10]*6+[100, 100]+[10]*20})
    return data


def execute(data, states=None):
    parent = pd.DataFrame({"origin": data.date, PARENT_A: 0.})
    risk = pd.DataFrame({"idx": np.arange(len(data)), "es95": .1})
    dividends = pd.DataFrame({key: pd.Series(dtype="datetime64[ns]")
                              for key in ("record_date", "ex_date", "payment_date")})
    dividends["cash_dividend_per_share"] = pd.Series(dtype=float)
    states = volume_lead_states(data)[0] if states is None else states
    result = account(data, dividends, parent, risk, account_signals(data, states, PRIMARY),
                     "STRESS", data.date.iloc[1], "VOLUME_PRICE_ONLY")
    verify_account(result)
    return result


def test_price_confirmation_enters_next_open_and_never_rewrites_fixed_reference_or_adds():
    data = market()
    data.loc[14, ["open", "high", "low", "close", "ac"]] = [15., 15.01, 14.99, 15., 15.]
    states = volume_lead_states(data)[0]
    assert states.reference_high_ticks.iloc[16] == 15000
    result = execute(data, states)
    trade = result["trades"].iloc[0]
    assert trade.entry_origin == data.date.iloc[9] and trade.entry_date == data.date.iloc[10]
    assert trade.entry_volume_floor_ticks == 10000 and trade.fixed_volume_floor_ticks == 10000
    assert trade.fixed_reference_high_ticks == 13000 and trade.fixed_reference_volume == 20
    assert trade.status == "RIGHT_CENSORED"
    assert result["orders"].side.eq("BUY").sum() == 1
    held = result["decisions"].loc[result["decisions"].shares_before.gt(0)]
    assert held.known_current_volume_floor_ticks.eq(10000).all()
    assert held.known_current_reference_high_ticks.eq(13000).all()
    assert held.known_current_reference_volume.eq(20).all()
    assert held.desired_shares.le(held.shares_before).all()


def test_reference_volume_and_price_must_fail_together_for_joint_exit():
    data = market()
    data.loc[10, ["high", "low", "close", "ac", "volume"]] = [14.051, 12.99, 13., 13., 500.]
    result = execute(data)
    trade = result["trades"].iloc[0]
    assert trade.exit_date == data.date.iloc[11]
    assert trade.exit_reason == "VOLUME_LEAD_PRICE_CONFIRM_REFERENCE_VOLUME_AND_PRICE_FAILED_CLOSE"
    assert trade.holding_sessions == 1


def test_volume_failure_alone_above_reference_high_does_not_create_exit():
    data = market()
    data.loc[10, ["high", "low", "close", "ac"]] = [14.1, 14.03, 14.09, 14.09]
    data.loc[11, ["high", "low", "close", "ac", "volume"]] = [14.06, 14.03, 14.041, 14.041, 1000.]
    states = volume_lead_states(data)[0]
    assert states.cumulative_signed_volume.iloc[11] <= 20
    assert states.known_cash_close_ticks.iloc[11] > 13000
    result = execute(data, states)
    assert result["trades"].iloc[0].status == "RIGHT_CENSORED"


def test_open_at_fixed_low_cancels_without_retry():
    data = market()
    data.loc[10, ["open", "low"]] = [10., 10.]
    result = execute(data)
    assert len(result["trades"]) == 0 and result["terminal"]["open_shares"] == 0
    assert result["rejections"].reason.eq("OPEN_AT_OR_BELOW_KNOWN_VOLUME_FLOOR_CANCELLED").sum() == 1


def test_entry_day_fixed_low_failure_cannot_rewrite_morning_buy_and_obeys_t_plus_one():
    data = market()
    baseline = execute(data)
    data.loc[10, ["high", "low", "close", "ac"]] = [14.051, 9.99, 10., 10.]
    changed = execute(data)
    for field in ("date", "origin", "quantity", "raw_open", "fill_price", "commission"):
        assert baseline["orders"].iloc[0][field] == changed["orders"].iloc[0][field]
    trade = changed["trades"].iloc[0]
    assert trade.exit_date == data.date.iloc[11] and trade.holding_sessions == 1
    assert trade.exit_reason == "VOLUME_LEAD_PRICE_CONFIRM_FIXED_LOW_FAILED_CLOSE"


def test_unknown_cumulative_volume_alone_keeps_inventory_but_known_low_still_exits():
    data = market()
    data.loc[10, "volume"] = np.nan
    data.loc[20, ["high", "low", "close", "ac"]] = [14.051, 9.99, 10., 10.]
    states = volume_lead_states(data)[0]
    assert not states.cumulative_volume_known.iloc[10:].any()
    result = execute(data, states)
    assert result["daily"].shares.iloc[0:10].max() > 0
    assert result["trades"].iloc[0].exit_date == data.date.iloc[21]
    assert result["trades"].iloc[0].exit_reason == "VOLUME_LEAD_PRICE_CONFIRM_FIXED_LOW_FAILED_CLOSE"
