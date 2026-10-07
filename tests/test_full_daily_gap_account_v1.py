"""验证完整日线缺口的下一开盘、回补、只上移下沿及未知时钟。"""
import numpy as np
import pandas as pd

from research.full_daily_gap_inputs_v1 import gap_states, account_signals, PRIMARY
from research.full_daily_gap_account_v1 import account
from research.point_account_nr7_inputs_v1 import PARENT_A
from research.point_account_nr7_complement_v1 import verify_account


def market():
    n = 28
    close = np.array([10.] * 15 + [10.02] + [10.021] * (n-16))
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=n).astype("datetime64[ns]"),
                         "open": close, "high": close+.002, "low": close-.002,
                         "close": close, "ac": close, "cash_shift": 0.,
                         "ema20": 9.9, "dividend": 0., "atr20": .1})
    data.loc[16:, "high"] = 10.024
    data.loc[16:, "low"] = 10.012
    return data


def execute(data, states=None):
    parent = pd.DataFrame({"origin": data.date, PARENT_A: 0.})
    risk = pd.DataFrame({"idx": np.arange(len(data)), "es95": .01})
    dividend = pd.DataFrame({k: pd.Series(dtype="datetime64[ns]") for k in
                             ("record_date", "ex_date", "payment_date")})
    dividend["cash_dividend_per_share"] = pd.Series(dtype=float)
    states = gap_states(data) if states is None else states
    result = account(data, dividend, parent, risk, account_signals(data, states, PRIMARY),
                     "STRESS", data.date.iloc[1], "GAP_ONLY")
    verify_account(result)
    return result


def test_entry_day_low_touch_exits_next_open_even_when_close_stays_above_floor():
    data = market()
    before = execute(data)
    changed = data.copy()
    changed.loc[16, "low"] = 10.002
    after = execute(changed)
    a = before["orders"].loc[before["orders"].side.eq("BUY")].iloc[0]
    b = after["orders"].loc[after["orders"].side.eq("BUY")].iloc[0]
    for key in ("date", "origin", "quantity", "raw_open", "fill_price", "commission"):
        assert a[key] == b[key]
    trade = after["trades"].iloc[0]
    assert trade.entry_origin == changed.date.iloc[15]
    assert trade.entry_date == changed.date.iloc[16]
    assert changed.close.iloc[16] > trade.entry_stop_index
    assert trade.exit_date == changed.date.iloc[17]
    assert trade.holding_sessions == 1
    assert trade.exit_reason == "FULL_DAILY_UP_GAP_FILLED_DAILY_LOW"
    assert not after["daily"].risk_stopped.any()


def test_open_exactly_at_gap_lower_cancels_without_a_delayed_retry():
    data = market()
    data.loc[16, ["open", "low"]] = [10.002, 10.002]
    result = execute(data)
    assert len(result["trades"]) == 0
    assert not result["daily"].shares.any()
    rejection = result["rejections"].iloc[0]
    assert rejection.origin == data.date.iloc[15]
    assert rejection.date == data.date.iloc[16]
    assert rejection.reason == "OPEN_AT_OR_BELOW_KNOWN_GAP_LOWER_CANCELLED"


def test_later_full_gap_only_raises_floor_and_its_touch_exits_next_open():
    data = market()
    data.loc[18:, ["open", "high", "low", "close", "ac"]] = [10.052, 10.054, 10.044, 10.052, 10.052]
    data.loc[18, "low"] = 10.045
    data.loc[20, "low"] = 10.024
    states = gap_states(data)
    assert states.gap_event.iloc[18]
    assert states.gap_lower_ticks.iloc[18] == 10024
    result = execute(data, states)
    trade = result["trades"].iloc[0]
    assert trade.entry_gap_lower_ticks == 10002
    assert trade.trailing_gap_floor_ticks == 10024
    assert data.low.iloc[20] > trade.entry_stop_index
    assert data.close.iloc[20] > trade.trailing_gap_floor_ticks*.001
    assert trade.exit_date == data.date.iloc[21]
    known = result["decisions"].loc[result["decisions"].origin.between(data.date.iloc[16], data.date.iloc[20]),
                                     "known_current_gap_floor_ticks"]
    assert known.diff().dropna().ge(0).all()


def test_unknown_quote_alone_does_not_force_exit_or_create_a_gap():
    data = market()
    data.loc[20, "low"] = 10.002
    unavailable = data.copy()
    unavailable.loc[16, "high"] = np.nan
    states = gap_states(unavailable)
    assert not states.current_quote_known.iloc[16]
    assert not states.gap_event.iloc[17]
    result = execute(data, states)
    day = result["daily"].loc[result["daily"].date.eq(data.date.iloc[16])].iloc[0]
    assert day.shares > 0
    assert result["trades"].iloc[0].exit_date == data.date.iloc[21]
    assert not result["daily"].risk_stopped.any()
