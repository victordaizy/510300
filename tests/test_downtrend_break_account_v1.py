"""突破失效、确认低点只上移、次开取消和T+1的实际账户行为。"""
import numpy as np
import pandas as pd

from research.downtrend_break_inputs_v1 import break_states, account_signals, PRIMARY
from research.downtrend_break_account_v1 import account
from research.point_account_nr7_inputs_v1 import PARENT_A
from research.point_account_nr7_complement_v1 import verify_account


def market():
    shape = np.array([11, 12, 13, 12, 11, 10, 11, 12, 12.5, 12, 11, 9, 10, 11, 11.5,
                      12.6, 13, 13.5, 13.2, 13.0, 13.3, 13.6, 13.4, 13.2, 13.3, 13.5, 13.0, 13.1])
    close = 11+.05*(shape-11)
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"date": pd.date_range("2020-01-02", periods=len(shape), freq="B"),
                         "open": op, "high": np.maximum(op, close)+.001, "low": np.minimum(op, close)-.001,
                         "close": close, "ac": close, "cash_shift": 0., "ema20": 10.8, "dividend": 0., "atr20": .1})


def execute(data, states=None):
    parents = pd.DataFrame({"origin": data.date, PARENT_A: 0.})
    risks = pd.DataFrame({"idx": np.arange(len(data)), "es95": .01})
    dividends = pd.DataFrame({k: pd.Series(dtype="datetime64[ns]") for k in ("record_date", "ex_date", "payment_date")})
    dividends["cash_dividend_per_share"] = pd.Series(dtype=float)
    if states is None:
        states, _ = break_states(data)
    result = account(data, dividends, parents, risks, account_signals(data, states, PRIMARY), "STRESS", data.date.iloc[1], "BREAK_ONLY")
    verify_account(result)
    return result


def test_confirmed_higher_low_raises_stop_and_breach_exits_next_open():
    data = market()
    states, _ = break_states(data)
    result = execute(data, states)
    trade = result["trades"].iloc[0]
    assert trade.entry_origin == data.date.iloc[15]
    assert trade.entry_date == data.date.iloc[16]
    assert trade.entry_stop_index == 11.075
    assert states.L2_index.iloc[21] == 19
    assert states.L2_index.iloc[25] == 23
    assert trade.trailing_structural_stop == 11.11
    assert data.ac.iloc[26] > trade.entry_stop_index
    assert trade.exit_date == data.date.iloc[27]
    assert trade.exit_reason == "DOWNTREND_BREAK_STRUCTURAL_STOP_FAILED_CLOSE"
    known = result["decisions"].loc[result["decisions"].origin.between(data.date.iloc[16], data.date.iloc[26]), "known_current_structural_stop"]
    assert known.diff().dropna().ge(0).all()


def test_open_at_initial_broken_high_is_cancelled_without_a_delayed_retry():
    data = market()
    data.loc[16, "open"] = 11.075
    data.loc[16, "low"] = min(data.low.iloc[16], 11.074)
    result = execute(data)
    assert len(result["trades"]) == 0
    assert not result["daily"].shares.any()
    rejection = result["rejections"].iloc[0]
    assert rejection.origin == data.date.iloc[15]
    assert rejection.date == data.date.iloc[16]
    assert rejection.reason == "OPEN_AT_OR_BELOW_KNOWN_BROKEN_HIGH_CANCELLED"


def test_entry_cannot_read_its_own_close_and_failure_exit_obeys_t_plus_one():
    data = market()
    before = execute(data)
    changed = data.copy()
    changed.loc[16, ["close", "ac", "low"]] = [11.06, 11.06, 11.059]
    after = execute(changed)
    a = before["orders"].loc[before["orders"].side.eq("BUY")].iloc[0]
    b = after["orders"].loc[after["orders"].side.eq("BUY")].iloc[0]
    for key in ("date", "origin", "quantity", "raw_open", "fill_price", "commission"):
        assert a[key] == b[key]
    trade = after["trades"].iloc[0]
    assert trade.entry_date == changed.date.iloc[16]
    assert trade.exit_date == changed.date.iloc[17]
    assert trade.holding_sessions == 1
    assert not after["daily"].risk_stopped.any()


def test_unknown_signal_does_not_force_exit_or_create_another_event():
    data = market()
    source = data.copy()
    source.loc[16, "close"] = np.nan
    states, _ = break_states(source)
    assert not states.structure_known.iloc[16]
    assert not states.break_event.iloc[17]
    result = execute(data, states)
    at_unknown = result["daily"].loc[result["daily"].date.eq(data.date.iloc[16])].iloc[0]
    assert at_unknown.shares > 0
    assert result["trades"].iloc[0].exit_date == data.date.iloc[27]
