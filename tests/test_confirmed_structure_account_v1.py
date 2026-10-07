"""已确认结构的次开入出、T+1及信号未知处理，核对实际资金身份。"""
import numpy as np
import pandas as pd

from research.confirmed_structure_inputs_v1 import structure_states, account_signals
from research.confirmed_structure_account_v1 import account
from research.point_account_nr7_inputs_v1 import PARENT_A
from research.point_account_nr7_complement_v1 import verify_account


def market():
    shape = np.array([11, 10, 9, 10, 11, 12, 11, 10, 9.5, 10, 11, 13, 12, 11,
                      10.5, 11, 12, 14, 13, 12, 11.5, 12, 13, 13.5, 13.2, 13, 12.6, 12.2, 12.3, 12.5])
    close = 11+.05*(shape-11)
    op = np.r_[close[0], close[:-1]]
    return pd.DataFrame({"date": pd.date_range("2020-01-02", periods=len(shape), freq="B"),
                         "open": op, "high": np.maximum(op, close)+.001, "low": np.minimum(op, close)-.001,
                         "close": close, "ac": close, "ema20": 10.8, "atr20": .1, "dividend": 0.})


def execute(data, states=None):
    parents = pd.DataFrame({"origin": data.date, PARENT_A: 0.})
    risks = pd.DataFrame({"idx": np.arange(len(data)), "es95": .01})
    dividends = pd.DataFrame({k: pd.Series(dtype="datetime64[ns]") for k in ("record_date", "ex_date", "payment_date")})
    dividends["cash_dividend_per_share"] = pd.Series(dtype=float)
    if states is None:
        states, _ = structure_states(data)
    signals = account_signals(data, states, "CONFIRMED_RISING_STRUCTURE")
    result = account(data, dividends, parents, risks, signals, "STRESS", data.date.iloc[1], "STRUCTURE_ONLY")
    verify_account(result)
    return result


def test_lower_confirmed_high_exits_next_open_even_above_ema_and_latest_low():
    data = market()
    states, _ = structure_states(data)
    result = execute(data, states)
    first = result["trades"].iloc[0]
    assert first.entry_origin == data.date.iloc[13]
    assert first.entry_date == data.date.iloc[14]
    assert states.H2_index.iloc[25] == 23
    assert states.structure_status.iloc[25] == "NONRISING_CONFIRMED_STRUCTURE"
    assert data.close.iloc[25] > states.latest_confirmed_low_price.iloc[25]
    assert data.ac.gt(data.ema20).all()
    assert first.exit_date == data.date.iloc[26]
    assert first.exit_reason == "CONFIRMED_STRUCTURE_FAILED_CLOSE"
    assert result["orders"].quantity.mod(100).eq(0).all()


def test_entry_ignores_its_own_close_and_low_breach_exit_obeys_t_plus_one():
    data = market()
    original = execute(data)
    changed = data.copy()
    changed.loc[14, ["close", "ac", "low"]] = [10.90, 10.90, 10.899]
    actual = execute(changed)
    a = original["orders"].loc[original["orders"].side.eq("BUY")].iloc[0]
    b = actual["orders"].loc[actual["orders"].side.eq("BUY")].iloc[0]
    for field in ("date", "origin", "quantity", "raw_open", "fill_price", "commission"):
        assert a[field] == b[field]
    first = actual["trades"].iloc[0]
    assert first.entry_date == changed.date.iloc[14]
    assert first.exit_date == changed.date.iloc[15]
    assert first.holding_sessions == 1
    assert first.exit_reason == "CONFIRMED_STRUCTURE_FAILED_CLOSE"
    assert not actual["daily"].risk_stopped.any()


def test_unknown_signal_keeps_existing_inventory_without_pretending_a_new_birth():
    data = market()
    data.loc[27, ["close", "ac", "low"]] = [10.90, 10.90, 10.899]
    signal_source = data.copy()
    signal_source.loc[15, "close"] = np.nan
    states, _ = structure_states(signal_source)
    assert states.structure_status.iloc[15] == "NO_VIEW"
    assert not states.rule_exit.iloc[15]
    assert not states.structure_onset.iloc[16]
    result = execute(data, states)
    at_unknown = result["daily"].loc[result["daily"].date.eq(data.date.iloc[15])].iloc[0]
    assert at_unknown.shares > 0
    assert states.structure_status.iloc[27] == "LATEST_CONFIRMED_LOW_FAILED"
    assert result["trades"].iloc[0].exit_date == data.date.iloc[28]
    assert not result["decisions"].loc[result["decisions"].origin.eq(data.date.iloc[15]), "known_rule_exit"].iloc[0]
