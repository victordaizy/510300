"""检验日周阶段失效、未知持有、次开和T+1的真实账户行为。"""

import numpy as np
import pandas as pd

from research.trend_expansion_phase_inputs_v1 import phase_states, account_signals
from research.trend_expansion_phase_account_v1 import account
from research.point_account_nr7_inputs_v1 import PARENT_A
from research.point_account_nr7_complement_v1 import verify_account


def market():
    dates = pd.Series(pd.date_range("2020-01-06", periods=9, freq="B"))
    prior = dates.dt.to_period("W-FRI").dt.start_time - pd.Timedelta(days=1)
    return pd.DataFrame({
        "date": dates, "open": 11., "high": 11.1, "low": 10.9, "close": 11.,
        "ac": 11., "ema20": 10., "dividend": 0.,
        "daily_dif": [-1., 1., 1., 1., 1., -1., 1., 1., 1.],
        "weekly_hist": 1., "weekly_last_date": prior, "relative_volume": 1.,
        "breakout20": False, "available": True, "atr20": 1.,
    })


def execute(d):
    parents = pd.DataFrame({"origin": d.date, PARENT_A: 0.})
    risks = pd.DataFrame({"idx": np.arange(len(d)), "es95": .01})
    dividends = pd.DataFrame({
        name: pd.Series(dtype="datetime64[ns]")
        for name in ["record_date", "ex_date", "payment_date"]
    })
    dividends["cash_dividend_per_share"] = pd.Series(dtype=float)
    phases = phase_states(d)
    signals = account_signals(d, phases, "JOINT_PHASE_START")
    result = account(d, dividends, parents, risks, signals, "STRESS", d.date.iloc[1], "PHASE_ONLY")
    verify_account(result)
    return result


def test_daily_phase_failure_exits_even_while_price_stays_above_ema():
    d = market()
    result = execute(d)
    trade = result["trades"].iloc[0]
    assert trade.entry_origin == d.date.iloc[1]
    assert trade.entry_date == d.date.iloc[2]
    assert trade.exit_date == d.date.iloc[6]
    assert trade.exit_reason == "JOINT_PHASE_FAILED_CLOSE"
    assert trade.holding_sessions == 4
    assert d.ac.gt(d.ema20).all()
    assert result["orders"].quantity.mod(100).eq(0).all()


def test_unknown_does_not_exit_but_a_known_weekly_failure_does():
    unknown = market()
    unknown.loc[3, "available"] = False
    a = execute(unknown)
    at_gap = a["daily"].loc[a["daily"].date.eq(unknown.date.iloc[3])].iloc[0]
    assert at_gap.shares > 0
    assert a["trades"].iloc[0].exit_date == unknown.date.iloc[6]
    weekly = market()
    weekly["daily_dif"] = [-1.] + [1.]*8
    weekly.loc[5:, "weekly_hist"] = -1.
    b = execute(weekly)
    assert b["trades"].iloc[0].exit_date == weekly.date.iloc[6]
    assert b["trades"].iloc[0].exit_reason == "JOINT_PHASE_FAILED_CLOSE"


def test_entry_does_not_read_its_own_close_and_exit_obeys_t_plus_one():
    d = market()
    original = execute(d)
    changed = d.copy()
    changed.loc[2, ["close", "ac", "high", "daily_dif"]] = [11.5, 11.5, 11.6, -10.]
    actual = execute(changed)
    first_a = original["orders"].loc[original["orders"].side.eq("BUY")].iloc[0]
    first_b = actual["orders"].loc[actual["orders"].side.eq("BUY")].iloc[0]
    for field in ["date", "origin", "quantity", "raw_open", "fill_price", "commission"]:
        assert first_a[field] == first_b[field]
    trade = actual["trades"].iloc[0]
    assert trade.entry_date == changed.date.iloc[2]
    assert trade.exit_date == changed.date.iloc[3]
    assert trade.holding_sessions == 1
    assert not actual["daily"].risk_stopped.any()

