"""窄幅确认时钟、完整账户和新增机会边界的必要测试。"""
import numpy as np
import pandas as pd

from research.point_account_nr7_inputs_v1 import (
    PARENT_A, PARENT_B, account, metrics, nr7_signals, return_statistics,
)


def fixture_data(n=14):
    dates = pd.bdate_range("2024-01-02", periods=n)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "high": 10.2,
                         "low": 9.8, "dividend": 0., "volume": 100000.})
    data["cash_shift"] = 0.
    for raw, adjusted in [("open", "ao"), ("close", "ac"), ("high", "ah"), ("low", "al")]:
        data[adjusted] = data[raw]
    signals = pd.DataFrame({"date": dates, "entry_event": False, "stop_index": np.nan,
                            "setup_date": pd.NaT})
    risks = pd.DataFrame({"idx": np.arange(n), "es95": .01})
    parents = pd.DataFrame({"origin": dates, PARENT_A: 0., PARENT_B: 0.})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    return data, signals, risks, parents, dividends


def event(signals, i, stop=9.5):
    signals.loc[i, ["entry_event", "stop_index", "setup_date"]] = [True, stop, signals.date.iloc[max(0, i - 1)]]


def test_nr7_requires_next_close_and_strict_range_then_prefix_is_stable():
    d, _, _, _, _ = fixture_data(12)
    d.loc[:5, ["high", "ah"]] = 10.5
    d.loc[:5, ["low", "al"]] = 9.5
    d.loc[6, ["high", "ah"]] = 10.1
    d.loc[6, ["low", "al"]] = 9.9
    d.loc[7, ["close", "ac"]] = 10.2
    f = nr7_signals(d)
    assert f.nr7_setup.iloc[6] and not f.entry_event.iloc[6]
    assert f.entry_event.iloc[7]
    assert f.setup_date.iloc[7] == d.date.iloc[6]
    assert f.stop_index.iloc[7] == 9.9
    assert not f.nr7_setup.iloc[8]  # 与之前宽度相同不能成为严格最窄。
    pd.testing.assert_frame_equal(f.iloc[:9], nr7_signals(d.iloc[:9]))
    d.loc[9:, ["ah", "ac", "high", "close"]] = 99.
    pd.testing.assert_frame_equal(f.iloc[:9], nr7_signals(d).iloc[:9])


def test_entry_close_stop_exits_only_next_open_and_end_is_not_liquidated():
    d, s, r, p, div = fixture_data(5)
    event(s, 0)
    d.loc[1, ["close", "ac"]] = 9.4
    d.loc[2, ["open", "ao", "close", "ac"]] = 9.2
    result = account(d, div, s, p, r, "NR7_ONLY", "STRESS", d.date.iloc[1])
    trade = result["trades"].iloc[0]
    assert trade.entry_date == d.date.iloc[1] and trade.exit_date == d.date.iloc[2]
    assert trade.exit_reason == "CLOSE_STRUCTURAL_STOP" and trade.net_return < -.08
    assert result["orders"].origin.lt(result["orders"].date).all()
    d2, s2, r2, p2, div2 = fixture_data(4)
    event(s2, 0)
    unfinished = account(d2, div2, s2, p2, r2, "NR7_ONLY", "STRESS", d2.date.iloc[1])
    assert unfinished["trades"].status.eq("RIGHT_CENSORED").all()
    assert unfinished["terminal"]["open_shares"] > 0
    assert metrics(unfinished)["completed_cycles"] == 0


def test_dividend_receivable_cash_and_full_calendar_accounting():
    d, s, r, p, div = fixture_data(7)
    event(s, 0, stop=8.)
    d.loc[2:, ["open", "close", "high", "low"]] -= .1
    d.loc[2, "dividend"] = .1
    d.loc[2:, "cash_shift"] = .1
    div = pd.DataFrame([{"record_date": d.date.iloc[1], "ex_date": d.date.iloc[2],
                         "payment_date": d.date.iloc[4], "cash_dividend_per_share": .1}])
    result = account(d, div, s, p, r, "NR7_ONLY", "STRESS", d.date.iloc[1])
    daily = result["daily"].set_index("date")
    assert daily.loc[d.date.iloc[2], "receivable"] > 0
    assert daily.loc[d.date.iloc[4], "receivable"] == 0
    assert daily.dividend_accrual.sum() == daily.dividend_paid.sum()
    assert daily.accounting_error.abs().max() < 1e-7
    assert len(daily) == len(d) - 1
    assert abs(daily.equity.iloc[-1] - 200000 + daily.commission.sum() + daily.slippage.sum()) < 1e-6


def test_core_priority_no_same_day_reentry_and_nr7_does_not_steal_active_cycle():
    d, s, r, p, div = fixture_data(8)
    p.loc[0:1, PARENT_A] = 1.
    event(s, 0)
    event(s, 2)
    event(s, 3)
    p.loc[4:6, PARENT_A] = 1.
    result = account(d, div, s, p, r, "POINT_A_PLUS_NR7", "STRESS", d.date.iloc[1])
    trades = result["trades"]
    assert trades.source.tolist() == ["CORE", "NR7"]
    assert trades.entry_date.tolist() == [d.date.iloc[1], d.date.iloc[4]]
    assert trades.exit_date.iloc[0] == d.date.iloc[3]
    assert trades.status.iloc[1] == "RIGHT_CENSORED"
    assert "EXISTING_POSITION_OR_SAME_DAY_EXIT" in set(result["rejections"].reason)


def test_blocked_exit_remains_pending_and_invalid_open_entry_is_cancelled():
    d, s, r, p, div = fixture_data(5)
    event(s, 0)
    d.loc[1, ["close", "ac"]] = 9.4
    d.loc[2, ["open", "ao"]] = 8.46  # 跌停开盘，不假设可立即卖出。
    d.loc[2, ["close", "ac"]] = 10.
    result = account(d, div, s, p, r, "NR7_ONLY", "STRESS", d.date.iloc[1])
    assert result["trades"].exit_date.iloc[0] == d.date.iloc[3]
    d2, s2, r2, p2, div2 = fixture_data(3)
    event(s2, 0, stop=10.1)
    invalid = account(d2, div2, s2, p2, r2, "NR7_ONLY", "STRESS", d2.date.iloc[1])
    assert invalid["trades"].empty
    assert "OPEN_ALREADY_BELOW_STRUCTURAL_STOP" in set(invalid["rejections"].reason)


def test_sharpe_includes_cash_days_and_uses_daily_net_wealth():
    r = np.array([.01, -.005, 0., 0., 0.])
    m = return_statistics(r)
    assert m["days"] == 5
    assert np.isclose(m["net_sharpe"], r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert np.isclose(m["cumulative_return"], np.prod(1 + r) - 1)
    assert m["net_sharpe"] != return_statistics(r[:2])["net_sharpe"]
