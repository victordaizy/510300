"""检验固定订单、期限、分红会计与永久停止对实际账户的影响。"""
import numpy as np
import pandas as pd

from research import constituent_dividend_calendar_account_v1 as s


def fixture():
    dates = pd.bdate_range("2016-12-30", periods=40)
    market = pd.DataFrame({"date": dates, "open": 10., "close": 10., "high": 10.1, "low": 9.9,
                           "previous_close": 10., "dividend": 0.})
    events = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    origins = {i: {"information_status": "READY", "selected": True} for i in [2, 7, 12, 17, 22]}
    risk = np.full(len(market), .02)
    engine = s.shared.engine(s.shared.OUT / "code/account_engine.py")
    return market, events, origins, risk, engine


def test_overlapping_signals_do_not_add_or_extend_twenty_open_contract():
    market, events, origins, risk, e = fixture()
    ledger, orders, _ = s.simulate(market, events, risk, origins, s.PRIMARY, 200000, "STRESS", e)
    entries = ledger.loc[ledger.filled_quantity.gt(0)]
    assert len(entries) == 1 and entries.idx.iloc[0] == 3
    exits = orders.loc[orders.reason.eq("原20日期限到达")]
    assert len(exits) == 1 and exits.idx.iloc[0] == 23
    assert ledger.loc[ledger.idx.ge(23), "shares"].eq(0).all()


def test_next_open_gap_cannot_recompute_the_precommitted_quantity():
    market, events, origins, risk, e = fixture()
    first, _, plans = s.simulate(market, events, risk, origins, s.PRIMARY, 200000, "STRESS", e)
    changed = market.copy()
    changed.loc[3, "open"] = 10.5
    second, _, later_plans = s.simulate(changed, events, risk, origins, s.PRIMARY, 200000, "STRESS", e)
    planned = int(plans.loc[plans.decision_idx.eq(2), "quantity"].iloc[0])
    assert planned == int(later_plans.loc[later_plans.decision_idx.eq(2), "quantity"].iloc[0])
    assert first.loc[first.idx.eq(3), "requested_quantity"].iloc[0] == planned
    assert second.loc[second.idx.eq(3), "requested_quantity"].iloc[0] == planned


def test_dividend_receivable_and_payment_are_not_double_counted():
    market, _, origins, risk, e = fixture()
    events = pd.DataFrame([{"record_date": market.date.iloc[5], "ex_date": market.date.iloc[6],
                            "payment_date": market.date.iloc[9], "cash_dividend_per_share": .1}])
    market.loc[6:, ["open", "close"]] = 9.9
    market.loc[7:, "previous_close"] = 9.9
    market.loc[6, "dividend"] = .1
    ledger, _, _ = s.simulate(market, events, risk, origins, s.PRIMARY, 200000, "STRESS", e)
    assert ledger.dividend_recognized.sum() > 0
    assert ledger.dividend_paid.sum() == ledger.dividend_recognized.sum()
    assert ledger.loc[ledger.idx.eq(7), "dividend_receivable"].iloc[0] > 0
    assert ledger.loc[ledger.idx.eq(9), "dividend_receivable"].iloc[0] == 0
    assert ledger.accounting_error.abs().max() < 1e-7


def test_drawdown_stop_is_permanent_even_when_new_signals_arrive():
    market, events, origins, risk, e = fixture()
    # 连续方向跌停令减仓单无法成交，回撤触发后首个可卖开盘退出。
    market.loc[4, ["open", "close"]] = 9.
    market.loc[5, ["open", "close"]] = 8.1
    market.loc[6:, ["open", "close"]] = 7.29
    market["previous_close"] = market.close.shift().fillna(10.)
    ledger, _, _ = s.simulate(market, events, risk, origins, s.PRIMARY, 200000, "STRESS", e)
    assert ledger.risk_stopped.any()
    first_stop = int(ledger.loc[ledger.risk_stopped, "idx"].iloc[0])
    assert ledger.loc[ledger.idx.ge(first_stop), "risk_stopped"].all()
    assert not ledger.loc[ledger.idx.gt(first_stop), "filled_quantity"].gt(0).any()
