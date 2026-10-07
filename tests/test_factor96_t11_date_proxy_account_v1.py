"""T11合成账户验证：因果时钟、退出、分红、风险和成交边界。"""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from research import factor96_t11_date_proxy_account_v1 as study


@pytest.fixture(scope="module")
def engine():
    path = Path(__file__).resolve().parents[1] / "reports/research/510300_factor96_daily_state_shrink_v1/code/account_engine.py"
    spec = importlib.util.spec_from_file_location("t11_test_frozen_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture(n=45):
    dates = pd.bdate_range("2020-01-01", periods=n)
    market = pd.DataFrame({"date": dates, "open": 2., "high": 2.1, "low": 1.9,
                           "close": 2., "previous_close": 2., "dividend": 0.})
    features = pd.DataFrame({"date": dates, "wealth": 1., "low_w": .9, "es95": .01})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    return market, features, dividends


def signal(i=5, stop=.9):
    return {"observation_idx": i, "FULL": True, "FINANCIAL_ALL": True, "FINANCIAL_COMMON": True,
            "ABOVE_MEDIAN": False, "signal_id": str(i), "frozen_stop": stop,
            "anchors": (("600001.SH", pd.Timestamp("2019-09-30")),)}


def run(engine, market, features, dividends, signals=None, facts=None, lag=1, end=None, risk=None):
    return study.simulate(market, features, dividends, signals if signals is not None else [signal()],
                          facts or {}, "FULL", lag, 200000, "STRESS", market.date.iloc[2],
                          end if end is not None else market.date.iloc[-1], engine,
                          risk or (lambda e, a, reference, peak, es: 1000))


@pytest.mark.parametrize("lag", [1, 2])
def test_entry_clock_twenty_opens_no_add_or_same_day_reentry(engine, lag):
    d, x, div = fixture()
    entry, exit_day = 5+lag, 25+lag
    l, decisions, orders, _ = run(engine, d, x, div, [signal(), signal(10), signal(exit_day-lag)], lag=lag)
    assert orders.idx.tolist() == [entry, exit_day]
    assert orders.filled_quantity.tolist() == [1000, -1000]
    assert decisions.loc[decisions.idx.eq(exit_day), "scheduled_signal_ignored_while_held"].item()
    assert l.loc[l.idx.lt(entry) | l.idx.gt(exit_day), "net_return"].eq(0).all()
    assert len(l) == len(d)-2 and l.accounting_error.abs().max() < 1e-7


def test_delayed_entry_uses_known_close_and_future_changes_do_not_rewrite_orders(engine):
    d, x, div = fixture()
    x.loc[6, "wealth"] = .89
    l, decisions, orders, _ = run(engine, d, x, div, lag=2)
    assert orders.empty and l.shares.eq(0).all()
    assert decisions.loc[decisions.idx.eq(7), "reason"].item() == "延迟入场前价格条件已失效"
    d1, x1, div = fixture()
    prior = run(engine, d1, x1, div)[0]
    d1.loc[20:, ["open", "close", "previous_close"]] = 2.05
    x1.loc[20:, "wealth"] = 1.025
    revised = run(engine, d1, x1, div)[0]
    pd.testing.assert_frame_equal(prior.loc[prior.idx.lt(20)], revised.loc[revised.idx.lt(20)])


@pytest.mark.parametrize("lag", [1, 2])
def test_unknown_newer_fact_exits_with_lag_and_limit_down_persists(engine, lag):
    d, x, div = fixture()
    front = pd.DataFrame([{"available_date": d.date.iloc[10], "report_period": pd.Timestamp("2019-12-31"),
                           "ts_code": "600001.SH", "announcement_id": "new-unknown"}])
    companies = pd.DataFrame(columns=["announcement_id", "known", "L02", "L04_change", "L04_industry_z"])
    facts, audit = study.fact_events(front, companies, d)
    assert audit.adverse.item() and not audit.measurement_known.item()
    d.loc[10+lag, "open"] = 1.8
    l, decisions, orders, checks = run(engine, d, x, div, facts=facts, lag=lag)
    assert orders.idx.tolist() == [5+lag, 10+lag, 11+lag]
    assert orders.status.iloc[1] == "UNFILLED_DIRECTIONAL_LIMIT"
    assert orders.filled_quantity.iloc[-1] == -1000
    assert checks.execution_idx.item() == 10+lag
    assert decisions.loc[decisions.idx.eq(11+lag), "reason"].item() == "后续财报反证或未知"
    assert l.accounting_error.abs().max() < 1e-7


def test_dividend_record_ex_receivable_and_payment_are_distinct(engine):
    d, x, _ = fixture()
    d.loc[8:, ["open", "close"]] = 1.9
    d.loc[9:, "previous_close"] = 1.9
    d.loc[8, "dividend"] = .1
    div = pd.DataFrame([{"record_date": d.date.iloc[7], "ex_date": d.date.iloc[8],
                         "payment_date": d.date.iloc[10], "cash_dividend_per_share": .1}])
    l, _, _, _ = run(engine, d, x, div)
    assert l.loc[l.idx.eq(8), "dividend_recognized"].item() == 100
    assert l.loc[l.idx.eq(9), "dividend_receivable"].item() == 100
    assert l.loc[l.idx.eq(10), "dividend_paid"].item() == 100
    assert l.loc[l.idx.eq(10), "dividend_receivable"].item() == 0
    assert l.loc[l.idx.eq(8), "net_return"].item() == 0
    assert l.accounting_error.abs().max() < 1e-7


def test_risk_reductions_never_restore_trimmed_quantity(engine):
    d, x, div = fixture()
    x.loc[7, "es95"] = .5
    risk = lambda e, a, reference, peak, es: 500 if es > .1 else 1000
    l, _, orders, _ = run(engine, d, x, div, risk=risk)
    assert orders.idx.tolist() == [6, 8, 26]
    assert orders.filled_quantity.tolist() == [1000, -500, -500]
    assert l.loc[l.idx.between(8, 25), "shares"].eq(500).all()


def test_drawdown_stop_never_restarts_after_flat(engine):
    d, x, div = fixture()
    d.loc[7:, ["open", "close"]] = 1.5
    d.loc[8:, "previous_close"] = 1.5
    x.loc[7:, "wealth"] = .75
    l, _, orders, _ = run(engine, d, x, div, [signal(), signal(15, stop=.6)],
                          risk=lambda e, a, reference, peak, es: 49000)
    assert l.loc[l.idx.ge(7), "risk_stopped"].all()
    assert orders.idx.tolist() == [6, 8]
    assert l.loc[l.idx.ge(8), "shares"].eq(0).all()


def test_terminal_reserve_and_failed_entry_no_retry(engine):
    d, x, div = fixture()
    l, _, orders, _ = run(engine, d, x, div, end=d.date.iloc[12])
    assert len(orders) == 1 and l.terminal_unliquidated.iloc[-1]
    assert l.terminal_exit_reserve.iloc[-1] > 0
    d.loc[6, "open"] = 2.2
    l, _, orders, _ = run(engine, d, x, div)
    assert len(orders) == 1 and orders.status.item() == "UNFILLED_DIRECTIONAL_LIMIT"
    assert l.shares.eq(0).all()


def test_t_plus_one_and_minimum_commission_in_frozen_engine(engine):
    a = engine.Account(20000.)
    cost, cfg = study.COSTS["STRESS"], {"lot": 100, "tick": .001, "limit_fraction": .1}
    buy = engine.execute_order(a, 100, 2., 2., 0., 5, cost, cfg)
    same_day = engine.execute_order(a, -100, 2., 2., 0., 5, cost, cfg)
    sell = engine.execute_order(a, -100, 2., 2., 0., 6, cost, cfg)
    assert buy["commission"] == sell["commission"] == 5.
    assert same_day["filled_quantity"] == 0 and sell["filled_quantity"] == -100


def test_policy_unknown_and_complementary_direction_partition():
    d, x, _ = fixture()
    dates = d.date.iloc[[5, 6, 7]]
    o02 = pd.DataFrame({"date": dates, "cohort_known": True, "positive_financial_measurement": True,
                        "condition_known": [False, True, True], "reference_count": [1, 2, 2],
                        "first_response": [-.01, -.01, .02], "reference_median": [np.nan, 0., 0.], "status": "合成"})
    companies = pd.DataFrame({"date": dates, "ts_code": "600001.SH", "report_period": pd.Timestamp("2019-09-30"), "known": True})
    records = study.signal_table(o02, companies, d, x)
    assert [r["FULL"] for r in records] == [False, True, False]
    assert [r["ABOVE_MEDIAN"] for r in records] == [False, False, True]
    assert [r["FINANCIAL_COMMON"] for r in records] == [False, True, True]
    assert all(r["FINANCIAL_ALL"] for r in records)
