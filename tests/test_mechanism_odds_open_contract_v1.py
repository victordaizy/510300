"""直接影响可实现性的必要测试：时序、订单、权益和成本。"""
import numpy as np
import pandas as pd

from research import mechanism_odds_open_contract_v1 as s


def market(n=650):
    dates = pd.bdate_range("2014-07-01", periods=n)
    index = np.arange(n)
    close = 3 * np.exp(.0001*index+.04*np.sin(index/17)+.008*np.cos(index/3))
    op = close*np.exp(.003*np.cos(index/5))
    return pd.DataFrame({"date": dates, "open": op, "close": close,
        "high": np.maximum(close, op)*1.01, "low": np.minimum(close, op)*.99,
        "previous_close": np.r_[close[0], close[:-1]], "dividend": np.zeros(n), "symbol": "510300.SH"})


def account_engine():
    return s.engine(s.PARENT / "code/account_engine.py")


def test_future_prices_and_unmatured_labels_cannot_change_forecast():
    m = market()
    i = 580
    x, y = s.features(m), s.labels(m)
    ids, fit = s.forecast_one(m, x, y, i)
    changed = m.copy()
    changed.loc[i+1:, ["open", "close", "high", "low"]] *= 9
    ids_changed, _ = s.forecast_one(changed, s.features(changed), s.labels(changed), i)
    np.testing.assert_array_equal(ids, ids_changed)
    assert fit["latest_maturity_idx"] <= i
    prefix = m.iloc[:i+1].copy()
    ids_prefix, _ = s.forecast_one(prefix, s.features(prefix), s.labels(prefix), i)
    np.testing.assert_array_equal(ids, ids_prefix)


def test_terminal_label_includes_owned_dividend_once():
    m = market(12)
    m.loc[:, "open"] = 10.
    m.loc[:, "close"] = 10.
    m.loc[3, "dividend"] = 1.
    m.loc[3:, ["open", "close"]] = 9.
    y = s.labels(m)
    assert y[1, 0] == 0.
    # 原点收盘若已经除息，后续不能再次取得这次分红。
    assert y[3, 0] == 0.


def test_t_plus_one_blocks_sale_of_new_inventory():
    e = account_engine()
    account = e.Account(20000.)
    row = market(12).iloc[5]
    row["open"], row["previous_close"] = 3., 3.
    buy = s.execute(e, account, {"quantity": 1000, "limit_price": 3.1}, row, 5, s.COSTS["STRESS"])
    assert buy["filled_quantity"] == 1000
    same_day = s.execute(e, account, {"quantity": -1000, "limit_price": None}, row, 5, s.COSTS["STRESS"])
    assert same_day["filled_quantity"] == 0
    next_day = s.execute(e, account, {"quantity": -1000, "limit_price": None}, row, 6, s.COSTS["STRESS"])
    assert next_day["filled_quantity"] == -1000


def test_limit_is_precommitted_and_low_gap_does_not_cancel():
    e = account_engine()
    row = market(12).iloc[5]
    row["previous_close"] = 3.
    plan = {"quantity": 1000, "limit_price": 3.01}
    row["open"] = 3.1
    nofill = s.execute(e, e.Account(20000.), plan, row, 5, s.COSTS["STRESS"])
    assert nofill["status"] == "UNFILLED_PRECOMMITTED_LIMIT"
    assert nofill["requested_quantity"] == 1000
    row["open"] = 2.9
    fill = s.execute(e, e.Account(20000.), plan, row, 5, s.COSTS["STRESS"])
    assert fill["filled_quantity"] == 1000
    assert fill["fill_price"] <= plan["limit_price"]
    assert plan == {"quantity": 1000, "limit_price": 3.01}


def test_price_quote_covers_minimum_commission_without_using_open():
    e = account_engine()
    scenarios = np.linspace(-.015, .045, 126)
    plan = s.buy_plan(e, 20000., 20000., 20000., 3., 0., scenarios, .02, "LIMIT_VALUE")
    assert plan["quantity"] > 0
    assert plan["quantity"] % 100 == 0
    assert plan["expected_net_at_limit"] > 0
    assert plan["limit_price"] < plan["expected_terminal_net_per_share"]
    assert plan["quantity"] * max(3., plan["limit_price"]) < .5*20000


def test_dividend_is_not_promised_to_ex_date_buyer():
    e = account_engine()
    scenarios = np.linspace(-.01, .03, 126)
    plain = s.buy_plan(e, 200000., 200000., 200000., 3., 0., scenarios, .02, "LIMIT_VALUE")
    ex = s.buy_plan(e, 200000., 200000., 200000., 3., .1, scenarios, .02, "LIMIT_VALUE")
    assert .098 <= plain["limit_price"]-ex["limit_price"] <= .102


def test_cash_sharpe_is_undefined():
    n = 50
    d = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "equity": 20000., "net_return": 0.,
        "exposure": 0., "filled_quantity": 0, "shares_before": 0, "commission": 0., "slippage_cost": 0.,
        "notional": 0., "terminal_exit_reserve": 0., "accounting_error": 0., "risk_stopped": False})
    result = s.metrics(d, 20000)
    assert result["net_sharpe"] is None
    assert result["hac20_sharpe_diagnostic"] is None
    assert result["calendar_cagr"] == 0.
