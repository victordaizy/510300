"""限价边界、可卖旧份额、两成交情景及分红路径。"""
import numpy as np
import pandas as pd
from research.standing_profit_limit_account_v1 import profit_limit_price, required_reference_price, standing_execution_reference, simulate_profit_limit
from research.simple_price_entry_exit_v1 import simulate_policy
from research.adaptive_allocation_v1 import fill_price

CFG = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1, "daily_volume_cap": .01}
BASE = {"commission": .0002, "minimum": 5., "slippage": .0005}
SPEC = {"cooldown": 2, "modes": {1: {"loss": .04, "trail": None, "take": .06, "days": 10}}}
EMPTY_DIV = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])


def market(n=14):
    d = pd.DataFrame({"date": pd.bdate_range("2020-01-02", periods=n), "open": 10., "close": 10., "high": 10., "low": 9.9, "previous_close": 10., "dividend": 0., "volume": 1.e8})
    entry = np.zeros(n, int)
    entry[2] = 1
    return d, {"entry": entry, "exit": {1: np.zeros(n, bool)}}


def simulate(d, rule, assumption="THROUGH_PRICE", div=EMPTY_DIV):
    return simulate_profit_limit(d, div, CFG, BASE, str(d.date.iloc[2].date()), rule, SPEC, assumption)


def test_limit_rounds_up_and_original_slippage_never_breaks_sell_limit():
    price = profit_limit_price(1000, 10005., 20., .06, .001)
    assert price*1000+20 >= 10005.*1.06-1e-10
    assert (price-.001)*1000+20 < 10005.*1.06
    for slippage in [0., .0005, .001]:
        cost = {**BASE, "slippage": slippage}
        required = required_reference_price(price, cost, .001)
        assert fill_price(required, -1, cost, .001) >= price-1e-10


def test_only_touch_missing_volume_and_daily_capacity_do_not_create_fill():
    limit = 10.6
    required = required_reference_price(limit, BASE, .001)
    args = [limit, 10., required, 10., 1.e8, 10., 0., 1000, BASE, CFG, "THROUGH_PRICE"]
    assert standing_execution_reference(*args)[0] is None
    args[2] = required+.001
    assert standing_execution_reference(*args)[0] == required
    args[4] = np.nan
    assert standing_execution_reference(*args)[1] == "NO_VIEW_INTRADAY_EXECUTION_INPUT"
    args[4] = 100.
    assert standing_execution_reference(*args)[1] == "UNFILLED_DAILY_VOLUME_CAP"


def test_new_buy_cannot_sell_same_day_and_old_shares_can_fill_before_close_reversal():
    d, rule = market()
    d.loc[3:4, "high"] = 11.
    ledger, decisions, cycles = simulate(d, rule)
    bought = ledger[ledger.date.eq(d.date.iloc[3])].iloc[0]
    sold = ledger[ledger.date.eq(d.date.iloc[4])].iloc[0]
    assert bought.filled_quantity > 0 and pd.isna(bought.standing_limit_price)
    assert sold.filled_quantity == -bought.shares and sold.sellable_before == bought.shares
    assert sold.fill_price >= sold.standing_limit_price and sold.execution_clock == "INTRADAY_CONDITIONAL_PROFIT_LIMIT"
    assert cycles.holding_intervals.iloc[0] == 1 and sold.mark == 10.
    assert ledger.accounting_error.abs().max() < 1e-6


def test_conservative_scenario_waits_for_close_but_both_take_known_open_gap():
    d, rule = market()
    d.loc[4, "high"] = 11.
    ledger, _, _ = simulate(d, rule, "CLOSE_STILL_ABOVE")
    assert ledger.loc[ledger.date.eq(d.date.iloc[4]), "shares"].iloc[0] > 0
    d.loc[4, "open"] = 10.9
    for mode in ["THROUGH_PRICE", "CLOSE_STILL_ABOVE"]:
        ledger, _, _ = simulate(d, rule, mode)
        row = ledger[ledger.date.eq(d.date.iloc[4])].iloc[0]
        assert row.shares == 0 and row.execution_clock == "OPEN_MARKETABLE_PROFIT_LIMIT" and row.open_price == 10.9


def test_dividend_changes_old_limit_and_rights_survive_sale_until_payment():
    d, rule = market()
    d.loc[4:, ["open", "close"]] = 9.8
    d.loc[4, "dividend"] = .2
    d.loc[4, "high"] = 10.7
    d.loc[5:, "previous_close"] = 9.8
    div = pd.DataFrame([{"record_date": d.date.iloc[3], "ex_date": d.date.iloc[4], "payment_date": d.date.iloc[7], "cash_dividend_per_share": .2}])
    ledger, _, cycles = simulate(d, rule, div=div)
    sale = ledger[ledger.date.eq(d.date.iloc[4])].iloc[0]
    assert sale.shares == 0 and sale.known_cycle_dividends_before == sale.shares_before*.2
    assert sale.dividend_receivable == sale.shares_before*.2
    paid = ledger[ledger.date.eq(d.date.iloc[7])].iloc[0]
    assert paid.dividend_receivable == 0 and paid.dividend_paid == sale.dividend_receivable and abs(paid.net_return) < 1e-12
    assert cycles.dividend_cny.iloc[0] == sale.dividend_receivable and ledger.accounting_error.abs().max() < 1e-6


def test_disabled_matches_original_and_future_prices_do_not_change_past_orders():
    d, rule = market()
    d.loc[4, "high"] = 11.
    old, old_decisions, old_cycles = simulate_policy(d, EMPTY_DIV, CFG, BASE, str(d.date.iloc[2].date()), rule, SPEC)
    disabled, decisions, cycles = simulate(d, rule, "DISABLED")
    for column in ["shares", "filled_quantity", "requested_quantity", "cash", "equity", "dividend_receivable", "net_return"]:
        np.testing.assert_allclose(disabled[column], old[column], atol=1e-10, rtol=0)
    pd.testing.assert_frame_equal(decisions, old_decisions)
    changed = d.copy()
    changed.loc[9:, ["open", "close", "high", "low"]] *= 1.05
    a, _, _ = simulate(d, rule)
    b, _, _ = simulate(changed, rule)
    pd.testing.assert_frame_equal(a[a.date.lt(d.date.iloc[9])], b[b.date.lt(d.date.iloc[9])])


def test_pending_original_exit_has_priority_over_new_profit_limit():
    d, rule = market()
    rule["exit"][1][3] = True
    d.loc[4, "high"] = 11.
    ledger, _, _ = simulate(d, rule)
    sale = ledger[ledger.date.eq(d.date.iloc[4])].iloc[0]
    assert sale.filled_quantity < 0 and sale.open_price == 10. and pd.isna(sale.standing_limit_price)
    assert sale.standing_limit_status == "SCHEDULED_EXIT_HAS_PRIORITY"
