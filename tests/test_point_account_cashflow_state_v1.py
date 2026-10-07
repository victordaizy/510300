"""用手算现金流、跨周期股息和未来前缀验证隔离研究的时间与归属。"""
import numpy as np
import pandas as pd
import pytest

from research.point_account_cashflow_state_v1 import reconstruct_states


def fixture():
    dates = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame({"date": dates, "close": [10., 11., 10., 12., 12., 13.]})
    orders = pd.DataFrame([
        {"date": dates[1], "origin": dates[0], "cycle_id": 1, "side": "BUY", "quantity": 100, "fill_price": 10.01, "commission": 5.},
        {"date": dates[2], "origin": dates[1], "cycle_id": 1, "side": "BUY", "quantity": 100, "fill_price": 11.01, "commission": 5.},
        {"date": dates[3], "origin": dates[2], "cycle_id": 1, "side": "SELL", "quantity": 100, "fill_price": 9.99, "commission": 5.},
        {"date": dates[4], "origin": dates[3], "cycle_id": 1, "side": "SELL", "quantity": 100, "fill_price": 11.99, "commission": 5.},
    ])
    dividends = pd.DataFrame({"record_date": [dates[2]], "ex_date": [dates[3]], "payment_date": [dates[5]], "cash_dividend_per_share": [.1]})
    return prices, dividends, orders


def test_multiple_fills_keep_realized_cashflow_and_prefix_buy_denominator():
    prices, dividends, orders = fixture()
    held, daily, terminal = reconstruct_states(prices, dividends, orders, "STRESS")
    np.testing.assert_allclose(held.known_buy_debit_cny, [1006., 2112., 2112.])
    np.testing.assert_allclose(held.actual_close_mark_pnl_cny, [94., -112., 102.])
    np.testing.assert_allclose(held.actual_close_mark_return, [94. / 1006., -112. / 2112., 102. / 2112.])
    assert held.shares.tolist() == [100, 200, 100]
    assert held.known_extra_buy_orders.tolist() == [0, 1, 1]
    assert held.known_sell_orders.tolist() == [0, 0, 1]
    assert terminal.known_sell_net_cny.iloc[0] == pytest.approx(2188.)
    assert terminal.known_terminal_mark_pnl_cny.iloc[0] == pytest.approx(96.)
    assert daily.equity.iloc[-1] == pytest.approx(200096.)
    np.testing.assert_allclose(daily.equity, [200000., 200094., 199888., 200102., 200096., 200096.])


def test_dividend_after_exit_belongs_to_record_cycle_and_payment_is_not_profit():
    prices, dividends, orders = fixture()
    dates = prices.date
    dividends.loc[0, "ex_date"] = dates.iloc[5]
    dividends.loc[0, "payment_date"] = dates.iloc[5]
    second = pd.DataFrame([{ "date": dates.iloc[5], "origin": dates.iloc[4], "cycle_id": 2, "side": "BUY", "quantity": 100, "fill_price": 12.01, "commission": 5.}])
    held, daily, terminal = reconstruct_states(prices, dividends, pd.concat([orders, second], ignore_index=True), "BASE")
    assert daily.dividend_accrual.tolist() == [0., 0., 0., 0., 0., 20.]
    assert daily.dividend_paid.tolist() == [0., 0., 0., 0., 0., 20.]
    by_cycle = terminal.set_index("actual_cycle_id")
    assert by_cycle.loc[1, "known_dividend_cny"] == pytest.approx(20.)
    assert by_cycle.loc[2, "known_dividend_cny"] == pytest.approx(0.)
    assert by_cycle.loc[1, "known_terminal_mark_pnl_cny"] == pytest.approx(96.)
    assert by_cycle.loc[2, "known_terminal_mark_pnl_cny"] == pytest.approx(94.)
    assert daily.receivable.iloc[-1] == pytest.approx(0.)
    assert daily.equity.iloc[-1] == pytest.approx(200190.)
    assert held.loc[held.actual_cycle_id.eq(2), "known_dividend_cny"].iloc[0] == 0.


def test_future_prices_orders_and_dividend_amount_cannot_change_known_prefix():
    prices, dividends, orders = fixture()
    cutoff = prices.date.iloc[2]
    full_held, full_daily, _ = reconstruct_states(prices, dividends, orders, "STRESS")
    prefix_held, prefix_daily, _ = reconstruct_states(prices, dividends, orders, "STRESS", cutoff=cutoff)
    pd.testing.assert_frame_equal(full_held.loc[full_held.origin.le(cutoff)].reset_index(drop=True), prefix_held)
    pd.testing.assert_frame_equal(full_daily.loc[full_daily.date.le(cutoff)].reset_index(drop=True), prefix_daily)
    changed_prices = prices.copy()
    changed_prices.loc[changed_prices.date.gt(cutoff), "close"] *= 3.
    changed_orders = orders.copy()
    changed_orders.loc[changed_orders.date.gt(cutoff), "fill_price"] *= 2.
    changed_dividends = dividends.copy()
    changed_dividends["cash_dividend_per_share"] *= 5.
    changed_held, changed_daily, _ = reconstruct_states(changed_prices, changed_dividends, changed_orders, "STRESS")
    pd.testing.assert_frame_equal(full_held.loc[full_held.origin.le(cutoff)].reset_index(drop=True), changed_held.loc[changed_held.origin.le(cutoff)].reset_index(drop=True))
    pd.testing.assert_frame_equal(full_daily.loc[full_daily.date.le(cutoff)].reset_index(drop=True), changed_daily.loc[changed_daily.date.le(cutoff)].reset_index(drop=True))


def test_reject_negative_inventory_and_cycle_misattribution():
    prices, dividends, orders = fixture()
    broken = orders.copy()
    broken.loc[broken.index[-1], "quantity"] = 200
    with pytest.raises(ValueError, match="负库存"):
        reconstruct_states(prices, dividends, broken, "STRESS")
    wrong_cycle = orders.copy()
    wrong_cycle.loc[wrong_cycle.index[2], "cycle_id"] = 2
    with pytest.raises(ValueError, match="活跃周期"):
        reconstruct_states(prices, dividends, wrong_cycle, "STRESS")
