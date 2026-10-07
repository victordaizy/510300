"""用可手算的订单与分红验证账户路径解释，防止把未来峰值当成信号。"""
import numpy as np
import pandas as pd
import pytest

from research.point_weight_path_inputs_v1 import cycle_path, describe_cycle


def fixture():
    dates = pd.bdate_range("2024-01-02", periods=6)
    p = pd.DataFrame({"date": dates, "open": [10., 10., 11., 10., 12., 12.], "close": [10., 11., 10., 12., 12., 13.]})
    orders = pd.DataFrame([
        {"date": dates[1], "origin": dates[0], "cycle_id": 1, "side": "BUY", "quantity": 100, "fill_price": 10.01, "commission": 5., "slippage": 1.},
        {"date": dates[2], "origin": dates[1], "cycle_id": 1, "side": "BUY", "quantity": 100, "fill_price": 11.01, "commission": 5., "slippage": 1.},
        {"date": dates[3], "origin": dates[2], "cycle_id": 1, "side": "SELL", "quantity": 100, "fill_price": 9.99, "commission": 5., "slippage": 1.},
        {"date": dates[4], "origin": dates[3], "cycle_id": 1, "side": "SELL", "quantity": 100, "fill_price": 11.99, "commission": 5., "slippage": 1.},
    ])
    div = pd.DataFrame({"record_date": [dates[2]], "ex_date": [dates[3]], "payment_date": [dates[5]], "cash_dividend_per_share": [.1]})
    trade = {"cycle_id": 1, "entry_date": dates[1], "exit_date": dates[4], "status": "COMPLETE", "net_pnl": 96., "net_return": 96. / 2112., "buy_debit": 2112., "entry_equity": 200000., "dividend_cny": 20., "exit_reason": "PARENT_ZERO"}
    decisions = pd.DataFrame({"origin": dates[1:5], "shares_before": [100, 200, 100, 0], "desired_shares": [200, 100, 0, 0]})
    return p, orders, div, trade, decisions


def test_actual_inventory_cash_dividend_and_next_open_bridge():
    p, orders, div, trade, decisions = fixture()
    path = cycle_path(p, div, orders, trade, "STRESS", p.date.iloc[-1])
    np.testing.assert_allclose(path.net_increment, [94., -206., 214., -6.], atol=1e-9)
    assert path.shares.tolist() == [100, 200, 100, 0]
    assert path.dividend_accrual.tolist() == [0., 0., 20., 0.]
    assert path.net_mark.iloc[-1] == pytest.approx(96.)
    description = describe_cycle(path, trade, decisions)
    assert description["entry_day_net_pnl"] == pytest.approx(94.)
    assert description["holding_interior_net_pnl"] == pytest.approx(8.)
    assert description["exit_day_net_pnl"] == pytest.approx(-6.)
    assert description["exit_intent_to_execution_sessions"] == 1
    assert description["first_exit_intent_origin"] == p.date.iloc[3]


def test_path_marks_use_only_observed_orders_prices_and_dividend_recognition():
    p, orders, div, trade, _ = fixture()
    full = cycle_path(p, div, orders, trade, "STRESS", p.date.iloc[-1])
    cutoff = p.date.iloc[2]
    prefix_trade = {**trade, "exit_date": pd.NaT, "status": "RIGHT_CENSORED", "net_pnl": np.nan, "net_return": np.nan, "dividend_cny": 0.}
    prefix = cycle_path(p.loc[p.date.le(cutoff)], div, orders.loc[orders.date.le(cutoff)], prefix_trade, "STRESS", cutoff)
    fields = ["date", "shares", "net_increment", "net_mark", "hypothetical_close_liquidation_pnl", "dividend_accrual"]
    pd.testing.assert_frame_equal(full.loc[full.date.le(cutoff), fields].reset_index(drop=True), prefix[fields])
    changed = p.copy()
    changed.loc[changed.date.gt(cutoff), "close"] *= 2
    censored_trade = {**trade, "exit_date": pd.NaT, "status": "RIGHT_CENSORED", "net_pnl": np.nan, "net_return": np.nan}
    tail_changed = cycle_path(changed, div, orders, censored_trade, "STRESS", changed.date.iloc[-1])
    pd.testing.assert_frame_equal(full.loc[full.date.le(cutoff), fields].reset_index(drop=True), tail_changed.loc[tail_changed.date.le(cutoff), fields].reset_index(drop=True))


def test_reject_negative_inventory_or_wrong_saved_profit():
    p, orders, div, trade, _ = fixture()
    broken = orders.copy()
    broken.loc[broken.index[-1], "quantity"] = 200
    with pytest.raises(ValueError, match="负库存"):
        cycle_path(p, div, broken, trade, "STRESS", p.date.iloc[-1])
    with pytest.raises(ValueError, match="最终净损益"):
        cycle_path(p, div, orders, {**trade, "net_pnl": 97.}, "STRESS", p.date.iloc[-1])
