"""核对事前假设状态、费用、未来模型隔离和入场暂缓。"""
import numpy as np
import pandas as pd
import pytest

from research.consistent_entry_exit_v1 import EntryGate
from research.consistent_entry_exit_account_v1 import simulate_consistent_exit
from research.learned_cycle_exit_v1 import ExitController
from research.intraday_overnight_increment_v1 import Account, affordable_quantity, commission, fill_price


def data():
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=10), "open": 10., "close": 10.,
                         "previous_close": 10., "dividend": 0., "mom5": .01, "mom20": .02, "sma120": .03, "vol20": .15})


def model(value, t=0):
    return {"fit_index": t, "latest_exit_index": t, "status": "FIT_COMPLETE", "model": {
        "kind": "RIDGE", "mean": [0.] * 8, "scale": [1.] * 8, "coefficients": [0.] * 8, "feature_clip": 5., "intercept": value}}


def test_negative_prediction_defers_entry_and_zero_is_allowed():
    frame = data()
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    gate = EntryGate(frame, [model(-.01), model(0., 2)], cost, {"tick": .001})
    assert not gate(0, Account(100000.), 1, 10000)["entry_allowed"]
    assert gate(2, Account(100000.), 1, 10000)["entry_allowed"]


def test_future_model_remains_missing_and_keeps_original_entry():
    frame = data()
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    gate = EntryGate(frame, [model(-.5, 4)], cost, {"tick": .001})
    result = gate(0, Account(100000.), 1, 10000)
    assert result["entry_allowed"]
    assert result["entry_prediction"] is None
    assert result["entry_model_status"] == "NO_VIEW_NO_MATURE_MODEL"


def test_hypothetical_state_uses_current_cost_and_does_not_change_account():
    frame = data()
    frame.loc[1, "open"] = 11.
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    account = Account(100000.)
    px = fill_price(10., 1, cost, .001)
    q = affordable_quantity(account.cash, px, cost, 100)
    result = EntryGate(frame, [model(.01)], cost, {"tick": .001})(0, account, 1, q)
    expected_cost = q * px + commission(q, px, cost)
    assert result["entry_hypothetical_cost"] == pytest.approx(expected_cost)
    assert result["entry_factor_cycle_return"] == pytest.approx(q * 10 / expected_cost - 1)
    assert result["entry_factor_log_holding_days"] == pytest.approx(np.log(2))
    assert account.cash == 100000. and account.shares == 0


def test_deferred_entry_preserves_permission_until_model_changes():
    frame = data()
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    rule = {"entry": np.ones(10, dtype=int), "exit": {1: np.zeros(10, dtype=bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    models = [model(-.01), model(.01, 4)]
    ledger, decisions, cycles = simulate_consistent_exit(frame, div, cfg, cost, str(frame.date.iloc[1].date()), rule, spec,
        ExitController(frame, models), EntryGate(frame, models, cost, cfg))
    buys = ledger[ledger.filled_quantity > 0]
    assert len(buys) == 1 and buys.date.iloc[0] == frame.date.iloc[5]
    assert (decisions[decisions.origin_index < 4].requested_quantity == 0).all()
    assert ledger.accounting_error.abs().max() < 1e-7
