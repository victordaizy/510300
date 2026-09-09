"""验证持续旧信号、信号重新出现及买入未成交时的重新进入状态。"""
import numpy as np
import pandas as pd

from research.learned_cycle_exit_v1 import ExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit


def setup():
    dates = pd.bdate_range("2020-01-01", periods=10)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.,
                         "mom5": .01, "mom20": .02, "sma120": .03, "vol20": .15})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    rule = {"entry": np.ones(10, dtype=int), "exit": {1: np.zeros(10, dtype=bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    model = {"fit_index": 0, "latest_exit_index": 0, "status": "FIT_COMPLETE", "model": {
        "kind": "RIDGE", "mean": [0.] * 8, "scale": [1.] * 8, "coefficients": [0.] * 8, "feature_clip": 5., "intercept": -.02}}
    return [data, div, cfg, cost, str(dates[1].date()), rule, spec, ExitController(data, [model])]


def test_continuously_active_signal_does_not_reenter_after_exit():
    args = setup()
    ledger, decisions, cycles = simulate_rearmed_exit(*args)
    assert (ledger.filled_quantity > 0).sum() == 1
    assert (ledger.filled_quantity < 0).sum() == 1
    assert len(cycles) == 1
    assert "旧入场条件" in decisions.iloc[-1].action


def test_signal_reset_and_original_wait_are_both_required():
    args = setup()
    args[5]["entry"][4] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args)
    buys = ledger[ledger.filled_quantity > 0].date.to_list()
    assert buys == [args[0].date.iloc[1], args[0].date.iloc[6]]
    assert decisions.loc[decisions.origin_index == 4, "entry_rearmed"].iloc[0]


def test_unfilled_buy_does_not_consume_entry_permission():
    args = setup()
    args[0].loc[1, "open"] = 11.
    ledger, decisions, cycles = simulate_rearmed_exit(*args)
    assert ledger.iloc[0].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger[ledger.filled_quantity > 0].date.iloc[0] == args[0].date.iloc[2]


def test_rearm_control_works_without_any_learning_model():
    args = setup()
    args[-1] = None
    args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args)
    assert (ledger.filled_quantity > 0).sum() == 1
    assert "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.accounting_error.abs().max() < 1e-7
