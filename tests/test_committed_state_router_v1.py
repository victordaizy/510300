"""确认真实持仓采用入场选定的策略直到它自身退出。"""
import numpy as np
import pandas as pd
from research.committed_state_router_v1 import make_rule
from research.simple_price_entry_exit_v1 import simulate_policy


def test_market_switch_cannot_replace_exit_rule_of_the_actual_cycle():
    dates = pd.bdate_range("2020-01-01", periods=10)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.})
    states = pd.DataFrame({"trend_state": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                           "learned_state": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                           "strong_market": [True, True, False, False, False, False, False, False, False, False],
                           "selector_available": True})
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    spec = {"cooldown": 0, "modes": {i: {"loss": None, "trail": None, "take": None, "days": None} for i in [1, 2]}}
    ledger, decisions, cycles = simulate_policy(data, dividends, cfg, cost, str(dates[1].date()), make_rule(states), spec)
    assert ledger[ledger.filled_quantity > 0].date.to_list() == [dates[1], dates[6]]
    assert ledger[ledger.filled_quantity < 0].date.to_list() == [dates[5], dates[8]]
    assert cycles["mode"].to_list() == [1, 2]
    assert ledger.accounting_error.abs().max() < 1e-7
