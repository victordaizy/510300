"""核对进入上限的事前时点、除息调整、整手成交及原退出保持。"""
import numpy as np
import pandas as pd
from research.capped_entry_exit_account_v1 import simulate_capped_entry
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit


def setup():
    dates = pd.bdate_range("2020-01-01", periods=10)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "high": 10.5, "low": 9.5,
                         "previous_close": 10., "dividend": 0., "vol20": .01 * np.sqrt(242)})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1, "annual_days": 242}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    rule = {"entry": np.ones(10, dtype=int), "exit": {1: np.zeros(10, dtype=bool)}}
    rule["exit"][1][4] = True
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    return [data, div, cfg, cost, str(dates[1].date()), rule, spec]


def test_nonbinding_cap_keeps_original_account_economics_and_exits():
    args = setup()
    old = simulate_rearmed_exit(*args)[0]
    new = simulate_capped_entry(*args)[0]
    fields = ["date", "cash", "shares", "equity", "net_return", "commission", "slippage_cost", "filled_quantity", "status"]
    pd.testing.assert_frame_equal(new[fields], old[fields])


def test_rejected_open_does_not_fill_at_later_daily_low_or_consume_permission():
    args = setup()
    args[0].loc[1, ["open", "low"]] = [10.2, 8.]
    ledger, decisions, cycles = simulate_capped_entry(*args)
    assert ledger.iloc[0].entry_price_cap_rejected
    assert ledger.iloc[0].filled_quantity == 0
    assert ledger.iloc[0].status == "UNFILLED_ENTRY_PRICE_ABOVE_PRESET_CAP"
    assert decisions[decisions.origin_index.eq(1)].entry_rearmed.iloc[0]
    assert ledger[ledger.filled_quantity.gt(0)].date.iloc[0] == args[0].date.iloc[2]


def test_open_decision_uses_signal_day_volatility_not_execution_day_value():
    args = setup()
    args[0].loc[1, "open"] = 10.2
    args[0].loc[1, "vol20"] = 5.
    ledger, decisions, cycles = simulate_capped_entry(*args)
    assert ledger.iloc[0].filled_quantity == 0
    assert abs(ledger.iloc[0].entry_price_ceiling_after_ex_adjustment - 10.1) < 1e-10


def test_ex_dividend_adjustment_prevents_false_price_acceptance():
    args = setup()
    data = args[0]
    data.loc[1:, ["open", "close"]] = 9.9
    data.loc[1, ["open", "dividend"]] = [10.05, .1]
    data["previous_close"] = data.close.shift(1).fillna(10.)
    args[1] = pd.DataFrame([{"record_date": data.date.iloc[0], "ex_date": data.date.iloc[1],
                             "payment_date": data.date.iloc[3], "cash_dividend_per_share": .1}])
    ledger, decisions, cycles = simulate_capped_entry(*args)
    assert abs(ledger.iloc[0].entry_price_ceiling_after_ex_adjustment - 10.) < 1e-10
    assert ledger.iloc[0].entry_price_cap_rejected
    assert ledger[ledger.filled_quantity.gt(0)].date.iloc[0] == data.date.iloc[2]
    assert ledger.dividend_recognized.sum() == 0.


def test_fill_at_limit_is_allowed_and_actual_cash_still_caps_quantity():
    args = setup()
    args[0].loc[1, "open"] = 10.094
    ledger, decisions, cycles = simulate_capped_entry(*args)
    first = ledger.iloc[0]
    assert not first.entry_price_cap_rejected
    assert first.filled_quantity > 0
    assert first.filled_quantity % 100 == 0
    assert first.cash >= 0
    assert abs(first.fill_price - first.entry_price_ceiling_after_ex_adjustment) < 1e-9
