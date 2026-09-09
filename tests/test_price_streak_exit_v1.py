"""核对含分红连续下跌以及退出后的原始进入许可。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from research.price_streak_exit_v1 import price_flags, PriceEntryGate
from research.price_streak_exit_account_v1 import simulate_price_streak_exit
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit


def setup():
    dates = pd.bdate_range("2020-01-01", periods=10)
    frame = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0., "wealth": 10.})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    rule = {"entry": np.ones(10, dtype=int), "exit": {1: np.zeros(10, dtype=bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}}
    return [frame, div, cfg, cost, str(dates[2].date()), rule, spec]


def test_equal_total_return_and_missing_history_do_not_create_two_down_signal():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=7), "wealth": [10., 9., 8., 8., 7., np.nan, 6.],
                         "close": [10., 9., 8., 7.8, 7., 6.5, 6.]})
    flags = price_flags(data)
    assert flags.two_day_down.to_list() == [False, False, True, False, False, False, False]
    assert not flags.prices_available.iloc[6]


def test_price_recovery_after_exit_does_not_rearm_continuous_old_entry_signal():
    args = setup()
    args[0].loc[2:3, "wealth"] = [9.9, 9.8]
    args[0]["close"] = args[0].wealth
    args[0]["open"] = args[0].wealth
    args[0]["previous_close"] = args[0].close.shift(1).fillna(10.)
    flags = price_flags(args[0])
    args[5]["exit"][1] |= flags.two_day_down.to_numpy(bool)
    ledger, decisions, cycles = simulate_price_streak_exit(*args, entry_gate=PriceEntryGate(flags))
    assert ledger[ledger.filled_quantity > 0].date.to_list() == [args[0].date.iloc[3]]
    assert ledger[ledger.filled_quantity < 0].date.to_list() == [args[0].date.iloc[4]]
    assert not decisions.iloc[-1].entry_rearmed


def test_known_two_down_exit_condition_defers_new_entry_until_recovery():
    args = setup()
    args[0]["wealth"] = [10., 9.9, 9.8, 9.9, 10., 10., 10., 10., 10., 10.]
    args[4] = str(args[0].date.iloc[3].date())
    flags = price_flags(args[0])
    args[5]["exit"][1] |= flags.two_day_down.to_numpy(bool)
    ledger, decisions, cycles = simulate_price_streak_exit(*args, entry_gate=PriceEntryGate(flags))
    assert decisions.iloc[0].entry_deferred_for_two_down
    assert ledger[ledger.filled_quantity > 0].date.iloc[0] == args[0].date.iloc[4]


def test_no_new_price_exit_recovers_original_nonlearning_account():
    args = setup()
    args[4] = str(args[0].date.iloc[3].date())
    args[5]["exit"][1][5] = True
    original, original_decisions, original_cycles = simulate_rearmed_exit(*args)
    ledger, decisions, cycles = simulate_price_streak_exit(*args, entry_gate=PriceEntryGate(price_flags(args[0])))
    assert_frame_equal(original, ledger)
    assert_frame_equal(original_cycles, cycles)
