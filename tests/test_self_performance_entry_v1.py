"""策略自身收益过滤的完整窗口、时钟和进入许可。"""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.self_performance_entry_account_v1 import simulate_self_performance_exit
from research.self_performance_entry_v1 import performance_flags, SelfPerformanceGate


def data_and_reference():
    dates = pd.bdate_range("2020-01-01", periods=9)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.})
    reference = pd.DataFrame({"date": dates, "net_return": [.01, 0., -.02, .03, 0., 0., 0., 0., 0.], "mark_clock": "CLOSE"})
    return data, reference


def test_compounding_includes_flat_days_and_requires_full_window():
    data, reference = data_and_reference()
    flags = performance_flags(data, reference, 3)
    assert not flags.history_available.iloc[:2].any()
    assert np.isclose(flags.reference_year_net_return.iloc[2], 1.01 * .98 - 1)
    assert not flags.positive_history.iloc[2]
    assert flags.reference_year_net_return.iloc[7] == 0 and not flags.positive_history.iloc[7]


def test_missing_calendar_row_is_not_dropped_and_terminal_open_is_not_close():
    data, reference = data_and_reference()
    reference = reference.drop(3)
    reference.loc[8, "mark_clock"] = "OPEN_TERMINAL"
    flags = performance_flags(data, reference, 3)
    assert not flags.history_available.iloc[3:6].any()
    assert flags.history_available.iloc[6] and not flags.history_available.iloc[8]
    gate = SelfPerformanceGate(flags)
    check = gate(3, None, 1, 100)
    assert not check["entry_allowed"] and check["reference_year_net_return"] is None


def test_future_reference_returns_cannot_change_previous_flags():
    data, reference = data_and_reference()
    original = performance_flags(data, reference, 3)
    reference.loc[5:, "net_return"] = 9.
    changed = performance_flags(data, reference, 3)
    assert_frame_equal(original.iloc[:5], changed.iloc[:5])


def account_args():
    data, reference = data_and_reference()
    dividends = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    config = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": 0., "minimum": 0., "slippage": 0.}
    rule = {"entry": np.ones(len(data), dtype=int), "exit": {1: np.zeros(len(data), dtype=bool)}}
    spec = {"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": 2}}}
    return [data, dividends, config, cost, str(data.date.iloc[1].date()), rule, spec]


def test_deferred_entry_preserves_unused_permission_but_does_not_rearm_after_sale():
    args = account_args()
    flags = pd.DataFrame({"history_available": True, "reference_year_net_return": [-.01] + [.01] * 8,
                          "positive_history": [False] + [True] * 8})
    ledger, decisions, cycles = simulate_self_performance_exit(*args, entry_gate=SelfPerformanceGate(flags))
    assert decisions.iloc[0].entry_rearmed and decisions.iloc[0].requested_quantity == 0
    assert ledger[ledger.filled_quantity > 0].date.to_list() == [args[0].date.iloc[2]]
    assert cycles.exit_date.to_list() == [args[0].date.iloc[4]]


def test_all_allowed_flags_recover_the_original_complete_account():
    args = account_args()
    flags = pd.DataFrame({"history_available": [True] * 9, "reference_year_net_return": [.01] * 9, "positive_history": [True] * 9})
    expected = simulate_rearmed_exit(*args)
    actual = simulate_self_performance_exit(*args, entry_gate=SelfPerformanceGate(flags))
    for original, filtered in zip(expected, actual):
        assert_frame_equal(original, filtered[original.columns])
