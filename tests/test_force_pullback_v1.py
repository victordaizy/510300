"""合成数据验证力度计算、缺失语义和真实账户时点。"""
import numpy as np
import pandas as pd
from research.force_pullback_inputs_v1 import economic_force, force_factors, ForcePullbackController
from research.joint_entry_exit_account_v1 import simulate_joint_policy

CFG = {"fast_period": 2, "slow_period": 13}


def frame(changes):
    prices = np.r_[100., 100.+np.cumsum(changes)]
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=len(prices)), "close": prices,
        "previous_close": np.r_[np.nan, prices[:-1]], "dividend": 0., "volume": 10.})


def test_decimal_dividend_neutrality_and_actual_share_volume_units():
    assert economic_force(9.9, 10., .1, 100000000.) == (0., 0.)
    assert economic_force(9.95, 10., .1, 100.) == (.05, 5.)
    assert economic_force(9.9, 10., 0., 100.) == (-.1, -10.)


def test_fast_mean_uses_two_value_seed_then_fixed_recursion():
    factors, _ = force_factors(frame([1., 2., -1., 0.]), CFG)
    assert factors.fast_force.iloc[:2].isna().all()
    assert factors.fast_force.iloc[2] == 15.
    assert np.isclose(factors.fast_force.iloc[3], -5/3)
    assert np.isclose(factors.fast_force.iloc[4], -5/9)


def test_slow_mean_needs_thirteen_observations_without_future_seed():
    data = frame(np.arange(1., 15.)); data.volume = 1.
    factors, summary = force_factors(data, CFG)
    assert factors.slow_force.iloc[:13].isna().all()
    assert factors.slow_force.iloc[13] == 7. and factors.slow_force.iloc[14] == 8.
    assert summary["available_indicator_days"] == 2 and summary["observations"] == 14
    assert ForcePullbackController(factors)(12, 0)["policy_action"] is None


def test_future_changes_and_shorter_prefix_do_not_change_past_indicator():
    data = frame(np.tile([1., -2., 3., -1.], 15))
    original, _ = force_factors(data, CFG)
    changed = data.copy(); changed.loc[30:, "volume"] *= 1000
    updated, _ = force_factors(changed, CFG)
    prefix, _ = force_factors(data.iloc[:30], CFG)
    pd.testing.assert_frame_equal(original.iloc[:30], updated.iloc[:30])
    pd.testing.assert_frame_equal(original.iloc[:30], prefix)


def test_positive_volume_unit_scaling_preserves_all_decisions():
    data = frame(np.tile([1., -2., 3., -1.], 12))
    original, _ = force_factors(data, CFG)
    data.volume *= 1000
    scaled, _ = force_factors(data, CFG)
    np.testing.assert_allclose(scaled[["raw_force", "fast_force", "slow_force"]], original[["raw_force", "fast_force", "slow_force"]]*1000,
        atol=1e-9, rtol=1e-12, equal_nan=True)
    a, b = ForcePullbackController(original), ForcePullbackController(scaled)
    for t in range(len(data)):
        for mode in [0, 1, 2]:
            assert a(t, mode)["policy_action"] == b(t, mode)["policy_action"]


def test_zero_observed_volume_is_valid_and_internal_gap_stops_remaining_averages():
    data = frame(np.tile([1., -1.], 20)); data.loc[15, "volume"] = 0.
    factors, _ = force_factors(data, CFG)
    assert factors.raw_force.iloc[15] == 0. and factors.factor_status.iloc[15] == "INDICATOR_AVAILABLE"
    data.loc[20, "volume"] = np.nan
    missing, summary = force_factors(data, CFG)
    assert len(missing) == len(data) and missing.factor_status.iloc[20:].eq("NO_VIEW_SOURCE_GAP_REMAINDER").all()
    assert missing[["fast_force", "slow_force"]].iloc[20:].isna().all().all() and summary["observations"] == 19
    zero, _ = force_factors(frame(np.zeros(20)), CFG)
    assert zero.slow_force.iloc[13:].eq(0).all() and ForcePullbackController(zero)(14, 0)["policy_action"] == 0


def test_entry_exit_zero_missing_and_locked_exit_are_separate_states():
    factors = pd.DataFrame({"fast_force": [-4., -2., -1., 0., 2., 1., -1., -.5, np.nan],
        "slow_force": [1., 1., 1., 1., 1., 1., 1., 0., 1.], "factor_status": "INDICATOR_AVAILABLE"})
    c = ForcePullbackController(factors)
    assert c(0, 0)["policy_action"] is None
    assert c(2, 0)["policy_action"] == 1 and c(2, 1)["policy_action"] == 1
    assert c(3, 0)["policy_action"] == 0 and c(3, 1)["policy_action"] == 1
    assert c(4, 1)["policy_action"] == 1 and c(5, 1)["policy_action"] == 0
    assert c(7, 1)["policy_action"] == 0 and c(7, 0)["policy_action"] == 0
    assert c(8, 1)["policy_action"] is None and c(8, 2)["policy_action"] == 0


def test_real_account_next_open_dividend_exit_lock_reentry_and_terminal():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=10), "open": 10., "close": 10., "previous_close": 10., "dividend": 0.})
    data.loc[4, "open"] = 9.
    data.loc[3, "dividend"] = .1
    div = pd.DataFrame({"record_date": [data.date.iloc[2]], "ex_date": [data.date.iloc[3]], "payment_date": [data.date.iloc[4]], "cash_dividend_per_share": [.1]})
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    factors = pd.DataFrame({"fast_force": [-4., -2., 3., 2., np.nan, -3., -2., 2., 3., 4.], "slow_force": 1., "factor_status": "INDICATOR_AVAILABLE"})
    ledger, decisions, cycles = simulate_joint_policy(data, div, cfg, cost, data.date.iloc[1], ForcePullbackController(factors))
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [data.date.iloc[2], data.date.iloc[7]]
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [data.date.iloc[5], data.date.iloc[9]]
    assert decisions[decisions.origin_index.eq(4)].iloc[0].decision_status == "LOCKED_EXIT_CONTINUES"
    assert ledger[ledger.date.eq(data.date.iloc[4])].iloc[0].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert np.isclose(ledger.dividend_recognized.sum(), ledger[ledger.date.eq(data.date.iloc[2])].shares.iloc[0]*.1)
    assert len(cycles) == 2 and cycles.holding_intervals.ge(1).all()
    assert ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1]
