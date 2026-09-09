"""检查等待价值的完整动作连接、策略评价、月度可用性和账户衔接。"""
import numpy as np
import pandas as pd
import pytest
from research.entry_wait_policy_inputs_v1 import FEATURES, FILLED, UNFILLED, action_arrays, evaluate_fixed_policy, decision_mask, transition_state, policy_hash, fit_policy, observed_signal_age
from research.entry_wait_policy_account_v1 import simulate_entry_wait_policy
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit


def test_policy_evaluation_follows_fixed_decisions_not_future_perfect_choice():
    reward = np.array([.10, -.20, .30])
    filled = np.array([True, True, True])
    successors = np.array([1, 2, -1])
    targets, values = evaluate_fixed_policy(np.array([False, True, True]), reward, filled, successors)
    np.testing.assert_allclose(targets, [[.10, -.20], [-.20, .30], [.30, 0.]])
    np.testing.assert_allclose(values, [-.20, -.20, .30])
    assert values[0] != .30


def test_unfilled_buy_keeps_later_cash_opportunity_and_unknown_boundary_fails():
    targets, values = evaluate_fixed_policy(np.array([True, True]), np.array([0., .08]), np.array([False, True]), np.array([1, -1]))
    np.testing.assert_allclose(targets, [[.08, .08], [.08, 0.]])
    np.testing.assert_allclose(values, [.08, .08])
    rows = pd.DataFrame({"path_id": [1], "episode_id": [1], "buy_outcome_kind": [FILLED], "buy_terminal_or_immediate_return": [.1], "next_path_id": [pd.NA], "wait_outcome_kind": ["NO_VIEW_UNRESOLVED_WAIT_BOUNDARY"], **{key: [1.] for key in FEATURES}})
    with pytest.raises(ValueError, match="未知等待边界"):
        action_arrays(rows)


def test_same_group_next_day_and_forward_order_are_required():
    rows = pd.DataFrame({"path_id": [1, 2], "episode_id": [1, 2], "buy_outcome_kind": [FILLED, FILLED], "buy_terminal_or_immediate_return": [.1, .2], "next_path_id": [2, pd.NA], "wait_outcome_kind": ["WAIT_TO_NEXT_CLOSE_IN_SAME_GROUP", "WAIT_EPISODE_END_CASH"], **{key: [1., 2.] for key in FEATURES}})
    with pytest.raises(ValueError, match="跨组"):
        action_arrays(rows)


def test_strict_cash_and_wait_comparison_and_policy_cycle_detection():
    np.testing.assert_array_equal(decision_mask([.1, .1, -.1, 0.], [.05, .1, -.2, -.1]), [True, False, False, False])
    a, b = np.array([True, False]), np.array([False, True])
    assert transition_state(a, a, {policy_hash(a)}) == "TRAINING_POLICY_STABLE"
    assert transition_state(a, b, {policy_hash(a), policy_hash(b)}) == "NO_VIEW_REPEATED_TRAINING_POLICY"


def test_policy_fit_stabilizes_on_known_terminal_choices_and_matches_equations():
    rng = np.random.default_rng(96)
    x = rng.normal(size=(20, 10))
    rows = pd.DataFrame(x, columns=FEATURES)
    rows["path_id"], rows["episode_id"] = np.arange(20), np.arange(20)
    rows["buy_outcome_kind"], rows["buy_terminal_or_immediate_return"] = FILLED, .1
    rows["next_path_id"], rows["wait_outcome_kind"], rows["sample_weight"] = pd.NA, "WAIT_EPISODE_END_CASH", 1.
    cfg = {"feature_clip": 5., "ridge_alpha": 1., "maximum_policy_iterations": 50}
    fitted = fit_policy(rows, cfg)
    assert fitted["status"] == "FIT_COMPLETE" and len(fitted["iterations"]) == 2
    np.testing.assert_allclose(fitted["model"]["buy_coefficients_with_intercept"], [.1]+[0.]*10, atol=1e-12)
    np.testing.assert_allclose(fitted["model"]["wait_coefficients_with_intercept"], 0., atol=1e-12)
    limited = fit_policy(rows, {**cfg, "maximum_policy_iterations": 1})
    assert limited["status"] == "NO_VIEW_POLICY_ITERATION_LIMIT" and limited["model"] is None


def test_signal_age_is_past_observation_run_only():
    f = pd.DataFrame({"signal_available": [True, True, True, False, True, True], "raw_entry": [0, 1, 1, 1, 1, 0]})
    full = observed_signal_age(f)
    np.testing.assert_allclose(full, [0., 1., 2., np.nan, 1., 0.], equal_nan=True)
    np.testing.assert_allclose(full[:3], observed_signal_age(f.iloc[:3]), equal_nan=True)


def setup_account():
    dates = pd.bdate_range("2020-01-01", periods=10)
    data = pd.DataFrame({"date": dates, "open": 10., "close": 10., "previous_close": 10., "dividend": 0.})
    div = pd.DataFrame({"record_date": [dates[3]], "ex_date": [dates[4]], "payment_date": [dates[7]], "cash_dividend_per_share": [.1]})
    data.loc[4, "dividend"] = .1
    cfg = {"initial_capital": 100000., "lot": 100, "tick": .001, "limit_fraction": .1}
    cost = {"commission": .0002, "minimum": 5., "slippage": .0005}
    rule = {"entry": np.ones(10, dtype=int), "exit": {1: np.zeros(10, dtype=bool)}}
    rule["exit"][1][5] = True
    spec = {"cooldown": 2, "modes": {1: {"loss": .06, "trail": .08, "take": None, "days": 60}}}
    views = pd.DataFrame({"date": dates, "entry_model_status": "ACTION_VALUES_AVAILABLE", "entry_fit_origin": dates[0], "entry_value_margin": .1, "buy_value": .2, "wait_value": .1, "entry_decision_rule": "WAIT_VALUE"})
    return data, div, cfg, cost, str(dates[1].date()), rule, spec, views


def test_always_entering_margin_preserves_original_account_with_dividends():
    args = list(setup_account())
    new, decisions, _ = simulate_entry_wait_policy(*args)
    old, _, _ = simulate_rearmed_exit(*args[:-1], None)
    pd.testing.assert_frame_equal(new, old)
    assert new.dividend_recognized.sum() > 0 and new.dividend_paid.sum() == new.dividend_recognized.sum()
    assert "旧入场条件" in decisions.iloc[-1].action


def test_waiting_and_no_view_do_not_consume_entry_permission_or_reset_raw_signal():
    args = list(setup_account())
    args[7].loc[0, "entry_value_margin"] = np.nan
    args[7].loc[1:2, "entry_value_margin"] = -.01
    ledger, decisions, _ = simulate_entry_wait_policy(*args)
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].iloc[0] == args[0].date.iloc[4]
    assert pd.isna(decisions.reference_weight.iloc[0])
    assert ledger.filled_quantity.gt(0).sum() == 1
