"""核对名义单日经济、未来截断、合法动作及真实账户连续进出。"""
import numpy as np
import pandas as pd
import pytest
from research.joint_state_action_inputs_v1 import market_state_frame
from research.joint_entry_exit_inputs_v1 import (PRIMARY, CONTROL, LEGAL, INITIAL, one_step_case, make_cases,
    empirical_model, solve_policy, myopic_policy, JointController)
from research.joint_entry_exit_account_v1 import simulate_joint_policy


def fixture():
    data = pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=12), "open": 10., "close": 10.,
        "previous_close": 10., "dividend": 0., "sma120": .03, "mom5": .01})
    div = pd.DataFrame(columns=["record_date", "ex_date", "payment_date", "cash_dividend_per_share"])
    cfg = {"initial_capital": 200000., "lot": 100, "tick": .001, "limit_fraction": .1,
           "costs": {"BASE": {"commission": .0002, "minimum": 5., "slippage": .0005}},
           "training_days": 8, "minimum_market_observations": 2, "discount": .99,
           "improvement_tolerance": 1e-12, "equation_residual_limit": 1e-10, "maximum_policy_iterations": 50}
    return data, div, market_state_frame(data), cfg


def test_one_day_buy_sell_hold_and_minimum_fee_match_hand_calculation():
    d, div, s, cfg = fixture()
    buy = one_step_case(d, div, s, 0, 0, 1, cfg)
    sell = one_step_case(d, div, s, 0, 1, 0, cfg)
    hold = one_step_case(d, div, s, 0, 1, 1, cfg)
    assert buy["filled_quantity"] == 19900 and buy["fill_price"] == pytest.approx(10.005)
    assert buy["end_nav"] == pytest.approx(200000-19900*.005-19900*10.005*.0002)
    assert sell["filled_quantity"] == -20000 and sell["next_account_mode"] == 0
    assert sell["end_nav"] == pytest.approx(200000-20000*.005-20000*9.995*.0002)
    assert hold["end_nav"] == 200000 and hold["reward"] == 0
    tiny = one_step_case(d, div, s, 0, 0, 1, {**cfg, "initial_capital": 2000.})
    assert tiny["filled_quantity"] == 100 and tiny["commission"] == 5
    assert tiny["end_nav"] == pytest.approx(1994.5)


def test_request_uses_origin_close_and_next_open_can_partially_fill_or_block():
    d, div, s, cfg = fixture()
    d.loc[1, ["open", "close"]] = [10.5, 10.55]
    got = one_step_case(d, div, s, 0, 0, 1, cfg)
    assert got["requested_quantity"] == 19900 and got["filled_quantity"] == 19000
    assert got["fill_price"] == pytest.approx(10.506)
    assert got["end_nav"] == pytest.approx(200000+19000*(10.55-10.506)-19000*10.506*.0002)
    d.loc[1, "open"] = 11.
    blocked_buy = one_step_case(d, div, s, 0, 0, 1, cfg)
    assert blocked_buy["next_account_mode"] == 0 and blocked_buy["filled_quantity"] == 0
    d.loc[1, ["open", "close"]] = [9., 9.2]
    blocked_sell = one_step_case(d, div, s, 0, 1, 0, cfg)
    already_locked = one_step_case(d, div, s, 0, 2, 0, cfg)
    assert blocked_sell["next_account_mode"] == already_locked["next_account_mode"] == 2
    assert blocked_sell["end_nav"] == pytest.approx(184000.)
    with pytest.raises(ValueError, match="不合法"):
        one_step_case(d, div, s, 0, 2, 1, cfg)


def test_dividend_rights_belong_to_origin_holder_and_payment_is_not_extra_profit():
    d, div, s, cfg = fixture()
    d.loc[1, ["open", "close", "dividend"]] = [9.9, 9.9, .1]
    div.loc[0] = [d.date.iloc[0], d.date.iloc[1], d.date.iloc[3], .1]
    held = one_step_case(d, div, s, 0, 1, 1, cfg)
    bought = one_step_case(d, div, s, 0, 0, 1, cfg)
    sold = one_step_case(d, div, s, 0, 1, 0, cfg)
    assert held["dividend_recognized"] == held["dividend_receivable"] == 2000.
    assert held["end_nav"] == 200000. and bought["dividend_recognized"] == 0
    assert sold["dividend_receivable"] == 2000. and sold["shares"] == 0
    div.loc[0, "payment_date"] = d.date.iloc[1]
    paid = one_step_case(d, div, s, 0, 1, 1, cfg)
    assert paid["cash"] == 2000. and paid["dividend_receivable"] == 0.
    assert paid["end_nav"] == held["end_nav"] and paid["dividend_paid"] == 2000.
    div.loc[0, "record_date"] = d.date.iloc[0]-pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="未知"):
        one_step_case(d, div, s, 0, 1, 1, cfg)


def test_market_states_are_causal_and_mature_window_keeps_every_required_day():
    d, div, s, cfg = fixture()
    d.loc[0, ["sma120", "mom5"]] = [0, 0]
    d.loc[1, "mom5"] = np.nan
    baseline = market_state_frame(d)
    d.loc[9:, ["sma120", "mom5"]] = -100
    pd.testing.assert_frame_equal(baseline.iloc[:9], market_state_frame(d).iloc[:9])
    assert baseline.market_state.iloc[0] == 0 and np.isnan(baseline.market_state.iloc[1])
    s["market_state"] = np.arange(len(s)) % 4
    cases = make_cases(d, div, s, [8, 9], cfg)
    a = empirical_model(cases, 8, cfg)
    changed = cases.copy()
    changed.loc[changed.origin_index.eq(8), "reward"] = 1000.
    b = empirical_model(changed, 8, cfg)
    assert a["status"] == "EMPIRICAL_MODEL_COMPLETE" and a["latest_maturity_index"] == 8
    np.testing.assert_allclose(a["reward"], b["reward"], equal_nan=True)
    np.testing.assert_allclose(a["transition"].sum(axis=2)[LEGAL], 1.)
    damaged = cases[cases.origin_index.ne(3)]
    assert empirical_model(damaged, 8, cfg)["status"].startswith("NO_VIEW")


def toy_mdp():
    r, p = np.zeros((12, 2)), np.zeros((12, 2, 12))
    for s in range(12):
        p[s, :, s] = 1.
    r[0, 1] = -.001
    p[0, 1] = 0.
    p[0, 1, 1] = 1.
    r[1, 1] = .002
    p[1, 0] = 0.
    p[1, 0, 0] = 1.
    r[~LEGAL] = 1000000.
    return r, p


def test_joint_policy_pays_small_entry_cost_for_future_reward_while_myopic_waits():
    cfg = fixture()[3]
    r, p = toy_mdp()
    solved = solve_policy(r, p, cfg)
    assert solved["status"] == "POLICY_COMPLETE"
    assert solved["policy"][0] == 1 and myopic_policy(r, cfg)[0] == 0
    assert solved["value"][1] == pytest.approx(.2)
    assert solved["value"][0] == pytest.approx(.197)
    assert all(solved["policy"][s] == 0 and solved["q"][s][1] is None for s in (2, 5, 8, 11))
    assert solved["residual"] <= cfg["equation_residual_limit"]


def test_policy_ties_keep_current_asset_and_iteration_failure_stays_no_view():
    cfg = fixture()[3]
    r, p = toy_mdp()
    failed = solve_policy(r, p, {**cfg, "maximum_policy_iterations": 1})
    assert failed["status"] == "NO_VIEW_POLICY_ITERATION_LIMIT"
    solved = solve_policy(np.zeros((12, 2)), p, cfg)
    assert solved["policy"] == INITIAL.tolist() and solved["linear_solves"] == 1


def test_controller_never_uses_a_future_model_and_keeps_locked_exit_without_inputs():
    d, div, states, cfg = fixture()
    policies = {PRIMARY: INITIAL.tolist(), CONTROL: INITIAL.tolist()}
    model = {"fit_index": 3, "latest_maturity_index": 3, "status": "FIT_COMPLETE", "policies": policies}
    controller = JointController(states, [model], PRIMARY)
    assert controller(2, 0)["policy_action"] is None
    assert controller(3, 1)["policy_action"] == 1
    states.loc[4, "market_state"] = np.nan
    missing = JointController(states, [model], PRIMARY)
    assert missing(4, 1)["policy_action"] is None
    assert missing(4, 2)["policy_action"] == 0


def test_real_account_can_reenter_immediately_and_locked_exit_survives_missing_signal():
    d, div, s, cfg = fixture()
    d.loc[2, "open"] = 9.
    seen = []
    def controller(t, mode):
        seen.append((t, mode))
        action = {0: 1, 1: 0, 3: 1}.get(t)
        return {"policy_action": action, "account_mode": mode}
    ledger, decisions, cycles = simulate_joint_policy(d, div, cfg, cfg["costs"]["BASE"], d.date.iloc[1], controller)
    assert ledger.filled_quantity.tolist()[:4] == [19900, 0, -19900, 19900]
    assert (2, 2) in seen and (3, 0) in seen
    assert len(cycles) == 2 and cycles.holding_intervals.ge(1).all()
    assert ledger.shares.iloc[-1] == 0 and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL"
    assert ledger.accounting_error.abs().max() < 1e-6


def test_real_account_preserves_record_rights_after_exit_and_missing_signal_holds():
    d, div, s, cfg = fixture()
    d.loc[3:, ["open", "close", "previous_close"]] = 9.9
    d.loc[3, ["previous_close", "dividend"]] = [10., .1]
    div.loc[0] = [d.date.iloc[2], d.date.iloc[3], d.date.iloc[5], .1]
    def controller(t, mode):
        return {"policy_action": 1 if t == 0 else (0 if t == 2 else None), "account_mode": mode}
    ledger, decisions, cycles = simulate_joint_policy(d, div, cfg, cfg["costs"]["BASE"], d.date.iloc[1], controller)
    assert ledger.shares.iloc[1] == 19900 and ledger.shares.iloc[2] == 0
    assert ledger.dividend_recognized.sum() == ledger.dividend_paid.sum() == 1990.
    assert ledger.dividend_receivable.iloc[2] == 1990. and ledger.dividend_receivable.iloc[4] == 0.
    assert ledger.accounting_error.abs().max() < 1e-6
