"""联合周期截距胡伯目标、尺度、收敛边界和实际确认规则。"""
import numpy as np
import pandas as pd
import pytest
from research.robust_cycle_exit_inputs_v1 import (FEATURES, weighted_median, objective_gradient,
    original_residual_scale, fit_robust_cycle, RobustCycleExitController)
from research.market_path_exit_inputs_v1 import fit_market_path_exit
from research.learned_cycle_exit_v1 import training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture

CFG = {"feature_clip": 5., "ridge_alpha": 1., "huber_constant": 1.345, "maximum_iterations": 1000,
       "objective_tolerance": 1e-15, "gradient_tolerance": 1e-9, "maximum_line_search_steps": 40, "accepted_gradient_bound": 1e-7}


def states():
    rng = np.random.default_rng(119)
    x = rng.normal(size=(120, 8))
    x[:, 3] = 1.
    rows = pd.DataFrame(x, columns=FEATURES)
    rows["cycle_id"] = np.repeat([1, 2, 3], 40)
    rows["target"] = .01*x[:, 1]-.03*x[:, 2]+np.repeat([-.1, .02, .12], 40)+rng.normal(0, .003, 120)
    rows["sample_weight"] = 1/40
    rows["origin_index"], rows["exit_index"] = np.arange(120), np.repeat([40, 80, 120], 40)
    rows.loc[15, "target"] += .7
    return rows


def stored(value, t=0, latest=None):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE", "model": {
        "kind": "MARKET_PATH_JOINT_CYCLE_HUBER", "features": FEATURES, "mean": [0.]*8, "scale": [1.]*8,
        "coefficients": [0.]*8, "feature_clip": 5., "intercept": value}}


def test_weighted_scale_preserves_independent_cycle_weights_and_outlier_count():
    assert weighted_median([0., 1., 2., 999.], [.4, .1, .4, .1]) == 1.
    rows = states(); initial = fit_market_path_exit(rows, CFG)
    scale = original_residual_scale(rows, initial)
    order = np.argsort(scale["residual"], kind="stable")
    expected = scale["residual"][order[np.searchsorted(np.cumsum(rows.sample_weight.to_numpy()[order]), rows.sample_weight.sum()/2)]]
    assert scale["median"] == expected and scale["scale"] == pytest.approx(scale["mad"]/.6744897501960817)
    assert abs(scale["residual"][15]) > 1.345*scale["scale"]


def test_analytic_gradient_matches_finite_difference_and_large_residual_has_bounded_push():
    z = np.array([[1., 0.], [0., 1.], [1., 1.]])
    y, w, groups = np.array([.1, -.2, 10.]), np.array([.5, .5, 1.]), np.array([0, 0, 1])
    theta = np.array([.02, -.03, .01, -.02])
    _, grad = objective_gradient(theta, z, y, w, groups, .5, 1.)
    finite = []
    for j in range(4):
        step = np.zeros(4); step[j] = 1e-6
        finite.append((objective_gradient(theta+step, z, y, w, groups, .5, 1.)[0]-objective_gradient(theta-step, z, y, w, groups, .5, 1.)[0])/2e-6)
    np.testing.assert_allclose(grad, finite, atol=1e-8, rtol=0)
    larger = y.copy(); larger[2] = 1000.
    np.testing.assert_allclose(objective_gradient(theta, z, larger, w, groups, .5, 1.)[1], grad, atol=1e-12, rtol=0)


def test_joint_fit_reduces_registered_loss_and_satisfies_cycle_intercept_gradients():
    rows = states(); initial = fit_market_path_exit(rows, CFG)
    result = fit_robust_cycle(rows, CFG, initial)
    assert result["final_huber_objective"] < result["initial_huber_objective"]
    assert result["maximum_gradient"] <= 1e-7 and result["limited_influence_rows"] > 0
    assert result["intercept"] == pytest.approx(np.mean([g["cycle_intercept"] for g in result["cycle_intercepts"]]))
    assert abs(result["cycle_intercepts"][0]["cycle_intercept"]+.1) < abs(initial["cycle_intercepts"][0]["cycle_intercept"]+.1)


def test_no_future_cycle_no_zero_scale_and_no_solver_rescue():
    rows = states()
    chosen, ids = training_rows(rows, 80, {"recent_cycles": 20})
    assert ids == [1, 2] and chosen.exit_index.le(80).all()
    initial = fit_market_path_exit(rows, CFG)
    with pytest.raises(RuntimeError, match="收敛条件"):
        fit_robust_cycle(rows, {**CFG, "maximum_iterations": 0}, initial)
    rows["target"] = 0.
    with pytest.raises(RuntimeError, match="尺度不足"):
        fit_robust_cycle(rows, CFG, fit_market_path_exit(rows, CFG))


def test_two_negative_closes_blocked_order_persists_and_reentry_needs_new_condition():
    args = fixture()
    args[0].loc[3, "open"] = 9.
    args[5]["entry"][5] = 0
    controller = RobustCycleExitController(args[0], [stored(-.02), stored(.03, 3)])
    ledger, decisions, cycles = simulate_rearmed_exit(*args, controller)
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    row = decisions[decisions.origin_index.eq(3)].iloc[0]
    assert row.continuation_prediction > 0 and row.requested_quantity < 0
    assert ledger.accounting_error.abs().max() < 1e-7 and ledger.shares.iloc[-1] == 0


def test_missing_state_resets_confirmation_and_future_fit_is_not_used():
    data = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_quantity": 100, "entry_cost_cny": 1005., "mode": 1}
    data.loc[2, "mom5"] = np.nan
    controller = RobustCycleExitController(data, [stored(-.1)])
    results = [controller(t, cycle, 1000., 1005.) for t in [1, 2, 3]]
    assert [r["negative_confirmation_count"] for r in results] == [1, 0, 1]
    assert results[1]["continuation_prediction"] is None
    assert RobustCycleExitController(data, [stored(-.1, 5)])(1, cycle, 1000., 1005.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未来周期"):
        RobustCycleExitController(data, [stored(-.1, latest=5)])(1, cycle, 1000., 1005.)
