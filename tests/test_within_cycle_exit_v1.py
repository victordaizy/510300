"""验证周期内估计、成熟隔离和真实账户退出。"""
import numpy as np
import pandas as pd
import pytest
from research.within_cycle_exit_inputs_v1 import FEATURES, fit_within_cycle_exit, within_cycle_prediction, WithinCycleExitController
from research.learned_cycle_exit_v1 import training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture

CFG = {"feature_clip": 5., "ridge_alpha": 1.}


def samples():
    rng = np.random.default_rng(114)
    rows = pd.DataFrame(rng.normal(size=(90, 8)), columns=FEATURES)
    rows["entry_mode"] = 1.
    rows["cycle_id"] = np.r_[np.full(20, 1), np.full(30, 2), np.full(40, 3)]
    rows["sample_weight"] = rows.cycle_id.map({1: 1/20, 2: 1/30, 3: 1/40})
    rows["target"] = .03*rows.mom20-.01*rows.cycle_drawdown+rows.cycle_id.map({1: .2, 2: -.1, 3: .05})
    return rows


def constant(value=-.01, t=0, latest=None):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE",
            "model": {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES.copy(), "mean": [0.]*8, "scale": [1.]*8,
                      "coefficients": [0.]*8, "intercept": value, "feature_clip": 5., "cycle_intercepts": [{"cycle_id": 999, "cycle_intercept": 100.}]}}


def test_within_fit_equals_joint_unpenalized_cycle_intercepts_ridge_equation():
    rows = samples(); m = fit_within_cycle_exit(rows, CFG)
    z = np.clip((rows[FEATURES].to_numpy()-m["mean"])/m["scale"], -5, 5)
    d = pd.get_dummies(rows.cycle_id, dtype=float).to_numpy()
    a = np.c_[z, d]; w = rows.sample_weight.to_numpy()
    b = np.linalg.solve(a.T@(w[:, None]*a)+np.diag([1.]*8+[0.]*3), a.T@(w*rows.target.to_numpy()))
    np.testing.assert_allclose(m["coefficients"], b[:8], atol=1e-12, rtol=0)
    np.testing.assert_allclose([x["cycle_intercept"] for x in m["cycle_intercepts"]], b[8:], atol=1e-12, rtol=0)
    assert m["intercept"] == pytest.approx(b[8:].mean())


def test_arbitrary_cycle_target_levels_do_not_change_within_slopes():
    rows = samples(); old = fit_within_cycle_exit(rows, CFG)
    rows["target"] += rows.cycle_id.map({1: .5, 2: -.3, 3: -.2})
    changed = fit_within_cycle_exit(rows, CFG)
    np.testing.assert_allclose(old["coefficients"], changed["coefficients"], atol=1e-12, rtol=0)
    assert old["intercept"] == pytest.approx(changed["intercept"], abs=1e-12)
    rows["target"] = rows.cycle_id.map({1: .2, 2: -.1, 3: .05})
    np.testing.assert_allclose(fit_within_cycle_exit(rows, CFG)["coefficients"], 0., atol=1e-12, rtol=0)


def test_repeating_a_complete_cycle_with_split_weight_does_not_change_model():
    rows = samples(); old = fit_within_cycle_exit(rows, CFG)
    first = rows[rows.cycle_id.eq(1)].copy(); first.sample_weight /= 2
    duplicate = pd.concat([first, first, rows[rows.cycle_id.ne(1)]], ignore_index=True)
    changed = fit_within_cycle_exit(duplicate, CFG)
    for key in ["mean", "scale", "coefficients", "intercept"]:
        np.testing.assert_allclose(old[key], changed[key], atol=1e-12, rtol=0)
    rows.loc[0, "mom5"] = np.nan
    with pytest.raises(ValueError, match="禁止删行"):
        fit_within_cycle_exit(rows, CFG)


def test_maturity_keeps_complete_cycles_and_future_targets_cannot_enter_training():
    r = samples()
    r["origin_index"] = np.arange(len(r)); r["exit_index"] = r.cycle_id.map({1: 25, 2: 55, 3: 95})
    old, ids = training_rows(r, 55, {"recent_cycles": 20})
    assert ids == [1, 2] and len(old) == 50
    m = fit_within_cycle_exit(old, CFG)
    r.loc[r.cycle_id.eq(3), ["target", "mom5"]] = 1000.
    newer, _ = training_rows(r, 55, {"recent_cycles": 20})
    assert fit_within_cycle_exit(newer, CFG) == m


def test_new_cycle_uses_saved_average_intercept_and_unknown_resets_confirmation():
    d = fixture()[0]; cycle = {"cycle_id": 999, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    assert within_cycle_prediction(constant()["model"], np.zeros(8)) == -.01
    c = WithinCycleExitController(d, [constant()])
    assert c(1, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
    d.loc[2, "mom5"] = np.nan
    missing = c(2, cycle, 100000., 100000.)
    assert missing["continuation_prediction"] is None and missing["negative_confirmation_count"] == 0
    assert WithinCycleExitController(d, [constant(t=5)])(1, cycle, 100000., 100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未来周期"):
        WithinCycleExitController(d, [constant(latest=5)])(1, cycle, 100000., 100000.)
    cycle.update(cycle_id=1000, entry_index=3)
    assert c(3, cycle, 100000., 100000.)["negative_confirmation_count"] == 1


def test_actual_next_open_exit_stays_locked_and_no_model_keeps_price_exit():
    args = fixture(); args[0].loc[3, "open"] = 9.; args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, WithinCycleExitController(args[0], [constant(), constant(.03, 3)]))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    args = fixture(); args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, WithinCycleExitController(args[0], []))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1]
