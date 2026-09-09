"""验证九因子周期内方程、当前强弱输入、成熟时钟及实际退出。"""
import numpy as np
import pandas as pd
import pytest
from research.session_strength_within_inputs_v1 import FEATURES, fit_session_strength_within, session_strength_prediction, SessionStrengthWithinController
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit
from research.learned_cycle_exit_v1 import training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_within_cycle_exit_v1 import samples as eight_samples
from tests.test_median_continuation_v1 import fixture

CFG = {"feature_clip": 5., "ridge_alpha": 1.}


def constant(value=-.01, t=0, latest=None, strength_coefficient=0.):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE",
            "model": {"kind": "SESSION_STRENGTH_WITHIN_CYCLE_RIDGE", "features": FEATURES.copy(), "mean": [0.]*9, "scale": [1.]*9,
                      "coefficients": [0.]*8+[strength_coefficient], "intercept": value, "feature_clip": 5.,
                      "cycle_intercepts": [{"cycle_id": 999, "cycle_intercept": 100.}]}}


def factors(data, value=0.):
    return pd.DataFrame({"date": data.date, "d60_factor": value})


def test_nine_factor_fit_equals_joint_unpenalized_cycle_intercept_equation():
    rows = eight_samples(); rows[FEATURES[-1]] = np.random.default_rng(127).uniform(-1, 1, len(rows))
    rows.target += .04*rows[FEATURES[-1]]
    m = fit_session_strength_within(rows, CFG)
    z = np.clip((rows[FEATURES].to_numpy()-m["mean"])/m["scale"], -5, 5)
    design = np.c_[z, pd.get_dummies(rows.cycle_id, dtype=float).to_numpy()]
    w = rows.sample_weight.to_numpy()
    b = np.linalg.solve(design.T@(w[:, None]*design)+np.diag([1.]*9+[0.]*3), design.T@(w*rows.target.to_numpy()))
    np.testing.assert_allclose(m["coefficients"], b[:9], atol=1e-12, rtol=0)
    np.testing.assert_allclose([g["cycle_intercept"] for g in m["cycle_intercepts"]], b[9:], atol=1e-12, rtol=0)
    assert m["intercept"] == pytest.approx(b[9:].mean()) and abs(m["coefficients"][-1]) > 1e-4
    x = rows[FEATURES].iloc[0].to_numpy()
    assert session_strength_prediction(m, x) == pytest.approx(b[9:].mean()+z[0]@b[:9])


def test_within_constant_strength_adds_no_slope_and_future_cycles_or_length_do_not_reweight():
    rows = eight_samples(); old = fit_within_cycle_exit(rows, CFG)
    rows[FEATURES[-1]] = rows.cycle_id.map({1: .1, 2: -.5, 3: .2})
    m = fit_session_strength_within(rows, CFG)
    np.testing.assert_allclose(m["coefficients"][:8], old["coefficients"], atol=1e-12, rtol=0)
    assert abs(m["coefficients"][-1]) < 1e-12
    first = rows[rows.cycle_id.eq(1)].copy(); first.sample_weight /= 2
    repeated = pd.concat([first, first, rows[rows.cycle_id.ne(1)]], ignore_index=True)
    for key in ["coefficients", "mean", "scale", "intercept"]:
        np.testing.assert_allclose(m[key], fit_session_strength_within(repeated, CFG)[key], atol=1e-12, rtol=0)
    rows["origin_index"] = np.arange(len(rows)); rows["exit_index"] = rows.cycle_id.map({1: 25, 2: 55, 3: 95})
    mature, ids = training_rows(rows, 55, {"recent_cycles": 20}); before = fit_session_strength_within(mature, CFG)
    assert ids == [1, 2]
    rows.loc[rows.cycle_id.eq(3), [FEATURES[-1], "target"]] = 1000.
    assert fit_session_strength_within(training_rows(rows, 55, {"recent_cycles": 20})[0], CFG) == before
    mature.loc[mature.index[0], FEATURES[-1]] = np.nan
    with pytest.raises(ValueError, match="禁止删行"):
        fit_session_strength_within(mature, CFG)


def test_actual_current_close_factor_missing_confirmation_and_future_model_clock():
    data = fixture()[0]; f = factors(data); f.loc[1, FEATURES[-1]] = -.2; f.loc[2, FEATURES[-1]] = np.nan
    cycle = {"cycle_id": 999, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    c = SessionStrengthWithinController(data, [constant(value=0., strength_coefficient=1.)], f)
    row = c(1, cycle, 100000., 100000.)
    assert row["continuation_prediction"] == -.2 and row["negative_confirmation_count"] == 1
    row = c(2, cycle, 100000., 100000.)
    assert row["learning_status"] == "NO_VIEW_INCOMPLETE_NINE_FEATURES" and row["continuation_prediction"] is None and row["negative_confirmation_count"] == 0
    assert SessionStrengthWithinController(data, [constant(t=5)], f)(1, cycle, 100000., 100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未来周期"):
        SessionStrengthWithinController(data, [constant(latest=5)], f)(1, cycle, 100000., 100000.)
    c = SessionStrengthWithinController(data, [constant()], factors(data))
    assert c(1, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
    cycle.update(cycle_id=1000, entry_index=3)
    assert c(3, cycle, 100000., 100000.)["negative_confirmation_count"] == 1


def test_actual_next_open_exit_lock_and_original_exit_without_model():
    args = fixture(); args[0].loc[3, "open"] = 9.; args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, SessionStrengthWithinController(args[0], [constant(), constant(.03, 3)], factors(args[0])))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    args = fixture(); args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, SessionStrengthWithinController(args[0], [], factors(args[0])))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1]


def test_current_continuous_strength_is_not_just_an_entry_flag_and_future_is_unread():
    data = fixture()[0]
    f = factors(data, value=2.)
    f.loc[2, FEATURES[-1]] = 3.
    future = f.copy()
    future.loc[4:, FEATURES[-1]] = -100.
    first = SessionStrengthWithinController(data, [constant(value=0., strength_coefficient=.01)], f)
    other = SessionStrengthWithinController(data, [constant(value=0., strength_coefficient=.01)], future)
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    for t, expected in [(1, .02), (2, .03), (3, .02)]:
        a = first(t, cycle, 100000., 100000.)
        b = other(t, cycle, 100000., 100000.)
        assert a == b and a["continuation_prediction"] == pytest.approx(expected)


def test_factor_date_alignment_is_required_even_when_values_are_complete():
    data = fixture()[0]
    f = factors(data)
    f.loc[2, "date"] += pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="日历不同"):
        SessionStrengthWithinController(data, [constant()], f)
