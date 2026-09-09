"""验证持仓高点时间、完整参考、成熟训练与实际退出。"""
import numpy as np
import pandas as pd
import pytest
from research.peak_age_exit_inputs_v1 import ADDED, FEATURES, PeakAgeTracker, attach_peak_age, fit_peak_age_exit, PeakAgeExitController
from research.learned_cycle_exit_v1 import training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture


def constant(value=-.01, t=0, latest=None):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE",
            "model": {"kind": "PEAK_AGE_RIDGE", "features": FEATURES.copy(), "mean": [0.]*9, "scale": [1.]*9,
                      "coefficients": [0.]*9, "intercept": value, "feature_clip": 5.}}


def test_peak_age_counts_entry_loss_and_resets_at_equal_or_higher_peak():
    tracker = PeakAgeTracker()
    values = [tracker.observe(1, 4, i, dd)["days_since_peak"] for i, dd in zip(range(4, 10), [-.01, 0., -.01, -.01, 0., -.01])]
    assert values == [1, 0, 1, 2, 0, 1]
    assert tracker.observe(2, 11, 11, -.02)["days_since_peak"] == 1


def test_missing_or_skipped_state_is_unknown_until_peak_is_reached_again():
    t = PeakAgeTracker()
    t.observe(1, 1, 1, 0.)
    assert np.isnan(t.observe(1, 1, 3, -.02)[ADDED])
    assert np.isnan(t.observe(1, 1, 4, -.01)[ADDED])
    assert t.observe(1, 1, 5, 0.)[ADDED] == 0.
    assert np.isnan(t.observe(1, 1, 6, np.nan)[ADDED])
    with pytest.raises(ValueError, match="逐日向前"):
        t.observe(1, 1, 6, -.01)


def test_full_reference_history_includes_rows_absent_from_training_and_no_future():
    dates = pd.bdate_range("2020-01-01", periods=10)
    decisions = pd.DataFrame({"learning_cycle_id": 1, "origin_index": [1, 2, 3, 4, 5], "origin": dates[1:6], "cycle_drawdown": [-.01, 0., -.02, -.02, 0.]})
    cycles = pd.DataFrame({"cycle_id": [1], "entry_index": [1]})
    samples = pd.DataFrame({"cycle_id": [1, 1], "origin_index": [3, 4], "origin": dates[3:5], "exit_index": [8, 8]})
    extended, states = attach_peak_age(samples, decisions, cycles)
    assert extended.days_since_peak.to_list() == [1, 2] and len(states) == 5
    pd.testing.assert_frame_equal(extended[samples.columns], samples)
    future = decisions.copy(); future.loc[4, "cycle_drawdown"] = -.99
    pd.testing.assert_frame_equal(extended, attach_peak_age(samples, future, cycles)[0])
    assert training_rows(extended, 7, {"recent_cycles": 20})[0].empty
    rows, ids = training_rows(extended, 8, {"recent_cycles": 20})
    assert ids == [1] and len(rows) == 2 and rows.sample_weight.eq(.5).all()


def test_weighted_nine_factor_fit_matches_equation_and_missing_is_not_deleted():
    rng = np.random.default_rng(113)
    rows = pd.DataFrame(rng.normal(size=(90, 8)), columns=FEATURES[:-1])
    rows["entry_mode"] = 1.
    rows[ADDED] = np.log1p(rng.integers(0, 20, len(rows)))
    rows["target"] = -.05*rows[ADDED]+.02*rows.mom20
    rows["sample_weight"] = np.r_[np.full(30, 1/30), np.full(60, 1/60)]
    model = fit_peak_age_exit(rows, {"feature_clip": 5., "ridge_alpha": 1.})
    z = np.clip((rows[FEATURES].to_numpy()-model["mean"])/model["scale"], -5, 5)
    a = np.c_[np.ones(len(rows)), z]; w = rows.sample_weight.to_numpy()
    expected = np.linalg.solve(a.T@(w[:, None]*a)+np.diag([0.]+[1.]*9), a.T@(w*rows.target.to_numpy()))
    np.testing.assert_allclose([model["intercept"]]+model["coefficients"], expected, atol=1e-12, rtol=0)
    assert abs(model["coefficients"][-1]) > .001
    rows.loc[0, ADDED] = np.nan
    with pytest.raises(ValueError, match="禁止删行"):
        fit_peak_age_exit(rows, {"feature_clip": 5., "ridge_alpha": 1.})


def test_actual_cycle_uses_own_peak_and_future_model_never_appears_early():
    d = fixture()[0]
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    c = PeakAgeExitController(d, [constant()])
    assert c(1, cycle, 99000., 100000.)["days_since_peak"] == 1
    assert c(2, cycle, 101000., 101000.)["days_since_peak"] == 0
    assert c(3, cycle, 100000., 101000.)["days_since_peak"] == 1
    assert c(4, cycle, 100000., 101000.)["days_since_peak"] == 2
    assert PeakAgeExitController(d, [constant(t=5)])(1, cycle, 100000., 100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未来周期"):
        PeakAgeExitController(d, [constant(latest=5)])(1, cycle, 100000., 100000.)
    cycle.update(cycle_id=2, entry_index=5)
    assert c(5, cycle, 99000., 100000.)["negative_confirmation_count"] == 1


def test_next_open_locked_exit_and_original_exit_with_no_model():
    args = fixture(); args[0].loc[3, "open"] = 9.; args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, PeakAgeExitController(args[0], [constant(), constant(.03, 3)]))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    args = fixture(); args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, PeakAgeExitController(args[0], []))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1]
