"""检验负隔夜风险定义、完整窗口、原权重拟合与真实退出。"""
import numpy as np
import pandas as pd
import pytest
from research.overnight_downside_exit_inputs_v1 import ADDED, FEATURES, downside_frame, attach_downside, fit_downside_exit, downside_prediction, OvernightDownsideExitController
from research.learned_cycle_exit_v1 import training_rows
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture


def source(n=60):
    return pd.DataFrame({"date": pd.bdate_range("2020-01-01", periods=n), "overnight_log": -.01, "intraday_log": .02})


def constant(value=-.01, t=0, latest=None):
    return {"fit_index": t, "latest_exit_index": t if latest is None else latest, "status": "FIT_COMPLETE",
            "model": {"kind": "OVERNIGHT_DOWNSIDE_SHARE_RIDGE", "features": FEATURES.copy(), "mean": [0.]*9, "scale": [1.]*9,
                      "coefficients": [0.]*9, "intercept": value, "feature_clip": 5.}}


def known_factors(data):
    return pd.DataFrame({"date": data.date, ADDED: .2, "factor_status": "OVERNIGHT_DOWNSIDE_SHARE_AVAILABLE"})


def test_ratio_is_uncentered_negative_overnight_only_and_known_zero_is_valid():
    d = source(3)
    d.overnight_log, d.intraday_log = [-.01, .02, -.02], [.03, -.01, 0.]
    f = downside_frame(d, 3)
    assert f[ADDED].iloc[:2].isna().all()
    assert f[ADDED].iloc[2] == pytest.approx(.0005/.0019)
    d.overnight_log = .01
    assert downside_frame(d, 3)[ADDED].iloc[2] == 0.


def test_dividend_neutral_return_zero_denominator_missing_window_and_future_prefix():
    d = source()
    d.overnight_log = np.log((9.9+.1)/10.)
    assert downside_frame(d)[ADDED].iloc[19:].eq(0.).all()
    d.intraday_log = 0.
    assert downside_frame(d)[ADDED].isna().all()
    d = source()
    original = downside_frame(d)
    pd.testing.assert_frame_equal(original.iloc[:35], downside_frame(d.iloc[:35]))
    d.loc[35:, "overnight_log"] = -100.
    pd.testing.assert_frame_equal(original.iloc[:35], downside_frame(d).iloc[:35])
    d = source()
    d.loc[30, "intraday_log"] = np.nan
    f = downside_frame(d)
    assert f[ADDED].iloc[30:50].isna().all() and f[ADDED].iloc[50] == pytest.approx(.2)


def test_same_date_reference_input_whole_maturity_and_actual_input_match():
    d = source()
    factors = downside_frame(d)
    samples = pd.DataFrame({"cycle_id": [1, 1, 2], "origin_index": [21, 22, 30], "origin": d.date.iloc[[21, 22, 30]].to_list(), "exit_index": [25, 25, 40]})
    extended = attach_downside(samples, factors)
    pd.testing.assert_frame_equal(samples, extended.drop(columns=ADDED))
    selected, ids = training_rows(extended, 25, {"recent_cycles": 20})
    assert ids == [1] and len(selected) == 2
    np.testing.assert_allclose(selected[ADDED], .2)
    np.testing.assert_allclose(selected.sample_weight, [.5, .5])
    bad = samples.copy(); bad.loc[0, "origin"] = d.date.iloc[20]
    with pytest.raises(ValueError, match="日期不同"):
        attach_downside(bad, factors)


def test_weighted_nine_factor_fit_matches_ridge_equation_and_keeps_missing_rows():
    rng = np.random.default_rng(107)
    rows = pd.DataFrame(rng.normal(size=(90, 8)), columns=FEATURES[:-1])
    rows["entry_mode"] = 1.
    rows[ADDED] = rng.uniform(0, 1, len(rows))
    rows["target"] = -.05*rows[ADDED]+.02*rows.mom20
    rows["sample_weight"] = np.r_[np.full(30, 1/30), np.full(60, 1/60)]
    model = fit_downside_exit(rows, {"feature_clip": 5., "ridge_alpha": 1.})
    z = np.clip((rows[FEATURES].to_numpy()-model["mean"])/model["scale"], -5, 5)
    a = np.c_[np.ones(len(rows)), z]; w = rows.sample_weight.to_numpy()
    expected = np.linalg.solve(a.T@(w[:, None]*a)+np.diag([0.]+[1.]*9), a.T@(w*rows.target.to_numpy()))
    np.testing.assert_allclose([model["intercept"]]+model["coefficients"], expected, atol=1e-12, rtol=0)
    assert abs(model["coefficients"][-1]) > .001
    rows.loc[0, ADDED] = np.nan
    with pytest.raises(ValueError, match="禁止删行"):
        fit_downside_exit(rows, {"feature_clip": 5., "ridge_alpha": 1.})


def test_unknown_required_factor_future_model_and_cycle_reset():
    d = fixture()[0]; factors = known_factors(d)
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    factors.loc[2, ADDED] = np.nan
    controller = OvernightDownsideExitController(d, [constant()], factors=factors)
    assert controller(1, cycle, 100000., 100000.)["negative_confirmation_count"] == 1
    missing = controller(2, cycle, 100000., 100000.)
    assert missing["continuation_prediction"] is None and missing["negative_confirmation_count"] == 0
    assert OvernightDownsideExitController(d, [constant(t=5)], factors=factors)(3, cycle, 100000., 100000.)["continuation_prediction"] is None
    with pytest.raises(ValueError, match="未来周期"):
        OvernightDownsideExitController(d, [constant(latest=5)], factors=factors)(3, cycle, 100000., 100000.)
    controller(3, cycle, 100000., 100000.)
    assert controller(4, cycle, 100000., 100000.)["negative_confirmation_count"] == 2
    cycle.update(cycle_id=2, entry_index=5)
    assert controller(5, cycle, 100000., 100000.)["negative_confirmation_count"] == 1


def test_next_open_locked_exit_survives_positive_model_and_original_exit_survives_unknown():
    args = fixture(); factors = known_factors(args[0])
    args[0].loc[3, "open"] = 9.; args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, OvernightDownsideExitController(args[0], [constant(), constant(.03, 3)], factors=factors))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [args[0].date.iloc[4], args[0].date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [args[0].date.iloc[1], args[0].date.iloc[7]]
    assert decisions[decisions.origin_index.eq(3)].iloc[0].requested_quantity < 0
    args = fixture(); factors = known_factors(args[0]); factors[ADDED] = np.nan
    args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, OvernightDownsideExitController(args[0], [constant()], factors=factors))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].iloc[0] == args[0].date.iloc[3]
