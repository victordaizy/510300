"""核对加权保序曲线、等权预测、固定时钟和真实进出场。"""
import copy
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import isotonic_regression
from research import marginal_monotone_exit_inputs_v1 as module
from research.learned_cycle_exit_v1 import FEATURES, training_rows
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture

CFG = {"feature_clip": 5., "curve_weight": .125, "direction_rule": "SIGN_OF_CYCLE_WEIGHTED_COVARIANCE", "recent_cycles": 20, "minimum_cycles": 4, "minimum_rows": 40}


def sample():
    rng = np.random.default_rng(20260909)
    rows = pd.DataFrame(rng.normal(size=(64, 8)), columns=FEATURES)
    rows["entry_mode"] = 1.
    rows["cycle_id"] = np.repeat(np.arange(1, 5), 16)
    rows["origin_index"] = np.concatenate([np.arange(1, 17) + offset for offset in [0, 20, 40, 60]])
    rows["exit_index"], rows["sample_weight"] = np.repeat([20, 40, 60, 80], 16), 1/16
    rows["target"] = np.where(np.arange(64) % 4 == 0, -.10, .01)
    return rows


def test_weighted_adjacent_blocks_and_exact_ties_have_known_solutions():
    for direction in [-1, 1]:
        curve = module.weighted_pava_curve([0, 1, 2, 3], np.array([3, 1, 2, 4])*direction, [1, 3, 1, 1], direction)
        np.testing.assert_allclose(np.interp([0, 1, 2, 3], curve["x"], curve["y"]), np.array([1.5, 1.5, 2, 4])*direction)
    curve = module.weighted_pava_curve([0, 0, 1, 2], [0, 2, 0, 3], [1, 3, 2, 1], 1)
    np.testing.assert_allclose(np.interp([0, 1, 2], curve["x"], curve["y"]), [1, 1, 3])


def test_weighted_directions_and_all_curves_match_independent_scipy_solution():
    rows = sample().iloc[:-7].copy()
    rows["sample_weight"] = 1. / rows.groupby("cycle_id").cycle_id.transform("count")
    model = module.fit_marginal_monotone(rows, CFG)
    x, y, w = rows[FEATURES].to_numpy(), rows.target.to_numpy(), rows.sample_weight.to_numpy()
    mean = np.average(x, axis=0, weights=w)
    mean[3] = 1.
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    scale[scale <= 1e-12] = 1.
    np.testing.assert_allclose(model["mean"], mean, atol=1e-12)
    np.testing.assert_allclose(model["scale"], scale, atol=1e-12)
    z = np.clip((x-mean)/scale, -5., 5.)
    target_mean = np.average(y, weights=w)
    estimates = []
    for j, curve in enumerate(model["curves"]):
        covariance = np.average((z[:, j]-np.average(z[:, j], weights=w))*(y-target_mean), weights=w)
        assert curve["direction"] == (0 if j == 3 else np.sign(covariance))
        support, inverse = np.unique(z[:, j], return_inverse=True)
        masses = np.bincount(inverse, weights=w)
        means = np.bincount(inverse, weights=w*y)/masses
        expected = np.full(len(support), target_mean) if curve["direction"] == 0 else isotonic_regression(means, weights=masses, increasing=curve["direction"] > 0).x
        np.testing.assert_allclose(np.interp(support, curve["x"], curve["y"]), expected, atol=1e-12, rtol=0)
        estimates.append(np.interp(z[7, j], support, expected))
    got = module.marginal_monotone_prediction(model, x[7])
    assert got["prediction"] == pytest.approx(np.mean(estimates))
    np.testing.assert_allclose([got[f"curve_prediction_{name}"] for name in FEATURES], estimates, atol=1e-12)


def test_cycle_duplication_preserves_total_weight_and_entire_prediction():
    rows = sample()
    duplicated = pd.concat([rows[rows.cycle_id.eq(1)].loc[rows[rows.cycle_id.eq(1)].index.repeat(2)], rows[rows.cycle_id.ne(1)]], ignore_index=True)
    duplicated.loc[duplicated.cycle_id.eq(1), "origin_index"] = np.arange(1, 33)
    duplicated.loc[duplicated.cycle_id.eq(1), "exit_index"] = 33
    duplicated["sample_weight"] = 1. / duplicated.groupby("cycle_id").cycle_id.transform("count")
    a, b = module.fit_marginal_monotone(rows, CFG), module.fit_marginal_monotone(duplicated, CFG)
    for key in ["mean", "scale"]:
        np.testing.assert_allclose(a[key], b[key], atol=1e-12, rtol=0)
    for values in rows[FEATURES].to_numpy():
        assert module.marginal_monotone_prediction(a, values)["prediction"] == pytest.approx(module.marginal_monotone_prediction(b, values)["prediction"], abs=1e-12)
    duplicated.loc[0, "sample_weight"] = 1.
    with pytest.raises(ValueError, match="总权重"):
        module.fit_marginal_monotone(duplicated, CFG)


def test_constant_input_zero_covariance_and_missing_factor_are_distinct():
    rows = sample()
    rows[FEATURES] = .1
    model = module.fit_marginal_monotone(rows, CFG)
    assert model["constant_curves"] == 8 and model["pava_solves"] == 0
    assert module.marginal_monotone_prediction(model, np.full(8, .1))["prediction"] == pytest.approx(-.0175)
    zero = sample(); zero["target"] = .125
    model = module.fit_marginal_monotone(zero, CFG)
    assert all(c["direction"] == 0 for c in model["curves"])
    assert model["curves"][0]["constant_reason"] == "EXACT_ZERO_COVARIANCE"
    assert model["curves"][3]["constant_reason"] == "CONSTANT_INPUT"
    with pytest.raises(ValueError, match="完整八因素"):
        module.marginal_monotone_prediction(model, [np.nan]+[.1]*7)
    rows.loc[0, FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="禁止删行"):
        module.fit_marginal_monotone(rows, CFG)


def test_linear_interpolation_endpoint_extension_equal_weights_and_identity():
    model = constant(.01)["model"]
    for j, curve in enumerate(model["curves"]):
        curve.update(x=[-1., 1.], y=[float(j), float(j+2)], direction=1)
    got = module.marginal_monotone_prediction(model, [-9., -1., 0., 1., 9., 0., 0., 0.])
    expected = [0., 1., 3., 5., 6., 6., 7., 8.]
    np.testing.assert_allclose([got[f"curve_prediction_{name}"] for name in FEATURES], expected)
    assert got["prediction"] == pytest.approx(np.mean(expected))
    changed = copy.deepcopy(model); changed["curves"][0]["y"][0] += .01
    assert module.prediction_identity({"model": model}) != module.prediction_identity({"model": changed})
    changed["curves"][0]["y"][0] = np.nan
    with pytest.raises(ValueError, match="断点数值"):
        module.marginal_monotone_prediction(changed, np.zeros(8))


def monthly_fixture():
    rows = sample()
    original = fit_within_cycle_exit(rows, {"feature_clip": 5., "ridge_alpha": 1.})
    records = []
    for t in [0, 100, 101, 102]:
        chosen, ids = training_rows(rows, t, CFG)
        date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=t)
        records.append({"fit_index": t, "fit_origin": str(date.date()), "fit_time": str(date + pd.Timedelta(hours=15, minutes=5)),
            "training_cycles": ids, "training_cycle_count": len(ids), "training_rows": len(chosen),
            "latest_exit_index": 80 if ids else None, "latest_exit_date": "2020-03-21" if ids else None,
            "status": "FIT_COMPLETE" if ids else "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS", "model": copy.deepcopy(original) if ids else None})
    return rows, records


def test_same_actual_input_is_fit_once_and_future_month_does_not_change_earlier_models(monkeypatch):
    rows, records = monthly_fixture()
    original, calls = module.fit_marginal_monotone, []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, "fit_marginal_monotone", counted)
    full, _, _, counts = module.build_monthly_models(rows, records, CFG)
    assert len(calls) == 1 and counts["new_model_fits"] == 1 and counts["reused_monthly_fits"] == 2
    prefix, _, _, _ = module.build_monthly_models(rows, records[:2], CFG)
    assert full[:2] == prefix and full[0]["model"] is None
    changed = rows.copy(); changed.loc[0, "target"] += .001
    assert module.input_identity(changed, CFG) != module.input_identity(rows, CFG)
    assert [r["parameter_first_fit_index"] for r in full] == [None, 100, 100, 100]


def test_failed_input_is_cached_without_retry_or_original_model_fallback(monkeypatch):
    rows, records = monthly_fixture()
    calls = []
    def failed(*args):
        calls.append(1)
        raise ValueError("合成求解失败")
    monkeypatch.setattr(module, "fit_marginal_monotone", failed)
    full, _, _, counts = module.build_monthly_models(rows, records, CFG)
    assert len(calls) == 1 and counts["failed_fits"] == 1
    assert all(r["model"] is None for r in full)
    assert all(r["status"] == "NO_VIEW_MODEL_FIT_FAILED" for r in full[1:])


def constant(value, t=0, latest=None):
    return {"fit_index": t, "parameter_first_fit_index": t, "latest_exit_index": t if latest is None else latest,
        "status": "FIT_COMPLETE", "model": {"kind": module.KIND, "features": FEATURES, "mean": [0.] * 8,
        "scale": [1.] * 8, "feature_clip": 5., "curve_weights": [.125]*8,
        "curves": [{"feature": name, "x": [0.], "y": [value], "direction": 0} for name in FEATURES]}}


def test_entry_version_is_fixed_missing_state_resets_count_and_future_model_is_rejected():
    data = fixture()[0]
    data.loc[2, "mom5"] = np.nan
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    controller = module.MarginalMonotoneExitController(data, [constant(-.01), constant(.05, 2)])
    got = [controller(t, cycle, 100000., 100000.) for t in range(1, 5)]
    assert [r["negative_confirmation_count"] for r in got] == [1, 0, 1, 2]
    assert len({r["fixed_prediction_identity"] for r in got}) == 1
    cycle.update(cycle_id=2, entry_index=5)
    assert controller(5, cycle, 100000., 100000.)["continuation_prediction"] == pytest.approx(.05)
    with pytest.raises(ValueError, match="未来周期"):
        module.MarginalMonotoneExitController(data, [constant(-.1, 0, 8)])(5, cycle, 100000., 100000.)
    with pytest.raises(ValueError, match="首次持仓收盘"):
        module.MarginalMonotoneExitController(data, [constant(-.1)])(6, cycle, 100000., 100000.)


def test_real_blocked_exit_reentry_dividend_and_price_exit_without_initial_model():
    args = fixture()
    data = args[0]
    data.loc[3:, ["open", "close"]] = 9.9
    data.loc[4:, "previous_close"] = 9.9
    data.loc[3, ["open", "dividend"]] = [8.91, .1]
    args[1] = pd.DataFrame({"record_date": [data.date.iloc[2]], "ex_date": [data.date.iloc[3]],
        "payment_date": [data.date.iloc[5]], "cash_dividend_per_share": [.1]})
    args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, module.MarginalMonotoneExitController(data, [constant(-.01), constant(.03, 3)]))
    assert ledger.iloc[2].status == "UNFILLED_DIRECTIONAL_LIMIT"
    assert ledger.loc[ledger.filled_quantity.lt(0), "date"].tolist() == [data.date.iloc[4], data.date.iloc[9]]
    assert ledger.loc[ledger.filled_quantity.gt(0), "date"].tolist() == [data.date.iloc[1], data.date.iloc[7]]
    by_date = ledger.set_index("date")
    amount = by_date.loc[data.date.iloc[2], "shares"] * .1
    assert by_date.loc[data.date.iloc[3], "dividend_recognized"] == pytest.approx(amount)
    assert by_date.loc[data.date.iloc[5], "dividend_paid"] == pytest.approx(amount)
    assert decisions.loc[decisions.origin_index.eq(3), "requested_quantity"].iloc[0] < 0
    assert ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1]
    args = fixture()
    args[5]["exit"][1][2] = True
    ledger, decisions, cycles = simulate_rearmed_exit(*args, module.MarginalMonotoneExitController(args[0], [constant(-.5, 2)]))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
