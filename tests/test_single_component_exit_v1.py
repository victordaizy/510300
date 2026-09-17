"""验证单成分最优解、周期权重、因果缓存、固定版本和真实进出场。"""
import copy
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import hadamard
from sklearn.cross_decomposition import PLSRegression
from research import single_component_exit_inputs_v1 as module
from research.learned_cycle_exit_v1 import FEATURES, training_rows
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture

CFG = {"feature_clip": 5., "n_components": 1, "orthogonality_tolerance": 1e-10, "recent_cycles": 20, "minimum_cycles": 4, "minimum_rows": 40}


def sample():
    values = np.tile(hadamard(16)[:, 1:9], (4, 1)).astype(float)
    values[:, 3] = 1.
    rows = pd.DataFrame(values, columns=FEATURES)
    rows["cycle_id"] = np.repeat(np.arange(1, 5), 16)
    rows["origin_index"] = np.concatenate([np.arange(1, 17) + offset for offset in [0, 20, 40, 60]])
    rows["exit_index"], rows["sample_weight"] = np.repeat([20, 40, 60, 80], 16), 1 / 16
    rows["target"] = .04 * rows[FEATURES[0]] - .02 * rows[FEATURES[1]] + rows.cycle_id * .1
    return rows


def test_single_component_recovers_known_orthogonal_response_and_constant_factor_is_zero():
    fitted = module.fit_single_component_cycle(sample(), CFG)
    expected = np.zeros(8); expected[:2] = [.04, -.02]
    np.testing.assert_allclose(fitted["coefficients"], expected, atol=1e-12, rtol=0)
    assert all(value != 0 for value in fitted["coefficients"][:2]) and fitted["coefficients"][3] == 0.
    assert fitted["intercept"] == pytest.approx(.25)
    assert abs(fitted["score_residual_covariance"]) < 1e-12


def test_correlated_nonuniform_cycles_match_independent_sklearn_pls1():
    rows = sample()
    rows.loc[rows.cycle_id.eq(1), FEATURES[0]] *= 1.7
    rows[FEATURES[2]] = .8 * rows[FEATURES[0]] + .2 * rows[FEATURES[2]]
    rows = rows[~(rows.cycle_id.eq(4) & rows.origin_index.gt(68))].copy()
    rows["sample_weight"] = 1. / rows.groupby("cycle_id").cycle_id.transform("count")
    fitted = module.fit_single_component_cycle(rows, CFG)
    dx, dy, w, _, _, _ = module.centered_inputs(rows, CFG)
    independent = PLSRegression(n_components=1, scale=False).fit(dx * np.sqrt(w[:, None]), dy * np.sqrt(w))
    np.testing.assert_allclose(fitted["coefficients"], independent.coef_.ravel(), atol=1e-12, rtol=1e-11)
    ols = np.linalg.lstsq(dx * np.sqrt(w[:, None]), dy * np.sqrt(w), rcond=None)[0]
    assert np.max(abs(ols - fitted["coefficients"])) > 1e-3
    assert not fitted["zero_covariance_model"]


def test_whole_cycle_weight_is_invariant_when_one_cycles_observations_are_duplicated():
    rows = sample()
    duplicated = pd.concat([rows[rows.cycle_id.eq(1)].loc[rows[rows.cycle_id.eq(1)].index.repeat(2)], rows[rows.cycle_id.ne(1)]], ignore_index=True)
    duplicated.loc[duplicated.cycle_id.eq(1), "origin_index"] = np.arange(1, 33)
    duplicated.loc[duplicated.cycle_id.eq(1), "exit_index"] = 33
    duplicated["sample_weight"] = 1. / duplicated.groupby("cycle_id").cycle_id.transform("count")
    a, b = module.fit_single_component_cycle(rows, CFG), module.fit_single_component_cycle(duplicated, CFG)
    np.testing.assert_allclose(a["coefficients"], b["coefficients"], atol=1e-12, rtol=0)
    assert a["intercept"] == pytest.approx(b["intercept"], abs=1e-12)
    duplicated.loc[0, "sample_weight"] = 1.
    with pytest.raises(ValueError, match="总权重"):
        module.fit_single_component_cycle(duplicated, CFG)


def test_exact_zero_solution_is_valid_and_missing_data_cannot_be_deleted():
    rows = sample()
    rows["target"] = rows.cycle_id * .125
    fitted = module.fit_single_component_cycle(rows, CFG)
    assert fitted["nonzero_factor_count"] == 0 and fitted["zero_covariance_model"]
    assert module.single_component_prediction(fitted, rows[FEATURES].iloc[0].to_numpy()) == pytest.approx(.3125)
    rows.loc[0, FEATURES[2]] = np.nan
    with pytest.raises(ValueError, match="禁止删行"):
        module.fit_single_component_cycle(rows, CFG)


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
    original, calls = module.fit_single_component_cycle, []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, "fit_single_component_cycle", counted)
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
    monkeypatch.setattr(module, "fit_single_component_cycle", failed)
    full, _, _, counts = module.build_monthly_models(rows, records, CFG)
    assert len(calls) == 1 and counts["failed_fits"] == 1
    assert all(r["model"] is None for r in full)
    assert all(r["status"] == "NO_VIEW_MODEL_FIT_FAILED" for r in full[1:])


def constant(value, t=0, latest=None):
    return {"fit_index": t, "parameter_first_fit_index": t, "latest_exit_index": t if latest is None else latest,
        "status": "FIT_COMPLETE", "model": {"kind": module.KIND, "features": FEATURES, "mean": [0.] * 8, "scale": [1.] * 8,
        "coefficients": [0.] * 8, "feature_clip": 5., "intercept": value}}


def test_entry_version_is_fixed_missing_state_resets_count_and_future_model_is_rejected():
    data = fixture()[0]
    data.loc[2, "mom5"] = np.nan
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    controller = module.SingleComponentExitController(data, [constant(-.01), constant(.05, 2)])
    got = [controller(t, cycle, 100000., 100000.) for t in range(1, 5)]
    assert [r["negative_confirmation_count"] for r in got] == [1, 0, 1, 2]
    assert len({r["fixed_prediction_identity"] for r in got}) == 1
    cycle.update(cycle_id=2, entry_index=5)
    assert controller(5, cycle, 100000., 100000.)["continuation_prediction"] == .05
    with pytest.raises(ValueError, match="未来周期"):
        module.SingleComponentExitController(data, [constant(-.1, 0, 8)])(5, cycle, 100000., 100000.)
    with pytest.raises(ValueError, match="首次持仓收盘"):
        module.SingleComponentExitController(data, [constant(-.1)])(6, cycle, 100000., 100000.)


def test_real_blocked_exit_reentry_dividend_and_price_exit_without_initial_model():
    args = fixture()
    data = args[0]
    data.loc[3:, ["open", "close"]] = 9.9
    data.loc[4:, "previous_close"] = 9.9
    data.loc[3, ["open", "dividend"]] = [8.91, .1]
    args[1] = pd.DataFrame({"record_date": [data.date.iloc[2]], "ex_date": [data.date.iloc[3]],
        "payment_date": [data.date.iloc[5]], "cash_dividend_per_share": [.1]})
    args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, module.SingleComponentExitController(data, [constant(-.01), constant(.03, 3)]))
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
    ledger, decisions, cycles = simulate_rearmed_exit(*args, module.SingleComponentExitController(args[0], [constant(-.5, 2)]))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
