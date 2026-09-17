"""核对加权两类矩、概率与盈亏幅度、固定时钟和真实进出场。"""
import copy
import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax
from scipy.stats import norm
from research import probability_payoff_exit_inputs_v1 as module
from research.learned_cycle_exit_v1 import FEATURES, training_rows
from research.within_cycle_exit_inputs_v1 import fit_within_cycle_exit
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from tests.test_median_continuation_v1 import fixture

CFG = {"feature_clip": 5., "variance_smoothing": 1e-9, "recent_cycles": 20, "minimum_cycles": 4, "minimum_rows": 40}


def sample():
    rng = np.random.default_rng(20260909)
    rows = pd.DataFrame(rng.normal(size=(64, 8)), columns=FEATURES)
    rows["entry_mode"] = 1.
    rows["cycle_id"] = np.repeat(np.arange(1, 5), 16)
    rows["origin_index"] = np.concatenate([np.arange(1, 17) + offset for offset in [0, 20, 40, 60]])
    rows["exit_index"], rows["sample_weight"] = np.repeat([20, 40, 60, 80], 16), 1/16
    rows["target"] = np.where(np.arange(64) % 4 == 0, -.10, .01)
    return rows


def test_weighted_class_moments_probabilities_and_payoffs_match_independent_density():
    rows = sample().iloc[:-7].copy()
    rows["sample_weight"] = 1. / rows.groupby("cycle_id").cycle_id.transform("count")
    model = module.fit_probability_payoff(rows, CFG)
    x, w = rows[FEATURES].to_numpy(), rows.sample_weight.to_numpy()
    mean = (x * w[:, None]).sum(axis=0) / w.sum()
    scale = np.sqrt(((x-mean)**2*w[:, None]).sum(axis=0)/w.sum())
    scale[scale <= 1e-12] = 1.
    z = np.clip((x-mean)/scale, -5., 5.)
    z[:, 3] = 0.
    global_mean = (z*w[:, None]).sum(axis=0)/w.sum()
    variance = ((z-global_mean)**2*w[:, None]).sum(axis=0)/w.sum()
    epsilon = variance.max()*1e-9
    active = np.array(model["active_features"])
    assert active.sum() == 7 and not active[3]
    density = []
    test_z = z[7]
    for label in [0, 1]:
        mask = (rows.target.to_numpy() > 0).astype(int) == label
        class_w = w[mask]
        cm = (z[mask]*class_w[:, None]).sum(axis=0)/class_w.sum()
        cv = ((z[mask]-cm)**2*class_w[:, None]).sum(axis=0)/class_w.sum()
        np.testing.assert_allclose(model["class_means"][label], cm, atol=1e-12, rtol=0)
        np.testing.assert_allclose(model["class_raw_variances"][label], cv, atol=1e-12, rtol=0)
        prior = class_w.sum()/w.sum()
        assert model["class_priors"][label] == pytest.approx(prior)
        density.append(np.log(prior)+norm.logpdf(test_z[active], loc=cm[active], scale=np.sqrt(cv[active]+epsilon)).sum())
    probabilities = softmax(density)
    got = module.probability_payoff_prediction(model, x[7])
    np.testing.assert_allclose([got["probability_disadvantage"], got["probability_advantage"]], probabilities, atol=1e-12, rtol=0)
    assert got["prediction"] == pytest.approx(probabilities @ [-.10, .01])
    assert got["prediction"] == pytest.approx(got["expected_gain_contribution"]+got["expected_loss_contribution"])


def test_all_constant_factors_use_priors_and_high_win_probability_can_still_lose():
    rows = sample()
    rows[FEATURES] = .1
    fitted = module.fit_probability_payoff(rows, CFG)
    assert fitted["all_constant_model"] and fitted["active_factor_count"] == 0 and fitted["variance_epsilon"] == 0.
    got = module.probability_payoff_prediction(fitted, np.full(8, .1))
    assert got["probability_advantage"] == pytest.approx(.75)
    assert got["prediction"] == pytest.approx(-.0175) and got["prediction"] < 0.
    changed = copy.deepcopy(fitted); changed["class_payoffs"][0] = -.02
    assert module.prediction_identity({"model": fitted}) != module.prediction_identity({"model": changed})


def test_class_constant_variance_is_stabilized_and_zero_label_is_kept_nonpositive():
    rows = sample()
    rows[FEATURES[0]] = (rows.target > 0).astype(float)
    rows.loc[0, "target"] = 0.
    fitted = module.fit_probability_payoff(rows, CFG)
    assert fitted["class_rows"] == [16, 48]
    assert fitted["class_raw_variances"][0][0] == fitted["class_raw_variances"][1][0] == 0.
    assert fitted["class_variances"][0][0] > 0.
    assert module.probability_payoff_prediction(fitted, rows[FEATURES].iloc[1].to_numpy())["probability_advantage"] > .99


def test_whole_cycle_weight_does_not_change_when_its_observations_are_duplicated():
    rows = sample()
    duplicated = pd.concat([rows[rows.cycle_id.eq(1)].loc[rows[rows.cycle_id.eq(1)].index.repeat(2)], rows[rows.cycle_id.ne(1)]], ignore_index=True)
    duplicated.loc[duplicated.cycle_id.eq(1), "origin_index"] = np.arange(1, 33)
    duplicated.loc[duplicated.cycle_id.eq(1), "exit_index"] = 33
    duplicated["sample_weight"] = 1. / duplicated.groupby("cycle_id").cycle_id.transform("count")
    a, b = module.fit_probability_payoff(rows, CFG), module.fit_probability_payoff(duplicated, CFG)
    for key in ["mean", "scale", "class_means", "class_variances", "class_priors", "class_payoffs"]:
        np.testing.assert_allclose(a[key], b[key], atol=1e-12, rtol=1e-12)
    duplicated.loc[0, "sample_weight"] = 1.
    with pytest.raises(ValueError, match="总权重"):
        module.fit_probability_payoff(duplicated, CFG)


def test_missing_class_and_missing_feature_remain_explicitly_unknown():
    rows, originals = monthly_fixture()
    rows["target"] = .01
    with pytest.raises(module.MissingClassError):
        module.fit_probability_payoff(rows, CFG)
    records, _, _, counts = module.build_monthly_models(rows, originals, CFG)
    assert counts["missing_class_distinct_models"] == 1 and counts["reused_monthly_fits"] == 2
    assert all(r["status"] == "NO_VIEW_BOTH_CLASSES_REQUIRED" and r["model"] is None for r in records[1:])
    rows.loc[0, FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="禁止删行"):
        module.fit_probability_payoff(rows, CFG)


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
    original, calls = module.fit_probability_payoff, []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(module, "fit_probability_payoff", counted)
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
    monkeypatch.setattr(module, "fit_probability_payoff", failed)
    full, _, _, counts = module.build_monthly_models(rows, records, CFG)
    assert len(calls) == 1 and counts["failed_fits"] == 1
    assert all(r["model"] is None for r in full)
    assert all(r["status"] == "NO_VIEW_MODEL_FIT_FAILED" for r in full[1:])


def constant(value, t=0, latest=None):
    return {"fit_index": t, "parameter_first_fit_index": t, "latest_exit_index": t if latest is None else latest,
        "status": "FIT_COMPLETE", "model": {"kind": module.KIND, "features": FEATURES, "mean": [0.] * 8,
        "scale": [1.] * 8, "feature_clip": 5., "classes": [0, 1], "active_features": [False]*8,
        "class_priors": [(1-value)/2, (1+value)/2], "class_payoffs": [-1., 1.],
        "class_means": [[0.]*8, [0.]*8], "class_variances": [[0.]*8, [0.]*8]}}


def test_entry_version_is_fixed_missing_state_resets_count_and_future_model_is_rejected():
    data = fixture()[0]
    data.loc[2, "mom5"] = np.nan
    cycle = {"cycle_id": 1, "entry_index": 1, "entry_cost_cny": 100000., "mode": 1}
    controller = module.ProbabilityPayoffExitController(data, [constant(-.01), constant(.05, 2)])
    got = [controller(t, cycle, 100000., 100000.) for t in range(1, 5)]
    assert [r["negative_confirmation_count"] for r in got] == [1, 0, 1, 2]
    assert len({r["fixed_prediction_identity"] for r in got}) == 1
    cycle.update(cycle_id=2, entry_index=5)
    assert controller(5, cycle, 100000., 100000.)["continuation_prediction"] == pytest.approx(.05)
    with pytest.raises(ValueError, match="未来周期"):
        module.ProbabilityPayoffExitController(data, [constant(-.1, 0, 8)])(5, cycle, 100000., 100000.)
    with pytest.raises(ValueError, match="首次持仓收盘"):
        module.ProbabilityPayoffExitController(data, [constant(-.1)])(6, cycle, 100000., 100000.)


def test_real_blocked_exit_reentry_dividend_and_price_exit_without_initial_model():
    args = fixture()
    data = args[0]
    data.loc[3:, ["open", "close"]] = 9.9
    data.loc[4:, "previous_close"] = 9.9
    data.loc[3, ["open", "dividend"]] = [8.91, .1]
    args[1] = pd.DataFrame({"record_date": [data.date.iloc[2]], "ex_date": [data.date.iloc[3]],
        "payment_date": [data.date.iloc[5]], "cash_dividend_per_share": [.1]})
    args[5]["entry"][5] = 0
    ledger, decisions, cycles = simulate_rearmed_exit(*args, module.ProbabilityPayoffExitController(data, [constant(-.01), constant(.03, 3)]))
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
    ledger, decisions, cycles = simulate_rearmed_exit(*args, module.ProbabilityPayoffExitController(args[0], [constant(-.5, 2)]))
    assert decisions.continuation_prediction.dropna().empty and "价格退出条件" in cycles.iloc[0].exit_reasons
