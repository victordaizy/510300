"""静态新增概率幅度退出的真实账户入口与九项必要测试。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(name, text):
    with (ROOT / name).open("x", encoding="utf-8") as stream:
        stream.write(text)


def main():
    entry = (ROOT / "research/single_component_exit_v1.py").read_text(encoding="utf-8")
    entry = entry.replace('single_component_exit', 'probability_payoff_exit').replace('SINGLE_COMPONENT_EXIT', 'PROBABILITY_PAYOFF_EXIT')
    entry = entry.replace('SingleComponentExitController', 'ProbabilityPayoffExitController').replace('单成分', '概率幅度')
    entry = entry.replace('157', '158').replace('ROUND156', 'ROUND157').replace('tests["passed"] == 8', 'tests["passed"] == 9').replace('八项', '九项')
    start, end = entry.index('        n_components=1,'), entry.index('        model_selection_clock=')
    entry = entry[:start] + '''        variance_smoothing=1e-9, class_definition="ORIGINAL_TARGET_STRICTLY_ABOVE_ZERO_VS_NONPOSITIVE",
        model_loss="CYCLE_EQUAL_WEIGHTED_GAUSSIAN_CLASS_MOMENTS_AND_CLASS_MEAN_NET_PAYOFFS",
        feature_normalization="CYCLE_EQUAL_GLOBAL_STANDARDIZATION_CLIP5_WITHOUT_WITHIN_CYCLE_CENTERING",
        zero_variance_rule="EXCLUDE_EXACT_GLOBAL_CONSTANT_FACTORS_KEEP_ALL_PARAMETERS_ALL_CONSTANT_USES_PRIORS",
        missing_class="NO_VIEW_BOTH_CLASSES_REQUIRED", solver="CLOSED_FORM_WEIGHTED_CLASS_MOMENTS",
        solver_fallback=False, sklearn_version=sklearn.__version__,
''' + entry[end:]
    entry = entry.replace('["510300_sparse_vintage_exit_v1"]', '["510300_single_component_exit_v1"]')
    entry = entry.replace('原八因素合成单一分数后固定版本退出', '原八因素估计概率与盈亏幅度后固定版本退出')
    save('research/probability_payoff_exit_v1.py', entry)
    old = (ROOT / "tests/test_single_component_exit_v1.py").read_text(encoding="utf-8")
    tests = '''"""核对加权两类矩、概率与盈亏幅度、固定时钟和真实进出场。"""
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


'''
    tail = old[old.index('def monthly_fixture():'):]
    tail = tail.replace('fit_single_component_cycle', 'fit_probability_payoff').replace('SingleComponentExitController', 'ProbabilityPayoffExitController')
    start, end = tail.index('def constant('), tail.index('def test_entry_version_')
    constant = '''def constant(value, t=0, latest=None):
    return {"fit_index": t, "parameter_first_fit_index": t, "latest_exit_index": t if latest is None else latest,
        "status": "FIT_COMPLETE", "model": {"kind": module.KIND, "features": FEATURES, "mean": [0.] * 8,
        "scale": [1.] * 8, "feature_clip": 5., "classes": [0, 1], "active_features": [False]*8,
        "class_priors": [(1-value)/2, (1+value)/2], "class_payoffs": [-1., 1.],
        "class_means": [[0.]*8, [0.]*8], "class_variances": [[0.]*8, [0.]*8]}}


'''
    tail = tail[:start]+constant+tail[end:]
    tail = tail.replace('["continuation_prediction"] == .05', '["continuation_prediction"] == pytest.approx(.05)')
    save('tests/test_probability_payoff_exit_v1.py', tests+tail)
    print("第158轮真实账户入口与九项必要测试已准备，尚未运行新历史。", flush=True)


if __name__ == "__main__":
    main()
