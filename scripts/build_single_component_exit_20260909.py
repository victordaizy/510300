"""静态生成新的单成分学习入口与必要测试，保留旧冻结文件。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("静态替换目标不唯一：" + old[:80])
    return text.replace(old, new, 1)


def save(name, text):
    with (ROOT / name).open("x", encoding="utf-8") as stream:
        stream.write(text)


def main():
    source = (ROOT / "research/sparse_vintage_exit_inputs_v1.py").read_text(encoding="utf-8")
    start, end = source.index("def kkt_violation("), source.index("def input_identity(")
    fitting = '''def fit_single_component_cycle(rows, cfg):
    require(cfg["n_components"] == 1 and cfg["orthogonality_tolerance"] == 1e-10,
            "单成分数量或数值核对设置改变")
    dx, dy, w, mean, scale, groups = centered_inputs(rows, cfg)
    total = float(w.sum())
    covariance = dx.T @ (w * dy) / total
    require(np.isfinite(covariance).all(), "因素与目标协动非有限")
    direction, beta = np.zeros(len(FEATURES)), np.zeros(len(FEATURES))
    slope = score_variance = score_covariance = residual_covariance = 0.
    magnitude = float(np.max(abs(covariance)))
    if magnitude > 0.:
        direction = covariance / magnitude
        direction /= np.linalg.norm(direction)
        score = dx @ direction
        score_variance = float(np.average(score ** 2, weights=w))
        score_covariance = float(np.average(score * dy, weights=w))
        require(np.isfinite(score_variance) and score_variance > 0 and np.isfinite(score_covariance),
                "非零协动的单成分分母必须为有限正数")
        slope = score_covariance / score_variance
        beta = slope * direction
        residual_covariance = float(np.average(score * (dx @ beta - dy), weights=w))
        require(abs(residual_covariance) <= cfg["orthogonality_tolerance"], "单成分回归残差未满足正交条件")
    require(np.isfinite(beta).all() and np.isfinite(slope), "单成分系数非有限")
    for group in groups:
        group["cycle_intercept"] = float(group["target_mean"] - np.asarray(group["standardized_feature_mean"]) @ beta)
    intercept = float(np.mean([group["cycle_intercept"] for group in groups]))
    require(np.isfinite(intercept), "新周期截距非有限")
    return {"kind": KIND, "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
        "coefficients": beta.tolist(), "intercept": intercept, "feature_clip": cfg["feature_clip"],
        "cycle_intercepts": groups, "n_components": 1, "effective_cycle_weight": total,
        "factor_target_covariance": covariance.tolist(), "component_direction": direction.tolist(),
        "component_slope": slope, "score_variance": score_variance, "score_target_covariance": score_covariance,
        "score_residual_covariance": residual_covariance, "zero_covariance_model": magnitude == 0.,
        "nonzero_factors": [name for name, value in zip(FEATURES, beta) if value != 0],
        "nonzero_factor_count": int((beta != 0).sum()),
        "training_weighted_mse": float(np.average((dx @ beta - dy) ** 2, weights=w)),
        "new_cycle_intercept_rule": "EQUAL_MEAN_OF_MATURE_TRAINING_CYCLE_INTERCEPTS"}


'''
    output = source[:start] + fitting + source[end:]
    output = output.replace('import warnings\n', '').replace('from sklearn.exceptions import ConvergenceWarning\n', '').replace('from sklearn.linear_model import Lasso\n', '')
    output = output.replace('WITHIN_CYCLE_COST_SCALED_LASSO', 'WITHIN_CYCLE_WEIGHTED_PLS1')
    output = output.replace('SparseVintageExitController', 'SingleComponentExitController')
    output = output.replace('fit_sparse_cycle', 'fit_single_component_cycle').replace('sparse_prediction', 'single_component_prediction')
    output = output.replace('稀疏', '单成分').replace('FloatingPointError, ConvergenceWarning', 'FloatingPointError')
    output = output.replace('["feature_clip", "lasso_alpha", "lasso_max_iter", "lasso_tolerance", "kkt_tolerance"]', '["feature_clip", "n_components", "orthogonality_tolerance"]')
    output = replace_once(output, '"kkt_max_violation": model["kkt_max_violation"] if model else None,\n            "solver_iterations": model["solver_iterations"] if model else None',
        '"score_residual_covariance": model["score_residual_covariance"] if model else None,\n            "zero_covariance_model": model["zero_covariance_model"] if model else None')
    save('research/single_component_exit_inputs_v1.py', output)
    entry = (ROOT / "research/sparse_vintage_exit_v1.py").read_text(encoding="utf-8")
    entry = entry.replace('sparse_vintage_exit', 'single_component_exit').replace('SPARSE_VINTAGE_EXIT', 'SINGLE_COMPONENT_EXIT')
    entry = entry.replace('SparseVintageExitController', 'SingleComponentExitController').replace('稀疏', '单成分')
    entry = entry.replace('156', '157').replace('ROUND155', 'ROUND156')
    entry = entry.replace('tests["passed"] == 7', 'tests["passed"] == 8').replace('七项', '八项')
    entry = replace_once(entry, 'lasso_alpha=.0014, lasso_max_iter=10000, lasso_tolerance=1e-10, kkt_tolerance=1e-8,\n        model_loss="CYCLE_EQUAL_AVERAGED_WITHIN_SQUARED_ERROR_OVER_TWO_PLUS_FIXED_L1_PENALTY",\n        zero_initial_solution="ACCEPT_EXACT_KKT_SATISFIED_ZERO_VECTOR_BEFORE_ITERATION",\n        solver="CYCLIC_COORDINATE_DESCENT_FROM_ZERO", solver_fallback=False, sklearn_version=sklearn.__version__,',
        'n_components=1, orthogonality_tolerance=1e-10,\n        model_loss="SINGLE_COVARIANCE_DIRECTION_THEN_CYCLE_EQUAL_WEIGHTED_ONE_SCORE_LEAST_SQUARES",\n        zero_initial_solution="EXACT_ZERO_COVARIANCE_RETURNS_ZERO_COEFFICIENTS_WITH_CYCLE_INTERCEPTS",\n        solver="CLOSED_FORM_WEIGHTED_SINGLE_COMPONENT", solver_fallback=False, sklearn_version=sklearn.__version__,')
    entry = replace_once(entry, '    require(abs(cfg["lasso_alpha"] - 2 * (cfg["costs"]["BASE"]["commission"] + cfg["costs"]["BASE"]["slippage"])) < 1e-15, "固定惩罚尺度与基础成本口径不同")\n', '')
    entry = entry.replace('["510300_cycle_serial_error_exit_v1", "510300_median_slope_risk_v1"]', '["510300_sparse_vintage_exit_v1"]')
    entry = entry.replace('原八因素单成分学习后固定版本退出', '原八因素合成单一分数后固定版本退出')
    save('research/single_component_exit_v1.py', entry)
    tests = (ROOT / "tests/test_sparse_vintage_exit_v1.py").read_text(encoding="utf-8")
    tests = tests.replace('sparse_vintage_exit', 'single_component_exit').replace('fit_sparse_cycle', 'fit_single_component_cycle')
    tests = tests.replace('SparseVintageExitController', 'SingleComponentExitController').replace('sparse_prediction', 'single_component_prediction').replace('稀疏', '单成分')
    tests = replace_once(tests, 'CFG = {"feature_clip": 5., "lasso_alpha": .0014, "lasso_max_iter": 10000, "lasso_tolerance": 1e-10,\n       "kkt_tolerance": 1e-8,',
        'CFG = {"feature_clip": 5., "n_components": 1, "orthogonality_tolerance": 1e-10,')
    tests = tests.replace('from scipy.linalg import hadamard', 'from scipy.linalg import hadamard\nfrom sklearn.cross_decomposition import PLSRegression')
    left, right = tests.index('def test_nonzero_solution_'), tests.index('def test_whole_cycle_weight_')
    new_tests = '''def test_single_component_recovers_known_orthogonal_response_and_constant_factor_is_zero():
    fitted = module.fit_single_component_cycle(sample(), CFG)
    expected = np.zeros(8); expected[:2] = [.04, -.02]
    np.testing.assert_allclose(fitted["coefficients"], expected, atol=1e-12, rtol=0)
    assert fitted["nonzero_factor_count"] == 2 and fitted["coefficients"][3] == 0.
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


'''
    tests = tests[:left] + new_tests + tests[right:]
    tests = tests.replace('fitted["solver_iterations"] == 0', 'fitted["zero_covariance_model"]')
    save('tests/test_single_component_exit_v1.py', tests)
    print("第157轮单成分模型、真实账户入口与八项必要测试已静态生成；尚未运行新历史。", flush=True)


if __name__ == "__main__":
    main()
