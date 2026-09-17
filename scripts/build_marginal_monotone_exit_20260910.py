"""静态生成八条保序曲线的新学习和实际账户入口，不改旧冻结规则。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def save(name, text):
    with (ROOT / name).open("x", encoding="utf-8") as stream:
        stream.write(text)


def main():
    text = (ROOT / "research/probability_payoff_exit_inputs_v1.py").read_text(encoding="utf-8")
    start, end = text.index('KIND = '), text.index('def weighted_moments(')
    text = text[:start] + '''KIND = "CYCLE_EQUAL_EIGHT_MARGINAL_MONOTONE_MEAN"
PREDICTION_KEYS = ["kind", "features", "mean", "scale", "feature_clip", "curves", "curve_weights"]


''' + text[end:]
    start, end = text.index('def fit_probability_payoff('), text.index('def input_identity(')
    fitting = '''def weighted_pava_curve(x, y, weights, direction):
    require(direction in [-1, 1], "保序方向必须为已确定的上升或下降")
    x, y, weights = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
    require(x.ndim == 1 and x.shape == y.shape == weights.shape and len(x) > 0, "保序输入维度不同")
    require(np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(weights).all() and (weights > 0).all(), "保序输入或权重无效")
    unique, inverse = np.unique(x, return_inverse=True)
    support_weight, support_sum = np.zeros(len(unique)), np.zeros(len(unique))
    np.add.at(support_weight, inverse, weights)
    np.add.at(support_sum, inverse, weights*y*direction)
    blocks = []
    for i in range(len(unique)):
        blocks.append([i, i+1, float(support_weight[i]), float(support_sum[i])])
        while len(blocks) >= 2 and blocks[-2][3]/blocks[-2][2] > blocks[-1][3]/blocks[-1][2]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2]+right[2], left[3]+right[3]])
    fitted = np.empty(len(unique))
    for first, last, weight, value in blocks:
        fitted[first:last] = direction*value/weight
    require(np.isfinite(fitted).all() and (np.diff(fitted)*direction >= 0).all(), "保序输出无效或违反方向")
    keep = np.ones(len(unique), bool)
    if len(unique) > 2:
        keep[1:-1] = (fitted[1:-1] != fitted[:-2]) | (fitted[1:-1] != fitted[2:])
    return {"x": unique[keep].tolist(), "y": fitted[keep].tolist(), "support_count": len(unique), "block_count": len(blocks)}


def fit_marginal_monotone(rows, cfg):
    require(cfg["curve_weight"] == .125 and cfg["direction_rule"] == "SIGN_OF_CYCLE_WEIGHTED_COVARIANCE", "曲线固定等权或方向规则改变")
    z, y, w, mean, scale = standardized_inputs(rows, cfg)
    target_mean = float(np.average(y, weights=w))
    feature_mean, _ = weighted_moments(z, w)
    covariances = np.average((z-feature_mean)*(y-target_mean)[:, None], axis=0, weights=w)
    require(np.isfinite(covariances).all(), "训练因素与目标协方差无效")
    curves, solves = [], 0
    for j, name in enumerate(FEATURES):
        x = z[:, j]
        constant = np.ptp(x) == 0.
        c = 0. if constant else float(covariances[j])
        if constant or c == 0.:
            curve = {"x": [float(x.min())], "y": [target_mean], "support_count": len(np.unique(x)), "block_count": 1,
                "direction": 0, "constant_reason": "CONSTANT_INPUT" if constant else "EXACT_ZERO_COVARIANCE"}
        else:
            direction = 1 if c > 0. else -1
            curve = weighted_pava_curve(x, y, w, direction)
            curve.update(direction=direction, constant_reason=None)
            solves += 1
        curve.update(feature=name, weighted_covariance=c, training_weighted_mse=float(np.average((y-np.interp(x, curve["x"], curve["y"]))**2, weights=w)))
        curves.append(curve)
    return {"kind": KIND, "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
        "feature_clip": cfg["feature_clip"], "curves": curves, "curve_weights": [.125]*8,
        "target_mean": target_mean, "pava_solves": solves, "constant_curves": 8-solves,
        "curve_count": 8, "total_saved_knots": sum(len(c["x"]) for c in curves),
        "prediction_interpretation": "EQUAL_MEAN_OF_EIGHT_UNIVARIATE_CONTINUATION_ESTIMATES"}


'''
    text = text[:start]+fitting+text[end:]
    text = text.replace('"positive_class": "ORIGINAL_TARGET_STRICTLY_ABOVE_ZERO",\n                **{key: cfg[key] for key in ["feature_clip", "variance_smoothing"]}',
        '**{key: cfg[key] for key in ["feature_clip", "curve_weight", "direction_rule"]}')
    text = text.replace('fit_probability_payoff(', 'fit_marginal_monotone(')
    text = text.replace('                except MissingClassError as error:\n                    item.update(status="NO_VIEW_BOTH_CLASSES_REQUIRED", failure=str(error))\n', '')
    first = text.index('            "active_factor_count": model[')
    last = text.index('        records.append(record)', first)
    text = text[:first]+'''            "pava_solves": model["pava_solves"] if model else None,
            "constant_curves": model["constant_curves"] if model else None,
            "total_saved_knots": model["total_saved_knots"] if model else None})
'''+text[last:]
    first = text.index('        "reused_monthly_fits": reused,')
    last = text.index('    return records,', first)
    text = text[:first]+'''        "reused_monthly_fits": reused,
        "new_scalar_curve_solves": sum(v["model"]["pava_solves"] for v in cache.values() if v["model"] is not None),
        "constant_curve_records": sum(v["model"]["constant_curves"] for v in cache.values() if v["model"] is not None),
        "curve_records": sum(v["model"]["curve_count"] for v in cache.values() if v["model"] is not None)}
'''+text[last:]
    first, last = text.index('def probability_payoff_prediction('), text.index('class ProbabilityPayoffExitController:')
    text = text[:first]+'''def marginal_monotone_prediction(model, values):
    require(model["kind"] == KIND and model["features"] == FEATURES and model["curve_weights"] == [.125]*8, "八曲线退出模型身份不同")
    x = np.asarray(values, float)
    require(x.shape == (8,) and np.isfinite(x).all(), "八曲线预测需要完整八因素")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    estimates = []
    for j, (name, curve) in enumerate(zip(FEATURES, model["curves"], strict=True)):
        require(name == curve["feature"] and len(curve["x"]) == len(curve["y"]) > 0, "曲线因素或断点维度不同")
        require(np.isfinite(curve["x"]).all() and np.isfinite(curve["y"]).all(), "曲线断点数值无效")
        estimates.append(float(np.interp(z[j], curve["x"], curve["y"])))
    require(len(estimates) == 8 and np.isfinite(estimates).all(), "八条曲线预测不完整")
    return {"prediction": float(np.mean(estimates)), **{f"curve_prediction_{name}": value for name, value in zip(FEATURES, estimates)}}


'''+text[last:]
    text = text.replace('ProbabilityPayoffExitController', 'MarginalMonotoneExitController').replace('probability_payoff_prediction(', 'marginal_monotone_prediction(')
    text = text.replace('["probability_advantage", "probability_disadvantage", "expected_gain_contribution", "expected_loss_contribution"]', '[f"curve_prediction_{name}" for name in FEATURES]')
    text = text.replace('概率幅度', '八曲线').replace('两类状态概率和盈亏幅度', '八个因素的单调曲线')
    save('research/marginal_monotone_exit_inputs_v1.py', text)
    entry = (ROOT / 'research/probability_payoff_exit_v1.py').read_text(encoding='utf-8')
    entry = entry.replace('probability_payoff_exit', 'marginal_monotone_exit').replace('PROBABILITY_PAYOFF_EXIT', 'MARGINAL_MONOTONE_EXIT')
    entry = entry.replace('ProbabilityPayoffExitController', 'MarginalMonotoneExitController').replace('概率幅度', '八曲线').replace('158', '159').replace('ROUND157', 'ROUND158')
    first, last = entry.index('        variance_smoothing='), entry.index('        model_selection_clock=')
    entry = entry[:first]+'''        curve_weight=.125, direction_rule="SIGN_OF_CYCLE_WEIGHTED_COVARIANCE",
        model_loss="EIGHT_INDEPENDENT_CYCLE_EQUAL_WEIGHTED_ISOTONIC_SQUARED_ERRORS",
        interpolation="LINEAR_BETWEEN_SAVED_KNOTS_CONSTANT_ENDPOINTS",
        constant_rule="EXACT_ZERO_COVARIANCE_OR_CONSTANT_INPUT_USES_GLOBAL_WEIGHTED_TARGET_MEAN",
        solver="EXACT_TIE_AGGREGATION_THEN_WEIGHTED_ADJACENT_BLOCK_MERGING", solver_fallback=False,
        sklearn_version=sklearn.__version__,
'''+entry[last:]
    entry = entry.replace('MARGINAL_MONOTONE_EXIT_NEXT_20260909', 'MARGINAL_MONOTONE_EXIT_NEXT_20260910')
    entry = entry.replace('["510300_single_component_exit_v1"]', '["510300_probability_payoff_exit_v1"]')
    entry = entry.replace('原八因素估计概率与盈亏幅度后固定版本退出', '原八因素保序曲线等权平均后固定版本退出')
    save('research/marginal_monotone_exit_v1.py', entry)
    print('第159轮八曲线模型和真实账户入口已静态生成，尚未训练新历史。', flush=True)


if __name__ == '__main__':
    main()
