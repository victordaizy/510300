"""原八项实际持仓状态的提升树继续价值，模型完整保存为可复算数值。"""
from bisect import bisect_right
import numpy as np
from threadpoolctl import threadpool_limits
from research.adaptive_allocation_v1 import model_for
from research.learned_cycle_exit_v1 import FEATURES, CN, state_values
from research.intraday_overnight_increment_v1 import require


def export_boosted_model(fitted):
    require(fitted.n_trees_per_iteration_ == 1 and fitted.loss == "squared_error", "只支持冻结的平方损失单目标提升回归")
    trees = []
    for iteration in fitted._predictors:
        require(len(iteration) == 1, "提升树每轮目标数量不符")
        nodes = iteration[0].nodes
        require(not nodes["is_categorical"].any(), "八项数值因子不得出现分类分支")
        tree = {"feature": nodes["feature_idx"].astype(int).tolist(), "threshold": nodes["num_threshold"].tolist(),
                "left": nodes["left"].astype(int).tolist(), "right": nodes["right"].astype(int).tolist(),
                "is_leaf": nodes["is_leaf"].astype(bool).tolist(), "value": nodes["value"].tolist(), "sample_count": nodes["count"].astype(int).tolist()}
        trees.append(tree)
    return {"kind": "HGB_CONDITIONAL_CONTINUATION", "features": FEATURES.copy(), "baseline": float(fitted._baseline_prediction[0, 0]),
            "trees": trees, "iterations": int(fitted.n_iter_), "parameters": fitted.get_params(),
            "leaf_value_scale": "ALREADY_INCLUDES_LEARNING_RATE_DO_NOT_MULTIPLY_AGAIN", "feature_transform": "RAW_EIGHT_FINITE_NUMERIC_FEATURES"}


def boosted_prediction(model, values):
    require(model["kind"] == "HGB_CONDITIONAL_CONTINUATION" and model["features"] == FEATURES, "提升退出模型身份不符")
    x = np.asarray(values, float)
    require(x.shape == (8,) and np.isfinite(x).all(), "提升退出必须有八项完整实际状态")
    result = model["baseline"]
    for tree in model["trees"]:
        node = 0
        while not tree["is_leaf"][node]:
            node = tree["left"][node] if x[tree["feature"][node]] <= tree["threshold"][node] else tree["right"][node]
        result += tree["value"][node]
    require(np.isfinite(result), "保存提升模型预测不是有限值")
    return float(result)


def fit_boosted_exit(rows, cfg):
    x = rows[FEATURES].to_numpy(float)
    y, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "提升退出原完整样本输入或目标缺失，禁止删行")
    require(np.isfinite(weights).all() and (weights > 0).all(), "完整周期权重不合法")
    fitted = model_for("HGB", cfg["random_seed"])
    require(fitted.get_params() == cfg["model_parameters"], "提升退出参数与冻结原算法不符")
    with threadpool_limits(limits=1):
        fitted.fit(x, y, sample_weight=weights)
        expected = fitted.predict(x)
    saved = export_boosted_model(fitted)
    actual = np.asarray([boosted_prediction(saved, v) for v in x])
    error = float(np.max(np.abs(actual-expected)))
    require(error < 1e-12, "保存树不能重现拟合库预测")
    saved["training_check"] = {"rows": len(rows), "maximum_library_prediction_difference": error,
                               "cycle_weighted_training_mse": float(np.average((actual-y)**2, weights=weights)),
                               "baseline_mse": float(np.average((saved["baseline"]-y)**2, weights=weights))}
    return saved


class BoostedExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "提升退出模型月度原点必须唯一递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        x = state_values(self.data, t, cycle, current_value, peak_value)
        k = bisect_right(self.indexes, t)-1
        record = self.models[k] if k >= 0 else None
        value, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= t, "提升退出使用未来周期或模型")
            if np.isfinite(x).all():
                value, status = boosted_prediction(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_EIGHT_FEATURES"
        elif record and record["status"] in {"NO_VIEW_MODEL_FIT_FAILED", "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"}:
            status = record["status"]
        self.negative_count = self.negative_count+1 if value is not None and value < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": value,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
                **dict(zip(FEATURES, x))}
