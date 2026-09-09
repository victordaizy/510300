"""原八项持仓状态加一项浮盈正部与回撤的连续交互。"""
from bisect import bisect_right
import numpy as np
from sklearn.linear_model import Ridge
from research.learned_cycle_exit_v1 import FEATURES as ORIGINAL_FEATURES, CN as ORIGINAL_CN, state_values
from research.intraday_overnight_increment_v1 import require

INTERACTION = "positive_profit_times_drawdown"
FEATURES = ORIGINAL_FEATURES + [INTERACTION]
CN = ORIGINAL_CN + ["浮盈正数部分乘周期回撤"]


def interaction_value(profit, drawdown):
    profit, drawdown = np.asarray(profit, float), np.asarray(drawdown, float)
    value = np.maximum(profit, 0.)*drawdown
    return np.where(np.isfinite(profit) & np.isfinite(drawdown), value, np.nan)


def attach_interaction(samples):
    output = samples.copy()
    output[INTERACTION] = interaction_value(output.cycle_return, output.cycle_drawdown)
    return output


def interaction_values(data, t, cycle, current_value, peak_value):
    require(t >= cycle["entry_index"] and cycle["entry_cost_cny"] > 0 and peak_value > 0, "实际持仓时间、成本或最高价值不合法")
    original = state_values(data, t, cycle, current_value, peak_value)
    return np.r_[original, interaction_value(original[1], original[2])]


def fit_interaction_exit(rows, config):
    x = rows[FEATURES].to_numpy(float)
    y, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "九项输入或原目标不完整，禁止删除部分周期样本")
    require(np.isfinite(weights).all() and (weights > 0).all(), "周期等权权重不完整")
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.clip((x-mean)/scale, -config["feature_clip"], config["feature_clip"])
    fitted = Ridge(alpha=config["ridge_alpha"], solver="svd", fit_intercept=True)
    fitted.fit(design, y, sample_weight=weights)
    if not (np.isfinite(fitted.coef_).all() and np.isfinite(fitted.intercept_)):
        raise FloatingPointError("浮盈回撤交互模型出现非有限系数")
    return {"kind": "PROFIT_DRAWDOWN_INTERACTION_RIDGE", "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
        "coefficients": fitted.coef_.tolist(), "intercept": float(fitted.intercept_), "feature_clip": config["feature_clip"]}


def interaction_prediction(model, values):
    require(model["kind"] == "PROFIT_DRAWDOWN_INTERACTION_RIDGE" and model["features"] == FEATURES, "浮盈回撤交互模型身份不符")
    normalized = np.clip((np.asarray(values)-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"]+normalized@np.asarray(model["coefficients"]))


class InteractionExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.fit_indexes = [r["fit_index"] for r in models]
        require(self.fit_indexes == sorted(set(self.fit_indexes)), "九项模型月度时点必须唯一且递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        values = interaction_values(self.data, t, cycle, current_value, peak_value)
        index = bisect_right(self.fit_indexes, t)-1
        stored = self.models[index] if index >= 0 else None
        estimate, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if stored and stored["status"] == "FIT_COMPLETE" and np.isfinite(values).all():
            require(stored["latest_exit_index"] <= stored["fit_index"] <= t, "交互退出模型使用未来周期或模型")
            estimate, status = interaction_prediction(stored["model"], values), "PREDICTION_AVAILABLE"
            require(np.isfinite(estimate), "九项模型预测不是有限数")
        elif stored and stored["status"] == "FIT_COMPLETE":
            status = "NO_VIEW_INCOMPLETE_NINE_FEATURES"
        elif stored and stored["status"] in {"NO_VIEW_MODEL_FIT_FAILED", "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"}:
            status = stored["status"]
        self.negative_count = self.negative_count+1 if estimate is not None and estimate < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": estimate,
            "learning_fit_origin": self.data.date.iloc[stored["fit_index"]] if stored else None,
            "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
            **dict(zip(FEATURES, values))}
