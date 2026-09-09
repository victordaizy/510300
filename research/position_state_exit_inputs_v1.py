"""只使用持有时间、实际盈亏与回撤拟合并执行继续收益退出。"""
from __future__ import annotations

from bisect import bisect_right

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from research.intraday_overnight_increment_v1 import require

FEATURES = ["log_holding_days", "cycle_return", "cycle_drawdown"]
CN = ["持仓时间对数", "含分红实际盈亏", "含分红持仓回撤"]


def position_values(t, cycle, current_value, peak_value):
    require(t >= cycle["entry_index"], "持仓判断必须在实际买入之后")
    require(cycle["entry_cost_cny"] > 0 and peak_value > 0, "持仓成本和最高价值必须为正")
    return np.array([np.log1p(t - cycle["entry_index"] + 1), current_value / cycle["entry_cost_cny"] - 1,
        current_value / peak_value - 1], dtype=float)


def fit_position_state(rows, config):
    x = rows[FEATURES].to_numpy(float)
    target, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(target).all(), "三项持仓训练输入或目标不完整")
    require(np.isfinite(weights).all() and (weights > 0).all(), "周期等权训练权重不完整")
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.clip((x - mean) / scale, -config["feature_clip"], config["feature_clip"])
    fitted = Ridge(alpha=config["ridge_alpha"], solver="svd", fit_intercept=True)
    fitted.fit(design, target, sample_weight=weights)
    if not (np.isfinite(fitted.coef_).all() and np.isfinite(fitted.intercept_)):
        raise FloatingPointError("三项模型数值求解出现非有限系数")
    return {"kind": "POSITION_STATE_RIDGE", "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
        "coefficients": fitted.coef_.tolist(), "intercept": float(fitted.intercept_), "feature_clip": config["feature_clip"]}


def position_prediction(model, values):
    require(model["kind"] == "POSITION_STATE_RIDGE" and model["features"] == FEATURES, "持仓三因子模型身份不符")
    normalized = np.clip((np.asarray(values) - model["mean"]) / model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"] + normalized @ np.asarray(model["coefficients"]))


def position_chinese_formula(model):
    lines = [f"预测继续持有相对提前退出收益的截距为{model['intercept']:.10f}。每项先减训练均值、除训练标准差，限制在负5至正5，再乘该项系数；三项结果与截距相加。", ""]
    lines.extend(f"- {name}：训练均值{mean:.10f}，训练标准差{scale:.10f}，标准化后的系数{coefficient:.10f}。"
        for name, mean, scale, coefficient in zip(CN, model["mean"], model["scale"], model["coefficients"], strict=True))
    return lines


class PositionStateExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.fit_indexes = [r["fit_index"] for r in models]
        require(self.fit_indexes == sorted(set(self.fit_indexes)), "三项模型月度时点必须唯一且递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        values = position_values(t, cycle, current_value, peak_value)
        index = bisect_right(self.fit_indexes, t) - 1
        stored = self.models[index] if index >= 0 else None
        estimate, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if stored and stored["status"] == "FIT_COMPLETE" and np.isfinite(values).all():
            require(stored["latest_exit_index"] <= stored["fit_index"] <= t, "三项退出模型使用未来周期或模型")
            estimate, status = position_prediction(stored["model"], values), "PREDICTION_AVAILABLE"
            require(np.isfinite(estimate), "三项模型预测不是有限数")
        elif stored and stored["status"] == "FIT_COMPLETE":
            status = "NO_VIEW_INCOMPLETE_HOLDING_FEATURES"
        elif stored and stored["status"] == "NO_VIEW_MODEL_FIT_FAILED":
            status = "NO_VIEW_MODEL_FIT_FAILED"
        self.negative_count = self.negative_count + 1 if estimate is not None and estimate < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": estimate,
            "learning_fit_origin": self.data.date.iloc[stored["fit_index"]] if stored else pd.NaT,
            "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
            **dict(zip(FEATURES, values))}
