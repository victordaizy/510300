"""按周期等权估计线性条件中位数，保留原连续负值退出语义。"""
from __future__ import annotations

from bisect import bisect_right
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import QuantileRegressor
from threadpoolctl import threadpool_limits

from research.learned_cycle_exit_v1 import FEATURES, CN, state_values
from research.intraday_overnight_increment_v1 import require


def fit_median(rows, config):
    x = rows[FEATURES].to_numpy(float)
    y, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "中位数训练输入或标签缺失")
    require(np.isfinite(weights).all() and (weights > 0).all(), "中位数训练权重必须为有限正值")
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.clip((x - mean) / scale, -config["feature_clip"], config["feature_clip"])
    estimator = QuantileRegressor(quantile=.5, alpha=0., fit_intercept=True, solver="highs")
    with warnings.catch_warnings(), threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        estimator.fit(design, y, sample_weight=weights)
    require(np.isfinite(estimator.coef_).all() and np.isfinite(estimator.intercept_), "中位数系数不是有限数")
    predictions = estimator.predict(design)
    return {"kind": "LINEAR_CONDITIONAL_MEDIAN", "quantile": .5, "alpha": 0., "solver": "highs",
        "mean": mean.tolist(), "scale": scale.tolist(), "coefficients": estimator.coef_.tolist(),
        "intercept": float(estimator.intercept_), "feature_clip": config["feature_clip"], "solver_iterations": int(estimator.n_iter_),
        "training_weighted_absolute_error": float(np.average(np.abs(y - predictions), weights=weights))}


def predict_median(model, values):
    require(model["kind"] == "LINEAR_CONDITIONAL_MEDIAN", "不是本轮中位数模型")
    normalized = np.clip((values - np.array(model["mean"])) / np.array(model["scale"]), -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"] + normalized @ np.array(model["coefficients"]))


def median_chinese_formula(model):
    lines = [f"预测继续持有收益的条件中位数：截距为{model['intercept']:.12f}，再加下列各项。各项先减训练均值、除训练标准差，限制在负5到5，再乘对应系数。", ""]
    lines += [f"- {name}：均值{mean:.12f}，标准差{scale:.12f}，标准化后系数{coefficient:.12f}。"
        for name, mean, scale, coefficient in zip(CN, model["mean"], model["scale"], model["coefficients"])]
    return lines


class MedianExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.fit_indexes = [m["fit_index"] for m in models]
        require(self.fit_indexes == sorted(set(self.fit_indexes)), "模型生效原点必须唯一且递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if cycle["cycle_id"] != self.cycle_id:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        values = state_values(self.data, t, cycle, current_value, peak_value)
        index = bisect_right(self.fit_indexes, t) - 1
        stored = self.models[index] if index >= 0 else None
        estimate, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if stored and stored["status"] == "FIT_COMPLETE" and np.isfinite(values).all():
            require(stored["latest_exit_index"] <= stored["fit_index"] <= t, "中位数退出使用了未来周期或模型")
            estimate = predict_median(stored["model"], values)
            status = "PREDICTION_AVAILABLE"
        elif stored and stored["status"] == "FIT_COMPLETE":
            status = "NO_VIEW_INCOMPLETE_HOLDING_FEATURES"
        elif stored and stored["status"] == "NO_VIEW_MODEL_FIT_FAILED":
            status = "NO_VIEW_MODEL_FIT_FAILED"
        self.negative_count = self.negative_count + 1 if estimate is not None and estimate < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": estimate,
            "learning_fit_origin": self.data.date.iloc[stored["fit_index"]] if stored else pd.NaT,
            "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
            **dict(zip(FEATURES, values))}
