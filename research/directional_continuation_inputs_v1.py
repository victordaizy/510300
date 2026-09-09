"""用原持仓状态估计继续持有严格占优的概率。"""
from bisect import bisect_right
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from research.intraday_overnight_increment_v1 import require
from research.learned_cycle_exit_v1 import FEATURES, CN, state_values


def fit_directional_continuation(rows, config):
    x = rows[FEATURES].to_numpy(float)
    target, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(target).all(), "概率训练输入或原始目标不完整")
    require(np.isfinite(weights).all() and (weights > 0).all(), "周期等权训练权重不完整")
    labels = (target > 0).astype(int)
    if len(np.unique(labels)) != 2:
        return None
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.clip((x - mean) / scale, -config["feature_clip"], config["feature_clip"])
    fitted = LogisticRegression(C=config["logistic_C"], l1_ratio=0., solver="lbfgs", max_iter=1000, tol=1e-8,
        fit_intercept=True, class_weight=None, warm_start=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            fitted.fit(design, labels, sample_weight=weights)
        except ConvergenceWarning as error:
            raise RuntimeError("概率模型没有收敛，不换求解器补救") from error
    require(fitted.classes_.tolist() == [0, 1], "继续占优的正类身份不符")
    if not np.isfinite(fitted.coef_).all() or not np.isfinite(fitted.intercept_).all():
        raise FloatingPointError("概率模型出现非有限系数")
    stored = {"kind": "DIRECTIONAL_CONTINUATION_LOGISTIC", "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
        "coefficients": fitted.coef_[0].tolist(), "intercept": float(fitted.intercept_[0]), "feature_clip": config["feature_clip"],
        "iterations": int(fitted.n_iter_[0]), "positive_rows": int(labels.sum()), "zero_target_rows": int((target == 0).sum()),
        "positive_weight": float(weights[labels == 1].sum()), "nonpositive_weight": float(weights[labels == 0].sum())}
    predicted = expit(stored["intercept"] + design @ np.array(stored["coefficients"]))
    require(np.allclose(predicted, fitted.predict_proba(design)[:, 1], rtol=0, atol=1e-14), "保存概率公式不等于拟合器正类概率")
    return stored


def directional_probability(model, values):
    require(model["kind"] == "DIRECTIONAL_CONTINUATION_LOGISTIC" and model["features"] == FEATURES, "概率模型身份或输入不符")
    normalized = np.clip((np.asarray(values) - model["mean"]) / model["scale"], -model["feature_clip"], model["feature_clip"])
    score = float(model["intercept"] + normalized @ np.asarray(model["coefficients"]))
    return float(expit(score)), score


def directional_chinese_formula(model):
    lines = [f"线性评分截距为{model['intercept']:.10f}。八项输入分别减训练均值、除训练标准差，限制在负5至正5，再乘对应系数；八项与截距相加。取线性评分相反数的自然指数、加1，再用1除以上述结果，得到继续持有严格占优的概率。概率不是预期收益。", ""]
    lines.extend(f"- {name}：均值{mean:.10f}，标准差{scale:.10f}，系数{coefficient:.10f}。"
        for name, mean, scale, coefficient in zip(CN, model["mean"], model["scale"], model["coefficients"], strict=True))
    return lines


class DirectionalExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.fit_indexes = [m["fit_index"] for m in models]
        require(self.fit_indexes == sorted(set(self.fit_indexes)), "概率模型时点必须唯一递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if cycle["cycle_id"] != self.cycle_id:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        values = state_values(self.data, t, cycle, current_value, peak_value)
        index = bisect_right(self.fit_indexes, t) - 1
        stored = self.models[index] if index >= 0 else None
        probability, score, status = None, None, "NO_VIEW_NO_MATURE_MODEL"
        if stored and stored["status"] == "FIT_COMPLETE" and np.isfinite(values).all():
            require(stored["latest_exit_index"] <= stored["fit_index"] <= t, "概率模型使用未来周期或模型")
            probability, score = directional_probability(stored["model"], values)
            require(np.isfinite(probability) and np.isfinite(score), "概率或评分不是有限数")
            status = "PREDICTION_AVAILABLE"
        elif stored and stored["status"] == "FIT_COMPLETE":
            status = "NO_VIEW_INCOMPLETE_HOLDING_FEATURES"
        elif stored:
            status = stored["status"]
        self.negative_count = self.negative_count + 1 if probability is not None and probability < .5 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": None,
            "continuation_probability": probability, "continuation_log_odds": score,
            "learning_fit_origin": self.data.date.iloc[stored["fit_index"]] if stored else pd.NaT,
            "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
            **dict(zip(FEATURES, values))}
