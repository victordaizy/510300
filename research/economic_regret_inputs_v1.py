"""以错误持有或退出的经济损失加权，训练持仓退出决策。"""
from bisect import bisect_right
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from research.intraday_overnight_increment_v1 import require
from research.learned_cycle_exit_v1 import FEATURES, CN, state_values


def economic_weights(target, base_weights):
    """周期基础权重乘绝对净差额，再归一到原基础权重总量。"""
    target, base = np.asarray(target, float), np.asarray(base_weights, float)
    require(target.shape == base.shape and len(target) > 0, "经济差额与基础权重长度不符")
    require(np.isfinite(target).all() and np.isfinite(base).all() and (base > 0).all(), "经济差额或基础权重无效")
    raw = base * np.abs(target)
    require(np.isfinite(raw).all() and np.isfinite(raw.sum()), "经济损失权重不是有限数")
    if raw.sum() == 0:
        return np.zeros(len(raw))
    return raw / raw.sum() * base.sum()


def economic_support(rows):
    weights = economic_weights(rows.target, rows.sample_weight)
    positive_weight = weights > 0
    if not positive_weight.any():
        return "NO_VIEW_NO_ECONOMIC_DIFFERENCE"
    if len(np.unique((rows.target.to_numpy(float)[positive_weight] > 0).astype(int))) != 2:
        return "NO_VIEW_SINGLE_ECONOMIC_CLASS"
    return "ECONOMIC_FIT_SUPPORT_AVAILABLE"


def fit_economic_regret(rows, config):
    x = rows[FEATURES].to_numpy(float)
    target, base = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(np.isfinite(x).all(), "经济退出训练因子不完整")
    weights = economic_weights(target, base)
    selected = weights > 0
    labels = (target > 0).astype(int)
    if not selected.any() or len(np.unique(labels[selected])) != 2:
        return None
    # 标准化仍按原周期基础权重，只改变拟合错误代价。
    mean = np.average(x, axis=0, weights=base)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=base))
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.clip((x - mean) / scale, -config["feature_clip"], config["feature_clip"])
    fitted = LogisticRegression(C=config["logistic_C"], l1_ratio=0., solver="lbfgs", max_iter=1000, tol=1e-8,
        fit_intercept=True, class_weight=None, warm_start=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            fitted.fit(design[selected], labels[selected], sample_weight=weights[selected])
        except ConvergenceWarning as error:
            raise RuntimeError("经济退出模型未收敛，不换求解器") from error
    require(fitted.classes_.tolist() == [0, 1], "经济退出正类身份不符")
    if not np.isfinite(fitted.coef_).all() or not np.isfinite(fitted.intercept_).all():
        raise FloatingPointError("经济退出模型系数非有限")
    model = {"kind": "ECONOMIC_REGRET_LOGISTIC", "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
        "coefficients": fitted.coef_[0].tolist(), "intercept": float(fitted.intercept_[0]), "feature_clip": config["feature_clip"],
        "iterations": int(fitted.n_iter_[0]), "positive_rows": int((labels[selected] == 1).sum()), "zero_target_rows": int((target == 0).sum()),
        "base_weight_sum": float(base.sum()), "economic_weight_sum": float(weights.sum()),
        "raw_absolute_advantage_mass": float((base*np.abs(target)).sum()),
        "positive_economic_weight": float(weights[labels == 1].sum()), "negative_economic_weight": float(weights[labels == 0].sum()),
        "maximum_state_weight": float(weights.max()), "score_is_calibrated_probability": False}
    score = expit(model["intercept"] + design@np.array(model["coefficients"]))
    require(np.allclose(score, fitted.predict_proba(design)[:, 1], rtol=0, atol=1e-14), "经济决策评分公式与拟合器不一致")
    return model


def economic_score(model, values):
    require(model["kind"] == "ECONOMIC_REGRET_LOGISTIC" and model["features"] == FEATURES, "经济模型或8因子身份不符")
    normalized = np.clip((np.asarray(values)-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    margin = float(model["intercept"] + normalized@np.array(model["coefficients"]))
    return float(expit(margin)), margin


def economic_chinese_formula(model):
    lines = [f"经济决策线性评分截距为{model['intercept']:.10f}。八项输入按原周期基础权重的训练均值和标准差标准化，限制在负5至5，各乘下列系数再与截距相加。取总分相反数的自然指数、加1，再以1除以上述数，得到零至一的经济持有评分；它不是实际胜率或预期收益。", "",
        f"拟合经济权重总和{model['economic_weight_sum']:.10f}，原基础权重总和{model['base_weight_sum']:.10f}；正差额经济权重{model['positive_economic_weight']:.10f}，负差额经济权重{model['negative_economic_weight']:.10f}，零差额状态{model['zero_target_rows']}条。", ""]
    lines.extend(f"- {name}：均值{mean:.10f}，标准差{scale:.10f}，系数{coefficient:.10f}。"
        for name, mean, scale, coefficient in zip(CN, model["mean"], model["scale"], model["coefficients"], strict=True))
    return lines


class EconomicExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.fit_indexes = [m["fit_index"] for m in models]
        require(self.fit_indexes == sorted(set(self.fit_indexes)), "经济模型时点必须唯一递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if cycle["cycle_id"] != self.cycle_id:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        values = state_values(self.data, t, cycle, current_value, peak_value)
        index = bisect_right(self.fit_indexes, t)-1
        stored = self.models[index] if index >= 0 else None
        score, margin, status = None, None, "NO_VIEW_NO_MATURE_MODEL"
        if stored and stored["status"] == "FIT_COMPLETE" and np.isfinite(values).all():
            require(stored["latest_exit_index"] <= stored["fit_index"] <= t, "经济模型使用未来周期或模型")
            score, margin = economic_score(stored["model"], values)
            require(np.isfinite(score) and np.isfinite(margin), "经济评分不是有限数")
            status = "PREDICTION_AVAILABLE"
        elif stored and stored["status"] == "FIT_COMPLETE":
            status = "NO_VIEW_INCOMPLETE_HOLDING_FEATURES"
        elif stored:
            status = stored["status"]
        self.negative_count = self.negative_count+1 if score is not None and score < .5 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": None, "continuation_probability": None,
            "economic_continuation_score": score, "economic_linear_margin": margin,
            "learning_fit_origin": self.data.date.iloc[stored["fit_index"]] if stored else pd.NaT,
            "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
            **dict(zip(FEATURES, values))}
