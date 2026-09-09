"""原八项继续持有因子加一项过去成交额流动性代理。"""
from __future__ import annotations
from bisect import bisect_right
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from research.intraday_overnight_increment_v1 import require
from research.learned_cycle_exit_v1 import FEATURES as ORIGINAL_FEATURES, CN as ORIGINAL_CN, state_values

LIQUIDITY = "log_amihud_20_cny_100m"
FEATURES = ORIGINAL_FEATURES + [LIQUIDITY]
CN = ORIGINAL_CN + ["二十日成交额流动性代理对数"]


def liquidity_features(data):
    require(data.index.equals(pd.RangeIndex(len(data))), "日行情索引必须完整递增")
    require(data.date.is_monotonic_increasing and not data.date.duplicated().any(), "交易日期重复或不递增")
    require("amount_unit" in data and data.amount_unit.eq("CNY").all(), "成交额必须明确为人民币元，禁止误乘一千")
    amount = pd.to_numeric(data.amount, errors="coerce").astype(float)
    returns = pd.to_numeric(data.total_simple, errors="coerce").astype(float)
    valid = np.isfinite(amount) & amount.gt(0) & np.isfinite(returns)
    daily = (returns.abs() / (amount / 100000000.)).where(valid)
    mean = daily.rolling(20, min_periods=20).mean()
    factor = np.log(mean.where(mean.gt(0)))
    return pd.DataFrame({"date": data.date, "amount_cny": amount, "daily_return": returns,
        "daily_illiquidity_per_100m_cny": daily, "mean_illiquidity_20d": mean, LIQUIDITY: factor,
        "liquidity_status": np.where(np.isfinite(factor), "AVAILABLE", "NO_VIEW_INCOMPLETE_OR_ZERO_20D_LIQUIDITY")})


def attach_sample_liquidity(samples, data):
    values = samples.origin_index.to_numpy()
    require(np.equal(values, values.astype(int)).all() and (values >= 0).all() and (values < len(data)).all(), "参考样本原点不合法")
    require(pd.DatetimeIndex(samples.origin).equals(pd.DatetimeIndex(data.date.iloc[values])), "参考样本日期和行情原点不一致")
    output = samples.copy()
    output[LIQUIDITY] = data[LIQUIDITY].iloc[values].to_numpy()
    return output


def liquidity_values(data, t, cycle, current_value, peak_value):
    require(t >= cycle["entry_index"] and cycle["entry_cost_cny"] > 0 and peak_value > 0, "实际持仓时间、成本或最高价值不合法")
    return np.r_[state_values(data, t, cycle, current_value, peak_value), data[LIQUIDITY].iloc[t]]


def fit_liquidity_exit(rows, config):
    x = rows[FEATURES].to_numpy(float)
    y, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "九项训练输入或原目标不完整，禁止删除部分周期样本")
    require(np.isfinite(weights).all() and (weights > 0).all(), "周期等权权重不完整")
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.clip((x - mean) / scale, -config["feature_clip"], config["feature_clip"])
    fitted = Ridge(alpha=config["ridge_alpha"], solver="svd", fit_intercept=True)
    fitted.fit(design, y, sample_weight=weights)
    if not (np.isfinite(fitted.coef_).all() and np.isfinite(fitted.intercept_)):
        raise FloatingPointError("九项模型出现非有限系数")
    return {"kind": "LIQUIDITY_INCREMENT_RIDGE", "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
        "coefficients": fitted.coef_.tolist(), "intercept": float(fitted.intercept_), "feature_clip": config["feature_clip"]}


def liquidity_prediction(model, values):
    require(model["kind"] == "LIQUIDITY_INCREMENT_RIDGE" and model["features"] == FEATURES, "九项模型身份不符")
    normalized = np.clip((np.asarray(values) - model["mean"]) / model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"] + normalized @ np.asarray(model["coefficients"]))


def liquidity_chinese_formula(model):
    lines = [f"预测继续持有相对提前退出收益的截距为{model['intercept']:.10f}。每项先减下列训练均值、除训练标准差，限制在负5至正5，再乘系数；九项结果与截距相加。", ""]
    lines.extend(f"- {name}：训练均值{mean:.10f}，训练标准差{scale:.10f}，标准化后的系数{coefficient:.10f}。"
        for name, mean, scale, coefficient in zip(CN, model["mean"], model["scale"], model["coefficients"], strict=True))
    return lines


class LiquidityExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.fit_indexes = [r["fit_index"] for r in models]
        require(self.fit_indexes == sorted(set(self.fit_indexes)), "九项模型月度时点必须唯一且递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        values = liquidity_values(self.data, t, cycle, current_value, peak_value)
        index = bisect_right(self.fit_indexes, t) - 1
        stored = self.models[index] if index >= 0 else None
        estimate, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if stored and stored["status"] == "FIT_COMPLETE" and np.isfinite(values).all():
            require(stored["latest_exit_index"] <= stored["fit_index"] <= t, "九项退出模型使用未来周期或模型")
            estimate, status = liquidity_prediction(stored["model"], values), "PREDICTION_AVAILABLE"
            require(np.isfinite(estimate), "九项模型预测不是有限数")
        elif stored and stored["status"] == "FIT_COMPLETE":
            status = "NO_VIEW_INCOMPLETE_NINE_FEATURES"
        elif stored and stored["status"] in {"NO_VIEW_MODEL_FIT_FAILED", "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"}:
            status = stored["status"]
        self.negative_count = self.negative_count + 1 if estimate is not None and estimate < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": estimate,
            "learning_fit_origin": self.data.date.iloc[stored["fit_index"]] if stored else pd.NaT,
            "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
            **dict(zip(FEATURES, values))}
