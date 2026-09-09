"""在原实际持仓状态上增加已知隔夜下行风险占比。"""
from bisect import bisect_right
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from research.learned_cycle_exit_v1 import FEATURES as OLD_FEATURES, CN as OLD_CN, state_values
from research.intraday_overnight_increment_v1 import require

ADDED = "overnight_downside_share20"
FEATURES = OLD_FEATURES+[ADDED]
CN = OLD_CN+["二十日隔夜下行风险占比"]


def downside_frame(data, window=20):
    require(isinstance(window, int) and window > 0, "隔夜风险窗口必须为正整数")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "隔夜风险日期必须唯一递增")
    night, day = data.overnight_log.to_numpy(float), data.intraday_log.to_numpy(float)
    valid = np.isfinite(night) & np.isfinite(day)
    squared_down = np.where(valid, np.minimum(night, 0.)**2, np.nan)
    squared_all = np.where(valid, night**2+day**2, np.nan)
    numerator = pd.Series(squared_down).rolling(window, min_periods=window).sum()
    denominator = pd.Series(squared_all).rolling(window, min_periods=window).sum()
    share = numerator/denominator.where(denominator > 0)
    require((share.dropna() >= -1e-12).all() and (share.dropna() <= 1.+1e-12).all(), "隔夜下行风险占比超出经济范围")
    status = np.where(share.notna(), "OVERNIGHT_DOWNSIDE_SHARE_AVAILABLE", "NO_VIEW_INCOMPLETE_OR_ZERO_SQUARED_RETURN_WINDOW")
    return pd.DataFrame({"date": dates, "downside_overnight_square_sum": numerator, "all_session_square_sum": denominator,
                         ADDED: share, "factor_status": status})


def attach_downside(samples, factors):
    indexes = samples.origin_index.to_numpy(int)
    require((indexes >= 0).all() and (indexes < len(factors)).all(), "参考状态索引不在隔夜因子日历")
    require(pd.DatetimeIndex(samples.origin).equals(pd.DatetimeIndex(factors.date.iloc[indexes])), "参考状态与隔夜因子日期不同")
    result = samples.copy()
    result[ADDED] = factors[ADDED].iloc[indexes].to_numpy(float)
    return result


def fit_downside_exit(rows, cfg):
    x = rows[FEATURES].to_numpy(float)
    y, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "隔夜风险退出完整训练样本缺失，禁止删行")
    require(np.isfinite(weights).all() and (weights > 0).all(), "原周期等权权重非法")
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.clip((x-mean)/scale, -cfg["feature_clip"], cfg["feature_clip"])
    fitted = Ridge(alpha=cfg["ridge_alpha"], solver="svd", fit_intercept=True)
    fitted.fit(design, y, sample_weight=weights)
    require(np.isfinite(fitted.coef_).all() and np.isfinite(fitted.intercept_), "隔夜风险退出出现非有限模型系数")
    return {"kind": "OVERNIGHT_DOWNSIDE_SHARE_RIDGE", "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": fitted.coef_.tolist(), "intercept": float(fitted.intercept_), "feature_clip": cfg["feature_clip"]}


def downside_prediction(model, values):
    require(model["kind"] == "OVERNIGHT_DOWNSIDE_SHARE_RIDGE" and model["features"] == FEATURES, "隔夜风险退出模型身份不同")
    x = np.asarray(values, float)
    require(x.shape == (9,) and np.isfinite(x).all(), "隔夜风险退出需要完整九项状态")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"]+z@np.asarray(model["coefficients"]))


class OvernightDownsideExitController:
    def __init__(self, data, models, confirmation_days=2, factors=None):
        self.data, self.models = data, models
        self.factors = downside_frame(data) if factors is None else factors
        require(pd.DatetimeIndex(self.factors.date).equals(pd.DatetimeIndex(data.date)), "实际持仓隔夜因子日历不符")
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "隔夜风险模型月度原点必须唯一递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        x = np.r_[state_values(self.data, t, cycle, current_value, peak_value), self.factors[ADDED].iloc[t]]
        k = bisect_right(self.indexes, t)-1
        record = self.models[k] if k >= 0 else None
        value, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= t, "隔夜风险退出使用未来周期或模型")
            if np.isfinite(x).all():
                value, status = downside_prediction(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_NINE_FEATURES"
        elif record and record["status"] in {"NO_VIEW_MODEL_FIT_FAILED", "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"}:
            status = record["status"]
        self.negative_count = self.negative_count+1 if value is not None and value < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": value,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
                "downside_factor_status": self.factors.factor_status.iloc[t], **dict(zip(FEATURES, x))}
