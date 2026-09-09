"""原周期内模型加入有符号成交份额，九项系数从完整成熟周期内部变化学习。"""
from bisect import bisect_right
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from research.learned_cycle_exit_v1 import FEATURES as BASE_FEATURES, CN as BASE_CN, state_values

FEATURES = BASE_FEATURES+["signed_volume_balance20"]
CN = BASE_CN+["二十日有符号成交份额占比"]
from research.intraday_overnight_increment_v1 import require


def fit_signed_volume_within(rows, cfg):
    x = rows[FEATURES].to_numpy(float)
    y, w = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "周期内训练输入缺失，禁止删行")
    require(np.isfinite(w).all() and (w > 0).all(), "周期内训练权重无效")
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    scale = np.where(scale > 1e-12, scale, 1.)
    z = np.clip((x-mean)/scale, -cfg["feature_clip"], cfg["feature_clip"])
    dx, dy = np.empty_like(z), np.empty_like(y)
    groups = []
    ids = rows.cycle_id.to_numpy(int)
    for cycle in sorted(set(ids)):
        mask = ids == cycle
        require(abs(w[mask].sum()-1.) < 1e-12, "每个原完整周期必须等总权重一")
        mz, my = np.average(z[mask], axis=0, weights=w[mask]), float(np.average(y[mask], weights=w[mask]))
        dx[mask], dy[mask] = z[mask]-mz, y[mask]-my
        groups.append({"cycle_id": int(cycle), "rows": int(mask.sum()), "standardized_feature_mean": mz.tolist(), "target_mean": my})
    fitted = Ridge(alpha=cfg["ridge_alpha"], solver="svd", fit_intercept=False)
    fitted.fit(dx, dy, sample_weight=w)
    require(np.isfinite(fitted.coef_).all(), "周期内模型系数不是有限数值")
    for group in groups:
        group["cycle_intercept"] = float(group["target_mean"]-np.asarray(group["standardized_feature_mean"])@fitted.coef_)
    intercept = float(np.mean([g["cycle_intercept"] for g in groups]))
    return {"kind": "SIGNED_VOLUME_WITHIN_CYCLE_RIDGE", "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": fitted.coef_.tolist(), "intercept": intercept, "feature_clip": cfg["feature_clip"], "cycle_intercepts": groups,
            "new_cycle_intercept_rule": "EQUAL_MEAN_OF_MATURE_TRAINING_CYCLE_INTERCEPTS"}


def signed_volume_prediction(model, values):
    require(model["kind"] == "SIGNED_VOLUME_WITHIN_CYCLE_RIDGE" and model["features"] == FEATURES, "周期内预测模型身份不同")
    x = np.asarray(values, float)
    require(x.shape == (9,) and np.isfinite(x).all(), "周期内预测需要完整九项状态")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"]+z@np.asarray(model["coefficients"]))


class SignedVolumeWithinController:
    def __init__(self, data, models, factors, confirmation_days=2):
        self.data, self.models, self.factors = data, models, factors
        require(pd.DatetimeIndex(data.date).equals(pd.DatetimeIndex(factors.date)), "成交份额因子与实际账户日历不同")
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "周期内模型日期必须唯一递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        x = np.r_[state_values(self.data, t, cycle, current_value, peak_value), float(self.factors.signed_volume_balance20.iloc[t])]
        k = bisect_right(self.indexes, t)-1
        record = self.models[k] if k >= 0 else None
        value, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= t, "周期内退出读取未来周期或模型")
            if np.isfinite(x).all():
                value, status = signed_volume_prediction(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_NINE_FEATURES"
        elif record and record["status"] in {"NO_VIEW_MODEL_FIT_FAILED", "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"}:
            status = record["status"]
        self.negative_count = self.negative_count+1 if value is not None and value < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": value,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
                **dict(zip(FEATURES, x))}
