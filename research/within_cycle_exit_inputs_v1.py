"""固定周期截距吸收共同目标水平，原八项系数从周期内变化学习。"""
from bisect import bisect_right
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from research.learned_cycle_exit_v1 import FEATURES, CN, state_values
from research.intraday_overnight_increment_v1 import require


def fit_within_cycle_exit(rows, cfg):
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
    return {"kind": "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE", "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": fitted.coef_.tolist(), "intercept": intercept, "feature_clip": cfg["feature_clip"], "cycle_intercepts": groups,
            "new_cycle_intercept_rule": "EQUAL_MEAN_OF_MATURE_TRAINING_CYCLE_INTERCEPTS"}


def within_cycle_prediction(model, values):
    require(model["kind"] == "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE" and model["features"] == FEATURES, "周期内预测模型身份不同")
    x = np.asarray(values, float)
    require(x.shape == (8,) and np.isfinite(x).all(), "周期内预测需要原完整八项状态")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"]+z@np.asarray(model["coefficients"]))


class WithinCycleExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "周期内模型日期必须唯一递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        x = state_values(self.data, t, cycle, current_value, peak_value)
        k = bisect_right(self.indexes, t)-1
        record = self.models[k] if k >= 0 else None
        value, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= t, "周期内退出读取未来周期或模型")
            if np.isfinite(x).all():
                value, status = within_cycle_prediction(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_EIGHT_FEATURES"
        elif record and record["status"] in {"NO_VIEW_MODEL_FIT_FAILED", "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"}:
            status = record["status"]
        self.negative_count = self.negative_count+1 if value is not None and value < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": value,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
                **dict(zip(FEATURES, x))}
