"""为原完整持仓状态增加距最近达到周期高点的时间，保持成熟训练口径。"""
from bisect import bisect_right
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from research.learned_cycle_exit_v1 import FEATURES as OLD_FEATURES, CN as OLD_CN, state_values
from research.intraday_overnight_increment_v1 import require

ADDED = "log_days_since_peak"
FEATURES = OLD_FEATURES+[ADDED]
CN = OLD_CN+["距最近达到含分红持仓高点的交易日数对数"]


class PeakAgeTracker:
    def __init__(self):
        self.cycle_id, self.last_origin, self.peak_origin = None, None, None

    def observe(self, cycle_id, entry_index, t, drawdown):
        require(t >= entry_index, "高点时间不能早于实际进入")
        if self.cycle_id != cycle_id:
            self.cycle_id, self.last_origin, self.peak_origin = cycle_id, entry_index-1, entry_index-1
        require(t > self.last_origin, "同一持仓高点时间必须逐日向前")
        if t != self.last_origin+1 or not np.isfinite(drawdown):
            self.peak_origin = None
        if np.isfinite(drawdown):
            require(drawdown <= 1e-12, "周期回撤不能高于已有最高价值")
            if drawdown >= 0:
                self.peak_origin = t
        self.last_origin = t
        age = t-self.peak_origin if self.peak_origin is not None else np.nan
        return {"days_since_peak": age, ADDED: float(np.log1p(age)), "peak_origin_index": self.peak_origin,
                "peak_age_status": "PEAK_AGE_AVAILABLE" if np.isfinite(age) else "NO_VIEW_INCOMPLETE_PEAK_HISTORY"}


def attach_peak_age(samples, decisions, cycles):
    require(not cycles.cycle_id.duplicated().any(), "原参考周期身份重复")
    entries = dict(zip(cycles.cycle_id.astype(int), cycles.entry_index.astype(int)))
    states = decisions[decisions.learning_cycle_id.notna()].copy()
    states["cycle_id"] = states.learning_cycle_id.astype(int)
    states = states.sort_values(["cycle_id", "origin_index"])
    require(not states.duplicated(["cycle_id", "origin_index"]).any(), "原参考持仓状态重复")
    tracker, ages = PeakAgeTracker(), []
    for row in states.itertuples():
        require(row.cycle_id in entries, "参考状态没有对应实际进入")
        value = tracker.observe(row.cycle_id, entries[row.cycle_id], int(row.origin_index), row.cycle_drawdown)
        ages.append({"cycle_id": row.cycle_id, "origin_index": int(row.origin_index), "origin": row.origin, **value})
    frame = pd.DataFrame(ages)
    require(not samples.duplicated(["cycle_id", "origin_index"]).any(), "原训练状态身份重复")
    attached = samples.merge(frame, on=["cycle_id", "origin_index", "origin"], how="left", validate="one_to_one", sort=False)
    require(len(attached) == len(samples), "新增高点时间改变原样本行数")
    require(attached.peak_age_status.notna().all(), "原训练状态缺少完整参考时间来源")
    return attached, frame


def fit_peak_age_exit(rows, cfg):
    x = rows[FEATURES].to_numpy(float)
    y, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "高点时间退出完整训练样本缺失，禁止删行")
    require(np.isfinite(weights).all() and (weights > 0).all(), "原周期等权权重非法")
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.clip((x-mean)/scale, -cfg["feature_clip"], cfg["feature_clip"])
    fitted = Ridge(alpha=cfg["ridge_alpha"], solver="svd", fit_intercept=True)
    fitted.fit(design, y, sample_weight=weights)
    require(np.isfinite(fitted.coef_).all() and np.isfinite(fitted.intercept_), "高点时间退出出现非有限模型系数")
    return {"kind": "PEAK_AGE_RIDGE", "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": fitted.coef_.tolist(), "intercept": float(fitted.intercept_), "feature_clip": cfg["feature_clip"]}


def peak_age_prediction(model, values):
    require(model["kind"] == "PEAK_AGE_RIDGE" and model["features"] == FEATURES, "高点时间退出模型身份不同")
    x = np.asarray(values, float)
    require(x.shape == (9,) and np.isfinite(x).all(), "高点时间退出需要完整九项状态")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"]+z@np.asarray(model["coefficients"]))


class PeakAgeExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "高点时间模型月度原点必须唯一递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0
        self.tracker = PeakAgeTracker()

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        original = state_values(self.data, t, cycle, current_value, peak_value)
        age = self.tracker.observe(cycle["cycle_id"], cycle["entry_index"], t, original[2])
        x = np.r_[original, age[ADDED]]
        k = bisect_right(self.indexes, t)-1
        record = self.models[k] if k >= 0 else None
        value, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= t, "高点时间退出使用未来周期或模型")
            if np.isfinite(x).all():
                value, status = peak_age_prediction(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_NINE_FEATURES"
        elif record and record["status"] in {"NO_VIEW_MODEL_FIT_FAILED", "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"}:
            status = record["status"]
        self.negative_count = self.negative_count+1 if value is not None and value < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": value,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
                **age, **dict(zip(FEATURES, x))}
