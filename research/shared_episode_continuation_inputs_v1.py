"""按共同持仓时段成熟、分层等权，学习三类自然参考的共享继续价值。"""
from bisect import bisect_right
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from research.learned_cycle_exit_v1 import FEATURES as OLD_FEATURES, CN as OLD_CN, state_values
from research.intraday_overnight_increment_v1 import require

SIGNALS = ["D60_INTRA", "S1_TREND_REBOUND", "R2_Z_CONFIRM"]
TASK_FEATURES = ["task_trend_rebound", "task_z_recovery"]
FEATURES = OLD_FEATURES+TASK_FEATURES
CN = OLD_CN+["来源为趋势反弹参考", "来源为均值偏离参考"]


def calendar_episodes(ledgers, terminal_date):
    require(set(ledgers) == set(SIGNALS), "共享训练必须有三类原参考")
    dates = pd.DatetimeIndex(ledgers[SIGNALS[0]].date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "参考日历必须唯一递增")
    holdings = []
    for signal in SIGNALS:
        ledger = ledgers[signal]
        require(pd.DatetimeIndex(ledger.date).equals(dates), "三个参考缺少相同完整日历")
        require(np.isfinite(ledger.shares).all() and ledger.shares.ge(0).all(), "每日参考持仓未知，不能当作空仓")
        holdings.append(ledger.shares.gt(0).to_numpy())
    union = np.column_stack(holdings).any(axis=1)
    active, groups, mapped = None, [], []
    for date, holding in zip(dates, union):
        if holding and active is None:
            active = len(groups)+1
            groups.append({"group_id": active, "group_start_date": date, "group_mature_date": pd.NaT})
        if not holding and active is not None:
            if date != pd.Timestamp(terminal_date):
                groups[-1]["group_mature_date"] = date
            active = None
        mapped.append({"origin": date, "group_id": active})
    return pd.DataFrame(groups, columns=["group_id", "group_start_date", "group_mature_date"]), pd.DataFrame(mapped)


def grouped_samples(samples, assignments, groups):
    keys = ["signal", "cycle_id", "origin_index"]
    require(not samples.duplicated(keys).any() and set(samples.signal) <= set(SIGNALS), "原参考状态重复或任务未知")
    require(not assignments.origin.duplicated().any(), "共同持仓时段原点重复")
    result = samples.merge(assignments[["origin", "group_id"]], on="origin", how="left", validate="many_to_one", sort=False)
    require(len(result) == len(samples) and result.group_id.notna().all(), "原参考状态缺少共同持仓时段，禁止删行")
    result["group_id"] = result.group_id.astype(int)
    result = result.merge(groups, on="group_id", how="left", validate="many_to_one", sort=False)
    require(result.groupby(["signal", "cycle_id"]).group_id.nunique().eq(1).all(), "原自然周期跨越共同空仓边界")
    known = result.group_mature_date.notna()
    require((result.loc[known, "mature_date"] <= result.loc[known, "group_mature_date"]).all(), "共同时段早于内部自然周期成熟")
    result["source_cycle_key"] = result.signal+"::"+result.cycle_id.astype(str)
    result[TASK_FEATURES[0]] = result.signal.eq("S1_TREND_REBOUND").astype(float)
    result[TASK_FEATURES[1]] = result.signal.eq("R2_Z_CONFIRM").astype(float)
    pd.testing.assert_frame_equal(result[samples.columns].reset_index(drop=True), samples.reset_index(drop=True))
    return result


def shared_training_rows(samples, origin, cfg):
    eligible = samples[samples.group_mature_date.notna() & samples.group_mature_date.le(pd.Timestamp(origin))]
    episodes = eligible[["group_id", "group_mature_date"]].drop_duplicates().sort_values(["group_mature_date", "group_id"])
    ids = episodes.tail(cfg["recent_groups"]).group_id.astype(int).to_list()
    rows = eligible[eligible.group_id.isin(ids)].sort_values(["group_id", "signal", "cycle_id", "origin_index"]).copy()
    if len(rows):
        cycle_count = rows.groupby("group_id").source_cycle_key.transform("nunique")
        state_count = rows.groupby(["group_id", "source_cycle_key"]).origin_index.transform("count")
        rows["sample_weight"] = 1./cycle_count/state_count
    else:
        rows["sample_weight"] = pd.Series(dtype=float)
    return rows, ids


def support_status(rows, groups, cfg):
    return len(groups) >= cfg["minimum_groups"] and len(rows) >= cfg["minimum_rows"] and set(rows.signal) == set(SIGNALS)


def fit_shared_exit(rows, cfg):
    x = rows[FEATURES].to_numpy(float)
    y, weights = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "共享训练完整时段存在未知输入或目标，禁止删行")
    require(np.isfinite(weights).all() and (weights > 0).all(), "共同时段分层权重不完整")
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale = np.where(scale > 1e-12, scale, 1.)
    design = np.clip((x-mean)/scale, -cfg["feature_clip"], cfg["feature_clip"])
    fitted = Ridge(alpha=cfg["ridge_alpha"], solver="svd", fit_intercept=True)
    fitted.fit(design, y, sample_weight=weights)
    require(np.isfinite(fitted.coef_).all() and np.isfinite(fitted.intercept_), "共享继续价值出现非有限系数")
    return {"kind": "SHARED_EPISODE_CONTINUATION_RIDGE", "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
            "coefficients": fitted.coef_.tolist(), "intercept": float(fitted.intercept_), "feature_clip": cfg["feature_clip"]}


def shared_prediction(model, values):
    require(model["kind"] == "SHARED_EPISODE_CONTINUATION_RIDGE" and model["features"] == FEATURES, "共享继续价值模型身份不符")
    x = np.asarray(values, float)
    require(x.shape == (10,) and np.isfinite(x).all(), "共享继续价值需要完整十项输入")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"]+z@np.asarray(model["coefficients"]))


class SharedEpisodeExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "共享模型原点必须唯一递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        x = np.r_[state_values(self.data, t, cycle, current_value, peak_value), 0., 0.]
        k = bisect_right(self.indexes, t)-1
        record = self.models[k] if k >= 0 else None
        value, status = None, "NO_VIEW_NO_MATURE_SHARED_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["latest_group_mature_index"] <= record["fit_index"] <= t, "共享模型使用未成熟共同时段或未来模型")
            if np.isfinite(x).all():
                value, status = shared_prediction(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_CURRENT_FEATURES"
        elif record and record["status"] in {"NO_VIEW_MODEL_FIT_FAILED", "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"}:
            status = record["status"]
        self.negative_count = self.negative_count+1 if value is not None and value < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": value,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
                "prediction_task": "D60_INTRA", **dict(zip(FEATURES, x))}
