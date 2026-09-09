"""复用已保存周期内系数，以买入前状态估计新周期截距。"""
from bisect import bisect_right
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from research.learned_cycle_exit_v1 import FEATURES, CN, state_values
from research.intraday_overnight_increment_v1 import require

CONTEXT_FEATURES = ["entry_mom20", "entry_sma120", "entry_vol20"]
CONTEXT_CN = ["买入前二十日涨跌", "买入前一百二十日均线偏离", "买入前二十日波动"]


def entry_values(data, entry_index, entry_origin=None):
    origin = int(entry_index)-1
    require(0 <= origin < len(data), "买入前完整日历位置无效")
    require(entry_origin is None or pd.Timestamp(entry_origin) == data.date.iloc[origin], "实际买入请求不是前一收盘")
    return data.loc[origin, ["mom20", "sma120", "vol20"]].to_numpy(float)


def cycle_entry_context(data, cycles):
    require(cycles.cycle_id.is_unique, "原参考周期编号不唯一")
    rows = []
    for cycle in cycles.itertuples():
        x = entry_values(data, cycle.entry_index, cycle.entry_origin)
        rows.append({"cycle_id": int(cycle.cycle_id), "entry_origin_index": int(cycle.entry_index)-1,
                     "entry_origin": pd.Timestamp(cycle.entry_origin), "exit_date": pd.Timestamp(cycle.exit_date),
                     **dict(zip(CONTEXT_FEATURES, x))})
    return pd.DataFrame(rows).set_index("cycle_id", drop=False)


def select_context(record, context):
    require(record["status"] == "FIT_COMPLETE" and record["latest_exit_index"] <= record["fit_index"], "周期上下文不能使用未来周期或无模型")
    groups = record["model"]["cycle_intercepts"]
    ids = record["training_cycles"]
    require(len(ids) == len(set(ids)) and set(ids) == {g["cycle_id"] for g in groups}, "周期内模型与上下文成员不同")
    selected = context.loc[ids].copy()
    require(selected.exit_date.le(pd.Timestamp(record["fit_origin"])).all() and selected.entry_origin_index.lt(record["fit_index"]).all(), "上下文训练周期尚未完成")
    targets = {g["cycle_id"]: g["cycle_intercept"] for g in groups}
    selected["target_intercept"] = [targets[cycle] for cycle in ids]
    return selected


def fit_entry_context_intercept(record, context, cfg):
    selected = select_context(record, context)
    x, y = selected[CONTEXT_FEATURES].to_numpy(float), selected.target_intercept.to_numpy(float)
    require(np.isfinite(x).all() and np.isfinite(y).all(), "进入上下文或截距缺失，禁止删周期")
    mean, scale = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = np.clip((x-mean)/scale, -cfg["feature_clip"], cfg["feature_clip"])
    regression = Ridge(alpha=cfg["context_ridge_alpha"], solver="svd", fit_intercept=True).fit(z, y)
    require(np.isfinite(regression.coef_).all() and np.isfinite(regression.intercept_), "进入上下文求解结果无效")
    return {"kind": "ENTRY_CONTEXT_PLUS_SAVED_WITHIN_CYCLE", "within_model": deepcopy(record["model"]),
            "context_model": {"features": CONTEXT_FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
                              "coefficients": regression.coef_.tolist(), "intercept": float(regression.intercept_),
                              "feature_clip": cfg["feature_clip"], "training_cycles": record["training_cycles"].copy(),
                              "target": "SAVED_MATURE_CYCLE_INTERCEPT", "cycle_weight": 1.}}


def entry_context_prediction(model, current_values, entry_context):
    require(model["kind"] == "ENTRY_CONTEXT_PLUS_SAVED_WITHIN_CYCLE", "两层退出预测模型身份不同")
    inside, between = model["within_model"], model["context_model"]
    require(inside["features"] == FEATURES and between["features"] == CONTEXT_FEATURES, "两层预测的因子定义不同")
    x, e = np.asarray(current_values, float), np.asarray(entry_context, float)
    require(x.shape == (8,) and e.shape == (3,) and np.isfinite(x).all() and np.isfinite(e).all(), "两层预测需要完整当前八因子及买入前三因子")
    a = float(between["intercept"]+np.clip((e-between["mean"])/between["scale"], -between["feature_clip"], between["feature_clip"])@np.asarray(between["coefficients"]))
    contribution = float(np.clip((x-inside["mean"])/inside["scale"], -inside["feature_clip"], inside["feature_clip"])@np.asarray(inside["coefficients"]))
    return a+contribution, a, contribution


class EntryContextInterceptController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "两层退出模型日期必须唯一递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        require(cycle["entry_index"] <= t, "实际持仓进入日期在未来")
        x = state_values(self.data, t, cycle, current_value, peak_value)
        e = entry_values(self.data, cycle["entry_index"], cycle.get("entry_origin"))
        k = bisect_right(self.indexes, t)-1
        record = self.models[k] if k >= 0 else None
        score, intercept, contribution, status = None, None, None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= t, "两层退出读取未来周期或模型")
            if not np.isfinite(e).all():
                status = "NO_VIEW_INCOMPLETE_ENTRY_CONTEXT"
            elif not np.isfinite(x).all():
                status = "NO_VIEW_INCOMPLETE_EIGHT_FEATURES"
            else:
                score, intercept, contribution = entry_context_prediction(record["model"], x, e)
                status = "PREDICTION_AVAILABLE"
        elif record and record["status"] in {"NO_VIEW_MODEL_FIT_FAILED", "NO_VIEW_INCOMPLETE_TRAINING_CONTEXT"}:
            status = record["status"]
        self.negative_count = self.negative_count+1 if score is not None and score < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": score,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "entry_context_origin": self.data.date.iloc[cycle["entry_index"]-1], "context_intercept_prediction": intercept,
                "within_cycle_component": contribution, "negative_confirmation_count": self.negative_count,
                "learned_exit_requested": self.negative_count >= self.confirmation_days,
                **dict(zip(FEATURES, x)), **dict(zip(CONTEXT_FEATURES, e))}
