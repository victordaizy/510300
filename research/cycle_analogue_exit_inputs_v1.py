"""从不同成熟参考周期各选一个相似状态，平均五个邻居的继续收益。"""
from bisect import bisect_right
import json
import numpy as np
from research.market_path_exit_inputs_v1 import FEATURES, CN
from research.market_path_state_v1 import MarketPathTracker
from research.learned_cycle_exit_v1 import state_values
from research.intraday_overnight_increment_v1 import require


def fit_cycle_analogue(rows, cfg):
    ordered = rows.sort_values(["cycle_id", "origin_index"]).copy()
    require(len(ordered) > 0 and not ordered.duplicated(["cycle_id", "origin_index"]).any(), "相似周期训练为空或状态重复")
    x = ordered[FEATURES].to_numpy(float)
    y, w = ordered.target.to_numpy(float), ordered.sample_weight.to_numpy(float)
    require(np.isfinite(x).all() and np.isfinite(y).all(), "相似周期训练必要输入缺失，禁止删行")
    require(np.isfinite(w).all() and (w > 0).all(), "相似周期训练权重无效")
    require(np.allclose(ordered.groupby("cycle_id").sample_weight.sum(), 1., atol=1e-12, rtol=0), "每个完整周期总权重必须为一")
    require(ordered.cycle_id.nunique() >= cfg["distinct_cycle_neighbors"], "不同成熟周期不足固定邻居数")
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    scale = np.where(scale > 1e-12, scale, 1.)
    z = np.clip((x-mean)/scale, -cfg["feature_clip"], cfg["feature_clip"])
    return {"kind": "DISTINCT_CYCLE_NEAREST_ANALOGUE", "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
            "feature_clip": cfg["feature_clip"], "neighbors": cfg["distinct_cycle_neighbors"], "standardized_inputs": z.tolist(),
            "targets": y.tolist(), "cycle_ids": ordered.cycle_id.to_list(), "origin_indices": ordered.origin_index.to_list(),
            "exit_indices": ordered.exit_index.to_list(), "selection_order": "SQUARED_DISTANCE_THEN_CYCLE_ID_THEN_ORIGIN_INDEX",
            "prediction_rule": "EQUAL_MEAN_OF_FIVE_DISTINCT_CYCLE_REPRESENTATIVES"}


def analogue_prediction(model, values):
    require(model["kind"] == "DISTINCT_CYCLE_NEAREST_ANALOGUE" and model["features"] == FEATURES, "相似周期模型身份不同")
    x = np.asarray(values, float)
    require(x.shape == (8,) and np.isfinite(x).all(), "相似周期预测需要完整八项状态")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    training = np.asarray(model["standardized_inputs"], float)
    distances = np.sum((training-z)**2, axis=1)
    cycles, origins = np.asarray(model["cycle_ids"], int), np.asarray(model["origin_indices"], int)
    order = np.lexsort((origins, cycles, distances))
    chosen, seen = [], set()
    for i in order:
        cycle = int(cycles[i])
        if cycle not in seen:
            chosen.append(int(i))
            seen.add(cycle)
            if len(chosen) == model["neighbors"]:
                break
    require(len(chosen) == model["neighbors"], "合法不同周期邻居不足，不能重复同周期补齐")
    targets = np.asarray(model["targets"], float)[chosen]
    value = float(targets.mean())
    require(np.isfinite(value), "相似周期继续收益不是有限值")
    return value, {"analogue_cycle_ids": json.dumps(cycles[chosen].tolist()), "analogue_origin_indices": json.dumps(origins[chosen].tolist()),
                   "analogue_squared_distances": json.dumps(distances[chosen].tolist()), "analogue_targets": json.dumps(targets.tolist()),
                   "analogue_count": len(chosen)}


class CycleAnalogueExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.market_path = MarketPathTracker(data)
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "相似周期模型日期必须唯一递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        x = np.asarray(state_values(self.data, t, cycle, current_value, peak_value), dtype=float)
        account_return, account_drawdown = float(x[1]), float(x[2])
        market = self.market_path.observe(t, cycle, current_value)
        x[1:3] = [market["market_cycle_return"], market["market_cycle_drawdown"]]
        k = bisect_right(self.indexes, t)-1
        record = self.models[k] if k >= 0 else None
        value, status, neighbors = None, "NO_VIEW_NO_MATURE_MODEL", {}
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= t, "相似周期退出读取未来周期或模型")
            if np.isfinite(x).all():
                value, neighbors = analogue_prediction(record["model"], x)
                status = "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_EIGHT_FEATURES"
        elif record:
            status = record["status"]
        self.negative_count = self.negative_count+1 if value is not None and value < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": value,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
                "account_cycle_return": account_return, "account_cycle_drawdown": account_drawdown, **market, **dict(zip(FEATURES, x)), **neighbors}
