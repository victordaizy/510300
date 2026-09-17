"""以成熟周期等权估计八个因素的单调曲线，固定每笔退出模型。"""
import copy
import hashlib
import json
from bisect import bisect_right
import numpy as np
import pandas as pd
from research.learned_cycle_exit_v1 import FEATURES, state_values, training_rows
from research.intraday_overnight_increment_v1 import require

KIND = "CYCLE_EQUAL_EIGHT_MARGINAL_MONOTONE_MEAN"
PREDICTION_KEYS = ["kind", "features", "mean", "scale", "feature_clip", "curves", "curve_weights"]


def weighted_moments(x, weights):
    x = np.asarray(x, float)
    mean = np.average(x, axis=0, weights=weights)
    constant = np.ptp(x, axis=0) == 0.
    mean = np.where(constant, x[0], mean)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    variance = np.where(constant, 0., variance)
    require(np.isfinite(mean).all() and np.isfinite(variance).all() and (variance >= 0).all(), "加权均值或方差无效")
    return mean, variance


def standardized_inputs(rows, cfg):
    x = rows[FEATURES].to_numpy(float)
    y, w = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "八曲线训练输入缺失，禁止删行")
    require(np.isfinite(w).all() and (w > 0).all(), "八曲线训练权重无效")
    require(not rows.duplicated(["cycle_id", "origin_index"]).any(), "训练周期原点重复")
    require((rows.origin_index < rows.exit_index).all(), "训练状态必须早于原自然退出")
    for cycle in sorted(set(rows.cycle_id)):
        mask = rows.cycle_id.eq(cycle).to_numpy()
        require((np.diff(rows.origin_index.to_numpy()[mask]) > 0).all(), "周期原点必须严格递增")
        require(np.allclose(w[mask], 1. / mask.sum(), atol=1e-14, rtol=0), "每周期总权重必须为一且状态等权")
    mean, variance = weighted_moments(x, w)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = np.clip((x - mean) / scale, -cfg["feature_clip"], cfg["feature_clip"])
    return z, y, w, mean, scale


def weighted_pava_curve(x, y, weights, direction):
    require(direction in [-1, 1], "保序方向必须为已确定的上升或下降")
    x, y, weights = np.asarray(x, float), np.asarray(y, float), np.asarray(weights, float)
    require(x.ndim == 1 and x.shape == y.shape == weights.shape and len(x) > 0, "保序输入维度不同")
    require(np.isfinite(x).all() and np.isfinite(y).all() and np.isfinite(weights).all() and (weights > 0).all(), "保序输入或权重无效")
    unique, inverse = np.unique(x, return_inverse=True)
    support_weight, support_sum = np.zeros(len(unique)), np.zeros(len(unique))
    np.add.at(support_weight, inverse, weights)
    np.add.at(support_sum, inverse, weights*y*direction)
    blocks = []
    for i in range(len(unique)):
        blocks.append([i, i+1, float(support_weight[i]), float(support_sum[i])])
        while len(blocks) >= 2 and blocks[-2][3]/blocks[-2][2] > blocks[-1][3]/blocks[-1][2]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append([left[0], right[1], left[2]+right[2], left[3]+right[3]])
    fitted = np.empty(len(unique))
    for first, last, weight, value in blocks:
        fitted[first:last] = direction*value/weight
    require(np.isfinite(fitted).all() and (np.diff(fitted)*direction >= 0).all(), "保序输出无效或违反方向")
    keep = np.ones(len(unique), bool)
    if len(unique) > 2:
        keep[1:-1] = (fitted[1:-1] != fitted[:-2]) | (fitted[1:-1] != fitted[2:])
    return {"x": unique[keep].tolist(), "y": fitted[keep].tolist(), "support_count": len(unique), "block_count": len(blocks)}


def fit_marginal_monotone(rows, cfg):
    require(cfg["curve_weight"] == .125 and cfg["direction_rule"] == "SIGN_OF_CYCLE_WEIGHTED_COVARIANCE", "曲线固定等权或方向规则改变")
    z, y, w, mean, scale = standardized_inputs(rows, cfg)
    target_mean = float(np.average(y, weights=w))
    feature_mean, _ = weighted_moments(z, w)
    covariances = np.average((z-feature_mean)*(y-target_mean)[:, None], axis=0, weights=w)
    require(np.isfinite(covariances).all(), "训练因素与目标协方差无效")
    curves, solves = [], 0
    for j, name in enumerate(FEATURES):
        x = z[:, j]
        constant = np.ptp(x) == 0.
        c = 0. if constant else float(covariances[j])
        if constant or c == 0.:
            curve = {"x": [float(x.min())], "y": [target_mean], "support_count": len(np.unique(x)), "block_count": 1,
                "direction": 0, "constant_reason": "CONSTANT_INPUT" if constant else "EXACT_ZERO_COVARIANCE"}
        else:
            direction = 1 if c > 0. else -1
            curve = weighted_pava_curve(x, y, w, direction)
            curve.update(direction=direction, constant_reason=None)
            solves += 1
        curve.update(feature=name, weighted_covariance=c, training_weighted_mse=float(np.average((y-np.interp(x, curve["x"], curve["y"]))**2, weights=w)))
        curves.append(curve)
    return {"kind": KIND, "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
        "feature_clip": cfg["feature_clip"], "curves": curves, "curve_weights": [.125]*8,
        "target_mean": target_mean, "pava_solves": solves, "constant_curves": 8-solves,
        "curve_count": 8, "total_saved_knots": sum(len(c["x"]) for c in curves),
        "prediction_interpretation": "EQUAL_MEAN_OF_EIGHT_UNIVARIATE_CONTINUATION_ESTIMATES"}


def input_identity(rows, cfg):
    settings = {"kind": KIND, "features": FEATURES, **{key: cfg[key] for key in ["feature_clip", "curve_weight", "direction_rule"]}}
    h = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode())
    h.update(rows[["cycle_id", "origin_index", "exit_index"]].to_numpy(dtype="<i8").tobytes())
    h.update(rows[FEATURES + ["target", "sample_weight"]].to_numpy(dtype="<f8").tobytes())
    return h.hexdigest()


def prediction_identity(record):
    if record is None or record["model"] is None:
        return "NO_MODEL"
    content = {key: record["model"][key] for key in PREDICTION_KEYS}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_monthly_models(samples, originals, cfg):
    require([r["fit_index"] for r in originals] == sorted(set(r["fit_index"] for r in originals)), "月度时点必须唯一递增")
    records, diagnostics, membership, cache = [], [], [], {}
    attempts = successful = reused = 0
    for source in originals:
        t = int(source["fit_index"])
        require(pd.Timestamp(source["fit_time"]) == pd.Timestamp(source["fit_origin"]) + pd.Timedelta(hours=15, minutes=5), "原月度收盘时钟不同")
        rows, ids = training_rows(samples, t, cfg)
        require(ids == source["training_cycles"] and len(rows) == source["training_rows"], "原完整训练成员改变")
        eligible = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        require(eligible == (source["status"] == "FIT_COMPLETE"), "原月度训练样本支持状态改变")
        require(not len(rows) or rows.exit_index.le(t).all(), "训练包含未来未结束周期")
        record = {key: copy.deepcopy(source[key]) for key in ["fit_index", "fit_origin", "fit_time", "training_cycles",
            "training_cycle_count", "training_rows", "latest_exit_index", "latest_exit_date"]}
        record.update(status="NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS", model=None, failure=None, source_input_identity=None,
            first_attempt_fit_index=None, parameter_first_fit_index=None, reused_previous_input=False)
        if eligible:
            identity = input_identity(rows, cfg)
            was_seen = identity in cache
            if not was_seen:
                item = {"first_attempt_fit_index": t, "parameter_first_fit_index": None, "model": None,
                        "status": "NO_VIEW_MODEL_FIT_FAILED", "failure": None}
                attempts += 1
                try:
                    item["model"] = fit_marginal_monotone(rows, cfg)
                    item.update(status="FIT_COMPLETE", parameter_first_fit_index=t)
                    successful += 1
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                    item["failure"] = str(error)
                cache[identity] = item
            else:
                reused += 1
            item = cache[identity]
            require(item["first_attempt_fit_index"] <= t, "相同输入缓存来自未来")
            record.update(copy.deepcopy(item))
            record.update(source_input_identity=identity, reused_previous_input=was_seen)
        record["fixed_prediction_identity"] = prediction_identity(record)
        model = record["model"]
        diagnostics.append({key: record[key] for key in ["fit_index", "fit_origin", "status", "failure", "source_input_identity",
            "first_attempt_fit_index", "parameter_first_fit_index", "reused_previous_input", "fixed_prediction_identity"]} | {
            "pava_solves": model["pava_solves"] if model else None,
            "constant_curves": model["constant_curves"] if model else None,
            "total_saved_knots": model["total_saved_knots"] if model else None})
        records.append(record)
        membership.extend({"fit_index": t, "cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index), "exit_index": int(r.exit_index),
                           "sample_weight": r.sample_weight, "fit_status": record["status"]} for r in rows.itertuples())
    counts = {"monthly_records": len(records), "eligible_monthly_records": sum(r["source_input_identity"] is not None for r in records),
        "available_monthly_records": sum(r["status"] == "FIT_COMPLETE" for r in records), "distinct_source_input_sets": len(cache),
        "new_model_fits": attempts, "successful_distinct_fits": successful, "failed_fits": attempts - successful,
        "reused_monthly_fits": reused,
        "new_scalar_curve_solves": sum(v["model"]["pava_solves"] for v in cache.values() if v["model"] is not None),
        "constant_curve_records": sum(v["model"]["constant_curves"] for v in cache.values() if v["model"] is not None),
        "curve_records": sum(v["model"]["curve_count"] for v in cache.values() if v["model"] is not None)}
    return records, pd.DataFrame(diagnostics), pd.DataFrame(membership), counts


def marginal_monotone_prediction(model, values):
    require(model["kind"] == KIND and model["features"] == FEATURES and model["curve_weights"] == [.125]*8, "八曲线退出模型身份不同")
    x = np.asarray(values, float)
    require(x.shape == (8,) and np.isfinite(x).all(), "八曲线预测需要完整八因素")
    z = np.clip((x-model["mean"])/model["scale"], -model["feature_clip"], model["feature_clip"])
    estimates = []
    for j, (name, curve) in enumerate(zip(FEATURES, model["curves"], strict=True)):
        require(name == curve["feature"] and len(curve["x"]) == len(curve["y"]) > 0, "曲线因素或断点维度不同")
        require(np.isfinite(curve["x"]).all() and np.isfinite(curve["y"]).all(), "曲线断点数值无效")
        estimates.append(float(np.interp(z[j], curve["x"], curve["y"])))
    require(len(estimates) == 8 and np.isfinite(estimates).all(), "八条曲线预测不完整")
    return {"prediction": float(np.mean(estimates)), **{f"curve_prediction_{name}": value for name, value in zip(FEATURES, estimates)}}


class MarginalMonotoneExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "八曲线退出月度时点必须唯一递增")
        self.confirmation_days = confirmation_days
        self.cycle_id, self.selection_index, self.record, self.identity, self.negative_count = None, None, None, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            require(t == cycle["entry_index"], "八曲线版本必须在实际买入日首次持仓收盘选择")
            self.cycle_id, self.selection_index = cycle["cycle_id"], t
            k = bisect_right(self.indexes, t) - 1
            self.record = self.models[k] if k >= 0 else None
            self.identity, self.negative_count = prediction_identity(self.record), 0
        require(self.selection_index == cycle["entry_index"] and t >= self.selection_index, "实际持仓模型选择起点改变")
        values = state_values(self.data, t, cycle, current_value, peak_value)
        record, prediction, status = self.record, None, "NO_VIEW_NO_MATURE_MODEL"
        detail = {key: None for key in [f"curve_prediction_{name}" for name in FEATURES]}
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= self.selection_index, "八曲线退出读取未来周期或模型")
            require(record["parameter_first_fit_index"] <= record["fit_index"], "八曲线参数来自未来缓存")
            if np.isfinite(values).all():
                try:
                    detail = marginal_monotone_prediction(record["model"], values)
                    prediction, status = detail.pop("prediction"), "PREDICTION_AVAILABLE"
                except (ValueError, FloatingPointError) as error:
                    status = "NO_VIEW_PREDICTION_NUMERIC_FAILURE"
                    detail["prediction_failure"] = str(error)
            else:
                status = "NO_VIEW_INCOMPLETE_EIGHT_FEATURES"
        elif record and record["status"] != "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS":
            status = record["status"]
        self.negative_count = self.negative_count + 1 if prediction is not None and prediction < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": prediction,
            "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
            "model_selection_index": self.selection_index, "model_selection_origin": self.data.date.iloc[self.selection_index],
            "fixed_prediction_identity": self.identity, "negative_confirmation_count": self.negative_count,
            "learned_exit_requested": self.negative_count >= self.confirmation_days, **detail, **dict(zip(FEATURES, values))}
