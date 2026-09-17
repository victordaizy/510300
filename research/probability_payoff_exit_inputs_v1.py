"""以成熟周期等权估计两类状态概率和盈亏幅度，固定每笔退出模型。"""
import copy
import hashlib
import json
from bisect import bisect_right
import numpy as np
import pandas as pd
from research.learned_cycle_exit_v1 import FEATURES, state_values, training_rows
from research.intraday_overnight_increment_v1 import require

KIND = "CYCLE_EQUAL_GAUSSIAN_PROBABILITY_TIMES_CLASS_PAYOFF"
PREDICTION_KEYS = ["kind", "features", "mean", "scale", "feature_clip", "classes", "active_features",
                   "class_priors", "class_means", "class_variances", "class_payoffs"]


class MissingClassError(ValueError):
    """成熟训练样本缺少占优或不占优类别，保持无观点。"""


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
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "概率幅度训练输入缺失，禁止删行")
    require(np.isfinite(w).all() and (w > 0).all(), "概率幅度训练权重无效")
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


def fit_probability_payoff(rows, cfg):
    require(cfg["variance_smoothing"] == 1e-9, "两类方差的固定稳定比例改变")
    z, y, w, mean, scale = standardized_inputs(rows, cfg)
    classes = (y > 0).astype(int)
    if set(classes) != {0, 1}:
        raise MissingClassError("成熟样本需要同时存在继续占优和不占优两类")
    global_mean, global_variance = weighted_moments(z, w)
    active = global_variance > 0.
    epsilon = float(global_variance.max() * cfg["variance_smoothing"])
    priors, means, variances, payoffs, weights, counts = [], [], [], [], [], []
    for label in [0, 1]:
        mask = classes == label
        class_mean, class_variance = weighted_moments(z[mask], w[mask])
        weight = float(w[mask].sum())
        priors.append(weight / w.sum())
        means.append(class_mean.tolist())
        variances.append(class_variance.tolist())
        payoffs.append(float(np.average(y[mask], weights=w[mask])))
        weights.append(weight)
        counts.append(int(mask.sum()))
    stable_variances = np.asarray(variances) + epsilon
    require(np.isfinite(stable_variances).all() and np.isfinite(payoffs).all(), "两类模型参数非有限")
    require((stable_variances[:, active] > 0).all(), "参与概率计算的因素类内方差非正")
    require(payoffs[0] <= 0 < payoffs[1] and abs(sum(priors)-1.) < 1e-12, "类别平均盈亏方向或先验比例错误")
    return {"kind": KIND, "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
        "feature_clip": cfg["feature_clip"], "classes": [0, 1], "active_features": active.tolist(),
        "global_standardized_mean": global_mean.tolist(), "global_standardized_variance": global_variance.tolist(),
        "variance_smoothing": cfg["variance_smoothing"], "variance_epsilon": epsilon,
        "class_priors": priors, "class_means": means, "class_raw_variances": variances,
        "class_variances": stable_variances.tolist(), "class_payoffs": payoffs,
        "class_weights": weights, "class_rows": counts, "total_weight": float(w.sum()),
        "active_factor_count": int(active.sum()), "all_constant_model": not bool(active.any()),
        "prediction_interpretation": "UNVALIDATED_CLASS_PROBABILITY_WEIGHTED_MEAN_NET_CONTINUATION_INCREMENT"}


def input_identity(rows, cfg):
    settings = {"kind": KIND, "features": FEATURES, "positive_class": "ORIGINAL_TARGET_STRICTLY_ABOVE_ZERO",
                **{key: cfg[key] for key in ["feature_clip", "variance_smoothing"]}}
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
                    item["model"] = fit_probability_payoff(rows, cfg)
                    item.update(status="FIT_COMPLETE", parameter_first_fit_index=t)
                    successful += 1
                except MissingClassError as error:
                    item.update(status="NO_VIEW_BOTH_CLASSES_REQUIRED", failure=str(error))
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
            "active_factor_count": model["active_factor_count"] if model else None,
            "prior_advantage": model["class_priors"][1] if model else None,
            "payoff_advantage": model["class_payoffs"][1] if model else None,
            "payoff_disadvantage": model["class_payoffs"][0] if model else None,
            "all_constant_model": model["all_constant_model"] if model else None})
        records.append(record)
        membership.extend({"fit_index": t, "cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index), "exit_index": int(r.exit_index),
                           "sample_weight": r.sample_weight, "fit_status": record["status"]} for r in rows.itertuples())
    counts = {"monthly_records": len(records), "eligible_monthly_records": sum(r["source_input_identity"] is not None for r in records),
        "available_monthly_records": sum(r["status"] == "FIT_COMPLETE" for r in records), "distinct_source_input_sets": len(cache),
        "new_model_fits": attempts, "successful_distinct_fits": successful, "failed_fits": attempts - successful,
        "reused_monthly_fits": reused, "missing_class_distinct_models": sum(v["status"] == "NO_VIEW_BOTH_CLASSES_REQUIRED" for v in cache.values()),
        "all_constant_distinct_models": sum(v["model"] is not None and v["model"]["all_constant_model"] for v in cache.values())}
    return records, pd.DataFrame(diagnostics), pd.DataFrame(membership), counts


def probability_payoff_prediction(model, values):
    require(model["kind"] == KIND and model["features"] == FEATURES and model["classes"] == [0, 1], "概率幅度退出模型身份不同")
    x = np.asarray(values, float)
    require(x.shape == (8,) and np.isfinite(x).all(), "概率幅度预测需要完整八因素")
    z = np.clip((x - model["mean"]) / model["scale"], -model["feature_clip"], model["feature_clip"])
    active = np.asarray(model["active_features"], bool)
    variances = np.asarray(model["class_variances"])[:, active]
    means = np.asarray(model["class_means"])[:, active]
    log_scores = np.log(model["class_priors"]) - .5 * np.sum(np.log(2*np.pi*variances) + (z[active]-means)**2/variances, axis=1)
    require(np.isfinite(log_scores).all(), "两类概率分数非有限")
    masses = np.exp(log_scores - log_scores.max())
    probabilities = masses / masses.sum()
    expected = float(probabilities @ np.asarray(model["class_payoffs"]))
    require(np.isfinite(expected) and abs(probabilities.sum()-1) < 1e-12, "两类概率或平均净增量无效")
    return {"prediction": expected, "probability_advantage": float(probabilities[1]),
            "probability_disadvantage": float(probabilities[0]),
            "expected_gain_contribution": float(probabilities[1]*model["class_payoffs"][1]),
            "expected_loss_contribution": float(probabilities[0]*model["class_payoffs"][0])}


class ProbabilityPayoffExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "概率幅度退出月度时点必须唯一递增")
        self.confirmation_days = confirmation_days
        self.cycle_id, self.selection_index, self.record, self.identity, self.negative_count = None, None, None, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            require(t == cycle["entry_index"], "概率幅度版本必须在实际买入日首次持仓收盘选择")
            self.cycle_id, self.selection_index = cycle["cycle_id"], t
            k = bisect_right(self.indexes, t) - 1
            self.record = self.models[k] if k >= 0 else None
            self.identity, self.negative_count = prediction_identity(self.record), 0
        require(self.selection_index == cycle["entry_index"] and t >= self.selection_index, "实际持仓模型选择起点改变")
        values = state_values(self.data, t, cycle, current_value, peak_value)
        record, prediction, status = self.record, None, "NO_VIEW_NO_MATURE_MODEL"
        detail = {key: None for key in ["probability_advantage", "probability_disadvantage", "expected_gain_contribution", "expected_loss_contribution"]}
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= self.selection_index, "概率幅度退出读取未来周期或模型")
            require(record["parameter_first_fit_index"] <= record["fit_index"], "概率幅度参数来自未来缓存")
            if np.isfinite(values).all():
                try:
                    detail = probability_payoff_prediction(record["model"], values)
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
