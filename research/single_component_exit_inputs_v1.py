"""按成熟周期等权拟合单成分退出系数，缓存相同输入并固定每笔版本。"""
import copy
import hashlib
import json
from bisect import bisect_right
import numpy as np
import pandas as pd
from research.learned_cycle_exit_v1 import FEATURES, state_values, training_rows
from research.intraday_overnight_increment_v1 import require

KIND = "WITHIN_CYCLE_WEIGHTED_PLS1"


def centered_inputs(rows, cfg):
    x = rows[FEATURES].to_numpy(float)
    y, w = rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "单成分训练输入缺失，禁止删行")
    require(np.isfinite(w).all() and (w > 0).all(), "单成分训练权重无效")
    require(not rows.duplicated(["cycle_id", "origin_index"]).any(), "单成分训练周期原点重复")
    require((rows.origin_index < rows.exit_index).all(), "训练状态必须早于原自然退出")
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=w))
    scale = np.where(scale > 1e-12, scale, 1.)
    z = np.clip((x - mean) / scale, -cfg["feature_clip"], cfg["feature_clip"])
    dx, dy, groups = np.empty_like(z), np.empty_like(y), []
    for cycle in sorted(set(rows.cycle_id)):
        mask = rows.cycle_id.eq(cycle).to_numpy()
        require((np.diff(rows.origin_index.to_numpy()[mask]) > 0).all(), "周期原点必须严格递增")
        require(np.allclose(w[mask], 1. / mask.sum(), atol=1e-14, rtol=0), "每周期总权重必须为一且状态等权")
        mz, my = np.average(z[mask], axis=0, weights=w[mask]), float(np.average(y[mask], weights=w[mask]))
        dx[mask], dy[mask] = z[mask] - mz, y[mask] - my
        groups.append({"cycle_id": int(cycle), "rows": int(mask.sum()),
                       "standardized_feature_mean": mz.tolist(), "target_mean": my})
    return dx, dy, w, mean, scale, groups


def fit_single_component_cycle(rows, cfg):
    require(cfg["n_components"] == 1 and cfg["orthogonality_tolerance"] == 1e-10,
            "单成分数量或数值核对设置改变")
    dx, dy, w, mean, scale, groups = centered_inputs(rows, cfg)
    total = float(w.sum())
    covariance = dx.T @ (w * dy) / total
    require(np.isfinite(covariance).all(), "因素与目标协动非有限")
    direction, beta = np.zeros(len(FEATURES)), np.zeros(len(FEATURES))
    slope = score_variance = score_covariance = residual_covariance = 0.
    magnitude = float(np.max(abs(covariance)))
    if magnitude > 0.:
        direction = covariance / magnitude
        direction /= np.linalg.norm(direction)
        score = dx @ direction
        score_variance = float(np.average(score ** 2, weights=w))
        score_covariance = float(np.average(score * dy, weights=w))
        require(np.isfinite(score_variance) and score_variance > 0 and np.isfinite(score_covariance),
                "非零协动的单成分分母必须为有限正数")
        slope = score_covariance / score_variance
        beta = slope * direction
        residual_covariance = float(np.average(score * (dx @ beta - dy), weights=w))
        require(abs(residual_covariance) <= cfg["orthogonality_tolerance"], "单成分回归残差未满足正交条件")
    require(np.isfinite(beta).all() and np.isfinite(slope), "单成分系数非有限")
    for group in groups:
        group["cycle_intercept"] = float(group["target_mean"] - np.asarray(group["standardized_feature_mean"]) @ beta)
    intercept = float(np.mean([group["cycle_intercept"] for group in groups]))
    require(np.isfinite(intercept), "新周期截距非有限")
    return {"kind": KIND, "features": FEATURES.copy(), "mean": mean.tolist(), "scale": scale.tolist(),
        "coefficients": beta.tolist(), "intercept": intercept, "feature_clip": cfg["feature_clip"],
        "cycle_intercepts": groups, "n_components": 1, "effective_cycle_weight": total,
        "factor_target_covariance": covariance.tolist(), "component_direction": direction.tolist(),
        "component_slope": slope, "score_variance": score_variance, "score_target_covariance": score_covariance,
        "score_residual_covariance": residual_covariance, "zero_covariance_model": magnitude == 0.,
        "nonzero_factors": [name for name, value in zip(FEATURES, beta) if value != 0],
        "nonzero_factor_count": int((beta != 0).sum()),
        "training_weighted_mse": float(np.average((dx @ beta - dy) ** 2, weights=w)),
        "new_cycle_intercept_rule": "EQUAL_MEAN_OF_MATURE_TRAINING_CYCLE_INTERCEPTS"}


def input_identity(rows, cfg):
    settings = {"features": FEATURES, **{key: cfg[key] for key in ["feature_clip", "n_components", "orthogonality_tolerance"]}}
    h = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode())
    h.update(rows[["cycle_id", "origin_index", "exit_index"]].to_numpy(dtype="<i8").tobytes())
    h.update(rows[FEATURES + ["target", "sample_weight"]].to_numpy(dtype="<f8").tobytes())
    return h.hexdigest()


def prediction_identity(record):
    if record is None or record["model"] is None:
        return "NO_MODEL"
    content = {key: record["model"][key] for key in ["kind", "features", "mean", "scale", "coefficients", "intercept", "feature_clip"]}
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
        require(eligible == (source["status"] == "FIT_COMPLETE"), "原月度训练支持状态改变")
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
                    item["model"] = fit_single_component_cycle(rows, cfg)
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
            "nonzero_factor_count": model["nonzero_factor_count"] if model else None,
            "nonzero_factors": ",".join(model["nonzero_factors"]) if model else None,
            "score_residual_covariance": model["score_residual_covariance"] if model else None,
            "zero_covariance_model": model["zero_covariance_model"] if model else None})
        records.append(record)
        membership.extend({"fit_index": t, "cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index), "exit_index": int(r.exit_index),
                           "sample_weight": r.sample_weight, "fit_status": record["status"]} for r in rows.itertuples())
    counts = {"monthly_records": len(records), "eligible_monthly_records": sum(r["source_input_identity"] is not None for r in records),
        "available_monthly_records": sum(r["status"] == "FIT_COMPLETE" for r in records), "distinct_source_input_sets": len(cache),
        "new_model_fits": attempts, "successful_distinct_fits": successful, "failed_fits": attempts - successful,
        "reused_monthly_fits": reused, "exact_zero_distinct_models": sum(v["model"] is not None and v["model"]["nonzero_factor_count"] == 0 for v in cache.values())}
    return records, pd.DataFrame(diagnostics), pd.DataFrame(membership), counts


def single_component_prediction(model, values):
    require(model["kind"] == KIND and model["features"] == FEATURES, "单成分退出模型身份不同")
    x = np.asarray(values, float)
    require(x.shape == (8,) and np.isfinite(x).all(), "单成分退出预测需要完整八因素")
    z = np.clip((x - model["mean"]) / model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"] + z @ np.asarray(model["coefficients"]))


class SingleComponentExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "单成分退出月度时点必须唯一递增")
        self.confirmation_days = confirmation_days
        self.cycle_id, self.selection_index, self.record, self.identity, self.negative_count = None, None, None, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            require(t == cycle["entry_index"], "单成分版本必须在实际买入日首次持仓收盘选择")
            self.cycle_id, self.selection_index = cycle["cycle_id"], t
            k = bisect_right(self.indexes, t) - 1
            self.record = self.models[k] if k >= 0 else None
            self.identity, self.negative_count = prediction_identity(self.record), 0
        require(self.selection_index == cycle["entry_index"] and t >= self.selection_index, "实际持仓模型选择起点改变")
        values = state_values(self.data, t, cycle, current_value, peak_value)
        record, prediction, status = self.record, None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= self.selection_index, "单成分退出读取未来周期或模型")
            require(record["parameter_first_fit_index"] <= record["fit_index"], "单成分参数来自未来缓存")
            if np.isfinite(values).all():
                prediction, status = single_component_prediction(record["model"], values), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_EIGHT_FEATURES"
        elif record and record["status"] != "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS":
            status = record["status"]
        self.negative_count = self.negative_count + 1 if prediction is not None and prediction < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": prediction,
            "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
            "model_selection_index": self.selection_index, "model_selection_origin": self.data.date.iloc[self.selection_index],
            "fixed_prediction_identity": self.identity, "negative_confirmation_count": self.negative_count,
            "learned_exit_requested": self.negative_count >= self.confirmation_days, **dict(zip(FEATURES, values))}
