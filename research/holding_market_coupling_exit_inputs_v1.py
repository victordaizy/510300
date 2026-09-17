"""以市场状态调节持仓状态的边际退出作用，按成熟输入向前缓存。"""
import copy
import hashlib
import json
from bisect import bisect_right
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from research.learned_cycle_exit_v1 import FEATURES, CN, state_values, training_rows
from research.intraday_overnight_increment_v1 import require

KIND = "WITHIN_CYCLE_HOLDING_MARKET_COUPLING_RIDGE"
PAIRS = [(i, j) for i in range(3) for j in range(4, 8)]
NAMES = FEATURES + [FEATURES[i]+"_times_"+FEATURES[j] for i,j in PAIRS]
CHINESE = CN + [CN[i]+"与"+CN[j]+"的联合作用" for i,j in PAIRS]
IDENTITY_KEYS = ["kind", "features", "mean", "scale", "coefficients", "intercept", "feature_clip",
                 "interaction_pairs", "product_mean", "product_scale"]


def standardize(x, w):
    mean = np.average(x, axis=0, weights=w)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=w))
    scale = np.where(scale > 1e-12, scale, 1.)
    return np.clip((x-mean)/scale, -5., 5.), mean, scale


def transformed(model, values):
    values = np.asarray(values, float)
    require(values.shape[-1] == 8 and np.isfinite(values).all(), "预测需要完整八项因素")
    z = np.clip((values-model["mean"])/model["scale"], -5., 5.)
    products = np.stack([z[...,i]*z[...,j] for i,j in PAIRS], axis=-1)
    joined = np.clip((products-model["product_mean"])/model["product_scale"], -5., 5.)
    return np.concatenate([z, joined], axis=-1)


def fit_coupling_cycle(rows, cfg):
    require(cfg["feature_clip"] == 5. and cfg["ridge_alpha"] == 1., "固定尺度或惩罚改变")
    require(cfg["interaction_pairs"] == [list(p) for p in PAIRS], "十二项联合作用身份不同")
    x, y, w = rows[FEATURES].to_numpy(float), rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows)>0 and np.isfinite(x).all() and np.isfinite(y).all(), "训练输入缺失，禁止删行")
    require(np.isfinite(w).all() and (w>0).all(), "训练权重无效")
    require(not rows.duplicated(["cycle_id", "origin_index"]).any(), "训练周期原点重复")
    require((rows.origin_index<rows.exit_index).all(), "训练状态必须早于自然退出")
    z, mean, scale = standardize(x, w)
    products = np.stack([z[:,i]*z[:,j] for i,j in PAIRS], axis=1)
    joint, product_mean, product_scale = standardize(products, w)
    design = np.column_stack([z, joint])
    dx, dy, groups = np.empty_like(design), np.empty_like(y), []
    for cycle in sorted(set(rows.cycle_id)):
        mask = rows.cycle_id.eq(cycle).to_numpy()
        require(np.allclose(w[mask], 1./mask.sum(), atol=1e-14, rtol=0), "每周期总权重必须为一且状态等权")
        mz, my = np.average(design[mask], axis=0, weights=w[mask]), float(np.average(y[mask], weights=w[mask]))
        dx[mask], dy[mask] = design[mask]-mz, y[mask]-my
        groups.append({"cycle_id": int(cycle), "rows": int(mask.sum()), "standardized_feature_mean": mz.tolist(), "target_mean": my})
    fit = Ridge(alpha=1., solver="svd", fit_intercept=False).fit(dx, dy, sample_weight=w)
    beta = fit.coef_
    require(np.isfinite(beta).all(), "联合作用系数非有限")
    residual = dx@beta-dy
    normal_error = float(np.max(abs(dx.T@(w*residual)+beta)))
    require(normal_error<1e-8, "岭回归正规方程未满足")
    for group in groups:
        group["cycle_intercept"] = float(group["target_mean"]-np.array(group["standardized_feature_mean"])@beta)
    return {"kind": KIND, "features": FEATURES.copy(), "expanded_features": NAMES.copy(), "mean": mean.tolist(),
        "scale": scale.tolist(), "product_mean": product_mean.tolist(), "product_scale": product_scale.tolist(),
        "interaction_pairs": [list(p) for p in PAIRS], "coefficients": beta.tolist(),
        "intercept": float(np.mean([g["cycle_intercept"] for g in groups])), "feature_clip": 5.,
        "cycle_intercepts": groups, "max_normal_error": normal_error,
        "nonzero_factors": [name for name,b in zip(NAMES,beta) if b!=0.], "nonzero_factor_count": int((beta!=0.).sum()),
        "training_weighted_mse": float(np.average(residual**2, weights=w)),
        "new_cycle_intercept_rule": "EQUAL_MEAN_OF_MATURE_TRAINING_CYCLE_INTERCEPTS"}


def input_identity(rows, cfg):
    settings = {"features": FEATURES, **{key: cfg[key] for key in ["feature_clip", "ridge_alpha", "interaction_pairs"]}}
    h = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode())
    h.update(rows[["cycle_id", "origin_index", "exit_index"]].to_numpy(dtype="<i8").tobytes())
    h.update(rows[FEATURES+["target", "sample_weight"]].to_numpy(dtype="<f8").tobytes())
    return h.hexdigest()


def prediction_identity(record):
    if record is None or record["model"] is None:
        return "NO_MODEL"
    content = {key:record["model"][key] for key in IDENTITY_KEYS}
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
                    item["model"] = fit_coupling_cycle(rows, cfg)
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
            "max_normal_error": model["max_normal_error"] if model else None,
            "zero_coefficients": model["nonzero_factor_count"] == 0 if model else None})
        records.append(record)
        membership.extend({"fit_index": t, "cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index), "exit_index": int(r.exit_index),
                           "sample_weight": r.sample_weight, "fit_status": record["status"]} for r in rows.itertuples())
    counts = {"monthly_records": len(records), "eligible_monthly_records": sum(r["source_input_identity"] is not None for r in records),
        "available_monthly_records": sum(r["status"] == "FIT_COMPLETE" for r in records), "distinct_source_input_sets": len(cache),
        "new_model_fits": attempts, "successful_distinct_fits": successful, "failed_fits": attempts - successful,
        "reused_monthly_fits": reused, "exact_zero_distinct_models": sum(v["model"] is not None and v["model"]["nonzero_factor_count"] == 0 for v in cache.values())}
    return records, pd.DataFrame(diagnostics), pd.DataFrame(membership), counts


def coupling_prediction(model, values):
    require(model["kind"] == KIND and model["features"] == FEATURES, "联合作用退出模型身份不同")
    require(model["interaction_pairs"] == [list(p) for p in PAIRS], "联合作用预测次序不同")
    return float(model["intercept"]+transformed(model, values)@np.array(model["coefficients"]))


class CouplingExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "联合作用退出月度时点必须唯一递增")
        self.confirmation_days = confirmation_days
        self.cycle_id, self.selection_index, self.record, self.identity, self.negative_count = None, None, None, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            require(t == cycle["entry_index"], "联合作用版本必须在实际买入日首次持仓收盘选择")
            self.cycle_id, self.selection_index = cycle["cycle_id"], t
            k = bisect_right(self.indexes, t) - 1
            self.record = self.models[k] if k >= 0 else None
            self.identity, self.negative_count = prediction_identity(self.record), 0
        require(self.selection_index == cycle["entry_index"] and t >= self.selection_index, "实际持仓模型选择起点改变")
        values = state_values(self.data, t, cycle, current_value, peak_value)
        record, prediction, status = self.record, None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= self.selection_index, "联合作用退出读取未来周期或模型")
            require(record["parameter_first_fit_index"] <= record["fit_index"], "联合作用参数来自未来缓存")
            if np.isfinite(values).all():
                prediction, status = coupling_prediction(record["model"], values), "PREDICTION_AVAILABLE"
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
