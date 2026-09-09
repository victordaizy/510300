"""按周期相关误差学习退出，完整保留首行，并缓存已经可用的同一输入。"""
import copy
import hashlib
import json
import math
from bisect import bisect_right
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from research.learned_cycle_exit_v1 import FEATURES, state_values, training_rows
from research.intraday_overnight_increment_v1 import require

KIND = "CYCLE_SERIAL_GLS_RIDGE"


def input_identity(rows, source_model):
    payload = {"source_model": source_model, "members": rows[["cycle_id", "origin_index"]].to_numpy(int).tolist()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def standardized_inputs(rows, source_model):
    require(source_model["kind"] == "WITHIN_CYCLE_FIXED_INTERCEPT_RIDGE" and source_model["features"] == FEATURES, "原残差模型身份不同")
    x, y, w = rows[FEATURES].to_numpy(float), rows.target.to_numpy(float), rows.sample_weight.to_numpy(float)
    require(len(rows) > 0 and np.isfinite(x).all() and np.isfinite(y).all(), "周期相关误差输入缺失，禁止删行")
    require(np.isfinite(w).all() and (w > 0).all(), "周期相关误差基础权重无效")
    require(not rows.duplicated(["cycle_id", "origin_index"]).any(), "周期状态原点重复")
    mean, scale = np.asarray(source_model["mean"]), np.asarray(source_model["scale"])
    require(np.isfinite(mean).all() and np.isfinite(scale).all() and (scale > 0).all(), "原标准化参数无效")
    for _, g in rows.groupby("cycle_id", sort=True):
        require((np.diff(g.origin_index.to_numpy()) > 0).all(), "周期原点没有严格递增")
        require(np.allclose(g.sample_weight, 1 / len(g), atol=1e-14, rtol=0), "原周期总权重或状态等权改变")
    return np.clip((x - mean) / scale, -source_model["feature_clip"], source_model["feature_clip"]), y, w


def residual_correlation(rows, source_model, bounds=(-.99, .99)):
    z, y, _ = standardized_inputs(rows, source_model)
    intercepts = {int(g["cycle_id"]): g["cycle_intercept"] for g in source_model["cycle_intercepts"]}
    errors = y - z @ np.asarray(source_model["coefficients"]) - np.array([intercepts[int(c)] for c in rows.cycle_id])
    cross, previous_square, pairs, gaps = [], [], 0, 0
    for cycle in sorted(set(rows.cycle_id)):
        mask = rows.cycle_id.eq(cycle).to_numpy()
        e, origins = errors[mask], rows.origin_index.to_numpy()[mask]
        adjacent = np.diff(origins) == 1
        count = int(adjacent.sum())
        gaps += int((~adjacent).sum())
        if count:
            a, b = e[:-1][adjacent], e[1:][adjacent]
            cross.append(math.fsum(a * b) / count)
            previous_square.append(math.fsum(a * a) / count)
        pairs += count
    denominator = math.fsum(previous_square)
    require(pairs > 0 and denominator > 0, "残差相邻相关无法识别")
    raw = math.fsum(cross) / denominator
    require(np.isfinite(raw), "残差相邻相关不是有限数值")
    rho = float(np.clip(raw, *bounds))
    return {"raw_rho": raw, "bounded_rho": rho, "bound_active": rho != raw, "adjacent_pairs": pairs, "nonadjacent_gaps": gaps}


def fit_serial_gls(rows, source_model, rho):
    require(np.isfinite(rho) and abs(rho) < 1, "相关矩阵要求绝对相关系数小于一")
    z, y, w = standardized_inputs(rows, source_model)
    information, rhs, groups = np.eye(8), np.zeros(8), []
    for cycle in sorted(set(rows.cycle_id)):
        mask = rows.cycle_id.eq(cycle).to_numpy()
        origins = rows.origin_index.to_numpy(int)[mask]
        values = np.column_stack([z[mask], y[mask], np.ones(mask.sum())])
        transformed = values.copy()
        a = rho ** np.diff(origins)
        transformed[1:] = (values[1:] - a[:, None] * values[:-1]) / np.sqrt(1 - a * a)[:, None]
        zz, yy, dd, ww = transformed[:, :8], transformed[:, 8], transformed[:, 9], w[mask]
        h, v, k = zz.T @ (ww * dd), float(dd @ (ww * dd)), float(dd @ (ww * yy))
        require(v > 0 and np.isfinite(v), "周期截距消元信息无效")
        information += zz.T @ (ww[:, None] * zz) - np.outer(h, h) / v
        rhs += zz.T @ (ww * yy) - h * k / v
        groups.append({"cycle_id": int(cycle), "rows": int(mask.sum()), "intercept_information": v,
                       "intercept_rhs": k, "intercept_cross": h.tolist()})
    beta = cho_solve(cho_factor(information, lower=True, check_finite=True), rhs, check_finite=True)
    require(np.isfinite(beta).all(), "广义最小二乘系数求解失败")
    for g in groups:
        g["cycle_intercept"] = float((g["intercept_rhs"] - np.asarray(g["intercept_cross"]) @ beta) / g["intercept_information"])
    intercept = math.fsum(g["cycle_intercept"] for g in groups) / len(groups)
    return {"kind": KIND, "features": FEATURES.copy(), "mean": source_model["mean"], "scale": source_model["scale"],
            "feature_clip": source_model["feature_clip"], "coefficients": beta.tolist(), "intercept": intercept,
            "rho": float(rho), "ridge_alpha": 1., "cycle_intercepts": groups,
            "first_row": "PRESERVED_UNSCALED", "new_cycle_intercept_rule": "EQUAL_MEAN_OF_MATURE_TRAINING_CYCLE_INTERCEPTS"}


def prediction_identity(record):
    if record is None or record["model"] is None:
        return "NO_MODEL"
    m = record["model"]
    content = {k: m[k] for k in ["kind", "features", "mean", "scale", "coefficients", "intercept", "feature_clip"]}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_monthly_models(samples, source_records, cfg, expected=None):
    require([r["fit_index"] for r in source_records] == sorted(set(r["fit_index"] for r in source_records)), "原月度时点必须唯一递增")
    cache, records, diagnostics, membership = {}, [], [], []
    attempted = successful = reused = 0
    expected_by_index = expected.set_index("fit_index") if expected is not None else None
    for original in source_records:
        t = int(original["fit_index"])
        rows, ids = training_rows(samples, t, cfg)
        require(ids == original["training_cycles"] and len(rows) == original["training_rows"], "原成熟训练成员改变")
        eligible = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        require(eligible == (original["status"] == "FIT_COMPLETE"), "原月度支持状态改变")
        require(not len(rows) or rows.exit_index.le(t).all(), "训练使用未来未结束周期")
        record = {k: copy.deepcopy(original[k]) for k in ["fit_index", "fit_origin", "fit_time", "training_cycles", "training_cycle_count", "training_rows", "latest_exit_index", "latest_exit_date"]}
        record.update(status="NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS", model=None, failure=None, source_input_identity=None,
                      first_attempt_fit_index=None, parameter_first_fit_index=None, reused_previous_input=False)
        if eligible:
            identity = input_identity(rows, original["model"])
            seen = identity in cache
            if not seen:
                item = {"first_attempt_fit_index": t, "parameter_first_fit_index": None, "status": "NO_VIEW_CORRELATION_NOT_IDENTIFIED",
                        "model": None, "failure": None, "correlation": None}
                try:
                    item["correlation"] = residual_correlation(rows, original["model"], tuple(cfg["correlation_bounds"]))
                except ValueError as error:
                    item["failure"] = str(error)
                if item["correlation"] is not None:
                    attempted += 1
                    try:
                        item["model"] = fit_serial_gls(rows, original["model"], item["correlation"]["bounded_rho"])
                        item.update(status="FIT_COMPLETE", parameter_first_fit_index=t)
                        successful += 1
                    except (ValueError, np.linalg.LinAlgError) as error:
                        item.update(status="NO_VIEW_MODEL_FIT_FAILED", failure=str(error))
                cache[identity] = item
            else:
                reused += 1
            item = cache[identity]
            require(item["first_attempt_fit_index"] <= t, "缓存读取未来月份")
            record.update({k: copy.deepcopy(item[k]) for k in item if k != "correlation"})
            record.update(source_input_identity=identity, reused_previous_input=seen)
            if expected_by_index is not None:
                expected_row = expected_by_index.loc[t]
                require(identity == expected_row.source_input_identity, "相关误差输入与准备绑定不同")
                require(item["correlation"] is not None and abs(item["correlation"]["raw_rho"] - expected_row.raw_rho) < 1e-12, "原保存残差相关不能复算")
            diagnostic = {"fit_index": t, "fit_origin": original["fit_origin"], "source_input_identity": identity,
                          "reused_previous_input": seen, "first_attempt_fit_index": item["first_attempt_fit_index"],
                          "parameter_first_fit_index": item["parameter_first_fit_index"], "status": item["status"],
                          **(item["correlation"] or {})}
        else:
            diagnostic = {"fit_index": t, "fit_origin": original["fit_origin"], "status": record["status"], "reused_previous_input": False}
        record["fixed_prediction_identity"] = prediction_identity(record)
        diagnostics.append({**diagnostic, "prediction_identity": record["fixed_prediction_identity"]})
        records.append(record)
        membership.extend({"fit_index": t, "cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index),
                           "exit_index": int(r.exit_index), "sample_weight": r.sample_weight, "fit_status": record["status"]} for r in rows.itertuples())
    counts = {"monthly_records": len(records), "eligible_monthly_records": sum(r["source_input_identity"] is not None for r in records),
              "available_monthly_records": sum(r["status"] == "FIT_COMPLETE" for r in records), "distinct_source_input_sets": len(cache),
              "new_model_fits": attempted, "successful_distinct_fits": successful, "failed_fits": attempted - successful,
              "reused_monthly_fits": reused, "unidentified_distinct_correlations": sum(v["correlation"] is None for v in cache.values())}
    return records, pd.DataFrame(diagnostics), pd.DataFrame(membership), counts


def serial_prediction(model, values):
    require(model["kind"] == KIND and model["features"] == FEATURES, "相关误差预测模型身份不同")
    x = np.asarray(values, float)
    require(x.shape == (8,) and np.isfinite(x).all(), "相关误差预测需要完整八因素")
    z = np.clip((x - model["mean"]) / model["scale"], -model["feature_clip"], model["feature_clip"])
    return float(model["intercept"] + z @ np.asarray(model["coefficients"]))


class SerialErrorExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.indexes = [r["fit_index"] for r in models]
        require(self.indexes == sorted(set(self.indexes)), "相关误差月度时点必须唯一递增")
        self.confirmation_days = confirmation_days
        self.cycle_id, self.selection_index, self.record, self.identity, self.negative_count = None, None, None, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            require(t == cycle["entry_index"], "相关误差版本必须在实际买入日首次持仓收盘选择")
            self.cycle_id, self.selection_index = cycle["cycle_id"], t
            k = bisect_right(self.indexes, t) - 1
            self.record = self.models[k] if k >= 0 else None
            self.identity, self.negative_count = prediction_identity(self.record), 0
        require(self.selection_index == cycle["entry_index"] and t >= self.selection_index, "实际持仓模型选择起点改变")
        x = state_values(self.data, t, cycle, current_value, peak_value)
        record, value, status = self.record, None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= self.selection_index, "相关误差读取未来周期或月度模型")
            require(record["parameter_first_fit_index"] <= record["fit_index"], "相关误差参数来自未来缓存")
            if np.isfinite(x).all():
                value, status = serial_prediction(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_EIGHT_FEATURES"
        elif record and record["status"] != "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS":
            status = record["status"]
        self.negative_count = self.negative_count + 1 if value is not None and value < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": value,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "model_selection_index": self.selection_index, "model_selection_origin": self.data.date.iloc[self.selection_index],
                "fixed_prediction_identity": self.identity, "negative_confirmation_count": self.negative_count,
                "learned_exit_requested": self.negative_count >= self.confirmation_days, **dict(zip(FEATURES, x))}
