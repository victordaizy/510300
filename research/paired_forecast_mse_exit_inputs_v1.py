"""用同期预测的已成熟误差校准两套退出模型，不使用未来模型或未结束周期。"""
from bisect import bisect_right
import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require
from research.learned_cycle_exit_v1 import FEATURES, predict, state_values
from research.profit_drawdown_interaction_inputs_v1 import INTERACTION, interaction_value, interaction_prediction


class ModelHistory:
    def __init__(self, records, interaction=False):
        self.records, self.interaction = records, interaction
        self.indexes = [int(r["fit_index"]) for r in records]
        require(self.indexes == sorted(set(self.indexes)), "保存模型时点必须唯一递增")

    def at(self, t):
        k = bisect_right(self.indexes, t)-1
        return self.records[k] if k >= 0 else None

    def forecast(self, t, values, reference_cycle_id=None):
        record = self.at(t)
        value, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= t, "保存模型使用未来周期或未来模型")
            if reference_cycle_id is not None:
                require(reference_cycle_id not in record["training_cycles"], "同期预测含当前未成熟参考周期")
            if np.isfinite(values).all():
                if self.interaction:
                    value = interaction_prediction(record["model"], values)
                else:
                    require(record["model"]["kind"] == "RIDGE", "原八项模型身份错误")
                    value = predict(record["model"], np.asarray(values, float))
                require(np.isfinite(value), "同期模型预测出现非有限值")
                status = "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_CURRENT_FEATURES"
        return value, status, record


def reconstruct_pairs(samples, old_models, new_models):
    """保留全部原状态；目标虽已保存，但校准只能在整周期成熟后读取。"""
    require(not samples.duplicated(["cycle_id", "origin_index"]).any(), "自然周期状态重复")
    old, new = ModelHistory(old_models), ModelHistory(new_models, True)
    require(old.indexes == new.indexes, "两套原模型月度日程不同")
    rows = []
    for r in samples.itertuples():
        require(r.origin_index < r.early_exit_index < r.exit_index, "原状态、提前卖出和自然退出时钟不符")
        x = np.asarray([getattr(r, c) for c in FEATURES], float)
        x9 = np.r_[x, interaction_value(x[1], x[2])]
        forecasts = [old.forecast(r.origin_index, x, r.cycle_id), new.forecast(r.origin_index, x9, r.cycle_id)]
        record = {"cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index), "origin": r.origin,
                  "exit_index": int(r.exit_index), "mature_date": r.mature_date, "target": float(r.target)}
        for role, (value, status, model) in zip(["old", "new"], forecasts):
            record.update({f"{role}_prediction": value, f"{role}_status": status,
                           f"{role}_fit_index": model["fit_index"] if model else None,
                           f"{role}_error": value-r.target if value is not None and np.isfinite(r.target) else None})
        record["both_available"] = all(item[0] is not None for item in forecasts)
        rows.append(record)
    return pd.DataFrame(rows)


def complete_mature_rows(pairs, fit_index, cfg):
    """整周期排除部分同期预测可用者；已入选周期的未知误差不得删行补足。"""
    eligible = []
    for cycle_id, group in pairs.groupby("cycle_id", sort=False):
        require(group.exit_index.nunique() == 1, "同一自然周期成熟日不唯一")
        maturity = int(group.exit_index.iloc[0])
        if maturity <= fit_index and bool(group.both_available.all()):
            eligible.append((maturity, int(cycle_id)))
    ids = [cycle for maturity, cycle in sorted(eligible)[-cfg["recent_cycles"]:]]
    rows = pairs[pairs.cycle_id.isin(ids)].sort_values(["cycle_id", "origin_index"]).copy()
    rows["sample_weight"] = 1./rows.groupby("cycle_id").origin_index.transform("count") if len(rows) else pd.Series(dtype=float)
    return rows, ids


def minimum_mse_weights(rows, previous):
    """最小化未去均值的预测误差平方，两个非负权重之和为一。"""
    require(np.isfinite(previous).all() and min(previous) >= 0 and abs(sum(previous)-1.) < 1e-12, "上次混合权重不合法")
    errors = rows[["old_error", "new_error"]].to_numpy(float)
    weights = rows.sample_weight.to_numpy(float)
    result = {"old_weight": float(previous[0]), "new_weight": float(previous[1]),
              "calibration_status": "NO_VIEW_INCOMPLETE_PAIRED_ERRORS", "old_mse": None,
              "new_mse": None, "error_cross_mean": None, "difference_mse": None, "raw_old_weight": None, "mixed_mse": None}
    if not len(rows) or not np.isfinite(errors).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        return result
    a, b = errors.T
    old_mse, new_mse, cross, difference = [float(np.average(v, weights=weights)) for v in [a*a, b*b, a*b, (a-b)**2]]
    result.update(old_mse=old_mse, new_mse=new_mse, error_cross_mean=cross, difference_mse=difference)
    if difference == 0.:
        result["calibration_status"] = "NO_VIEW_IDENTICAL_PAIRED_ERRORS_KEEP_PREVIOUS"
        return result
    require(np.isfinite([old_mse, new_mse, cross, difference]).all() and difference > 0, "误差平方均值计算异常")
    raw = (new_mse-cross)/difference
    old_weight = float(np.clip(raw, 0., 1.))
    new_weight = 1.-old_weight
    result.update(old_weight=old_weight, new_weight=new_weight, raw_old_weight=raw,
                  mixed_mse=float(np.average((old_weight*a+new_weight*b)**2, weights=weights)), calibration_status="CALIBRATION_COMPLETE")
    return result


def calibrate_months(pairs, schedule, cfg):
    previous, previous_origin = (1., 0.), None
    records, memberships = [], []
    indexes = [int(r["fit_index"]) for r in schedule]
    require(indexes == sorted(set(indexes)), "混合权重月度日程不合法")
    for original in schedule:
        t = int(original["fit_index"])
        rows, ids = complete_mature_rows(pairs, t, cfg)
        enough = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        if enough:
            result = minimum_mse_weights(rows, previous)
        else:
            result = {"old_weight": previous[0], "new_weight": previous[1], "calibration_status": "NO_VIEW_MINIMUM_COMPLETE_PAIRED_CYCLES_OR_ROWS"}
        if result["calibration_status"] == "CALIBRATION_COMPLETE":
            previous = result["old_weight"], result["new_weight"]
            previous_origin = original["fit_origin"]
        record = {"fit_index": t, "fit_origin": original["fit_origin"], "training_cycles": ids,
                  "training_cycle_count": len(ids), "training_rows": len(rows), "supports_minimum": enough,
                  "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None,
                  "weight_update_origin": previous_origin, **result}
        records.append(record)
        if enough:
            memberships.extend({"fit_index": t, "cycle_id": int(r.cycle_id), "origin_index": int(r.origin_index),
                                "exit_index": int(r.exit_index), "sample_weight": float(r.sample_weight)} for r in rows.itertuples())
    return records, pd.DataFrame(memberships)


def mix_predictions(old, new, old_weight, new_weight):
    require(np.isfinite([old_weight, new_weight]).all() and min(old_weight, new_weight) >= 0 and abs(old_weight+new_weight-1.) < 1e-12, "当前混合权重非法")
    total = 0.
    for value, weight in [(old, old_weight), (new, new_weight)]:
        if weight > 0.:
            if value is None or not np.isfinite(value):
                return None
            total += weight*value
    return float(total)


class PairedForecastExitController:
    def __init__(self, data, old_models, new_models, calibrations, confirmation_days=2):
        self.data, self.old, self.new = data, ModelHistory(old_models), ModelHistory(new_models, True)
        self.calibrations = calibrations
        self.indexes = [int(r["fit_index"]) for r in calibrations]
        require(self.indexes == sorted(set(self.indexes)), "校准时点必须唯一递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        x = state_values(self.data, t, cycle, current_value, peak_value)
        x9 = np.r_[x, interaction_value(x[1], x[2])]
        old_value, old_status, old_record = self.old.forecast(t, x)
        new_value, new_status, new_record = self.new.forecast(t, x9)
        k = bisect_right(self.indexes, t)-1
        calibration = self.calibrations[k] if k >= 0 else None
        if calibration:
            require(calibration["fit_index"] <= t and (calibration["latest_exit_index"] is None or calibration["latest_exit_index"] <= calibration["fit_index"]), "混合权重用了未成熟周期")
        weights = (calibration["old_weight"], calibration["new_weight"]) if calibration else (1., 0.)
        estimate = mix_predictions(old_value, new_value, *weights)
        self.negative_count = self.negative_count+1 if estimate is not None and estimate < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": "PREDICTION_AVAILABLE" if estimate is not None else "NO_VIEW_REQUIRED_PREDICTION_MISSING",
                "continuation_prediction": estimate, "negative_confirmation_count": self.negative_count,
                "learned_exit_requested": self.negative_count >= self.confirmation_days,
                "old_prediction": old_value, "new_prediction": new_value, "old_prediction_status": old_status, "new_prediction_status": new_status,
                "old_fit_index": old_record["fit_index"] if old_record else None, "new_fit_index": new_record["fit_index"] if new_record else None,
                "old_weight": weights[0], "new_weight": weights[1], "calibration_fit_index": calibration["fit_index"] if calibration else None,
                "calibration_status": calibration["calibration_status"] if calibration else "NO_VIEW_INITIAL_OLD_BASELINE",
                "weight_update_origin": calibration["weight_update_origin"] if calibration else None,
                **dict(zip(FEATURES, x)), INTERACTION: x9[-1]}
