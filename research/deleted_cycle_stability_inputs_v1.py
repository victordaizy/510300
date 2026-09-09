"""逐个删除完整成熟周期，要求全样本及每个删除模型一致支持退出。"""
from __future__ import annotations

from bisect import bisect_right

import numpy as np
import pandas as pd

from research.learned_cycle_exit_v1 import FEATURES, state_values, fit_one, predict
from research.intraday_overnight_increment_v1 import require


def deletion_subsets(rows, fit_index):
    require(len(rows) > 0 and rows.exit_index.le(fit_index).all(), "删除检查必须使用已经完整结束的周期")
    ids = rows[["cycle_id", "exit_index"]].drop_duplicates().sort_values(["exit_index", "cycle_id"]).cycle_id.tolist()
    require(len(ids) >= 2 and len(ids) == len(set(ids)), "周期日期不唯一或不足以删除")
    for removed in ids:
        selected = rows[rows.cycle_id.ne(removed)].copy()
        selected["sample_weight"] = 1. / selected.groupby("cycle_id").origin_index.transform("count")
        yield int(removed), selected


def fit_deletions(rows, fit_index, config):
    records = []
    for removed, selected in deletion_subsets(rows, fit_index):
        stored, failure, status = None, None, "FIT_COMPLETE"
        try:
            stored = fit_one(selected, "RIDGE", config)
            if not (np.isfinite(stored["coefficients"]).all() and np.isfinite(stored["intercept"])):
                raise FloatingPointError("删除模型出现非有限系数")
        except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            stored, failure, status = None, str(error), "MODEL_FIT_FAILED"
        records.append({"deleted_cycle_id": removed, "training_cycles": sorted(int(x) for x in selected.cycle_id.unique()),
            "training_rows": len(selected), "latest_exit_index": int(selected.exit_index.max()), "model": stored,
            "status": status, "failure": failure})
    return records


def committee_values(stored, values):
    require(stored["kind"] == "DELETED_CYCLE_COMMITTEE" and len(stored["deletions"]) > 0, "删除周期模型集合不完整")
    base = predict(stored["base_model"], values)
    deleted = np.array([predict(m["model"], values) for m in stored["deletions"]])
    require(np.isfinite(deleted).all() and np.isfinite(base), "模型集合预测不是有限数")
    return {"continuation_prediction": max(base, float(deleted.max())), "base_continuation_prediction": base,
        "deleted_max_prediction": float(deleted.max()), "deleted_min_prediction": float(deleted.min()),
        "committee_size": len(deleted) + 1, "nonnegative_predictions": int(np.sum(deleted >= 0)) + int(base >= 0)}


class DeletedCycleExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.fit_indexes = [m["fit_index"] for m in models]
        require(self.fit_indexes == sorted(set(self.fit_indexes)), "模型生效原点必须唯一且递增")
        self.confirmation_days, self.cycle_id, self.negative_count = confirmation_days, None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if cycle["cycle_id"] != self.cycle_id:
            self.cycle_id, self.negative_count = cycle["cycle_id"], 0
        values = state_values(self.data, t, cycle, current_value, peak_value)
        pos = bisect_right(self.fit_indexes, t) - 1
        stored = self.models[pos] if pos >= 0 else None
        result = {"continuation_prediction": None, "base_continuation_prediction": None, "deleted_max_prediction": None,
            "deleted_min_prediction": None, "committee_size": None, "nonnegative_predictions": None}
        status = "NO_VIEW_NO_MATURE_MODEL"
        if stored and stored["status"] == "FIT_COMPLETE" and np.isfinite(values).all():
            require(stored["latest_exit_index"] <= stored["fit_index"] <= t, "删除周期退出使用未来模型或周期")
            require(len(stored["model"]["deletions"]) == stored["training_cycle_count"], "未删除检查每一个原周期")
            result, status = committee_values(stored["model"], values), "PREDICTION_AVAILABLE"
        elif stored and stored["status"] == "FIT_COMPLETE":
            status = "NO_VIEW_INCOMPLETE_HOLDING_FEATURES"
        elif stored and stored["status"] == "NO_VIEW_MODEL_FIT_FAILED":
            status = "NO_VIEW_MODEL_FIT_FAILED"
        maximum = result["continuation_prediction"]
        self.negative_count = self.negative_count + 1 if maximum is not None and maximum < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status,
            "learning_fit_origin": self.data.date.iloc[stored["fit_index"]] if stored else pd.NaT,
            "negative_confirmation_count": self.negative_count, "learned_exit_requested": self.negative_count >= self.confirmation_days,
            **result, **dict(zip(FEATURES, values))}
