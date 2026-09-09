"""每笔实际买入日首次收盘固定原八项退出模型，下一笔再选择。"""
import hashlib
import json
from bisect import bisect_right
import numpy as np
from research.within_cycle_exit_inputs_v1 import FEATURES, state_values, within_cycle_prediction
from research.intraday_overnight_increment_v1 import require


def prediction_identity(record):
    if record is None or record["model"] is None:
        return "NO_MODEL"
    model = record["model"]
    content = {key: model[key] for key in ["features", "mean", "scale", "coefficients", "intercept", "feature_clip"]}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class EntryVintageExitController:
    def __init__(self, data, models, confirmation_days=2):
        self.data, self.models = data, models
        self.indexes = [record["fit_index"] for record in models]
        require(self.indexes == sorted(set(self.indexes)), "原保存模型时点必须唯一递增")
        self.confirmation_days = confirmation_days
        self.cycle_id, self.selection_index, self.record = None, None, None
        self.identity, self.negative_count = None, 0

    def __call__(self, t, cycle, current_value, peak_value):
        if self.cycle_id != cycle["cycle_id"]:
            require(t == cycle["entry_index"], "固定版本必须在实际买入日首次持仓收盘选择")
            self.cycle_id, self.selection_index = cycle["cycle_id"], t
            k = bisect_right(self.indexes, t)-1
            self.record = self.models[k] if k >= 0 else None
            self.identity = prediction_identity(self.record)
            self.negative_count = 0
        require(self.selection_index == cycle["entry_index"] and t >= self.selection_index, "本笔模型选择时点或实际周期改变")
        x = state_values(self.data, t, cycle, current_value, peak_value)
        record = self.record
        value, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if record and record["status"] == "FIT_COMPLETE":
            require(record["latest_exit_index"] <= record["fit_index"] <= self.selection_index, "固定版本使用了首次收盘之后的模型或未来周期")
            if np.isfinite(x).all():
                value, status = within_cycle_prediction(record["model"], x), "PREDICTION_AVAILABLE"
            else:
                status = "NO_VIEW_INCOMPLETE_EIGHT_FEATURES"
        elif record and record["status"] in {"NO_VIEW_MODEL_FIT_FAILED", "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"}:
            status = record["status"]
        self.negative_count = self.negative_count+1 if value is not None and value < 0 else 0
        return {"learning_cycle_id": cycle["cycle_id"], "learning_status": status, "continuation_prediction": value,
                "learning_fit_origin": self.data.date.iloc[record["fit_index"]] if record else None,
                "model_selection_index": self.selection_index, "model_selection_origin": self.data.date.iloc[self.selection_index],
                "fixed_prediction_identity": self.identity, "negative_confirmation_count": self.negative_count,
                "learned_exit_requested": self.negative_count >= self.confirmation_days, **dict(zip(FEATURES, x))}
