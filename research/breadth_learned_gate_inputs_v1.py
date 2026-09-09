"""广度只筛选既有学习策略的进入，缺失时显式沿用原策略。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.intraday_overnight_increment_v1 import require


def gate_rule(base_rule, factors, enabled=True):
    original = np.asarray(base_rule["entry"], dtype=int)
    require(len(original) == len(factors), "广度与原进入条件长度不同")
    require(np.isin(original, [0, 1]).all(), "本轮只允许原单一日内强弱进入模式")
    valid = factors.breadth_valid.to_numpy(bool)
    majority = factors.breadth_majority.to_numpy(bool)
    rejected = enabled & valid & ~majority
    effective = np.where(rejected, 0, original)
    state = np.where(valid, np.where(majority, "有效广度多数上涨", "有效广度未过半，暂停新进入"), "NO_VIEW_广度缺失，沿用原学习策略进入决定")
    if not enabled:
        state = np.full(len(original), "仅作验证：关闭附加筛选，沿用原策略", dtype=object)
    trace = pd.DataFrame({"date": factors.date.to_numpy(), "base_raw_entry": original, "effective_raw_entry": effective,
        "breadth_gate_state": state, "breadth_gate_rejected": rejected,
        "base_eligible_origin_rejected": (original == 1) & rejected,
        "breadth_missing_baseline_fallback": enabled & ~valid,
        "base_eligible_origin_with_missing_breadth": enabled & ~valid & (original == 1),
        "base_condition_reset": original == 0})
    rule = {"entry": effective, "exit": {k: np.asarray(v, dtype=bool).copy() for k, v in base_rule["exit"].items()},
            "rearm_allowed": original == 0}
    return rule, trace
