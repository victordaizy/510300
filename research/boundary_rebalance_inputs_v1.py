"""保留143目标，只把已有持仓越界后的调整终点改为最近边界。"""
import math
import numpy as np
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = "BOUNDARY_REBALANCE"
CANDIDATES = {PRIMARY: "已有持仓越界后调整到最近边界"}
MODELS = ["TREND_NOISE_REFERENCE_BLEND"]


def boundary_request(account, close, target, cfg):
    require(cfg["weight_band"] == .1 and cfg["boundary_equality_tolerance"] == 1e-12, "边界固定宽度或数值等号容差改变")
    nav = account.value(close)
    require(np.isfinite(close) and close > 0 and np.isfinite(nav) and nav > 0, "边界请求需要已知正收盘和自身净值")
    actual = account.shares * close / nav
    if np.isnan(target):
        return {"requested_quantity": 0, "reference_weight": np.nan, "rebalance_weight": np.nan,
                "request_origin_nav": nav, "request_origin_stock_weight": actual, "lower_weight": np.nan, "upper_weight": np.nan,
                "request_rule": "NO_VIEW", "action": "目标未知，保留实际份额"}
    require(np.isfinite(target) and 0 <= target <= 1, "边界目标超出无融资范围")
    lower, upper = max(0., target-cfg["weight_band"]), min(1., target+cfg["weight_band"])
    selected, rule, description = np.nan, "HOLD_INSIDE_INTERVAL", "处于持仓区间内，保持份额"
    if target == 0:
        selected, rule, description = 0., "EXPLICIT_FULL_EXIT", "目标明确为零，申请全部退出"
    elif account.shares == 0:
        selected, rule, description = target, "FIRST_ENTRY_AT_CENTER", "空仓首次进入，按原中心目标申请"
    elif actual < lower-cfg["boundary_equality_tolerance"]:
        selected, rule, description = lower, "LOWER_BOUNDARY", "低于下边界，仅申请补至下边界附近"
    elif actual > upper+cfg["boundary_equality_tolerance"]:
        selected, rule, description = upper, "UPPER_BOUNDARY", "高于上边界，仅申请减至上边界附近"
    desired = int(math.floor(selected*nav/close/cfg["lot"]))*cfg["lot"] if np.isfinite(selected) else account.shares
    return {"requested_quantity": desired-account.shares, "reference_weight": target, "rebalance_weight": selected,
            "request_origin_nav": nav, "request_origin_stock_weight": actual, "lower_weight": lower, "upper_weight": upper,
            "request_rule": rule, "action": description}


def boundary_rebalance_frames(data, parents_by_cost, cfg, start):
    require(cfg["candidate_models"] == list(CANDIDATES) and cfg["parent_models"] == MODELS, "边界候选或父来源改变")
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    indices = np.arange(first-1, len(data)-1)
    summaries = []
    for cost, frame in frames.items():
        target = frame[MODELS[0]+"_parent_target"].to_numpy(float)
        frame[PRIMARY+"_target"] = target.copy()
        x = target[indices]
        summaries.append({"model": PRIMARY, "cost": cost, "decision_origins": len(indices), "positive_target_origins": int((x > 0).sum()),
            "zero_target_origins": int((x == 0).sum()), "unknown_target_origins": int(np.isnan(x).sum()),
            "parent_target_unchanged": True, "execution_quantity_requires_own_account": True})
    return frames, summaries
