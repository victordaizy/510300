"""在已保存正目标上增加上下行平方幅度条件，明确零目标优先。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require

PARENT = "DOWNSIDE_REFERENCE_RISK"
PRIMARY = "DOWNSIDE_BALANCE_GATE"


def balance_gate_targets(data, parent, cfg, start, cost_id):
    """相等允许，未知不能变成退出；不改原有风险规模。"""
    require(cfg["balance_multiple"] == 2. and cfg["weight_band"] == .10, "方向条件或既有带宽发生改变")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "方向条件需要递增完整日历")
    require(pd.DatetimeIndex(parent.date).equals(dates), "方向条件父目标日期不匹配")
    require(np.array_equal(parent.origin_index, np.arange(len(data))), "方向条件父目标索引不完整")
    require(parent.reference_cost.eq(cost_id).all() and parent.model.eq(PARENT).all(), "方向条件父费用或策略身份不同")
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])
    require(first > 0 and len(data) > first, "方向条件没有准备收盘或交易区间")
    old = parent.target.to_numpy(float)
    require((np.isnan(old) | (np.isfinite(old) & (old >= 0) & (old <= 1))).all(), "方向条件父目标超出不融资范围")
    require(np.isnan(old[:first-1]).all() and np.isnan(old[-1]), "方向条件父目标含未知准备期或终点收盘的数值")
    full = parent.full_second_moment20.to_numpy(float)
    downside = parent.downside_second_moment20.to_numpy(float)
    for values in [full, downside]:
        require((np.isnan(values) | (np.isfinite(values) & (values >= 0))).all(), "方向条件风险平方值非法")
    known = np.isfinite(full) & np.isfinite(downside)
    require((downside[known] <= full[known]+1e-15).all(), "方向条件下行平方值超过全部平方值")
    allowed = np.full(len(data), np.nan)
    allowed[known] = (full[known] >= 2.*downside[known]).astype(float)
    target = np.full(len(data), np.nan)
    status = np.full(len(data), "NO_VIEW_PARENT_TARGET", dtype=object)
    explicit_zero = old == 0
    positive = old > 0
    accepted = positive & (allowed == 1.)
    rejected = positive & (allowed == 0.)
    target[explicit_zero | rejected] = 0.
    target[accepted] = old[accepted]
    status[explicit_zero] = "EXPLICIT_PARENT_ZERO"
    status[accepted] = "PARENT_SIZE_ALLOWED_BY_BALANCE"
    status[rejected] = "EXPLICIT_EXIT_DOWNSIDE_DOMINANCE"
    status[positive & ~known] = "NO_VIEW_BALANCE_INPUT"
    status[:first-1] = "NO_VIEW_OUTSIDE_PARENT_DECISION_RANGE"
    status[-1] = "NO_VIEW_OUTSIDE_PARENT_DECISION_RANGE"
    frame = parent.rename(columns={"target": "parent_target", "model": "parent_model", "target_status": "parent_target_status"}).copy()
    frame["model"] = PRIMARY
    frame["balance_allowed"] = allowed
    frame["target"] = target
    frame["target_status"] = status
    indices = np.arange(first-1, len(data)-1)
    summary = {"cost": cost_id, "calendar_rows": len(data), "decision_origins": len(indices),
        "parent_positive_origins": int(positive[indices].sum()), "positive_origins_allowed": int(accepted[indices].sum()),
        "parent_positive_origins_rejected": int(rejected[indices].sum()), "zero_target_origins": int((target[indices] == 0).sum()),
        "unknown_target_origins": int(np.isnan(target[indices]).sum()), "new_model_fits": 0, "new_reference_accounts": 0}
    return frame, summary
