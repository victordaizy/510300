"""两条既有参考策略只按已实现风险更新预算，保留缺失与日终目标。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def active_budget_frame(dates, reference_returns, expert_states, first, active_exposure, window=242):
    dates = pd.DatetimeIndex(dates)
    returns = np.asarray(reference_returns, dtype=float)
    states = np.asarray(expert_states, dtype=float)
    active = np.asarray(active_exposure, dtype=float)
    require(active.shape == states.shape and (np.isnan(active) | np.isin(active, [0., 1.])).all(), "直接股票风险日标记无效")
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "风险预算日历必须严格递增")
    require(returns.shape == states.shape == (len(dates), 2), "两条参考策略的日历及形状不符")
    require(1 <= first < len(dates)-1 and isinstance(window, int) and window >= 2, "评价起点或风险窗口无效")
    require((np.isnan(returns) | (np.isfinite(returns) & (returns > -1))).all(), "参考收益存在无穷或净值失效")
    require((np.isnan(states) | np.isin(states, [0., 1.])).all(), "参考状态必须为零、一或缺失")
    rows = []
    weights = np.array([.5, .5])
    sd = np.full(2, np.nan)
    status, attempted, successful, start = "INITIAL_EQUAL_BUDGET_NO_RISK_ESTIMATE", pd.NaT, pd.NaT, pd.NaT
    count = 0
    selected_counts = [0, 0]
    for t in range(len(dates)):
        scheduled = False
        if t < first-1 or t == len(dates)-1:
            rows.append({"date": dates[t], "panic_budget": np.nan, "learned_budget": np.nan, "panic_sd": np.nan, "learned_sd": np.nan,
                "risk_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD", "risk_update_scheduled": False, "risk_attempt_origin": pd.NaT,
                "last_successful_risk_origin": pd.NaT, "risk_window_start": pd.NaT, "risk_window_observations": 0,
                "panic_state": states[t, 0], "learned_state": states[t, 1], "target": np.nan, "panic_active_days": 0, "learned_active_days": 0})
            continue
        if t >= first and dates[t].to_period("M") != dates[t-1].to_period("M"):
            scheduled, attempted = True, dates[t]
            count = min(window, t-first+1)
            start = dates[t-count+1]
            values = returns[t-count+1:t+1]
            active_window = active[t-count+1:t+1]
            selected_counts = [int(np.sum(active_window[:, k] == 1)) for k in range(2)]
            sd = np.full(2, np.nan)
            if count < window:
                status = "NO_VIEW_WARMUP_KEEP_BUDGET"
            elif not np.isfinite(values).all() or not np.isfinite(active_window).all():
                status = "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
            elif min(selected_counts) < 2:
                status = "NO_VIEW_TOO_FEW_ACTIVE_DAYS_KEEP_BUDGET"
            else:
                sd = np.array([np.std(values[active_window[:, k] == 1, k], ddof=1) for k in range(2)])
                if not np.isfinite(sd).all() or (sd <= 0).any():
                    status = "NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET"
                else:
                    # 与标准差倒数归一化相同，避免显式计算极大倒数。
                    weights = np.array([sd[1], sd[0]]) / sd.sum()
                    successful, status = dates[t], "RISK_BUDGET_AVAILABLE"
        target = float(states[t] @ weights) if np.isfinite(states[t]).all() else np.nan
        require(np.isfinite(weights).all() and (weights >= 0).all() and np.isclose(weights.sum(), 1.), "风险预算超出完整无杠杆资金")
        rows.append({"date": dates[t], "panic_budget": weights[0], "learned_budget": weights[1], "panic_sd": sd[0], "learned_sd": sd[1],
            "risk_status": status, "risk_update_scheduled": scheduled, "risk_attempt_origin": attempted,
            "last_successful_risk_origin": successful, "risk_window_start": start, "risk_window_observations": count, "panic_active_days": selected_counts[0], "learned_active_days": selected_counts[1],
            "panic_state": states[t, 0], "learned_state": states[t, 1], "target": target})
    return pd.DataFrame(rows)
