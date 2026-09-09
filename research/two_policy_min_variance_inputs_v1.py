"""两条既有参考策略按过去组合方差最小化分预算，完整保留日历。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def budget_frame(dates, reference_returns, expert_states, first, window=242):
    dates = pd.DatetimeIndex(dates)
    returns = np.asarray(reference_returns, dtype=float)
    states = np.asarray(expert_states, dtype=float)
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
    covariance, difference_variance, raw_panic_budget = np.nan, np.nan, np.nan
    for t in range(len(dates)):
        scheduled = False
        if t < first-1 or t == len(dates)-1:
            rows.append({"date": dates[t], "panic_budget": np.nan, "learned_budget": np.nan, "panic_sd": np.nan, "learned_sd": np.nan,
                "reference_covariance": np.nan, "difference_variance": np.nan, "raw_panic_budget": np.nan,
                "risk_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD", "risk_update_scheduled": False, "risk_attempt_origin": pd.NaT,
                "last_successful_risk_origin": pd.NaT, "risk_window_start": pd.NaT, "risk_window_observations": 0,
                "panic_state": states[t, 0], "learned_state": states[t, 1], "target": np.nan})
            continue
        if t >= first and dates[t].to_period("M") != dates[t-1].to_period("M"):
            scheduled, attempted = True, dates[t]
            count = min(window, t-first+1)
            start = dates[t-count+1]
            values = returns[t-count+1:t+1]
            sd = np.full(2, np.nan)
            covariance, difference_variance, raw_panic_budget = np.nan, np.nan, np.nan
            if count < window:
                status = "NO_VIEW_WARMUP_KEEP_BUDGET"
            elif not np.isfinite(values).all():
                status = "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
            else:
                sd = np.std(values, axis=0, ddof=1)
                if not np.isfinite(sd).all() or (sd <= 0).any():
                    status = "NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET"
                else:
                    centered = values-values.mean(axis=0)
                    covariance = float((centered[:, 0]*centered[:, 1]).sum()/(window-1))
                    # 直接计算收益差的方差，避免两项近似相等方差相减造成负值。
                    difference_variance = float(np.var(values[:, 0]-values[:, 1], ddof=1))
                    if not np.isfinite(difference_variance) or difference_variance <= 0:
                        status = "NO_VIEW_DEGENERATE_DIFFERENCE_KEEP_BUDGET"
                    else:
                        raw_panic_budget = float((sd[1]**2-covariance)/difference_variance)
                        panic_weight = float(np.clip(raw_panic_budget, 0., 1.))
                        weights = np.array([panic_weight, 1.-panic_weight])
                        successful, status = dates[t], "MIN_VARIANCE_BUDGET_AVAILABLE"
        target = float(states[t] @ weights) if np.isfinite(states[t]).all() else np.nan
        require(np.isfinite(weights).all() and (weights >= 0).all() and np.isclose(weights.sum(), 1.), "风险预算超出完整无杠杆资金")
        rows.append({"date": dates[t], "panic_budget": weights[0], "learned_budget": weights[1], "panic_sd": sd[0], "learned_sd": sd[1],
            "reference_covariance": covariance, "difference_variance": difference_variance, "raw_panic_budget": raw_panic_budget,
            "risk_status": status, "risk_update_scheduled": scheduled, "risk_attempt_origin": attempted,
            "last_successful_risk_origin": successful, "risk_window_start": start, "risk_window_observations": count,
            "panic_state": states[t, 0], "learned_state": states[t, 1], "target": target})
    return pd.DataFrame(rows)
