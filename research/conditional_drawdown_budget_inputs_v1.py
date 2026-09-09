"""复用连续参考的完整日历，按月首收盘更新条件回撤预算。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require
from research.conditional_drawdown_budget_optimizer_v1 import optimal_drawdown_budget

DETAILS = ["optimal_drawdown", "selected_drawdown", "optimal_budget_lower", "optimal_budget_upper"]


def outside_row(date, states):
    return {"date": date, "panic_budget": np.nan, "learned_budget": np.nan, "panic_sd": np.nan, "learned_sd": np.nan,
            "difference_variance": np.nan, **{key: np.nan for key in DETAILS}, "linear_programs": 0,
            "risk_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD", "risk_update_scheduled": False, "risk_attempt_origin": pd.NaT,
            "last_successful_risk_origin": pd.NaT, "risk_window_start": pd.NaT, "risk_window_observations": 0,
            "panic_state": states[0], "learned_state": states[1], "target": np.nan}


def budget_frame(dates, reference_returns, expert_states, first, window=242, confidence=.95):
    dates = pd.DatetimeIndex(dates)
    returns, states = np.asarray(reference_returns, float), np.asarray(expert_states, float)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "参考日历必须严格递增")
    require(returns.shape == states.shape == (len(dates), 2), "两条参考的形状与日历不符")
    require(1 <= first < len(dates)-1 and isinstance(window, int) and window >= 2, "参考起点或风险窗口无效")
    require((np.isnan(returns) | (np.isfinite(returns) & (returns > -1))).all(), "参考收益存在无穷或净值失效")
    require((np.isnan(states) | np.isin(states, [0., 1.])).all(), "参考状态必须为零、一或缺失")
    rows, certificates = [], []
    weights, sd = np.array([.5, .5]), np.full(2, np.nan)
    status = "INITIAL_EQUAL_BUDGET_NO_RISK_ESTIMATE"
    attempted, successful, start = pd.NaT, pd.NaT, pd.NaT
    count, difference_variance = 0, np.nan
    details = {**{key: np.nan for key in DETAILS}, "linear_programs": 0}
    for t in range(len(dates)):
        scheduled = False
        if t < first-1 or t == len(dates)-1:
            rows.append(outside_row(dates[t], states[t])); continue
        if t >= first and dates[t].to_period("M") != dates[t-1].to_period("M"):
            scheduled, attempted = True, dates[t]
            count = min(window, t-first+1)
            start = dates[t-count+1]
            values = returns[t-count+1:t+1]
            sd, difference_variance = np.full(2, np.nan), np.nan
            details = {**{key: np.nan for key in DETAILS}, "linear_programs": 0}
            if count < window:
                status = "NO_VIEW_WARMUP_KEEP_BUDGET"
            elif not np.isfinite(values).all():
                status = "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
            else:
                sd = np.std(values, axis=0, ddof=1)
                if not np.isfinite(sd).all() or (sd <= 0).any():
                    status = "NO_VIEW_ZERO_OR_INVALID_RISK_KEEP_BUDGET"
                else:
                    difference_variance = float(np.var(values[:, 0]-values[:, 1], ddof=1))
                    if not np.isfinite(difference_variance) or difference_variance <= 0:
                        status = "NO_VIEW_DEGENERATE_DIFFERENCE_KEEP_BUDGET"
                    else:
                        optimized = optimal_drawdown_budget(values, weights[0], confidence)
                        certificates.append({"origin": str(dates[t].date()), "origin_index": t, "window_start_index": t-count+1,
                            "previous_budget": float(weights[0]), "status": optimized["status"], "solves": optimized["certificate"]})
                        status = optimized["status"]
                        details = {key: optimized[key] for key in details}
                        if status == "CONDITIONAL_DRAWDOWN_BUDGET_AVAILABLE":
                            weights = np.array([optimized["panic_budget"], 1-optimized["panic_budget"]])
                            successful = dates[t]
        target = float(states[t] @ weights) if np.isfinite(states[t]).all() else np.nan
        require(np.isfinite(weights).all() and (weights >= 0).all() and np.isclose(weights.sum(), 1.), "条件回撤预算超出完整无杠杆资金")
        rows.append({"date": dates[t], "panic_budget": weights[0], "learned_budget": weights[1], "panic_sd": sd[0], "learned_sd": sd[1],
            "difference_variance": difference_variance, **details, "risk_status": status, "risk_update_scheduled": scheduled,
            "risk_attempt_origin": attempted, "last_successful_risk_origin": successful, "risk_window_start": start,
            "risk_window_observations": count, "panic_state": states[t, 0], "learned_state": states[t, 1], "target": target})
    return pd.DataFrame(rows), certificates


def prefix_factors(full, source):
    require(len(source) <= len(full) and pd.DatetimeIndex(full.date.iloc[:len(source)]).equals(pd.DatetimeIndex(source.date)), "裁切期不是同一完整日期前缀")
    frame = full.iloc[:len(source)].copy()
    frame.loc[frame.index[-1], :] = outside_row(source.date.iloc[-1], source[["panic_state", "learned_state"]].iloc[-1].to_numpy(float))
    return frame
