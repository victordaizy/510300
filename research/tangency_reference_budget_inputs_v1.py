"""根据当时完整均值与协方差求两策略的非负切点预算。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def tangency_budget(mean, covariance, previous):
    mean, cov, old = np.asarray(mean, float), np.asarray(covariance, float), np.asarray(previous, float)
    require(mean.shape == (2,) and cov.shape == (2, 2) and old.shape == (2,), "切点预算形状不符")
    if not np.isfinite(mean).all() or not np.isfinite(cov).all():
        return old.copy(), "NO_VIEW_INVALID_MOMENTS_KEEP_BUDGET", float("nan")
    if mean.max() <= 0:
        return np.zeros(2), "EXPLICIT_CASH_NO_POSITIVE_PAST_MEAN", float("nan")
    eigen = np.linalg.eigvalsh(cov)
    if eigen[-1] <= 0 or eigen[0] <= np.finfo(float).eps*eigen[-1]:
        return old.copy(), "NO_VIEW_NONPOSITIVE_DEFINITE_COVARIANCE_KEEP_BUDGET", float("nan")
    base, delta = mean[1], mean[0]-mean[1]
    variance, cross = cov[1, 1], cov[0, 1]-cov[1, 1]
    difference = cov[0, 0]+cov[1, 1]-2*cov[0, 1]
    denominator = base*difference-delta*cross
    previous_mix = float(old[0]/old.sum()) if old.sum() > 0 else .5
    candidates = [0., 1., previous_mix]
    if denominator != 0:
        critical = (delta*variance-base*cross)/denominator
        if np.isfinite(critical) and 0 < critical < 1:
            candidates.append(float(critical))
    choices = []
    for w in candidates:
        weights = np.array([w, 1-w])
        score = float(weights@mean/np.sqrt(weights@cov@weights))
        choices.append((score, w))
    best = max(s for s, _ in choices)
    require(best > 0 and np.isfinite(best), "正均值资料没有给出有效正切点")
    w = min((w for s, w in choices if abs(s-best) <= 1e-12), key=lambda w: (abs(w-previous_mix), w))
    return np.array([w, 1-w]), "TANGENCY_BUDGET_AVAILABLE", best


def budget_frame(dates, reference_returns, expert_states, first, window=242):
    dates = pd.DatetimeIndex(dates)
    returns, states = np.asarray(reference_returns, float), np.asarray(expert_states, float)
    require(dates.is_monotonic_increasing and not dates.has_duplicates and returns.shape == states.shape == (len(dates), 2), "切点参考日历或形状不符")
    require(1 <= first < len(dates)-1 and isinstance(window, int) and window >= 2 and (np.isnan(states) | np.isin(states, [0., 1.])).all(), "切点起点、窗口或状态不符")
    weights = np.array([.5, .5])
    mean, cov, score = np.full(2, np.nan), np.full((2, 2), np.nan), float("nan")
    status, attempted, successful = "INITIAL_EQUAL_BUDGET_NO_MOMENT_ESTIMATE", pd.NaT, pd.NaT
    rows, count, left = [], 0, pd.NaT
    for t in range(len(dates)):
        outside = t < first-1 or t == len(dates)-1
        scheduled = not outside and t >= first and dates[t].to_period("M") != dates[t-1].to_period("M")
        if scheduled:
            attempted, count = dates[t], min(window, t-first+1)
            left = dates[t-count+1]
            values = returns[t-count+1:t+1]
            mean, cov, score = np.full(2, np.nan), np.full((2, 2), np.nan), float("nan")
            if count < window:
                status = "NO_VIEW_WARMUP_KEEP_BUDGET"
            elif not np.isfinite(values).all() or (values <= -1).any():
                status = "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
            else:
                mean, cov = values.mean(axis=0), np.cov(values, rowvar=False, ddof=1)
                weights, status, score = tangency_budget(mean, cov, weights)
                if status in ["TANGENCY_BUDGET_AVAILABLE", "EXPLICIT_CASH_NO_POSITIVE_PAST_MEAN"]:
                    successful = dates[t]
        require(np.isfinite(weights).all() and (weights >= 0).all() and weights.sum() <= 1+1e-12, "切点预算超出自身资金")
        target = float("nan")
        if not outside:
            if weights.sum() == 0:
                target = 0.
            elif np.isfinite(states[t]).all():
                target = float(states[t]@weights)
        rows.append({"date": dates[t], "panic_budget": float("nan") if outside else weights[0], "learned_budget": float("nan") if outside else weights[1],
            "budget_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD" if outside else status, "budget_update_scheduled": scheduled, "moment_attempt_origin": attempted,
            "last_successful_budget_origin": successful, "window_start": left, "window_observations": count, "panic_mean": mean[0], "learned_mean": mean[1],
            "panic_variance": cov[0, 0], "learned_variance": cov[1, 1], "reference_covariance": cov[0, 1], "estimated_daily_ratio": score,
            "panic_state": states[t, 0], "learned_state": states[t, 1], "target": target, "panic_reference_return": returns[t, 0], "learned_reference_return": returns[t, 1]})
    return pd.DataFrame(rows)
