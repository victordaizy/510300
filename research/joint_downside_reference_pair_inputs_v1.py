"""直接优化合成后负收益平方，保留真实抵消、旧预算和完整时点。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require
from research.recent_reference_selection_inputs_v1 import aligned_reference_inputs, MODELS

PRIMARY = "JOINT_DOWNSIDE_REFERENCE_PAIR"


def minimum_joint_downside(returns, previous_weight):
    """在完整区间内求凸目标最小点；多解时取离旧预算最近的解。"""
    values = np.asarray(returns, dtype=float)
    require(values.ndim == 2 and values.shape[1] == 2 and len(values) >= 2 and np.isfinite(values).all(), "共同下行优化需要完整两列收益")
    require((values > -1).all() and np.isfinite(previous_weight) and 0 <= previous_weight <= 1, "共同下行输入或旧预算非法")
    second, difference = values[:, 1], values[:, 0]-values[:, 1]

    def risk(weight):
        losses = np.minimum(second+weight*difference, 0.)
        return float(np.dot(losses, losses)/len(losses))

    def gradient(weight):
        return float(2*np.dot(np.minimum(second+weight*difference, 0.), difference)/len(values))

    prior = float(previous_weight)
    g_prior, g_left, g_right = gradient(prior), gradient(0.), gradient(1.)
    require(np.isfinite([g_prior, g_left, g_right, risk(prior)]).all(), "共同下行目标或导数非有限")
    if g_prior == 0.:
        weight, status = prior, "JOINT_DOWNSIDE_MINIMUM_KEEP_PREVIOUS"
    elif g_left > 0.:
        weight, status = 0., "JOINT_DOWNSIDE_LEFT_BOUNDARY"
    elif g_right < 0.:
        weight, status = 1., "JOINT_DOWNSIDE_RIGHT_BOUNDARY"
    else:
        left, right = (prior, 1.) if g_prior < 0 else (0., prior)
        for _ in range(64):
            middle = (left+right)/2
            if middle == left or middle == right:
                break
            value = gradient(middle)
            if g_prior < 0:
                if value >= 0:
                    right = middle
                else:
                    left = middle
            elif value <= 0:
                left = middle
            else:
                right = middle
        weight, status = (left+right)/2, "JOINT_DOWNSIDE_CONVEX_MINIMUM"
    return {"downside_budget": weight, "continuous_budget": 1.-weight,
        "joint_downside_second_moment": risk(weight), "previous_budget_downside_second_moment": risk(prior),
        "downside_gradient": gradient(weight), "optimizer_status": status}


def joint_downside_reference_frames(data, base_ledgers, parents_by_cost, cfg, start):
    """月首只用已有基础收益求一次预算，再采用各档费用收盘目标。"""
    require(cfg["risk_window"] == 242 and cfg["downside_benchmark"] == 0., "共同下行窗口或下行基准改变")
    first, returns, target_arrays = aligned_reference_inputs(data, base_ledgers, parents_by_cost, {**cfg, "selection_window": 242}, start)
    dates, rows = pd.DatetimeIndex(data.date), []
    weight, risk, prior_risk, derivative = .5, np.nan, np.nan, np.nan
    attempted, successful, window_start = pd.NaT, pd.NaT, pd.NaT
    count, status = 0, "INITIAL_EQUAL_BUDGET_NO_DOWNSIDE_ESTIMATE"
    for t, date in enumerate(dates):
        outside, scheduled = t < first-1 or t == len(dates)-1, False
        if not outside and t >= first and date.to_period("M") != dates[t-1].to_period("M"):
            scheduled, attempted = True, date
            count = min(242, t-first+1)
            window_start = dates[t-count+1]
            values = returns[t-count+1:t+1]
            risk, prior_risk, derivative = np.nan, np.nan, np.nan
            if count < 242:
                status = "NO_VIEW_WARMUP_KEEP_BUDGET"
            elif not np.isfinite(values).all():
                status = "NO_VIEW_INCOMPLETE_WINDOW_KEEP_BUDGET"
            else:
                choice = minimum_joint_downside(values, weight)
                weight = choice["downside_budget"]
                risk, prior_risk, derivative = choice["joint_downside_second_moment"], choice["previous_budget_downside_second_moment"], choice["downside_gradient"]
                status, successful = choice["optimizer_status"], date
        rows.append({"date": date, "origin_index": t, "risk_cost": "BASE", "downside_budget": np.nan if outside else weight,
            "continuous_budget": np.nan if outside else 1.-weight, "joint_downside_second_moment": np.nan if outside else risk,
            "previous_budget_downside_second_moment": np.nan if outside else prior_risk, "downside_gradient": np.nan if outside else derivative,
            "risk_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD" if outside else status, "risk_update_scheduled": scheduled,
            "risk_attempt_origin": pd.NaT if outside else attempted, "last_successful_risk_origin": pd.NaT if outside else successful,
            "risk_window_start": pd.NaT if outside else window_start, "risk_window_observations": 0 if outside else count,
            "downside_reference_return": returns[t, 0], "continuous_reference_return": returns[t, 1]})
    risk_frame = pd.DataFrame(rows)
    weights = risk_frame[["downside_budget", "continuous_budget"]].to_numpy(float)
    frames, summaries = {}, []
    for cost_id, targets in target_arrays.items():
        factors = risk_frame.copy()
        factors["target_cost"] = cost_id
        factors["downside_parent_target"], factors["continuous_parent_target"] = targets[:, 0], targets[:, 1]
        factors["target"] = np.sum(weights*targets, axis=1)
        eligible = factors.iloc[first-1:-1]
        frames[cost_id] = factors
        summaries.append({"cost": cost_id, "decision_origins": len(eligible), "monthly_attempts": int(factors.risk_update_scheduled.sum()),
            "budget_changes": int(eligible.downside_budget.diff().abs().gt(1e-12).sum()),
            "positive_target_origins": int(eligible.target.gt(0).sum()), "zero_target_origins": int(eligible.target.eq(0).sum()),
            "unknown_target_origins": int(eligible.target.isna().sum()), "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries
