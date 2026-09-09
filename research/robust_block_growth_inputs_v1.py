"""每月用三个已实现历史子阶段的最弱对数增长分配来源。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = "ROBUST_BLOCK_GROWTH_REFERENCE_PAIR"
MODELS = ["VINTAGE_REFERENCE_RISK", "TREND_NOISE_REFERENCE_BLEND"]


def block_growth(returns, weight):
    combined = weight*returns[:, 0]+(1-weight)*returns[:, 1]
    values = np.log1p(combined)
    return np.array([part.mean() for part in np.split(values, [81, 162])])


def estimate_budget(returns, previous, cfg):
    require(cfg["training_rows"] == 242 and cfg["block_lengths"] == [81, 81, 80] and cfg["optimizer_iterations"] == 80 and
        cfg["numeric_tolerance"] == 1e-12, "最弱增长预算窗口或求解规则改变")
    returns = np.asarray(returns, float)
    require(returns.shape == (242, 2) and np.isfinite(returns).all() and (returns > -1).all(), "预算优化需要完整合法242日两来源收益")
    require(np.isfinite(previous) and 0 <= previous <= 1, "上次有效预算非法")

    def objective(weight):
        return float(block_growth(returns, weight).min())

    left, right = 0., 1.
    for _ in range(80):
        a, b = left+(right-left)/3, right-(right-left)/3
        fa, fb = objective(a), objective(b)
        if fa < fb:
            left = a
        elif fa > fb:
            right = b
        else:
            left, right = a, b
    candidate = (left+right)/2
    if candidate <= 1e-12:
        candidate = 0.
    elif candidate >= 1-1e-12:
        candidate = 1.
    candidates = sorted(set([0., 1., float(previous), candidate]))
    values = {weight: objective(weight) for weight in candidates}
    maximum = max(values.values())
    tied = [weight for weight in candidates if maximum-values[weight] <= 1e-12]
    chosen = float(previous) if float(previous) in tied else min(tied)
    return chosen, {"previous_worst_block_growth": values[float(previous)], "chosen_worst_block_growth": values[chosen],
        "chosen_block_growths": block_growth(returns, chosen).tolist(), "numerical_candidate_weight": candidate}


def robust_block_growth_frames(data, parents_by_cost, base_ledgers, cfg, start):
    require(cfg["combination"] == "MONTHLY_MAXIMUM_WORST_THREE_BLOCK_LOG_GROWTH" and cfg["initial_131_budget"] == 0,
        "月度最弱增长预算或初始143规则改变")
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    dates, indices = pd.DatetimeIndex(data.date), np.arange(first-1, len(data)-1)
    require(set(base_ledgers) == set(MODELS), "基础费用训练来源集合不同")
    history = np.full((len(data), 2), np.nan)
    for column, model in enumerate(MODELS):
        ledger = base_ledgers[model]
        require(ledger.source_cost.eq("BASE").all() and ledger.source_model.eq(model).all(), "预算训练混用费用或模型")
        require(pd.DatetimeIndex(ledger.date).equals(dates[first:]) and ledger.mark_clock.iloc[:-1].eq("CLOSE").all(), "预算训练账本未按完整当日收盘对齐")
        values = ledger.net_return.to_numpy(float)
        require((np.isnan(values) | (np.isfinite(values) & (values > -1))).all(), "预算训练收益有无穷值或不合法财富")
        history[first:, column] = values
    # 最后一个交易日是统一开盘终点，从不进入当日收盘判断或训练。
    history[-1] = np.nan
    weights = np.full(len(data), np.nan)
    states = np.full(len(data), "OUTSIDE_DECISION_CALENDAR", dtype=object)
    estimate_dates = np.full(len(data), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    current, previous_valid, state, estimate_date = 0., 0., "WARMUP_USE143", None
    estimates = []
    for t in indices:
        monthly = t == first-1 or dates[t].to_period("M") != dates[t-1].to_period("M")
        if monthly:
            estimate_date = dates[t].to_datetime64()
            available_rows = t-first+1
            detail = {"origin": str(dates[t].date()), "origin_index": int(t), "decision_time": str(dates[t]+pd.Timedelta(hours=15, minutes=5)),
                "training_source_cost": "BASE", "available_calendar_rows": int(available_rows), "previous_valid_131_budget": previous_valid,
                "training_start": None, "training_end": None}
            if available_rows < 242:
                current, state = 0., "WARMUP_USE143"
            else:
                values = history[t-241:t+1]
                detail.update(training_start=str(dates[t-241].date()), training_end=str(dates[t].date()),
                    complete_pair_rows=int(np.isfinite(values).all(axis=1).sum()))
                if not np.isfinite(values).all():
                    current, state = np.nan, "NO_VIEW_INCOMPLETE_REFERENCE_RETURNS"
                else:
                    current, optimization = estimate_budget(values, previous_valid, cfg)
                    previous_valid, state = current, "BUDGET_OPTIMIZATION_COMPLETE"
                    detail.update(optimization)
            estimates.append({**detail, "status": state, "reference_budget131": current})
        weights[t], states[t], estimate_dates[t] = current, state, estimate_date
    summaries = []
    count = sum(row["status"] == "BUDGET_OPTIMIZATION_COMPLETE" for row in estimates)
    for cost, frame in frames.items():
        frame["reference_budget131"] = weights
        frame["reference_budget143"] = 1-weights
        frame["budget_estimation_state"] = states
        frame["budget_estimate_origin"] = estimate_dates
        x, y = (frame[model+"_parent_target"].to_numpy(float) for model in MODELS)
        known = np.isfinite(weights) & np.isfinite(x) & np.isfinite(y)
        target = np.full(len(data), np.nan)
        target[known] = np.clip(weights[known]*x[known]+(1-weights[known])*y[known], 0, 1)
        frame["target"] = target
        eligible = frame.iloc[first-1:-1]
        summaries.append({"cost": cost, "decision_origins": len(indices), "positive_target_origins": int(eligible.target.gt(0).sum()),
            "zero_target_origins": int(eligible.target.eq(0).sum()), "unknown_target_origins": int(eligible.target.isna().sum()),
            "monthly_budget_checks": len(estimates), "shared_base_budget_optimizations": count,
            "mean_131_budget": float(eligible.reference_budget131.mean()), "maximum_131_budget": float(eligible.reference_budget131.max()),
            "positive_131_budget_origins": int(eligible.reference_budget131.gt(0).sum()), "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries, estimates
