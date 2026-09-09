"""复用已有120日趋势因素，在两套保存收盘目标之间选择。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require

PRIMARY = "TREND_REFERENCE_ROUTER"
MODELS = ["VINTAGE_REFERENCE_RISK", "MODEL_SUPPORT_REFERENCE_ROUTER"]


def checked_trend(data):
    """只核对已有因素的历史窗口，缺失保持缺失。"""
    wealth = data.wealth.to_numpy(float)
    require((np.isnan(wealth) | (np.isfinite(wealth) & (wealth > 0))).all(), "含分红财富出现非法值")
    expected = data.wealth / data.wealth.rolling(120, min_periods=120).mean() - 1
    np.testing.assert_allclose(data.sma120, expected, atol=1e-12, rtol=1e-12, equal_nan=True)
    require((np.isnan(data.sma120) | np.isfinite(data.sma120)).all(), "趋势因素出现无限值")
    return data.sma120.to_numpy(float)


def trend_routed_frames(data, parents_by_cost, cfg, start):
    """15:05以明确趋势选择来源，使用对应费用的原收盘目标。"""
    require(cfg["decision_clock"] == "15:05:00" and cfg["trend_window"] == 120 and cfg["trend_threshold"] == 0
        and cfg["weight_band"] == .10, "趋势时钟、窗口、门槛或带宽改变")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "趋势选择需要完整递增日期")
    require(set(parents_by_cost) == set(cfg["costs"]), "趋势选择费用集合不同")
    trend = checked_trend(data)
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])
    require(1 <= first < len(data)-1, "趋势选择缺少准备或执行区间")
    indices = np.arange(first-1, len(data)-1)
    valid = np.zeros(len(data), bool)
    valid[indices] = np.isfinite(trend[indices])
    selected = np.full(len(data), None, object)
    selected[valid & (trend > 0)] = MODELS[0]
    selected[valid & (trend <= 0)] = MODELS[1]
    changes = np.zeros(len(data), bool)
    previous = None
    for t in indices:
        if selected[t] is not None:
            changes[t] = previous is not None and selected[t] != previous
            previous = selected[t]
    state = np.full(len(data), "NO_VIEW_OUTSIDE_DECISION_PERIOD", object)
    state[indices] = "NO_VIEW_TREND_INPUT"
    state[valid & (trend > 0)] = "ABOVE_TRAILING_MEAN"
    state[valid & (trend <= 0)] = "AT_OR_BELOW_TRAILING_MEAN"
    decision_times = np.full(len(data), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    decision_times[indices] = (dates[indices].normalize()+pd.Timedelta(hours=15, minutes=5)).to_numpy()
    source = pd.DataFrame({"date": dates, "origin_index": np.arange(len(data)), "decision_time": decision_times,
        "trend_deviation120": trend, "trend_state": state, "selected_parent": selected, "selection_changed": changes})
    frames, summaries = {}, []
    for cost_id, parents in parents_by_cost.items():
        require(set(parents) == set(MODELS), "趋势选择父来源不同")
        factors = source.copy()
        factors["source_cost"] = cost_id
        target = np.full(len(data), np.nan)
        for j, model in enumerate(MODELS):
            parent = parents[model]
            require(parent.source_cost.eq(cost_id).all() and parent.source_model.eq(model).all(), "趋势选择目标费用或身份不同")
            require(np.array_equal(parent.origin_index, indices), "趋势选择父目标索引不同")
            require(pd.DatetimeIndex(parent.origin).equals(dates[indices]) and pd.DatetimeIndex(parent.execution_date).equals(dates[indices+1]), "趋势选择父目标时钟不同")
            values = parent.reference_weight.to_numpy(float)
            require((np.isnan(values) | (np.isfinite(values) & (values >= 0) & (values <= 1))).all(), "趋势选择父目标超出无融资范围")
            aligned = np.full(len(data), np.nan)
            aligned[indices] = values
            factors["trend_parent_target" if j == 0 else "other_parent_target"] = aligned
            chosen = selected == model
            target[chosen] = aligned[chosen]
        factors["target"] = target
        eligible = factors.iloc[first-1:-1]
        frames[cost_id] = factors
        summaries.append({"cost": cost_id, "decision_origins": len(eligible), "selection_changes": int(eligible.selection_changed.sum()),
            "selected_origins": {str(k): int(v) for k, v in eligible.selected_parent.value_counts().items()},
            "positive_target_origins": int(eligible.target.gt(0).sum()), "zero_target_origins": int(eligible.target.eq(0).sum()),
            "unknown_target_origins": int(eligible.target.isna().sum()), "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries
