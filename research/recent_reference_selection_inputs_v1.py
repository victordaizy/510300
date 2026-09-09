"""复用既有月首历史夏普控制器，明确基础评分与对应费用目标的边界。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require
from research.past_sharpe_parent_selector_inputs_v1 import selection_frame, PARENTS as LEGACY_PARENTS

MODELS = ["DOWNSIDE_REFERENCE_RISK", "CONTINUOUS_REFERENCE_MIN_VARIANCE"]
PRIMARY = "RECENT_REFERENCE_SELECTION"


def aligned_reference_inputs(data, base_ledgers, parents_by_cost, cfg, start):
    """完整日历对齐；准备期和末日开盘收益不能成为收盘评分资料。"""
    require(cfg["selection_window"] == 242 and cfg["annual_days"] == 242 and cfg["weight_band"] == .10, "本轮选择窗口、年化或带宽改变")
    require(set(base_ledgers) == set(MODELS) and set(parents_by_cost) == set(cfg["costs"]) and "BASE" in parents_by_cost, "本轮来源或费用集合不同")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "历史选择需要完整递增日期")
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])
    require(1 <= first < len(data)-1, "历史选择缺少准备或执行区间")
    indices = np.arange(first-1, len(data)-1)
    returns = np.full((len(data), 2), np.nan)
    for j, model in enumerate(MODELS):
        ledger = base_ledgers[model]
        require(ledger.source_cost.eq("BASE").all() and ledger.source_model.eq(model).all(), "历史选择评分必须来自原基础费用账户")
        require(pd.DatetimeIndex(ledger.date).equals(dates[first:]), "历史选择父收益日历不完整")
        require(ledger.mark_clock.iloc[:-1].eq("CLOSE").all() and ledger.mark_clock.iloc[-1] == "OPEN_TERMINAL", "历史选择父收益的收盘与终点开盘时钟不同")
        values = ledger.net_return.to_numpy(float)
        require((np.isnan(values) | (np.isfinite(values) & (values > -1))).all(), "历史选择父收益非法")
        returns[first:len(data)-1, j] = values[:-1]
    target_arrays = {}
    for cost_id, parents in parents_by_cost.items():
        require(set(parents) == set(MODELS), "历史选择父目标来源集合不同")
        values = np.full((len(data), 2), np.nan)
        for j, model in enumerate(MODELS):
            parent = parents[model]
            require(parent.source_cost.eq(cost_id).all() and parent.source_model.eq(model).all(), "历史选择目标费用或身份不同")
            require(np.array_equal(parent.origin_index, indices), "历史选择目标索引不同")
            require(pd.DatetimeIndex(parent.origin).equals(dates[indices]), "历史选择目标不是对应收盘")
            require(pd.DatetimeIndex(parent.execution_date).equals(dates[indices+1]), "历史选择目标不是下一开盘")
            target = parent.reference_weight.to_numpy(float)
            require((np.isnan(target) | (np.isfinite(target) & (target >= 0) & (target <= 1))).all(), "历史选择父目标非法")
            values[indices, j] = target
        target_arrays[cost_id] = values
    return first, returns, target_arrays


def selected_reference_frames(data, base_ledgers, parents_by_cost, cfg, start):
    """两档费用只计算一次基础评分，再采用各自被选来源的目标。"""
    first, returns, target_arrays = aligned_reference_inputs(data, base_ledgers, parents_by_cost, cfg, start)
    original = selection_frame(data.date, returns, target_arrays["BASE"], first, window=242, annual_days=242)
    selection = original.rename(columns={"two_parent_target": "downside_parent_target", "three_parent_target": "continuous_parent_target",
        "two_past_sharpe": "downside_past_sharpe", "three_past_sharpe": "continuous_past_sharpe"}).copy()
    selection["selected_parent"] = selection.selected_parent.replace(dict(zip(LEGACY_PARENTS, MODELS)))
    selection["selection_status"] = selection.selection_status.replace({"INITIAL_TWO_POLICY_PARENT": "INITIAL_DOWNSIDE_REFERENCE",
        "EQUAL_POSITIVE_SCORE_KEEP_PARENT_OR_TWO_AFTER_CASH": "EQUAL_POSITIVE_SCORE_KEEP_PARENT_OR_DOWNSIDE_AFTER_CASH"})
    selection["origin_index"] = np.arange(len(data))
    selection["scoring_cost"] = "BASE"
    selection["downside_scoring_return"] = returns[:, 0]
    selection["continuous_scoring_return"] = returns[:, 1]
    frames, summaries = {}, []
    for cost_id, values in target_arrays.items():
        frame = selection.copy()
        frame["downside_parent_target"] = values[:, 0]
        frame["continuous_parent_target"] = values[:, 1]
        selected = frame.selected_parent.to_numpy()
        target = np.full(len(data), np.nan)
        target[selected == "CASH"] = 0.
        for j, model in enumerate(MODELS):
            mask = selected == model
            target[mask] = values[mask, j]
        frame["target"] = target
        frame["target_cost"] = cost_id
        if cost_id == "BASE":
            np.testing.assert_allclose(target, original.target, atol=0, rtol=0, equal_nan=True)
        eligible = frame.iloc[first-1:-1]
        frames[cost_id] = frame
        summaries.append({"cost": cost_id, "decision_origins": len(eligible),
            "monthly_attempts": int(frame.selection_update_scheduled.sum()), "selection_changes": int(frame.selection_changed.sum()),
            "selected_origins": {str(k): int(v) for k, v in eligible.selected_parent.value_counts().items()},
            "positive_target_origins": int(eligible.target.gt(0).sum()), "zero_target_origins": int(eligible.target.eq(0).sum()),
            "unknown_target_origins": int(eligible.target.isna().sum()), "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries
