"""只按当时已形成的原训练支持记录选择策略，不预测未来或按年份选择。"""
from bisect import bisect_right
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require

PRIMARY = "MODEL_SUPPORT_REFERENCE_ROUTER"
MODELS = ["VINTAGE_REFERENCE_RISK", "JOINT_DOWNSIDE_REFERENCE_PAIR"]
INSUFFICIENT = "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS"


def validate_support_records(data, records):
    """检查训练记录的日期和支持状态；未来记录只用于未来判断。"""
    times = []
    for record in records:
        fit_index = record["fit_index"]
        fit_origin, fit_time = pd.Timestamp(record["fit_origin"]), pd.Timestamp(record["fit_time"])
        require(isinstance(fit_index, int) and fit_index >= 0, "模型支持索引非法")
        require(fit_time.normalize() == fit_origin.normalize() and fit_time >= fit_origin.normalize()+pd.Timedelta(hours=15), "模型拟合时点早于其完整收盘资料")
        if fit_index < len(data):
            require(pd.Timestamp(data.date.iloc[fit_index]).normalize() == fit_origin.normalize(), "模型支持拟合日与原完整日历不一致")
        count, rows = record["training_cycle_count"], record["training_rows"]
        require(count == len(record["training_cycles"]) and count >= 0 and rows >= 0, "模型支持周期或行数不一致")
        latest_exit = record["latest_exit_index"]
        if latest_exit is not None:
            require(0 <= latest_exit <= fit_index and pd.Timestamp(record["latest_exit_date"]) <= fit_origin, "模型支持使用未来退出周期")
            if latest_exit < len(data):
                require(pd.Timestamp(data.date.iloc[latest_exit]).normalize() == pd.Timestamp(record["latest_exit_date"]).normalize(), "训练退出日期与索引不同")
        status = record["status"]
        require(status in {INSUFFICIENT, "FIT_COMPLETE"}, "不认识的模型支持状态不能自动选策略")
        if status == "FIT_COMPLETE":
            require(record["eligible_for_fit"] and count >= 10 and rows >= 100 and record["model"] is not None
                and record["missing_feature_rows"] == 0 and record["failure"] is None and latest_exit is not None, "成熟模型不满足原支持或完整性要求")
        else:
            require(not record["eligible_for_fit"] and record["model"] is None and (count < 10 or rows < 100), "支持不足记录与原门槛不一致")
        times.append(fit_time)
    require(times == sorted(set(times)) and [r["fit_index"] for r in records] == sorted(set(r["fit_index"] for r in records)), "训练支持记录时点必须唯一递增")
    return times


def support_routed_frames(data, records, parents_by_cost, cfg, start):
    """每天15:05可用的最近记录决定来源，两费用共享支持状态。"""
    require(cfg["decision_clock"] == "15:05:00" and cfg["weight_band"] == .10 and cfg["minimum_training_cycles"] == 10 and cfg["minimum_training_rows"] == 100,
        "模型支持时钟、门槛或带宽改变")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "模型支持路由需要完整递增日期")
    require(set(parents_by_cost) == set(cfg["costs"]), "模型支持路由费用集合不同")
    times = validate_support_records(data, records)
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])
    require(1 <= first < len(data)-1, "模型支持路由缺少准备或执行区间")
    indices = np.arange(first-1, len(data)-1)
    rows = []
    previous = None
    for t, date in enumerate(dates):
        outside = t < first-1 or t == len(data)-1
        decision_time = date.normalize()+pd.Timedelta(hours=15, minutes=5)
        k = bisect_right(times, decision_time)-1 if not outside else -1
        record = records[k] if k >= 0 else None
        mature = record is not None and record["status"] == "FIT_COMPLETE"
        selected = None if outside else MODELS[1] if mature else MODELS[0]
        changed = not outside and previous is not None and selected != previous
        if not outside:
            previous = selected
        rows.append({"date": date, "origin_index": t, "decision_time": pd.NaT if outside else decision_time,
            "model_support_status": "NO_VIEW_OUTSIDE_DECISION_PERIOD" if outside else record["status"] if record else "NO_RECORD_AVAILABLE",
            "support_fit_index": np.nan if record is None else record["fit_index"], "support_fit_time": pd.NaT if record is None else pd.Timestamp(record["fit_time"]),
            "support_training_cycles": np.nan if record is None else record["training_cycle_count"], "support_training_rows": np.nan if record is None else record["training_rows"],
            "selected_parent": selected, "selection_changed": changed})
    support = pd.DataFrame(rows)
    frames, summaries = {}, []
    for cost_id, parents in parents_by_cost.items():
        require(set(parents) == set(MODELS), "模型支持父目标来源集合不同")
        factors = support.copy()
        factors["source_cost"] = cost_id
        target = np.full(len(data), np.nan)
        for j, model in enumerate(MODELS):
            parent = parents[model]
            require(parent.source_cost.eq(cost_id).all() and parent.source_model.eq(model).all(), "模型支持目标费用或身份不同")
            require(np.array_equal(parent.origin_index, indices), "模型支持父目标索引不同")
            require(pd.DatetimeIndex(parent.origin).equals(dates[indices]) and pd.DatetimeIndex(parent.execution_date).equals(dates[indices+1]), "模型支持父目标收盘或下一开盘不同")
            values = parent.reference_weight.to_numpy(float)
            require((np.isnan(values) | (np.isfinite(values) & (values >= 0) & (values <= 1))).all(), "模型支持父目标超出无融资范围")
            aligned = np.full(len(data), np.nan)
            aligned[indices] = values
            factors["fallback_parent_target" if j == 0 else "mature_parent_target"] = aligned
            chosen = factors.selected_parent.eq(model).to_numpy()
            target[chosen] = aligned[chosen]
        factors["target"] = target
        eligible = factors.iloc[first-1:-1]
        frames[cost_id] = factors
        summaries.append({"cost": cost_id, "decision_origins": len(eligible), "selection_changes": int(eligible.selection_changed.sum()),
            "selected_origins": {str(k): int(v) for k, v in eligible.selected_parent.value_counts().items()},
            "positive_target_origins": int(eligible.target.gt(0).sum()), "zero_target_origins": int(eligible.target.eq(0).sum()),
            "unknown_target_origins": int(eligible.target.isna().sum()), "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries
