"""共用保存收盘目标的身份、费用、完整日历与下一开盘对齐。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require


def aligned_target_frames(data, parents_by_cost, models, cfg, start):
    require(cfg["decision_clock"] == "15:05:00" and cfg["weight_band"] == .1, "保存目标时钟或带宽不同")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "保存目标需要完整递增日历")
    require(set(parents_by_cost) == set(cfg["costs"]), "保存目标费用集合不同")
    matches = np.flatnonzero(dates >= pd.Timestamp(start))
    require(len(matches) > 0, "保存目标缺少研究起点")
    first = int(matches[0])
    require(1 <= first < len(data)-1, "保存目标缺少准备收盘或执行区间")
    indices = np.arange(first-1, len(data)-1)
    times = np.full(len(data), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    times[indices] = (dates[indices]+pd.Timedelta(hours=15, minutes=5)).to_numpy()
    frames = {}
    for cost_id, parents in parents_by_cost.items():
        require(set(parents) == set(models), "保存父目标集合不同")
        factors = pd.DataFrame({"date": dates, "origin_index": np.arange(len(data)), "decision_time": times, "source_cost": cost_id})
        for model in models:
            parent = parents[model]
            require(parent.source_cost.eq(cost_id).all() and parent.source_model.eq(model).all(), "父目标费用或身份不同")
            require(np.array_equal(parent.origin_index, indices), "父目标索引不同")
            require(pd.DatetimeIndex(parent.origin).equals(dates[indices]), "父目标收盘日期不同")
            require(pd.DatetimeIndex(parent.execution_date).equals(dates[indices+1]), "父目标执行不是下一开盘")
            values = parent.reference_weight.to_numpy(float)
            require((np.isnan(values) | (np.isfinite(values) & (values >= 0) & (values <= 1))).all(), "父目标超出无融资范围")
            aligned = np.full(len(data), np.nan)
            aligned[indices] = values
            factors[model+"_parent_target"] = aligned
        frames[cost_id] = factors
    return frames, first
