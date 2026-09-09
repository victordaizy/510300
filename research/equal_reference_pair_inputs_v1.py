"""把两条匹配费用的已知收盘目标各占一半，不重分配未知预算。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require

MODELS = ["DOWNSIDE_REFERENCE_RISK", "CONTINUOUS_REFERENCE_MIN_VARIANCE"]
PRIMARY = "EQUAL_REFERENCE_PAIR"


def equal_reference_targets(data, parents, cfg, start, cost_id):
    """每条输入仍属于原参考；输出用于一个独立实际账户。"""
    require(cfg["reference_weights"] == [.5, .5] and cfg["weight_band"] == .10, "固定各半或既有带宽发生改变")
    require(set(parents) == set(MODELS), "各半组合缺少或增加了未登记来源")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "各半组合行情日历不完整")
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])
    require(first > 0, "各半组合缺少准备收盘")
    indices = np.arange(first-1, len(data)-1)
    columns = []
    for model in MODELS:
        parent = parents[model]
        require(parent.source_cost.eq(cost_id).all() and parent.source_model.eq(model).all(), "各半组合费用或来源身份错误")
        require(np.array_equal(parent.origin_index, indices), "各半组合参考收盘索引不同")
        require(pd.DatetimeIndex(parent.origin).equals(dates[indices]), "各半组合参考收盘日期不同")
        require(pd.DatetimeIndex(parent.execution_date).equals(dates[indices+1]), "各半组合参考不是下一开盘")
        values = parent.reference_weight.to_numpy(float)
        require((np.isnan(values) | (np.isfinite(values) & (values >= 0) & (values <= 1))).all(), "各半组合参考目标非法")
        full = np.full(len(data), np.nan)
        full[indices] = values
        columns.append(full)
    a, b = columns
    known = np.isfinite(a) & np.isfinite(b)
    target = np.full(len(data), np.nan)
    target[known] = .5*a[known]+.5*b[known]
    status = np.full(len(data), "NO_VIEW_ANY_REFERENCE_UNKNOWN", dtype=object)
    status[known] = "TWO_KNOWN_REFERENCES_EQUAL_WEIGHT"
    status[:first-1] = "NO_VIEW_OUTSIDE_DECISION_RANGE"
    status[-1] = "NO_VIEW_OUTSIDE_DECISION_RANGE"
    factors = pd.DataFrame({"date": dates, "origin_index": np.arange(len(data)), "source_cost": cost_id,
        "downside_reference_target": a, "continuous_reference_target": b, "target": target, "target_status": status})
    summary = {"cost": cost_id, "calendar_rows": len(data), "decision_origins": len(indices),
        "positive_target_origins": int((target[indices] > 0).sum()), "zero_target_origins": int((target[indices] == 0).sum()),
        "unknown_target_origins": int(np.isnan(target[indices]).sum()), "new_model_fits": 0, "new_reference_accounts": 0}
    return factors, summary
