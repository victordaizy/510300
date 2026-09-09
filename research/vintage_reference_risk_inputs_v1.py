"""按同费用参考收盘意向乘既有风险乘数，不重算参考模型。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require

MULTIPLIER = "ROLLING_VARIANCE_BUDGET_CONTROL_multiplier"
VOLATILITY = "realized_annual_volatility20"


def vintage_risk_targets(data, risk, references, cfg, start, cost_id):
    """准备和终点未知收盘保留空值；只合成已知收盘的新目标。"""
    require(cfg["target_volatility"] == .10 and cfg["weight_band"] == .10, "本轮只允许既定风险水平与交易带宽")
    require(cost_id in {"BASE", "STRESS"} and cost_id in references, "没有匹配费用的已保存参考")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "组合行情日历必须完整递增")
    require(pd.DatetimeIndex(risk.date).equals(dates), "组合风险乘数日历不同")
    multiplier = risk[MULTIPLIER].to_numpy(float)
    require(np.isfinite(multiplier).all() and ((multiplier >= 0) & (multiplier <= 1)).all(), "保存风险乘数缺失或超出不融资范围")
    ref = references[cost_id]
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])
    require(first > 0 and len(data) > first, "组合需要准备收盘与至少一个实际交易日")
    indices = np.arange(first-1, len(data)-1)
    require(np.array_equal(ref.origin_index.to_numpy(int), indices), "参考原始收盘索引不连续或来自不同历史")
    require(pd.DatetimeIndex(ref.origin).equals(dates[indices]), "参考意向不是对应的已知收盘")
    require(pd.DatetimeIndex(ref.execution_date).equals(dates[indices+1]), "参考意向被挪到执行日或使用未来时点")
    values = ref.reference_weight.to_numpy(float)
    require((np.isnan(values) | np.isin(values, [0., 1.])).all(), "参考意向必须是明确零一或无观点")
    intent = np.full(len(data), np.nan)
    intent[indices] = values
    target = intent*multiplier
    status = np.full(len(data), "NO_VIEW_OUTSIDE_REFERENCE_DECISION_RANGE", dtype=object)
    status[indices] = np.where(np.isfinite(values), "REFERENCE_AND_RISK_TARGET_AVAILABLE", "NO_VIEW_REFERENCE_INTENT")
    vol = risk[VOLATILITY].to_numpy(float)
    frame = pd.DataFrame({"date": dates, "origin_index": np.arange(len(data)), "reference_cost": cost_id,
        "reference_intent": intent, "realized_annual_volatility20": vol, "risk_multiplier": multiplier,
        "risk_view_status": np.where(np.isfinite(vol) & (vol > 0), "EXISTING_REALIZED_VOLATILITY_VIEW", "NO_NEW_VOLATILITY_VIEW_CARRIED_MULTIPLIER"),
        "target": target, "target_status": status})
    summary = {"cost": cost_id, "calendar_rows": len(frame), "reference_decision_rows": len(indices),
        "unknown_reference_rows": int(np.isnan(values).sum()), "positive_targets": int(np.sum(target[indices] > 0)),
        "zero_targets": int(np.sum(target[indices] == 0)), "new_models_fit": 0, "new_reference_accounts": 0}
    return frame, summary
