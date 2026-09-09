"""只对既有最终股票目标设置上限，直接复用已保存的普通波动乘数。"""
import numpy as np
import pandas as pd
from research.intraday_overnight_increment_v1 import require

PRIMARY = "FINAL_TARGET_VOLATILITY_CAP"
PARENT = "JOINT_DOWNSIDE_REFERENCE_PAIR"
MULTIPLIER = "ROLLING_VARIANCE_BUDGET_CONTROL_multiplier"
VOLATILITY = "realized_annual_volatility20"


def capped_reference_targets(data, risk, references, cfg, start, cost_id):
    """同费用收盘目标与既有上限取较小者，未知目标保持未知。"""
    require(cfg["target_volatility"] == .10 and cfg["weight_band"] == .10, "最终上限的原风险水平或带宽改变")
    require(cost_id in cfg["costs"] and cost_id in references, "最终上限没有对应费用来源")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "最终上限需要完整递增日历")
    require(pd.DatetimeIndex(risk.date).equals(dates), "最终上限风险日历不一致")
    caps, vol = risk[MULTIPLIER].to_numpy(float), risk[VOLATILITY].to_numpy(float)
    require(np.isfinite(caps).all() and ((caps >= 0) & (caps <= 1)).all(), "既有风险上限缺失或超出零至一")
    require((np.isnan(vol) | (np.isfinite(vol) & (vol >= 0))).all(), "既有普通波动非法")
    first = int(np.flatnonzero(data.date.ge(pd.Timestamp(start)))[0])
    require(1 <= first < len(data)-1, "最终上限缺少准备收盘或执行区间")
    indices = np.arange(first-1, len(data)-1)
    parent = references[cost_id]
    require(parent.source_model.eq(PARENT).all() and parent.source_cost.eq(cost_id).all(), "最终上限来源身份或费用不符")
    require(np.array_equal(parent.origin_index, indices), "最终上限父目标索引不同")
    require(pd.DatetimeIndex(parent.origin).equals(dates[indices]), "最终上限父目标不是对应收盘")
    require(pd.DatetimeIndex(parent.execution_date).equals(dates[indices+1]), "最终上限父目标不是下一开盘")
    values = parent.reference_weight.to_numpy(float)
    require((np.isnan(values) | (np.isfinite(values) & (values >= 0) & (values <= 1))).all(), "最终上限父目标超出无融资范围")
    previous = np.full(len(data), np.nan)
    previous[indices] = values
    targets = np.minimum(previous, caps)
    factors = pd.DataFrame({"date": dates, "origin_index": np.arange(len(data)), "source_cost": cost_id,
        "parent_target": previous, "volatility_cap": caps, "realized_annual_volatility20": vol,
        "cap_binding": np.isfinite(previous) & (previous > caps), "target": targets,
        "risk_view_status": np.where(np.isfinite(vol) & (vol > 0), "EXISTING_REALIZED_VOLATILITY_VIEW", "NO_NEW_VOLATILITY_VIEW_CARRIED_CAP")})
    eligible = factors.iloc[first-1:-1]
    summary = {"cost": cost_id, "decision_origins": len(eligible), "cap_binding_origins": int(eligible.cap_binding.sum()),
        "positive_target_origins": int(eligible.target.gt(0).sum()), "zero_target_origins": int(eligible.target.eq(0).sum()),
        "unknown_target_origins": int(eligible.target.isna().sum()), "new_model_fits": 0, "new_reference_accounts": 0}
    return factors, summary
