"""用已有趋势与波动幅度连续分配两套来源，不拟合额外参数。"""
import numpy as np
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.trend_reference_router_inputs_v1 import checked_trend
from research.intraday_overnight_increment_v1 import require

PRIMARY = "TREND_NOISE_REFERENCE_BLEND"
MODELS = ["VINTAGE_REFERENCE_RISK", "MODEL_SUPPORT_REFERENCE_ROUTER"]


def market_amplitudes(data, cfg):
    require(cfg["trend_window"] == 120 and cfg["noise_window"] == 20 and cfg["annual_days"] == 242, "趋势或波动尺度改变")
    trend = checked_trend(data)
    expected = data.total_simple.rolling(20, min_periods=20).std(ddof=1)*np.sqrt(242)
    np.testing.assert_allclose(data.vol20, expected, atol=1e-12, rtol=1e-12, equal_nan=True)
    volatility = data.vol20.to_numpy(float)
    require((np.isnan(volatility) | (np.isfinite(volatility) & (volatility >= 0))).all(), "波动因素出现负值或无穷值")
    return np.maximum(trend, 0), volatility*np.sqrt(120/242)


def trend_noise_frames(data, parents_by_cost, cfg, start):
    require(cfg["combination"] == "POSITIVE_TREND_DIVIDED_BY_TREND_PLUS_NOISE", "连续预算定义改变")
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    positive, noise = market_amplitudes(data, cfg)
    known = np.isfinite(positive) & np.isfinite(noise)
    budget = np.full(len(data), np.nan)
    denominator = positive+noise
    budget[known] = np.divide(positive[known], denominator[known], out=np.zeros(known.sum()), where=denominator[known] > 0)
    summaries = []
    for cost, frame in frames.items():
        frame["positive_trend_amplitude120"] = positive
        frame["noise_amplitude120"] = noise
        frame["reference_budget131"] = budget
        frame["reference_budget139"] = 1-budget
        x, y = (frame[model+"_parent_target"].to_numpy(float) for model in MODELS)
        available = known & np.isfinite(x) & np.isfinite(y)
        target = np.full(len(data), np.nan)
        target[available] = np.clip(budget[available]*x[available]+(1-budget[available])*y[available], 0, 1)
        frame["target"] = target
        eligible = frame.iloc[first-1:-1]
        summaries.append({"cost": cost, "decision_origins": len(eligible), "positive_target_origins": int(eligible.target.gt(0).sum()),
            "zero_target_origins": int(eligible.target.eq(0).sum()), "unknown_target_origins": int(eligible.target.isna().sum()),
            "mean_131_budget": float(eligible.reference_budget131.mean()), "maximum_131_budget": float(eligible.reference_budget131.max()),
            "positive_131_budget_origins": int(eligible.reference_budget131.gt(0).sum()), "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries
