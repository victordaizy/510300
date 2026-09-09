"""以分红修正日内高低区间与隔夜方差缩放保存目标。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = "RANGE_OVERNIGHT_RISK"
MODELS = ["TREND_NOISE_REFERENCE_BLEND"]


def range_overnight_statistics(data, cfg):
    require(cfg["risk_window"] == 20 and cfg["target_volatility"] == .10 and cfg["annual_days"] == 242, "高低区间风险窗口或目标改变")
    for column in ["open", "high", "low", "close", "previous_close"]:
        values = data[column].to_numpy(float)
        require((np.isnan(values) | (np.isfinite(values) & (values > 0))).all(), "价格出现非正值或无穷值")
    dividend = data.dividend.to_numpy(float)
    require((np.isnan(dividend) | (np.isfinite(dividend) & (dividend >= 0))).all(), "分红出现负值或无穷值")
    quotes = data[["open", "high", "low", "close"]]
    known_quotes = quotes.notna().all(axis=1)
    require((~known_quotes | (data.high.ge(quotes.max(axis=1)) & data.low.le(quotes.min(axis=1)))).all(), "已知开高低收顺序矛盾")
    np.testing.assert_allclose(data.previous_close, data.close.shift(), atol=0, rtol=0, equal_nan=True)
    overnight = np.log((data.open+data.dividend)/data.previous_close)
    np.testing.assert_allclose(data.overnight_log, overnight, atol=1e-12, rtol=1e-12, equal_nan=True)
    adjusted_range = np.log((data.high+data.dividend)/(data.low+data.dividend))
    range_variance = adjusted_range.pow(2)/(4*np.log(2))
    overnight_variance = overnight.rolling(20, min_periods=20).var(ddof=1)
    range_mean = range_variance.rolling(20, min_periods=20).mean()
    variance = overnight_variance+range_mean
    require((variance.isna() | (np.isfinite(variance) & variance.ge(0))).all(), "合成风险方差非法")
    risk = np.sqrt(242*variance)
    multiplier = np.full(len(data), np.nan)
    known = np.isfinite(risk)
    multiplier[known] = np.minimum(1., np.divide(.10, risk.to_numpy()[known], out=np.ones(known.sum()), where=risk.to_numpy()[known] > 0))
    return pd.DataFrame({"date": data.date.to_numpy(), "adjusted_range_log": adjusted_range.to_numpy(),
        "range_variance_daily": range_variance.to_numpy(), "overnight_variance20": overnight_variance.to_numpy(),
        "range_mean_variance20": range_mean.to_numpy(), "risk_volatility20": risk.to_numpy(), "risk_multiplier": multiplier})


def range_overnight_frames(data, parents_by_cost, cfg, start):
    require(cfg["combination"] == "PARENT143_TIMES_RANGE_PLUS_OVERNIGHT_RISK_MULTIPLIER", "高低区间仓位定义改变")
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    statistics = range_overnight_statistics(data, cfg)
    summaries = []
    for cost, frame in frames.items():
        for column in statistics.columns.drop("date"):
            frame[column] = statistics[column].to_numpy()
        parent = frame[MODELS[0]+"_parent_target"].to_numpy(float)
        multiplier = frame.risk_multiplier.to_numpy(float)
        known = np.isfinite(parent) & np.isfinite(multiplier)
        target = np.full(len(data), np.nan)
        target[known] = parent[known]*multiplier[known]
        frame["target"] = target
        eligible = frame.iloc[first-1:-1]
        summaries.append({"cost": cost, "decision_origins": len(eligible), "positive_target_origins": int(eligible.target.gt(0).sum()),
            "zero_target_origins": int(eligible.target.eq(0).sum()), "unknown_target_origins": int(eligible.target.isna().sum()),
            "mean_risk_multiplier": float(eligible.risk_multiplier.mean()), "minimum_risk_multiplier": float(eligible.risk_multiplier.min()),
            "maximum_risk_multiplier": float(eligible.risk_multiplier.max()), "reduced_multiplier_origins": int(eligible.risk_multiplier.lt(1).sum()),
            "new_model_fits": 0, "new_reference_accounts": 0})
    return frames, summaries
