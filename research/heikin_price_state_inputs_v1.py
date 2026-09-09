"""从真实含分红价格构造平均K线，保持计算价格和成交价格分离。"""
import numpy as np
import pandas as pd
from research.saved_parent_target_alignment_v1 import aligned_target_frames
from research.intraday_overnight_increment_v1 import require

PRIMARY = "HEIKIN_PRICE_STATE"
CANDIDATES = {PRIMARY: "平均K线独立进出场", "HEIKIN_CONFIRMED_REFERENCE": "平均K线确认原143目标"}
MODELS = ["TREND_NOISE_REFERENCE_BLEND"]


def heikin_candles(data, tolerance=1e-12):
    require(tolerance == 1e-12 and len(data) > 0, "平均K线数值容差或日历为空")
    dates = pd.DatetimeIndex(data.date)
    require(dates.is_monotonic_increasing and not dates.has_duplicates, "平均K线日历不是唯一递增")
    fields = ["open", "high", "low", "close", "previous_close", "dividend", "wealth"]
    values = data[fields].to_numpy(float)
    require((np.isnan(values) | np.isfinite(values)).all(), "平均K线来源包含无穷值")
    for column in ["open", "high", "low", "close", "previous_close", "wealth"]:
        x = data[column].to_numpy(float)
        require((np.isnan(x) | (x > 0)).all(), "平均K线已知价格或财富非正")
    require((data.dividend.isna() | data.dividend.ge(0)).all(), "平均K线现金分红为负")
    raw = data[["open", "high", "low", "close"]].to_numpy(float)
    complete = np.isfinite(raw).all(axis=1)
    require((raw[complete, 1] >= np.maximum(raw[complete, 0], raw[complete, 3])).all() and
        (raw[complete, 2] <= np.minimum(raw[complete, 0], raw[complete, 3])).all(), "平均K线原始最高最低次序不合法")
    previous = data.previous_close.to_numpy(float)
    paired = np.isfinite(previous[1:]) & np.isfinite(raw[:-1, 3])
    require(np.allclose(previous[1:][paired], raw[:-1, 3][paired], atol=1e-12, rtol=1e-12), "平均K线前收盘不等于原始上一收盘")
    wealth, dividends = data.wealth.to_numpy(float), data.dividend.to_numpy(float)
    scales = np.r_[wealth[0]/(raw[0, 3]+dividends[0]), wealth[:-1]/previous[1:]]
    adjusted = (raw+dividends[:, None])*scales[:, None]
    known = complete & np.isfinite(dividends) & np.isfinite(wealth) & np.isfinite(scales)
    adjusted[~known] = np.nan
    require(np.allclose(adjusted[known, 3], wealth[known], atol=1e-12, rtol=1e-12), "平均K线分红换算收盘与已有累计财富不同")
    candle = np.full((len(data), 4), np.nan)
    for t in range(len(data)):
        if known[t]:
            candle[t, 3] = adjusted[t].sum()/4
        if t == 0 and known[t]:
            candle[t, 0] = (adjusted[t, 0]+adjusted[t, 3])/2
        elif t > 0 and np.isfinite(candle[t-1, [0, 3]]).all():
            candle[t, 0] = (candle[t-1, 0]+candle[t-1, 3])/2
        if known[t] and np.isfinite(candle[t, 0]):
            candle[t, 1] = max(adjusted[t, 1], candle[t, 0], candle[t, 3])
            candle[t, 2] = min(adjusted[t, 2], candle[t, 0], candle[t, 3])
    body = candle[:, 3]/candle[:, 0]-1
    direction = np.where(np.isnan(body), np.nan, np.where(body > tolerance, 1., np.where(body < -tolerance, -1., 0.)))
    output = pd.DataFrame({"date": dates})
    for j, name in enumerate(["open", "high", "low", "close"]):
        output["economic_"+name] = adjusted[:, j]
        output["heikin_"+name] = candle[:, j]
    output["heikin_body_ratio"] = body
    output["heikin_direction"] = direction
    return output


def heikin_price_frames(data, parents_by_cost, cfg, start):
    require(cfg["combination"] == "HEIKIN_STANDALONE_AND_REFERENCE_CONFIRMATION" and cfg["candidate_models"] == list(CANDIDATES), "平均K线候选集合改变")
    frames, first = aligned_target_frames(data, parents_by_cost, MODELS, cfg, start)
    candles = heikin_candles(data, cfg["numeric_tolerance"])
    indices = np.arange(first-1, len(data)-1)
    direction = candles.heikin_direction.to_numpy(float)
    summaries = []
    for cost, frame in frames.items():
        for column in candles.columns[1:]:
            frame[column] = candles[column].to_numpy()
        parent = frame[MODELS[0]+"_parent_target"].to_numpy(float)
        independent, confirmed = np.full(len(data), np.nan), np.full(len(data), np.nan)
        known = indices[np.isfinite(direction[indices])]
        independent[known] = (direction[known] > 0).astype(float)
        zero = indices[parent[indices] == 0]
        confirmed[zero] = 0.
        eligible = indices[(parent[indices] > 0) & np.isfinite(direction[indices])]
        confirmed[eligible] = np.where(direction[eligible] > 0, parent[eligible], 0.)
        for model, target in zip(CANDIDATES, [independent, confirmed]):
            frame[model+"_target"] = target
            x = target[indices]
            summaries.append({"model": model, "cost": cost, "decision_origins": len(indices), "positive_target_origins": int((x > 0).sum()),
                "zero_target_origins": int((x == 0).sum()), "unknown_target_origins": int(np.isnan(x).sum()),
                "bullish_candle_origins": int((direction[indices] > 0).sum()), "flat_candle_origins": int((direction[indices] == 0).sum()),
                "bearish_candle_origins": int((direction[indices] < 0).sum()), "parent_positive_vetoed": int(((parent[indices] > 0) & (x == 0)).sum()) if model != PRIMARY else None})
    return frames, summaries
