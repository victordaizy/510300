"""量、日MACD和价格当前修复的先后顺序；不使用事后低点或未来结果。"""
from __future__ import annotations

import numpy as np
import pandas as pd

POLICIES = ("FRESH_ORDERED_REPAIR", "PRICE_CONFIRMATION")
REQUIRED = ["date", "ac", "ema20", "daily_hist", "up_volume_balance5", "available", "atr20"]


def signals(data):
    d = data.reset_index(drop=True)
    if not set(REQUIRED).issubset(d.columns) or not d.date.is_unique or not d.date.is_monotonic_increasing:
        raise ValueError("修复顺序缺少原字段或日期不完整。")
    volume_start = macd_start = under_start = -1
    previous_volume = previous_macd = np.nan
    previous_above = None
    rows = []
    for i, r in enumerate(d.itertuples(index=False)):
        known = np.isfinite([r.ac, r.ema20, r.daily_hist, r.up_volume_balance5]).all()
        if not known:
            volume_start = macd_start = under_start = -1
            previous_volume = previous_macd = np.nan
            previous_above = None
            rows.append({"date": r.date, "origin_index": i, "status": "NO_VIEW_PRICE_OR_VOLUME_OR_MACD",
                         "price_cross": False, "ordered_repair": False, "known_under_start": -1,
                         "current_volume_positive_start": -1, "current_macd_positive_start": -1,
                         "volume_birth_confirmed": False, "macd_birth_confirmed": False})
            continue
        above = bool(r.ac > r.ema20)
        if r.up_volume_balance5 > 0:
            if not np.isfinite(previous_volume) or previous_volume <= 0:
                volume_start = i
        else:
            volume_start = -1
        if r.daily_hist > 0:
            if not np.isfinite(previous_macd) or previous_macd <= 0:
                macd_start = i
        else:
            macd_start = -1
        if not above and (previous_above is None or previous_above):
            under_start = i
        cross = bool(above and previous_above is False)
        available = bool(r.available)
        volume_birth = bool(volume_start > 0 and np.isfinite(d.up_volume_balance5.iloc[volume_start-1])
                            and d.up_volume_balance5.iloc[volume_start-1] <= 0)
        macd_birth = bool(macd_start > 0 and np.isfinite(d.daily_hist.iloc[macd_start-1])
                          and d.daily_hist.iloc[macd_start-1] <= 0)
        ordered = bool(available and cross and volume_birth and macd_birth
                       and 0 <= under_start <= volume_start < macd_start < i)
        rows.append({"date": r.date, "origin_index": i, "status": "AVAILABLE" if available else "NO_VIEW_ORIGINAL_FEATURE_WARMUP",
                     "price_cross": bool(available and cross), "ordered_repair": ordered,
                     "known_under_start": under_start if cross or not above else -1,
                     "current_volume_positive_start": volume_start, "current_macd_positive_start": macd_start,
                     "volume_birth_confirmed": volume_birth, "macd_birth_confirmed": macd_birth})
        previous_volume, previous_macd, previous_above = r.up_volume_balance5, r.daily_hist, above
        if above:
            under_start = -1
    return pd.DataFrame(rows)


def account_signals(data, sequence, policy):
    if policy not in POLICIES or len(data) != len(sequence):
        raise ValueError("未知固定政策或原点长度不同。")
    if not pd.DatetimeIndex(data.date).equals(pd.DatetimeIndex(sequence.date)):
        raise ValueError("修复顺序与日线日期不同。")
    mask = sequence.ordered_repair if policy == "FRESH_ORDERED_REPAIR" else sequence.price_cross
    rows = sequence.copy()
    rows["entry_event"] = mask.astype(bool)
    rows["event_id"] = [f"{policy}_{i}" if flag else None for i, flag in enumerate(mask)]
    rows["atr"] = data.atr20.to_numpy(float)
    rows["stop_index"], rows["target_index"] = np.nan, np.nan
    return rows
