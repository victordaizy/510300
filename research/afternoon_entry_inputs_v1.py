"""在下午只读取已完成历史与当前已知价格，计算原六十日进入条件。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def partial_entry(data, t, price):
    result = {"entry_signal": False, "factor_status": "NO_VIEW_INCOMPLETE_INPUT", "previous_d60": np.nan,
        "partial_d60": np.nan, "partial_overnight_log": np.nan, "partial_intraday_log": np.nan,
        "partial_difference": np.nan, "signal_price": price}
    if t < 60 or not np.isfinite(price) or price <= 0:
        return result
    valid = data.feature_valid.iloc[t - 1]
    if not (pd.notna(valid) and bool(valid)):
        return result
    known = np.array([data.open.iloc[t], data.close.iloc[t - 1], data.dividend.iloc[t]], dtype=float)
    history = (data.intraday_log.iloc[t - 60:t] - data.overnight_log.iloc[t - 60:t]).to_numpy(float)
    if not np.isfinite(known).all() or not np.isfinite(history).all() or (known[:2] <= 0).any() or known[2] < 0:
        return result
    opening, previous, dividend = known
    night = np.log((opening + dividend) / previous)
    intraday = np.log((price + dividend) / (opening + dividend))
    partial = np.r_[history[1:], intraday - night]
    original_scale = np.std(history, ddof=1) * np.sqrt(60)
    partial_scale = np.std(partial, ddof=1) * np.sqrt(60)
    result.update(partial_overnight_log=night, partial_intraday_log=intraday, partial_difference=intraday - night)
    if not np.isfinite(original_scale) or not np.isfinite(partial_scale) or min(original_scale, partial_scale) <= 0:
        result["factor_status"] = "NO_VIEW_ZERO_OR_INVALID_SCALE"
        return result
    previous_d60, partial_d60 = history.sum() / original_scale, partial.sum() / partial_scale
    result.update(factor_status="PARTIAL_ENTRY_VIEW_AVAILABLE", previous_d60=float(previous_d60), partial_d60=float(partial_d60),
        entry_signal=bool(previous_d60 > 1 and partial_d60 > 1))
    return result


class EntryPreview:
    def __init__(self, data):
        self.data = data

    def __call__(self, t, price, signal_time):
        return partial_entry(self.data, t, price)
