"""只用当前及过去状态，还原上次价格确认、失效和当前动量是否保留。"""

from __future__ import annotations

import numpy as np
import pandas as pd


CLASSES = (
    "MOMENTUM_RETAINED", "MOMENTUM_REBUILT", "MOMENTUM_NOT_POSITIVE",
    "NO_PREVIOUS_CONFIRMATION", "NO_VIEW_CONTINUITY",
)


def known_paths(data: pd.DataFrame, sequence: pd.DataFrame) -> pd.DataFrame:
    d, s = data.reset_index(drop=True), sequence.reset_index(drop=True)
    required = {"date", "ac", "ema20", "daily_hist", "up_volume_balance5", "available"}
    if not required.issubset(d.columns) or len(d) != len(s):
        raise ValueError("路径还原缺少原字段或序列长度不同。")
    if not d.date.is_unique or not d.date.is_monotonic_increasing or not d.date.equals(s.date):
        raise ValueError("日线原点和原顺序没有按同一唯一日期排列。")
    previous_confirmation = -1
    first_failure = -1
    rows = []
    for i in range(len(d)):
        x, y = d.iloc[i], s.iloc[i]
        known = bool(np.isfinite([x.ac, x.ema20, x.daily_hist, x.up_volume_balance5]).all())
        previous_at_origin = previous_confirmation
        failure_at_origin = first_failure
        cross = bool(y.price_cross)
        category = "NOT_A_CONFIRMATION_EVENT"
        retained = False
        volume_state = "NOT_A_CONFIRMATION_EVENT"
        if not known:
            previous_confirmation = first_failure = -1
            previous_at_origin = failure_at_origin = -1
            category = volume_state = "NO_VIEW_CONTINUITY"
        else:
            if previous_confirmation >= 0 and x.ac <= x.ema20 and first_failure < 0:
                first_failure = failure_at_origin = i
            if cross:
                start = int(y.current_macd_positive_start)
                vstart = int(y.current_volume_positive_start)
                if start > i or vstart > i:
                    raise ValueError("当前正段起点包含未来原点。")
                if previous_confirmation < 0:
                    category = "NO_PREVIOUS_CONFIRMATION"
                elif first_failure < 0:
                    category = "NO_VIEW_CONTINUITY"
                elif x.daily_hist <= 0:
                    category = "MOMENTUM_NOT_POSITIVE"
                elif not y.macd_birth_confirmed:
                    category = "NO_VIEW_CONTINUITY"
                elif start <= previous_confirmation:
                    category = "MOMENTUM_RETAINED"
                    retained = True
                else:
                    category = "MOMENTUM_REBUILT"
                if x.up_volume_balance5 <= 0:
                    volume_state = "VOLUME_NOT_POSITIVE"
                elif not y.volume_birth_confirmed or previous_confirmation < 0:
                    volume_state = "NO_VIEW_PREVIOUS_VOLUME_RUN"
                elif vstart <= previous_confirmation:
                    volume_state = "VOLUME_RETAINED"
                else:
                    volume_state = "VOLUME_REBUILT"
        rows.append({
            "date": x.date, "origin_index": i, "price_confirmation": cross,
            "known_previous_confirmation_index": previous_at_origin,
            "known_previous_confirmation_date": d.date.iloc[previous_at_origin] if previous_at_origin >= 0 else pd.NaT,
            "known_first_price_failure_index": failure_at_origin,
            "known_first_price_failure_date": d.date.iloc[failure_at_origin] if failure_at_origin >= 0 else pd.NaT,
            "known_sessions_since_previous_confirmation": i - previous_at_origin if previous_at_origin >= 0 else None,
            "known_sessions_since_first_failure": i - failure_at_origin if failure_at_origin >= 0 else None,
            "current_macd_positive_start": int(y.current_macd_positive_start),
            "current_volume_positive_start": int(y.current_volume_positive_start),
            "momentum_path_class": category, "momentum_retained_through_failure": retained,
            "volume_path_state": volume_state,
            "current_daily_hist": float(x.daily_hist), "current_volume_balance5": float(x.up_volume_balance5),
            "original_feature_available": bool(x.available),
        })
        if known and cross:
            previous_confirmation, first_failure = i, -1
    result = pd.DataFrame(rows)
    for name in ["known_sessions_since_previous_confirmation", "known_sessions_since_first_failure"]:
        result[name] = pd.array(result[name], dtype="Int64")
    return result


def descriptive_returns(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    x = x[np.isfinite(x)]
    positive, negative = x[x > 0], x[x < 0]
    p = len(positive) / len(x) if len(x) else np.nan
    q = len(negative) / len(x) if len(x) else np.nan
    b = positive.mean() / -negative.mean() if len(positive) and len(negative) else np.nan
    return {
        "observations": len(x), "wins": len(positive), "losses": len(negative),
        "win_fraction": p, "mean_return": float(x.mean()) if len(x) else np.nan,
        "mean_positive_over_mean_negative": float(b), "p_times_b": float(p * b),
        "standard_expectation_loss_units": float(p * b - q),
        "independent_strategy_metrics": False,
    }
