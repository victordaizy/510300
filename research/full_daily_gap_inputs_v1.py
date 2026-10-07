"""日线相邻整日区间的完整向上缺口；现金平移比较使用整数价格刻度。"""
from __future__ import annotations

import numpy as np
import pandas as pd

PRIMARY = "FULL_DAILY_UP_GAP"
TICK = .001


def quote_tick(value):
    """仅消除原报价的二进制表示误差，非刻度价格返回未知。"""
    if not np.isfinite(value) or value <= 0:
        return None
    scaled = float(value) / TICK
    rounded = int(round(scaled))
    return rounded if abs(scaled-rounded) <= 1e-8 else None


def gap_states(data):
    required = {"date", "open", "high", "low", "close", "dividend"}
    if not required.issubset(data) or data.empty or not data.date.is_monotonic_increasing or not data.date.is_unique:
        raise ValueError("完整日线缺口需要有序唯一原报价及已发生分红。")
    rows, shift, shift_known = [], 0, True
    previous = None
    for i, row in enumerate(data.itertuples(index=False)):
        dividend = float(row.dividend)
        valid_dividend = bool(np.isfinite(dividend) and dividend >= 0 and abs(dividend/TICK-round(dividend/TICK)) <= 1e-8)
        shift_known = bool(shift_known and valid_dividend)
        if shift_known:
            shift += int(round(dividend/TICK))
        raw = {key: quote_tick(float(getattr(row, key))) for key in ("open", "high", "low", "close")}
        coherent = all(value is not None for value in raw.values())
        if coherent:
            coherent = raw["low"] <= min(raw["open"], raw["close"]) <= max(raw["open"], raw["close"]) <= raw["high"]
        known = bool(coherent and shift_known)
        current = {key: value+shift if known else -1 for key, value in raw.items()}
        pair_known = bool(known and previous is not None and previous["known"])
        up = bool(pair_known and current["low"] > previous["high"])
        down = bool(pair_known and current["high"] < previous["low"])
        lower = previous["high"] if up else -1
        upper = current["low"] if up else -1
        status = ("NO_PREVIOUS_QUOTE" if previous is None else "NO_VIEW_CURRENT_OR_PRIOR_QUOTE" if not pair_known
                  else "FULL_UP_GAP" if up else "FULL_DOWN_GAP_OBSERVATION_ONLY" if down else "OVERLAPPING_OR_TOUCHING_RANGES")
        rows.append({
            "date": row.date, "origin_index": i, "current_quote_known": known,
            "previous_and_current_quote_known": pair_known, "cash_shift_known": shift_known,
            "cash_shift_ticks": shift if shift_known else -1,
            "known_cash_open_ticks": current["open"], "known_cash_high_ticks": current["high"],
            "known_cash_low_ticks": current["low"], "known_cash_close_ticks": current["close"],
            "gap_status": status, "gap_event": up, "down_gap_observation": down,
            "gap_anchor_id": f"DAILY_GAP_{i}_PREVIOUS_{i-1}" if up else None,
            "gap_lower_ticks": lower, "gap_upper_ticks": upper,
            "gap_width_ticks": upper-lower if up else 0,
            "gap_lower_reference_index": i-1 if up else -1,
            "gap_lower_reference_date": data.date.iloc[i-1] if up else pd.NaT,
        })
        previous = {**current, "known": known}
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame.date).astype("datetime64[ns]")
    frame["gap_lower_reference_date"] = pd.to_datetime(frame.gap_lower_reference_date).astype("datetime64[ns]")
    frame["gap_anchor_id"] = pd.array(frame.gap_anchor_id, dtype="string")
    return frame


def account_signals(data, states, policy):
    if not data.date.reset_index(drop=True).equals(states.date) or policy not in (PRIMARY, "PRICE_CONFIRMATION"):
        raise ValueError("完整缺口政策或原点日期不一致。")
    if policy == "PRICE_CONFIRMATION":
        from research import point_fresh_repair_order_inputs_v1 as old
        frame = old.account_signals(data, old.signals(data), policy)
        for name in ("cash_shift_ticks", "cash_shift_known"):
            frame[name] = states[name].to_numpy()
        return frame
    frame = states.copy()
    frame["entry_event"] = frame.gap_event
    frame["event_id"] = pd.Series([f"{PRIMARY}_{i}" if event else None for i, event in enumerate(frame.gap_event)], dtype=object)
    frame["stop_index"] = np.where(frame.gap_event, frame.gap_lower_ticks*TICK, np.nan)
    frame["target_index"] = np.nan
    frame["atr"] = data.atr20.to_numpy(float)
    frame["rule_exit"] = False
    return frame
