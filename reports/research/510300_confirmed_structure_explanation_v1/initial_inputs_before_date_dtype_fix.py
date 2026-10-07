"""严格两日确认的日收盘交替枢轴与双抬升状态；只向前更新。"""
from __future__ import annotations

import numpy as np
import pandas as pd

POLICIES = ("CONFIRMED_RISING_STRUCTURE", "PRICE_CONFIRMATION")


def structure_states(data: pd.DataFrame):
    needed = {"date", "close", "dividend"}
    if not needed.issubset(data.columns):
        raise ValueError("结构状态缺少日期、原价收盘或当日现金分红。")
    d = data.reset_index(drop=True)
    dates = pd.DatetimeIndex(d.date)
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("结构原点日期必须唯一递增。")
    close = pd.to_numeric(d.close, errors="coerce").to_numpy(float)
    dividends = pd.to_numeric(d.dividend, errors="coerce").to_numpy(float)
    if (np.isfinite(dividends) & (dividends < 0)).any():
        raise ValueError("结构现金分红不能是负值。")
    price = close + np.cumsum(dividends)
    valid = np.isfinite(price) & np.isfinite(close) & (close > 0)
    sequence, states, arrivals = [], [], []
    previous_known, previous_active = False, False
    for t in range(len(d)):
        low_today, high_today, action = False, False, "NO_NEW_PIVOT"
        center = t-2
        if t >= 4 and valid[t-4:t+1].all():
            others = price[[center-2, center-1, center+1, center+2]]
            low_today = bool((price[center] < others).all())
            high_today = bool((price[center] > others).all())
            if low_today or high_today:
                kind = "LOW" if low_today else "HIGH"
                current = {"kind": kind, "center_index": center, "confirmation_index": t,
                           "price": float(price[center])}
                replaced = -1
                if not sequence or sequence[-1]["kind"] != kind:
                    sequence.append(current)
                    action = "APPENDED_ALTERNATING"
                else:
                    last = sequence[-1]
                    extreme = current["price"] < last["price"] if kind == "LOW" else current["price"] > last["price"]
                    if extreme:
                        replaced = last["center_index"]
                        sequence[-1] = current
                        action = "REPLACED_CURRENT_TAIL_MORE_EXTREME"
                    else:
                        action = "IGNORED_SAME_KIND_LESS_EXTREME"
                arrivals.append({"confirmation_date": dates[t], "center_date": dates[center],
                                 **current, "action": action, "replaced_tail_center_index": replaced})
        known = bool(valid[t])
        complete = len(sequence) >= 4
        high_rising, low_rising, latest_low = False, False, np.nan
        detail = {}
        for name in ("H1", "H2", "L1", "L2"):
            detail.update({f"{name}_index": -1, f"{name}_confirmation_index": -1,
                           f"{name}_date": pd.NaT, f"{name}_confirmation_date": pd.NaT, f"{name}_price": np.nan})
        if complete:
            tail = sequence[-4:]
            if any(a["kind"] == b["kind"] for a, b in zip(tail, tail[1:])):
                raise AssertionError("已知尾部枢轴没有交替。")
            highs = [p for p in tail if p["kind"] == "HIGH"]
            lows = [p for p in tail if p["kind"] == "LOW"]
            for name, pivot in zip(("H1", "H2", "L1", "L2"), (*highs, *lows)):
                if pivot["confirmation_index"] > t or pivot["center_index"]+2 != pivot["confirmation_index"]:
                    raise AssertionError("使用尚未确认的高低点。")
                detail.update({f"{name}_index": pivot["center_index"], f"{name}_confirmation_index": pivot["confirmation_index"],
                               f"{name}_date": dates[pivot["center_index"]], f"{name}_confirmation_date": dates[pivot["confirmation_index"]],
                               f"{name}_price": pivot["price"]})
            high_rising = highs[-1]["price"] > highs[-2]["price"]
            low_rising = lows[-1]["price"] > lows[-2]["price"]
            latest_low = float(lows[-1]["price"])
        active = bool(known and complete and high_rising and low_rising and price[t] > latest_low)
        status = ("NO_VIEW" if not known else "INSUFFICIENT_CONFIRMED_PIVOTS" if not complete
                  else "NONRISING_CONFIRMED_STRUCTURE" if not (high_rising and low_rising)
                  else "LATEST_CONFIRMED_LOW_FAILED" if price[t] <= latest_low else "RISING_CONFIRMED_STRUCTURE")
        born = bool(known and previous_known and active and not previous_active)
        states.append({
            "date": dates[t], "origin_index": t, "known_cash_adjusted_close": float(price[t]),
            "structure_known": known, "structure_status": status, "structure_active": active if known else pd.NA,
            "structure_onset": born, "rule_exit": bool(known and not active),
            "confirmed_sequence_length": len(sequence), "confirmed_low_today": low_today, "confirmed_high_today": high_today,
            "pivot_update_action": action, "two_highs_rising": bool(high_rising), "two_lows_rising": bool(low_rising),
            "latest_confirmed_low_price": latest_low, **detail,
        })
        previous_known, previous_active = known, active
    frame = pd.DataFrame(states)
    frame["structure_active"] = pd.array(frame.structure_active, dtype="boolean")
    for name in ("H1", "H2", "L1", "L2"):
        frame[f"{name}_date"] = pd.to_datetime(frame[f"{name}_date"])
        frame[f"{name}_confirmation_date"] = pd.to_datetime(frame[f"{name}_confirmation_date"])
    pivot_columns = ["confirmation_date", "center_date", "kind", "center_index", "confirmation_index", "price", "action",
                     "replaced_tail_center_index"]
    return frame, pd.DataFrame(arrivals, columns=pivot_columns)


def account_signals(data, states, policy):
    if policy not in POLICIES or not data.date.reset_index(drop=True).equals(states.date):
        raise ValueError("结构政策或已知状态日期不一致。")
    if policy == "PRICE_CONFIRMATION":
        from research import point_fresh_repair_order_inputs_v1 as original_price
        return original_price.account_signals(data, original_price.signals(data), "PRICE_CONFIRMATION")
    rows = states.copy()
    rows["entry_event"] = states.structure_onset
    rows["event_id"] = [f"{policy}_{i}" if x else None for i, x in enumerate(rows.entry_event)]
    rows["atr"] = data.atr20.to_numpy(float)
    rows["stop_index"] = states.latest_confirmed_low_price
    rows["target_index"] = np.nan
    return rows
