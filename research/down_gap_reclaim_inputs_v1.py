"""完整向下日线缺口的最新锚和首次收盘收复，全部使用当时整数现金坐标。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.full_daily_gap_inputs_v1 import gap_states, TICK

PRIMARY = "DOWN_GAP_RECLAIM"


def reclaim_states(data):
    """新向下缺口替换旧锚；首次严格收复后消耗，不反复重试。"""
    coordinate = gap_states(data)
    fields = [name for name in coordinate if not name.startswith("gap_") and name != "down_gap_observation"]
    frame = coordinate[fields].copy()
    rows, active = [], None
    for i, day in enumerate(coordinate.itertuples(index=False)):
        born, replaced, reclaimed = bool(day.down_gap_observation), None, False
        if born:
            replaced = active["anchor_id"] if active is not None else None
            active = {"anchor_id": f"DOWN_GAP_{i}_PREVIOUS_{i-1}", "index": i,
                      "lower": int(day.known_cash_high_ticks),
                      "upper": int(coordinate.known_cash_low_ticks.iloc[i-1])}
        snapshot = active.copy() if active is not None else None
        known = bool(day.current_quote_known)
        if snapshot is not None and i > snapshot["index"] and known:
            reclaimed = int(day.known_cash_close_ticks) > snapshot["upper"]
        status = ("NO_VIEW_CURRENT_QUOTE_ANCHOR_KEPT" if not known else "NEW_FULL_DOWN_GAP" if born
                  else "FIRST_STRICT_CLOSE_RECLAIM" if reclaimed else "WAITING_LATEST_GAP_RECLAIM"
                  if snapshot is not None else "NO_ACTIVE_DOWN_GAP")
        if reclaimed:
            active = None
        rows.append({
            "new_down_gap": born, "replaced_anchor_id": replaced, "reclaim_event": reclaimed,
            "reclaim_status": status, "reclaim_anchor_id": snapshot["anchor_id"] if reclaimed else None,
            "reclaim_floor_ticks": snapshot["lower"] if reclaimed else -1,
            "reclaim_upper_ticks": snapshot["upper"] if reclaimed else -1,
            "reclaim_reference_index": snapshot["index"] if reclaimed else -1,
            "reclaim_age_sessions": i-snapshot["index"] if reclaimed else -1,
            "observed_anchor_id": snapshot["anchor_id"] if snapshot is not None else None,
            "observed_anchor_index": snapshot["index"] if snapshot is not None else -1,
            "observed_anchor_lower_ticks": snapshot["lower"] if snapshot is not None else -1,
            "observed_anchor_upper_ticks": snapshot["upper"] if snapshot is not None else -1,
            "anchor_active_after_origin": active is not None,
        })
    extra = pd.DataFrame(rows)
    for name in ("replaced_anchor_id", "reclaim_anchor_id", "observed_anchor_id"):
        extra[name] = pd.array(extra[name], dtype="string")
    return pd.concat([frame, extra], axis=1)


def anchor_paths(states):
    """每一个原锚均保留，未来终态只作解释，不能作为输入。"""
    endings = {}
    for day in states.itertuples(index=False):
        if pd.notna(day.replaced_anchor_id):
            endings[str(day.replaced_anchor_id)] = (day.origin_index, day.date, "REPLACED_BEFORE_RECLAIM")
        if day.reclaim_event:
            endings[str(day.reclaim_anchor_id)] = (day.origin_index, day.date, "RECLAIMED")
    rows = []
    for day in states.loc[states.new_down_gap].itertuples(index=False):
        index, date, status = endings.get(str(day.observed_anchor_id), (-1, pd.NaT, "RIGHT_CENSORED"))
        rows.append({"anchor_id": str(day.observed_anchor_id), "birth_index": int(day.origin_index),
                     "birth_date": day.date, "gap_lower_ticks": int(day.observed_anchor_lower_ticks),
                     "gap_upper_ticks": int(day.observed_anchor_upper_ticks), "terminal_index": int(index),
                     "terminal_date": date, "status": status,
                     "sessions_to_terminal": int(index-day.origin_index) if index >= 0 else -1,
                     "future_terminal_is_not_decision_input": True})
    return pd.DataFrame(rows, columns=["anchor_id", "birth_index", "birth_date", "gap_lower_ticks",
                                      "gap_upper_ticks", "terminal_index", "terminal_date", "status",
                                      "sessions_to_terminal", "future_terminal_is_not_decision_input"])


def account_signals(data, states, policy):
    if policy not in (PRIMARY, "PRICE_CONFIRMATION") or not data.date.reset_index(drop=True).equals(states.date):
        raise ValueError("向下缺口收复用途或原点日期不一致。")
    if policy == "PRICE_CONFIRMATION":
        from research import point_fresh_repair_order_inputs_v1 as old
        frame = old.account_signals(data, old.signals(data), policy)
        for name in ("cash_shift_ticks", "cash_shift_known"):
            frame[name] = states[name].to_numpy()
        return frame
    frame = states.copy()
    frame["entry_event"] = frame.reclaim_event
    frame["event_id"] = pd.Series([f"{PRIMARY}_{i}" if event else None
                                     for i, event in enumerate(frame.entry_event)], dtype=object)
    frame["stop_index"] = np.where(frame.entry_event, frame.reclaim_floor_ticks*TICK, np.nan)
    frame["target_index"] = np.nan
    frame["atr"] = data.atr20.to_numpy(float)
    frame["rule_exit"] = frame.new_down_gap
    return frame
