"""整日区间反转的因果输入：严格跨昨低且收盘跨昨高，反向对称。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.full_daily_gap_inputs_v1 import gap_states, TICK

PRIMARY = "FULL_RANGE_REVERSAL"


def reversal_states(data):
    """复用冻结报价/现金坐标规则；形态不读取未来、量或结果标签。"""
    coordinate = gap_states(data)
    names = [name for name in coordinate if not name.startswith("gap_") and name != "down_gap_observation"]
    frame = coordinate[names].copy()
    previous_low = frame.known_cash_low_ticks.shift(1).fillna(-1).astype(np.int64)
    previous_high = frame.known_cash_high_ticks.shift(1).fillna(-1).astype(np.int64)
    known = frame.previous_and_current_quote_known
    up = known & frame.known_cash_low_ticks.lt(previous_low) & frame.known_cash_close_ticks.gt(previous_high)
    down = known & frame.known_cash_high_ticks.gt(previous_high) & frame.known_cash_close_ticks.lt(previous_low)
    if (up & down).any():
        raise AssertionError("同一完整日线不能同时具有两个相反的收盘跨区间事件。")
    frame["bullish_range_reversal"] = up
    frame["bearish_range_reversal"] = down
    frame["reversal_status"] = np.select(
        [frame.origin_index.eq(0), ~known, up, down],
        ["NO_PREVIOUS_QUOTE", "NO_VIEW_CURRENT_OR_PRIOR_QUOTE", "BULLISH_FULL_RANGE_REVERSAL", "BEARISH_FULL_RANGE_REVERSAL"],
        default="NO_FULL_RANGE_REVERSAL")
    frame["reversal_anchor_id"] = pd.array(
        [f"RANGE_REVERSAL_{i}_PREVIOUS_{i-1}" if event else None for i, event in enumerate(up)], dtype="string")
    frame["reversal_floor_ticks"] = np.where(up, frame.known_cash_low_ticks, -1)
    frame["previous_cash_low_ticks"] = previous_low
    frame["previous_cash_high_ticks"] = previous_high
    frame["reversal_reference_index"] = np.where(up | down, frame.origin_index-1, -1)
    return frame


def account_signals(data, states, policy):
    if policy not in (PRIMARY, "PRICE_CONFIRMATION") or not data.date.reset_index(drop=True).equals(states.date):
        raise ValueError("整日反转政策或原点日期不一致。")
    if policy == "PRICE_CONFIRMATION":
        from research import point_fresh_repair_order_inputs_v1 as old
        frame = old.account_signals(data, old.signals(data), policy)
        for name in ("cash_shift_ticks", "cash_shift_known"):
            frame[name] = states[name].to_numpy()
        return frame
    frame = states.copy()
    frame["entry_event"] = frame.bullish_range_reversal
    frame["event_id"] = pd.Series(
        [f"{PRIMARY}_{i}" if event else None for i, event in enumerate(frame.entry_event)], dtype=object)
    frame["stop_index"] = np.where(frame.entry_event, frame.reversal_floor_ticks*TICK, np.nan)
    frame["target_index"] = np.nan
    frame["atr"] = data.atr20.to_numpy(float)
    frame["rule_exit"] = frame.bearish_range_reversal
    return frame
