"""日DIF与上一完整周MACD的展开阶段；不读取未来结果或事后分段。"""

from __future__ import annotations

import numpy as np
import pandas as pd


PHASES = (
    "DAILY_AND_WEEKLY_POSITIVE", "DAILY_POSITIVE_WEEKLY_NONPOSITIVE",
    "DAILY_NONPOSITIVE_WEEKLY_POSITIVE", "DAILY_AND_WEEKLY_NONPOSITIVE",
    "PRICE_NOT_ABOVE_EMA", "NO_VIEW",
)
POLICIES = ("JOINT_PHASE_START", "PRICE_CONFIRMATION")


def phase_states(data: pd.DataFrame) -> pd.DataFrame:
    d = data.reset_index(drop=True)
    needed = {"date", "ac", "ema20", "daily_dif", "weekly_hist", "weekly_last_date", "relative_volume", "breakout20", "available"}
    if not needed.issubset(d.columns) or not d.date.is_unique or not d.date.is_monotonic_increasing:
        raise ValueError("展开阶段缺少原字段，或日期不唯一递增。")
    finite = np.isfinite(d[["ac", "ema20", "daily_dif", "weekly_hist", "relative_volume"]]).all(axis=1)
    known = d.available.eq(True) & finite & d.weekly_last_date.notna()
    week_start = d.date.dt.to_period("W-FRI").dt.start_time
    if (known & d.weekly_last_date.ge(week_start)).any():
        raise ValueError("展开阶段读取了当前尚未完成周的值。")
    above, daily, weekly = d.ac.gt(d.ema20), d.daily_dif.gt(0), d.weekly_hist.gt(0)
    kind = np.select([
        ~known, ~above, daily & weekly, daily & ~weekly, ~daily & weekly,
    ], ["NO_VIEW", "PRICE_NOT_ABOVE_EMA", "DAILY_AND_WEEKLY_POSITIVE",
        "DAILY_POSITIVE_WEEKLY_NONPOSITIVE", "DAILY_NONPOSITIVE_WEEKLY_POSITIVE"],
        default="DAILY_AND_WEEKLY_NONPOSITIVE")
    active = pd.Series(kind == "DAILY_AND_WEEKLY_POSITIVE")
    previous_known = known.shift(1, fill_value=False)
    joint_onset = known & previous_known & active & ~active.shift(1, fill_value=False)
    raw_price_cross = above & ~above.shift(1, fill_value=False)
    price_cross = known & previous_known & raw_price_cross
    raw_breakout = d.breakout20.eq(True)
    breakout_onset = known & previous_known & raw_breakout & ~raw_breakout.shift(1, fill_value=False)
    high_volume = d.relative_volume.ge(1.5)
    volume_onset = known & previous_known & high_volume & ~high_volume.shift(1, fill_value=False)
    output = pd.DataFrame({
        "date": d.date, "origin_index": np.arange(len(d)), "phase_known": known,
        "phase": kind, "joint_phase_active": pd.array(active.where(known), dtype="boolean"),
        "joint_phase_onset": joint_onset, "price_confirmation": price_cross,
        "range20_breakout_onset": breakout_onset, "relative_volume_expansion_onset": volume_onset,
        "previous_phase": pd.Series(kind).shift().fillna("NO_PREVIOUS_STATE"),
        "current_daily_dif": d.daily_dif, "previous_complete_week_hist": d.weekly_hist,
        "previous_complete_week_date": d.weekly_last_date,
    })
    return output


def account_signals(data: pd.DataFrame, phases: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy not in POLICIES or len(data) != len(phases) or not data.date.reset_index(drop=True).equals(phases.date):
        raise ValueError("展开政策或当时阶段日期不一致。")
    rows = phases.copy()
    if policy == "JOINT_PHASE_START":
        rows["entry_event"] = phases.joint_phase_onset
        rows["rule_exit"] = phases.phase_known & ~phases.joint_phase_active.fillna(False)
    else:
        rows["entry_event"] = phases.price_confirmation
        rows["rule_exit"] = np.isfinite(data[["ac", "ema20"]]).all(axis=1).to_numpy() & data.ac.le(data.ema20).to_numpy()
    rows["event_id"] = [f"{policy}_{i}" if x else None for i, x in enumerate(rows.entry_event)]
    rows["atr"] = data.atr20.to_numpy(float)
    rows["stop_index"] = rows["target_index"] = np.nan
    return rows
