"""只用当前和前一已知状态解释联合阶段的最后到达项。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from research.trend_expansion_phase_inputs_v1 import phase_states

CLOCKS = ("PRICE_LAST", "DAILY_DIF_LAST", "PREVIOUS_COMPLETE_WEEK_LAST", "MULTIPLE_SAME_ORIGIN")


def clocks(data: pd.DataFrame) -> pd.DataFrame:
    d = data.reset_index(drop=True)
    phases = phase_states(d)
    known = phases.phase_known
    prior_known = known.shift(1, fill_value=False)
    eligible = known & prior_known
    price = d.ac.gt(d.ema20)
    daily = d.daily_dif.gt(0)
    weekly = d.weekly_hist.gt(0)
    price_flip = eligible & price & ~price.shift(1, fill_value=False)
    daily_flip = eligible & daily & ~daily.shift(1, fill_value=False)
    weekly_flip = eligible & weekly & ~weekly.shift(1, fill_value=False)
    born = phases.joint_phase_onset
    changed = price_flip.astype(int) + daily_flip.astype(int) + weekly_flip.astype(int)
    if (born & changed.eq(0)).any():
        raise ValueError("联合出生没有任何当时新增确认，不能归因。")
    same_week_source = d.weekly_last_date.eq(d.weekly_last_date.shift(1))
    if (born & weekly_flip & same_week_source).any():
        raise ValueError("同一完整周来源的柱值发生转正，来源时钟不一致。")
    kind = np.select([
        ~known, born & changed.gt(1), born & price_flip, born & daily_flip, born & weekly_flip,
    ], ["NO_VIEW", "MULTIPLE_SAME_ORIGIN", "PRICE_LAST", "DAILY_DIF_LAST",
        "PREVIOUS_COMPLETE_WEEK_LAST"], default="NO_JOINT_BIRTH")
    return pd.DataFrame({
        "date": d.date, "origin_index": phases.origin_index,
        "phase_known": known, "joint_phase_onset": born, "birth_clock": kind,
        "price_newly_positive": price_flip.where(born, False),
        "daily_dif_newly_positive": daily_flip.where(born, False),
        "weekly_hist_newly_positive": weekly_flip.where(born, False),
        "simultaneous_arrivals": changed.where(born, 0),
        "previous_phase": phases.previous_phase,
        "known_daily_dif": d.daily_dif,
        "known_weekly_hist": d.weekly_hist, "known_previous_week_date": d.weekly_last_date,
    })
