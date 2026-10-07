"""已确认下降结构的最后高点首次被收盘突破；只使用上一原点的结构。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.confirmed_structure_inputs_v1 import structure_states

PRIMARY = "FIRST_CONFIRMED_DOWNTREND_BREAK"
POLICIES = (PRIMARY, "PRICE_CONFIRMATION")


def break_states(data):
    original, pivots = structure_states(data)
    frame = original.rename(columns={"structure_onset": "old_rising_structure_onset", "rule_exit": "old_rising_structure_rule_exit"}).copy()
    consumed, rows = set(), []
    for i, current in enumerate(original.itertuples(index=False)):
        prior = original.iloc[i-1] if i else None
        known = bool(current.structure_known and prior is not None and prior.structure_known)
        complete = bool(known and prior.confirmed_sequence_length >= 4)
        down = bool(complete and prior.H2_price < prior.H1_price and prior.L2_price < prior.L1_price)
        high = float(prior.H2_price) if complete else np.nan
        cross = bool(down and prior.known_cash_adjusted_close <= high < current.known_cash_adjusted_close)
        anchor = f"H_{int(prior.H2_index)}_C_{int(prior.H2_confirmation_index)}" if complete else None
        seen = bool(anchor in consumed) if anchor is not None else False
        entry = bool(cross and not seen)
        if entry:
            if int(prior.H2_confirmation_index) >= i:
                raise AssertionError("突破引用了尚未在上一原点确认的高点。")
            consumed.add(anchor)
        status = ("NO_PREVIOUS_QUOTE" if prior is None else "NO_VIEW_CURRENT_OR_PRIOR_QUOTES" if not known
                  else "INSUFFICIENT_PRIOR_PIVOTS" if not complete else "PRIOR_STRUCTURE_NOT_BOTH_FALLING" if not down
                  else "NEW_BROKEN_DOWNTREND_EVENT" if entry else "ANCHOR_ALREADY_CONSUMED" if cross and seen
                  else "NO_FIRST_CLOSE_CROSS")
        detail = {}
        for name in ("H1", "H2", "L1", "L2"):
            detail.update({
                f"prior_{name}_index": int(prior[f"{name}_index"]) if complete else -1,
                f"prior_{name}_confirmation_index": int(prior[f"{name}_confirmation_index"]) if complete else -1,
                f"prior_{name}_date": prior[f"{name}_date"] if complete else pd.NaT,
                f"prior_{name}_confirmation_date": prior[f"{name}_confirmation_date"] if complete else pd.NaT,
                f"prior_{name}_price": float(prior[f"{name}_price"]) if complete else np.nan,
            })
        rows.append({"break_status": status, "prior_and_current_known": known, "prior_complete_downtrend": down,
                     "break_candidate": cross, "break_event": entry, "break_anchor_id": anchor,
                     "anchor_consumed_before_today": seen, "initial_broken_high_stop": high,
                     "previous_cash_adjusted_close": float(prior.known_cash_adjusted_close) if prior is not None else np.nan,
                     "consumed_anchor_count": len(consumed), **detail})
    added = pd.DataFrame(rows)
    for name in ("H1", "H2", "L1", "L2"):
        for suffix in ("date", "confirmation_date"):
            added[f"prior_{name}_{suffix}"] = pd.to_datetime(added[f"prior_{name}_{suffix}"]).astype("datetime64[ns]")
    return pd.concat([frame, added], axis=1), pivots


def account_signals(data, states, policy):
    if policy not in POLICIES or not data.date.reset_index(drop=True).equals(states.date):
        raise ValueError("下降结构突破政策或原点日期不一致。")
    if policy == "PRICE_CONFIRMATION":
        from research import point_fresh_repair_order_inputs_v1 as original
        return original.account_signals(data, original.signals(data), policy)
    rows = states.copy()
    rows["entry_event"] = states.break_event
    rows["event_id"] = [f"{PRIMARY}_{i}" if event else None for i, event in enumerate(rows.entry_event)]
    rows["stop_index"] = states.initial_broken_high_stop
    rows["target_index"] = np.nan
    rows["atr"] = data.atr20.to_numpy(float)
    rows["rule_exit"] = False
    return rows
