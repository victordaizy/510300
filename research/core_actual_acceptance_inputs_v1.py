"""原A进入后的价格接受、完整买后周与新增信息；不使用未来盈亏。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import actual_holding_information_inputs_v1 as description
from research.point_account_nr7_inputs_v1 import PARENT_A

EPS = 1e-10
BASELINE = "A_CONTROL"
POLICIES = (
    "CORE_ACTUAL_ACCEPTANCE_INFORMATION_CARRY",
    "CORE_ACTUAL_ACCEPTANCE_PRICE_CARRY",
    "CORE_ACTUAL_ACCEPTANCE_PROTECTION_ONLY",
)
NAMES = dict(zip(POLICIES, ("原A进入与新增信息延续", "同进入仅价格延续", "同进入仅接受后保护")))


def observations(frame: pd.DataFrame, receipts: pd.DataFrame) -> pd.DataFrame:
    d = description.normalize_calendar(frame)
    groups = {pd.Timestamp(k): v.to_dict("records") for k, v in receipts.groupby("first_original_decision_date")}
    reverse_clocks, reverse_ids = [], []
    for row in d.itertuples():
        clocks, identities = [], []
        decision = description.clock(row.decision_time)
        for item in groups.get(row.date, []):
            available = description.clock(item["source_available_upper"])
            if pd.isna(available) or available > decision:
                raise ValueError("原节点的公布上界不在本次决定之前。")
            if item["actual_information_role"] == "OBSERVED_RATE_HIGHER":
                clocks.append(available)
                identities.append(item["node_id"])
        reverse_clocks.append(max(clocks) if clocks else pd.NaT)
        reverse_ids.append("|".join(identities))
    d["reverse_source_available_upper"] = pd.to_datetime(pd.Series(reverse_clocks), utc=True).dt.tz_convert("Asia/Shanghai").astype("datetime64[ns, Asia/Shanghai]")
    d["actual_carry_reverse_node_ids"] = reverse_ids
    return d


def parent_episodes(parents: pd.DataFrame) -> pd.DataFrame:
    """未知不结束旧正目标段；只有明确零之后的新正目标才产生新身份。"""
    p = parents[["origin", PARENT_A]].copy().sort_values("origin").reset_index(drop=True)
    p["origin"] = pd.to_datetime(p.origin).astype("datetime64[ns]")
    if not p.origin.is_unique:
        raise ValueError("原A来源日期重复。")
    positive, identity, values = False, 0, []
    for value in p[PARENT_A].to_numpy(float):
        if np.isfinite(value):
            if not 0 <= value <= 1:
                raise ValueError("原A保存权重超出0至1。")
            if value == 0:
                positive = False
            elif not positive:
                identity += 1
                positive = True
        values.append(identity)
    p["parent_episode_id"] = values
    return p


def participation(row: pd.Series) -> tuple[bool, bool]:
    values = [row.up_volume_balance5, row.industry_positive5_fraction, row.leader_mean_return5]
    known = bool(row.industry_view_allowed and np.isfinite(values).all())
    source_date = pd.Timestamp(row.industry_source_date)
    if known and pd.notna(source_date) and source_date > row.date:
        raise ValueError("行业参与观察含未来日期。")
    known = bool(known and pd.notna(source_date))
    positive = bool(known and row.up_volume_balance5 > 0 and row.industry_positive5_fraction > .5 and row.leader_mean_return5 > 0)
    return known, positive


def fresh_adverse(row: pd.Series, entry_date: pd.Timestamp) -> dict:
    entry_clock = description.clock(entry_date) + pd.Timedelta(hours=9, minutes=30)
    decision = description.clock(row.decision_time)
    available = description.clock(row.orders_available_at)
    if bool(row.orders_known) and pd.notna(available) and available > decision:
        raise ValueError("订单调查信息晚于当前决定。")
    order_new = bool(row.orders_known and pd.notna(available) and entry_clock < available <= decision
                     and row.orders_publication_since_previous_decision and np.isfinite(row.pmi_orders_change))
    reverse = description.clock(row.reverse_source_available_upper)
    if pd.notna(reverse) and reverse > decision:
        raise ValueError("反向操作信息晚于当前决定。")
    reverse_new = bool(row.reverse_operation_arrived and pd.notna(reverse) and entry_clock < reverse <= decision)
    return {"orders_fresh_deterioration": bool(order_new and row.pmi_orders_change < 0),
            "reverse_fresh_operation": reverse_new, "orders_available_at": available,
            "reverse_available_at": reverse, "entry_open_clock": entry_clock}


def advance(active: dict, row: pd.Series, policy: str) -> dict:
    """只在真实持仓收盘推进；进入日高低在该收盘建立，之后才允许保护。"""
    if policy not in POLICIES:
        raise ValueError("未知原A接受后持有用途。")
    if row.date < active["entry_date"]:
        raise ValueError("当前收盘早于实际进入。")
    if row.date == active["entry_date"]:
        active["entry_day_high_anchor"] = float(row.cash_high)
        active["entry_day_low_anchor"] = float(row.cash_low)
    entry = pd.Series({"date": active["entry_date"], "cash_high": active["entry_day_high_anchor"], "cash_low": active["entry_day_low_anchor"]})
    price = description.price_view(row, entry)
    known, positive = participation(row)
    news = fresh_adverse(row, active["entry_date"])
    adverse = bool(news["orders_fresh_deterioration"] or news["reverse_fresh_operation"])
    before = active["holding_stage"]
    if policy == POLICIES[0] and adverse:
        active["extension_revoked"] = True
        if active["holding_stage"] == "CARRY":
            active["holding_stage"] = "ACCEPTED"
        if pd.isna(active["first_extension_revocation"]):
            active["first_extension_revocation"] = row.date
    if price["post_entry_price_accepted_now"] and not active["protection_armed"]:
        active["protection_armed"] = True
        active["first_price_acceptance"] = row.date
        active["structural_stop"] = active["entry_day_low_anchor"]
        active["holding_stage"] = "ACCEPTED"
    promote = bool(policy != POLICIES[2] and not active["extension_revoked"] and active["protection_armed"]
                   and price["post_entry_price_accepted_now"] and price["post_entry_complete_week_acceptance_now"]
                   and (policy == POLICIES[1] or positive))
    if promote and active["holding_stage"] != "CARRY":
        active["holding_stage"] = "CARRY"
        if pd.isna(active["promotion_date"]):
            active["promotion_date"] = row.date
    if active["holding_stage"] == "CARRY" and price["complete_week_all_observed_days_after_entry"] and np.isfinite(row.complete_week_low):
        active["structural_stop"] = max(active["structural_stop"], float(row.complete_week_low))
    failure = "NONE"
    if active["protection_armed"] and np.isfinite(row.ac) and row.ac <= active["structural_stop"] + EPS:
        failure = "CARRY_TRAIL_STRUCTURE_FAILED" if active["holding_stage"] == "CARRY" else "ACTUAL_ACCEPTANCE_STRUCTURE_FAILED"
    return {**price, **news, "participation_known": known, "participation_positive": positive,
            "fresh_adverse_information": adverse, "stage_before": before, "holding_stage": active["holding_stage"],
            "protection_armed": active["protection_armed"], "structural_stop": active["structural_stop"],
            "extension_revoked": active["extension_revoked"], "extra_exit_reason": failure}
