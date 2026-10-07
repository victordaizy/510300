"""固定实际进入锚，描述持仓与退出后新增信息；不读取盈亏来判别状态。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.original_policy_announcement_trace_inputs_v1 import clock

EPS = 1e-10
INFORMATION = ("orders", "funding", "margin")
CLOCK_COLUMNS = {"orders": "orders_available_at", "funding": "funding_available_at", "margin": "margin_available_at"}
SUPPORT_ROLES = {"RECORDED_SUPPORT_INFORMATION", "OBSERVED_RATE_LOWER"}


def normalize_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d.date).astype("datetime64[ns]")
    if d.date.isna().any() or not d.date.is_unique or not d.date.is_monotonic_increasing:
        raise ValueError("原日历缺失、重复或未排序。")
    groups = d.groupby(d.date.dt.to_period("W-FRI"), sort=True).date
    first = groups.min()
    d["complete_week_first"] = d.complete_week_last.dt.to_period("W-FRI").map(first)
    known = d.complete_week_last.notna()
    if not d.loc[known, "complete_week_last"].lt(d.loc[known, "date"]).all():
        raise ValueError("上一完整周包含当前未完成价格。")
    return d


def price_view(row: pd.Series, entry: pd.Series) -> dict:
    """实际进入日完整高低仅用于该日收盘后的持有描述，不倒用于进入。"""
    known = np.isfinite([row.ac, row.daily_hist, entry.cash_high, entry.cash_low]).all()
    after = row.date > entry.date
    if not known:
        zone = "UNKNOWN_PRICE"
    elif row.ac > entry.cash_high + EPS:
        zone = "ABOVE_ACTUAL_ENTRY_DAY_RANGE"
    elif row.ac <= entry.cash_low + EPS:
        zone = "AT_OR_BELOW_ACTUAL_ENTRY_DAY_LOW"
    else:
        zone = "INSIDE_ACTUAL_ENTRY_DAY_RANGE"
    whole_week = (pd.notna(row.complete_week_first) and pd.notna(row.complete_week_last)
                  and row.complete_week_first >= entry.date and row.complete_week_last < row.date)
    accepted_week = bool(whole_week and np.isfinite([row.complete_week_low, row.complete_week_close]).all()
                         and row.complete_week_low > entry.cash_low + EPS
                         and row.complete_week_close > entry.cash_high + EPS)
    return {"entry_day_high_anchor": float(entry.cash_high), "entry_day_low_anchor": float(entry.cash_low),
        "current_price_zone": zone, "post_entry_price_accepted_now": bool(after and known and zone == "ABOVE_ACTUAL_ENTRY_DAY_RANGE" and row.daily_hist > 0),
        "entry_day_low_failed_now": bool(after and known and zone == "AT_OR_BELOW_ACTUAL_ENTRY_DAY_LOW"),
        "complete_week_all_observed_days_after_entry": bool(whole_week),
        "post_entry_complete_week_acceptance_now": accepted_week,
        "complete_week_first": row.complete_week_first, "complete_week_last": row.complete_week_last,
        "previous_week_low_and_close_raised": bool(row.higher_complete_week_low_and_close)}


def macro_view(row: pd.Series, entry_clock: pd.Timestamp) -> dict:
    result = {}
    for name in INFORMATION:
        available = clock(row.get(CLOCK_COLUMNS[name], pd.NaT))
        admitted = bool(row.get(name + "_known", False))
        if pd.notna(available) and available > clock(row.decision_time) and admitted:
            raise ValueError("宏观已知身份含决定时点之后的来源。")
        new_after_entry = bool(admitted and pd.notna(available) and available > entry_clock)
        publication = bool(row.get(name + "_publication_since_previous_decision", False))
        first = bool(row.get(name + "_first_observed_here", False))
        result.update({name + "_currently_admitted": admitted, name + "_source_published_after_actual_entry_open": new_after_entry,
            name + "_fresh_publication_event_today_after_entry": bool(new_after_entry and publication),
            name + "_first_observed_today_without_new_publication": bool(first and not publication),
            name + "_available_at": available, name + "_source_identity": str(row.get(name + "_source_identity", ""))})
    return result


def policy_view(records: list[dict], entry_clock: pd.Timestamp, decision: pd.Timestamp) -> dict:
    fresh, old, confirmations, reverse, common = [], [], [], [], set()
    for item in records:
        available = clock(item["source_available_upper"])
        if pd.isna(available) or available > decision:
            raise ValueError("已观察节点没有合格过去公布钟。")
        role = item["actual_information_role"]
        if role in SUPPORT_ROLES:
            if available > entry_clock:
                fresh.append(item["node_id"])
                common.add(item["common_source_id"])
            else:
                old.append(item["node_id"])
        elif role == "ANNOUNCED_TARGET_OPERATION_CONFIRMATION":
            confirmations.append(item["node_id"])
        elif role == "OBSERVED_RATE_HIGHER":
            reverse.append(item["node_id"])
    return {"policy_new_support_published_after_entry_ids_today": "|".join(fresh),
        "policy_older_support_first_observed_after_entry_ids_today": "|".join(old),
        "policy_target_confirmation_ids_today": "|".join(confirmations),
        "policy_observed_reverse_ids_today": "|".join(reverse),
        "policy_fresh_common_sources_today": len(common),
        "policy_scope": "RECORDED_POSITIVE_EVIDENCE_EMPTY_NOT_POLICY_ABSENCE"}


def sequences(observed: pd.DataFrame, account: dict, receipts: pd.DataFrame, period: str, cost: str,
              policy: str, end: str) -> pd.DataFrame:
    d = normalize_calendar(observed)
    daily = account["daily"].copy()
    daily["date"] = pd.to_datetime(daily.date).astype("datetime64[ns]")
    if not daily.date.is_unique:
        raise ValueError("账户同一天有重复记录。")
    daily = daily.set_index("date")
    trades = account["trades"].copy()
    trades["entry_date"] = pd.to_datetime(trades.entry_date).astype("datetime64[ns]")
    trades = trades.loc[trades.entry_date.le(d.date.iloc[-1])].sort_values("entry_date")
    if trades.entry_date.isna().any() or not trades.entry_date.is_unique:
        raise ValueError("实际进入日期重复或缺失。")
    receipt_groups = {pd.Timestamp(date): frame.to_dict("records") for date, frame in receipts.groupby("first_original_decision_date")}
    records = []
    for i, trade in enumerate(trades.to_dict("records")):
        entry_date = pd.Timestamp(trade["entry_date"])
        entry_clock = clock(entry_date) + pd.Timedelta(hours=9, minutes=30)
        entry = d.loc[d.date.eq(entry_date)].iloc[0]
        next_entry = trades.entry_date.iloc[i+1] if i+1 < len(trades) else pd.Timestamp.max
        segment = d.loc[d.date.ge(entry_date) & d.date.lt(next_entry) & d.date.le(pd.Timestamp(end)) & d.date.isin(daily.index)]
        first_accept, first_low_failure, first_week_accept = pd.NaT, pd.NaT, pd.NaT
        fresh_ids, old_ids, confirmation_ids = [], [], []
        previous_above = False
        held_closes = 0
        for _, row in segment.iterrows():
            date, decision = row.date, clock(row.decision_time)
            price = price_view(row, entry)
            macro = macro_view(row, entry_clock)
            policy_info = policy_view(receipt_groups.get(date, []), entry_clock, decision)
            actual = daily.loc[date]
            held = bool(actual.shares > 0)
            prior_held = held_closes
            held_closes += int(held)
            if price["post_entry_price_accepted_now"] and pd.isna(first_accept):
                first_accept = date
            if price["entry_day_low_failed_now"] and pd.isna(first_low_failure):
                first_low_failure = date
            if price["post_entry_complete_week_acceptance_now"] and pd.isna(first_week_accept):
                first_week_accept = date
            above = price["post_entry_price_accepted_now"]
            reaccept = bool(above and not previous_above and pd.notna(first_low_failure))
            previous_above = above
            for field, target in (("policy_new_support_published_after_entry_ids_today", fresh_ids),
                                  ("policy_older_support_first_observed_after_entry_ids_today", old_ids),
                                  ("policy_target_confirmation_ids_today", confirmation_ids)):
                target.extend(node for node in policy_info[field].split("|") if node)
            industry_known = bool(row.get("industry_view_allowed", False))
            records.append({"date": date, "period": period, "cost": cost, "policy": policy,
                "cycle_id": int(trade["cycle_id"]), "entry_date": entry_date,
                "actual_account_phase": "ACTUALLY_HOLDING" if held else "AFTER_ACTUAL_EXIT_BEFORE_NEXT_ENTRY",
                "actual_holding_closes_prior_to_decision": prior_held,
                "actual_close_shares": int(actual.shares), "actual_close_exposure": float(actual.exposure),
                "first_post_entry_price_acceptance": first_accept, "first_actual_entry_low_failure": first_low_failure,
                "first_wholly_post_entry_complete_week_acceptance": first_week_accept,
                "price_reacceptance_after_low_failure_event": reaccept,
                "policy_fresh_support_ids_since_entry": "|".join(fresh_ids),
                "policy_old_support_first_observed_ids_since_entry": "|".join(old_ids),
                "policy_target_confirmations_since_entry": "|".join(confirmation_ids),
                **price, **macro, **policy_info,
                "ac": float(row.ac), "daily_hist": float(row.daily_hist), "weekly_hist": float(row.weekly_hist),
                "relative_volume": float(np.exp(row.log_relative_volume)), "up_volume_balance5": float(row.up_volume_balance5),
                "volatility20_60": float(row.volatility20_60), "funding_gap_pp": float(row.funding_gap_pp),
                "funding_gap_change5": float(row.funding_gap_change5), "financing_net_change5": float(row.financing_net_change5),
                "pmi_new_orders_level": float(row.pmi_orders_level + 50), "pmi_new_orders_change": float(row.pmi_orders_change),
                "industry_view_allowed": industry_known, "industry_positive5_fraction": float(row.industry_positive5_fraction),
                "industry_relative_positive5_fraction": float(row.industry_relative_positive5_fraction),
                "industry_source_age_days": float(row.industry_source_age_days),
                "industry_snapshot_id": str(row.industry_snapshot_id), "leader_ids": str(row.leader_ids),
                "leader_names": str(row.leader_names), "leader_mean_return5": float(row.leader_mean_return5),
                "leader_mean_relative5": float(row.leader_mean_relative5), "rotation_churn5blocks": float(row.rotation_churn5blocks),
                "industry_scope": "POINT_VIEW_NOT_CERTIFIED_FIXED_ENTRY_COHORT_DIFFUSION",
                "role": "CAUSAL_POST_ENTRY_DESCRIPTION_NO_NEW_TRADING_RULE_OR_TARGET"})
    return pd.DataFrame(records)
