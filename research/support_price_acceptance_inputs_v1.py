"""已记录支持信息、固定价格区域与延续结构；不使用未来数据或结果选择。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.original_policy_announcement_trace_inputs_v1 import clock

EPS = 1e-10
POLICIES = ("SUPPORT_PRICE_ACCEPTANCE_CARRY", "PRICE_ACCEPTANCE_CARRY_ONLY", "SUPPORT_PRICE_ACCEPTANCE_FIXED_EXIT")
NAMES = {POLICIES[0]: "支持信息与价格接受后延续", POLICIES[1]: "只有价格接受后延续", POLICIES[2]: "同支持进入但固定退出"}
SUPPORT_NODES = {
    "LOCAL_GOV_20190105": "同期人民币降准政府转载",
    "SOURCE_001": "人民币定向降准与超额准备金利率宣布",
    "SOURCE_002": "人民币降准宣布",
    "SOURCE_010": "同期人民币降准政府转载",
    "SOURCE_020": "同期人民币降准政府转载",
    "SOURCE_021": "同期人民币降准政府转载",
    "CHAIN_R01": "人民币降准及政策利率下调宣布",
    "CHAIN_R03": "人民币降准及政策利率下调宣布",
    "CHAIN_C01": "资本市场工具首次宣布，同源不独立计分",
}
ANNOUNCED_RATE_TARGETS = {"CHAIN_R01": 1.5, "CHAIN_R03": 1.4}


def messages(nodes: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """完整节点各给用途资格；操作变化相对前一保存值，不认证即时政策降幅。"""
    if not nodes.node_id.is_unique or not rates.node_id.is_unique:
        raise ValueError("原节点或原利率身份重复。")
    known_rates = rates.set_index("node_id")
    rows = []
    for item in nodes.to_dict("records"):
        identity = item["node_id"]
        action, reason, target, value = "NON_ACTIVATING_RECORD", "其他工具、实施、迟发回顾或重复确认，不在本激活用途", np.nan, np.nan
        if identity.startswith("RATE_"):
            if identity not in known_rates.index:
                raise ValueError("操作节点没有原逐文核对记录。")
            rate = known_rates.loc[identity]
            value = float(rate.seven_day_rate_percent)
            before = rate.previous_observed_rate_percent
            if pd.isna(before):
                reason = "前一保存操作未知，不能认定宽松或收紧"
            elif value < float(before)-EPS:
                action, reason = "OBSERVED_RATE_LOWER", "相对前一已保存操作较低，不等于最早宣布或瞬时降息幅度"
            elif value > float(before)+EPS:
                action, reason = "OBSERVED_RATE_HIGHER", "相对前一已保存操作较高，不等于完整政策立场"
            else:
                reason = "相对前一保存操作持平"
        elif identity in SUPPORT_NODES:
            action, reason = "RECORDED_SUPPORT_INFORMATION", SUPPORT_NODES[identity]
            target = ANNOUNCED_RATE_TARGETS.get(identity, np.nan)
        rows.append({**item, "source_available_upper": clock(item["source_available_upper"]),
                     "registered_action": action, "qualification_reason": reason,
                     "announced_seven_day_target": target, "observed_seven_day_value": value,
                     "historical_first_vintage_authenticated": False})
    return pd.DataFrame(rows).sort_values(["source_available_upper", "node_id"]).reset_index(drop=True)


def source_states(data: pd.DataFrame, records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在原决定区间观察信息；宣布同目标的后续操作只确认，不再激活。"""
    d = data[["date", "decision_time"]].copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d.date).astype("datetime64[ns]")
    if not d.date.is_unique or not d.date.is_monotonic_increasing:
        raise ValueError("原观察日历未排序或重复。")
    queue = records.loc[records.source_available_upper.notna()].sort_values(["source_available_upper", "node_id"]).to_dict("records")
    cursor, pending_target, previous = 0, np.nan, pd.NaT
    rows, observed = [], []
    for date, at in d.itertuples(index=False):
        decision = clock(at)
        supports, reverse, confirmations = [], [], []
        while cursor < len(queue) and clock(queue[cursor]["source_available_upper"]) <= decision:
            item = queue[cursor]
            cursor += 1
            known_at = clock(item["source_available_upper"])
            action = item["registered_action"]
            current_target = item["announced_seven_day_target"]
            value = item["observed_seven_day_value"]
            actual = action
            if action == "RECORDED_SUPPORT_INFORMATION":
                if np.isfinite(current_target):
                    pending_target = float(current_target)
                supports.append(item)
            elif action == "OBSERVED_RATE_LOWER":
                if np.isfinite(pending_target) and np.isclose(value, pending_target, rtol=0, atol=EPS):
                    actual = "ANNOUNCED_TARGET_OPERATION_CONFIRMATION"
                    confirmations.append(item)
                    pending_target = np.nan
                else:
                    supports.append(item)
            elif action == "OBSERVED_RATE_HIGHER":
                reverse.append(item)
                pending_target = np.nan
            observed.append({"node_id": item["node_id"], "source_available_upper": known_at,
                "first_original_decision_date": date, "decision_time": decision, "registered_action": action,
                "actual_information_role": actual, "common_source_id": item["common_source_id"],
                "entered_after_previous_original_decision": bool(not pd.isna(previous) and previous < known_at),
                "source_url": item["source_url"], "source_path": item["source_path"]})
        ids = "|".join(item["node_id"] for item in supports)
        rows.append({"date": date, "support_information_arrived": bool(supports),
            "support_new_node_ids": ids, "support_unique_common_sources": len({item["common_source_id"] for item in supports}),
            "reverse_operation_arrived": bool(reverse), "reverse_node_ids": "|".join(item["node_id"] for item in reverse),
            "announced_target_confirmation_ids": "|".join(item["node_id"] for item in confirmations),
            "support_activation_allowed": bool(supports and not reverse),
            "coverage": "RECORDED_POSITIVE_EVIDENCE_ONLY_EMPTY_NOT_POLICY_ABSENCE"})
        previous = decision
    return pd.DataFrame(rows), pd.DataFrame(observed)


def weekly_structure(data: pd.DataFrame) -> pd.DataFrame:
    """两最近完整自然周的低点、收盘；周末可用，不使用本周未完成结构。"""
    d = data.copy().reset_index(drop=True)
    d["date"] = pd.to_datetime(d.date).astype("datetime64[ns]")
    d["cash_low"] = d.low+d.cash_shift
    d["cash_high"] = d.high+d.cash_shift
    periods = d.date.dt.to_period("W-FRI")
    weeks = d.groupby(periods).agg(week_low=("cash_low", "min"), week_close=("ac", "last"), week_last=("date", "last"))
    weeks["older_week_low"] = weeks.week_low.shift()
    weeks["older_week_close"] = weeks.week_close.shift()
    weeks["available"] = (weeks.index.to_timestamp(how="end").normalize()+pd.Timedelta(days=1)).astype("datetime64[ns]")
    lookup = pd.merge_asof(d[["date"]], weeks.reset_index(drop=True).sort_values("available"), left_on="date", right_on="available", direction="backward")
    for field in ["week_low", "week_close", "week_last", "older_week_low", "older_week_close", "available"]:
        d["complete_"+field] = lookup[field]
    valid = d.complete_week_last.notna()
    if not d.loc[valid, "complete_week_last"].lt(d.loc[valid, "date"]).all():
        raise ValueError("完整周结构包含当前未完成日。")
    d["higher_complete_week_low_and_close"] = d.complete_week_low.gt(d.complete_older_week_low+EPS) & d.complete_week_close.gt(d.complete_older_week_close+EPS)
    return d


def price_states(data: pd.DataFrame, use_support: bool) -> pd.DataFrame:
    """每次观察固定高低价；接受或失效后消费，不凭旧信息反复尝试。"""
    d = data.reset_index(drop=True)
    watch, records = None, []
    prefix = "support" if use_support else "price"
    for i, row in enumerate(d.itertuples()):
        event, reason = False, "NO_ACTIVE_OBSERVATION"
        inherited = watch
        reverse = bool(row.reverse_operation_arrived) if use_support else False
        known_price = np.isfinite([row.ac, row.cash_high, row.cash_low, row.daily_hist, row.atr14]).all()
        if reverse:
            reason, watch = "OBSERVED_REVERSE_OPERATION_INVALIDATES", None
        elif watch is not None:
            if row.ac <= watch["low"]+EPS:
                reason, watch = "ANCHOR_LOW_FAILED", None
            elif i > watch["idx"] and known_price and row.ac > watch["high"]+EPS and row.daily_hist > 0:
                event, reason, watch = True, "PRICE_ACCEPTED_AFTER_ANCHOR", None
            else:
                reason = "WAIT_PRICE_ACCEPTANCE_FIXED_ANCHOR"
        elif known_price and (bool(row.support_activation_allowed) if use_support else True):
            watch = {"idx": i, "date": row.date, "high": float(row.cash_high), "low": float(row.cash_low),
                     "source_ids": row.support_new_node_ids if use_support else "PRICE_ONLY_NO_POLICY"}
            reason = "NEW_FIXED_ANCHOR_OBSERVATION"
        displayed = inherited if inherited is not None else watch
        records.append({"date": row.date, prefix+"_entry_event": event, prefix+"_observation_reason": reason,
            prefix+"_setup_idx": displayed["idx"] if displayed else np.nan,
            prefix+"_setup_date": displayed["date"] if displayed else pd.NaT,
            prefix+"_setup_high": displayed["high"] if displayed else np.nan,
            prefix+"_setup_low": displayed["low"] if displayed else np.nan,
            prefix+"_setup_source_ids": displayed["source_ids"] if displayed else "",
            prefix+"_watch_alive_after_decision": watch is not None})
    out = pd.DataFrame(records)
    out[prefix+"_setup_date"] = pd.to_datetime(out[prefix+"_setup_date"]).astype("datetime64[ns]")
    return out


def observations(data: pd.DataFrame, records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = weekly_structure(data)
    states, source_receipts = source_states(d, records)
    d = d.merge(states, on="date", validate="one_to_one")
    for enabled in [True, False]:
        d = d.merge(price_states(d, enabled), on="date", validate="one_to_one")
    return d, source_receipts


def exit_decision(active: dict, row, i: int, policy: str):
    """仅真实持仓确认后改持有方法；风险退出之外，明确结构与反向信息。"""
    if policy != POLICIES[1] and bool(row.reverse_operation_arrived):
        return "KNOWN_REVERSE_OPERATION"
    if row.ac <= active["structural_stop"]+EPS:
        return "KNOWN_ANCHOR_OR_TRAILING_LOW_FAILED"
    if policy == POLICIES[2]:
        if row.ac <= active["fixed_stop"]+EPS:
            return "FIXED_LOSS_CLOSE"
        if row.ac >= active["fixed_target"]-EPS:
            return "FIXED_PROFIT_CLOSE"
        return "FIXED_TWENTY_CLOSES" if i-active["entry_idx"]+1 >= 20 else None
    if (active["holding_stage"] == "EARLY" and bool(row.higher_complete_week_low_and_close)
            and not pd.isna(row.complete_week_last) and row.complete_week_last >= active["entry_date"]):
        active["holding_stage"] = "CARRY"
        active["promotion_date"] = row.date
    if active["holding_stage"] == "CARRY":
        if np.isfinite(row.complete_week_low):
            active["structural_stop"] = max(active["structural_stop"], float(row.complete_week_low))
        if row.ac <= active["structural_stop"]+EPS:
            return "COMPLETE_WEEK_TRAILING_LOW_FAILED"
    elif np.isfinite(row.daily_hist) and row.daily_hist < 0 and row.ac < row.ema20-EPS:
        return "EARLY_PRICE_MOMENTUM_FAILED"
    return None
