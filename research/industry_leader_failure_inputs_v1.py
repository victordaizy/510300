"""实际进入后固定领先行业失效；在独立原阶段引擎中增加一次退出检查。"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from research import broker_stage_policy_inputs_v1 as stage
from research import industry_structure_description_inputs_v1 as industry
from research.broker_fixed_cohort_inputs_v1 import Panel

POLICIES = ("FIXED_INDUSTRY_LEADERS_FAILED_ETF_POSITIVE", "ETF_POSITIVE_SAME_LEADER_COVERAGE", "ETF_POSITIVE_ALL_DAYS")
NAMES = {POLICIES[0]: "固定领先行业失效且ETF仍正", POLICIES[1]: "同领先行业覆盖ETF正时退出", POLICIES[2]: "全部日期ETF正时退出"}
CHECK_LAG, EPS = 5, 1e-12
CHECK_COLUMNS = ["cycle_id", "entry_date", "decision_date", "execution_date", "relative_session",
    "anchor_source_date", "observation_source_date", "classification_snapshot_id", "classification_available_at",
    "classification_source_age_days", "anchor_status", "anchor_eligible20_count", "fixed_leader_ids", "fixed_leader_names",
    "leader_groups", "known_leader_groups", "leader_unknown_members", "leader_view_allowed", "full_description_view_allowed",
    "leader_mean_cumulative_return", "etf_lagged_five_return", "etf_price_known", "etf_price_positive",
    "original_exit", "extra_exit", "extra_reason", "policy"]


@dataclass
class IndustryData:
    panel: Panel
    members: pd.DataFrame
    calendar: pd.DataFrame
    locations: dict = field(init=False, repr=False)
    slots: dict = field(init=False, repr=False)

    def __post_init__(self):
        if not np.array_equal(pd.to_datetime(self.calendar.date).astype("datetime64[ns]").to_numpy(), self.panel.dates.to_numpy()):
            raise ValueError("行业公布钟与完整成分日历不一致。")
        self.locations = self.members.groupby("date", sort=False).indices
        self.slots = self.calendar.set_index("date").to_dict("index")

    def anchor(self, entry_idx, etf_logs):
        date = self.panel.dates[entry_idx]
        slot = dict(self.slots[date])
        available = pd.Timestamp(slot.get("available_at", pd.NaT))
        opening = date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
        if pd.isna(available):
            slot["row_source_snapshot_eligible"] = False
        else:
            available = available.tz_localize("Asia/Shanghai") if available.tzinfo is None else available.tz_convert("Asia/Shanghai")
            if available > opening:
                slot["row_source_snapshot_eligible"] = False
        members = self.members.iloc[self.locations[date]] if date in self.locations else self.members.iloc[:0]
        feature, anchor, groups = industry.describe(self.panel, entry_idx, members, slot, etf_logs)
        return feature, anchor, groups, available


class FailureController:
    def __init__(self, source: IndustryData, data, policy, original_exit):
        if policy not in POLICIES:
            raise ValueError("未知行业失效退出政策。")
        self.source, self.data, self.policy, self.original_exit = source, data, policy, original_exit
        self.etf_logs = np.log((data.close + data.dividend) / data.close.shift()).to_numpy(float)
        self.checks, self.group_checks = [], []

    def __call__(self, active, row, i, original_policy):
        original_reason = self.original_exit(active, row, i, original_policy)
        entry_idx = int(active["entry_idx"])
        if i - entry_idx != CHECK_LAG:
            return original_reason
        feature, anchor, initial, available = self.source.anchor(entry_idx, self.etf_logs)
        profile, groups = industry.observe_fixed(self.source.panel, anchor, i, self.etf_logs)
        leaders = [group for group in groups if group["leading"]]
        known = [group for group in leaders if group["group_view_allowed"]]
        source_idx, panel = i - 1, self.source.panel
        current_allowed = panel.allowed[source_idx] and panel.members[source_idx].sum() == 300
        # 此用途只判断三领先行业；跟随行业的完整性独立保存，不进入此退出资格。
        leader_allowed = bool(anchor.status == "KNOWN_FIXED_INDUSTRY_ANCHOR" and current_allowed
            and len(leaders) == industry.LEADING_INDUSTRIES and len(known) == len(leaders))
        leader_mean = float(np.mean([group["known_mean_cumulative_return"] for group in known])) if leader_allowed else np.nan
        values = self.etf_logs[entry_idx:i]
        price_known = bool(len(values) == CHECK_LAG and np.isfinite(values).all())
        price_return = float(np.expm1(values.sum())) if price_known else np.nan
        price_positive = bool(price_known and price_return > EPS)
        if self.policy == POLICIES[0]:
            condition = leader_allowed and leader_mean <= EPS and price_positive
        elif self.policy == POLICIES[1]:
            condition = leader_allowed and price_positive
        else:
            condition = price_positive
        extra = bool(condition and original_reason is None)
        reason = f"{self.policy}_FIVE_SOURCE_CLOSES" if extra else None
        self.checks.append({"cycle_id": int(active["cycle_id"]), "entry_date": active["entry_date"], "decision_date": row.date,
            "execution_date": self.data.date.iloc[i + 1] if i + 1 < len(self.data) else pd.NaT,
            "relative_session": i - entry_idx, "anchor_source_date": profile["anchor_source_date"],
            "observation_source_date": profile["observation_source_date"], "classification_snapshot_id": anchor.snapshot_id,
            "classification_available_at": available, "classification_source_age_days": anchor.source_age_days,
            "anchor_status": anchor.status, "anchor_eligible20_count": feature["eligible20_member_count"],
            "fixed_leader_ids": feature["leader_ids"], "fixed_leader_names": feature["leader_names"],
            "leader_groups": len(leaders), "known_leader_groups": len(known),
            "leader_unknown_members": int(sum(group["unknown_members"] for group in leaders)),
            "leader_view_allowed": leader_allowed, "full_description_view_allowed": profile["view_allowed"],
            "leader_mean_cumulative_return": leader_mean, "etf_lagged_five_return": price_return,
            "etf_price_known": price_known, "etf_price_positive": price_positive,
            "original_exit": original_reason, "extra_exit": extra, "extra_reason": reason, "policy": self.policy})
        self.group_checks.extend({"cycle_id": int(active["cycle_id"]), "entry_date": active["entry_date"],
            "decision_date": row.date, "policy": self.policy, **group} for group in groups)
        return original_reason if original_reason is not None else reason


def private_engine():
    spec = importlib.util.spec_from_file_location("_510300_private_stage_industry_leader_failure_v1", Path(stage.__file__))
    if spec is None or spec.loader is None:
        raise RuntimeError("无法建立独立的原阶段引擎。")
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    return engine


def account(data, dividends, risks, source, policy, cost, start):
    if len(data) > len(source.panel.dates) or not np.array_equal(pd.to_datetime(data.date).astype("datetime64[ns]").to_numpy(), source.panel.dates[:len(data)].to_numpy()):
        raise ValueError("账户与行业源的日历/行索引错位。")
    engine = private_engine()
    controller = None
    if policy != "R212_CONTROL":
        controller = FailureController(source, data, policy, engine.exit_decision)
        engine.exit_decision = controller
    result = engine.account(data, dividends, risks, "STAGE_ENTRY_AND_EXIT", cost, start)
    result["industry_checks"] = pd.DataFrame(controller.checks if controller else [], columns=CHECK_COLUMNS)
    result["industry_group_checks"] = pd.DataFrame(controller.group_checks if controller else [])
    return result
