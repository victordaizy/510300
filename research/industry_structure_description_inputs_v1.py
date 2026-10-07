"""当时固定行业集合的相对强弱与已形成传播，不推测未知归属或回报。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.broker_fixed_cohort_inputs_v1 import Panel

MINIMUM_MEMBERS = 294
MINIMUM_INDUSTRY_MEMBERS = 5
LEADING_INDUSTRIES = 3
COVERAGE = .98
EPS = 1e-12


@dataclass(frozen=True)
class IndustryGroup:
    key: str
    name: str
    members: tuple[int, ...]
    leading: bool


@dataclass(frozen=True)
class IndustryAnchor:
    anchor_idx: int
    source_idx: int
    snapshot_id: str | None
    taxonomy: str | None
    source_age_days: float
    groups: tuple[IndustryGroup, ...]
    status: str


def affirmative(value):
    return bool(value) if not pd.isna(value) else False


def etf_interval(logs, start, stop):
    values = np.asarray(logs[start:stop], dtype=float)
    return float(np.expm1(values.sum())) if np.isfinite(values).all() else np.nan


def summarize_groups(groups, etf20, etf5, etf_previous5):
    frame = pd.DataFrame(groups).sort_values("industry_key").reset_index(drop=True)
    current_rank = frame.industry_return5.rank(method="average", ascending=False)
    previous_rank = frame.industry_previous5.rank(method="average", ascending=False)
    frame["current5_rank"], frame["previous5_rank"] = current_rank, previous_rank
    frame["rank_change5blocks"] = current_rank - previous_rank
    order = frame.sort_values(["industry_return20", "industry_key"], ascending=[False, True])
    leader_ids = order.industry_key.head(LEADING_INDUSTRIES).tolist()
    frame["leading"] = frame.industry_key.isin(leader_ids)
    frame["industry_relative20"] = frame.industry_return20 - etf20
    frame["industry_relative5"] = frame.industry_return5 - etf5
    leaders = frame.loc[frame.leading]
    mean5 = float(leaders.industry_return5.mean())
    relative5 = mean5 - etf5
    previous_relative5 = float(leaders.industry_previous5.mean()) - etf_previous5
    sign = ("ETF_AND_LEADER5_POSITIVE" if etf5 > EPS and mean5 > EPS else
        "ETF5_NONPOSITIVE_LEADER5_POSITIVE" if etf5 <= EPS and mean5 > EPS else
        "ETF5_POSITIVE_LEADER5_NONPOSITIVE" if etf5 > EPS else "ETF_AND_LEADER5_NONPOSITIVE")
    features = {"retained_industry_count": len(frame), "retained_member_count": int(frame.member_count.sum()),
        "industry_positive5_fraction": float(frame.industry_return5.gt(EPS).mean()),
        "industry_relative_positive5_fraction": float(frame.industry_relative5.gt(EPS).mean()),
        "rotation_churn5blocks": float(frame.rank_change5blocks.abs().mean() / (len(frame) - 1)),
        "leader_ids": ";".join(leader_ids), "leader_names": ";".join(order.major_name.head(LEADING_INDUSTRIES)),
        "leader_mean_relative20": float(leaders.industry_relative20.mean()), "leader_mean_return5": mean5,
        "leader_mean_relative5": relative5, "leader_previous_relative5": previous_relative5,
        "leader_relative_advantage_change5blocks": relative5 - previous_relative5,
        "descriptive_structure_state": sign}
    return features, frame.to_dict("records")


def describe(panel: Panel, decision_idx: int, members: pd.DataFrame, source_slot, etf_logs):
    source_idx = decision_idx - 1
    fields = {"date": panel.dates[decision_idx], "industry_source_date": panel.dates[source_idx] if source_idx >= 0 else pd.NaT,
        "industry_snapshot_id": source_slot.get("snapshot_id"), "industry_taxonomy": source_slot.get("snapshot_taxonomy"),
        "industry_source_age_days": source_slot.get("source_age_calendar_days", np.nan),
        "member_count": len(members), "classified_member_count": int(members.row_source_known.eq(True).sum()), "eligible20_member_count": 0,
        "retained_industry_count": 0, "retained_member_count": 0, "view_allowed": False,
        "leader_ids": "", "leader_names": "", "industry_positive5_fraction": np.nan,
        "industry_relative_positive5_fraction": np.nan, "rotation_churn5blocks": np.nan,
        "leader_mean_relative20": np.nan, "leader_mean_return5": np.nan, "leader_mean_relative5": np.nan,
        "leader_previous_relative5": np.nan, "leader_relative_advantage_change5blocks": np.nan,
        "etf_return20_same_lagged_interval": np.nan, "etf_return5_same_lagged_interval": np.nan,
        "descriptive_structure_state": "NO_VIEW_INDUSTRY_MEMBERS_OR_QUOTES"}
    groups = ()
    def finish():
        return fields, IndustryAnchor(decision_idx, source_idx, fields["industry_snapshot_id"], fields["industry_taxonomy"],
            fields["industry_source_age_days"], groups, fields["descriptive_structure_state"]), []
    if source_idx < 19:
        fields["descriptive_structure_state"] = "NO_VIEW_LESS_THAN_20_PRIOR_SESSIONS"
        return finish()
    if pd.Timestamp(source_slot.get("membership_source_date")) != panel.dates[source_idx]:
        raise ValueError("行业成员源不是观察日前一完整ETF日。")
    if members.symbol.duplicated().any():
        raise ValueError("当时行业成员证券重复。")
    if len(members) != 300 or not affirmative(source_slot.get("row_source_snapshot_eligible", False)) or not panel.allowed[source_idx]:
        return finish()
    indices = panel.symbols.get_indexer(members.symbol)
    if (indices < 0).any():
        raise ValueError("行业成员不在原价格成员全集中。")
    known = members.row_source_known.eq(True).to_numpy(bool)
    complete20 = np.isfinite(panel.return20[source_idx, indices])
    fields["classified_member_count"] = int(known.sum())
    fields["eligible20_member_count"] = int((known & complete20).sum())
    if fields["eligible20_member_count"] < MINIMUM_MEMBERS:
        return finish()
    past20 = etf_interval(etf_logs, source_idx - 19, source_idx + 1)
    past5 = etf_interval(etf_logs, source_idx - 4, source_idx + 1)
    previous5 = etf_interval(etf_logs, source_idx - 9, source_idx - 4)
    if not np.isfinite([past20, past5, previous5]).all():
        fields["descriptive_structure_state"] = "NO_VIEW_ETF_INTERVAL"
        return finish()
    chosen = members.loc[known].copy()
    chosen["panel_symbol_index"] = indices[known]
    records, identities = [], {}
    for key, values in chosen.groupby("industry_key", sort=True):
        selected = values.panel_symbol_index.to_numpy(int)
        complete = np.isfinite(panel.return20[source_idx, selected])
        if len(selected) < MINIMUM_INDUSTRY_MEMBERS or complete.mean() < COVERAGE:
            continue
        history = selected[complete]
        identities[key] = tuple(selected)
        records.append({"industry_key": key, "major_name": values.major_name.iloc[0], "member_count": len(selected),
            "complete20_member_count": int(complete.sum()),
            "industry_return20": float(np.expm1(panel.return20[source_idx, history]).mean()),
            "industry_return5": float(np.expm1(panel.logs[source_idx - 4:source_idx + 1, history].sum(axis=0)).mean()),
            "industry_previous5": float(np.expm1(panel.logs[source_idx - 9:source_idx - 4, history].sum(axis=0)).mean())})
    if len(records) <= LEADING_INDUSTRIES:
        fields["descriptive_structure_state"] = "NO_VIEW_TOO_FEW_RETAINED_INDUSTRIES"
        return finish()
    features, stats = summarize_groups(records, past20, past5, previous5)
    fields.update(features)
    fields.update(view_allowed=True, etf_return20_same_lagged_interval=past20, etf_return5_same_lagged_interval=past5)
    groups = tuple(IndustryGroup(row["industry_key"], row["major_name"], identities[row["industry_key"]], row["leading"]) for row in stats)
    anchor = IndustryAnchor(decision_idx, source_idx, fields["industry_snapshot_id"], fields["industry_taxonomy"],
        fields["industry_source_age_days"], groups, "KNOWN_FIXED_INDUSTRY_ANCHOR")
    return fields, anchor, stats


def observe_fixed(panel: Panel, anchor: IndustryAnchor, decision_idx: int, etf_logs):
    source_idx = decision_idx - 1
    if decision_idx < anchor.anchor_idx or decision_idx >= len(panel.dates):
        raise ValueError("固定行业观察不在锚点之后或日历内。")
    row = {"date": panel.dates[decision_idx], "anchor_date": panel.dates[anchor.anchor_idx],
        "anchor_source_date": panel.dates[anchor.source_idx] if anchor.source_idx >= 0 else pd.NaT,
        "observation_source_date": panel.dates[source_idx] if source_idx >= 0 else pd.NaT,
        "relative_session": decision_idx - anchor.anchor_idx, "anchor_status": anchor.status,
        "industry_snapshot_id": anchor.snapshot_id, "industry_taxonomy": anchor.taxonomy,
        "anchor_industry_source_age_days": anchor.source_age_days, "view_allowed": False,
        "leader_fixed_mean_cumulative_return": np.nan, "follower_fixed_mean_cumulative_return": np.nan,
        "leader_fixed_excess_over_etf": np.nan, "descriptive_fixed_state": "NO_VIEW_FIXED_INDUSTRY_OR_POST_QUOTES"}
    records = []
    for group in anchor.groups:
        selected = np.asarray(group.members, dtype=int)
        window = panel.logs[anchor.source_idx + 1:source_idx + 1, selected]
        complete = np.isfinite(window).all(axis=0)
        known = int(complete.sum())
        cumulative = np.expm1(window[:, complete].sum(axis=0))
        observed = known / len(selected) >= COVERAGE
        records.append({"date": panel.dates[decision_idx], "industry_key": group.key, "major_name": group.name,
            "leading": group.leading, "anchor_member_count": len(selected), "known_members": known,
            "unknown_members": len(selected) - known, "group_view_allowed": observed,
            "known_mean_cumulative_return": float(cumulative.mean()) if known else np.nan,
            "positive_member_lower": float(np.sum(cumulative > EPS) / len(selected)),
            "positive_member_upper": float((np.sum(cumulative > EPS) + len(selected) - known) / len(selected))})
    for leading, name in ((True, "leader"), (False, "follower")):
        values = [record for record in records if record["leading"] == leading]
        valid = [record for record in values if record["group_view_allowed"]]
        row[name + "_industry_count"] = len(values)
        row[name + "_known_industry_count"] = len(valid)
        row[name + "_known_fraction"] = len(valid) / len(values) if values else np.nan
        row[name + "_fixed_mean_cumulative_return"] = float(np.mean([record["known_mean_cumulative_return"] for record in valid])) if valid else np.nan
    etf = etf_interval(etf_logs, anchor.source_idx + 1, source_idx + 1) if anchor.source_idx >= 0 else np.nan
    row["etf_cumulative_same_lagged_interval"] = etf
    current_allowed = source_idx >= 0 and panel.allowed[source_idx] and panel.members[source_idx].sum() == 300
    observed = bool(anchor.status == "KNOWN_FIXED_INDUSTRY_ANCHOR" and current_allowed
        and row["leader_known_fraction"] >= COVERAGE and row["follower_known_fraction"] >= COVERAGE and np.isfinite(etf))
    row["view_allowed"] = observed
    if observed:
        leader = row["leader_fixed_mean_cumulative_return"]
        row["leader_fixed_excess_over_etf"] = leader - etf
        row["descriptive_fixed_state"] = ("ANCHOR_NO_POST_EVENT_INFORMATION" if decision_idx == anchor.anchor_idx else
            "ETF_AND_FIXED_LEADERS_POSITIVE" if etf > EPS and leader > EPS else
            "ETF_NONPOSITIVE_FIXED_LEADERS_POSITIVE" if etf <= EPS and leader > EPS else
            "ETF_POSITIVE_FIXED_LEADERS_NONPOSITIVE" if etf > EPS else "ETF_AND_FIXED_LEADERS_NONPOSITIVE")
    return row, records
