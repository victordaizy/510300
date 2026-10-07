"""信号前固定价格领先组；后续只用滞后已形成的成分回报观察扩散。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MINIMUM_MEMBERS = 294
EPS = 1e-12


@dataclass(frozen=True)
class Panel:
    dates: pd.DatetimeIndex
    symbols: pd.Index
    logs: np.ndarray
    members: np.ndarray
    allowed: np.ndarray
    return20: np.ndarray


@dataclass(frozen=True)
class Cohort:
    anchor_idx: int
    source_idx: int
    leaders: tuple[int, ...]
    followers: tuple[int, ...]
    member_count: int
    eligible_count: int
    status: str


def make_panel(dates, classified, membership, coverage):
    dates = pd.DatetimeIndex(pd.to_datetime(dates)).astype("datetime64[ns]")
    if not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("观察日历必须唯一递增。")
    q, m, c = classified.copy(), membership.copy(), coverage.copy()
    q["date"] = pd.to_datetime(q.date).astype("datetime64[ns]")
    m["membership_date"] = pd.to_datetime(m.membership_date).astype("datetime64[ns]")
    c["date"] = pd.to_datetime(c.date).astype("datetime64[ns]")
    if q.duplicated(["date", "symbol"]).any() or m.duplicated(["membership_date", "symbol"]).any() or c.date.duplicated().any():
        raise ValueError("源表存在重复日期/成员，不能创建观察。")
    if not m.groupby("membership_date").symbol.nunique().eq(300).all():
        raise ValueError("源成员表不是逐日300只。")
    symbols = pd.Index(sorted(m.symbol.unique()))
    valid = (q.return_is_usable.eq(True) & q.constituent_return_state.isin(["TRADED_VALID", "OFFICIAL_SUSPENSION"])
             & np.isfinite(q.daily_total_shareholder_return) & q.daily_total_shareholder_return.gt(-1))
    values = q.assign(value=q.daily_total_shareholder_return.where(valid)).pivot(index="date", columns="symbol", values="value")
    values = values.reindex(index=dates, columns=symbols)
    logs = np.log1p(values)
    member = m.assign(present=True).pivot(index="membership_date", columns="symbol", values="present")
    member = member.reindex(index=dates, columns=symbols).eq(True)
    allowed = c.set_index("date").aggregation_state.reindex(dates).eq("VIEW_ALLOWED")
    return Panel(dates, symbols, logs.to_numpy(float), member.to_numpy(bool), allowed.to_numpy(bool),
                 logs.rolling(20, min_periods=20).sum().to_numpy(float))


def make_cohort(panel: Panel, anchor_idx: int) -> Cohort:
    source_idx = anchor_idx - 1
    if source_idx < 0 or source_idx >= len(panel.dates):
        return Cohort(anchor_idx, source_idx, (), (), 0, 0, "NO_VIEW_NO_PRIOR_SOURCE_DAY")
    member_count = int(panel.members[source_idx].sum())
    eligible = np.flatnonzero(panel.members[source_idx] & np.isfinite(panel.return20[source_idx]))
    if member_count != 300 or not panel.allowed[source_idx] or len(eligible) < MINIMUM_MEMBERS:
        return Cohort(anchor_idx, source_idx, (), (), member_count, len(eligible), "NO_VIEW_ANCHOR_MEMBERS_OR_RETURNS")
    # 只按信号前一个交易日的过去20日回报排序；代码序号只用于确定并列。
    order = np.lexsort((eligible, -panel.return20[source_idx, eligible]))
    ranked = eligible[order]
    middle = len(ranked) // 2
    return Cohort(anchor_idx, source_idx, tuple(ranked[:middle]), tuple(ranked[middle:]), member_count,
                  len(eligible), "KNOWN_FIXED_PRICE_RANKED_COHORT")


def group_measurements(panel: Panel, cohort: Cohort, decision_idx: int, group: tuple[int, ...]):
    source_idx = decision_idx - 1
    if source_idx < cohort.source_idx or decision_idx >= len(panel.dates):
        raise ValueError("观察必须在锚点之后且落在日历内。")
    size = len(group)
    if not size:
        return {"size": 0, "known": 0, "unknown": 0, "positive": 0, "positive_lower": np.nan,
                "positive_upper": np.nan, "positive_known_fraction": np.nan, "known_median_return": np.nan,
                "known_mean_return": np.nan, "still_members": 0, "view_allowed": False}
    selected = np.asarray(group, dtype=int)
    window = panel.logs[cohort.source_idx + 1:source_idx + 1, selected]
    complete = np.isfinite(window).all(axis=0)
    cumulative = np.expm1(window.sum(axis=0))
    positive = int(np.sum(complete & (cumulative > EPS)))
    known = int(complete.sum())
    unknown = size - known
    source_allowed = bool(panel.allowed[source_idx] and panel.members[source_idx].sum() == 300)
    return {"size": size, "known": known, "unknown": unknown, "positive": positive,
            "positive_lower": positive / size, "positive_upper": (positive + unknown) / size,
            "positive_known_fraction": positive / known if known else np.nan,
            "known_median_return": float(np.median(cumulative[complete])) if known else np.nan,
            "known_mean_return": float(np.mean(cumulative[complete])) if known else np.nan,
            "still_members": int(panel.members[source_idx, selected].sum()),
            "view_allowed": bool(source_allowed and known / size >= .98)}


def observe(panel: Panel, cohort: Cohort, decision_idx: int):
    source_idx = decision_idx - 1
    if decision_idx < cohort.anchor_idx or source_idx < 0:
        raise ValueError("不允许在固定锚点之前观察。")
    row = {"date": panel.dates[decision_idx], "anchor_date": panel.dates[cohort.anchor_idx],
           "cohort_source_date": panel.dates[cohort.source_idx] if cohort.source_idx >= 0 else pd.NaT,
           "observation_source_date": panel.dates[source_idx], "relative_decision_session": decision_idx - cohort.anchor_idx,
           "anchor_status": cohort.status, "anchor_member_count": cohort.member_count,
           "anchor_eligible_count": cohort.eligible_count}
    for name, group in (("leaders", cohort.leaders), ("followers", cohort.followers)):
        row.update({f"{name}_{key}": value for key, value in group_measurements(panel, cohort, decision_idx, group).items()})
    known = (cohort.status == "KNOWN_FIXED_PRICE_RANKED_COHORT" and row["leaders_view_allowed"] and row["followers_view_allowed"])
    row["view_allowed"] = bool(known)
    if not known:
        state = "NO_VIEW_COHORT_OR_POST_ANCHOR_RETURNS"
    elif row["relative_decision_session"] == 0:
        state = "ANCHOR_NO_POST_EVENT_INFORMATION"
    else:
        l_low, l_high = row["leaders_positive_lower"], row["leaders_positive_upper"]
        f_low, f_high = row["followers_positive_lower"], row["followers_positive_upper"]
        if l_low <= .5 < l_high or f_low <= .5 < f_high:
            state = "UNKNOWN_MAJORITY_BOUNDARY"
        else:
            state = ("BOTH_POSITIVE_MAJORITY" if l_low > .5 and f_low > .5 else
                     "LEADERS_ONLY_POSITIVE_MAJORITY" if l_low > .5 else
                     "FOLLOWERS_ONLY_POSITIVE_MAJORITY" if f_low > .5 else "NEITHER_POSITIVE_MAJORITY")
    row["descriptive_propagation_state"] = state
    return row
