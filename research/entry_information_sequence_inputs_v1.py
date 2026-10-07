"""因果描述宏观来源身份、公布区间及原进入重复；不计算未来标签或策略。"""
from __future__ import annotations

import numpy as np
import pandas as pd

SOURCES = {
    "orders": ("orders_available_at", "orders_known", ("orders_reference_period",)),
    "policy_rate": ("funding_policy_known_at", None, ("rate",)),
    "funding": ("funding_available_at", "funding_known", ("fund_stat_date",)),
    "margin": ("margin_available_at", "margin_known", ("margin_stat_date",)),
}
TZ = "Asia/Shanghai"


def stamp(value):
    if pd.isna(value):
        return pd.NaT
    result = pd.Timestamp(value)
    return result.tz_localize(TZ) if result.tzinfo is None else result.tz_convert(TZ)


def identity(row, clock, fields):
    values = []
    for field in fields:
        value = row[field]
        if pd.isna(value):
            return None
        values.append(pd.Timestamp(value).isoformat() if isinstance(value, (pd.Timestamp, np.datetime64)) else str(value))
    return "|".join([clock.isoformat(), *values])


def describe(frame):
    data = frame.copy().reset_index(drop=True)
    data["date"] = pd.to_datetime(data.date).astype("datetime64[ns]")
    if not data.date.is_unique or not data.date.is_monotonic_increasing:
        raise ValueError("进入信息日历未唯一排序。")
    seen = {name: {} for name in SOURCES}
    route_counts = {name: {} for name in ("orders", "policy_rate")}
    last_route_idx = {name: {} for name in route_counts}
    rows = []
    previous_decision = pd.NaT
    for i, row in enumerate(data.itertuples(index=False)):
        values = row._asdict()
        decision = stamp(values["decision_time"])
        if pd.isna(decision) or decision.normalize().tz_localize(None) != row.date:
            raise ValueError("原决定钟不在原观察日。")
        result = {"date": row.date, "decision_time": decision, "stage_entry_type": row.stage_entry_type,
            "any_original_stage_event": row.stage_entry_type != "NONE"}
        for name, (clock_field, original_known_field, fields) in SOURCES.items():
            clock = stamp(values[clock_field])
            key = identity(values, clock, fields) if not pd.isna(clock) else None
            original_known = bool(values[original_known_field]) if original_known_field is not None else np.isfinite(values["rate"])
            admitted = bool(original_known and key is not None and not pd.isna(clock) and clock <= decision)
            future_clock = bool(not pd.isna(clock) and clock > decision)
            first_seen = admitted and key not in seen[name]
            if first_seen:
                seen[name][key] = i
            first_idx = seen[name].get(key) if admitted else None
            source_first_date = data.date.iloc[first_idx] if first_idx is not None else pd.NaT
            new_publication = bool(admitted and not pd.isna(previous_decision) and previous_decision < clock <= decision)
            first_observed_without_new_pub = bool(first_seen and not new_publication)
            source_age = (decision-clock).total_seconds()/86400 if admitted else np.nan
            since_first = i-first_idx if first_idx is not None else np.nan
            result.update({f"{name}_source_identity": key, f"{name}_clock_admitted": admitted,
                f"{name}_future_clock": future_clock, f"{name}_first_observed_here": bool(first_seen),
                f"{name}_publication_since_previous_decision": new_publication,
                f"{name}_first_observed_without_new_publication": first_observed_without_new_pub,
                f"{name}_first_observed_date": source_first_date, f"{name}_decisions_since_first_observed": since_first,
                f"{name}_source_age_calendar_days": source_age})
            if name in route_counts:
                event = row.stage_entry_type != "NONE"
                route_key = (key, row.stage_entry_type) if admitted else None
                count = route_counts[name].get(route_key, 0) if admitted else np.nan
                previous_idx = last_route_idx[name].get(route_key) if admitted else None
                result[f"{name}_earlier_same_source_route_events"] = count if event else np.nan
                result[f"{name}_decisions_since_previous_same_source_route_event"] = i-previous_idx if event and previous_idx is not None else np.nan
                if event and admitted:
                    route_counts[name][route_key] = int(count) + 1
                    last_route_idx[name][route_key] = i
        rows.append(result)
        previous_decision = decision
    return pd.DataFrame(rows)
