"""点时指数驱动归因 V1.1：归一化同一级行业的重叠区间。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.index_driver_attribution import (
    AttributionRules,
    IndexDriverAttribution,
    build_index_driver_attribution,
)


def coalesce_same_l1_intervals(industry_intervals: pd.DataFrame) -> pd.DataFrame:
    """合并同证券、同一级行业的重叠或相邻半开区间。"""

    required = {
        "con_code",
        "industry_l1",
        "in_date",
        "out_date",
        "classification_usage",
        "source",
    }
    missing = sorted(required.difference(industry_intervals.columns))
    if missing:
        raise ValueError(f"点时行业区间缺少字段：{missing}")
    data = industry_intervals[list(required)].copy()
    data["con_code"] = data["con_code"].astype(str)
    data["industry_l1"] = data["industry_l1"].astype("string")
    data["in_date"] = pd.to_datetime(data["in_date"], errors="coerce")
    data["out_date"] = pd.to_datetime(data["out_date"], errors="coerce")
    if data[["con_code", "industry_l1", "in_date"]].isna().any().any():
        raise ValueError("点时行业区间存在空证券、空一级行业或空生效日")

    rows: list[dict[str, Any]] = []
    group_columns = [
        "con_code",
        "industry_l1",
        "classification_usage",
        "source",
    ]
    for keys, group in data.groupby(group_columns, sort=True, dropna=False):
        ordered = group.sort_values(["in_date", "out_date"], na_position="last")
        current_start: pd.Timestamp | None = None
        current_end: pd.Timestamp | None = None
        current_is_open = False
        for interval in ordered.itertuples(index=False):
            start = pd.Timestamp(interval.in_date)
            end_is_open = pd.isna(interval.out_date)
            end = None if end_is_open else pd.Timestamp(interval.out_date)
            if current_start is None:
                current_start = start
                current_end = end
                current_is_open = end_is_open
                continue
            overlaps_or_touches = current_is_open or (
                current_end is not None and start <= current_end
            )
            if overlaps_or_touches:
                if end_is_open:
                    current_end = None
                    current_is_open = True
                elif not current_is_open and current_end is not None:
                    current_end = max(current_end, end)
                continue
            rows.append(
                {
                    "con_code": keys[0],
                    "industry_l1": keys[1],
                    "classification_usage": keys[2],
                    "source": keys[3],
                    "in_date": current_start,
                    "out_date": pd.NaT if current_is_open else current_end,
                }
            )
            current_start = start
            current_end = end
            current_is_open = end_is_open
        if current_start is not None:
            rows.append(
                {
                    "con_code": keys[0],
                    "industry_l1": keys[1],
                    "classification_usage": keys[2],
                    "source": keys[3],
                    "in_date": current_start,
                    "out_date": pd.NaT if current_is_open else current_end,
                }
            )
    return pd.DataFrame(rows).sort_values(["con_code", "in_date"]).reset_index(
        drop=True
    )


def build_index_driver_attribution_v1_1(
    weights: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    industry_intervals: pd.DataFrame,
    rules: AttributionRules,
) -> IndexDriverAttribution:
    """归一化同一级行业区间后复用 V1 冻结归因实现。"""

    normalized_intervals = coalesce_same_l1_intervals(industry_intervals)
    return build_index_driver_attribution(
        weights, constituent_daily, normalized_intervals, rules
    )


__all__ = [
    "AttributionRules",
    "IndexDriverAttribution",
    "build_index_driver_attribution_v1_1",
    "coalesce_same_l1_intervals",
]

