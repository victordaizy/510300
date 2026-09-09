"""点时指数驱动归因 V1.2：以最新生效日解决跨一级行业重叠。"""

from __future__ import annotations

from typing import Any

import pandas as pd

from research.index_driver_attribution import AttributionRules, IndexDriverAttribution
from research.index_driver_attribution_v1_1 import (
    build_index_driver_attribution_v1_1,
    coalesce_same_l1_intervals,
)


def truncate_superseded_l1_intervals(
    industry_intervals: pd.DataFrame,
) -> pd.DataFrame:
    """新一级行业生效时截断仍重叠的旧一级行业区间。"""

    normalized = coalesce_same_l1_intervals(industry_intervals)
    rows: list[dict[str, Any]] = []
    for con_code, group in normalized.groupby("con_code", sort=True):
        ordered = group.sort_values(["in_date", "industry_l1"]).reset_index(drop=True)
        same_start_counts = ordered.groupby("in_date")["industry_l1"].nunique()
        ambiguous_dates = same_start_counts.loc[same_start_counts.gt(1)]
        if not ambiguous_dates.empty:
            dates = [str(pd.Timestamp(value).date()) for value in ambiguous_dates.index]
            raise ValueError(
                f"{con_code}同一最新生效日存在多个一级行业：{dates}"
            )
        records = ordered.to_dict("records")
        for index, record in enumerate(records):
            current_start = pd.Timestamp(record["in_date"])
            current_end = (
                None
                if pd.isna(record["out_date"])
                else pd.Timestamp(record["out_date"])
            )
            later_different_starts = [
                pd.Timestamp(candidate["in_date"])
                for candidate in records[index + 1 :]
                if candidate["industry_l1"] != record["industry_l1"]
                and pd.Timestamp(candidate["in_date"]) > current_start
            ]
            if later_different_starts:
                superseded_at = min(later_different_starts)
                if current_end is None or superseded_at < current_end:
                    current_end = superseded_at
            if current_end is not None and current_end <= current_start:
                continue
            rows.append(
                {
                    "con_code": str(record["con_code"]),
                    "industry_l1": str(record["industry_l1"]),
                    "classification_usage": record["classification_usage"],
                    "source": record["source"],
                    "in_date": current_start,
                    "out_date": pd.NaT if current_end is None else current_end,
                }
            )
    resolved = pd.DataFrame(rows)
    return coalesce_same_l1_intervals(resolved)


def build_index_driver_attribution_v1_2(
    weights: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    industry_intervals: pd.DataFrame,
    rules: AttributionRules,
) -> IndexDriverAttribution:
    """按最新一级行业生效日归一化后复用冻结归因实现。"""

    resolved = truncate_superseded_l1_intervals(industry_intervals)
    return build_index_driver_attribution_v1_1(
        weights, constituent_daily, resolved, rules
    )


__all__ = [
    "AttributionRules",
    "IndexDriverAttribution",
    "build_index_driver_attribution_v1_2",
    "truncate_superseded_l1_intervals",
]
