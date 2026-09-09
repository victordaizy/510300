"""点时指数驱动归因 V1.3：先限定冻结研究窗口。"""

from __future__ import annotations

import pandas as pd

from research.index_driver_attribution import AttributionRules, IndexDriverAttribution
from research.index_driver_attribution_v1_1 import build_index_driver_attribution_v1_1
from research.index_driver_attribution_v1_2 import truncate_superseded_l1_intervals


def restrict_industry_intervals_to_research_window(
    industry_intervals: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """保留与冻结研究窗口相交的半开行业区间。"""

    required = {"in_date", "out_date"}
    missing = sorted(required.difference(industry_intervals.columns))
    if missing:
        raise ValueError(f"点时行业区间缺少窗口字段：{missing}")
    data = industry_intervals.copy()
    data["in_date"] = pd.to_datetime(data["in_date"], errors="coerce")
    data["out_date"] = pd.to_datetime(data["out_date"], errors="coerce")
    if data["in_date"].isna().any():
        raise ValueError("点时行业区间存在空生效日")
    return data.loc[
        data["in_date"].le(pd.Timestamp(end_date))
        & (data["out_date"].isna() | data["out_date"].gt(pd.Timestamp(start_date)))
    ].copy()


def build_index_driver_attribution_v1_3(
    weights: pd.DataFrame,
    constituent_daily: pd.DataFrame,
    industry_intervals: pd.DataFrame,
    rules: AttributionRules,
) -> IndexDriverAttribution:
    """限定窗口并解析行业区间后复用冻结归因实现。"""

    relevant = restrict_industry_intervals_to_research_window(
        industry_intervals, rules.start_date, rules.end_date
    )
    resolved = truncate_superseded_l1_intervals(relevant)
    return build_index_driver_attribution_v1_1(
        weights, constituent_daily, resolved, rules
    )


__all__ = [
    "AttributionRules",
    "IndexDriverAttribution",
    "build_index_driver_attribution_v1_3",
    "restrict_industry_intervals_to_research_window",
]

