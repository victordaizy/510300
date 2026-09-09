"""点时指数驱动归因 V1.1 区间归一化测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from research.index_driver_attribution import AttributionRules
from research.index_driver_attribution_v1_1 import (
    build_index_driver_attribution_v1_1,
    coalesce_same_l1_intervals,
)


def _intervals(second_industry: str = "汽车") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "con_code": ["A.SH", "A.SH", "B.SZ"],
            "industry_l1": ["汽车", second_industry, "银行"],
            "in_date": pd.to_datetime(["2020-01-01", "2024-01-01", "2020-01-01"]),
            "out_date": pd.to_datetime(["2025-01-01", None, None]),
            "classification_usage": ["POINT_IN_TIME_INTERVAL_CITIC"] * 3,
            "source": ["中信测试"] * 3,
        }
    )


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index_code": ["000300.SH", "000300.SH"],
            "con_code": ["A.SH", "B.SZ"],
            "trade_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "weight": [60.0, 40.0],
            "source": ["官方", "官方"],
        }
    )


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]
            ),
            "con_code": ["A.SH", "B.SZ", "A.SH", "B.SZ"],
            "total_return_close": [100.0, 100.0, 101.0, 99.0],
            "is_suspended": [False] * 4,
        }
    )


def _rules() -> AttributionRules:
    return AttributionRules(
        start_date=pd.Timestamp("2024-01-03"),
        end_date=pd.Timestamp("2024-01-03"),
        member_count_minimum=2,
        member_count_maximum=2,
        weight_sum_minimum=0.98,
        weight_sum_maximum=1.02,
        maximum_weight_age_days=45,
        minimum_price_coverage_weight=0.99,
        minimum_return_coverage_weight=0.95,
        minimum_industry_coverage_weight=0.95,
        maximum_identity_error=1.0e-12,
    )


def test_same_l1_overlapping_intervals_are_coalesced() -> None:
    normalized = coalesce_same_l1_intervals(_intervals())
    a_intervals = normalized.loc[normalized["con_code"].eq("A.SH")]
    assert len(a_intervals) == 1
    assert a_intervals.iloc[0]["in_date"] == pd.Timestamp("2020-01-01")
    assert pd.isna(a_intervals.iloc[0]["out_date"])
    result = build_index_driver_attribution_v1_1(
        _weights(), _prices(), _intervals(), _rules()
    )
    assert result.daily.iloc[0]["output"] == "ATTRIBUTION"


def test_conflicting_l1_overlapping_intervals_remain_hard_failure() -> None:
    with pytest.raises(ValueError, match="重复有效行业区间"):
        build_index_driver_attribution_v1_1(
            _weights(), _prices(), _intervals(second_industry="机械"), _rules()
        )
