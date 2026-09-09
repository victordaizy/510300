"""点时指数截面与板块贡献归因测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.index_driver_attribution import (
    AttributionRules,
    build_index_driver_attribution,
)


def _rules(
    start: str = "2025-01-03",
    end: str = "2025-01-06",
    industry_coverage: float = 0.95,
    maximum_weight_age_days: int = 45,
) -> AttributionRules:
    return AttributionRules(
        start_date=pd.Timestamp(start),
        end_date=pd.Timestamp(end),
        member_count_minimum=2,
        member_count_maximum=2,
        weight_sum_minimum=0.98,
        weight_sum_maximum=1.02,
        maximum_weight_age_days=maximum_weight_age_days,
        minimum_price_coverage_weight=0.99,
        minimum_return_coverage_weight=0.95,
        minimum_industry_coverage_weight=industry_coverage,
        maximum_identity_error=1.0e-12,
    )


def _weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "index_code": ["000300.SH"] * 4,
            "con_code": ["A.SH", "B.SZ", "A.SH", "B.SZ"],
            "trade_date": pd.to_datetime(
                ["2025-01-02", "2025-01-02", "2025-01-06", "2025-01-06"]
            ),
            "weight": [60.0, 40.0, 20.0, 80.0],
            "source": ["官方"] * 4,
        }
    )


def _prices() -> pd.DataFrame:
    dates = pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"])
    rows = []
    for code, values in (("A.SH", [100.0, 110.0, 121.0]), ("B.SZ", [100.0, 90.0, 99.0])):
        for date, value in zip(dates, values, strict=True):
            rows.append(
                {
                    "date": date,
                    "con_code": code,
                    "total_return_close": value,
                    "is_suspended": False,
                }
            )
    return pd.DataFrame(rows)


def _industries() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "con_code": ["A.SH", "A.SH", "B.SZ"],
            "industry_l1": ["旧行业", "新行业", "稳定行业"],
            "in_date": pd.to_datetime(["2020-01-01", "2025-01-03", "2020-01-01"]),
            "out_date": pd.to_datetime(["2025-01-03", None, None]),
            "classification_usage": ["POINT_IN_TIME_INTERVAL_CITIC"] * 3,
            "source": ["测试点时行业"] * 3,
        }
    )


def test_uses_latest_prior_weights_and_right_open_industry_intervals() -> None:
    result = build_index_driver_attribution(
        _weights(), _prices(), _industries(), _rules()
    )
    january_third = result.daily.loc[
        result.daily["date"].eq(pd.Timestamp("2025-01-03"))
    ].iloc[0]
    january_sixth = result.daily.loc[
        result.daily["date"].eq(pd.Timestamp("2025-01-06"))
    ].iloc[0]
    assert january_third["weight_snapshot_date"] == pd.Timestamp("2025-01-02")
    assert january_sixth["weight_snapshot_date"] == pd.Timestamp("2025-01-06")
    assert np.isclose(january_third["snapshot_weighted_constituent_return_1d"], 0.02)
    assert np.isclose(january_sixth["snapshot_weighted_constituent_return_1d"], 0.10)
    a_row = result.cross_section.loc[
        result.cross_section["date"].eq(pd.Timestamp("2025-01-03"))
        & result.cross_section["con_code"].eq("A.SH")
    ].iloc[0]
    assert a_row["industry_l1"] == "新行业"
    assert result.daily["valid_for_attribution"].all()


def test_component_and_industry_contributions_have_same_identity() -> None:
    result = build_index_driver_attribution(
        _weights(), _prices(), _industries(), _rules()
    )
    assert np.allclose(
        result.daily["attributed_component_contribution_1d"],
        result.daily["industry_contribution_sum_1d"],
        atol=1.0e-15,
    )
    assert result.daily["contribution_identity_absolute_error"].max() <= 1.0e-15
    grouped = result.industry.groupby("date")[
        "weighted_return_contribution_1d"
    ].sum()
    expected = result.daily.set_index("date")[
        "attributed_component_contribution_1d"
    ]
    assert np.allclose(grouped.loc[expected.index], expected)


def test_security_gap_does_not_create_fake_one_day_return() -> None:
    prices = _prices()
    prices = prices.loc[
        ~(
            prices["con_code"].eq("A.SH")
            & prices["date"].eq(pd.Timestamp("2025-01-03"))
        )
    ].copy()
    result = build_index_driver_attribution(
        _weights(), prices, _industries(), _rules(start="2025-01-06")
    )
    a_row = result.cross_section.loc[
        result.cross_section["date"].eq(pd.Timestamp("2025-01-06"))
        & result.cross_section["con_code"].eq("A.SH")
    ].iloc[0]
    assert pd.isna(a_row["constituent_return_1d"])
    assert result.daily.iloc[0]["output"] == "NO_VIEW"
    assert "RETURN_COVERAGE_BELOW_GATE" in result.daily.iloc[0]["failure_category"]


def test_industry_coverage_below_gate_outputs_no_view() -> None:
    industries = _industries().loc[lambda frame: frame["con_code"].eq("A.SH")]
    result = build_index_driver_attribution(
        _weights(), _prices(), industries, _rules(industry_coverage=0.95)
    )
    january_sixth = result.daily.loc[
        result.daily["date"].eq(pd.Timestamp("2025-01-06"))
    ].iloc[0]
    assert january_sixth["industry_coverage_weight"] == pytest.approx(0.20)
    assert january_sixth["output"] == "NO_VIEW"
    assert "INDUSTRY_COVERAGE_BELOW_GATE" in january_sixth["failure_category"]
    assert january_sixth["unattributed_contribution_1d"] == pytest.approx(0.08)


def test_stale_weight_snapshot_outputs_no_view() -> None:
    weights = _weights().loc[
        _weights()["trade_date"].eq(pd.Timestamp("2025-01-02"))
    ]
    result = build_index_driver_attribution(
        weights,
        _prices(),
        _industries(),
        _rules(start="2025-01-06", maximum_weight_age_days=2),
    )
    assert result.daily.iloc[0]["output"] == "NO_VIEW"
    assert "STALE_WEIGHT_SNAPSHOT" in result.daily.iloc[0]["failure_category"]


def test_future_industry_interval_cannot_enter_current_attribution() -> None:
    industries = _industries().copy()
    industries.loc[
        industries["con_code"].eq("B.SZ"), "in_date"
    ] = pd.Timestamp("2025-01-07")
    result = build_index_driver_attribution(
        _weights(), _prices(), industries, _rules()
    )
    b_row = result.cross_section.loc[
        result.cross_section["date"].eq(pd.Timestamp("2025-01-06"))
        & result.cross_section["con_code"].eq("B.SZ")
    ].iloc[0]
    assert not b_row["industry_available"]
    assert result.daily.iloc[-1]["output"] == "NO_VIEW"


def test_duplicate_active_industry_intervals_are_rejected() -> None:
    industries = pd.concat(
        [
            _industries(),
            pd.DataFrame(
                {
                    "con_code": ["B.SZ"],
                    "industry_l1": ["冲突行业"],
                    "in_date": pd.to_datetime(["2024-01-01"]),
                    "out_date": pd.to_datetime([None]),
                    "classification_usage": ["POINT_IN_TIME_INTERVAL_CITIC"],
                    "source": ["冲突测试"],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="重复有效行业区间"):
        build_index_driver_attribution(
            _weights(), _prices(), industries, _rules()
        )


def test_duplicate_security_date_is_rejected() -> None:
    prices = pd.concat([_prices(), _prices().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="重复证券日期"):
        build_index_driver_attribution(
            _weights(), prices, _industries(), _rules()
        )
