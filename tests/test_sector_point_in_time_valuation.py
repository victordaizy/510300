"""点时板块估值面板的数据时点与覆盖测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.sector_point_in_time_valuation import (
    SectorPanelRules,
    build_sector_point_in_time_panel,
)


def _rules(**overrides: object) -> SectorPanelRules:
    values: dict[str, object] = {
        "start_date": pd.Timestamp("2025-04-30"),
        "end_date": pd.Timestamp("2025-04-30"),
        "minimum_snapshot_count": 1,
        "forecast_horizon_days": 1,
        "minimum_nonoverlapping_blocks": 1,
        "member_count_minimum": 4,
        "member_count_maximum": 4,
        "weight_sum_minimum": 0.98,
        "weight_sum_maximum": 1.02,
        "minimum_price_coverage": 0.99,
        "minimum_industry_coverage": 0.95,
        "minimum_earnings_coverage": 0.90,
        "minimum_book_coverage": 0.90,
        "minimum_roe_coverage": 0.90,
        "minimum_sector_component_count": 2,
        "minimum_sector_feature_coverage": 0.80,
        "minimum_eligible_sector_weight_coverage": 0.90,
    }
    values.update(overrides)
    return SectorPanelRules(**values)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    symbols = ["A1", "A2", "B1", "B2"]
    weights = pd.DataFrame(
        {
            "con_code": symbols,
            "trade_date": pd.Timestamp("2025-04-30"),
            "weight": [30.0, 20.0, 30.0, 20.0],
            "source": "测试官方权重",
        }
    )
    rows: list[dict[str, object]] = []
    periods = pd.to_datetime(
        ["2023-12-31", "2024-03-31", "2024-12-31", "2025-03-31"]
    )
    for symbol in symbols:
        for index, period in enumerate(periods):
            rows.append(
                {
                    "con_code": symbol,
                    "report_period": period,
                    "available_at": period + pd.Timedelta(days=30),
                    "revenue_cny": [80.0, 25.0, 100.0, 30.0][index],
                    "net_profit_parent_cny": [8.0, 2.5, 10.0, 3.0][index],
                    "equity_parent_cny": 50.0,
                    "total_shares": 10.0,
                    "bps": 5.0,
                }
            )
    rows.append(
        {
            "con_code": "A1",
            "report_period": pd.Timestamp("2025-03-31"),
            "available_at": pd.Timestamp("2025-05-10"),
            "revenue_cny": 300.0,
            "net_profit_parent_cny": 30.0,
            "equity_parent_cny": 50.0,
            "total_shares": 10.0,
            "bps": 5.0,
        }
    )
    financials = pd.DataFrame(rows)
    daily = pd.DataFrame(
        {
            "date": pd.Timestamp("2025-04-30"),
            "con_code": symbols,
            "raw_close": [10.0, 20.0, 10.0, 20.0],
        }
    )
    industries = pd.DataFrame(
        {
            "con_code": symbols,
            "industry_l1": ["行业A", "行业A", "行业B", "行业B"],
            "in_date": pd.Timestamp("2020-01-01"),
            "out_date": pd.NaT,
            "classification_usage": "POINT_IN_TIME_INTERVAL_CITIC",
            "source": "测试点时行业",
        }
    )
    return weights, financials, daily, industries


def test_future_financial_revision_does_not_rewrite_snapshot() -> None:
    result = build_sector_point_in_time_panel(*_inputs(), _rules())
    sector_a = result.sector_panel.loc[
        result.sector_panel["industry_l1"].eq("行业A")
    ].iloc[0]
    expected_a1_ttm_eps = (3.0 + 10.0 - 2.5) / 10.0
    expected_a2_ttm_eps = expected_a1_ttm_eps
    expected = np.average(
        [expected_a1_ttm_eps / 10.0, expected_a2_ttm_eps / 20.0],
        weights=[0.30, 0.20],
    )
    assert np.isclose(sector_a["weighted_earnings_yield"], expected)
    assert result.feasibility["status"] == "READY_FOR_TARGET_FREEZE"


def test_sector_component_gate_cannot_be_hidden_by_reweighting() -> None:
    weights, financials, daily, industries = _inputs()
    rules = _rules(minimum_sector_component_count=3)
    result = build_sector_point_in_time_panel(
        weights, financials, daily, industries, rules
    )
    assert not result.sector_panel["eligible_for_target_freeze"].any()
    assert result.snapshot_audit.iloc[0]["output"] == "NO_VIEW"
    assert result.feasibility["status"] == "NO_VIEW"
