"""VAL01_NORM_EY_5Y 补采后数据闸门专项测试。"""

from __future__ import annotations

import pandas as pd
import pytest

from research.val01_norm_ey_5y_post_acquisition_audit import (
    build_normalized_metric_audit_panel,
    independent_valuation_cross_check,
)


def _synthetic_financials() -> pd.DataFrame:
    periods = pd.to_datetime(
        [
            "2012-06-30", "2012-12-31", "2013-06-30", "2013-12-31",
            "2014-06-30", "2014-12-31", "2015-06-30", "2015-12-31",
            "2016-06-30",
        ]
    )
    cumulative_profit = [4.0, 10.0, 5.0, 12.0, 6.0, 14.0, 7.0, 16.0, 8.0]
    cumulative_revenue = [40.0, 100.0, 50.0, 120.0, 60.0, 140.0, 70.0, 160.0, 80.0]
    return pd.DataFrame(
        {
            "con_code": ["A"] * len(periods),
            "report_period": periods,
            "available_at": periods + pd.Timedelta(days=40),
            "revenue_cny": cumulative_revenue,
            "net_profit_parent_cny": cumulative_profit,
            "equity_parent_cny": [50.0] * len(periods),
            "total_shares": [10.0] * len(periods),
        }
    )


def test_正常化面板只用信号日前财务且不含收益字段() -> None:
    weights = pd.DataFrame(
        {"trade_date": [pd.Timestamp("2016-08-31")], "con_code": ["A"], "weight": [100.0]}
    )
    prices = pd.DataFrame(
        {"date": [pd.Timestamp("2016-08-31")], "con_code": ["A"], "raw_close": [10.0]}
    )
    index_daily = pd.DataFrame(
        {"date": [pd.Timestamp("2016-08-31")], "close": [3000.0]}
    )
    panel = build_normalized_metric_audit_panel(
        weights,
        _synthetic_financials(),
        prices,
        index_daily,
        pd.Timestamp("2016-08-01"),
        pd.Timestamp("2016-08-31"),
    )
    assert len(panel) == 1
    assert panel.loc[0, "normalized_earnings_weight_coverage"] == pytest.approx(1.0)
    assert panel.loc[0, "weighted_normalized_earnings_yield"] > 0
    assert not any("return" in column.lower() or "forward" in column.lower() for column in panel.columns)


def test_独立交叉核验只比较同日尺度() -> None:
    dates = pd.date_range("2016-08-31", periods=6, freq="ME")
    normalized = pd.DataFrame(
        {
            "date": dates,
            "weighted_normalized_earnings_yield": [0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
            "component_normalized_pe": [25.0, 20.0, 16.67, 14.29, 12.5, 11.11],
        }
    )
    official = pd.DataFrame(
        {"date": dates, "pe_official": [30.0, 25.0, 20.0, 18.0, 15.0, 12.0]}
    )
    vendor = pd.DataFrame(
        {"date": dates, "pe_ttm": [28.0, 24.0, 21.0, 17.0, 14.0, 11.0]}
    )
    result = independent_valuation_cross_check(normalized, official, vendor)
    assert result["status"] == "CROSS_CHECK_ONLY_NON_VINTAGE_NOT_A_SIGNAL_INPUT"
    assert result["official_inverse_pe"]["overlap_rows"] == 6
    assert result["official_inverse_pe"]["spearman_correlation"] == pytest.approx(1.0)
    assert result["vendor_inverse_pe"]["spearman_correlation"] == pytest.approx(1.0)
