from __future__ import annotations

import numpy as np
import pandas as pd

from research.val04_pb_roe_residual_data_feasibility_v1 import (
    active_industry_for_date,
    audit_snapshot,
    design_matrix_audit,
    prepare_snapshot_prices,
)


def _config() -> dict:
    return {
        "quality_gates": {
            "member_count_range": [300, 300],
            "weight_sum_range": [0.98, 1.02],
            "minimum_price_weight_coverage": 0.99,
            "minimum_industry_weight_coverage": 0.95,
            "minimum_book_to_price_weight_coverage": 0.90,
            "minimum_roe_weight_coverage": 0.90,
            "minimum_revenue_growth_weight_coverage": 0.90,
            "minimum_complete_case_weight_coverage": 0.90,
            "minimum_complete_company_count": 200,
            "minimum_active_industry_count": 20,
            "minimum_design_residual_degrees_of_freedom": 150,
            "duplicate_weight_security_count": 0,
            "future_financial_event_count": 0,
        }
    }


def test_snapshot_price_prefers_dedicated_historical_table() -> None:
    date = pd.Timestamp("2020-01-31")
    current = pd.DataFrame(
        {"date": [date], "con_code": ["000001.SZ"], "raw_close": [10.0]}
    )
    historical = pd.DataFrame(
        {"date": [date], "con_code": ["000001.SZ"], "raw_close": [10.5]}
    )
    result = prepare_snapshot_prices(
        historical, current, pd.DatetimeIndex([date])
    )
    assert result.loc[0, "raw_close"] == 10.5


def test_active_industry_uses_half_open_interval() -> None:
    intervals = pd.DataFrame(
        {
            "con_code": ["A", "A"],
            "industry_l1": ["旧行业", "新行业"],
            "in_date": pd.to_datetime(["2019-01-01", "2020-01-01"]),
            "out_date": pd.to_datetime(["2020-01-01", None]),
        }
    )
    result = active_industry_for_date(intervals, pd.Timestamp("2020-01-01"))
    assert result.to_dict("records") == [
        {"con_code": "A", "industry_l1": "新行业"}
    ]


def test_design_matrix_audit_does_not_fit_residuals() -> None:
    frame = pd.DataFrame(
        {
            "industry_l1": ["A", "A", "B", "B", "C", "C"],
            "ttm_roe": [0.1, 0.2, 0.0, 0.3, -0.1, 0.4],
            "ttm_revenue_growth_yoy": [0.0, 0.1, 0.2, -0.1, 0.3, 0.4],
        }
    )
    audit = design_matrix_audit(frame)
    assert audit["design_row_count"] == 6
    assert audit["design_column_count"] == 5
    assert audit["design_rank"] == 5
    assert audit["design_residual_degrees_of_freedom"] == 1


def test_full_synthetic_snapshot_passes_all_gates() -> None:
    rows = 300
    date = pd.Timestamp("2024-12-31")
    panel = pd.DataFrame(
        {
            "con_code": [f"S{i:03d}" for i in range(rows)],
            "snapshot_weight": np.repeat(1.0 / rows, rows),
            "raw_close": np.linspace(5.0, 20.0, rows),
            "industry_l1": [f"行业{i % 25:02d}" for i in range(rows)],
            "book_to_price": np.linspace(0.2, 1.2, rows),
            "ttm_roe": np.linspace(-0.2, 0.5, rows),
            "ttm_revenue_growth_yoy": np.sin(np.arange(rows)) / 5,
            "selected_available_at": pd.Timestamp("2024-10-31"),
        }
    )
    result = audit_snapshot(panel, date, _config())
    assert result["output"] == "DATA_READY"
    assert result["failure_category"] == "PASS"
    assert result["design_residual_degrees_of_freedom"] >= 150


def test_future_financial_event_forces_no_view() -> None:
    rows = 300
    date = pd.Timestamp("2024-12-31")
    panel = pd.DataFrame(
        {
            "con_code": [f"S{i:03d}" for i in range(rows)],
            "snapshot_weight": np.repeat(1.0 / rows, rows),
            "raw_close": 10.0,
            "industry_l1": [f"行业{i % 25:02d}" for i in range(rows)],
            "book_to_price": 0.5,
            "ttm_roe": np.linspace(-0.2, 0.5, rows),
            "ttm_revenue_growth_yoy": np.sin(np.arange(rows)) / 5,
            "selected_available_at": pd.Timestamp("2025-01-01"),
        }
    )
    result = audit_snapshot(panel, date, _config())
    assert result["output"] == "NO_VIEW"
    assert "FUTURE_FINANCIAL_EVENT" in result["failure_category"]
