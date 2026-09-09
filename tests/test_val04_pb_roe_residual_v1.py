from __future__ import annotations

import numpy as np
import pandas as pd

from research.normalized_valuation_5y_models_v1 import apply_frozen_percentile
from research.val04_pb_roe_residual_v1 import fit_snapshot_residuals


def _synthetic_frame() -> pd.DataFrame:
    rows = 60
    sectors = np.array(["银行", "电子", "医药"] * 20)
    roe = np.linspace(-0.1, 0.4, rows)
    growth = np.sin(np.arange(rows)) / 10
    sector_intercept = {"银行": -0.2, "电子": 0.6, "医药": 0.3}
    log_pb = np.array([sector_intercept[value] for value in sectors]) + 1.2 * roe - 0.5 * growth
    return pd.DataFrame(
        {
            "date": pd.Timestamp("2024-12-31"),
            "con_code": [f"S{i:03d}" for i in range(rows)],
            "snapshot_weight": np.repeat(1.0 / rows, rows),
            "industry_l1": sectors,
            "log_price_to_book": log_pb,
            "ttm_roe": roe,
            "ttm_revenue_growth_yoy": growth,
            "complete_case": True,
        }
    )


def test_exact_linear_cross_section_has_zero_residual() -> None:
    companies, audit = fit_snapshot_residuals(_synthetic_frame())
    assert np.allclose(companies["pb_roe_growth_residual"], 0.0, atol=1e-12)
    assert abs(audit["roe_coefficient"] - 1.2) < 1e-12
    assert abs(audit["revenue_growth_coefficient"] + 0.5) < 1e-12
    assert abs(audit["weighted_pb_roe_residual_cheapness"]) < 1e-12


def test_negative_residual_is_positive_cheapness() -> None:
    frame = _synthetic_frame()
    frame.loc[0, "log_price_to_book"] -= 0.5
    companies, _ = fit_snapshot_residuals(frame)
    row = companies.loc[companies["con_code"].eq("S000")].iloc[0]
    assert row["pb_roe_growth_residual"] < 0
    assert row["company_residual_cheapness"] > 0


def test_frozen_percentile_uses_only_trailing_sixty_including_current() -> None:
    panel = pd.DataFrame(
        {
            "value": np.arange(61, dtype=float),
            "status": "PASS",
        }
    )
    result = apply_frozen_percentile(panel, "value", "status", "pct", 60)
    assert result.loc[:58, "pct"].isna().all()
    assert result.loc[59, "pct"] == (59.5 / 60.0)
    assert result.loc[60, "pct"] == (59.5 / 60.0)


def test_incomplete_rows_are_excluded_before_fit() -> None:
    frame = _synthetic_frame()
    frame.loc[0, "complete_case"] = False
    companies, audit = fit_snapshot_residuals(frame)
    assert "S000" not in set(companies["con_code"])
    assert audit["complete_company_count"] == 59
