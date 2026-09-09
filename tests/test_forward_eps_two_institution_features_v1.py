"""防止机构样本数量加权、跨机构修正和缺失值填零。"""
import numpy as np
import pandas as pd
import pytest

from research.forward_eps_monthly_policy_v1 import company_features
from research.forward_eps_two_institution_features_v1 import aggregate_institutions


def test_company_equal_weight_after_within_company_mean():
    rows = pd.DataFrame([
        dict(ts_code="A", institution="guosen", eps_growth=0.2, profit_revision=0.1, reported_earnings_yield=0.05, report_age_days=10),
        dict(ts_code="A", institution="soochow", eps_growth=0.6, profit_revision=-0.1, reported_earnings_yield=0.15, report_age_days=30),
        dict(ts_code="B", institution="soochow", eps_growth=0.8, profit_revision=0.3, reported_earnings_yield=0.2, report_age_days=60),
    ])
    companies, result = aggregate_institutions(rows)
    assert result["pooled_eps_growth_median"] == pytest.approx(0.6)
    assert result["pooled_eps_growth_company_count"] == 2
    assert result["dual_institution_growth_share"] == 0.5
    assert result["pooled_report_age_mean_days"] == 40
    assert result["pooled_profit_revision_breadth"] == 0.5
    assert not result["common_sources_valid"]


def test_missing_institution_is_not_a_zero_forecast():
    rows = pd.DataFrame([
        dict(ts_code="A", institution="guosen", eps_growth=np.nan, profit_revision=np.nan, reported_earnings_yield=np.nan, report_age_days=np.nan),
        dict(ts_code="A", institution="soochow", eps_growth=0.8, profit_revision=np.nan, reported_earnings_yield=0.2, report_age_days=60),
    ])
    companies, result = aggregate_institutions(rows)
    assert companies.eps_growth.iloc[0] == 0.8
    assert np.isnan(result["pooled_profit_revision_median"])
    assert np.isnan(result["pooled_profit_revision_breadth"])
    assert result["dual_institution_growth_share"] == 0


def test_duplicate_institution_company_rejected():
    row = dict(ts_code="A", institution="soochow", eps_growth=0.1, profit_revision=0.1, reported_earnings_yield=0.1, report_age_days=30)
    with pytest.raises(ValueError):
        aggregate_institutions(pd.DataFrame([row, row]))


def test_missing_latest_report_blocks_stale_prior_view():
    old = {"report_id": "OLD", "information_date": "2024-01-01", "published_sequence": "2024-01-01 09:00:00",
           "facts": [{"target_fiscal_year": 2024, "eps_value_exact": "1", "pe_value_exact": "10"},
                     {"target_fiscal_year": 2025, "eps_value_exact": "2", "pe_value_exact": "5"}]}
    latest = {"report_id": "NEW", "information_date": "2024-03-01", "published_sequence": "2024-03-01 09:00:00", "facts": []}
    feature = company_features([old, latest], pd.Timestamp("2024-03-29"))
    assert np.isnan(feature["eps_growth"])
    assert feature["status"] == "NO_VIEW_LATEST_REPORT_MISSING_OR_STALE"
