"""核对历史版本差异、百分比误读、同表分季和缺失状态。"""
from decimal import Decimal

import pytest

from research.financial_ttm_vintage_bridge_v1 import build


@pytest.fixture(scope="module")
def saved():
    return build()


def one(saved,name,metric="PARENT_NET_PROFIT_YTD"):
    return next(r for r in saved[0] if r["sec_name"]==name and r["metric_id"]==metric)


def test_cms_later_tax_comparative_differs_from_original_and_annual(saved):
    r=one(saved,"招商证券")
    assert Decimal(r["current_comparative_minus_original"])==Decimal("1914155.99")
    assert r["annual_vintage_minus_original"]=="0.00"
    assert r["quarter_bridge_status"].startswith("NO_VIEW_ANNUAL_QUARTERS_DIFFER")
    assert r["known_policy_evidence"]["kind"]=="DEFERRED_TAX_INTERPRETATION_16_TRANSITION"


def test_life_three_distinct_profit_vintages_are_preserved(saved):
    r=one(saved,"中国人寿")
    assert r["provenance"]["original_prior"]["value_exact"]=="48502000000"
    assert r["annual_vintage_prior_ytd"]=="48488000000"
    assert r["current_report_comparative_prior"]=="48486000000"
    assert not r["full_accounting_basis_reconciliation_completed"]


def test_pab_summary_growth_is_never_comparative_profit(saved):
    r=one(saved,"平安银行")
    assert r["current_report_comparative_prior"] is None
    assert r["comparison_status"]=="NO_VIEW_SUMMARY_ADJACENT_COLUMN_IS_PERCENT_NOT_PARENT_PROFIT"
    assert r["original_three_report_arithmetic"]=="46549000000"


def test_missing_currency_survives_exact_arithmetic(saved):
    r=one(saved,"广发证券")
    assert r["original_three_report_arithmetic"]=="7447563822.60"
    assert r["research_value_status"]=="NO_VIEW_CURRENCY_NOT_EXPLICIT"


def test_ccb_first_quarter_uses_first_of_eight_annual_columns(saved):
    r=one(saved,"建设银行")
    assert r["annual_vintage_prior_ytd"]=="83115000000"
    assert r["original_three_report_arithmetic"]=="308139000000"


def test_missing_operating_profit_is_not_zero(saved):
    for name in ["农业银行","建设银行"]:
        assert one(saved,name,"OPERATING_PROFIT_YTD")["original_three_report_arithmetic"] is None


def test_bocom_equal_parent_number_does_not_override_known_basis_change(saved):
    r=one(saved,"交通银行")
    assert r["current_comparative_minus_original"]=="0"
    assert r["research_value_status"]=="NO_VIEW_KNOWN_NEW_INSTRUMENT_RULES_WITHOUT_PRIOR_RESTATEMENT"


def test_eps_and_new_accounts_not_inferred_from_source_success(saved):
    assert saved[2]["new_basic_eps_ttm_values"]==0
    assert saved[2]["new_account_evaluations"]==0
    assert len(saved[0])==36
    assert all(not r["basic_eps_arithmetic_performed"] for r in saved[0])
