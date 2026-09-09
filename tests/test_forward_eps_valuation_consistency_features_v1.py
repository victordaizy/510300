"""检验来源关系条件不会用错误价格、单年度或重复机构制造有效值。"""
import numpy as np
import pandas as pd
import pytest

from research.forward_eps_valuation_consistency_features_v1 import qualify_report, aggregate_qualified, QUALIFIED


def facts(eps, pe):
    return [{"target_fiscal_year": 2024 + i, "source_eps_cell": e, "pe_value_exact": p}
            for i, (e, p) in enumerate(zip(eps, pe))]


def test_integer_pe_display_precision_can_admit_report():
    result = qualify_report(facts(["1.83", "2.00"], ["11", "10"]), "19.47")
    assert result["eligible"]
    assert not qualify_report(facts(["1.83", "2.00"], ["11.00", "10.00"]), "19.47")["eligible"]


def test_actual_source_inconsistency_disqualifies_whole_report():
    result = qualify_report(facts(["1.43", "1.52", "1.59"], ["9.2", "8.6", "7.5"]), "11.97")
    assert not result["eligible"]
    assert result["annual_relations"][-1]["status"] == "COMPATIBLE_WITH_REPORTED_DISPLAY_PRECISION"


def test_internally_common_price_does_not_override_reference():
    result = qualify_report(facts(["1.31", "1.64", "1.95"], ["11.29", "9.02", "7.56"]), "20.99")
    assert not result["eligible"]


@pytest.mark.parametrize("quote,pe", [(None, ["10", "10"]), ("0", ["10", "10"]),
                                      ("10.00", ["10.00", None]), ("10.00", ["10.00", "0"])])
def test_missing_single_year_zero_quote_and_zero_pe_not_admitted(quote, pe):
    assert not qualify_report(facts(["1.00", "1.00"], pe), quote)["eligible"]


def test_negative_eps_and_negative_pe_do_not_become_missing():
    assert qualify_report(facts(["-1.00", "-2.00"], ["-10.00", "-5.00"]), "10.00")["eligible"]


def test_company_weight_and_missing_preserved():
    date = pd.Timestamp("2024-08-30")
    institution = pd.DataFrame({"origin": [date] * 4, "ts_code": ["A", "A", "B", "B"],
                               "institution": ["guosen", "soochow"] * 2, QUALIFIED: [.1, np.nan, .2, .4]})
    company = pd.DataFrame({"origin": [date] * 2, "ts_code": ["A", "B"], "eps_growth": [.5, .6]})
    monthly = pd.DataFrame({"origin": [date], "pooled_eps_growth_company_count": [30],
                           "pooled_profit_revision_company_count": [15], "pooled_eps_growth_median": [.5],
                           "pooled_profit_revision_median": [.1], "pooled_reported_earnings_yield_median": [.2]})
    pooled, result = aggregate_qualified(institution, company, monthly)
    np.testing.assert_allclose(pooled[QUALIFIED], [.1, .3])
    pd.testing.assert_frame_equal(pooled[company.columns], company)
    assert result.qualified_pe_company_count.iloc[0] == 2
    assert not result.common_quality_features_valid.iloc[0]
    with pytest.raises(ValueError, match="重复"):
        aggregate_qualified(pd.concat([institution, institution.iloc[[0]]]), company, monthly)
