"""针对真实年报混列、附注编号、股东归属和每股收益误读的必要回归验证。"""
from decimal import Decimal

import pytest

from research.financial_annual_components_v1 import parse_tail, process, row_matches, select_row


def line(text):
    return [{"page": 1, "raw": text, "text": text}]


def test_parenthesized_expense_is_negative_not_annotation():
    r = select_row(line("二、营业支出 (391,990) (360,963)"), "营业支出", [2])
    assert r["values"] == [Decimal(-391990), Decimal(-360963)]


def test_numeric_footnote_is_not_eps():
    r = select_row(line("基本每股收益(人民币元) 48 2.25 2.20"), "基本每股收益", [2])
    assert r["footnote_or_formula"] == "48"
    assert r["values"][0] == Decimal("2.25")


def test_formula_is_only_allowed_in_registered_formula_table():
    assert parse_tail("7=3÷6 0.86 1.25", [2], True) is None
    assert parse_tail("7=3÷6 0.86 1.25", [2], True, True)["values"] == [Decimal(".86"), Decimal("1.25")]


def test_dashes_preserved_in_mother_company_eps_columns():
    r = parse_tail("56 1.42 1.32 - -", [2, 4], True)
    assert r["values"] == [Decimal("1.42"), Decimal("1.32"), None, None]
    assert r["footnote_or_formula"] == "56"


def test_same_name_different_profit_requires_explicit_occurrence():
    ls = line("归属于本公司普通股股东的当年净利润 23,099,624,927.77 14,902,324,215.75") + line("归属于本公司普通股股东的当年净利润 22,941,989,311.35 14,902,324,215.75")
    with pytest.raises(ValueError, match="同名行有不同金额"):
        select_row(ls, "归属于本公司普通股股东的当年净利润", [2])


def test_labels_do_not_match_deducted_or_diluted_variants():
    assert not row_matches(line("扣除非经常性损益的基本每股收益 0.87 1.24"), "基本每股收益", [2])
    assert not row_matches(line("稀释每股收益 1.62 1.19"), "基本每股收益", [2])


@pytest.fixture(scope="module")
def records():
    return {aid: process(aid) for aid in ["1212709852", "1212751519", "1212730963", "1208663570", "1212745096", "1219376072", "1212669927", "1204547754", "1214964494", "1219306493"]}


def val(record, metric):
    return next(f["metric_value_exact"] for f in record["core_facts"]+record["additional_facts"] if f["metric_id"] == metric)


def test_citic_two_printed_pages_never_select_mother_profit(records):
    d = records["1212709852"]
    assert val(d, "OPERATING_REVENUE_YTD") == "76523716526.93"
    assert val(d, "PARENT_NET_PROFIT_YTD") == "23099624927.77"
    assert val(d, "ORDINARY_NET_PROFIT_YTD") == "22941989311.35"


def test_group_and_bank_mixed_columns(records):
    d = records["1212730963"]
    assert val(d, "PARENT_NET_PROFIT_YTD") == "302513000000"
    assert val(d, "BASIC_EPS_YTD") == "1.19"
    assert val(d, "ORDINARY_NET_PROFIT_YTD") == "297975000000"


def test_gf_parent_profit_has_two_cells_while_revenue_has_four(records):
    d = records["1212751519"]
    assert len(d["statement_rows"]["OPERATING_REVENUE_YTD"]["cells"]) == 4
    assert len(d["statement_rows"]["PARENT_NET_PROFIT_YTD"]["cells"]) == 2
    assert val(d, "ORDINARY_NET_PROFIT_YTD") == "10841453883.56"


def test_summary_ytd_not_quarter_or_percent(records):
    d = records["1208663570"]
    assert val(d, "PARENT_NET_PROFIT_YTD") == "165335000000"
    assert val(d, "BASIC_EPS_YTD") == "0.45"
    assert len(d["core_facts"]) == 3


def test_restricted_stock_dividend_and_repurchased_weighted_shares(records):
    d = records["1212745096"]
    assert val(d, "ORDINARY_NET_PROFIT_YTD") == "14567735310"
    assert val(d, "RESTRICTED_STOCK_DIVIDEND") == "43244320"
    assert val(d, "WEIGHTED_ORDINARY_SHARES") == "8819448820"


def test_employee_and_asset_management_share_deductions(records):
    d = records["1219376072"]
    assert val(d, "WEIGHTED_ORDINARY_SHARES") == "17717000000"
    assert val(d, "BASIC_EPS_YTD") == "4.84"
    assert val(d, "CONSOLIDATED_PRODUCT_WEIGHTED_SHARES") == "-33000000"


@pytest.mark.parametrize("aid", ["1212669927", "1204547754"])
def test_unknown_share_multiplier_is_not_backsolved(records, aid):
    assert val(records[aid], "WEIGHTED_ORDINARY_SHARES") is None
    assert val(records[aid], "ORDINARY_NET_PROFIT_YTD") is not None


def test_original_prior_cms_profit_not_later_restatement(records):
    assert val(records["1214964494"], "PARENT_NET_PROFIT_YTD") == "6280444248.30"


def test_bank_impairment_not_counted_as_operating_profit(records):
    assert val(records["1219306493"], "OPERATING_PROFIT_YTD") == "57928000000"
    assert val(records["1219306493"], "ORDINARY_NET_PROFIT_YTD") == "43606000000"
