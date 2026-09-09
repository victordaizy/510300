"""针对真实财报主体、期间、归母利润和普通股收益错误的必要验证。"""
import json
from copy import deepcopy
from decimal import Decimal

import pytest

from research.financial_original_facts_v1_1 import (
    INVENTORY, REPAIR, get_row, header_and_layout, identity,
    ordinary_eps_diagnostic, parse_document, statement_lines,
)


def document(aid):
    base = REPAIR if aid in {"1218186515", "1221451424"} else INVENTORY
    return json.loads((base/"page_texts"/(aid+".json")).read_text("utf-8"))


def parsed(aid):
    doc = document(aid)
    result = parse_document(doc["pages"], doc["source"])
    assert result["facts"], result
    return result


def values(aid):
    return {r["metric_id"]: r["metric_value"] for r in parsed(aid)["facts"]}


def test_subsidiary_attachment_is_rejected_before_financial_numbers():
    doc = document("1218137429")
    out = parse_document(doc["pages"], doc["source"])
    assert out["status"] == "NO_VIEW_WRONG_SUBJECT_CODE_OR_PERIOD"
    assert out["facts"] == []


def test_empty_graphic_cover_uses_explicit_second_title_page():
    doc = document("1211419143")
    out = identity(doc["pages"], doc["source"])
    assert out["status"].startswith("PASS_") and out["identity_page"] == 2


def test_cmb_cumulative_consolidated_profit_preserves_original_verified_value():
    out = values("1211361859")
    assert out["OPERATING_PROFIT_YTD"] == 116581000000
    assert out["OPERATING_REVENUE_YTD"] == 251410000000
    assert out["BASIC_EPS_YTD"] == 3.62


@pytest.mark.parametrize("aid,year,col,expected", [
    ("1204089069", 2017, 1, 258957000000),
    ("1211420908", 2021, 2, 544897000000),
    ("1204089363", 2017, 2, 149716000000),
])
def test_different_annual_column_order_is_read_from_actual_headers(aid, year, col, expected):
    doc = document(aid)
    layout = header_and_layout(statement_lines(doc["pages"], year), year)
    assert layout["column"] == col
    assert values(aid)["OPERATING_REVENUE_YTD"] == expected


def test_parent_company_statement_does_not_extend_group_section():
    doc = document("1218135621")
    lines = statement_lines(doc["pages"], 2023)
    assert "126,186" not in "\n".join(r["text"] for r in lines)
    assert get_row(lines, ["营业收入"], [4])["source_page"] == 35


def test_pab_impairment_and_other_equity_returns_are_not_ordinary_eps():
    doc = document("1218135621")
    out = parsed("1218135621")
    vals = {r["metric_id"]: r["metric_value"] for r in out["facts"]}
    assert vals["OPERATING_PROFIT_YTD"] == 49047000000
    diag = ordinary_eps_diagnostic(doc["pages"], doc["source"], out)
    assert Decimal(diag["ordinary_profit_after_other_equity_returns_cny"]) == 37606000000
    assert Decimal(diag["corrected_ordinary_profit_divided_by_latest_shares"]).quantize(Decimal(".01")) == Decimal("1.94")
    assert not diag["latest_share_count_is_proven_ytd_weighted_average_share_count"]


def test_group_company_columns_and_two_cell_parent_ownership_rows():
    out = values("1214937073")
    assert out["OPERATING_REVENUE_YTD"] == 701012000000
    assert out["PARENT_NET_PROFIT_YTD"] == 31117000000
    assert out["BASIC_EPS_YTD"] == 1.10


def test_explicit_group_q1_summary_keeps_missing_operating_profit_missing():
    out = parsed("1209861041")
    assert len(out["facts"]) == 3
    assert out["missing_core_metrics"] == ["OPERATING_PROFIT_YTD"]
    assert values("1209861041")["PARENT_NET_PROFIT_YTD"] == 83115000000


def test_unclear_currency_is_not_changed_to_cny():
    out = parsed("1214963358")
    assert all(r["strategy_input_status"] == "NO_VIEW_CURRENCY_NOT_EXPLICIT" for r in out["facts"])


def test_conflicting_duplicate_rows_are_rejected():
    rows = [{"page": 1, "text": "营业收入 10 20"}, {"page": 2, "text": "营业收入 11 20"}]
    with pytest.raises(ValueError, match="数值冲突"):
        get_row(rows, ["营业收入"], [2])


def test_wrong_report_year_cannot_supply_current_year_column():
    doc = document("1211420908")
    with pytest.raises(ValueError):
        header_and_layout(statement_lines(doc["pages"], 2021), 2022)


def test_restated_comparison_is_retained_only_as_later_vintage_cells():
    old = values("1211419194")
    later = parsed("1214937073")
    revenue = next(r for r in later["facts"] if r["metric_id"] == "OPERATING_REVENUE_YTD")
    assert old["OPERATING_REVENUE_YTD"] == 727711000000
    assert revenue["all_numeric_cells_for_verification"][1] == "727,785"
    assert not revenue["previous_year_comparative_is_original_prior_vintage"]


@pytest.mark.parametrize("aid,revenue,parent", [
    ("1204088714", 9524489003.64, 4171268021.94),
    ("1218186515", 704938000000, 87575000000),
    ("1221451424", 775383000000, 119182000000),
])
def test_explicit_owner_synonym_and_no_item_heading(aid, revenue, parent):
    out = values(aid)
    assert out["OPERATING_REVENUE_YTD"] == revenue
    assert out["PARENT_NET_PROFIT_YTD"] == parent


def test_every_original_v1_fact_keeps_its_value_scope_column_and_source():
    from research.financial_original_facts_v1 import OUT as OLD_OUT
    for f in (OLD_OUT/"document_records").glob("*.json"):
        old = json.loads(f.read_text("utf-8"))
        if not old["facts"]:
            continue
        current = parsed(f.stem)
        assert current["facts"] == old["facts"]
