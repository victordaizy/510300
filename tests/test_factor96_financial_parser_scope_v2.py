"""白云山原年报揭示的独立报表串入与本年列头回归反例。"""
from importlib import import_module
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_financial_parser_repair_v1"


def selected_parser():
    return import_module(os.environ.get("FACTOR96_FINANCIAL_PARSER", "research.factor96_financial_row_parser_v1"))


def test_baiyunshan_real_annual_consolidated_fields():
    target = next(t for t in json.loads((OUT / "batch_targets.json").read_text(encoding="utf-8"))
                  if t["announcement_id"] == "1209410909")
    parsed = selected_parser().extract_official_pdf_facts((ROOT / target["pdf_relative_path"]).read_bytes(),
        period_type="FY", report_period=target["report_period"])
    values = {m["metric_id"]: m for m in parsed["metrics"]}
    for metric, expected in [("OPERATING_CASH_FLOW_YTD", 585185023.09),
                             ("PARENT_NET_PROFIT_YTD", 2915244576.05),
                             ("TOTAL_ASSETS_END", 59760062879.12)]:
        assert values.get(metric, {}).get("metric_value_cny") == pytest.approx(expected, abs=.011, rel=0)
    assert json.loads(values["OPERATING_CASH_FLOW_YTD"]["source_locator"])["section"] == "CASH"
    assert values["OPERATING_CASH_FLOW_YTD"]["source_page"] == 152


@pytest.mark.parametrize("title", ["资产负债表", "利润表", "现金流量表", "股东权益变动表", "所有者权益变动表"])
def test_unqualified_statement_ends_consolidated_scope(title):
    context = selected_parser().Context()
    context.advance("合并现金流量表\n2020年度\n单位：人民币元", 152)
    context.headers = [(300, 460, "本年")]
    context.advance(title + "\n2020年度\n单位：人民币元", 155)
    assert context.section == "EXCLUDED"
    assert context.headers is None


def test_current_year_column_is_explicit_and_previous_year_is_excluded():
    parser = selected_parser()
    assert parser.HEADER_RE.search("本年")
    assert parser.header_is_current("本年", "OPERATING_CASH_FLOW_YTD", "FY", 2020, False)
    assert not parser.header_is_current("上年", "OPERATING_CASH_FLOW_YTD", "FY", 2020, False)


def test_new_statement_heading_does_not_inherit_a_prior_table_header():
    context = selected_parser().Context()
    context.advance("合并现金流量表\n2020年度\n单位：元", 1)
    context.headers = [(300, 460, "2020年12月31日")]
    context.advance("合并现金流量表\n2020年度\n单位：元", 8)
    assert context.headers is None


def test_explicit_continuation_keeps_same_statement_unit():
    context = selected_parser().Context()
    context.advance("合并现金流量表\n2020年度\n单位：万元", 1)
    context.headers = [(300, 460, "本年")]
    context.advance("合并现金流量表（续）", 2)
    assert context.section == "CASH" and context.unit == "万元"
    assert context.headers == [(300, 460, "本年")]
