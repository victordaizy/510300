"""七条原PDF金额反例及完整数值、脚注、期间和跨页单位边界。"""
from functools import lru_cache
from importlib import import_module
import json
import os
from pathlib import Path
from decimal import Decimal

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_financial_parser_repair_v1"
CASES = json.loads((OUT / "inputs/confirmed_contradictions.json").read_text(encoding="utf-8")) if OUT.exists() else json.loads(
    (ROOT / "reports/research/510300_factor96_earnings_cashflow_measurement_v1/anomaly_evidence/confirmed_field_contradictions.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def extracted(announcement_id):
    parser = import_module(os.environ.get("FACTOR96_FINANCIAL_PARSER", "research.csi300_pit_fundamental_underreaction_official_facts_v1_6"))
    item = next(c for c in CASES if c["announcement_id"] == announcement_id)
    q = pd.Timestamp(item["report_period"]).quarter
    result = parser.extract_official_pdf_facts((OUT / "inputs/pdf" / (announcement_id + ".pdf")).read_bytes(),
                                              period_type={1: "Q1", 2: "H1", 3: "Q3", 4: "FY"}[q])
    return {r["metric_id"]: r for r in result["metrics"]}


@pytest.mark.parametrize("case", CASES, ids=[c["announcement_id"] + "_" + c["metric_id"] for c in CASES])
def test_seven_original_pdf_amounts(case):
    row = extracted(case["announcement_id"]).get(case["metric_id"])
    assert row is not None, "原PDF有明确金额，当前解析结果缺失"
    assert row["metric_value_cny"] == pytest.approx(case["original_page_value_cny"], abs=.011, rel=0), (
        case["report_title"], case["error_kind"], row["metric_value_cny"], case["original_page_value_cny"])


def test_complete_comma_amount_and_blank_are_distinct():
    from research.factor96_financial_row_parser_v1 import parse_amount_cell
    assert parse_amount_cell("40,\n895,706,832.37")[0] == Decimal("40895706832.37")
    assert parse_amount_cell("40,") is None
    assert parse_amount_cell("40,895,706,832.37 31.69%") is None
    assert parse_amount_cell("—") is None
    assert parse_amount_cell("0")[0] == 0
    assert parse_amount_cell("2")[0] == 2  # 真实两元金额不能仅因较小而被删除。


def test_footnote_and_amount_cannot_form_one_numeric_cell():
    from research.factor96_financial_row_parser_v1 import parse_amount_cell
    assert parse_amount_cell("注2 764,757,750.06") is None
    assert parse_amount_cell("2 764,757,750.06") is None
    assert parse_amount_cell("(602,204,358.37)")[0] == Decimal("-602204358.37")


def test_q3_uses_ytd_not_first_quarter_column():
    from research.factor96_financial_row_parser_v1 import header_is_current
    assert not header_is_current("本报告期", "PARENT_NET_PROFIT_YTD", "Q3", 2024, False)
    assert header_is_current("年初至报告期末", "PARENT_NET_PROFIT_YTD", "Q3", 2024, False)
    assert not header_is_current("年初至报告期末比上年同期增减", "PARENT_NET_PROFIT_YTD", "Q3", 2024, False)
    assert not header_is_current("2023年1-9月", "PARENT_NET_PROFIT_YTD", "Q3", 2024, True)
    assert header_is_current("本期发生额", "OPERATING_CASH_FLOW_YTD", "Q3", 2024, True)


def test_fy_quarterly_breakdown_is_not_annual_profit():
    from research.factor96_financial_row_parser_v1 import header_is_current
    assert not header_is_current("2024年第一季度", "PARENT_NET_PROFIT_YTD", "FY", 2024, False)
    assert header_is_current("2024年", "PARENT_NET_PROFIT_YTD", "FY", 2024, False)


def test_cross_page_unit_carries_only_until_new_section():
    from research.factor96_financial_row_parser_v1 import Context
    c = Context()
    c.advance("主要财务数据 单位：百万元", 1)
    c.advance("2024年第一季度报告", 2)
    assert c.section == "SUMMARY" and c.unit == "百万元" and c.unit_page == 1
    c.advance("非经常性损益项目 单位：元", 2)
    assert c.section == "EXCLUDED" and c.unit == "元"
    c.advance("母公司利润表 单位：元", 3)
    assert c.section == "EXCLUDED"
    c.advance("合并利润表 2024年1-9月 单位：元", 4)
    assert c.section == "INCOME" and c.ytd_explicit


def test_rounding_agreement_and_true_conflict():
    from research.factor96_financial_row_parser_v1 import select_consistent_candidates
    precise = {"value_cny": Decimal("76677224967"), "resolution_cny": Decimal(1), "section": "BALANCE", "source_page": 7}
    rounded = {"value_cny": Decimal("76677200000"), "resolution_cny": Decimal("100000"), "section": "SUMMARY", "source_page": 2}
    assert select_consistent_candidates([rounded, precise])[0] is precise
    wrong = {**rounded, "value_cny": Decimal("76677.2"), "resolution_cny": Decimal("0.1")}
    assert select_consistent_candidates([wrong, precise])[0] is None
