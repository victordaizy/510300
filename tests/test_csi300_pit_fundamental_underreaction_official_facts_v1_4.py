from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path

import yaml

from research import csi300_pit_fundamental_underreaction_official_facts_v1_4 as parser


def _empty_result() -> dict[str, object]:
    return {
        "metrics": [],
        "missing_metrics": list(parser.REQUIRED_METRICS),
        "document_complete": False,
    }


def test_parallel_consolidated_and_company_columns_validate_statement_section() -> None:
    pages = (
        (
            "资产负债表 单位：千元 项目 期末 期初 合并数 公司数 合并数 公司数 "
            "流动资产 货币资金 100 50 90 45 应收账款 20 2 18 1 "
            "存货 30 3 28 2 资产总计 500 300 450 280"
        ),
        (
            "流动负债 负债合计 200 100 190 90 股东权益合计 300 200 260 190 "
            "负债和股东权益总计 500 300 450 280"
        ),
    )
    section = parser.SectionRange(
        name="CONSOLIDATED_BALANCE_SHEET",
        page_indices=(0, 1),
        unit="千元",
    )
    assert parser._section_has_consolidated_statement_content(
        pages,
        section,
        headings=("合并资产负债表", "合并及公司资产负债表"),
        marker_groups=(("流动资产", "货币资金", "资产总计"),),
    )


def test_balance_details_are_independent_and_totals_need_only_core_identity() -> None:
    pages = (
        (
            "资产负债表 单位：千元 合并数 公司数 合并数 公司数 流动资产 "
            "应收账款 100 10 90 9 存货 200 20 180 18 资产总计 1,000 700 900 650"
        ),
        "负债合计 600 400 550 360 股东权益合计 400 300 350 290",
    )
    section = parser.SectionRange(
        name="CONSOLIDATED_BALANCE_SHEET",
        page_indices=(0, 1),
        unit="千元",
    )
    result = _empty_result()
    parser._add_logical_balance_overrides(result, pages, section)
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert metrics["ACCOUNTS_RECEIVABLE_END"]["metric_value_cny"] == 100_000.0
    assert metrics["INVENTORY_END"]["metric_value_cny"] == 200_000.0
    assert metrics["TOTAL_ASSETS_END"]["metric_value_cny"] == 1_000_000.0
    assert metrics["TOTAL_LIABILITIES_END"]["metric_value_cny"] == 600_000.0


def test_total_equity_excludes_parent_common_shareholder_subtotal() -> None:
    pages = (
        "合并资产负债表 单位：千元 流动资产 货币资金 100 应收账款 20 存货 30 资产总计 500",
        (
            "负债合计 200 归属于母公司普通股股东权益合计 250 "
            "其他权益工具 50 少数股东权益 0 股东权益合计 300 "
            "负债和股东权益总计 500"
        ),
    )
    section = parser.SectionRange(
        name="CONSOLIDATED_BALANCE_SHEET",
        page_indices=(0, 1),
        unit="千元",
    )
    result = _empty_result()
    parser._add_logical_balance_overrides(result, pages, section)
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert metrics["TOTAL_LIABILITIES_END"]["metric_value_cny"] == 200_000.0


def test_q3_inline_unit_summary_selects_second_nonpercentage_ytd_amount() -> None:
    pages = (
        (
            "主要财务数据 项目 2018年7-9月 本报告期同比 2018年1-9月 年初至报告期末同比\n"
            "归属于上市公司普通股股东的扣除非经常性损益的净利润\n"
            "（千元人民币）\n"
            "120,448 130.95% (2,258,756) (244.58%)"
        ),
    )
    candidate = parser._summary_text_candidate(
        pages,
        (0,),
        (
            r"归属于上市公司(?:普通股)?股东的扣除非经常性损益的净利润",
        ),
        period_type="Q3",
        metric_id="CORE_PARENT_NET_PROFIT_YTD",
    )
    assert candidate is not None
    assert candidate.raw_value == "(2,258,756)"
    assert candidate.value_cny == Decimal("-2258756000")
    assert candidate.selected_amount_index == 1


def test_specific_operating_revenue_precedes_operating_total_revenue() -> None:
    pages = (
        (
            "合并利润表 单位：元 2015年度 2014年度 "
            "一、营业总收入 76,033,142,505.96 62,599,104,189.86 "
            "其中：营业收入 75,954,585,964.64 62,590,772,604.67"
        ),
    )
    candidate = parser._first_logical_pattern_candidate(
        pages,
        (0,),
        (
            r"其中[:：]?营业收入",
            r"(?:一[、.]?)?营业收入",
            r"(?:一[、.]?)?营业总收入",
        ),
        unit="元",
    )
    assert candidate is not None
    assert candidate.raw_value == "75,954,585,964.64"


def test_parenthesized_operating_loss_label_reconciles_to_profit_total() -> None:
    pages = (
        (
            "合并及公司利润表 单位：千元 合并 公司 合并 公司 "
            "一、营业收入 10,000 5,000 9,000 4,500 "
            "二、营业(亏损)/利润 (610,021) 130,065 (1,458,933) (721,597) "
            "加：营业外收入 1,665,224 22,547 2,652,150 33,247 "
            "减：营业外支出 166,246 16,035 220,604 23,432 "
            "三、利润总额 888,957 136,577 972,613 (711,782) "
            "四、净利润 757,732 329,667 824,038 (533,486) "
            "归属于本公司股东的净利润 872,504 329,667 866,915 (533,486) "
            "少数股东损益 (114,772) 0 (42,877) 0"
        ),
    )
    section = parser.SectionRange(
        name="CONSOLIDATED_INCOME_STATEMENT",
        page_indices=(0,),
        unit="千元",
    )
    result = _empty_result()
    parser._add_logical_income_overrides(
        result,
        pages,
        section,
        {},
        period_type="FY",
    )
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert metrics["OPERATING_PROFIT_YTD"]["metric_value_cny"] == -610_021_000.0


def test_v1_3_fallback_fills_only_when_entire_legacy_document_is_complete() -> None:
    pdf_sha256 = "a" * 64
    result = _empty_result()
    result["metrics"] = [
        {
            "metric_id": "OPERATING_REVENUE_YTD",
            "metric_value_cny": 101.0,
            "source_locator": "{}",
        }
    ]
    parser._refresh_result(result)
    legacy_metrics = [
        {
            "metric_id": metric_id,
            "metric_value_cny": float(index + 1),
            "source_locator": "{}",
            "verification_status": "PASS",
        }
        for index, metric_id in enumerate(parser.REQUIRED_METRICS)
    ]
    legacy = {
        "parser_version": "LEGACY_TEST",
        "official_pdf_sha256": pdf_sha256,
        "document_complete": True,
        "missing_metrics": [],
        "metrics": legacy_metrics,
    }
    receipt = parser._merge_complete_v1_3_fallback(
        result,
        legacy,
        pdf_sha256=pdf_sha256,
    )
    assert result["document_complete"] is True
    assert len(receipt["filled_metric_ids"]) == len(parser.REQUIRED_METRICS) - 1
    revenue = next(
        row for row in result["metrics"] if row["metric_id"] == "OPERATING_REVENUE_YTD"
    )
    assert revenue["metric_value_cny"] == 101.0

    rejected_result = _empty_result()
    legacy["document_complete"] = False
    legacy["missing_metrics"] = ["TOTAL_LIABILITIES_END"]
    legacy["metrics"] = legacy_metrics[:-1]
    rejected = parser._merge_complete_v1_3_fallback(
        rejected_result,
        legacy,
        pdf_sha256=pdf_sha256,
    )
    assert rejected["status"] == "REJECTED_V1_3_FALLBACK_DOCUMENT_NOT_COMPLETE"
    assert rejected_result["metrics"] == []


def test_manual_appendix_is_hash_bound_and_covers_exactly_nine_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (
            root
            / "config/csi300_pit_fundamental_underreaction_official_facts_v1_4_manual_appendix.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == "1.0"
    assert len(payload["documents"]) == 4
    for pdf_sha256, document in payload["documents"].items():
        assert len(pdf_sha256) == 64
        assert document["official_pdf_sha256"] == pdf_sha256
        assert int(document["official_pdf_size_bytes"]) > 0
        assert {fact["metric_id"] for fact in document["facts"]} == set(
            parser.REQUIRED_METRICS
        )


def test_v1_4_config_and_collector_use_isolated_checkpoint_root() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = (
        root / "config/csi300_pit_fundamental_underreaction_official_facts_v1_4.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["extraction"]["parser_version"] == parser.PARSER_VERSION
    assert config["artifacts"]["checkpoint_root"].endswith("/checkpoints_v1_4")
    assert config["artifacts"]["checkpoint_root"] != config["supersedes"][
        "legacy_checkpoint_root"
    ]
    from scripts import (
        collect_csi300_pit_fundamental_underreaction_official_facts_v1_4 as collector,
    )

    collector._patch_base_collector()
    assert collector._base.PARSER_VERSION == parser.PARSER_VERSION
    assert collector._base.CONFIG_PATH == collector.CONFIG_PATH
    assert collector._base.process_document is collector.process_document
