from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import yaml

from research import csi300_pit_fundamental_underreaction_official_facts_v1_3 as parser


def test_multiple_note_references_are_skipped_before_current_amount() -> None:
    text = (
        "合并利润表 单位：元 项目 2021年度合并 2020年度合并 "
        "一、营业收入 (39),十五(12) 30,166,805,377 27,759,710,926"
    )
    candidate = parser._find_candidate(
        (text,),
        (0,),
        (r"一[、.]营业收入",),
        unit="元",
    )
    assert candidate is not None
    assert candidate.raw_value == "30,166,805,377"
    assert candidate.value_cny == Decimal("30166805377")


def test_q3_income_selection_keeps_continuation_pages_and_stops_at_parent() -> None:
    pages = [
        "合并利润表 2020年第三季度（7-9月） 2020年前三季度（1-9月） 单位：元",
        (
            "归属于母公司股东的净利润（净亏损以\"-\"号填 "
            "882,436,711.81 275,381,721.80 2,141,280,033.46 1,018,557,167.44"
        ),
        "列） 少数股东损益 110,572,841.55 -7,059,465.96 453,725,959.97 11,807,699.71",
        "母公司利润表 2020年第三季度（7-9月） 2020年前三季度（1-9月） 单位：元",
    ]
    section = parser.SectionRange(
        name="CONSOLIDATED_INCOME_STATEMENT",
        page_indices=(0, 1, 2, 3),
        unit="元",
    )
    selected, amount_index = parser._income_selection(pages, section, "Q3")
    assert selected == (0, 1, 2)
    assert amount_index == 2
    candidate = parser._find_candidate(
        pages,
        selected,
        (r"归属于母公司(?:所有者|股东)的净利润\s*[（(]净亏损以[^）)]{0,40}号填(?:列)?[）)]?",),
        unit="元",
        amount_index=amount_index,
    )
    assert candidate is not None
    assert candidate.value_cny == Decimal("2141280033.46")


def test_content_classified_balance_requires_and_uses_accounting_identity() -> None:
    pages = [
        (
            "2021年度合并及公司利润表 单位：元 合并 公司 附注 流动资产 货币资金 100 90 "
            "应收账款 (4),十五(1) 20,000 18,000 存货 (7),十五(3) 30,000 28,000 "
            "其他项目 1 2 3 4 5 6 7 8 9 10 11 12 资产总计 500,000 450,000"
        ),
        (
            "流动负债 100 90 非流动负债 100 90 其他项目 1 2 3 4 5 6 7 8 9 10 11 12 "
            "负债合计 200,000 190,000 股东权益合计 300,000 260,000 "
            "负债及股东权益总计 500,000 450,000"
        ),
    ]
    section = parser._content_classified_balance_pair(pages)
    assert section.page_indices == (0, 1)
    evidence = parser._validated_balance_metrics(pages, section, text_engine="PDFIUM")
    assert evidence["ACCOUNTS_RECEIVABLE_END"].metric_value_cny == 20_000.0
    assert evidence["INVENTORY_END"].metric_value_cny == 30_000.0
    assert evidence["TOTAL_ASSETS_END"].metric_value_cny == 500_000.0
    assert evidence["TOTAL_LIABILITIES_END"].metric_value_cny == 200_000.0


def test_combined_receivable_note_requires_current_and_prior_reconciliation() -> None:
    pages = [
        "合并资产负债表 单位：元 流动资产 货币资金 100 应收账款 20 存货 30 资产总计 500",
        "合并利润表 单位：元 营业收入 300 营业成本 180 营业利润 60",
        "合并现金流量表 单位：元 经营活动产生的现金流量 销售商品 350 经营活动产生的现金流量净额 45",
        (
            "合并财务报表主要项目注释 3.应收票据及应收账款 项目 年末余额 年初余额 "
            "应收票据 164,348,594.49 117,617,801.49 "
            "应收账款 8,146,589,520.29 7,150,244,020.49 "
            "合计 8,310,938,114.78 7,267,861,821.98"
        ),
        "母公司财务报表主要项目注释",
    ]
    result = {"metrics": [], "missing_metrics": list(parser.REQUIRED_METRICS)}
    parser._add_reconciled_combined_receivable_note(
        result,
        pages,
        text_engine="PDFIUM",
    )
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert metrics["ACCOUNTS_RECEIVABLE_END"]["metric_value_cny"] == 8_146_589_520.29


def test_explicit_blank_accounts_receivable_table_row_is_zero() -> None:
    rows = [(0, 0, 3, ["应收账款", "", ""])]
    evidence = parser._blank_accounts_receivable_evidence(
        ("合并资产负债表 单位：元",),
        rows,
        fallback_unit="元",
    )
    assert evidence is not None
    assert evidence.metric_value_cny == 0.0
    assert "EXACT_ACCOUNTS_RECEIVABLE_LABEL" in evidence.source_locator


def test_ocr_number_normalization_and_dual_scale_agreement_are_exact() -> None:
    assert parser._normalize_ocr_number("108,769,214)") == Decimal("-108769214")
    assert parser._normalize_ocr_number("巧,875,475") == Decimal("15875475")
    first = parser.OcrObservation(
        key="TOTAL_ASSETS_END",
        page_index=97,
        render_scale=2.25,
        raw_value="3,198,004,370",
        value=Decimal("3198004370"),
        row_text="资产总计3,198,004,370",
        pixel_width=100,
        pixel_height=200,
    )
    second = parser.OcrObservation(
        key="TOTAL_ASSETS_END",
        page_index=97,
        render_scale=3.0,
        raw_value="3,198,004,370",
        value=Decimal("3198004370"),
        row_text="资产总计3,198,004,370",
        pixel_width=120,
        pixel_height=240,
    )
    by_scale = {
        2.25: {"TOTAL_ASSETS_END": [first]},
        3.0: {"TOTAL_ASSETS_END": [second]},
    }
    assert parser._agreed_ocr_observation(by_scale, "TOTAL_ASSETS_END") == (first, second)


def test_image_statement_unit_uses_bounded_nearby_financial_statement_region() -> None:
    pages = [""] * 140
    pages[60] = "单位：亿元 币种：人民币"
    pages[113] = "除有特别说明外，均以人民币千元为单位表示"
    unit, page = parser._find_image_statement_unit(pages, tuple(range(95, 104)))
    assert unit == "千元"
    assert page == 114


def test_v1_3_config_collector_and_ocr_script_use_isolated_frozen_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "config/csi300_pit_fundamental_underreaction_official_facts_v1_3.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["extraction"]["parser_version"] == parser.PARSER_VERSION
    assert config["artifacts"]["checkpoint_root"].endswith("/checkpoints_v1_3")
    assert config["artifacts"]["checkpoint_root"] != config["supersedes"]["legacy_checkpoint_root"]
    assert (root / "scripts/windows_ocr_financial_statement_v1_3.ps1").exists()
    from scripts import (
        collect_csi300_pit_fundamental_underreaction_official_facts_v1_3 as collector,
    )

    collector._patch_base_collector()
    assert collector._base.PARSER_VERSION == parser.PARSER_VERSION
    assert collector._base.CONFIG_PATH == collector.CONFIG_PATH
    assert collector._base.process_document is collector.process_document
