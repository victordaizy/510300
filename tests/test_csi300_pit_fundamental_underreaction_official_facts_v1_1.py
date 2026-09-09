from __future__ import annotations

from pathlib import Path

import yaml

from research.csi300_pit_fundamental_underreaction_official_facts_v1_1 import (
    ADMITTED_VERIFICATION_STATUS,
    PARSER_VERSION,
    detect_unique_unit,
    extract_metrics_from_page_texts,
    find_amount,
    find_core_profit_from_summary_tables,
    locate_statement_sections,
)


def complete_pages() -> list[str]:
    return [
        """
        2020年年度报告
        主要会计数据和财务指标 单位：元
        归属于上市公司股东的扣除非经常性损益的净利润 800 700
        """,
        """
        合并资产负债表 人民币百万元
        流动资产： 货币资金 100 90
        应收账款 20 18
        存货 30 28
        资产总计 500 450
        负债合计 200 190
        """,
        """
        合并利润表 人民币百万元
        一、营业总收入 300 250
        营业成本 180 160
        三、营业利润 60 50
        1.归属于母公司股东的净利润 40 35
        """,
        """
        合并现金流量表 人民币百万元
        经营活动产生的现金流量：
        销售商品、提供劳务收到的现金 350 300
        经营活动产生的现金净额 45 38
        """,
    ]


def test_coherent_statement_triple_rejects_early_income_reference() -> None:
    pages = [
        "经营分析详见合并利润表",
        "营业收入 10 营业成本 8",
        *complete_pages()[1:],
    ]
    sections = locate_statement_sections(pages)
    assert sections["balance"].page_indices[0] == 2
    assert sections["income"].page_indices[0] == 3
    assert sections["cash"].page_indices[0] == 4


def test_repeated_statement_heading_keeps_first_page_with_inventory() -> None:
    pages = complete_pages()
    pages.insert(
        2,
        "合并资产负债表（续） 人民币百万元\n其他资产 10 9\n流动资产合计 200 180",
    )
    sections = locate_statement_sections(pages)
    assert sections["balance"].page_indices[0] == 1


def test_statement_heading_at_previous_page_tail_uses_next_page_markers() -> None:
    pages = [
        *complete_pages()[:3],
        "公司负责人：甲 主管会计工作负责人：乙 合并现金流量表",
        """
        2025年1—9月 编制单位：甲公司 人民币百万元 项目
        一、经营活动产生的现金流量：
        销售商品、提供劳务收到的现金 100 90
        经营活动产生的现金流量净额 20 18
        """,
    ]
    sections = locate_statement_sections(pages)
    assert sections["cash"].page_indices[0] == 3
    assert 4 in sections["cash"].page_indices


def test_total_revenue_old_cash_label_and_standalone_currency_unit() -> None:
    pages = complete_pages()
    assert detect_unique_unit(pages, [1]) == "百万元"
    result = extract_metrics_from_page_texts(pages, period_type="FY")
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert result["document_complete"] is True
    assert metrics["OPERATING_REVENUE_YTD"]["metric_value_cny"] == 300_000_000.0
    assert metrics["OPERATING_CASH_FLOW_YTD"]["metric_value_cny"] == 45_000_000.0


def test_q3_summary_table_selects_ytd_midpoint_with_bare_percentage_cells() -> None:
    pages = [
        "第三季度报告 本报告期 年初至报告期末 人民币百万元",
    ]
    rows = [
        (
            0,
            0,
            2,
            [
                "归属于上市公司股东的扣除非经常性损益的净利润",
                "14,807",
                "11,578",
                "11,578",
                "27.9",
                "40,369",
                "31,934",
                "31,934",
                "26.4",
            ],
        )
    ]
    evidence = find_core_profit_from_summary_tables(
        pages,
        rows,
        period_type="Q3",
        fallback_unit="百万元",
    )
    assert evidence is not None
    assert evidence.metric_value_cny == 40_369_000_000.0
    assert evidence.verification_status == "PASS_OFFICIAL_ORIGINAL_PDF_LABEL_VALUE_UNIT_VERIFIED"


def test_summary_table_joins_label_split_across_adjacent_pages() -> None:
    pages = ["年初至报告期末 单位：元", "季度报告"]
    rows = [
        (0, 0, 8, ["归属于上市公司", "8,851,504,370", "4,816,333,200", "83.78"]),
        (1, 0, 0, ["股东的扣除非经常性损益的净利润", "", "", ""]),
    ]
    evidence = find_core_profit_from_summary_tables(
        pages,
        rows,
        period_type="Q3",
        fallback_unit="元",
    )
    assert evidence is not None
    assert evidence.metric_value_cny == 8_851_504_370.0
    assert "continuation" in evidence.source_locator


def test_summary_table_accepts_label_truncated_after_net_character() -> None:
    pages = ["第三季度报告 本报告期 年初至报告期末 单位：元"]
    rows = [
        (
            0,
            0,
            4,
            [
                "归属于上市公司股东的扣除非经常性损益的净",
                "663,826,285",
                "28.90",
                "2,869,415,539",
                "0.48",
            ],
        )
    ]
    evidence = find_core_profit_from_summary_tables(
        pages,
        rows,
        period_type="Q3",
        fallback_unit="元",
    )
    assert evidence is not None
    assert evidence.metric_value_cny == 2_869_415_539.0


def test_statement_parenthesized_note_number_is_not_an_amount() -> None:
    pages = [
        """
        合并利润表 （除特别注明外，金额单位为人民币千元）
        营业成本 225,273,833 234,072,360
        一、营业收入四(42) 258,409,403 267,490,414
        """
    ]
    candidate = find_amount(
        pages,
        [0],
        label_patterns=(r"一[、.]营业收入",),
        fallback_unit="千元",
        skip_statement_note_number=True,
    )
    assert candidate is not None
    assert candidate.raw_value == "258,409,403"
    assert candidate.value_cny == 258_409_403_000


def test_nested_statement_note_number_is_not_cash_flow_amount() -> None:
    pages = [
        """
        合并现金流量表 人民币百万元
        销售商品、提供劳务收到的现金 235,360 221,100
        经营活动产生的现金流量净额 （六）51(1) 118,554 99,618
        """
    ]
    candidate = find_amount(
        pages,
        [0],
        label_patterns=(r"经营活动产生的现金流量净额",),
        fallback_unit="百万元",
        skip_statement_note_number=True,
    )
    assert candidate is not None
    assert candidate.raw_value == "118,554"
    assert candidate.value_cny == 118_554_000_000


def test_fy_core_profit_prefers_annual_table_and_nearest_unit() -> None:
    pages = [
        """
        市值统计单位：亿元
        主要会计数据（人民币千元）
        归属于上市公司股东的扣除非经常性损益的净利润
        7,133,730 5,336,924 33.67% 1,834,199
        """,
        """
        （人民币千元）第一季度第二季度第三季度第四季度
        归属于上市公司股东的扣除非经常性损益的净利润
        1,517,130 2,188,179 1,840,603 1,587,818
        """,
        *complete_pages()[1:],
    ]
    result = extract_metrics_from_page_texts(pages, period_type="FY")
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == 7_133_730_000.0
    assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["source_page"] == 1


def test_v1_1_config_uses_new_checkpoint_root_and_frozen_main_status() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load(
        (root / "config/csi300_pit_fundamental_underreaction_official_facts_v1_1.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert config["extraction"]["parser_version"] == PARSER_VERSION
    assert (
        config["extraction"]["admitted_verification_status"]
        == ADMITTED_VERIFICATION_STATUS
    )
    assert config["artifacts"]["checkpoint_root"].endswith("/checkpoints_v1_1")
    assert (
        config["artifacts"]["checkpoint_root"]
        != config["supersedes"]["legacy_checkpoint_root"]
    )
    assert config["admission"]["pass_status"] == (
        "PASS_OFFICIAL_PDF_VERIFIED_PIT_FINANCIAL_FACTS_V1"
    )


def test_v1_1_collector_wrapper_binds_new_parser_and_config() -> None:
    from scripts import (
        collect_csi300_pit_fundamental_underreaction_official_facts_v1_1 as collector,
    )

    collector._patch_base_collector()
    assert collector._base.PARSER_VERSION == PARSER_VERSION
    assert collector._base.extract_official_pdf_facts is collector.extract_official_pdf_facts
    assert collector._base.CONFIG_PATH == collector.CONFIG_PATH
    assert collector._base.process_document is collector.process_document
