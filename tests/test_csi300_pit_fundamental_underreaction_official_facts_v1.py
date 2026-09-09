from __future__ import annotations

from decimal import Decimal

from research.csi300_pit_fundamental_underreaction_official_facts_v1 import (
    REQUIRED_METRICS,
    extract_metrics_from_page_texts,
    find_amount,
    find_explicit_blank_current_and_prior_inventory,
    locate_statement_sections,
    parse_decimal,
)


def synthetic_pages() -> list[str]:
    return [
        """
        2025年年度报告
        主要会计数据和财务指标
        单位：万元 币种：人民币
        归属于上市公司股东的扣除
        非经常性损益的净利润 8,800.50 7,900.00
        """,
        """
        合并资产负债表
        2025年12月31日
        单位：元
        流动资产：
        货币资金 9,000,000.00 8,000,000.00
        应收账款 5 1,200,000.00 1,000,000.00
        存货 6 2,300,000.00 2,100,000.00
        资产总计 20,000,000.00 18,000,000.00
        负债合计 8,000,000.00 7,500,000.00
        """,
        """
        合并利润表
        单位：元
        一、营业总收入 30,000,000.00 25,000,000.00
        其中：营业收入 12 29,500,000.00 24,500,000.00
        三、营业利润（亏损以“-”号填列） 4,000,000.00 3,500,000.00
        1.归属于母公司所有者的净利润 3,000,000.00 2,500,000.00
        """,
        """
        合并现金流量表
        单位：元
        一、经营活动产生的现金流量：
        销售商品、提供劳务收到的现金 35,000,000.00 30,000,000.00
        经营活动产生的现金流量净额 3,200,000.00 2,800,000.00
        """,
    ]


def test_parse_decimal_supports_commas_minus_and_parentheses() -> None:
    assert parse_decimal("1,234.50") == Decimal("1234.50")
    assert parse_decimal("-1,234.50") == Decimal("-1234.50")
    assert parse_decimal("(1,234.50)") == Decimal("-1234.50")
    assert parse_decimal("-") == Decimal("0")


def test_statement_sections_ignore_table_of_contents_like_text() -> None:
    pages = ["目录 合并资产负债表 22 合并利润表 24"] + synthetic_pages()
    sections = locate_statement_sections(pages)
    assert sections["balance"].page_indices[0] == 2
    assert sections["income"].page_indices[0] == 3
    assert sections["cash"].page_indices[0] == 4


def test_statement_sections_ignore_narrative_metrics_before_heading_reference() -> None:
    pages = [
        "经营分析 营业收入 100 营业成本 80 详见合并利润表",
        *synthetic_pages(),
    ]
    sections = locate_statement_sections(pages)
    assert sections["income"].page_indices[0] == 3


def test_find_amount_skips_statement_note_number() -> None:
    pages = ["合并利润表 单位：元\n其中：营业收入 53 129,771,799,874.60 140,879,136,397.02"]
    candidate = find_amount(
        pages,
        [0],
        label_patterns=[r"其中[:：]?营业收入"],
        fallback_unit="元",
        skip_statement_note_number=True,
    )
    assert candidate is not None
    assert candidate.value_cny == Decimal("129771799874.60")
    assert candidate.raw_value == "129,771,799,874.60"


def test_extract_all_nine_metrics_from_synthetic_document() -> None:
    result = extract_metrics_from_page_texts(synthetic_pages())
    assert result["document_complete"] is True
    assert result["missing_metrics"] == []
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert set(metrics) == set(REQUIRED_METRICS)
    assert metrics["OPERATING_REVENUE_YTD"]["metric_value_cny"] == 29_500_000.0
    assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == 88_005_000.0
    assert metrics["ACCOUNTS_RECEIVABLE_END"]["metric_value_cny"] == 1_200_000.0


def test_current_period_is_always_first_amount_column() -> None:
    result = extract_metrics_from_page_texts(synthetic_pages())
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert metrics["OPERATING_PROFIT_YTD"]["metric_value_cny"] == 4_000_000.0
    assert metrics["PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == 3_000_000.0
    assert metrics["OPERATING_CASH_FLOW_YTD"]["metric_value_cny"] == 3_200_000.0
    assert metrics["TOTAL_ASSETS_END"]["metric_value_cny"] == 20_000_000.0
    assert metrics["TOTAL_LIABILITIES_END"]["metric_value_cny"] == 8_000_000.0


def test_q3_core_profit_uses_explicit_year_to_date_column() -> None:
    pages = synthetic_pages()
    pages[0] = """
    2025年第三季度报告
    本报告期 本报告期比上年同期增减 年初至报告期末 年初至报告期末比上年同期增减
    单位：元
    归属于上市公司股东的扣除非经常性损益的净利润
    100,000.00 10.00% 350,000.00 20.00%
    """
    result = extract_metrics_from_page_texts(pages, period_type="Q3")
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == 350_000.0


def test_old_q3_ytd_only_summary_uses_first_amount_not_prior_year() -> None:
    pages = synthetic_pages()
    pages[0] = """
    2020年第三季度报告
    年初至报告期末（1-9月） 上年初至上年报告期末（1-9月） 比上年同期增减（%）
    单位：元
    归属于上市公司股东的扣除非经常性损益的净利
    350,000.00 280,000.00 25.00%
    """
    result = extract_metrics_from_page_texts(pages, period_type="Q3")
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == 350_000.0


def test_q3_income_uses_ytd_columns_when_current_quarter_is_also_present() -> None:
    pages = synthetic_pages()
    pages[2] = """
    合并利润表
    2025年第三季度（7-9月） 2024年第三季度（7-9月）
    2025年前三季度（1-9月） 2024年前三季度（1-9月）
    单位：元
    其中：营业收入 10,000.00 9,000.00 35,000.00 30,000.00
    三、营业利润（亏损以“-”号填列） 2,000.00 1,800.00 7,500.00 6,800.00
    1.归属于母公司所有者的净利润 1,500.00 1,300.00 5,500.00 4,800.00
    """
    result = extract_metrics_from_page_texts(pages, period_type="Q3")
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert metrics["OPERATING_REVENUE_YTD"]["metric_value_cny"] == 35_000.0
    assert metrics["OPERATING_PROFIT_YTD"]["metric_value_cny"] == 7_500.0
    assert metrics["PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == 5_500.0


def test_old_combined_and_company_heading_and_unit_wording_are_supported() -> None:
    pages = synthetic_pages()
    pages[1] = pages[1].replace("合并资产负债表", "合并及公司资产负债表").replace(
        "单位：元", "金额单位为人民币百万元"
    )
    pages[2] = pages[2].replace("合并利润表", "合并及公司利润表").replace(
        "单位：元", "金额单位为人民币百万元"
    )
    pages[3] = pages[3].replace("合并现金流量表", "合并及公司现金流量表").replace(
        "单位：元", "金额单位为人民币百万元"
    )
    result = extract_metrics_from_page_texts(pages)
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert result["document_complete"] is True
    assert metrics["OPERATING_REVENUE_YTD"]["metric_value_cny"] == 29_500_000_000_000.0
    assert metrics["TOTAL_ASSETS_END"]["metric_value_cny"] == 20_000_000_000_000.0


def test_explicit_current_column_dash_is_zero_not_prior_period_value() -> None:
    pages = ["合并资产负债表 单位：元\n流动资产：\n货币资金 1,000 900\n存货 - 800"]
    candidate = find_amount(
        pages,
        [0],
        label_patterns=[r"存货"],
        fallback_unit="元",
        skip_statement_note_number=True,
    )
    assert candidate is not None
    assert candidate.value_cny == Decimal("0")
    assert candidate.raw_value == "-"


def test_combined_receivable_line_is_not_mislabelled_as_accounts_receivable() -> None:
    pages = synthetic_pages()
    pages[1] = pages[1].replace(
        "应收账款 5 1,200,000.00 1,000,000.00",
        "应收票据及应收账款 5 1,500,000.00 1,300,000.00",
    )
    pages.append(
        """
        合并财务报表项目注释
        单位：元
        应收票据及应收账款
        应收票据 300,000.00 300,000.00
        应收账款（b） 1,200,000.00 1,000,000.00
        """
    )
    result = extract_metrics_from_page_texts(pages)
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert metrics["ACCOUNTS_RECEIVABLE_END"]["metric_value_cny"] == 1_200_000.0
    assert "CONSOLIDATED_STATEMENT_NOTES" in metrics[
        "ACCOUNTS_RECEIVABLE_END"
    ]["source_locator"]


def test_blank_metric_row_does_not_borrow_next_rows_amount() -> None:
    pages = [
        """
        合并资产负债表
        单位：元
        流动资产：
        货币资金 1,000.00 900.00
        存货
        其他流动资产 8,000.00 7,000.00
        """
    ]
    candidate = find_amount(
        pages,
        [0],
        label_patterns=[r"存货"],
        fallback_unit="元",
        skip_statement_note_number=True,
    )
    assert candidate is None


def test_parent_profit_accepts_numbered_prefix_with_whitespace() -> None:
    pages = synthetic_pages()
    pages[2] = pages[2].replace(
        "1.归属于母公司所有者的净利润",
        "1. 归属于母公司股东的净利润",
    )
    result = extract_metrics_from_page_texts(pages)
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert metrics["PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == 3_000_000.0


def test_core_profit_accepts_unit_line_between_label_and_amount() -> None:
    pages = synthetic_pages()
    pages[0] = """
    2024年半年度报告
    主要会计数据和财务指标
    扣除非经常性损益后的净利润
    （元）
    88,005,000.00 79,000,000.00
    """
    result = extract_metrics_from_page_texts(pages, period_type="H1")
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == 88_005_000.0


def test_inventory_zero_requires_explicit_blank_current_and_prior_table_cells() -> None:
    page_texts = ["合并资产负债表\n单位：元\n存货"]
    evidence = find_explicit_blank_current_and_prior_inventory(
        page_texts,
        [(0, 1, 3, [None, "存 货", None, ""])],
        fallback_unit="元",
    )
    assert evidence is not None
    assert evidence.metric_value_cny == 0.0
    assert evidence.source_page == 1
    assert "table_index" in evidence.source_locator

    nonblank_prior = find_explicit_blank_current_and_prior_inventory(
        page_texts,
        [(0, 1, 3, [None, "存货", None, "800.00"])],
        fallback_unit="元",
    )
    assert nonblank_prior is None
