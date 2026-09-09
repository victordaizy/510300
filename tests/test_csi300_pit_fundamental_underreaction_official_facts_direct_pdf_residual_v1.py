from __future__ import annotations

from decimal import Decimal

from research import csi300_pit_fundamental_underreaction_official_facts_direct_pdf_residual_v1 as residual


def _empty_candidates() -> dict[str, list[dict[str, object]]]:
    return {metric_id: [] for metric_id in residual.SUMMARY_METRICS}


def test_summary_selection_uses_earliest_main_table_instead_of_later_rounded_copy() -> None:
    candidates = [
        {
            "value_cny": Decimal("90942694239.83"),
            "page_number": 9,
            "table_index": 0,
            "row_index": 2,
            "engine": "PDFPLUMBER_DIRECT_TABLE_DOCUMENT_STATISTICS",
        },
        {
            "value_cny": Decimal("90943000000.00"),
            "page_number": 38,
            "table_index": 1,
            "row_index": 4,
            "engine": "PDFPLUMBER_DIRECT_TABLE_DOCUMENT_STATISTICS",
        },
    ]

    selected, receipt = residual._unique_summary_candidates(candidates)

    assert selected is not None
    assert selected["value_cny"] == Decimal("90942694239.83")
    assert receipt["primary_page_number"] == 9
    assert receipt["unique_value_count"] == 1


def test_adjusted_or_mixed_revenue_rows_are_not_operating_revenue() -> None:
    assert residual._metric_matches_label(
        "OPERATING_REVENUE_YTD",
        "营业收入",
    )
    assert not residual._metric_matches_label(
        "OPERATING_REVENUE_YTD",
        "扣除与主营业务无关的业务收入后的营业收入",
    )
    assert not residual._metric_matches_label(
        "OPERATING_REVENUE_YTD",
        "营业收入归属于上市公司股东的净利润",
    )


def test_text_summary_uses_first_accounting_row_not_later_narrative_year() -> None:
    candidates = _empty_candidates()
    page_texts = [
        (
            "主要会计数据 单位：元\n"
            "经营活动产生的现金流量净额 5,013,772,777.95 4,000,000,000.00\n"
            "经营活动产生的现金流量净额同比增长，2019年进一步改善"
        )
    ]

    residual._supplement_summary_text_candidates(
        candidates,
        page_texts,
        period_type="FY",
    )
    selected, _ = residual._unique_summary_candidates(
        candidates["OPERATING_CASH_FLOW_YTD"]
    )

    assert selected is not None
    assert selected["value_cny"] == Decimal("5013772777.95")


def test_cross_page_parent_profit_label_keeps_full_precision_wanyuan_value() -> None:
    candidates = _empty_candidates()
    page_texts = [
        (
            "主要会计数据 单位：万元\n"
            "营业收入 15,226,642.31575 9,506,275.29385\n"
            "归属于上市公司股东 2,113,968.786096 1,486,010.797902"
        ),
        (
            "的净利润\n"
            "归属于上市公司股东的扣除非经常性损益的净利润 "
            "2,047,982.94 909,356.15"
        ),
    ]

    residual._supplement_summary_text_candidates(
        candidates,
        page_texts,
        period_type="FY",
    )
    selected, _ = residual._unique_summary_candidates(
        candidates["PARENT_NET_PROFIT_YTD"]
    )

    assert selected is not None
    assert selected["unit"] == "万元"
    assert selected["value_cny"] == Decimal("21139687860.960000")


def test_balance_identity_uses_statement_rows_instead_of_earlier_narrative_value() -> None:
    result = {
        "metrics": [],
        "missing_metrics": list(residual.parser.REQUIRED_METRICS),
    }
    page_texts = [
        "经营情况讨论与分析\n应收账款 1,111 1,000",
        (
            "合并资产负债表\n单位：元\n"
            "应收账款 3,000 2,500\n"
            "存货 3,500 2,800"
        ),
        "资产总计 10,000 9,000",
        (
            "负债合计 6,000 5,500\n"
            "所有者权益（或股东权益）合计 4,000 3,500\n"
            "负债和所有者权益（或股东权益）\n"
            "总计 10,000 9,000"
        ),
    ]

    receipt = residual._add_residual_balance_metrics(
        result,
        page_texts,
        source_context={"announcement_id": "synthetic-balance-identity"},
    )
    metrics = residual._metric_map(result)

    assert receipt["status"] == "PASS_RESIDUAL_BALANCE_METRICS_ADMITTED"
    assert set(receipt["admitted_metric_ids"]) == {
        "ACCOUNTS_RECEIVABLE_END",
        "INVENTORY_END",
        "TOTAL_ASSETS_END",
        "TOTAL_LIABILITIES_END",
    }
    assert metrics["ACCOUNTS_RECEIVABLE_END"]["metric_value_cny"] == 3000.0
    assert metrics["INVENTORY_END"]["metric_value_cny"] == 3500.0
    assert metrics["TOTAL_ASSETS_END"]["metric_value_cny"] == 10000.0
    assert metrics["TOTAL_LIABILITIES_END"]["metric_value_cny"] == 6000.0


def test_derived_receivable_accepts_current_and_prior_net_values_on_adjacent_pages() -> None:
    result = {
        "metrics": [],
        "missing_metrics": ["ACCOUNTS_RECEIVABLE_END"],
    }
    page_texts = [
        (
            "合并资产负债表\n单位：元\n"
            "应收票据及应收账款 9,000 8,000\n"
            "资产总计 20,000 18,000"
        ),
        (
            "负债合计 12,000 11,000\n"
            "所有者权益合计 8,000 7,000\n"
            "负债和所有者权益总计 20,000 18,000"
        ),
        (
            "应收票据及应收账款\n单位：元\n"
            "应收票据 3,000 2,500"
        ),
        "应收账款分类披露\n本期账面价值合计 6,000",
        "应收账款分类披露\n上期账面价值合计 5,500",
    ]

    receipt = residual._add_derived_receivable_metric(
        result,
        page_texts,
        source_context={"announcement_id": "synthetic-cross-page-receivable"},
    )
    metric = residual._metric_map(result)["ACCOUNTS_RECEIVABLE_END"]

    assert receipt["status"] == (
        "PASS_DERIVED_RECEIVABLE_DUAL_PERIOD_RECONCILED_AND_ADMITTED"
    )
    assert receipt["explicit_display_pages"] == [4, 5]
    assert metric["metric_value_cny"] == 6000.0


def test_generic_income_statement_is_admitted_by_dual_summary_anchors() -> None:
    result = {
        "metrics": [
            {
                "metric_id": "OPERATING_REVENUE_YTD",
                "metric_value_cny": 100000.0,
                "source_unit": "元",
            },
            {
                "metric_id": "PARENT_NET_PROFIT_YTD",
                "metric_value_cny": 15000.0,
                "source_unit": "元",
            },
        ],
        "missing_metrics": ["OPERATING_PROFIT_YTD"],
    }
    page_texts = [
        "利润表\n单位：元\n营业收入 100,000 90,000",
        (
            "营业利润 20,000 18,000\n"
            "净利润 15,000 14,000\n"
            "归属于母公司所有者的净利润 15,000 14,000\n"
            "少数股东损益\n"
            "综合收益总额 15,000 14,000"
        ),
    ]

    receipt = residual._add_income_metrics(
        result,
        page_texts,
        period_type="FY",
        source_context={"announcement_id": "synthetic-generic-income"},
    )
    metric = residual._metric_map(result)["OPERATING_PROFIT_YTD"]

    assert receipt["status"] == "PASS_RESIDUAL_INCOME_METRICS_ADMITTED"
    assert metric["metric_value_cny"] == 20000.0


def test_operating_profit_is_admitted_by_revenue_anchor_and_profit_bridge() -> None:
    result = {
        "metrics": [
            {
                "metric_id": "OPERATING_REVENUE_YTD",
                "metric_value_cny": 100000.0,
                "source_unit": "元",
            },
        ],
        "missing_metrics": ["OPERATING_PROFIT_YTD"],
    }
    page_texts = [
        "利润表\n单位：元\n营业收入 100,000 90,000",
        (
            "营业利润 20,000 18,000\n"
            "加：营业外收入 3,000 2,800\n"
            "减：营业外支出 2,500 2,300\n"
            "利润总额 20,500 18,500"
        ),
    ]

    receipt = residual._add_income_metrics(
        result,
        page_texts,
        period_type="FY",
        source_context={"announcement_id": "synthetic-profit-bridge"},
    )
    metric = residual._metric_map(result)["OPERATING_PROFIT_YTD"]

    assert receipt["status"] == "PASS_RESIDUAL_INCOME_METRICS_ADMITTED"
    assert metric["metric_value_cny"] == 20000.0
