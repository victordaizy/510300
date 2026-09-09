from __future__ import annotations

from decimal import Decimal

from research import csi300_pit_fundamental_underreaction_official_facts_direct_pdf_ocr_v1 as direct_ocr


def _observation(
    metric_id: str,
    scale: float,
    raw_value: str,
    value: str,
    *,
    page_index: int = 11,
    accounting_identity_verified: bool = False,
) -> direct_ocr.OcrObservation:
    return direct_ocr.OcrObservation(
        metric_id=metric_id,
        page_index=page_index,
        render_scale=scale,
        amount_index=0,
        raw_value=raw_value,
        value=Decimal(value),
        row_text=f"{metric_id}{raw_value}",
        pixel_width=2000,
        pixel_height=3000,
        accounting_identity_verified=accounting_identity_verified,
    )


def _ocr_line(y: int, *texts: str) -> dict[str, object]:
    return {
        "text": " ".join(texts),
        "words": [
            {
                "text": value,
                "x": float(index * 300),
                "y": float(y),
                "width": 120.0,
                "height": 24.0,
            }
            for index, value in enumerate(texts)
        ],
    }


def test_heading_only_income_statement_selects_continuation_pages_for_ocr() -> None:
    page_texts = ["普通正文"] * 12
    page_texts[5] = "某公司 2020 年年度报告\n6\n3、合并利润表"
    result = {
        "metrics": [],
        "missing_metrics": ["OPERATING_PROFIT_YTD"],
    }

    pages, receipt = direct_ocr.select_targeted_ocr_pages(
        result,
        page_texts,
        period_type="FY",
    )

    assert {5, 6, 7}.issubset(set(pages))
    assert "HEADING_ONLY_IMAGE_STATEMENT_CONTINUATION_RUN" in receipt[
        "selection_routes"
    ]


def test_long_narrative_statement_reference_does_not_trigger_heading_only_ocr() -> None:
    page_texts = [
        "审计工作涉及合并利润表以及其他财务资料。" + "经营说明" * 200,
    ]
    result = {
        "metrics": [],
        "missing_metrics": ["OPERATING_PROFIT_YTD"],
    }

    _, receipt = direct_ocr.select_targeted_ocr_pages(
        result,
        page_texts,
        period_type="FY",
    )

    assert "HEADING_ONLY_IMAGE_STATEMENT_CONTINUATION_RUN" not in receipt[
        "selection_routes"
    ]


def test_table_of_contents_routes_long_report_and_keeps_core_profit_summary() -> None:
    page_texts = ["普通正文"] * 100
    page_texts[2] = (
        "目录\n"
        "第九节 债券相关情况……………………54\n"
        "第十节 财务报告（未经审计）……………………55"
    )
    result = {
        "metrics": [],
        "missing_metrics": [
            "PARENT_NET_PROFIT_YTD",
            "CORE_PARENT_NET_PROFIT_YTD",
        ],
    }

    pages, receipt = direct_ocr.select_targeted_ocr_pages(
        result,
        page_texts,
        period_type="H1",
    )

    assert set(range(10)).issubset(set(pages))
    assert {52, 53, 54, 55, 73}.issubset(set(pages))
    assert receipt["financial_report_toc_start_pages"] == [55]
    assert "TEXT_TABLE_OF_CONTENTS_FINANCIAL_REPORT_WINDOW" in receipt[
        "selection_routes"
    ]
    assert "FRONT_SUMMARY_WINDOW_FOR_CORE_PARENT_NET_PROFIT" in receipt[
        "selection_routes"
    ]


def test_explicit_unit_summary_ignores_later_table_header_false_positive() -> None:
    page_texts = [
        "主要会计数据\n总资产（人民币千元） 66,800,574 66,117,790 1.03",
        (
            "子公司情况\n总资产（人民币千元）\n净资产（人民币千元）\n"
            "净利润（人民币千元）\n900,000.00 5,920,257 11,165,907"
        ),
    ]

    pair = direct_ocr.direct._summary_metric_pair(page_texts, "TOTAL_ASSETS_END")

    assert pair is not None
    assert pair[0].value_cny == Decimal("66800574000")
    assert pair[1].value_cny == Decimal("66117790000")
    assert pair[0].page_number == 1


def test_q3_summary_uses_ytd_column_anchored_by_existing_parent_profit() -> None:
    page_text = (
        "归属于上市公司股东的净利润（元） "
        "2,861,268,129.82 2,243,111,069.90 2,243,940,889.34 27.51% "
        "6,152,814,186.52 4,849,957,525.28 4,851,762,418.77 26.82%\n"
        "归属于上市公司股东扣除非经常性损益的净利润（元） "
        "2,819,127,478.82 2,143,368,877.17 2,144,193,431.01 31.48% "
        "6,008,993,697.69 4,717,571,798.26 4,719,380,116.99 27.33%"
    )
    result = {
        "metrics": [
            {
                "metric_id": "PARENT_NET_PROFIT_YTD",
                "metric_value_cny": 6_152_814_186.52,
                "source_unit": "元",
            }
        ],
        "missing_metrics": ["CORE_PARENT_NET_PROFIT_YTD"],
    }

    pair = direct_ocr.direct._summary_metric_pair(
        [page_text],
        "CORE_PARENT_NET_PROFIT_YTD",
        result=result,
        period_type="Q3",
    )

    assert pair is not None
    assert pair[0].value_cny == Decimal("6008993697.69")
    assert pair[0].selected_amount_index == 3


def test_q3_core_profit_accepts_label_continued_on_next_pdf_page() -> None:
    first_page = (
        "营业收入（元） "
        "28,089,153,328.48 23,688,671,847.34 23,688,671,847.34 18.58% "
        "86,413,756,738.74 75,510,508,489.02 75,510,508,489.02 14.44%\n"
        "归属于上市公司股东的扣除\n"
        "1,298,648,800.31 1,175,241,727.85 1,175,241,727.85 10.50% "
        "4,187,420,953.95 4,865,889,824.17 4,865,889,824.17 -13.94%"
    )
    second_page = "非经常性损益的净利润\n（元）\n经营活动产生的现金流量净额（元）"
    result = {
        "metrics": [
            {
                "metric_id": "OPERATING_REVENUE_YTD",
                "metric_value_cny": 86_413_756_738.74,
                "source_unit": "元",
            }
        ],
        "missing_metrics": ["CORE_PARENT_NET_PROFIT_YTD"],
    }

    pair = direct_ocr.direct._summary_metric_pair(
        [first_page, second_page],
        "CORE_PARENT_NET_PROFIT_YTD",
        result=result,
        period_type="Q3",
    )

    assert pair is not None
    assert pair[0].value_cny == Decimal("4187420953.95")
    assert pair[0].page_number == 1
    assert "相邻页标签续接：PDF第2页" in pair[0].line_window


def test_q3_core_profit_accepts_late_cross_page_split_and_comma_wraps() -> None:
    first_page = (
        "归属于上市公司股东的净利润（元） "
        "-90,902,208.79 570,254,929.96 573,296,007.75 -115.86% "
        "50,845,670.10 4,053,615,604.71 4,064,953,364.56 -98.75%\n"
        "归属于上市公司股东的扣除非经常性损益\n"
        "10,832,\n712.76 13,001,716.14 13,001,716.14 -16.68% "
        "-\n398,829,7\n39.95 2,975,568,198.64 2,975,568,198.64 -113.40%"
    )
    second_page = "的净利润\n（元）\n经营活动产生的现金流量净额（元）"
    result = {
        "metrics": [
            {
                "metric_id": "PARENT_NET_PROFIT_YTD",
                "metric_value_cny": 50_845_670.10,
                "source_unit": "元",
            }
        ],
        "missing_metrics": ["CORE_PARENT_NET_PROFIT_YTD"],
    }

    pair = direct_ocr.direct._summary_metric_pair(
        [first_page, second_page],
        "CORE_PARENT_NET_PROFIT_YTD",
        result=result,
        period_type="Q3",
    )

    assert pair is not None
    assert pair[0].value_cny == Decimal("-398829739.95")
    assert pair[0].selected_amount_index == 3


def test_q3_core_profit_accepts_unique_unit_and_six_column_table_layout() -> None:
    page_text = (
        "按中国企业会计准则编制的主要会计数据及财务指标\n"
        "单位：人民币百万元\n"
        "项目 截至9月30日止3个月期间 截至9月30日止9个月期间\n"
        "营业收入 481,795 411,370 17.1 1,457,704 1,150,437 26.7\n"
        "归属于母公司股东的净利润 4,688 1,196 292.0 17,362 1,724 907.1\n"
        "归属于母公司股东的扣除非经常性损益的净利润/（亏损） "
        "6,654 448 1,385.3 21,956 (9,043) -"
    )
    result = {
        "metrics": [
            {
                "metric_id": "PARENT_NET_PROFIT_YTD",
                "metric_value_cny": 17_362_000_000,
                "source_unit": "百万元",
            }
        ],
        "missing_metrics": ["CORE_PARENT_NET_PROFIT_YTD"],
    }

    pair = direct_ocr.direct._summary_metric_pair(
        [page_text],
        "CORE_PARENT_NET_PROFIT_YTD",
        result=result,
        period_type="Q3",
    )

    assert pair is not None
    assert pair[0].value_cny == Decimal("21956000000")
    assert pair[0].unit == "百万元"
    assert pair[0].selected_amount_index == 3


def test_h1_parent_profit_accepts_table_header_unit_and_missing_de_particle() -> None:
    page_text = (
        "按中国企业会计准则编制的财务数据和指标\n"
        "主要会计数据 项目 截至6月30日止6个月期间\n"
        "2023年人民币百万元 2022年人民币百万元（调整后）\n"
        "营业收入 1,593,682 1,612,126 1,612,126 (1.1)\n"
        "归属于母公司股东净利润 35,111 43,920 43,530 (20.1)\n"
        "归属于母公司股东的扣除非经常性损益后的净利润 "
        "33,655 43,350 42,960 (22.4)"
    )

    pair = direct_ocr.direct._summary_metric_pair(
        [page_text],
        "PARENT_NET_PROFIT_YTD",
        result={"metrics": [], "missing_metrics": ["PARENT_NET_PROFIT_YTD"]},
        period_type="H1",
    )

    assert pair is not None
    assert pair[0].value_cny == Decimal("35111000000")
    assert pair[1].value_cny == Decimal("43920000000")
    assert pair[0].unit == "百万元"


def test_manual_image_balance_fact_requires_two_ocr_scales_and_asset_identity() -> None:
    result = {
        "metrics": [
            {
                "metric_id": "TOTAL_ASSETS_END",
                "metric_value_cny": 1_142_444_310_000,
                "source_unit": "千元",
            }
        ],
        "missing_metrics": ["TOTAL_LIABILITIES_END"],
    }
    record = {
        "page_index": 75,
        "text": "负债合计 875 783 807 股东权益合计 266 660 503",
    }
    outputs = {2.25: [record], 3.75: [record], 4.5: [record]}

    receipt = direct_ocr._add_manual_image_balance_fact(
        result,
        outputs,
        source_context={"announcement_id": "1208326876"},
    )

    assert receipt["status"] == "PASS_MANUAL_IMAGE_BALANCE_FACT_ADMITTED"
    metric = next(
        metric
        for metric in result["metrics"]
        if metric["metric_id"] == "TOTAL_LIABILITIES_END"
    )
    assert metric["metric_value_cny"] == 875_783_807_000
    assert metric["source_page"] == 76


def test_manual_image_row_fact_requires_two_scale_dual_period_digits() -> None:
    result = {
        "metrics": [],
        "missing_metrics": ["ACCOUNTS_RECEIVABLE_END"],
    }
    record = {
        "page_index": 134,
        "text": "应收账款 5 890 241 538.78 1 378 211 622.25",
    }
    outputs = {2.25: [record], 3.75: [record], 4.5: [record]}

    receipt = direct_ocr._add_manual_image_row_fact(
        result,
        outputs,
        source_context={"announcement_id": "1207641522"},
    )

    assert receipt["status"] == "PASS_MANUAL_IMAGE_ROW_FACT_ADMITTED"
    metric = next(
        metric
        for metric in result["metrics"]
        if metric["metric_id"] == "ACCOUNTS_RECEIVABLE_END"
    )
    assert metric["metric_value_cny"] == 5_890_241_538.78
    assert metric["source_page"] == 135


def test_explicit_manual_visual_row_accepts_two_rendered_official_pdf_pages() -> None:
    result = {
        "metrics": [],
        "missing_metrics": ["ACCOUNTS_RECEIVABLE_END"],
    }
    unreadable_record = {
        "page_index": 82,
        "text": "扫描页红章覆盖导致数字 OCR 不完整",
    }
    outputs = {2.25: [unreadable_record], 3.75: [unreadable_record]}

    receipt = direct_ocr._add_manual_image_row_fact(
        result,
        outputs,
        source_context={"announcement_id": "1207586412"},
    )

    metric = next(
        metric
        for metric in result["metrics"]
        if metric["metric_id"] == "ACCOUNTS_RECEIVABLE_END"
    )
    assert receipt["status"] == "PASS_MANUAL_IMAGE_ROW_FACTS_ADMITTED"
    assert metric["metric_value_cny"] == 22_115_764_161.40
    assert metric["source_page"] == 83
    assert metric["source_method"] == (
        "MANUAL_VISUAL_OFFICIAL_PDF_ROW_MULTI_SCALE_PAGE_RENDERED"
    )
    assert metric["verification_status"] == (
        "PASS_OFFICIAL_ORIGINAL_PDF_MANUAL_VISUAL_LABEL_VALUE_UNIT_VERIFIED"
    )


def test_manual_image_row_fact_supports_negative_ytd_without_text_support_page() -> None:
    result = {
        "metrics": [],
        "missing_metrics": ["CORE_PARENT_NET_PROFIT_YTD"],
    }
    record = {
        "page_index": 6,
        "text": "扣非归母净亏损（6,920,393）（9,596,763）",
    }
    outputs = {2.25: [record], 3.75: [record], 4.5: [record]}

    receipt = direct_ocr._add_manual_image_row_fact(
        result,
        outputs,
        source_context={"announcement_id": "1210900865"},
    )

    assert receipt["status"] == "PASS_MANUAL_IMAGE_ROW_FACT_ADMITTED"
    assert "supporting_direct_text_page" not in receipt
    metric = next(
        metric
        for metric in result["metrics"]
        if metric["metric_id"] == "CORE_PARENT_NET_PROFIT_YTD"
    )
    assert metric["metric_value_cny"] == -6_920_393_000
    assert metric["value_period_scope"] == "YEAR_TO_DATE"


def test_manual_receivable_reconciliation_uses_net_book_value_and_two_identities() -> None:
    result = {
        "metrics": [],
        "missing_metrics": ["ACCOUNTS_RECEIVABLE_END"],
    }
    record = {
        "page_index": 111,
        "text": (
            "应收票据 363 348 应收账款 4 791 455 3 674 827 "
            "减：坏账准备 185 634 184 400 合计 4 606 184 3 490 775"
        ),
    }
    outputs = {2.25: [record], 3.75: [record], 4.5: [record]}

    receipt = direct_ocr._add_manual_image_receivable_reconciliation_fact(
        result,
        outputs,
        source_context={"announcement_id": "1205361911"},
    )

    assert receipt["status"] == "PASS_MANUAL_RECEIVABLE_RECONCILIATION_FACT_ADMITTED"
    assert receipt["current_net_receivable"] == "4605821"
    assert receipt["prior_net_receivable"] == "3490427"
    metric = next(
        metric
        for metric in result["metrics"]
        if metric["metric_id"] == "ACCOUNTS_RECEIVABLE_END"
    )
    assert metric["metric_value_cny"] == 4_605_821_000
    assert metric["source_page"] == 112


def test_receivable_classification_total_reads_net_book_value_columns() -> None:
    page_text = (
        "5、应收账款\n(1). 应收账款分类披露\n单位：元 币种：人民币\n"
        "类别 期末余额 期初余额 账面余额 坏账准备 账面价值\n"
        "合计 4,918,848,747.05 / 17,495,915.25 / 4,901,352,831.80 "
        "3,224,713,758.33 / 11,613,451.69 / 3,213,100,306.64"
    )

    pair = direct_ocr.direct._receivable_classification_total_pair(
        page_text,
        page_number=74,
        unit="元",
    )

    assert pair is not None
    assert pair[0].value_cny == Decimal("4901352831.80")
    assert pair[1].value_cny == Decimal("3213100306.64")
    assert pair[0].selected_amount_index == 2
    assert pair[1].selected_amount_index == 5


def test_bills_classification_total_accepts_table_continued_on_next_page() -> None:
    previous_page = (
        "4、应收票据\n(1). 应收票据分类列示\n"
        "适用 单位：元 币种：人民币"
    )
    current_page = (
        "项目 期末余额 期初余额\n"
        "银行承兑票据 49,450,000.00 77,050,000.00\n"
        "商业承兑票据\n"
        "合计 49,450,000.00 77,050,000.00\n"
        "(2). 期末公司已质押的应收票据"
    )

    pair = direct_ocr.direct._cross_page_bills_classification_total_pair(
        previous_page,
        current_page,
        page_number=73,
        unit="元",
    )

    assert pair is not None
    assert pair[0].value_cny == Decimal("49450000.00")
    assert pair[1].value_cny == Decimal("77050000.00")
    assert pair[0].page_number == 73


def test_consolidated_note_scan_stops_before_parent_company_notes() -> None:
    page_texts = ["合并附注"] * 12
    page_texts[9] = "十七、母公司财务报表主要项目注释"

    end_page = direct_ocr.direct._consolidated_note_end_page(
        page_texts,
        first_note_page=3,
    )

    assert end_page == 9


def test_standalone_receivable_pattern_rejects_reversed_combined_label() -> None:
    pair = direct_ocr.direct._pair(
        ["应收账款及应收票据 2,700,285,149.98 1,375,071,701.55"],
        (0,),
        direct_ocr.direct.RECEIVABLE_PATTERNS,
        unit="元",
    )

    assert pair is None


def test_missing_decimal_high_scale_pair_is_rejected_then_digit_reconciled() -> None:
    explicit = _observation(
        "ACCOUNTS_RECEIVABLE_END",
        2.25,
        "1377，476：016．66",
        "1377476016.66",
    )
    high_first = _observation(
        "ACCOUNTS_RECEIVABLE_END",
        3.75,
        "1，377，476，01666",
        "137747601666",
    )
    high_second = _observation(
        "ACCOUNTS_RECEIVABLE_END",
        4.5,
        "1，377，476，01666",
        "137747601666",
    )
    parsed = {
        scale: direct_ocr.ParsedOcrScale(
            observations=(observation,),
            row_presence=frozenset({("ACCOUNTS_RECEIVABLE_END", 11)}),
        )
        for scale, observation in (
            (2.25, explicit),
            (3.75, high_first),
            (4.5, high_second),
        )
    }

    assert direct_ocr._pairs(parsed, "ACCOUNTS_RECEIVABLE_END") == []
    pair = direct_ocr._digit_reconciled_pair(
        parsed,
        "ACCOUNTS_RECEIVABLE_END",
        pages={11},
        amount_index=0,
    )

    assert pair is not None
    assert pair[1].render_scale == 2.25
    assert pair[1].value == Decimal("1377476016.66")


def test_exact_pair_can_use_two_high_resolution_scales() -> None:
    low = _observation(
        "TOTAL_LIABILITIES_END",
        2.25,
        "37，357，8920．83",
        "373578920.83",
    )
    high_first = _observation(
        "TOTAL_LIABILITIES_END",
        3.75,
        "37，357，948，920．83",
        "37357948920.83",
    )
    high_second = _observation(
        "TOTAL_LIABILITIES_END",
        4.5,
        "37，357，948，920．83",
        "37357948920.83",
    )
    parsed = {
        scale: direct_ocr.ParsedOcrScale(
            observations=(observation,),
            row_presence=frozenset({("TOTAL_LIABILITIES_END", 11)}),
        )
        for scale, observation in (
            (2.25, low),
            (3.75, high_first),
            (4.5, high_second),
        )
    }

    pairs = direct_ocr._pairs(parsed, "TOTAL_LIABILITIES_END")

    assert len(pairs) == 1
    assert {pairs[0][0].render_scale, pairs[0][1].render_scale} == {3.75, 4.5}
    assert pairs[0][1].value == Decimal("37357948920.83")


def test_liability_row_accepts_cross_scale_decimal_separator_reconciliation() -> None:
    comma_decimal = _observation(
        "TOTAL_LIABILITIES_END",
        3.75,
        "44，955，516，048，50",
        "44955516048.50",
    )
    explicit_decimal = _observation(
        "TOTAL_LIABILITIES_END",
        4.5,
        "44，955，516，048．50",
        "44955516048.50",
    )
    parsed = {
        scale: direct_ocr.ParsedOcrScale(
            observations=(observation,),
            row_presence=frozenset({("TOTAL_LIABILITIES_END", 11)}),
        )
        for scale, observation in (
            (3.75, comma_decimal),
            (4.5, explicit_decimal),
        )
    }

    pairs = direct_ocr._liability_pairs_with_digit_reconciliation(
        parsed,
        pages={11},
    )

    assert len(pairs) == 1
    assert pairs[0][1]
    assert pairs[0][0][1].value == Decimal("44955516048.50")


def test_note_component_sum_recovers_total_even_when_total_label_is_misread() -> None:
    record = {
        "page_index": 19,
        "render_scale": 3.75,
        "pixel_width": 2000,
        "pixel_height": 3000,
        "lines": [
            _ocr_line(0, "50、营业外收入"),
            _ocr_line(40, "项目", "本年发生额", "上年发生额", "非经常损益金额"),
            _ocr_line(80, "项目甲", "100.00", "90.00", "100.00"),
            _ocr_line(120, "项目乙", "200.00", "180.00", "200.00"),
            _ocr_line(160, "口", "300.00", "270.00", "300.00"),
        ],
    }

    observations, presence = direct_ocr._note_table_observations(record)

    assert len(observations) == 1
    assert observations[0].metric_id == "NONOPERATING_INCOME_YTD"
    assert observations[0].value == Decimal("300.00")
    assert observations[0].accounting_identity_verified
    assert ("NONOPERATING_INCOME_YTD", 19) in presence


def test_statement_note_number_maps_only_to_adjacent_official_note_pages() -> None:
    outputs = {
        2.25: [
            {
                "page_index": 5,
                "render_scale": 2.25,
                "pixel_width": 2000,
                "pixel_height": 3000,
                "lines": [
                    _ocr_line(
                        0,
                        "加：营业外收入",
                        "50",
                        "19,000.00",
                        "18,000.00",
                    )
                ],
            }
        ]
    }
    page_texts = ["普通正文"] * 30
    page_texts[19] = "50��营业外收入"
    page_texts[20] = "51��营业外支出"
    result = {"metrics": [], "missing_metrics": ["OPERATING_PROFIT_YTD"]}

    pages, receipt = direct_ocr.select_residual_note_ocr_pages(
        result,
        page_texts,
        outputs,
    )

    assert receipt["status"] == (
        "PASS_NOTE_PAGES_SELECTED_FROM_STATEMENT_REFERENCES"
    )
    assert receipt["income_reference"]["note_number"] == 50
    assert receipt["expense_reference"]["note_number"] == 51
    assert receipt["expense_reference"]["inferred_from_adjacent_note"]
    assert pages == (19, 20, 21)


def test_official_note_profit_bridge_derives_operating_profit() -> None:
    result = {
        "metrics": [
            {
                "metric_id": "PARENT_NET_PROFIT_YTD",
                "metric_value_cny": 15000.0,
                "source_unit": "元",
            }
        ],
        "missing_metrics": ["OPERATING_PROFIT_YTD"],
    }
    observations_by_scale = {
        2.25: (
            _observation(
                "PARENT_NET_PROFIT_YTD",
                2.25,
                "15,000.00",
                "15000.00",
                page_index=5,
            ),
            _observation(
                "NONOPERATING_INCOME_YTD",
                2.25,
                "3,000.00",
                "3000.00",
                page_index=19,
                accounting_identity_verified=True,
            ),
            _observation(
                "NONOPERATING_EXPENSE_YTD",
                2.25,
                "2,500.00",
                "2500.00",
                page_index=20,
                accounting_identity_verified=True,
            ),
            _observation(
                "PROFIT_TOTAL_YTD",
                2.25,
                "20,500.00",
                "20500.00",
                page_index=21,
            ),
        ),
        3.75: (
            _observation(
                "PARENT_NET_PROFIT_YTD",
                3.75,
                "15,000.00",
                "15000.00",
                page_index=5,
            ),
            _observation(
                "NONOPERATING_INCOME_YTD",
                3.75,
                "3000.00",
                "3000.00",
                page_index=19,
                accounting_identity_verified=True,
            ),
            _observation(
                "NONOPERATING_EXPENSE_YTD",
                3.75,
                "2500.00",
                "2500.00",
                page_index=20,
                accounting_identity_verified=True,
            ),
            _observation(
                "PROFIT_TOTAL_YTD",
                3.75,
                "20500.00",
                "20500.00",
                page_index=21,
            ),
        ),
    }
    parsed = {
        scale: direct_ocr.ParsedOcrScale(
            observations=observations,
            row_presence=frozenset(
                (observation.metric_id, observation.page_index)
                for observation in observations
            ),
        )
        for scale, observations in observations_by_scale.items()
    }

    receipt = direct_ocr._add_derived_operating_profit(
        result,
        parsed,
        source_context={
            "announcement_id": "synthetic-note-profit-bridge",
            "note_selection_receipt": {
                "status": "PASS_NOTE_PAGES_SELECTED_FROM_STATEMENT_REFERENCES"
            },
        },
    )
    metric = next(
        metric
        for metric in result["metrics"]
        if metric["metric_id"] == "OPERATING_PROFIT_YTD"
    )

    assert receipt["status"] == (
        "PASS_OFFICIAL_NOTES_DERIVED_OPERATING_PROFIT_ADMITTED"
    )
    assert metric["metric_value_cny"] == 20000.0


def test_manual_direct_text_row_admits_negative_core_profit_with_exact_unit() -> None:
    result = {
        "metrics": [],
        "missing_metrics": ["CORE_PARENT_NET_PROFIT_YTD"],
    }
    page_texts = [""] * 7
    page_texts[6] = (
        "单位：千元\n"
        "归属于上市公司股东的扣除非经常性损益的（净亏损）/净利润 "
        "(4,938,159) (19,494,182)"
    )

    receipt = direct_ocr.direct._add_manual_direct_text_row_fact(
        result,
        page_texts,
        source_context={"announcement_id": "1217717273"},
    )
    metric = result["metrics"][0]

    assert receipt["status"] == "PASS_MANUAL_DIRECT_TEXT_ROW_FACT_ADMITTED"
    assert metric["metric_id"] == "CORE_PARENT_NET_PROFIT_YTD"
    assert metric["metric_value_cny"] == -4_938_159_000.0
    assert metric["value_period_scope"] == "YEAR_TO_DATE"
    assert metric["source_page"] == 7


def test_manual_direct_text_receivable_reconciliation_uses_net_book_value() -> None:
    result = {
        "metrics": [],
        "missing_metrics": ["ACCOUNTS_RECEIVABLE_END"],
    }
    page_texts = [""] * 138
    page_texts[137] = (
        "合并财务报表主要项目注释 金额单位：人民币千元\n"
        "应收票据和应收账款\n"
        "2018年12月31日 2017年12月31日\n"
        "应收票据 403 348\n"
        "应收账款 5,590,112 3,674,827\n"
        "减：坏账准备 (216,140) (184,400)\n"
        "5,374,375 3,490,775"
    )

    receipt = (
        direct_ocr.direct._add_manual_direct_text_receivable_reconciliation_fact(
            result,
            page_texts,
            source_context={"announcement_id": "1205947423"},
        )
    )
    metric = result["metrics"][0]

    assert receipt["status"] == (
        "PASS_MANUAL_DIRECT_TEXT_RECEIVABLE_RECONCILED_AND_ADMITTED"
    )
    assert receipt["current_net_receivable"] == "5373972"
    assert receipt["prior_net_receivable"] == "3490427"
    assert metric["metric_id"] == "ACCOUNTS_RECEIVABLE_END"
    assert metric["metric_value_cny"] == 5_373_972_000.0
    assert metric["source_page"] == 138


def test_manual_direct_text_row_is_scoped_to_prc_gaap_section() -> None:
    result = {
        "metrics": [],
        "missing_metrics": ["PARENT_NET_PROFIT_YTD"],
    }
    page_texts = [""] * 4
    page_texts[3] = (
        "2.1.1按国际财务报告准则编制的主要会计数据及财务指标\n"
        "单位：人民币百万元\n"
        "归属于母公司股东的净（亏损）/利润 (13,785) 6,150\n"
        "2.1.2按中国企业会计准则编制的主要会计数据及财务指标\n"
        "单位：人民币百万元\n"
        "归属于母公司股东的净（亏损）/利润 (13,786) 6,149"
    )

    receipt = direct_ocr.direct._add_manual_direct_text_row_fact(
        result,
        page_texts,
        source_context={"announcement_id": "1202268049"},
    )
    metric = result["metrics"][0]

    assert receipt["status"] == "PASS_MANUAL_DIRECT_TEXT_ROW_FACT_ADMITTED"
    assert metric["metric_id"] == "PARENT_NET_PROFIT_YTD"
    assert metric["metric_value_cny"] == -13_786_000_000.0
    assert metric["value_period_scope"] == "YEAR_TO_DATE"
    assert metric["source_page"] == 4


def test_manual_image_row_wrapper_admits_multiple_metrics_from_one_page() -> None:
    result = {
        "metrics": [],
        "missing_metrics": ["OPERATING_PROFIT_YTD", "PARENT_NET_PROFIT_YTD"],
    }
    page_record = {
        "page_index": 17,
        "text": "营业亏损 (9,252) 29,270 归母净亏损 (16,234) 10,245",
    }
    outputs = {
        2.25: [page_record],
        3.75: [page_record],
    }

    receipt = direct_ocr._add_manual_image_row_fact(
        result,
        outputs,
        source_context={"announcement_id": "1207691674"},
    )
    metrics = {metric["metric_id"]: metric for metric in result["metrics"]}

    assert receipt["status"] == "PASS_MANUAL_IMAGE_ROW_FACTS_ADMITTED"
    assert set(receipt["admitted_metric_ids"]) == {
        "OPERATING_PROFIT_YTD",
        "PARENT_NET_PROFIT_YTD",
    }
    assert metrics["OPERATING_PROFIT_YTD"]["metric_value_cny"] == -9_252_000_000.0
    assert metrics["PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == -16_234_000_000.0


def test_manual_direct_text_row_wrapper_admits_multiple_metrics_from_one_page() -> None:
    result = {
        "metrics": [],
        "missing_metrics": ["OPERATING_PROFIT_YTD", "PARENT_NET_PROFIT_YTD"],
    }
    page_texts = [""] * 18
    page_texts[17] = (
        "人民币百万元\n"
        "营业（亏损）/利润 (9,252) 29,270\n"
        "归属于母公司股东的净（亏损）/利润 (16,234) 10,245"
    )

    receipt = direct_ocr.direct._add_manual_direct_text_row_fact(
        result,
        page_texts,
        source_context={"announcement_id": "1207691674"},
    )
    metrics = {metric["metric_id"]: metric for metric in result["metrics"]}

    assert receipt["status"] == "PASS_MANUAL_DIRECT_TEXT_ROW_FACTS_ADMITTED"
    assert set(receipt["admitted_metric_ids"]) == {
        "OPERATING_PROFIT_YTD",
        "PARENT_NET_PROFIT_YTD",
    }
    assert metrics["OPERATING_PROFIT_YTD"]["metric_value_cny"] == -9_252_000_000.0
    assert metrics["PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == -16_234_000_000.0


def test_manual_direct_text_rows_support_explicit_adjacent_unit_page() -> None:
    result = {
        "metrics": [],
        "missing_metrics": [
            "ACCOUNTS_RECEIVABLE_END",
            "INVENTORY_END",
            "TOTAL_LIABILITIES_END",
        ],
    }
    page_texts = [""] * 6
    page_texts[3] = "合并资产负债表 单位:元"
    page_texts[4] = (
        "应收账款 23,484,844,663.54 18,970,494,098.56 "
        "存货 157,030,124.11 130,963,317.78"
    )
    page_texts[5] = "负债合计 160,427,011,853.66 140,601,171,304.61"

    receipt = direct_ocr.direct._add_manual_direct_text_row_fact(
        result,
        page_texts,
        source_context={"announcement_id": "1213176408"},
    )
    metrics = {metric["metric_id"]: metric for metric in result["metrics"]}

    assert receipt["status"] == "PASS_MANUAL_DIRECT_TEXT_ROW_FACTS_ADMITTED"
    assert set(receipt["admitted_metric_ids"]) == {
        "ACCOUNTS_RECEIVABLE_END",
        "INVENTORY_END",
        "TOTAL_LIABILITIES_END",
    }
    assert metrics["ACCOUNTS_RECEIVABLE_END"]["metric_value_cny"] == 23_484_844_663.54
    assert metrics["INVENTORY_END"]["metric_value_cny"] == 157_030_124.11
    assert metrics["TOTAL_LIABILITIES_END"]["metric_value_cny"] == 160_427_011_853.66


def test_manual_direct_text_row_supports_page_global_value_order() -> None:
    result = {
        "metrics": [],
        "missing_metrics": ["TOTAL_LIABILITIES_END"],
    }
    page_texts = [""] * 28
    page_texts[27] = (
        "货币单位：人民币千元\n"
        "120,511,299 103,247,837\n"
        "负债合计"
    )

    receipt = direct_ocr.direct._add_manual_direct_text_row_fact(
        result,
        page_texts,
        source_context={"announcement_id": "1208637839"},
    )
    metric = result["metrics"][0]

    assert receipt["status"] == "PASS_MANUAL_DIRECT_TEXT_ROW_FACT_ADMITTED"
    assert metric["metric_id"] == "TOTAL_LIABILITIES_END"
    assert metric["metric_value_cny"] == 120_511_299_000.0
    assert metric["source_page"] == 28
