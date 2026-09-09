from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path

from research import csi300_pit_fundamental_underreaction_official_facts_v1_7 as parser


ROOT = Path(__file__).resolve().parents[1]


def _empty_result() -> dict[str, object]:
    return {
        "metrics": [],
        "missing_metrics": list(parser.REQUIRED_METRICS),
        "document_complete": False,
        "document_status": "NO_VIEW_REQUIRED_OFFICIAL_FINANCIAL_FACTS_INCOMPLETE",
        "parser_version": parser.PARSER_VERSION,
    }


def test_adjacent_note_pages_require_both_period_reconciliation() -> None:
    result = _empty_result()
    page_texts = [
        (
            "合并资产负债表 单位：元 流动资产 货币资金 "
            "16,264,512,796 9,364,823,477 "
            "应收票据及应收账款 24,878,005,303 25,447,594,596 "
            "资产总计 414,343,673,366 400,292,347,689"
        ),
        (
            "四、合并财务报表主要项目附注 "
            "3、应收票据及应收账款 2018年6月30日 2017年12月31日 "
            "应收票据 3,962,251,263 3,610,927,507 "
            "应收账款 20,915,754,040 21,836,667,089 "
            "合计 24,878,005,303 25,447,594,596"
        ),
        (
            "四、合并财务报表主要项目附注（续） "
            "应收账款总体分析如下：2018年6月30日 2017年12月31日 "
            "应收账款 21,224,626,966 22,144,630,543 "
            "减：坏账准备 308,872,926 307,963,454 "
            "合计 20,915,754,040 21,836,667,089"
        ),
    ]

    receipt = parser._add_adjacent_note_net_receivable_reconciliation(
        result,
        page_texts,
    )
    metrics = {row["metric_id"]: row for row in result["metrics"]}

    assert receipt["status"] == (
        "PASS_ADJACENT_NOTE_NET_ACCOUNTS_RECEIVABLE_RECONCILED_AND_ADMITTED"
    )
    assert receipt["current_period_identity_passed"] is True
    assert receipt["prior_period_identity_passed"] is True
    assert metrics["ACCOUNTS_RECEIVABLE_END"]["metric_value_cny"] == 20_915_754_040.0
    assert metrics["ACCOUNTS_RECEIVABLE_END"]["source_page"] == 3


def test_adjacent_note_pages_reject_when_prior_period_identity_fails() -> None:
    result = _empty_result()
    page_texts = [
        (
            "合并资产负债表 单位：元 流动资产 货币资金 "
            "16,264,512,796 9,364,823,477 "
            "应收票据及应收账款 24,878,005,303 25,447,594,696 "
            "资产总计 414,343,673,366 400,292,347,689"
        ),
        (
            "四、合并财务报表主要项目附注 "
            "3、应收票据及应收账款 2018年6月30日 2017年12月31日 "
            "应收票据 3,962,251,263 3,610,927,507"
        ),
        (
            "四、合并财务报表主要项目附注（续） "
            "应收账款 21,224,626,966 22,144,630,543 "
            "减：坏账准备 308,872,926 307,963,454"
        ),
    ]

    receipt = parser._add_adjacent_note_net_receivable_reconciliation(
        result,
        page_texts,
    )

    assert receipt["status"] == (
        "NO_MATCH_ADJACENT_NOTE_DUAL_PERIOD_IDENTITY_NOT_PROVEN"
    )
    assert "ACCOUNTS_RECEIVABLE_END" in result["missing_metrics"]


def test_real_china_unicom_q1_combined_contract_asset_is_not_pure_receivable() -> None:
    result = _empty_result()
    receipt = parser._add_adjacent_note_net_receivable_reconciliation(
        result,
        _real_pdf_page_texts("1204679732"),
    )

    assert receipt["status"] == "NO_MATCH_COMBINED_BALANCE_ROW_NOT_FOUND"
    assert receipt["admitted_metric_ids"] == []
    assert "ACCOUNTS_RECEIVABLE_END" in result["missing_metrics"]


def test_real_huaneng_2018_h1_pdf_recovers_exact_net_receivable() -> None:
    pdf_path = ROOT / "tmp/pit_v1_7_diagnostics/1205243868.PDF"
    content = pdf_path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == (
        "24efaef2639f70b83df92f93720f9c6f8e74e6c44c06a0905a2a644c3106fe47"
    )
    page_texts, _ = parser._base.extract_pdf_page_texts(content)
    normalized = [
        parser._v1_4.normalize_financial_text(text) for text in page_texts
    ]
    result = _empty_result()
    receipt = parser._add_adjacent_note_net_receivable_reconciliation(
        result,
        normalized,
    )
    metrics = {row["metric_id"]: row for row in result["metrics"]}

    assert receipt["status"] == (
        "PASS_ADJACENT_NOTE_NET_ACCOUNTS_RECEIVABLE_RECONCILED_AND_ADMITTED"
    )
    assert metrics["ACCOUNTS_RECEIVABLE_END"]["metric_value_cny"] == 20_915_754_040.0
    assert metrics["ACCOUNTS_RECEIVABLE_END"]["verification_status"] == (
        parser.ADMITTED_VERIFICATION_STATUS
    )


def _real_pdf_page_texts(announcement_id: str) -> list[str]:
    content = (ROOT / f"tmp/pit_v1_7_diagnostics/{announcement_id}.PDF").read_bytes()
    page_texts, _ = parser._base.extract_pdf_page_texts(content)
    return [parser._v1_4.normalize_financial_text(text) for text in page_texts]


CORE_DIAGNOSTIC_HASHES = {
    "1203425597": "7a2a37aba6419e493979b50b6e33d0baaac34cdf51dcd12886c0abdf3cbbdbe5",
    "1203836216": "11fc16cec126be2731c5f8cb27c72e2ac964af315f2340a3a4384ed44914f280",
    "1204069823": "ee8e04307b15a6741318c125e93aec7e42ed51e37de62e9a84cb1fd2f5669036",
    "1204701228": "4b4c0c8261c3e6bc0366614578bf35d2b50d23e342f2a558db015013ee9a3965",
    "1205316870": "a160cce0f3faf418377273ab86ca75fe6d64a82b22164dcb6d01aabd7be7fc0b",
    "1205538145": "862eddcacf7b1607c924711db4ea4b5599bfb64c488daf050ca5bae5cf4aa93d",
    "1205938323": "1d0c5346de3b6cd376102b7336e8e9fe9e9f2e1b98ec1fde4bfeb2fd2f9032b2",
    "1207008772": "0799602fef929285450c682f6df57d4ec44a4dbd48c5e21ee850b168bd8c8392",
}
CORE_GAP_CENSUS_HASHES = {
    "1207046095": "3617a0625a2041c6a74576d92dfd380f38b3ba4d0175f062f51eca79a38d02c1",
    "1208282896": "662a7fdcdc491129f0844db370fc94e86f047940a4d967c188b2fa43eafd76b7",
    "1210602400": "f22faaed99859c68e8b443e9d05b0fff7aee9e81297bbfa1d635b986d2197bb9",
    "1211407101": "6e97be1eb3d5d2fc0c1f36b6d195e3ba6df3de890a6fa20ddb804dd0f39cc892",
    "1213106333": "9b42b7d27a123acf242b91bcf002328d0aa97545fd758c3054d4ba1ac972be73",
    "1214967561": "d3d92a5a427739acad35313503e03ccc475fc2ba04a75e0ee5bf599cc478f1c0",
    "1216222070": "839e01613d5229a4519addfea97e095de2d3e231ccb23408a9741c74d054025e",
    "1218181302": "de4e5578c2e0118ac01cec02ce557913536c66f11b929c1b6fa2e5761d7ff9d0",
    "1221588414": "5b5c30e950b7f7ba1305528a27b6d76e6514687e64d7f2a0722cba2bf7437c79",
}


def _core_diagnostic_page_texts(announcement_id: str) -> list[str]:
    path = ROOT / f"tmp/pit_v1_7_core_diagnostics/{announcement_id}.PDF"
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == CORE_DIAGNOSTIC_HASHES[
        announcement_id
    ]
    page_texts, _ = parser._base.extract_pdf_page_texts(content)
    return [parser._v1_4.normalize_financial_text(text) for text in page_texts]


def _core_gap_census_page_texts(announcement_id: str) -> list[str]:
    path = ROOT / f"tmp/pit_v1_7_core_gap_census/{announcement_id}.pdf"
    content = path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == CORE_GAP_CENSUS_HASHES[
        announcement_id
    ]
    page_texts, _ = parser._base.extract_pdf_page_texts(content)
    return [parser._v1_4.normalize_financial_text(text) for text in page_texts]


def test_broad_summary_core_profit_covers_seven_real_layout_families() -> None:
    cases = (
        ("1203836216", "H1", 5_123_433_000.0),
        ("1204069823", "Q3", 1_360_878_324.51),
        ("1204701228", "Q1", 441_111_531.66),
        ("1205316870", "H1", 12_744_902_000.0),
        ("1205538145", "Q3", 13_946_432_686.30),
        ("1205938323", "FY", 8_009_141_060.32),
        ("1207008772", "Q3", 1_522_252_000.0),
    )
    for announcement_id, period_type, expected in cases:
        result = _empty_result()
        receipt = parser._add_extended_core_parent_profit(
            result,
            _core_diagnostic_page_texts(announcement_id),
            period_type=period_type,
        )
        metrics = {row["metric_id"]: row for row in result["metrics"]}

        assert receipt["status"] == (
            "PASS_EXTENDED_OR_CROSS_PAGE_CORE_PARENT_PROFIT_ADMITTED"
        )
        assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == expected


def test_broad_summary_core_profit_keeps_official_non_disclosure_missing() -> None:
    result = _empty_result()
    receipt = parser._add_extended_core_parent_profit(
        result,
        _core_diagnostic_page_texts("1203425597"),
        period_type="Q1",
    )

    assert receipt["status"] == "NO_MATCH_EXTENDED_OR_CROSS_PAGE_CORE_LABEL_NOT_FOUND"
    assert receipt["admitted_metric_ids"] == []
    assert "CORE_PARENT_NET_PROFIT_YTD" in result["missing_metrics"]


def test_real_core_gap_layouts_recover_only_the_year_to_date_amount() -> None:
    cases = (
        ("1207046095", "Q3", 41_546_000_000.0, "EXACT_EXTENDED_MULTILINE_LABEL_IN_MAIN_FINANCIAL_HIGHLIGHTS"),
        ("1208282896", "H1", 541_915_000.0, "EXACT_EXTENDED_MULTILINE_LABEL_IN_MAIN_FINANCIAL_HIGHLIGHTS"),
        ("1210602400", "H1", 13_195_175_828.82, "EXACT_LABEL_SPLIT_ACROSS_ADJACENT_SUMMARY_PAGES"),
        ("1211407101", "Q3", 2_417_303_119.11, "EXACT_LABEL_SPLIT_ACROSS_ADJACENT_SUMMARY_PAGES"),
        ("1213106333", "Q1", 670_677_786.49, "EXACT_LABEL_SPLIT_ACROSS_ADJACENT_SUMMARY_PAGES"),
        ("1214967561", "Q3", 1_133_321_886.66, "EXACT_EXTENDED_MULTILINE_LABEL_IN_MAIN_FINANCIAL_HIGHLIGHTS"),
        ("1218181302", "Q3", 7_075_095_109.32, "EXACT_LABEL_SPLIT_ACROSS_ADJACENT_SUMMARY_PAGES"),
        ("1221588414", "Q3", 7_380_280_728.90, "EXACT_LABEL_SPLIT_ACROSS_ADJACENT_SUMMARY_PAGES"),
    )
    for announcement_id, period_type, expected, method in cases:
        result = _empty_result()
        receipt = parser._add_extended_core_parent_profit(
            result,
            _core_gap_census_page_texts(announcement_id),
            period_type=period_type,
        )
        metrics = {row["metric_id"]: row for row in result["metrics"]}

        assert receipt["status"] == (
            "PASS_EXTENDED_OR_CROSS_PAGE_CORE_PARENT_PROFIT_ADMITTED"
        )
        assert receipt["method"] == method
        assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == expected
        assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["verification_status"] == (
            parser.ADMITTED_VERIFICATION_STATUS
        )


def test_real_fy_supplemental_core_profit_requires_three_way_identity() -> None:
    result = _empty_result()
    result["metrics"] = [
        {
            "metric_id": "PARENT_NET_PROFIT_YTD",
            "metric_value_cny": 20_042_045_977.0,
            "verification_status": parser.ADMITTED_VERIFICATION_STATUS,
        }
    ]
    receipt = parser._add_extended_core_parent_profit(
        result,
        _core_gap_census_page_texts("1216222070"),
        period_type="FY",
    )
    metrics = {row["metric_id"]: row for row in result["metrics"]}
    locator = json.loads(metrics["CORE_PARENT_NET_PROFIT_YTD"]["source_locator"])

    assert receipt["method"] == (
        "EXACT_SUPPLEMENTAL_CORE_PARENT_PROFIT_TRIPLE_IDENTITY"
    )
    assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == (
        19_531_070_917.0
    )
    assert locator["validation"]["nonrecurring_profit_cny"] == 510_975_060.0
    assert locator["validation"]["identity_residual_cny"] == 0.0
    assert locator["validation"]["unit_anchor"] == (
        "SUPPLEMENT_PAGE_HEADER_RENMINBI_YUAN"
    )


def test_real_china_petroleum_extended_parent_label_recovers_core_profit() -> None:
    result = _empty_result()
    receipt = parser._add_extended_core_parent_profit(
        result,
        _real_pdf_page_texts("1202613205"),
        period_type="H1",
    )
    metrics = {row["metric_id"]: row for row in result["metrics"]}

    assert receipt["status"] == (
        "PASS_EXTENDED_OR_CROSS_PAGE_CORE_PARENT_PROFIT_ADMITTED"
    )
    assert receipt["method"] == (
        "EXACT_EXTENDED_MULTILINE_LABEL_IN_MAIN_FINANCIAL_HIGHLIGHTS"
    )
    assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == (
        -9_491_000_000.0
    )


def test_real_h1_multiline_labels_keep_columns_and_local_units_separate() -> None:
    expected = {
        "1202642581": 275_000_000.0,
        "1202657301": -4_559_830_000.0,
    }
    for announcement_id, expected_value in expected.items():
        result = _empty_result()
        receipt = parser._add_extended_core_parent_profit(
            result,
            _real_pdf_page_texts(announcement_id),
            period_type="H1",
        )
        metrics = {row["metric_id"]: row for row in result["metrics"]}

        assert receipt["status"] == (
            "PASS_EXTENDED_OR_CROSS_PAGE_CORE_PARENT_PROFIT_ADMITTED"
        )
        assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == (
            expected_value
        )


def test_real_q3_cross_page_label_selects_year_to_date_core_profit() -> None:
    result = _empty_result()
    receipt = parser._add_extended_core_parent_profit(
        result,
        _real_pdf_page_texts("1202805449"),
        period_type="Q3",
    )
    metrics = {row["metric_id"]: row for row in result["metrics"]}

    assert receipt["status"] == (
        "PASS_EXTENDED_OR_CROSS_PAGE_CORE_PARENT_PROFIT_ADMITTED"
    )
    assert receipt["method"] == "EXACT_LABEL_SPLIT_ACROSS_ADJACENT_SUMMARY_PAGES"
    assert receipt["source_page"] == 3
    assert receipt["next_page"] == 4
    assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == (
        199_926_294.93
    )


def test_real_q1_cross_page_label_selects_current_core_profit() -> None:
    result = _empty_result()
    receipt = parser._add_extended_core_parent_profit(
        result,
        _real_pdf_page_texts("1203423195"),
        period_type="Q1",
    )
    metrics = {row["metric_id"]: row for row in result["metrics"]}

    assert receipt["status"] == (
        "PASS_EXTENDED_OR_CROSS_PAGE_CORE_PARENT_PROFIT_ADMITTED"
    )
    assert receipt["method"] == "EXACT_LABEL_SPLIT_ACROSS_ADJACENT_SUMMARY_PAGES"
    assert metrics["CORE_PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == 8_997_462.46


def test_real_eastern_airlines_q1_image_appendix_recovers_four_metrics() -> None:
    pdf_path = ROOT / "tmp/pit_v1_7_diagnostics/1202268047.PDF"
    content = pdf_path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == (
        "7f2f1f73794dda83789aabe5c12d96094fcfc80c0219d81c2fa4c4f06e25b55b"
    )
    checkpoint_path = ROOT / (
        "data/raw/cninfo/csi300_pit_fundamental_underreaction_enhancement_v1/"
        "checkpoints_v1_6/2016/1202268047.json.gz"
    )
    with gzip.open(checkpoint_path, "rt", encoding="utf-8") as stream:
        result = json.load(stream)
    page_texts, _ = parser._base.extract_pdf_page_texts(content)
    normalized = [
        parser._v1_4.normalize_financial_text(text) for text in page_texts
    ]

    receipt = parser._add_quarterly_image_ocr(
        result,
        content,
        normalized,
        period_type="Q1",
    )
    metrics = {row["metric_id"]: row for row in result["metrics"]}

    assert receipt["status"] == "PASS_QUARTERLY_IMAGE_OCR_METRICS_ADMITTED"
    assert receipt["unit"] == "百万元"
    assert receipt["admitted_metric_ids"] == [
        "ACCOUNTS_RECEIVABLE_END",
        "INVENTORY_END",
        "OPERATING_PROFIT_YTD",
        "TOTAL_LIABILITIES_END",
    ]
    assert result["document_complete"] is True
    assert metrics["OPERATING_PROFIT_YTD"]["metric_value_cny"] == 2_461_000_000.0
    assert metrics["ACCOUNTS_RECEIVABLE_END"]["metric_value_cny"] == 2_395_000_000.0
    assert metrics["INVENTORY_END"]["metric_value_cny"] == 2_079_000_000.0
    assert metrics["TOTAL_LIABILITIES_END"]["metric_value_cny"] == (
        149_348_000_000.0
    )


def test_real_huaneng_2016_h1_selects_total_equity_not_parent_equity() -> None:
    pdf_path = ROOT / "tmp/pit_v1_7_diagnostics/1202532867.PDF"
    content = pdf_path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == (
        "4d5053abbafe8417d7846ace2581f6c361216a5393ce81ccc39c5f299e783ee4"
    )
    page_texts, _ = parser._base.extract_pdf_page_texts(content)
    normalized = [
        parser._v1_4.normalize_financial_text(text) for text in page_texts
    ]
    result = _empty_result()
    result["metrics"] = [
        {
            "metric_id": "TOTAL_ASSETS_END",
            "metric_value_cny": 298_287_207_162.0,
        }
    ]
    result["missing_metrics"] = [
        metric_id
        for metric_id in parser.REQUIRED_METRICS
        if metric_id != "TOTAL_ASSETS_END"
    ]

    receipt = parser._add_dual_period_text_liability(result, normalized)
    metrics = {row["metric_id"]: row for row in result["metrics"]}

    assert receipt["status"] == "PASS_DUAL_PERIOD_TEXT_LIABILITY_IDENTITY_ADMITTED"
    assert receipt["current_period_identity_passed"] is True
    assert receipt["prior_period_identity_passed"] is True
    assert metrics["TOTAL_LIABILITIES_END"]["metric_value_cny"] == (
        201_445_532_646.0
    )


def test_real_parent_profit_identity_covers_owner_and_cross_page_labels() -> None:
    cases = (
        ("1202636509", "H1", 843_459_014.16, 0, None),
        ("1202772943", "Q3", 153_065_906.44, 2, None),
        ("1211403108", "Q3", 3_558_157_949.42, 0, "利润"),
    )
    for announcement_id, period_type, expected, amount_index, suffix in cases:
        result = _empty_result()
        receipt = parser._add_extended_parent_net_profit(
            result,
            _real_pdf_page_texts(announcement_id),
            period_type=period_type,
        )
        metrics = {row["metric_id"]: row for row in result["metrics"]}

        assert receipt["status"] == (
            "PASS_EXTENDED_PARENT_NET_PROFIT_IDENTITY_ADMITTED"
        )
        assert receipt["q3_selected_amount_index"] == amount_index
        assert receipt["cross_page_label_suffix"] == suffix
        assert metrics["PARENT_NET_PROFIT_YTD"]["metric_value_cny"] == expected
