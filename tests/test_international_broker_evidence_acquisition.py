from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.international_broker_evidence_acquisition import (
    _ticker_code_present,
    build_merged_contract,
    build_verified_registry,
    build_verified_universe,
    load_evidence_config,
    select_full_annual_report,
    verify_frozen_v1,
)
from research.international_broker_data_contract import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_evidence_config()


def test_frozen_v1_files_are_unchanged_before_v1_1_acquisition() -> None:
    result = verify_frozen_v1(ROOT, CONFIG)
    assert result["status"] == "PASS"
    assert result["mismatches"] == []


def test_v1_1_contract_preserves_all_non_trading_governance() -> None:
    merged = build_merged_contract(ROOT, CONFIG)
    evidence = validate_contract(merged)
    assert evidence["status"] == "PASS"
    assert merged["protocol"]["version"] == "1.1.0"
    assert merged["governance"]["return_test_allowed"] is False
    assert merged["governance"]["allow_as_510300_alpha_input"] is False
    assert "cninfo.com.cn" in merged["source_policy"]["accepted_official_domains"]


def test_full_annual_report_selection_excludes_summary_and_uses_latest() -> None:
    records = [
        {
            "announcementTitle": "示例券商2025年年度报告摘要",
            "announcementTime": 3,
            "announcementId": 30,
            "adjunctUrl": "summary.PDF",
        },
        {
            "announcementTitle": "示例券商2025年年度报告",
            "announcementTime": 1,
            "announcementId": 10,
            "adjunctUrl": "old.PDF",
        },
        {
            "announcementTitle": "示例券商2025年年度报告（修订版）",
            "announcementTime": 2,
            "announcementId": 20,
            "adjunctUrl": "latest.PDF",
        },
        {
            "announcementTitle": "示例券商2025年度报告",
            "announcementTime": 4,
            "announcementId": 40,
            "adjunctUrl": "official-short-title.PDF",
        },
    ]
    selected = select_full_annual_report(records, 2025)
    assert selected is not None
    assert selected["adjunctUrl"] == "official-short-title.PDF"


def test_h_share_code_accepts_official_format_without_leading_zero() -> None:
    normalized = "H股股票代码：6806；A股股票代码：000166"
    assert _ticker_code_present(normalized, "06806") is True
    assert _ticker_code_present(normalized, "06030") is False


def test_missing_annual_document_is_not_registered_as_primary_evidence() -> None:
    base_registry = pd.DataFrame(
        columns=[
            "source_id",
            "source_name",
            "source_type",
            "authority_level",
            "url",
            "publication_date",
            "retrieved_at",
            "snapshot_sha256",
            "coverage",
            "limitations",
        ]
    )
    snapshots = pd.DataFrame(columns=["source_id"])
    annual = pd.DataFrame(
        [
            {
                "a_ticker": "600001.SH",
                "company_name": "甲证券股份有限公司",
                "announcement_title": "",
                "source_url": "",
                "publication_date": "",
                "sha256": "",
                "validation_status": "BLOCKED_ACQUISITION_ERROR",
                "error": "未找到年报",
                "local_file": "data/raw/missing.pdf",
            }
        ]
    )
    output = build_verified_registry(
        base_registry, snapshots, annual, "2026-08-16T00:00:00+08:00"
    )
    assert output.loc[0, "authority_level"] == "BLOCKED"
    assert output.loc[0, "source_type"] == "MISSING_ANNUAL_REPORT"


def test_verified_universe_only_marks_passed_pdf_identity() -> None:
    universe = pd.DataFrame(
        [
            {
                "parent_id": "BRK_A",
                "company_name": "甲证券股份有限公司",
                "a_ticker": "600001.SH",
                "h_ticker": "",
                "issuer_type": "SECURITIES_ISSUER",
                "scope_status": "CORE_A_LISTED",
                "listing_status": "CURRENT",
                "official_listing_verified": "false",
                "listing_source_id": "OLD",
                "notes": "旧状态",
            },
            {
                "parent_id": "BRK_B",
                "company_name": "乙证券股份有限公司",
                "a_ticker": "000002.SZ",
                "h_ticker": "",
                "issuer_type": "SECURITIES_ISSUER",
                "scope_status": "CORE_A_LISTED",
                "listing_status": "CURRENT",
                "official_listing_verified": "false",
                "listing_source_id": "OLD",
                "notes": "旧状态",
            },
        ]
    )
    manifest = pd.DataFrame(
        [
            {
                "a_ticker": "600001.SH",
                "validation_status": "PASS",
                "sha256": "a" * 64,
            },
            {
                "a_ticker": "000002.SZ",
                "validation_status": "BLOCKED_PDF_IDENTITY_MISMATCH",
                "sha256": "b" * 64,
            },
        ]
    )
    output = build_verified_universe(universe, manifest).set_index("a_ticker")
    assert output.loc["600001.SH", "official_listing_verified"] == "true"
    assert output.loc["000002.SZ", "official_listing_verified"] == "false"
