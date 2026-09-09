from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.international_broker_priority_reconstruction import (
    _existing_2025_rows,
    expected_report_company_name,
    load_reconstruction_config,
    verify_g1_prerequisite,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_reconstruction_config()


def test_g1_prerequisite_is_currently_satisfied() -> None:
    result = verify_g1_prerequisite(ROOT, CONFIG)
    assert result["status"] == "PASS"
    assert result["errors"] == []


def test_priority_reconstruction_is_frozen_to_six_tickers_and_five_years() -> None:
    assert len(CONFIG["priority_tickers"]) == 6
    assert CONFIG["report_years"] == [2021, 2022, 2023, 2024, 2025]
    assert CONFIG["governance"]["allow_return_test"] is False
    assert CONFIG["governance"]["allow_510300_input"] is False


def test_gtn_historical_name_is_versioned_before_merger() -> None:
    historical = expected_report_company_name(
        "601211.SH", 2024, "国泰海通证券股份有限公司", CONFIG
    )
    current = expected_report_company_name(
        "601211.SH", 2025, "国泰海通证券股份有限公司", CONFIG
    )
    assert historical == "国泰君安证券股份有限公司"
    assert current == "国泰海通证券股份有限公司"


def test_reused_2025_manifest_keeps_ticker_as_a_column(tmp_path: Path) -> None:
    annual_path = tmp_path / "annual.csv"
    pd.DataFrame(
        [
            {
                "a_ticker": "600030.SH",
                "announcement_title": "中信证券2025年年度报告",
                "validation_status": "PASS",
            }
        ]
    ).to_csv(annual_path, index=False, encoding="utf-8-sig")
    local_config = {
        **CONFIG,
        "priority_tickers": ["600030.SH"],
    }
    evidence_config = {"paths": {"annual_report_manifest": "annual.csv"}}
    universe = pd.DataFrame(
        [
            {
                "a_ticker": "600030.SH",
                "company_name": "中信证券股份有限公司",
            }
        ]
    )
    rows = _existing_2025_rows(
        tmp_path, local_config, evidence_config, universe
    )
    assert rows[0]["a_ticker"] == "600030.SH"
