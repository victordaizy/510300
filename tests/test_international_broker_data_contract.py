from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.international_broker_data_contract import (
    EXPECTED_TABLES,
    audit_table,
    audit_universe,
    load_config,
    run_audit,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config()


def test_contract_has_exactly_five_analysis_tables() -> None:
    evidence = validate_contract(CONFIG)
    assert evidence["status"] == "PASS"
    assert evidence["table_count"] == 5
    assert evidence["table_names"] == list(EXPECTED_TABLES)


def test_governance_is_isolated_from_510300_and_trading() -> None:
    governance = CONFIG["governance"]
    assert governance["trading_authorization"] == "RESEARCH_ONLY_NO_POSITION_CHANGE"
    assert governance["isolated_from_510300"] is True
    assert governance["allow_as_510300_alpha_input"] is False
    assert governance["position_mapping_enabled"] is False
    assert governance["order_generation_enabled"] is False
    assert governance["broker_connection_enabled"] is False
    assert governance["live_trading_authorized"] is False
    assert governance["return_test_allowed"] is False


def test_frozen_materiality_and_profitability_thresholds() -> None:
    thresholds = CONFIG["frozen_thresholds"]
    assert thresholds["minimum_comparable_years"] == 3
    assert thresholds["target_comparable_years"] == 5
    assert thresholds["minimum_intl_profit_share"] == 0.10
    assert thresholds["minimum_two_year_earnings_materiality"] == 0.05
    assert thresholds["minimum_group_roe_uplift_percentage_points"] == 0.5
    assert thresholds["intl_roe_above_cost_of_equity_consecutive_years"] == 2


def test_initial_universe_has_42_core_a_and_13_dual_ah_issuers() -> None:
    registry = pd.read_csv(
        ROOT / CONFIG["evidence_registry"]["file"],
        dtype=str,
        keep_default_na=False,
    )
    result = audit_universe(ROOT, CONFIG, set(registry["source_id"]))
    assert result["core_a_issuer_count"] == 42
    assert result["dual_ah_count"] == 13
    assert result["official_verified_core_count"] == 0
    assert result["unverified_core_count"] == 42
    assert result["status"] == "BLOCKED_UNVERIFIED_OFFICIAL_LISTINGS"


def test_haitong_delisted_tickers_are_not_current_core_issuers() -> None:
    universe = pd.read_csv(
        ROOT / CONFIG["universe"]["file"], dtype=str, keep_default_na=False
    )
    core = universe.loc[universe["scope_status"].eq("CORE_A_LISTED")]
    current = set(core["a_ticker"].str.strip()) | set(core["h_ticker"].str.strip())
    assert "600837.SH" not in current
    assert "06837.HK" not in current


def test_nasdaq_23_hour_event_is_approved_but_not_live_revenue() -> None:
    events = pd.read_csv(
        ROOT
        / CONFIG["tables"]["international_broker_events"]["seed_file"],
        dtype=str,
        keep_default_na=False,
    ).set_index("event_id")
    approval = events.loc["EVT_NASDAQ_23H_APPROVAL"]
    expected_launch = events.loc["EVT_NASDAQ_23H_EXPECTED_LAUNCH"]
    assert approval["approval_date"] == "2026-04-10"
    assert approval["implementation_status"] == "APPROVED_NOT_LIVE"
    assert expected_launch["effective_date"] == "2026-12-06"
    assert expected_launch["implementation_status"] == "EXPECTED_CONDITIONAL"
    assert expected_launch["first_revenue_date"] == ""


def test_source_registry_does_not_treat_unhashed_links_as_frozen_evidence() -> None:
    report = run_audit(write_reports=False)
    registry = report["evidence_registry"]
    assert registry["status"] == "PARTIAL_UNHASHED_PRIMARY_SOURCES"
    assert registry["unhashed_primary_source_count"] > 0


def test_report_publish_date_before_period_is_blocked(tmp_path: Path) -> None:
    table_config = CONFIG["tables"]["broker_parent_quarterly"]
    row = {column: "" for column in table_config["required_columns"]}
    row.update(
        {
            "ticker": "600030.SH",
            "company_name": "中信证券股份有限公司",
            "report_period": "2025-12-31",
            "report_publish_date": "2025-12-01",
            "segment_scope_version": "v1",
            "evidence_ids": "SRC_SSE_DISCLOSURE",
            "source_publish_date": "2025-12-01",
            "retrieved_at": "2026-08-16T22:30:00+08:00",
            "data_status": "TEST",
            "notes": "测试披露日防穿越",
        }
    )
    path = tmp_path / "broker_parent_quarterly.csv"
    pd.DataFrame([row]).to_csv(path, index=False, encoding="utf-8")
    result = audit_table(
        tmp_path,
        "broker_parent_quarterly",
        table_config,
        {"SRC_SSE_DISCLOSURE"},
        override_path=path,
    )
    assert result["status"] == "BLOCKED_INVALID_DATA"
    assert any("report_publish_date早于report_period" in item for item in result["errors"])


def test_current_missing_primary_tables_block_all_return_tests() -> None:
    report = run_audit(write_reports=False)
    assert report["return_test_allowed"] is False
    assert report["gates"]["G1_UNIVERSE"]["status"] == "BLOCKED"
    assert report["gates"]["G6_PRICING"]["status"] == (
        "BLOCKED_EXPLICIT_APPROVAL_REQUIRED"
    )
    for table_name in EXPECTED_TABLES[:-1]:
        assert report["tables"][table_name]["status"] == "BLOCKED_MISSING_TABLE"
    assert report["tables"]["international_broker_events"]["status"] == "PASS"
    assert report["tables"]["international_broker_events"]["row_count"] == 4
