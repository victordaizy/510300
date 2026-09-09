from __future__ import annotations

from pathlib import Path

import yaml

from research.creation_data_readiness_ledger_v1 import LEDGER_SCHEMA, PROJECT_ID, _qualification


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_creation_data_readiness_ledger_v1.yaml"


def load_protocol() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def complete_row() -> dict:
    return {
        "trade_date": "2026-09-01",
        "actual_fund_shares": 8_500_000_000.0,
        "actual_share_available_at_verified": True,
        "actual_share_available_at": "2026-09-01T15:01:00+08:00",
        "actual_share_raw_source_document_id": "share-doc-1",
        "actual_share_raw_document_sha256": "a" * 64,
        "revision_flag": False,
        "split_flag": False,
        "iopv_snapshot_count": 240,
        "iopv_valid_snapshot_count": 238,
        "bid_ask_valid_snapshot_count": 230,
        "iopv_first_exchange_timestamp": "2026-09-01T09:30:00+08:00",
        "iopv_last_exchange_timestamp": "2026-09-01T14:59:59+08:00",
        "iopv_raw_source_document_id": "iopv-doc-1",
        "iopv_raw_document_sha256": "b" * 64,
    }


def test_protocol_is_data_only_and_does_not_modify_parent_v1() -> None:
    config = load_protocol()
    protocol = config["protocol"]
    assert protocol["project_id"] == PROJECT_ID
    assert protocol["research_mode"] == "FORWARD_DATA_BUILD_ONLY"
    assert protocol["return_view"] == "FORBIDDEN"
    assert protocol["strategy_code"] == "FORBIDDEN"
    assert protocol["parent_strategy_protocol_change"] == "FORBIDDEN"
    assert protocol["maturity_complete_days"] == 120
    assert protocol["backfill_allowed"] is False
    assert not any(config["boundaries"].values())


def test_ledger_has_no_strategy_return_or_signal_fields() -> None:
    forbidden_fragments = ("gross_return", "net_return", "sharpe", "signal", "position_size")
    for column in LEDGER_SCHEMA:
        assert not any(fragment in column.lower() for fragment in forbidden_fragments)
    assert LEDGER_SCHEMA["return_view"] == "string"


def test_strict_day_requires_all_provenance_and_coverage_gates() -> None:
    config = load_protocol()
    strict, close_coverage, reasons = _qualification(complete_row(), config)
    assert strict is True
    assert close_coverage is True
    assert reasons == []


def test_missing_bid_ask_and_available_at_are_not_complete_days() -> None:
    config = load_protocol()
    row = complete_row()
    row["actual_share_available_at_verified"] = False
    row["bid_ask_valid_snapshot_count"] = 0
    strict, close_coverage, reasons = _qualification(row, config)
    assert strict is False
    assert close_coverage is True
    assert "ACTUAL_SHARE_AVAILABLE_AT_NOT_VERIFIED" in reasons
    assert "BID_ASK_COVERAGE_BELOW_95_PERCENT" in reasons


def test_close_coverage_is_a_hard_gate() -> None:
    config = load_protocol()
    row = complete_row()
    row["iopv_last_exchange_timestamp"] = "2026-09-01T13:37:40+08:00"
    strict, close_coverage, reasons = _qualification(row, config)
    assert strict is False
    assert close_coverage is False
    assert "IOPV_CLOSE_COVERAGE_FAILED" in reasons


def test_impossible_counts_and_invalid_hashes_cannot_form_a_complete_day() -> None:
    config = load_protocol()
    row = complete_row()
    row["iopv_valid_snapshot_count"] = 241
    row["actual_share_raw_document_sha256"] = "not-a-hash"
    strict, _, reasons = _qualification(row, config)
    assert strict is False
    assert "IOPV_VALID_SNAPSHOT_COUNT_OUT_OF_RANGE" in reasons
    assert "ACTUAL_SHARE_RAW_DOCUMENT_SHA256_MISSING_OR_INVALID" in reasons
