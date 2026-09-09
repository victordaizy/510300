from __future__ import annotations

import copy
from pathlib import Path

from scripts.validate_research_registry import (
    DEFAULT_REGISTRY,
    latest_records,
    load_registry,
    validate_records,
)


ROOT = Path(__file__).resolve().parents[1]


def test_registry_schema_and_declared_hashes_are_valid() -> None:
    records = load_registry(DEFAULT_REGISTRY)
    result = validate_records(
        records,
        root=ROOT,
        verify_hashes=True,
        audit_coverage=False,
    )
    assert result["errors"] == []
    assert result["record_count"] == len(records)
    assert result["record_count"] >= 19
    assert result["model_count"] >= 16
    assert result["status"] == "BLOCKED_GOVERNANCE_UNVERSIONED"


def test_registry_contains_no_unicode_replacement_character() -> None:
    assert "\ufffd" not in DEFAULT_REGISTRY.read_text(encoding="utf-8")


def test_orthogonal_status_axes_and_safety_are_explicit() -> None:
    latest = latest_records(load_registry(DEFAULT_REGISTRY))
    for record in latest:
        assert record["run_status"] in {"SUCCESS", "FAILED"}
        assert record["collection_status"] in {
            "NOT_STARTED",
            "COLLECTING",
            "COMPLETE",
            "BLOCKED",
        }
        assert record["research_status"] in {
            "DISCOVERY_ONLY",
            "PREREGISTERED",
            "FORWARD_COLLECTING",
            "EVALUATION_ELIGIBLE",
            "PASSED_RESEARCH",
            "REJECTED_FROZEN",
        }
        assert record["view_status"] in {"VIEW", "NO_VIEW"}
        assert record["authorization"] in {"NONE", "PAPER_ONLY", "REAL_DISABLED"}
        assert record["live_trading_authorized"] is False


def test_project_wide_manifest_coverage_is_honestly_blocked() -> None:
    records = load_registry(DEFAULT_REGISTRY)
    result = validate_records(
        records,
        root=ROOT,
        verify_hashes=False,
        audit_coverage=True,
    )
    assert result["errors"] == []
    assert result["status"] == "BLOCKED_GOVERNANCE_UNVERSIONED"
    assert result["unregistered_manifest_count"] > 0


def test_rejected_frozen_is_terminal() -> None:
    records = load_registry(DEFAULT_REGISTRY)
    frozen = next(
        record
        for record in records
        if record["model_id"] == "510300_X1_COMMON_TAIL_PREDICTION_V1"
    )
    illegal = copy.deepcopy(frozen)
    illegal["event_id"] = "20260819_ILLEGAL_X1_UNFREEZE"
    illegal["recorded_at"] = "2026-08-19T23:00:00+08:00"
    illegal["as_of_date"] = "2026-08-19"
    illegal["research_status"] = "DISCOVERY_ONLY"
    result = validate_records(
        [*records, illegal],
        root=ROOT,
        verify_hashes=False,
        audit_coverage=False,
    )
    assert any("REJECTED_FROZEN终态" in error for error in result["errors"])
