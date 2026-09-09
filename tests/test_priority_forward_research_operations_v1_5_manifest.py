from __future__ import annotations

import csv
import hashlib
import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "priority_forward_research_operations_v1_5_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v1_5_manifest_frozen_files_and_automations_are_exact() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == (
        "FROZEN_ZERO_PAID_FOUNDATION_ACTIVE_AWAITING_NEXT_WINDOW"
    )
    assert manifest["supersedes"]["previous_manifest_preserved"] is True
    assert manifest["supersedes"]["previous_repository_frozen_files_verified"] is True
    assert manifest["supersedes"][
        "previous_external_automation_evidence_is_historical_snapshot"
    ] is True
    assert manifest["supersedes"]["path"].endswith(
        "priority_forward_research_operations_v1_4_manifest.json"
    )
    assert len(manifest["frozen_files"]) == 16
    for item in manifest["frozen_files"]:
        path = ROOT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert _sha256(path) == item["sha256"]
    for item in manifest["external_automation_evidence"]:
        path = Path(item["path"])
        assert path.stat().st_size == item["bytes"]
        assert _sha256(path) == item["sha256"]
        automation = tomllib.loads(path.read_text(encoding="utf-8"))
        assert automation["status"] == "ACTIVE"
        assert (
            f"run_priority_forward_codex_automation_v1_5.py --phase {item['entrypoint_phase']}"
            in automation["prompt"]
        )


def test_v1_5_zero_paid_registry_and_cash_admission_are_frozen() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["contract"]["data_purchase_budget_cny"] == 0
    assert manifest["contract"]["maximum_active_research_streams"] == 2
    with (ROOT / "data" / "governance" / "FREE_SOURCE_REGISTRY.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        registry = list(csv.DictReader(handle))
    assert len(registry) == 16
    assert len({row["source_id"] for row in registry}) == 16
    assert all(float(row["access_cost_cny"]) == 0 for row in registry)
    matrix_path = (
        ROOT
        / "data"
        / "curated"
        / "a_share_hs_official_cash_option_floor_alpha_v1_0_2_free_data_admission"
        / "free_data_admission_matrix_v1_0_2.csv"
    )
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        matrix = list(csv.DictReader(handle))
    assert len(matrix) == 21
    assert len({row["event_id"] for row in matrix}) == 21
    assert all(row["price_values_read_by_matrix"] == "false" for row in matrix)
    assert all(row["return_values_read_by_matrix"] == "false" for row in matrix)
    assert all(
        row["data_gate_status"]
        == "BLOCKED_FREE_DATA_INCOMPLETE_BEFORE_FINAL_REPAIR"
        for row in matrix
    )
    assert manifest["current_state"]["cash_option_data_gate_pass_count"] == 0


def test_v1_5_deployment_is_research_only_and_awaits_real_window() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    deployment = manifest["deployment"]
    assert deployment["codex_morning_heartbeat_active"] is True
    assert deployment["codex_close_heartbeat_active"] is True
    assert deployment["single_entrypoint_v1_5_active"] is True
    assert deployment["atomic_same_day_claim_enabled"] is True
    assert deployment["source_receipt_contract_enabled"] is True
    assert deployment["next_real_collection_window_observed"] is False
    authorization = manifest["authorization"]
    assert authorization["research_only"] is True
    assert authorization["shadow_enabled"] is False
    assert authorization["broker_connection_enabled"] is False
    assert authorization["position_mapping_enabled"] is False
    assert authorization["order_generation_enabled"] is False
    assert authorization["live_trading_enabled"] is False
