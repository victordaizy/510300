from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from research.project_evidence_contract_v1 import EvidenceContractError, read_json_strict
from scripts.freeze_510300_stress_transmission_hazard_v2_g1_event_prediction_admission_v1_0_1 import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
    verify_frozen_manifest,
    verify_immutable_parent,
)
from scripts.replay_510300_stress_transmission_hazard_v2_g1_event_prediction_admission_v1_0_1 import (
    verify_file_identity_subset,
)


ROOT = Path(__file__).resolve().parents[1]


def test_correction_scope_changes_only_file_identity_comparison() -> None:
    config = load_config(DEFAULT_CONFIG)
    validate_config(config)
    correction = config["correction_contract"]
    assert correction["scope"] == "FILE_EVIDENCE_COMPARISON_SUBSET_ONLY"
    assert correction["exact_file_identity_fields"] == ["path", "bytes", "sha256"]
    assert correction["non_identity_fields_verified_separately"] == [
        "row_count",
        "persisted_semantic_sha256",
    ]
    for key in (
        "table_values_changed",
        "table_files_rewritten",
        "feature_formula_changed",
        "b1_formula_changed",
        "label_definition_changed",
        "common_sample_rule_changed",
        "event_weight_rule_changed",
        "g1_thresholds_changed",
        "model_training_allowed",
        "prediction_metric_generation_allowed",
        "performance_read_allowed",
    ):
        assert correction[key] is False


def test_file_identity_subset_accepts_separate_row_and_semantic_metadata() -> None:
    config = load_config(DEFAULT_CONFIG)
    receipt_path = ROOT / config["immutable_parent"]["original_build_receipt"]["path"]
    receipt = read_json_strict(receipt_path)
    metadata = receipt["outputs"]["b1_feature_panel"]
    assert "row_count" in metadata
    assert "persisted_semantic_sha256" in metadata
    path = verify_file_identity_subset(metadata, label="测试 B1")
    assert path.is_file()


def test_file_identity_subset_still_rejects_real_identity_drift() -> None:
    config = load_config(DEFAULT_CONFIG)
    receipt_path = ROOT / config["immutable_parent"]["original_build_receipt"]["path"]
    receipt = read_json_strict(receipt_path)
    metadata = deepcopy(receipt["outputs"]["b1_feature_panel"])
    metadata["bytes"] = int(metadata["bytes"]) + 1
    with pytest.raises(EvidenceContractError, match="文件身份漂移"):
        verify_file_identity_subset(metadata, label="测试 B1")


def test_expected_gate_remains_the_frozen_negative_result() -> None:
    config = load_config(DEFAULT_CONFIG)
    expected = config["expected_gate_result"]
    assert expected["G1_DATA_AND_EVENTS"] == (
        "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY"
    )
    assert expected["b2_identifiable_event_count"] == 23
    assert expected["b3_identifiable_event_count"] == 23
    assert expected["next_allowed_step"] == (
        "STOP_V2_NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY"
    )


def test_original_manifest_build_and_failure_chain_remain_immutable() -> None:
    config = load_config(DEFAULT_CONFIG)
    evidence = verify_immutable_parent(config)
    assert set(evidence) == {
        "original_config",
        "original_manifest",
        "original_freeze_receipt",
        "original_build_receipt",
        "original_clean_replay_failure",
    }


def test_correction_manifest_verifies_when_present() -> None:
    config = load_config(DEFAULT_CONFIG)
    manifest_path = ROOT / config["freeze_contract"]["manifest_output"]
    if not manifest_path.exists():
        return
    manifest = verify_frozen_manifest(DEFAULT_CONFIG)
    assert manifest["status"] == (
        "FROZEN_APPEND_ONLY_FILE_EVIDENCE_COMPARISON_CORRECTION"
    )
    assert manifest["this_freeze_actual_label_artifact_parsed"] is False
    assert manifest["model_trained"] is False
    assert manifest["position_impact"] == 0
