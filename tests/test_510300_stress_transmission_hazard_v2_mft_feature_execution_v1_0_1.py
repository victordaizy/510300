from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.project_evidence_contract_v1 import EvidenceContractError, sha256_file
from scripts.build_510300_stress_transmission_hazard_v2_mft_features_v1 import (
    frame_semantic_sha256,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_mft_feature_execution_v1_0_1 import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
    verify_frozen_manifest,
)
from scripts.replay_510300_stress_transmission_hazard_v2_mft_features_v1_0_1 import (
    canonicalize_declared_persisted_dtypes,
)


ROOT = Path(__file__).resolve().parents[1]


def test_correction_scope_changes_only_declared_persisted_dtype_comparison() -> None:
    config = load_config(DEFAULT_CONFIG)
    validate_config(config)
    rule = config["canonicalization_rule"]
    assert config["program"]["correction_scope"] == (
        "PERSISTED_SCHEMA_CANONICALIZATION_BEFORE_REPLAY_COMPARISON_ONLY"
    )
    assert rule["feature_formula_change_allowed"] is False
    assert rule["information_clock_change_allowed"] is False
    assert rule["input_change_allowed"] is False
    assert rule["coverage_gate_change_allowed"] is False
    assert rule["no_view_change_allowed"] is False
    assert rule["any_value_difference_allowed"] is False


def test_only_tail_diffusion_object_to_float64_is_canonicalized() -> None:
    config = load_config(DEFAULT_CONFIG)
    rebuilt = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "tail_diffusion5": pd.Series([0.25, None], dtype=object),
            "F": pd.Series([0.4, 0.5], dtype="float64"),
        }
    )
    stored = rebuilt.copy()
    stored["tail_diffusion5"] = pd.to_numeric(
        stored["tail_diffusion5"], errors="coerce"
    ).astype("float64")
    normalized, applied = canonicalize_declared_persisted_dtypes(
        output_name="internal_features",
        rebuilt=rebuilt,
        stored=stored,
        correction=config,
    )
    pd.testing.assert_frame_equal(normalized, stored, check_exact=True)
    assert applied == [
        {"column": "tail_diffusion5", "from": "object", "to": "float64"}
    ]


def test_undeclared_dtype_difference_still_fails_closed() -> None:
    config = load_config(DEFAULT_CONFIG)
    rebuilt = pd.DataFrame(
        {
            "tail_diffusion5": pd.Series([0.25], dtype=object),
            "F": pd.Series([1], dtype="int64"),
        }
    )
    stored = pd.DataFrame(
        {
            "tail_diffusion5": pd.Series([0.25], dtype="float64"),
            "F": pd.Series([1.0], dtype="float64"),
        }
    )
    with pytest.raises(EvidenceContractError, match="未声明 dtype 差异"):
        canonicalize_declared_persisted_dtypes(
            output_name="internal_features",
            rebuilt=rebuilt,
            stored=stored,
            correction=config,
        )


def test_affected_persisted_outputs_match_declared_identity_and_semantics() -> None:
    config = load_config(DEFAULT_CONFIG)
    for contract in config["affected_outputs"].values():
        path = ROOT / contract["path"]
        assert path.stat().st_size == contract["bytes"]
        assert sha256_file(path) == contract["sha256"]
        frame = pd.read_parquet(path)
        assert str(frame["tail_diffusion5"].dtype) == "float64"
        assert frame_semantic_sha256(frame) == contract["persisted_semantic_sha256"]
        assert contract["persisted_semantic_sha256"] != (
            contract["original_prewrite_semantic_sha256"]
        )


def test_original_failed_replay_outputs_remain_absent() -> None:
    assert not (
        ROOT
        / "reports/audit/510300_stress_transmission_hazard_v2_mft_clean_replay_v1.json"
    ).exists()
    assert not (
        ROOT / "reports/research/510300_stress_transmission_hazard_v2_g0_status_v1.json"
    ).exists()


def test_correction_manifest_verifies_when_present() -> None:
    config = load_config(DEFAULT_CONFIG)
    manifest_path = ROOT / config["freeze_contract"]["manifest_output"]
    if not manifest_path.exists():
        return
    manifest = verify_frozen_manifest(DEFAULT_CONFIG)
    assert manifest["correction_id"] == config["program"]["correction_id"]
    assert manifest["actual_label_artifact_read"] is False
    assert manifest["actual_performance_artifact_read"] is False
    assert manifest["position_impact"] == 0
