from __future__ import annotations

import pytest

from research.project_evidence_contract_v1 import EvidenceContractError
from scripts.correct_510300_stress_transmission_hazard_v2_g1_historical_remediation_status_v1_0_1 import (
    derive_corrected_branch_state,
)


def test_mechanism_only_pass_is_not_collapsed_to_total_failure() -> None:
    result = derive_corrected_branch_state(
        {
            "G1_DATA_AND_EVENTS": "PASS_B2_MECHANISM_ONLY_B3_NOT_IDENTIFIABLE",
            "mechanism_discovery_prerequisite_passed": True,
            "full_three_coefficient_model_prerequisite_passed": False,
            "next_allowed_step": "FREEZE_B2_VS_B1_EVENT_LEVEL_PREDICTION_EXECUTION",
        }
    )
    assert result["G1_DATA_AND_EVENTS"] == "PASS_B2_MECHANISM_ONLY_B3_NOT_IDENTIFIABLE"
    assert result["B2_MECHANISM_G1"] == "PASS_MECHANISM_DISCOVERY_PREREQUISITE"
    assert result["B3_FULL_MODEL_G1"] == "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY"
    assert result["next_allowed_step"] == "FREEZE_B2_VS_B1_EVENT_LEVEL_PREDICTION_EXECUTION"


def test_full_pass_requires_mechanism_pass() -> None:
    with pytest.raises(EvidenceContractError):
        derive_corrected_branch_state(
            {
                "G1_DATA_AND_EVENTS": "INVALID",
                "mechanism_discovery_prerequisite_passed": False,
                "full_three_coefficient_model_prerequisite_passed": True,
                "next_allowed_step": "INVALID",
            }
        )


def test_full_pass_preserves_both_branches() -> None:
    result = derive_corrected_branch_state(
        {
            "G1_DATA_AND_EVENTS": "PASS",
            "mechanism_discovery_prerequisite_passed": True,
            "full_three_coefficient_model_prerequisite_passed": True,
            "next_allowed_step": "FREEZE_B2_VS_B1_EVENT_LEVEL_PREDICTION_EXECUTION",
        }
    )
    assert result["G1_DATA_AND_EVENTS"] == "PASS_FULL_B2_AND_B3_EVENT_IDENTIFIABILITY"
    assert result["B2_MECHANISM_G1"] == "PASS_MECHANISM_DISCOVERY_PREREQUISITE"
    assert result["B3_FULL_MODEL_G1"] == "PASS_FULL_MODEL_PREREQUISITE"
