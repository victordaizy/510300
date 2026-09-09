from __future__ import annotations

from pathlib import Path

from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_sources_v1 import (
    DEFAULT_CONFIG as V1_CONFIG,
    load_config as load_v1_config,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_sources_v1_0_1 import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
    verify_frozen_manifest,
)
from scripts.probe_510300_stress_transmission_hazard_v2_four_state_sources_v1_0_1 import (
    effective_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_correction_is_only_optional_dividend_schema() -> None:
    correction = load_config(DEFAULT_CONFIG)
    validate_config(correction)
    schema = correction["schema_correction"]
    assert schema["source"] == "dividend"
    assert schema["optional_fields"] == ["base_date", "base_share"]
    assert "base_date" not in schema["required_fields_after_correction"]
    assert "base_share" not in schema["required_fields_after_correction"]
    assert correction["program"]["return_evaluation"] == "NOT_ALLOWED"
    assert correction["program"]["position_impact"] == 0


def test_effective_config_changes_no_probe_scope_or_safety_boundary() -> None:
    base = load_v1_config(V1_CONFIG)
    correction = load_config(DEFAULT_CONFIG)
    effective = effective_config(correction)
    for field in (
        "daily_symbol",
        "daily_start",
        "daily_end",
        "suspension_trade_date",
        "suspension_type",
        "dividend_symbol",
        "adj_factor_symbol",
        "adj_factor_start",
        "adj_factor_end",
    ):
        assert effective["probe_contract"][field] == base["probe_contract"][field]
    assert effective["provider_contract"]["suspend_d"][
        "official_exchange_evidence_eligible"
    ] is False
    assert effective["provider_contract"]["suspend_d"][
        "zero_return_authorization"
    ] is False
    assert effective["provider_contract"]["adj_factor"][
        "direct_return_construction_allowed"
    ] is False
    assert effective["probe_contract"]["full_history_or_bulk_member_download_allowed"] is False


def test_frozen_correction_manifest_verifies_when_present() -> None:
    correction = load_config(DEFAULT_CONFIG)
    manifest_path = ROOT / correction["freeze_contract"]["manifest_output"]
    if not manifest_path.exists():
        return
    manifest = verify_frozen_manifest(DEFAULT_CONFIG)
    assert manifest["correction_id"] == correction["program"]["correction_id"]
    assert manifest["return_evaluation"] == "NOT_ALLOWED"
    assert manifest["portfolio_results_read"] is False
