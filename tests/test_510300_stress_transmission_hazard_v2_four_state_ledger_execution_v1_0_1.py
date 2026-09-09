from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.acquire_510300_stress_transmission_hazard_v2_four_state_inputs_v1_0_1 import (
    effective_config,
    select_fresh_daily_symbols,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1_0_1 import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
    verify_frozen_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MEMBERSHIP = (
    ROOT
    / "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/"
    "000300_daily_pit_membership_20150101_20260814.parquet"
)


def test_selector_uses_uncovered_period_not_query_overlap_start() -> None:
    correction = load_config(DEFAULT_CONFIG)
    validate_config(correction)
    membership = pd.read_parquet(MEMBERSHIP)
    symbols = select_fresh_daily_symbols(
        membership,
        legacy_seed_last_date=correction["selector_correction"][
            "legacy_seed_last_date"
        ],
    )
    assert len(symbols) == 493
    assert not set(correction["selector_correction"]["excluded_overlap_only_symbols"]).intersection(symbols)


def test_excluded_overlap_only_symbols_are_fully_inside_legacy_period() -> None:
    correction = load_config(DEFAULT_CONFIG)
    membership = pd.read_parquet(MEMBERSHIP)
    membership["membership_date"] = pd.to_datetime(membership["membership_date"])
    excluded = set(correction["selector_correction"]["excluded_overlap_only_symbols"])
    rows = membership.loc[membership["symbol"].astype(str).isin(excluded)]
    assert rows["symbol"].nunique() == 12
    assert rows["membership_date"].max() <= pd.Timestamp(
        correction["selector_correction"]["legacy_seed_last_date"]
    )


def test_effective_config_changes_only_selector_and_output_namespace() -> None:
    correction = load_config(DEFAULT_CONFIG)
    effective = effective_config(correction)
    assert effective["acquisition"]["fresh_daily"]["expected_symbol_count"] == 493
    assert effective["acquisition"]["fresh_daily"]["start_date"] == "2019-12-01"
    assert effective["source_admission_dependencies"]["selected_endpoint"] == (
        "https://fast.xiaodefa.cn"
    )
    assert effective["acquisition"]["request_control"]["maximum_workers"] == 2
    assert effective["four_state_execution"]["blanket_zero_fill_allowed"] is False
    assert effective["program"]["return_evaluation"] == "NOT_ALLOWED"
    assert effective["program"]["position_impact"] == 0


def test_frozen_correction_manifest_verifies_when_present() -> None:
    correction = load_config(DEFAULT_CONFIG)
    path = ROOT / correction["freeze_contract"]["manifest_output"]
    if not path.exists():
        return
    manifest = verify_frozen_manifest(DEFAULT_CONFIG)
    assert manifest["correction_id"] == correction["program"]["correction_id"]
    assert manifest["return_evaluation"] == "NOT_ALLOWED"
    assert manifest["portfolio_results_read"] is False
