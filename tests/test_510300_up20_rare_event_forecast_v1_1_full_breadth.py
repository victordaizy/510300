from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from up20_rare_event_forecast_v1_1_full_breadth import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    PROJECT_ID,
    load_config,
    reconstruct_spliced_price_panel,
    validate_manifest,
)
from up20_rare_event_forecast_v1 import load_config as load_parent_config  # noqa: E402


def test_v1_1_changes_only_breadth_coverage() -> None:
    config = load_config()
    parent = load_parent_config()
    assert config["protocol"]["project_id"] == PROJECT_ID
    assert config["protocol"]["parent_v1_result_known"] is True
    assert config["protocol"]["parent_v1_rejection_preserved"] is True
    assert config["model"] == parent["model"]
    assert config["decision_rule"] == parent["decision_rule"]
    assert config["evaluation"] == parent["evaluation"]
    assert config["historical_acceptance_gates"] == parent[
        "historical_acceptance_gates"
    ]
    assert config["feature_modules"] == parent["feature_modules"]


def test_splice_uses_native_returns_not_raw_level_jump() -> None:
    dates = pd.Series(pd.bdate_range("2020-01-02", periods=8))
    external = pd.DataFrame(
        {
            "date": dates.iloc[:5],
            "con_code": "000001.SZ",
            "total_return_close": [100.0, 101.0, 102.01, 103.0301, 104.060401],
        }
    )
    current = pd.DataFrame(
        {
            "date": dates.iloc[2:],
            "con_code": "000001.SZ",
            "total_return_close": [500.0, 505.0, 510.05, 515.1505, 520.302005, 525.50502505],
        }
    )
    config = load_config()
    config["dates"]["evaluation_end"] = str(dates.iloc[-1].date())
    config["price_splice_contract"]["external_return_last_date"] = str(
        dates.iloc[4].date()
    )
    config["price_splice_contract"]["current_return_first_date"] = str(
        dates.iloc[5].date()
    )
    config["feature_data_contract"]["expected_external_panel_rows"] = len(external)
    config["feature_data_contract"]["expected_external_valid_panel_rows"] = len(external)
    config["feature_data_contract"][
        "expected_external_invalid_price_rows_before_required_from"
    ] = 0
    config["feature_data_contract"]["external_price_validity_required_from"] = str(
        dates.iloc[0].date()
    )
    config["feature_data_contract"]["expected_external_panel_dates"] = len(external)
    config["feature_data_contract"]["expected_external_first_date"] = str(
        dates.iloc[0].date()
    )
    config["feature_data_contract"]["expected_external_last_date"] = str(
        dates.iloc[4].date()
    )
    config["feature_data_contract"]["expected_current_rows_through_cutoff"] = len(
        current
    )
    config["feature_data_contract"]["expected_current_valid_rows_through_cutoff"] = len(
        current
    )
    config["feature_data_contract"][
        "expected_current_invalid_price_rows_before_required_from"
    ] = 0
    config["feature_data_contract"]["current_price_validity_required_from"] = str(
        dates.iloc[2].date()
    )
    config["feature_data_contract"]["expected_current_dates_through_cutoff"] = len(
        current
    )
    config["feature_data_contract"]["minimum_overlap_native_return_correlation"] = 0.99
    config["feature_data_contract"][
        "maximum_overlap_median_absolute_return_difference"
    ] = 1e-12
    config["feature_data_contract"][
        "maximum_overlap_p99_absolute_return_difference"
    ] = 1e-12
    config["feature_data_contract"]["minimum_overlap_share_within_one_basis_point"] = 1.0
    reconstructed, diagnostics = reconstruct_spliced_price_panel(
        external, current, dates, config
    )
    returns = reconstructed["native_return_1d"].dropna().to_numpy()
    assert np.allclose(returns, 0.01, atol=1e-12, rtol=0.0)
    assert diagnostics["raw_price_level_splice_used"] is False


def test_external_membership_and_weights_are_never_admitted() -> None:
    config = load_config()
    assert config["additional_inputs"]["external_constituent_price_panel"][
        "external_membership_flag_used"
    ] is False
    assert config["additional_inputs"]["external_constituent_price_panel"][
        "external_weights_used"
    ] is False
    assert config["price_splice_contract"]["external_membership_used"] is False
    assert config["price_splice_contract"]["current_panel_membership_used"] is False


def test_governance_stays_research_only() -> None:
    config = load_config()
    for key in (
        "paper_signal_allowed",
        "shadow_signal_allowed",
        "position_mapping_enabled",
        "order_generation",
        "broker_connection",
        "position_change",
        "live_trading_authorized",
    ):
        assert config["governance"][key] is False


def test_manifest_integrity_when_frozen() -> None:
    if not MANIFEST_PATH.exists():
        pytest.skip("V1.1首次冻结前尚无清单")
    manifest = validate_manifest(load_config(CONFIG_PATH))
    assert manifest["state"] == "FROZEN_BEFORE_NEW_MODEL_OUTCOME_CALCULATION"
    assert manifest["parent_v1_result_known"] is True
    assert manifest["new_model_outcomes_computed_before_freeze"] is False


def test_result_contract_when_written() -> None:
    path = ROOT / "reports" / "research" / (
        "510300_up20_rare_event_forecast_v1_1_full_breadth_result.json"
    )
    if not path.exists():
        pytest.skip("V1.1正式结果尚未写入")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["only_change_vs_parent"][
        "full_period_official_pit_equal_weight_breadth"
    ] is True
    assert report["only_change_vs_parent"]["model_changed"] is False
    assert report["only_change_vs_parent"]["external_membership_used"] is False
    assert report["parent_v1_comparison"]["result_unchanged"] is True
    assert report["adjudication"]["verified_forward_observations"] == 0
    assert report["adjudication"]["goal_achieved"] is False
    assert report["governance"]["live_trading_authorized"] is False
