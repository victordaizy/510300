from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.return_tail_data_feasibility import (
    audit_earnings_consensus,
    audit_options,
    validate_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = yaml.safe_load(
    (ROOT / "config" / "return_tail_hypothesis_registry.yaml").read_text(
        encoding="utf-8"
    )
)


def test_registry_has_exactly_nine_frozen_configurations() -> None:
    evidence = validate_registry(REGISTRY)
    assert evidence["candidate_count"] == 9
    assert evidence["candidate_ids"] == [
        "O1",
        "O2",
        "O3",
        "O4",
        "X1",
        "E1",
        "C1_EQUAL_PROBABILITY",
        "C2_SIGN_CONSTRAINED_LOGISTIC",
        "P1_100_50_HYSTERESIS",
    ]


def test_governance_keeps_all_execution_capabilities_disabled() -> None:
    governance = REGISTRY["governance"]
    assert governance["position_mapping_enabled"] is False
    assert governance["order_generation_enabled"] is False
    assert governance["broker_connection_enabled"] is False
    assert governance["live_trading_authorized"] is False
    assert REGISTRY["protocol"]["true_forward_start"] is None


def test_missing_option_chain_blocks_all_option_return_tests(tmp_path: Path) -> None:
    result = audit_options(tmp_path, REGISTRY["data_contracts"]["options"])
    assert result["status"] == "BLOCKED_MISSING_LOCAL_OPTION_HISTORY"
    assert result["return_test_allowed"] is False
    assert "quote" in result["missing_primary_inputs"]
    assert "contract_master" in result["missing_primary_inputs"]


def test_actual_financials_do_not_substitute_for_consensus_vintages(
    tmp_path: Path,
) -> None:
    actual_path = (
        tmp_path
        / "data"
        / "raw"
        / "fundamentals"
        / "csi300_financials_point_in_time.parquet"
    )
    actual_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "con_code": ["600000.SH"],
            "announcement_date": [pd.Timestamp("2026-01-01")],
            "eps": [1.0],
        }
    ).to_parquet(actual_path, index=False)
    result = audit_earnings_consensus(
        tmp_path, REGISTRY["data_contracts"]["earnings_consensus"]
    )
    assert result["status"] == "TERMINATED_NO_POINT_IN_TIME_CONSENSUS_HISTORY"
    assert result["return_test_allowed"] is False
    assert result["files"]["actual_company_financials_non_substitute"]["exists"]


def test_position_policy_threshold_is_economically_anchored() -> None:
    policy = REGISTRY["position_policy"]
    threshold = (
        policy["assumed_false_exit_loss"]
        + policy["assumed_round_trip_cost_per_unit"]
    ) / (policy["assumed_avoidable_loss"] + policy["assumed_false_exit_loss"])
    assert threshold == pytest.approx(0.395)
    assert policy["exit_probability"] == 0.40
    assert policy["reentry_probability"] < policy["exit_probability"]
