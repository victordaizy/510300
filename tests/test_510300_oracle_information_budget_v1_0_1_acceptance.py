from __future__ import annotations

import json
import math
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

import oracle_information_budget_v1 as base  # noqa: E402
from oracle_information_budget_v1_0_1_acceptance import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    _timing_boundary,
    load_config,
    simulate_dense_signals_with_conservative_metrics,
    validate_manifest,
)


def test_config_freezes_all_eight_required_fields_and_conservative_metric() -> None:
    config = load_config()
    assert config["frozen_measurement"]["conservative_scenario_sharpe_definition"] == (
        "MIN_OF_MONTHLY_20D_INTERVAL_AND_LO_HAC"
    )
    assert config["reported_budget_fields"] == [
        "MIN_UP_RECALL_FOR_SHARPE_1_2",
        "MIN_UP_PRECISION_FOR_SHARPE_1_2",
        "MAX_DOWN_FALSE_LONG_RATE",
        "MAX_RANGE_FALSE_LONG_RATE",
        "MAX_ENTRY_DELAY_DAYS",
        "MAX_EXIT_DELAY_DAYS",
        "MIN_ORACLE_RETURN_CAPTURE",
        "SHARPE_1_2_FEASIBLE_REGION",
    ]


def test_config_preserves_original_seed_grid_and_governance() -> None:
    config = load_config()
    measurement = config["frozen_measurement"]
    assert measurement["random_seed"] == 51030020260831
    assert measurement["repetitions_per_random_cell"] == 500
    assert measurement["joint_grid"]["captured_bull_counts"] == [4, 8, 12, 16, 20, 23]
    assert measurement["timing_error_days"] == list(range(11))
    assert config["atlas_governance"]["atlas_state"] == "REJECTED_FROZEN"
    assert config["atlas_governance"]["router_state"] == "SKIPPED_NOT_CREATED"
    assert config["preexisting_up20_protocol"]["accepted_as_authorized_successor"] is False
    assert config["governance"]["preexisting_up20_protocol_present"] is True
    assert config["governance"]["preexisting_up20_protocol_accepted"] is False
    assert config["governance"]["position_output"] == "NONE"
    assert config["governance"]["live_trading_authorized"] is False


def test_timing_boundary_reports_right_censoring_instead_of_false_point() -> None:
    timing = pd.DataFrame(
        {
            "family": ["LATE_EXIT"] * 3,
            "error_days": [0, 1, 2],
            "conservative_deterministic_budget_pass": [True, True, True],
        }
    )
    boundary = _timing_boundary(timing, "LATE_EXIT")
    assert boundary["point_estimate_days"] is None
    assert boundary["lower_bound_days"] == 2
    assert boundary["display"] == ">=2"
    assert boundary["boundary_status"] == "RIGHT_CENSORED_AT_TEST_GRID_MAXIMUM"


def test_conservative_engine_replays_perfect_oracle_and_serial_metrics() -> None:
    if not MANIFEST_PATH.exists():
        pytest.skip("V1.0.1首次冻结前尚无清单")
    acceptance = load_config()
    base_config = base.load_config()
    context = base.load_context(base_config)
    signal = base.build_baseline_signals(context)
    metrics = simulate_dense_signals_with_conservative_metrics(
        signal[None, :], context, base_config, acceptance
    ).iloc[0]
    source = json.loads(
        (ROOT / "reports/research/510300_oracle_information_budget_v1_result.json").read_text(
            encoding="utf-8"
        )
    )
    serial = source["robustness"]["serial_correlation"]
    assert math.isclose(
        float(metrics["stress_net_sharpe"]),
        float(serial["ordinary_daily_annualized_sharpe"]),
        rel_tol=0.0,
        abs_tol=5e-12,
    )
    assert math.isclose(
        float(metrics["monthly_compounded_return_sharpe"]),
        float(serial["monthly_compounded_return_sharpe"]),
        rel_tol=0.0,
        abs_tol=5e-12,
    )
    assert math.isclose(
        float(metrics["twenty_day_interval_return_sharpe"]),
        float(serial["twenty_day_interval_return_sharpe"]),
        rel_tol=0.0,
        abs_tol=5e-12,
    )
    assert math.isclose(
        float(metrics["lo_hac_adjusted_sharpe_q20"]),
        float(serial["lo_adjusted_annualized_sharpe"]),
        rel_tol=0.0,
        abs_tol=5e-12,
    )
    assert math.isclose(
        float(metrics["conservative_scenario_sharpe"]),
        min(
            float(serial["monthly_compounded_return_sharpe"]),
            float(serial["twenty_day_interval_return_sharpe"]),
            float(serial["lo_adjusted_annualized_sharpe"]),
        ),
        rel_tol=0.0,
        abs_tol=5e-12,
    )


def test_manifest_integrity_when_frozen() -> None:
    if not MANIFEST_PATH.exists():
        pytest.skip("V1.0.1首次冻结前尚无清单")
    manifest = validate_manifest(load_config(CONFIG_PATH))
    assert manifest["state"] == "FROZEN_BEFORE_CONSERVATIVE_SCENARIO_METRICS"
    assert manifest["source_v1_outputs_known"] is True
    assert manifest["source_robustness_audit_known"] is True
    assert manifest["conservative_scenario_metrics_computed_before_freeze"] is False


def test_result_contract_when_written() -> None:
    result_path = ROOT / "reports/research/510300_oracle_information_budget_v1_0_1_acceptance_result.json"
    if not result_path.exists():
        pytest.skip("V1.0.1正式结果尚未写入")
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert set(report["information_budget"]) >= {
        "MIN_UP_RECALL_FOR_SHARPE_1_2",
        "MIN_UP_PRECISION_FOR_SHARPE_1_2",
        "MAX_DOWN_FALSE_LONG_RATE",
        "MAX_RANGE_FALSE_LONG_RATE",
        "MAX_ENTRY_DELAY_DAYS",
        "MAX_EXIT_DELAY_DAYS",
        "MIN_ORACLE_RETURN_CAPTURE",
        "SHARPE_1_2_FEASIBLE_REGION",
    }
    assert report["adjudication"]["realistic_up20_forecast"] == "NOT_EVALUATED"
    assert report["adjudication"]["position_output"] == "NONE"
    assert report["adjudication"]["live_trading_authorized"] is False
