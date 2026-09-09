"""V2 G1 历史覆盖审计 V1 的冻结边界测试。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from research.stress_transmission_hazard_v2_g1_historical_coverage_audit_v1 import (
    FORBIDDEN_OUTPUT_TOKENS,
    HistoricalCoverageAuditError,
    build_historical_coverage_audit,
    cumulative_tail_deficit_upper_bounds,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_g1_historical_coverage_audit_v1 import (
    DEFAULT_CONFIG,
    load_config,
    validate_config,
    verify_frozen_manifest,
)


def _synthetic_inputs() -> dict[str, pd.DataFrame | str]:
    all_dates = pd.bdate_range("2019-10-01", periods=80)
    member_dates = all_dates[60:]
    symbols = ["000001.SZ", "600000.SH"]

    classified_rows: list[dict[str, object]] = []
    for date in all_dates:
        for symbol in symbols:
            classified_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "source_observed": True,
                    "supplier_conflict": False,
                    "official_suspension": False,
                    "suspension_evidence_id": "",
                    "corporate_action_status": "NONE_CONFIRMED",
                    "constituent_return_state": "TRADED_VALID",
                    "daily_total_shareholder_return": 0.001,
                    "return_is_usable": True,
                    "state_reason": "SYNTHETIC_TEST_TRADED_VALID",
                }
            )
    membership_rows = [
        {
            "membership_date": date,
            "index_code": "000300",
            "symbol": symbol,
        }
        for date in member_dates
        for symbol in symbols
    ]
    coverage_rows: list[dict[str, object]] = []
    for position, date in enumerate(member_dates):
        return20_count = 2 if position >= 19 else 0
        coverage_rows.append(
            {
                "date": date,
                "point_in_time_member_count": 2,
                "return20_scoreable_member_count": return20_count,
                "tail_scoreable_member_count": 0,
                "comovement_scoreable_member_ratio": 1.0,
                "four_state_daily_coverage_state": "VIEW_ALLOWED",
                "internal_formula_coverage_state": "NO_VIEW",
                "internal_feature_state": "NO_VIEW",
                "F_state": "NO_VIEW",
                "M_state": "VIEW_ALLOWED",
                "T_state": "NO_VIEW",
                "B2_feature_state": "NO_VIEW",
                "B3_feature_state": "NO_VIEW",
                "NO_VIEW_REASON": "INTERNAL_20D_60D_OR_COMOVEMENT_COVERAGE_FAILED",
            }
        )
    origin = member_dates[0]
    sample = pd.DataFrame(
        [
            {
                "origin_date": origin,
                "bad10": 1,
                "event_id": "BAD10_V2_EVENT_TEST_0001",
                "B1_state": "VIEW_ALLOWED",
                "B2_feature_state": "NO_VIEW",
                "B3_feature_state": "NO_VIEW",
                "b1_eligible": True,
                "b2_vs_b1_eligible": False,
                "b3_vs_b2_eligible": False,
                "b2_vs_b1_no_view_reason": "B2_FEATURE_NO_VIEW",
                "b3_vs_b2_no_view_reason": "B2_FEATURE_NO_VIEW;B3_FEATURE_NO_VIEW",
            }
        ]
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "BAD10_V2_EVENT_TEST_0001",
                "total_positive_origin_count": 1,
                "b2_eligible_positive_origin_count": 0,
                "b2_identifiable_event": False,
                "b3_eligible_positive_origin_count": 0,
                "b3_identifiable_event": False,
            }
        ]
    )
    return {
        "sample_eligibility": sample,
        "event_eligibility": events,
        "mft_coverage": pd.DataFrame(coverage_rows),
        "classified_returns": pd.DataFrame(classified_rows),
        "membership": pd.DataFrame(membership_rows),
        "historical_cutoff": all_dates[-1].date().isoformat(),
    }


def _build_synthetic(**overrides: object):
    arguments = _synthetic_inputs()
    arguments.update(overrides)
    return build_historical_coverage_audit(
        **arguments,
        expected_members_per_day=2,
    )


def test_prehistory_is_used_only_for_individual_return_windows() -> None:
    artifacts = _build_synthetic()
    first = artifacts.prehistory_coverage_comparison.iloc[0]
    assert int(first["recomputed_current_tail_scoreable_member_count"]) == 0
    assert int(first["prehistory_inclusive_tail_scoreable_member_count"]) == 2
    assert int(first["prehistory_tail_member_delta"]) == 2
    assert bool(first["prehistory_tail_gate_pass"])
    assert artifacts.metrics["implementation_protocol_mismatch_confirmed"] is True
    assert artifacts.metrics[
        "prehistory_price_and_frozen_comovement_diagnostic_event_upper_bound"
    ] == 1
    assert artifacts.audit_state["G1_DATA_AND_EVENTS"] == (
        "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY"
    )
    assert artifacts.audit_state["branch_state"] == (
        "STOPPED_AT_G1_POST_G1_DIAGNOSTIC_ONLY"
    )
    assert artifacts.member_window_gap_ledger["primary_gap_reason"].eq(
        "IMPLEMENTATION_DISCARDED_ADMITTED_PRE_OBSERVATION_HISTORY"
    ).all()


def test_no_forbidden_prediction_or_performance_columns_are_generated() -> None:
    artifacts = _build_synthetic()
    for frame in (
        artifacts.prehistory_coverage_comparison,
        artifacts.event_gap_ledger,
        artifacts.member_window_gap_ledger,
    ):
        assert not FORBIDDEN_OUTPUT_TOKENS.intersection(
            {str(column).casefold() for column in frame.columns}
        )
    assert artifacts.metrics["model_trained"] is False
    assert artifacts.metrics["prediction_metric_generated"] is False
    assert artifacts.metrics["performance_artifact_read"] is False
    assert artifacts.metrics["g1_gate_promoted"] is False


def test_date_after_historical_cutoff_fails_closed() -> None:
    arguments = _synthetic_inputs()
    classified = arguments["classified_returns"].copy()
    classified.loc[classified.index[-1], "date"] = pd.Timestamp(
        arguments["historical_cutoff"]
    ) + pd.Timedelta(days=1)
    arguments["classified_returns"] = classified
    with pytest.raises(HistoricalCoverageAuditError, match="历史截止日之后"):
        build_historical_coverage_audit(
            **arguments,
            expected_members_per_day=2,
        )


def test_future_path_column_fails_closed_even_if_not_used() -> None:
    arguments = _synthetic_inputs()
    sample = arguments["sample_eligibility"].copy()
    sample["minimum_path_return"] = -0.05
    arguments["sample_eligibility"] = sample
    with pytest.raises(HistoricalCoverageAuditError, match="禁止读取列"):
        build_historical_coverage_audit(
            **arguments,
            expected_members_per_day=2,
        )


def test_frozen_windows_and_thresholds_cannot_be_changed() -> None:
    with pytest.raises(HistoricalCoverageAuditError, match="20/5/60"):
        _build_synthetic(tail_days=59)
    with pytest.raises(HistoricalCoverageAuditError, match="98%"):
        _build_synthetic(member_coverage_minimum=0.97)
    with pytest.raises(HistoricalCoverageAuditError, match="90%"):
        _build_synthetic(comovement_member_ratio_minimum=0.89)


def test_tail_deficit_upper_bound_is_explicit_and_deterministic() -> None:
    ledger = pd.DataFrame(
        {
            "current_b2_identifiable_event": [True, False, False, False],
            "tail_member_deficit": [0, 1, 4, 10],
        }
    )
    assert cumulative_tail_deficit_upper_bounds(ledger, [10, 0, 4, 1]) == {
        "0": 1,
        "1": 2,
        "4": 3,
        "10": 4,
    }


def test_current_config_is_valid_and_frozen_manifest_verifies_if_present() -> None:
    config = load_config(DEFAULT_CONFIG)
    validate_config(config)
    manifest_path = Path(config["freeze_contract"]["manifest_output"])
    if manifest_path.is_file():
        verify_frozen_manifest(DEFAULT_CONFIG)
