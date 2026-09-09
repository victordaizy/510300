from __future__ import annotations

import pandas as pd

from research.expectations_news_price_response_engine_v1 import (
    DEFAULT_NEUTRAL_QUADRANT_MAP,
    build_neutral_atlas,
    build_neutral_state_panel,
    build_permanent_rejection_receipt,
    neutralize_condition_id,
)


def _parent_state() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    quadrants = list(DEFAULT_NEUTRAL_QUADRANT_MAP)
    origins = pd.date_range("2021-01-31", periods=len(quadrants), freq="ME")
    for index, quadrant in enumerate(quadrants):
        rows.append(
            {
                "origin": origins[index],
                "origin_completion_status": "COMPLETE_CALENDAR_MONTH_LAST_TRADING_SESSION",
                "cf_level": 0.1 + index,
                "cf_direction": 1.0,
                "cf_acceleration": 0.0,
                "cf_breadth": 0.5,
                "cf_concentration": 0.1,
                "cash_flow_expectation_factor": 0.2,
                "cf_news": 0.1,
                "cf_news_sign": "UP",
                "cf_state_status": "PASS_EXPANDING_POINT_IN_TIME_CF_STATE",
                "cf_data_status": "PASS_PIT_CF_STATE_WITH_MARKET_CAP_PROXY_WEIGHT",
                "cross_sectional_expected_return_factor": 0.0,
                "equity_risk_premium_state": 0.05,
                "equity_risk_premium_expanding_z": 0.1,
                "discount_rate_easing_news": 0.1,
                "discount_rate_easing_news_sign": "UP",
                "valuation_concentration": 0.1,
                "duration_compression": 0.0,
                "present_value_data_status": "PASS_EXPLANATORY_CROSS_SECTIONAL_PRESENT_VALUE_STATE",
                "rc_asof_date": origins[index],
                "rc_asof_age_calendar_days": 0,
                "risk_bearing_capacity_factor": 0.1,
                "risk_capacity_news": 0.1,
                "risk_capacity_news_sign": "UP",
                "rc_capacity_expanding_median": 0.0,
                "member_median_return_60d": 0.01,
                "member_positive_breadth_60d": 0.6,
                "large_weight_minus_median_member_return_60d": 0.0,
                "industry_negative_breadth_20d": 0.4,
                "rc_state_status": "PASS_EXPLANATORY_RISK_BEARING_CAPACITY_FACTOR",
                "cf_dr_quadrant": quadrant,
                "rc_direction": "RC_SUPPORTIVE",
                "condition_quality_status": "PASS_FROZEN_PARENT_STATES",
                "condition_evaluation_eligible": True,
                "rc_interaction_evaluation_eligible": True,
            }
        )
    return pd.DataFrame(rows)


def test_neutral_condition_id_maps_plain_and_joint_states() -> None:
    assert neutralize_condition_id("CF_IMPROVING_DR_EASING") == "CF_UP_DR_EASING"
    assert neutralize_condition_id(
        "CF_DETERIORATING_DR_TIGHTENING__RC_CONSTRAINING"
    ) == "CF_DOWN_DR_TIGHTENING__RC_CONSTRAINING"


def test_neutral_state_panel_has_no_return_target_or_old_semantic_labels() -> None:
    result = build_neutral_state_panel(_parent_state())

    assert result["state_quadrant"].tolist() == list(
        DEFAULT_NEUTRAL_QUADRANT_MAP.values()
    )
    assert result["return_target_present"].eq(False).all()
    assert result["return_prediction_allowed"].eq(False).all()
    assert result["portfolio_evaluation_allowed"].eq(False).all()
    assert result["model_position_target"].eq("UNSET").all()
    assert "mechanism_state" not in result.columns
    assert "rc_interaction_role" not in result.columns
    text = result.astype(str).to_string()
    assert "CF_IMPROVING" not in text
    assert "CF_DETERIORATING" not in text


def test_neutral_atlas_excludes_archetypes_and_direction_fields() -> None:
    parent = pd.DataFrame(
        {
            "sample_scope": ["FULL", "FULL", "FULL"],
            "view_type": ["QUADRANT", "QUADRANT_X_RC", "ARCHETYPE"],
            "condition_id": [
                "CF_IMPROVING_DR_EASING",
                "CF_DETERIORATING_DR_TIGHTENING__RC_CONSTRAINING",
                "MOMENTUM_FUNDAMENTAL_SUPPORT",
            ],
            "horizon_market_days": [60, 60, 60],
            "condition_origin_count": [12, 10, 8],
            "observed_origin_count": [12, 10, 8],
            "censored_origin_count": [0, 0, 0],
            "forward_observation_coverage": [1.0, 1.0, 1.0],
            "parent_state_pass_share": [1.0, 1.0, 1.0],
            "distinct_calendar_years": [4, 4, 3],
            "first_observed_origin": [pd.Timestamp("2021-01-29")] * 3,
            "last_observed_origin": [pd.Timestamp("2024-12-31")] * 3,
            "mean_total_return": [0.01, -0.01, 0.02],
            "median_total_return": [0.01, -0.01, 0.02],
            "positive_rate": [0.6, 0.4, 0.7],
            "loss_rate": [0.4, 0.6, 0.3],
            "q10_total_return": [-0.1, -0.2, -0.1],
            "q25_total_return": [-0.05, -0.1, 0.0],
            "q75_total_return": [0.05, 0.1, 0.1],
            "q90_total_return": [0.1, 0.2, 0.2],
            "minimum_total_return": [-0.2, -0.3, -0.1],
            "maximum_total_return": [0.2, 0.3, 0.3],
            "expected_direction": ["UNSPECIFIED", "UNSPECIFIED", "POSITIVE"],
            "directional_result": ["NONE", "NONE", "ALIGNED"],
            "reliability_status": ["PASS", "PARTIAL", "PASS"],
            "overlapping_forward_windows": [True, True, True],
        }
    )
    result = build_neutral_atlas(
        parent,
        canonical_source_views=["QUADRANT", "QUADRANT_X_RC"],
    )

    assert len(result) == 2
    assert "expected_direction" not in result.columns
    assert "directional_result" not in result.columns
    assert result["return_prediction_allowed"].eq(False).all()
    text = result.astype(str).to_string()
    assert "ARCHETYPE" not in text
    assert "MOMENTUM_FUNDAMENTAL_SUPPORT" not in text


def test_permanent_rejection_keeps_censored_forecasts_diagnostic_only() -> None:
    source_status = {
        "status": "STAGE_3_COMPLETE_PREDICTIVE_VALIDATION_FAILED_OR_NO_VIEW",
        "status_payload_sha256": "frozen-status-hash",
        "validation_summary": {"structural_prediction_validated": False},
        "latest_forecasts": [
            {
                "origin": "2026-07-31",
                "target_id": "EXPECTED_60D_TOTAL_RETURN",
                "model_id": "STRUCTURE_WITH_INTERACTIONS",
                "prediction": 0.1,
                "actual": None,
                "actual_status": "ACTUAL_CENSORED",
            }
        ],
    }
    adjudication = {
        "current_forecast_specification": "REJECTED_FROZEN",
        "structural_economic_idea_status": "UNPROVEN_NOT_REJECTED_AS_MATHEMATICAL_IMPOSSIBILITY",
        "waiting_for_existing_labels": "PASSIVE_MONITORING_ONLY",
        "next_research_action": "REBUILD_MEASUREMENT_ARCHITECTURE",
        "portfolio_action": "ABSTAIN",
        "position_state": "POSITION_UNSET",
        "future_requalification_requires_new_program": "510300_STRUCTURAL_PREDICTION_V2",
        "mathematical_impossibility_proven": False,
        "accessible_free_information_feasibility_validated": False,
        "current_investable_strategy": "NONE",
        "existing_forecast_status": "REJECTED_MODEL_DIAGNOSTIC_FORECAST_ONLY",
        "existing_forecast_trading_status": "NOT_TRADABLE",
        "existing_forecast_origin": "2026-07-31",
    }
    result = build_permanent_rejection_receipt(source_status, adjudication)

    assert result["adjudication_status"] == "REJECTED_FROZEN_PERMANENT_NO_RESCUE"
    assert result["v1_requalification_allowed"] is False
    forecast = result["immutable_diagnostic_forecasts"][0]
    assert forecast["forecast_record_status"] == (
        "REJECTED_MODEL_DIAGNOSTIC_FORECAST_ONLY"
    )
    assert forecast["trading_status"] == "NOT_TRADABLE"
    assert forecast["actual"] is None
    assert forecast["actual_status"] == "ACTUAL_CENSORED"
