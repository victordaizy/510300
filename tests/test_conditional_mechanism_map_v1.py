from __future__ import annotations

import numpy as np
import pandas as pd

from research.conditional_mechanism_map_v1 import (
    ARCHETYPE_FLAG_COLUMNS,
    assign_mechanism_archetypes,
    attach_total_return_context,
    build_archetype_conclusions,
    build_condition_state_panel,
    summarize_condition_map,
)


RC_INTERACTION_MAP = {
    "CF_IMPROVING_DR_EASING": {
        "RC_SUPPORTIVE": "AMPLIFIES_EXPANSION",
        "RC_CONSTRAINING": "WEAKENS_EXPANSION",
        "RC_NEUTRAL": "NEUTRAL_TO_EXPANSION",
    },
    "CF_IMPROVING_DR_TIGHTENING": {
        "RC_SUPPORTIVE": "WEAKENS_DISCOUNT_RATE_SQUEEZE",
        "RC_CONSTRAINING": "AMPLIFIES_DISCOUNT_RATE_SQUEEZE",
        "RC_NEUTRAL": "NEUTRAL_TO_DISCOUNT_RATE_SQUEEZE",
    },
    "CF_DETERIORATING_DR_EASING": {
        "RC_SUPPORTIVE": "AMPLIFIES_DISCOUNT_RATE_REPAIR",
        "RC_CONSTRAINING": "WEAKENS_DISCOUNT_RATE_REPAIR",
        "RC_NEUTRAL": "NEUTRAL_TO_DISCOUNT_RATE_REPAIR",
    },
    "CF_DETERIORATING_DR_TIGHTENING": {
        "RC_SUPPORTIVE": "WEAKENS_TRUE_CONTRACTION",
        "RC_CONSTRAINING": "AMPLIFIES_TRUE_CONTRACTION",
        "RC_NEUTRAL": "NEUTRAL_TO_TRUE_CONTRACTION",
    },
}


def _parent_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    origins = pd.to_datetime(
        [
            "2020-01-31",
            "2020-02-28",
            "2020-03-31",
            "2020-04-30",
            "2020-05-29",
            "2020-06-30",
        ]
    )
    cashflow = pd.DataFrame(
        {
            "origin": origins,
            "cf_level": [0.0, 1.0, 2.0, 1.0, 0.0, 1.0],
            "cf_direction": [np.nan, 1.0, 1.0, -1.0, -1.0, 1.0],
            "cf_acceleration": [np.nan, np.nan, 0.0, -2.0, 0.0, 2.0],
            "cf_breadth": [0.5] * 6,
            "cf_concentration": [0.1] * 6,
            "cf_state_status": ["PASS_EXPANDING_POINT_IN_TIME_CF_STATE"] * 6,
            "cf_data_status": [
                "PASS_PIT_CF_STATE_WITH_MARKET_CAP_PROXY_WEIGHT"
            ]
            * 6,
            "origin_completion_status": [
                "COMPLETE_CALENDAR_MONTH_LAST_TRADING_SESSION"
            ]
            * 6,
            "cash_flow_expectation_factor": [0.0] * 6,
        }
    )
    present_value = pd.DataFrame(
        {
            "origin": origins,
            "cross_sectional_expected_return_factor": [0.0] * 6,
            "equity_risk_premium_state": [0.05] * 6,
            "equity_risk_premium_expanding_z": [0.0, -1.0, 0.0, -1.0, 0.0, -1.0],
            "valuation_concentration": [0.1] * 6,
            "duration_compression": [0.0] * 6,
            "present_value_data_status": [
                "PASS_EXPLANATORY_CROSS_SECTIONAL_PRESENT_VALUE_STATE"
            ]
            * 6,
        }
    )
    risk_capacity = pd.DataFrame(
        {
            "date": origins,
            "risk_bearing_capacity_factor": [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            "rc_capacity_expanding_median": [0.0] * 6,
            "member_median_return_60d": [0.0] * 6,
            "member_positive_breadth_60d": [0.5] * 6,
            "large_weight_minus_median_member_return_60d": [0.0] * 6,
            "industry_negative_breadth_20d": [0.5] * 6,
            "rc_state_status": [
                "PASS_EXPLANATORY_RISK_BEARING_CAPACITY_FACTOR"
            ]
            * 6,
        }
    )
    return cashflow, present_value, risk_capacity


def test_condition_state_reuses_frozen_first_differences_and_maps_all_quadrants() -> None:
    cashflow, present_value, risk_capacity = _parent_frames()
    result = build_condition_state_panel(
        cashflow,
        present_value,
        risk_capacity,
        asof_tolerance_calendar_days=10,
        numerical_zero_tolerance=1.0e-12,
        rc_interaction_map=RC_INTERACTION_MAP,
    )

    assert result.loc[1:5, "cf_dr_quadrant"].tolist() == [
        "CF_IMPROVING_DR_EASING",
        "CF_IMPROVING_DR_TIGHTENING",
        "CF_DETERIORATING_DR_EASING",
        "CF_DETERIORATING_DR_TIGHTENING",
        "CF_IMPROVING_DR_EASING",
    ]
    assert result.loc[1:5, "rc_direction"].tolist() == [
        "RC_SUPPORTIVE",
        "RC_CONSTRAINING",
        "RC_SUPPORTIVE",
        "RC_CONSTRAINING",
        "RC_NEUTRAL",
    ]
    assert result.loc[1, "rc_interaction_role"] == "AMPLIFIES_EXPANSION"
    assert result.loc[4, "rc_interaction_role"] == "AMPLIFIES_TRUE_CONTRACTION"
    assert result.loc[0, "condition_quality_status"].startswith("NO_VIEW")
    assert result.loc[1:, "condition_quality_status"].eq(
        "PASS_FROZEN_PARENT_STATES"
    ).all()
    assert not result["future_return_values_read"].any()


def test_condition_state_propagates_parent_partial_status() -> None:
    cashflow, present_value, risk_capacity = _parent_frames()
    cashflow.loc[2, "cf_data_status"] = (
        "PARTIAL_PIT_CF_STATE_COVERAGE_BELOW_ONE_OR_MORE_GATES"
    )
    result = build_condition_state_panel(
        cashflow,
        present_value,
        risk_capacity,
        asof_tolerance_calendar_days=10,
        numerical_zero_tolerance=1.0e-12,
        rc_interaction_map=RC_INTERACTION_MAP,
    )

    assert result.loc[2, "condition_quality_status"].startswith("PARTIAL")
    assert result.loc[3, "condition_quality_status"].startswith("PARTIAL")
    assert result.loc[2, "condition_evaluation_eligible"]


def test_total_return_context_uses_exact_market_day_offsets_and_censors_tail() -> None:
    dates = pd.bdate_range("2020-01-01", periods=220)
    closes = np.arange(100.0, 320.0)
    total_return = pd.DataFrame({"date": dates, "close": closes})
    condition = pd.DataFrame(
        {
            "origin": [dates[120], dates[200]],
            "future_return_values_read": [False, False],
        }
    )
    result = attach_total_return_context(
        condition,
        total_return,
        horizons_market_days=[20, 60, 120],
        trailing_horizons_market_days=[20, 60],
        observation_cutoff=dates[-1],
    )

    expected_trailing = closes[120] / closes[60] - 1.0
    expected_forward = closes[180] / closes[120] - 1.0
    assert result.loc[0, "trailing_total_return_60d"] == expected_trailing
    assert result.loc[0, "forward_total_return_60d"] == expected_forward
    assert pd.isna(result.loc[1, "forward_total_return_20d"])
    assert result.loc[1, "forward_total_return_20d_status"] == "CENSORED_NO_VIEW"
    assert result["future_return_values_read"].all()


def test_archetype_rules_are_fixed_and_nonexclusive() -> None:
    frame = pd.DataFrame(
        {
            "cf_dr_quadrant": [
                "CF_IMPROVING_DR_EASING",
                "CF_IMPROVING_DR_EASING",
                "CF_IMPROVING_DR_EASING",
                "CF_DETERIORATING_DR_TIGHTENING",
                "CF_IMPROVING_DR_TIGHTENING",
            ],
            "rc_direction": [
                "RC_SUPPORTIVE",
                "RC_SUPPORTIVE",
                "RC_NEUTRAL",
                "RC_CONSTRAINING",
                "RC_CONSTRAINING",
            ],
            "condition_quality_status": ["PASS_FROZEN_PARENT_STATES"] * 5,
            "equity_risk_premium_expanding_z": [0.5, 0.5, 0.5, -0.5, -0.5],
            "trailing_total_return_20d": [0.02, -0.03, 0.01, -0.10, -0.08],
            "trailing_total_return_60d": [0.10, -0.10, 0.03, -0.20, -0.15],
        }
    )
    result = assign_mechanism_archetypes(
        frame, numerical_zero_tolerance=1.0e-12
    )

    assert result.loc[0, "is_momentum_fundamental_support"]
    assert result.loc[0, "is_value_safety_margin"]
    assert result.loc[1, "is_reversal_economic_basis"]
    assert result.loc[2, "is_value_safety_margin"]
    assert result.loc[3, "is_true_contraction_decline"]
    assert result.loc[4, "is_temporary_liquidity_shock_candidate"]


def test_summary_reliability_gate_and_scope_conclusion() -> None:
    origins = pd.to_datetime(
        [
            "2019-01-31",
            "2019-04-30",
            "2019-07-31",
            "2019-10-31",
            "2020-01-31",
            "2020-04-30",
            "2020-07-31",
            "2020-10-30",
            "2021-01-29",
            "2021-04-30",
            "2021-07-30",
            "2021-10-29",
        ]
    )
    frame = pd.DataFrame(
        {
            "origin": origins,
            "cf_dr_quadrant": ["CF_IMPROVING_DR_EASING"] * 12,
            "rc_direction": ["RC_SUPPORTIVE"] * 12,
            "condition_quality_status": ["PASS_FROZEN_PARENT_STATES"] * 12,
            "forward_total_return_60d": np.linspace(0.01, 0.12, 12),
        }
    )
    for flag in ARCHETYPE_FLAG_COLUMNS.values():
        frame[flag] = flag == "is_momentum_fundamental_support"
    archetypes = {
        "MOMENTUM_FUNDAMENTAL_SUPPORT": {
            "frozen_definition": "TOY_FIXED_RULE",
            "expected_forward_direction": "POSITIVE",
            "primary_evaluation_horizon_market_days": 60,
        }
    }
    gate = {
        "pass_minimum_observed_origins": 12,
        "partial_minimum_observed_origins": 6,
        "pass_minimum_distinct_calendar_years": 3,
        "partial_minimum_distinct_calendar_years": 2,
        "minimum_forward_observation_coverage": 0.80,
        "minimum_parent_state_pass_share": 0.80,
    }
    scopes = {
        "FULL": {"start": "2019-01-01", "end": "2021-12-31"},
        "RECENT": {"start": "2020-01-01", "end": "2021-12-31"},
    }
    summary = summarize_condition_map(
        frame,
        horizons_market_days=[60],
        sample_scopes=scopes,
        reliability_gate=gate,
        mechanism_archetypes=archetypes,
    )
    full = summary.loc[
        summary["sample_scope"].eq("FULL")
        & summary["view_type"].eq("ARCHETYPE")
        & summary["condition_id"].eq("MOMENTUM_FUNDAMENTAL_SUPPORT")
    ].iloc[0]
    assert full["reliability_status"] == "PASS_DESCRIPTIVE_COVERAGE"
    assert full["directional_result"] == "ALIGNED_WITH_EXPECTED_DIRECTION"

    conclusions = build_archetype_conclusions(
        summary,
        mechanism_archetypes=archetypes,
        full_scope_id="FULL",
        user_scope_id="RECENT",
    )
    assert conclusions[0]["conclusion"] == (
        "PARTIAL_DIRECTIONAL_SUPPORT_NOT_FULLY_RELIABLE"
    )
