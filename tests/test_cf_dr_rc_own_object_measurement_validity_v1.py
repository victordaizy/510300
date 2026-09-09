from __future__ import annotations

import numpy as np
import pandas as pd

from research.cf_dr_rc_own_object_measurement_validity_v1 import (
    build_object_measurement_panel,
    summarize_measurement_validity,
)


def test_object_panel_contains_only_registered_own_objects() -> None:
    dates = pd.bdate_range("2021-01-04", periods=61)
    origin, target = dates[0], dates[-1]
    ledger = pd.DataFrame(
        {
            "origin": [origin],
            "horizon_market_days": [60],
            "target_date": [target],
            "scope": ["ORIGIN_FIXED_FINANCIAL_COVERAGE_COHORT_SCOPE"],
            "earnings_growth_component_factor": [1.10],
            "multiple_repricing_component_factor": [0.95],
            "ledger_status": ["PASS_FIXED_FINANCIAL_COHORT_COVERAGE"],
            "financial_weight_coverage": [0.98],
        }
    )
    coverage = pd.DataFrame(
        {
            "origin": [origin],
            "horizon_market_days": [60],
            "target_date": [target],
        }
    )
    cf = pd.DataFrame(
        {
            "origin": [origin, target],
            "weighted_operating_cashflow_yoy": [0.1, 0.2],
            "operating_cashflow_weight_coverage": [0.95, 0.96],
            "cf_breadth": [0.5, 0.6],
            "operating_profit_weight_coverage": [0.97, 0.98],
            "cf_data_status": [
                "PASS_PIT_CF_STATE_WITH_MARKET_CAP_PROXY_WEIGHT",
                "PASS_PIT_CF_STATE_WITH_MARKET_CAP_PROXY_WEIGHT",
            ],
        }
    )
    pv = pd.DataFrame(
        {
            "origin": [origin, target],
            "equity_risk_premium_state": [0.03, 0.04],
            "present_value_data_status": [
                "PASS_EXPLANATORY_CROSS_SECTIONAL_PRESENT_VALUE_STATE",
                "PASS_EXPLANATORY_CROSS_SECTIONAL_PRESENT_VALUE_STATE",
            ],
        }
    )
    risk = pd.DataFrame(
        {
            "date": dates,
            "member_average_pairwise_correlation_60d": np.linspace(0.2, 0.3, 61),
            "etf_amihud_20d": np.linspace(0.001, 0.002, 61),
        }
    )
    total = pd.DataFrame(
        {"date": dates, "close": 100.0 * np.cumprod(np.repeat(1.001, 61))}
    )
    panel = build_object_measurement_panel(
        ledger=ledger,
        coverage=coverage,
        cf_state=cf,
        present_value_state=pv,
        risk_capacity_daily=risk,
        total_return_index=total,
        minimum_cf_weight_coverage=0.90,
        annualization_days=252,
    )

    assert len(panel) == 9
    assert set(panel.loc[panel["module_id"].eq("CF"), "object_id"]) == {
        "EARNINGS_GROWTH",
        "OPERATING_CASHFLOW_GROWTH",
        "EARNINGS_CASHFLOW_BREADTH",
    }
    assert set(panel.loc[panel["module_id"].eq("DR"), "object_id"]) == {
        "VALUATION_MULTIPLE_CHANGE",
        "EQUITY_RISK_PREMIUM_GAP_CHANGE",
    }
    assert set(panel.loc[panel["module_id"].eq("RC"), "object_id"]) == {
        "REALIZED_VOLATILITY",
        "DOWNSIDE_SEMIVARIANCE",
        "MEMBER_CORRELATION",
        "LIQUIDITY_SHOCK",
    }
    assert not panel["direct_total_return_target"].any()
    assert not panel["return_prediction_allowed"].any()
    assert panel.loc[
        panel["object_id"].eq("REALIZED_VOLATILITY"),
        "daily_observation_count",
    ].iloc[0] == 60


def test_summary_distinguishes_partial_measurement_from_pass() -> None:
    origins = pd.to_datetime(
        ["2021-01-31", "2022-01-31", "2023-01-31", "2024-01-31", "2025-01-31"]
    )
    panel = pd.DataFrame(
        {
            "origin": list(origins) * 2,
            "module_id": ["CF"] * 5 + ["RC"] * 5,
            "object_id": ["EARNINGS_GROWTH"] * 5 + ["REALIZED_VOLATILITY"] * 5,
            "horizon_market_days": [60] * 10,
            "measurement_value": np.linspace(0.1, 1.0, 10),
            "measurement_status": (
                ["PARTIAL_PIT_VALUE_REVISION_AND_PROXY_WEIGHT_LIMITATION"] * 5
                + ["PASS_EXACT_FORWARD_DAILY_MEASUREMENT"] * 5
            ),
        }
    )
    summary = summarize_measurement_validity(
        panel,
        minimum_observed_share=0.90,
        minimum_pass_share=0.80,
        minimum_distinct_years=5,
    )

    cf_status = summary.loc[summary["module_id"].eq("CF"), "measurement_validity_status"].iloc[0]
    rc_status = summary.loc[summary["module_id"].eq("RC"), "measurement_validity_status"].iloc[0]
    assert cf_status == "PARTIAL_OWN_OBJECT_MEASUREMENT_VALIDITY"
    assert rc_status == "PASS_OWN_OBJECT_MEASUREMENT_VALIDITY"
