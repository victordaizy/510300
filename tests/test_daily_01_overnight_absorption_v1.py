from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.daily_01_overnight_absorption_v1 import (
    audit_and_load_inputs,
    build_daily_decomposition,
    build_signal_states,
    load_config,
    point_in_time_percentile,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config()


def test_ex_dividend_gap_is_not_misclassified_as_overnight_selling() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-16", "2026-01-19"]),
            "open": [10.0, 9.5],
            "high": [10.1, 9.9],
            "low": [9.9, 9.4],
            "close": [10.0, 9.8],
        }
    )
    dividends = pd.DataFrame(
        {
            "ex_date": [pd.Timestamp("2026-01-19")],
            "cash_dividend_per_share": [0.5],
        }
    )

    result = build_daily_decomposition(market, dividends)

    assert result.loc[1, "overnight_contribution"] == pytest.approx(0.0)
    assert result.loc[1, "intraday_contribution"] == pytest.approx(0.03)
    assert result.loc[1, "total_return"] == pytest.approx(0.03)
    assert result.loc[1, "decomposition_error"] == pytest.approx(0.0)


def test_point_in_time_percentile_excludes_current_value() -> None:
    values = pd.Series([1.0, 2.0, 3.0, 100.0, 0.0])

    result = point_in_time_percentile(values, history=3)

    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == pytest.approx(1.0)
    assert result.iloc[4] == pytest.approx(0.0)


def test_entry_exit_equality_and_maximum_holding_are_frozen() -> None:
    features = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-05", periods=6),
            "absorption_percentile": [0.80, 0.50, 0.49, 0.90, 0.90, 0.90],
            "entry_sign_eligible": [True, False, False, True, True, True],
        }
    )

    result = build_signal_states(
        features,
        entry_percentile=0.80,
        exit_percentile=0.50,
        maximum_holding_days=10,
    )

    assert result["signal_action"].tolist() == ["ENTER", "HOLD", "EXIT", "ENTER", "HOLD", "HOLD"]
    assert result["research_state_target"].tolist() == [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]


def test_maximum_holding_exit_is_not_optional() -> None:
    features = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-05", periods=2),
            "absorption_percentile": [0.90, 0.90],
            "entry_sign_eligible": [True, True],
        }
    )

    result = build_signal_states(
        features,
        entry_percentile=0.80,
        exit_percentile=0.50,
        maximum_holding_days=1,
    )

    assert result["signal_action"].tolist() == ["ENTER", "EXIT"]


def test_protocol_registers_one_daily_only_candidate_and_no_outcome_authority() -> None:
    assert CONFIG["trial_registry"]["registered_candidate_count"] == 1
    assert CONFIG["feature_contract"]["aggregation_window_trading_days"] == 5
    assert CONFIG["feature_contract"]["volatility_window_trading_days"] == 20
    assert CONFIG["feature_contract"]["percentile_history_trading_days"] == 252
    assert CONFIG["prediction_contract"]["primary_horizon_trading_days"] == 20
    assert CONFIG["governance"]["minute_data_allowed"] is False
    assert CONFIG["protocol"]["predictive_outcome_test_authorized"] is False
    assert CONFIG["protocol"]["strategy_backtest_authorized"] is False


def test_real_data_gate_passes_without_computing_features_or_outcomes() -> None:
    market, dividends, audit = audit_and_load_inputs(ROOT, CONFIG)

    assert audit["status"] == "PASS"
    assert len(market) == 2429
    assert len(dividends) == 14
    assert audit["maximum_decomposition_identity_error"] <= 1e-12
    assert audit["feature_values_computed_on_real_data"] is False
    assert audit["signal_state_computed_on_real_data"] is False
    assert audit["future_return_columns_loaded"] is False
    assert audit["predictive_outcomes_computed"] is False
    assert audit["strategy_backtest_computed"] is False


def test_live_safety_switches_are_disabled() -> None:
    assert CONFIG["protocol"]["true_forward_start"] is None
    assert CONFIG["protocol"]["live_trading_authorized"] is False
    assert CONFIG["governance"]["position_mapping_enabled"] is False
    assert CONFIG["governance"]["order_generation_enabled"] is False
    assert CONFIG["governance"]["broker_connection_enabled"] is False
