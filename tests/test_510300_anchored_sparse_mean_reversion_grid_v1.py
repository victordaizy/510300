from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.anchored_sparse_mean_reversion_grid_v1 import (
    CONFIG_PATH,
    _prior_rolling_location_scale,
    atomic_json,
    attach_event_outcomes,
    audit_inputs,
    cost_model,
    evaluate_phase_a,
    extract_events,
    load_config,
    load_inputs,
)


def test_contract_freezes_scope_cost_and_phase_gate() -> None:
    config = load_config(CONFIG_PATH)
    model = cost_model(config)
    assert config["scope"]["execution_asset"] == "510300.SH"
    assert set(config["scope"]["allowed_holdings"]) == {"510300.SH", "CASH_CNY"}
    assert config["scope"]["observational_anchor_is_tradable"] is False
    assert config["scope"]["martingale_position_sizing_allowed"] is False
    assert config["account"]["initial_capital_cny"] == 200_000.0
    assert config["account"]["grid_notional_cny"] == 20_000.0
    assert np.isclose(model.commission_rate_effective_per_leg, 0.00025)
    assert np.isclose(model.base_round_trip_rate, 0.0015)
    assert np.isclose(model.stress_round_trip_rate, 0.00225)
    assert config["phase_b"]["portfolio_backtest_implemented_in_v1"] is False
    assert config["governance"]["order_generation"] == "DISABLED"
    assert config["governance"]["live_trading_authorized"] is False


def test_prior_rolling_location_scale_excludes_current_observation() -> None:
    values = pd.Series(np.arange(25, dtype=float))
    location, scale = _prior_rolling_location_scale(values, window=20)
    assert np.isnan(location.iloc[19])
    assert np.isclose(location.iloc[20], 9.5)
    expected_mad = np.median(np.abs(np.arange(20, dtype=float) - 9.5)) * 1.4826
    assert np.isclose(scale.iloc[20], expected_mad)
    changed = values.copy()
    changed.iloc[20] = 1_000_000.0
    changed_location, changed_scale = _prior_rolling_location_scale(changed, window=20)
    assert np.isclose(changed_location.iloc[20], location.iloc[20])
    assert np.isclose(changed_scale.iloc[20], scale.iloc[20])


def _event_panel() -> pd.DataFrame:
    times = pd.date_range("2026-01-05 10:00", periods=8, freq="min")
    z_values = [0.2, 1.2, 1.5, 1.4, 0.4, 0.2, 0.1, 0.0]
    amount_values = [100.0, 150.0, 200.0, 150.0, 100.0, 100.0, 100.0, 100.0]
    scale = 0.004
    frame = pd.DataFrame(
        {
            "trade_time": times,
            "date": pd.Timestamp("2026-01-05"),
            "bar_index": np.arange(len(times)),
            "minute_of_day": times.hour * 60 + times.minute,
            "feature_ready": True,
            "basis_z": z_values,
            "centered_log_basis": np.array(z_values) * scale,
            "etf_amount_5m": amount_values,
            "etf_amount_5m_ratio": 2.0,
            "raw_log_basis": np.array(z_values) * scale,
            "basis_center_prior20": 0.0,
            "basis_scale": scale,
            "path_efficiency_30m": 0.2,
            "time_segment": "MORNING_MIDDLE_1000_1130",
            "market_state": "MEDIUM_VOL",
            "etf_log_return_5m": 0.005,
            "index_log_return_5m": 0.001,
            "relative_log_return_5m": 0.004,
            "base_round_trip_cost": 0.0015,
            "stress_round_trip_cost": 0.00225,
            "first_grid_band": 0.00375,
            "second_grid_band": 0.012,
            "outcome_only_full_day_path_efficiency": 0.2,
        }
    )
    return frame


def test_event_requires_non_widening_and_observed_volume_decay() -> None:
    config = load_config(CONFIG_PATH)
    events = extract_events(_event_panel(), config)
    assert len(events) == 1
    event = events.iloc[0]
    assert event["signal_time"] == pd.Timestamp("2026-01-05 10:03")
    assert bool(event["future_information_used_in_signal"]) is False
    assert event["liquidity_shock_qualified"]
    assert np.isclose(event["liquidity_decay_fraction"], 0.75)
    assert event["first_grid_qualified"]
    assert event["cost_grid_qualified"]


def test_outcome_identity_starts_at_next_record_open() -> None:
    config = load_config(CONFIG_PATH)
    times = pd.date_range("2026-01-05 10:00", periods=61, freq="min")
    etf_open = np.linspace(4.0, 4.06, len(times))
    etf_close = etf_open + 0.001
    index_open = np.linspace(4000.0, 4030.0, len(times))
    index_close = index_open + 0.5
    panel = pd.DataFrame(
        {
            "trade_time": times,
            "date": pd.Timestamp("2026-01-05"),
            "bar_index": np.arange(len(times)),
            "etf_open": etf_open,
            "etf_close": etf_close,
            "index_open": index_open,
            "index_close": index_close,
            "centered_log_basis": np.linspace(-0.006, 0.0, len(times)),
            "basis_scale": 0.002,
        }
    )
    events = pd.DataFrame(
        [
            {
                "event_id": "synthetic",
                "date": pd.Timestamp("2026-01-05"),
                "signal_time": times[0],
                "signal_bar_index": 0,
                "direction_sign": -1,
                "direction": "DISCOUNT_BUY_WITH_CASH",
                "base_round_trip_cost": 0.0015,
                "stress_round_trip_cost": 0.00225,
                "cost_grid_qualified": True,
                "candidate_time_allowed": True,
                "liquidity_shock_qualified": True,
                "basis_scale": 0.002,
                "abs_centered_log_basis": 0.006,
                "path_efficiency_bin": "LOW",
                "severity_bin": "3.0_PLUS",
                "time_segment": "MORNING_MIDDLE_1000_1130",
                "market_state": "MEDIUM_VOL",
                "volume_ratio_bin": "EXTREME",
                "outcome_only_full_day_path_efficiency": 0.2,
            }
        ]
    )
    outcomes = attach_event_outcomes(events, panel, config)
    result = outcomes.iloc[0]
    assert result["entry_time"] == times[1]
    assert np.isclose(
        result["convergence_score_5m"],
        result["etf_reversal_score_5m"] + result["anchor_catchup_score_5m"],
    )
    assert np.isclose(
        result["net_base_score_5m"], result["etf_reversal_score_5m"] - 0.0015
    )


def test_insufficient_event_budget_blocks_phase_b() -> None:
    config = load_config(CONFIG_PATH)
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-02", periods=300, freq="B"),
            "outcome_only_full_day_path_efficiency": np.linspace(0.0, 1.0, 300),
        }
    )
    evaluation = evaluate_phase_a(pd.DataFrame(), panel, config)
    assert evaluation["phase_a_passed"] is False
    assert evaluation["phase_b_portfolio_backtest"] == "NOT_ALLOWED"
    assert evaluation["status"] == "REJECTED_PHASE_A_NO_GRID_BACKTEST_NO_RESCUE"
    assert evaluation["gates"]["completed_event_count_at_least_300"] is False
    assert evaluation["portfolio_sharpe_uplift_vs_same_average_exposure"] == (
        "NOT_EVALUATED_PHASE_A"
    )
    assert evaluation["metrics"]["latest_two_calendar_years"] == [2025, 2026]


def test_strict_json_converts_non_finite_values_to_null(tmp_path: Path) -> None:
    output = tmp_path / "strict.json"
    atomic_json(output, {"finite": 1.0, "nan": float("nan"), "inf": float("inf")})
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {"finite": 1.0, "nan": None, "inf": None}


def test_bound_real_inputs_pass_prefreeze_audit_without_future_outcomes() -> None:
    config = load_config(CONFIG_PATH)
    inputs = load_inputs(config)
    audit = audit_inputs(inputs, config)
    assert audit["passed"] is True
    assert audit["status"] == "PASS_DISCOVERY_INDEX_PROXY_INPUTS"
    assert audit["sections"]["timestamp_alignment"]["exact_timestamp_match_count"] == 291_851
    assert audit["sections"]["index_minute"]["unused_or_full_ohlc_anomaly_count"] == 1
    assert audit["sections"]["etf_daily_crosscheck"]["passed"] is True
    assert audit["sections"]["index_daily_crosscheck"]["passed"] is True
    assert audit["return_evaluation"] == "ALLOWED_AFTER_MANIFEST_VERIFICATION"
