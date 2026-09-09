from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from three_state_trend_router_v1_0_1 import (  # noqa: E402
    ALL_STATES,
    CONFIG_PATH,
    MANIFEST_PATH,
    STATE_BEAR,
    STATE_BULL,
    STATE_RANGE,
    build_directional_movement_panel,
    build_state_panel,
    build_targets,
    classify_three_states,
    load_config,
    validate_manifest,
    wilder_rma,
)


def test_frozen_config_core_contract() -> None:
    config = load_config()
    assert config["protocol"]["project_id"] == "510300_THREE_STATE_TREND_ROUTER_V1_0_1"
    assert config["directional_movement_system"]["period_trading_days"] == 14
    assert config["directional_movement_system"]["trend_entry_adx"] == 25.0
    assert config["directional_movement_system"]["trend_exit_adx"] == 20.0
    assert config["range_policy"]["historical_policy"] == "NO_TRADE"
    assert config["range_policy"]["range_t_backtest_allowed"] is False


def test_wilder_rma_uses_full_window_then_recursive_update() -> None:
    values = np.array([1.0, 2.0, 3.0, 6.0, 9.0])
    observed = wilder_rma(values, 3)
    expected = np.array([np.nan, np.nan, 2.0, 10.0 / 3.0, 47.0 / 9.0])
    np.testing.assert_allclose(observed, expected, equal_nan=True)


def test_state_machine_entry_exit_and_hysteresis() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "true_range": 1.0,
            "smoothed_true_range": 1.0,
            "plus_di": [np.nan, 30.0, 10.0, 10.0, 10.0, 30.0],
            "minus_di": [np.nan, 10.0, 30.0, 30.0, 30.0, 10.0],
            "dx": [np.nan, 50.0, 50.0, 50.0, 50.0, 50.0],
            "adx": [np.nan, 26.0, 22.0, 26.0, 19.0, 26.0],
        }
    )
    result = classify_three_states(panel, 25.0, 20.0)
    assert result["state"].tolist() == [
        STATE_RANGE,
        STATE_BULL,
        STATE_BULL,
        STATE_BEAR,
        STATE_RANGE,
        STATE_BULL,
    ]


def test_directional_movement_is_causal_under_future_append() -> None:
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    close = np.linspace(100.0, 140.0, len(dates)) + np.sin(np.arange(len(dates)))
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
        }
    )
    prefix = build_directional_movement_panel(frame.iloc[:80], 14)
    full = build_directional_movement_panel(frame, 14).iloc[:80]
    np.testing.assert_allclose(prefix["plus_di"], full["plus_di"], equal_nan=True)
    np.testing.assert_allclose(prefix["minus_di"], full["minus_di"], equal_nan=True)
    np.testing.assert_allclose(prefix["adx"], full["adx"], equal_nan=True)


def test_target_mapping_is_bull_full_other_states_cash() -> None:
    config = load_config()
    dates = pd.to_datetime(["2015-01-05", "2015-01-06", "2015-01-07"])
    panel = pd.DataFrame(
        {
            "date": dates,
            "state": [STATE_BULL, STATE_BEAR, STATE_RANGE],
            "target_position": [1.0, 0.0, 0.0],
        }
    )
    targets = build_targets(panel, config)
    assert targets["target_position"].tolist() == [1.0, 0.0, 0.0]
    assert targets["risk_off_override"].tolist() == [False, True, True]


def test_real_input_state_panel_has_only_fixed_states() -> None:
    config = load_config()
    index = pd.read_parquet(ROOT / config["inputs"]["index_daily"]["path"])
    panel = build_state_panel(index, config)
    assert set(panel["state"].unique()).issubset(set(ALL_STATES))
    assert panel["target_position"].isin([0.0, 1.0]).all()
    assert panel["date"].is_monotonic_increasing


def test_prefreeze_range_t_gate_forces_no_trade() -> None:
    config = load_config()
    audit_path = ROOT / config["inputs"]["input_audit"]["path"]
    audit = __import__("json").loads(audit_path.read_text(encoding="utf-8"))
    assert audit["range_t_data_gate"]["coverage_complete"] is False
    assert audit["range_t_data_gate"]["range_t_return_evaluation"] == "NOT_ALLOWED"
    assert audit["range_t_data_gate"]["fallback_range_policy"] == "NO_TRADE"


def test_governance_has_no_signal_or_trading_authorization() -> None:
    config = load_config()
    assert config["scope"]["live_trading_authorized"] is False
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
        pytest.skip("首次冻结前尚无清单")
    config = load_config(CONFIG_PATH)
    manifest = validate_manifest(config)
    assert manifest["state"] == "FROZEN_BEFORE_FIRST_CANDIDATE_EVALUATION"
    assert manifest["candidate_2015_plus_outcomes_read_before_freeze"] is False
    assert manifest["candidate_portfolio_returns_read_before_freeze"] is False


def test_result_replay_contract_when_frozen() -> None:
    if not MANIFEST_PATH.exists():
        pytest.skip("首次冻结前尚无清单")
    from three_state_trend_router_v1_0_1 import run_study

    report = run_study(write=False)
    assert report["portfolio_evaluation"]["evaluated"] is True
    assert report["adjudication"]["verified_forward_target_achieved"] is False
    assert report["adjudication"]["goal_achieved"] is False
    assert report["range_policy"]["range_t_return_evaluation"] == "NOT_ALLOWED"
