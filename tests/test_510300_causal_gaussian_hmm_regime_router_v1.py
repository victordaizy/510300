from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from causal_gaussian_hmm_regime_router_v1 import (  # noqa: E402
    ALL_STATES,
    CONFIG_PATH,
    FEATURE_NAMES,
    MANIFEST_PATH,
    MODEL_PATH,
    STATE_BEAR,
    STATE_BULL,
    STATE_RANGE,
    build_state_panel,
    build_targets,
    causal_filter,
    classify_next_state_probabilities,
    compute_feature_panel,
    forward_backward,
    load_config,
    load_frozen_model,
    validate_manifest,
)


def test_config_freezes_user_requested_three_stage_actions() -> None:
    config = load_config()
    assert config["protocol"]["project_id"] == (
        "510300_CAUSAL_GAUSSIAN_HMM_REGIME_ROUTER_V1"
    )
    assert config["dates"]["training_end"] == "2014-12-31"
    assert config["dates"]["evaluation_start"] == "2015-01-05"
    assert config["states"][STATE_BULL]["target_position"] == 1.0
    assert config["states"][STATE_BEAR]["target_position"] == 0.0
    assert config["states"][STATE_RANGE]["target_position"] == 0.0
    assert config["range_policy"]["range_t_backtest_allowed"] is False


def test_feature_panel_is_causal_under_future_append() -> None:
    dates = pd.date_range("2020-01-01", periods=90, freq="B")
    close = 100.0 * np.exp(np.linspace(0.0, 0.2, len(dates)))
    frame = pd.DataFrame({"date": dates, "close": close})
    prefix = compute_feature_panel(frame.iloc[:70]).reset_index(drop=True)
    full_prefix = compute_feature_panel(frame).loc[
        lambda value: value["date"].le(dates[69])
    ].reset_index(drop=True)
    pd.testing.assert_series_equal(prefix["date"], full_prefix["date"])
    np.testing.assert_allclose(
        prefix[list(FEATURE_NAMES)], full_prefix[list(FEATURE_NAMES)], atol=0.0, rtol=0.0
    )


def test_forward_backward_and_causal_filter_probabilities_are_normalized() -> None:
    observations = np.array(
        [[-1.0, -0.5], [-0.8, -0.4], [0.1, 0.0], [0.9, 0.6], [1.1, 0.7]]
    )
    initial = np.array([0.5, 0.5])
    transition = np.array([[0.9, 0.1], [0.2, 0.8]])
    means = np.array([[-0.8, -0.4], [0.9, 0.6]])
    variances = np.full((2, 2), 0.25)
    log_likelihood, gamma, expected_transitions = forward_backward(
        observations, initial, transition, means, variances, 1e-12
    )
    filtered = causal_filter(
        observations, initial, transition, means, variances, 1e-12
    )
    assert np.isfinite(log_likelihood)
    np.testing.assert_allclose(gamma.sum(axis=1), 1.0)
    np.testing.assert_allclose(filtered.sum(axis=1), 1.0)
    assert (expected_transitions >= 0).all()


def test_probability_decision_uses_next_state_majority() -> None:
    mapping = {STATE_BEAR: 0, STATE_RANGE: 1, STATE_BULL: 2}
    assert classify_next_state_probabilities(
        np.array([0.10, 0.20, 0.70]), mapping, 0.50
    )[0] == STATE_BULL
    assert classify_next_state_probabilities(
        np.array([0.60, 0.30, 0.10]), mapping, 0.50
    )[0] == STATE_BEAR
    assert classify_next_state_probabilities(
        np.array([0.40, 0.35, 0.25]), mapping, 0.50
    )[0] == STATE_RANGE
    assert classify_next_state_probabilities(
        np.array([0.50, 0.00, 0.50]), mapping, 0.50
    )[0] == STATE_RANGE


def test_prefit_model_uses_only_pre_2015_training_data_when_present() -> None:
    if not MODEL_PATH.exists():
        pytest.skip("预冻结模型尚未生成")
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    assert model["training"]["start"] == "2012-06-26"
    assert model["training"]["end"] == "2014-12-31"
    assert model["training"]["rows"] == 613
    assert model["training"]["candidate_2015_plus_states_computed"] is False
    assert model["training"]["candidate_2015_plus_outcomes_read_or_computed"] is False
    assert model["training"]["candidate_portfolio_returns_read_or_computed"] is False
    assert model["training_diagnostics"]["training_gate_passed"] is True


def test_real_state_panel_is_causal_and_uses_only_fixed_states_when_model_present() -> None:
    if not MODEL_PATH.exists():
        pytest.skip("预冻结模型尚未生成")
    config = load_config()
    model = load_frozen_model(config)
    index = pd.read_parquet(ROOT / config["inputs"]["index_daily"]["path"])
    panel = build_state_panel(index, model, config)
    assert set(panel["state"].unique()).issubset(set(ALL_STATES))
    assert panel["target_position"].isin([0.0, 1.0]).all()
    assert panel["date"].is_monotonic_increasing
    probability_columns = [f"next_probability::{state}" for state in ALL_STATES]
    np.testing.assert_allclose(panel[probability_columns].sum(axis=1), 1.0)
    cutoff = pd.Timestamp("2020-12-31")
    prefix_index = index.loc[pd.to_datetime(index["date"]).le(cutoff)].copy()
    prefix_panel = build_state_panel(prefix_index, model, config)
    full_prefix = panel.loc[panel["date"].le(cutoff)].reset_index(drop=True)
    prefix_panel.reset_index(drop=True, inplace=True)
    pd.testing.assert_series_equal(prefix_panel["state"], full_prefix["state"])
    np.testing.assert_allclose(
        prefix_panel[probability_columns], full_prefix[probability_columns], atol=1e-12
    )


def test_target_mapping_is_bull_full_and_other_states_cash() -> None:
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
    assert manifest["state"] == "FROZEN_BEFORE_FIRST_2015_PLUS_CANDIDATE_EVALUATION"
    assert manifest["candidate_2015_plus_states_read_before_freeze"] is False
    assert manifest["candidate_2015_plus_outcomes_read_before_freeze"] is False
    assert manifest["candidate_portfolio_returns_read_before_freeze"] is False


def test_result_replay_contract_when_frozen() -> None:
    if not MANIFEST_PATH.exists():
        pytest.skip("首次冻结前尚无清单")
    from causal_gaussian_hmm_regime_router_v1 import run_study

    report = run_study(write=False)
    assert report["portfolio_evaluation"]["evaluated"] is True
    assert report["adjudication"]["verified_forward_target_achieved"] is False
    assert report["adjudication"]["goal_achieved"] is False
    assert report["range_policy"]["range_t_return_evaluation"] == "NOT_ALLOWED"
