from __future__ import annotations

import json
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

from up20_rare_event_forecast_v1 import (  # noqa: E402
    CONFIG_PATH,
    FEATURE_COLUMNS,
    MANIFEST_PATH,
    STATE_DOWN,
    STATE_RANGE,
    STATE_UP,
    build_forward_labels,
    build_phase_blocks,
    compute_price_features,
    fit_sign_constrained_multinomial,
    load_config,
    validate_manifest,
)


def test_config_freezes_low_dimension_up20_question() -> None:
    config = load_config()
    assert config["research_question"]["target_label"] == STATE_UP
    assert config["model"]["features"] == list(FEATURE_COLUMNS)
    assert config["decision_rule"]["minimum_up_probability"] == 0.40
    assert config["decision_rule"]["maximum_down_probability"] == 0.10
    assert config["evaluation"]["robustness_phase_offsets"] == list(range(20))
    assert config["inputs"]["pit_membership"]["historical_weights_used"] is False


def test_price_features_do_not_change_before_modified_future_date() -> None:
    dates = pd.bdate_range("2018-01-02", periods=320)
    close = 3000.0 * np.exp(np.linspace(0.0, 0.18, len(dates)))
    base = pd.DataFrame({"date": dates, "close": close})
    altered = base.copy()
    altered.loc[altered.index[-1], "close"] *= 1.50
    original_features = compute_price_features(base)
    altered_features = compute_price_features(altered)
    cutoff = len(dates) - 2
    pd.testing.assert_frame_equal(
        original_features.iloc[:cutoff].reset_index(drop=True),
        altered_features.iloc[:cutoff].reset_index(drop=True),
        check_exact=True,
    )


def test_forward_label_matures_exactly_twenty_trading_days_later() -> None:
    dates = pd.bdate_range("2020-01-02", periods=45)
    features = pd.DataFrame(
        {
            "date": dates,
            "tri_close": np.linspace(100.0, 130.0, len(dates)),
        }
    )
    labels = build_forward_labels(features, horizon=20)
    assert labels.loc[0, "label_maturity_date"] == dates[20]
    assert labels.loc[0, "true_class"] == STATE_UP
    assert labels.loc[24, "label_maturity_date"] == dates[44]
    assert pd.isna(labels.loc[25, "label_maturity_date"])
    assert pd.isna(labels.loc[25, "true_class"])


def test_sign_constrained_model_probabilities_and_coefficient_bounds() -> None:
    rng = np.random.default_rng(20260831)
    features = rng.uniform(0.0, 1.0, size=(600, 4))
    latent_up = 1.8 * features[:, 0] + 1.2 * features[:, 2] - 1.5 * features[:, 3]
    latent_down = -1.4 * features[:, 0] - 0.8 * features[:, 2] + 1.7 * features[:, 3]
    labels = np.where(latent_up > 1.45, 0, np.where(latent_down > 0.85, 2, 1))
    fit = fit_sign_constrained_multinomial(
        features,
        labels.astype(int),
        l2_lambda=0.10,
        max_iterations=500,
        ftol=1e-12,
    )
    parameters = fit["parameters"]
    assert fit["success"] is True
    assert np.all(parameters[1:4] >= -1e-10)
    assert parameters[4] <= 1e-10
    assert np.all(parameters[6:9] <= 1e-10)
    assert parameters[9] >= -1e-10


def test_phase_blocks_apply_signal_only_to_complete_blocks() -> None:
    dates = pd.Series(pd.bdate_range("2020-01-02", periods=43))
    predictions = pd.DataFrame(
        {
            "date": dates,
            "primary_long_signal": True,
            "model_valid": True,
            "p_up": 0.60,
            "p_range": 0.35,
            "p_down": 0.05,
            "base_p_up": 0.20,
            "base_p_range": 0.60,
            "base_p_down": 0.20,
            "expected_net_utility": 0.03,
            "stress_risk": 0.20,
            "up_probability_gate": True,
            "down_probability_veto_pass": True,
            "expected_utility_gate": True,
            "stress_hard_veto_pass": True,
        }
    )
    labels = pd.DataFrame(
        {
            "date": dates,
            "future_20d_total_return": 0.06,
            "label_maturity_date": dates.shift(-20),
            "true_class": STATE_UP,
        }
    )
    blocks = build_phase_blocks(
        predictions, labels, dates, offset=0, horizon=20, signal_column="primary_long_signal"
    )
    assert blocks["complete"].tolist() == [True, True, False]
    assert blocks["target_position"].tolist() == [1.0, 1.0, 0.0]


def test_governance_never_promotes_historical_result_to_live() -> None:
    config = load_config()
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
        pytest.skip("UP20预测首次冻结前尚无清单")
    manifest = validate_manifest(load_config(CONFIG_PATH))
    assert manifest["state"] == "FROZEN_BEFORE_MODEL_OUTCOME_CALCULATION"
    assert manifest["model_outcomes_computed_before_freeze"] is False


def test_result_contract_when_written() -> None:
    result_path = ROOT / "reports" / "research" / "510300_up20_rare_event_forecast_v1_result.json"
    if not result_path.exists():
        pytest.skip("UP20预测正式结果尚未写入")
    report = json.loads(result_path.read_text(encoding="utf-8"))
    assert report["data_and_clock_boundary"]["historical_weights_used"] is False
    assert report["adjudication"]["verified_forward_observations"] == 0
    assert report["adjudication"]["goal_achieved"] is False
    assert report["governance"]["live_trading_authorized"] is False
