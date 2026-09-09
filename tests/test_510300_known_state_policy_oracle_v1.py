from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from known_state_policy_oracle_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    POLICY_PRIMARY,
    STATE_BEAR,
    STATE_BULL,
    STATE_CENSORED,
    STATE_RANGE,
    build_oracle_blocks,
    build_targets,
    classify_oracle_state,
    load_config,
    validate_manifest,
)


def test_config_separates_conditional_policy_from_state_prediction() -> None:
    config = load_config()
    assert config["protocol"]["oracle_future_information_used"] is True
    assert config["protocol"]["state_predictability_evaluated"] is False
    assert config["protocol"]["strategy_mapping_evaluated"] is True
    assert config["policies"][POLICY_PRIMARY] == {
        STATE_BULL: 1.0,
        STATE_BEAR: 0.0,
        STATE_RANGE: 0.0,
    }


def test_oracle_state_threshold_boundaries() -> None:
    assert classify_oracle_state(0.05, 0.05) == STATE_BULL
    assert classify_oracle_state(-0.05, 0.05) == STATE_BEAR
    assert classify_oracle_state(0.049999, 0.05) == STATE_RANGE
    assert classify_oracle_state(-0.049999, 0.05) == STATE_RANGE
    assert classify_oracle_state(np.nan, 0.05) == STATE_CENSORED


def test_synthetic_nonoverlapping_blocks_have_twenty_future_days() -> None:
    config = load_config()
    config = {**config, "data_contract": {**config["data_contract"]}}
    config["data_contract"]["expected_evaluation_trading_days"] = 43
    config["data_contract"]["expected_complete_oracle_blocks"] = 2
    config["data_contract"]["expected_censored_blocks"] = 1
    config["data_contract"]["expected_trailing_days_after_censored_anchor"] = 2
    dates = pd.Series(pd.date_range("2020-01-01", periods=43, freq="B"))
    closes = 100.0 * np.exp(np.linspace(0.0, 0.12, len(dates)))
    tri = pd.DataFrame({"date": dates, "close": closes})
    blocks = build_oracle_blocks(dates, tri, 0.05, config)
    assert len(blocks) == 3
    assert blocks["complete"].tolist() == [True, True, False]
    assert blocks["execution_trading_days"].tolist() == [20, 20, 2]
    assert blocks["anchor_position"].tolist() == [0, 20, 40]
    assert blocks.loc[2, "state"] == STATE_CENSORED


def test_primary_targets_map_bull_full_and_other_states_cash() -> None:
    config = load_config()
    blocks = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"]
            ),
            "state": [STATE_BULL, STATE_BEAR, STATE_RANGE, STATE_CENSORED],
            "complete": [True, True, True, False],
        }
    )
    targets = build_targets(blocks, config["policies"][POLICY_PRIMARY], POLICY_PRIMARY)
    assert targets["target_position"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert targets["risk_off_override"].tolist() == [False, True, True, True]


def test_governance_never_promotes_oracle_to_signal() -> None:
    config = load_config()
    assert config["governance"]["historical_strategy_target_can_pass"] is False
    assert config["governance"]["state_predictability_can_pass"] is False
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
        pytest.skip("Oracle首次冻结前尚无清单")
    manifest = validate_manifest(load_config(CONFIG_PATH))
    assert manifest["state"] == "FROZEN_BEFORE_ORACLE_OUTCOME_CALCULATION"
    assert manifest["oracle_outcomes_read_before_freeze"] is False
    assert manifest["state_predictability_evaluated"] is False


def test_real_result_contract_when_frozen() -> None:
    if not MANIFEST_PATH.exists():
        pytest.skip("Oracle首次冻结前尚无清单")
    from known_state_policy_oracle_v1 import run_study

    report = run_study(write=False)
    assert report["oracle_information_boundary"]["future_information_used"] is True
    assert report["oracle_information_boundary"]["state_predictability_evaluated"] is False
    assert report["adjudication"]["state_prediction_hypothesis"] == "NOT_EVALUATED"
    assert report["adjudication"]["historical_tradable_strategy_target_achieved"] is False
    assert report["adjudication"]["goal_achieved"] is False
