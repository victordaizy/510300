from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from research.post_close_offshore_information_transfer_v1 import (
    OBSERVATION_SCHEMA,
    PROJECT_ID,
    _empty_frame,
    estimate_beta_lcb,
    evaluate_entry_gate,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/510300_post_close_offshore_information_transfer_v1.yaml"
SOURCE_RECEIPT = ROOT / "config/510300_post_close_offshore_information_transfer_v1_source_receipt.json"


def load_protocol() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_protocol_is_forward_only_no_backfill_and_zero_position() -> None:
    config = load_protocol()
    protocol = config["protocol"]
    assert protocol["project_id"] == PROJECT_ID
    assert protocol["research_mode"] == "FORWARD_ONLY_DISCOVERY"
    assert protocol["historical_tradable_backtest"] == "NOT_ALLOWED"
    assert protocol["pre_start_strategy_return_reading"] == "FORBIDDEN"
    assert protocol["backfill_allowed"] is False
    assert config["scope"]["position_impact"] == 0
    assert config["scope"]["current_holding_route"] == "CASH_CNY"
    assert not any(config["boundaries"].values())


def test_sse_rule_receipt_captures_order_acceptance_nuance() -> None:
    receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    sse = receipt["sse_rule"]
    assert sse["status"] == "OFFICIAL_RULE_VERIFIED_CURRENT_AND_NOT_DELAYED"
    assert sse["effective_date"] == "2026-07-06"
    assert "9:30" in sse["verified_clauses"]["3.7.3"]
    config = load_protocol()
    assert config["exchange_rules"]["research_submit_time_is_signal_choice_not_exchange_earliest_time"] is True


def test_beta_is_no_view_before_60_mature_forward_days() -> None:
    config = load_protocol()
    frame = _empty_frame(OBSERVATION_SCHEMA)
    rows = []
    for index in range(59):
        row = {column: None for column in OBSERVATION_SCHEMA}
        row.update(
            {
                "authoritative_forward_record": True,
                "target_mature": True,
                "r_post": index / 100000.0,
                "target_return_close_to_next_0935": index / 200000.0,
            }
        )
        rows.append(row)
    frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    result = estimate_beta_lcb(frame, config)
    assert result["sample_n"] == 59
    assert result["beta_lcb"] is None
    assert result["status"] == "NO_VIEW_INSUFFICIENT_MATURE_OBSERVATIONS"


def test_beta_lcb_uses_all_mature_forward_observations_and_can_be_positive() -> None:
    config = load_protocol()
    x = np.linspace(-0.01, 0.01, 80)
    noise = np.sin(np.arange(80)) * 0.00001
    y = 0.65 * x + noise
    frame = _empty_frame(OBSERVATION_SCHEMA)
    rows = []
    for current_x, current_y in zip(x, y, strict=True):
        row = {column: None for column in OBSERVATION_SCHEMA}
        row.update(
            {
                "authoritative_forward_record": True,
                "target_mature": True,
                "r_post": current_x,
                "target_return_close_to_next_0935": current_y,
            }
        )
        rows.append(row)
    frame = pd.concat([frame, pd.DataFrame(rows)], ignore_index=True)
    result = estimate_beta_lcb(frame, config)
    assert result["sample_n"] == 80
    assert result["beta_hat"] == pytest.approx(0.65, abs=0.01)
    assert result["beta_lcb"] > 0
    assert result["status"] == "BETA_LCB_POSITIVE"


def test_entry_gate_is_cost_derived_and_never_assumes_beta_one() -> None:
    config = load_protocol()
    assert config["beta_estimator"]["beta_equals_one_assumption"] == "FORBIDDEN"
    blocked = evaluate_entry_gate(0.01, None, config)
    assert blocked["signal_qualified"] is False
    assert blocked["gross_edge_lcb"] is None
    below = evaluate_entry_gate(0.008, 0.50, config)
    assert below["gross_edge_lcb"] == pytest.approx(0.004)
    assert below["signal_qualified"] is False
    pass_gate = evaluate_entry_gate(0.009, 0.50, config)
    assert pass_gate["gross_edge_lcb"] == pytest.approx(0.0045)
    assert pass_gate["signal_qualified"] is True
    invalid = evaluate_entry_gate(float("nan"), 0.50, config)
    assert invalid["status"] == "NO_VIEW_INVALID_R_POST"
    assert invalid["signal_qualified"] is False


def test_no_unfrozen_filters_and_no_assumed_fills() -> None:
    config = load_protocol()
    assert set(config["forbidden_filters"]) == {
        "MACD",
        "MOVING_AVERAGE",
        "BULL_BEAR_REGIME",
        "OPTION_SKEW",
        "INTRADAY_BREADTH",
        "SAME_DAY_RETURN",
        "OVERNIGHT_NEWS_CLASSIFICATION",
        "RESULT_SELECTED_EXIT_TIME",
    }
    assert config["ledgers"]["assumed_fills_allowed"] is False
    assert config["ledgers"]["order_intent_authorized_quantity_shares"] == 0
    assert config["ledgers"]["actual_fill_recording_before_separate_live_authorization"] is False
    assert config["stop_gates"]["stop_action"] == "NO_NEW_ENTRY"
    assert "LAST_10_VALID_SIGNALS_ALL_UNFILLED" in config["stop_gates"]["rules"]
