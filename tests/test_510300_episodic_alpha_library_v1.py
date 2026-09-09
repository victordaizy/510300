from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from research.episodic_alpha_library_v1 import (
    BRANCH_IDS,
    confirmation_from_minutes,
    evaluate_active_eligibility,
    event_metrics,
    historical_screen_decision,
    load_config,
    negative_page_hinkley_alarm,
    prior_rolling_quantile,
    prior_rolling_z,
    robust_flow_surprise_z,
    route_active_strategies,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_episodic_alpha_library_v1.yaml"


def test_protocol_simultaneously_contains_three_branches_and_closed_trade_switches() -> None:
    config = load_config(CONFIG_PATH)
    assert tuple(item["model_id"] for item in config["branches"].values()) == BRANCH_IDS
    assert config["protocol"]["simultaneous_branch_freeze_required"] is True
    assert config["formal_status"]["510300_20D_REGIME_TIMING_FAMILY"] == "CLOSED_REJECTED_FROZEN"
    assert config["formal_status"]["LIVE_TRADING_AUTHORIZED"] is False
    assert config["boundaries"]["position_mapping_enabled"] is False
    assert config["boundaries"]["order_generation_enabled"] is False
    assert config["boundaries"]["broker_connection_enabled"] is False
    assert config["boundaries"]["live_trading_enabled"] is False


def test_all_rolling_statistics_exclude_current_observation() -> None:
    base = pd.Series([1.0, 2.0, 3.0, 4.0])
    altered = base.copy()
    altered.iloc[-1] = 4000.0
    z_base = prior_rolling_z(base, window=3, minimum_history=3)
    z_altered = prior_rolling_z(altered, window=3, minimum_history=3)
    history_mean = 2.0
    history_std = 1.0
    assert math.isclose(z_base.iloc[-1], (4.0 - history_mean) / history_std)
    assert math.isclose(z_altered.iloc[-1], (4000.0 - history_mean) / history_std)

    q_base = prior_rolling_quantile(base, 0.5, window=3, minimum_history=3)
    q_altered = prior_rolling_quantile(altered, 0.5, window=3, minimum_history=3)
    assert q_base.iloc[-1] == q_altered.iloc[-1] == 2.0


def test_robust_flow_surprise_uses_prior_window_only() -> None:
    flow = pd.Series([0.00, 0.01, -0.01, 0.00, 0.02])
    z = robust_flow_surprise_z(flow, window=4, minimum_history=4, mad_scale=1.4826)
    expected = 0.02 / (1.4826 * 0.005)
    assert math.isclose(z.iloc[-1], expected, rel_tol=1e-12)


def test_confirmation_uses_0930_to_0944_and_executes_at_0945_open() -> None:
    times = pd.date_range("2026-01-05 09:30:00", periods=16, freq="min")
    close = np.array([
        99.0, 98.0, 97.0, 96.0, 95.0,
        96.0, 97.0, 98.0, 99.0, 100.0,
        100.1, 100.2, 100.3, 100.4, 100.5,
        101.0,
    ])
    low = np.array([
        98.5, 97.5, 96.5, 95.5, 90.0,
        95.0, 96.0, 97.0, 98.0, 99.0,
        99.5, 99.6, 99.7, 99.8, 99.9,
        100.8,
    ])
    volume = np.full(16, 10000.0)
    frame = pd.DataFrame({
        "trade_time": times,
        "open": np.r_[100.0, close[:-1]],
        "high": close + 0.2,
        "low": low,
        "close": close,
        "vol": volume,
        "amount": close * volume,
    })
    confirmation = confirmation_from_minutes(frame)
    assert confirmation["confirmation_complete"] is True
    assert confirmation["no_continued_new_low"] is True
    assert confirmation["above_first15_vwap"] is True
    assert confirmation["entry_price_0945"] == frame.iloc[15]["open"]
    changed = frame.copy()
    changed.loc[15, "open"] = 10000.0
    changed_confirmation = confirmation_from_minutes(changed)
    assert changed_confirmation["first15_vwap"] == confirmation["first15_vwap"]
    assert changed_confirmation["first10_low"] == confirmation["first10_low"]


def _synthetic_events(count: int) -> pd.DataFrame:
    gross = np.tile(np.array([0.0060, 0.0045, 0.0055, 0.0035]), math.ceil(count / 4))[:count]
    stress_net = gross - 0.0014
    double_stress_net = gross - 0.0024
    return pd.DataFrame({
        "candidate_preconfirmation": np.ones(count, dtype=bool),
        "signal": np.ones(count, dtype=bool),
        "mature_event": np.ones(count, dtype=bool),
        "exit_date": pd.date_range("2021-01-05", periods=count, freq="14D"),
        "gross_return": gross,
        "stress_net_return": stress_net,
        "double_stress_net_return": double_stress_net,
    })


def test_historical_metrics_can_never_supply_forward_active_events() -> None:
    config = load_config(CONFIG_PATH)
    metrics = event_metrics(_synthetic_events(60), config)
    assert metrics["mature_event_count"] == 60
    assert metrics["forward_mature_event_count"] == 0
    eligibility = evaluate_active_eligibility(
        metrics,
        config,
        independent_forward_mature_events=0,
        point_in_time_qualified=False,
    )
    assert eligibility["eligible"] is False
    assert eligibility["checks"]["forward_events"] is False
    assert eligibility["checks"]["point_in_time"] is False


def test_fewer_than_40_mature_events_remains_insufficient_not_rescued() -> None:
    config = load_config(CONFIG_PATH)
    metrics = event_metrics(_synthetic_events(39), config)
    assert historical_screen_decision(metrics, config) == "RESEARCH_OBSERVED_DISCOVERY_INSUFFICIENT_EVENTS"


def test_router_uses_edge_then_frozen_priority_and_never_stacks_positions() -> None:
    strategies = [
        {
            "model_id": BRANCH_IDS[2],
            "status": "ACTIVE",
            "current_signal": True,
            "conservative_net_edge": 0.003,
            "hard_veto": False,
        },
        {
            "model_id": BRANCH_IDS[0],
            "status": "ACTIVE",
            "current_signal": True,
            "conservative_net_edge": 0.003,
            "hard_veto": False,
        },
    ]
    route = route_active_strategies(strategies)
    assert route["selected_model_id"] == BRANCH_IDS[0]
    assert route["target_holding"] == "510300.SH"
    assert route["target_position"] == 1.0

    cash = route_active_strategies([
        {**strategies[0], "status": "SHADOW"},
        {**strategies[1], "hard_veto": True},
    ])
    assert cash["selected_model_id"] is None
    assert cash["target_holding"] == "CASH_CNY"
    assert cash["target_position"] == 0.0


def test_negative_page_hinkley_alarms_on_persistent_post_activation_decay() -> None:
    baseline = np.tile(np.array([0.01, 0.012, 0.008, 0.011]), 10)
    decayed = np.full(20, -0.02)
    result = negative_page_hinkley_alarm(np.r_[baseline, decayed])
    assert result["alarm"] is True
    assert result["status"] == "ALARM"
