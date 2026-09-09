from __future__ import annotations

from pathlib import Path

import pandas as pd

from research.t_only_forward_v1_1 import (
    apply_causal_weekly_states,
    build_daily_guide,
    latest_diagnostic,
    load_calendar,
    load_config,
)
from research.weekly_daily_technical_v1 import (
    build_features,
    load_config as load_base_config,
)
from scripts.run_t_only_forward_v1 import load_dividends


ROOT = Path(__file__).resolve().parents[1]
V1_1_CONFIG = load_config()
BASE_CONFIG = load_base_config()


def _current_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    market = pd.read_parquet(
        ROOT / "data" / "raw" / "t_only_forward_v1" / "510300_daily.parquet"
    )
    market["date"] = pd.to_datetime(market["date"], errors="raise")
    dividends = load_dividends(ROOT / "data" / "reference" / "510300_dividends.csv")
    return build_features(market, dividends, BASE_CONFIG)


def test_2026_calendar_matches_frozen_contract() -> None:
    calendar = load_calendar(V1_1_CONFIG)
    assert len(calendar) == 242
    assert calendar["trade_date"].min() == pd.Timestamp("2026-01-05")
    assert calendar["trade_date"].max() == pd.Timestamp("2026-12-31")
    assert int(calendar["is_week_last_trade_date"].sum()) == 51


def test_partial_week_terminal_day_uses_previous_completed_week() -> None:
    features, weekly = _current_features()
    calendar = load_calendar(V1_1_CONFIG)
    corrected, audit = apply_causal_weekly_states(
        features,
        weekly,
        calendar,
        signal_start=calendar["trade_date"].min(),
    )
    row = corrected.loc[corrected["date"].eq(pd.Timestamp("2026-08-18"))].iloc[0]
    assert row["v1_uncorrected_weekly_state_at_close"] == "W_RANGE"
    assert row["weekly_state_at_close"] == "W_BEAR"
    assert row["weekly_state_source_date"] == pd.Timestamp("2026-08-14")
    assert bool(row["is_completed_week_close"]) is False
    assert audit["rows_with_state_change_vs_v1_runtime"] >= 1


def test_latest_diagnostic_uses_corrected_state() -> None:
    features, weekly = _current_features()
    calendar = load_calendar(V1_1_CONFIG)
    corrected, _ = apply_causal_weekly_states(
        features,
        weekly,
        calendar,
        signal_start=calendar["trade_date"].min(),
    )
    diagnostic = latest_diagnostic(corrected)
    assert diagnostic["date"] == "2026-08-18"
    assert diagnostic["weekly_state_at_close"] == "W_BEAR"
    assert diagnostic["trend_entry_condition_met"] is False


def test_daily_guide_maps_registered_shadow_action_without_share_order() -> None:
    report = {
        "status": "COLLECTING",
        "current_view": "NO_VIEW_UNTIL_FORWARD_MATURITY",
        "as_of_market_date": "2026-08-21",
        "current_shadow_signal": {
            "signal_date": "2026-08-21",
            "reason": "TREND_ENTRY",
            "target_exposure": 0.325,
        },
    }
    diagnostic = {
        "date": "2026-08-21",
        "raw_close": 4.90,
        "adjusted_close": 4.90,
        "weekly_state_at_close": "W_BULL",
        "weekly_state_source_date": "2026-08-21",
        "is_completed_week_close": True,
        "donchian_entry_hh20": 4.85,
        "donchian_exit_ll10": 4.60,
        "atr14": 0.08,
        "trend_entry_condition_met": True,
        "close_to_entry_threshold_gap": 4.90 / 4.85 - 1.0,
    }
    guide = build_daily_guide(
        status_report=report,
        diagnostic=diagnostic,
        config=V1_1_CONFIG,
        shadow_state={"mode": "CASH", "tier": 0, "shares": 0, "equity_cny": 20000},
    )
    assert guide["decision"] == "SHADOW_ACTION_PENDING_NEXT_OPEN"
    assert guide["model_action"]["target_exposure"] == 0.325
    assert guide["model_action"]["share_quantity"] is None
    assert guide["real_money_boundary"]["system_authorization"] == "NOT_AUTHORIZED"


def test_v1_1_changes_no_strategy_parameter_and_keeps_safety_off() -> None:
    assert V1_1_CONFIG["protocol"]["historical_strategy_parameters_changed"] is False
    assert V1_1_CONFIG["protocol"]["forward_signal_start"] == "2026-08-19"
    assert V1_1_CONFIG["governance"]["live_position_mapping_enabled"] is False
    assert V1_1_CONFIG["governance"]["order_generation_enabled"] is False
    assert V1_1_CONFIG["governance"]["broker_connection_enabled"] is False
