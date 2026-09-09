from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.graph_regime_martin_turtle_v2 import (
    add_point_in_time_adjusted_ohlc,
    build_features,
    load_config,
    load_inputs,
    simulate_variant,
)
from scripts.run_graph_regime_martin_turtle_v2 import _markdown_report


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config()


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype=str),
            "record_date": pd.Series(dtype="datetime64[ns]"),
            "ex_date": pd.Series(dtype="datetime64[ns]"),
            "payment_date": pd.Series(dtype="datetime64[ns]"),
            "cash_dividend_per_share": pd.Series(dtype=float),
            "source": pd.Series(dtype=str),
        }
    )


def _synthetic_features() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=7)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 10.0, 8.5, 7.0, 6.8, 6.7, 6.6],
            "close": [10.0, 8.5, 7.0, 6.8, 6.7, 6.6, 6.5],
            "adjusted_close": [10.0, 8.5, 7.0, 6.8, 6.7, 6.6, 6.5],
            "adjustment_factor": 1.0,
            "chart_state": "RANGE",
            "range_eligible": True,
            "trend_down": False,
            "turtle_breakout": False,
            "turtle_regular_exit": False,
            "bottom_divergence_event": ["D0001", None, None, None, None, None, None],
            "bottom_divergence_active": [True, False, False, False, False, False, False],
            "bottom_divergence_active_id": [
                "D0001",
                None,
                None,
                None,
                None,
                None,
                None,
            ],
            "bottom_divergence_invalidation": [5.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            "top_divergence_event": None,
            "bias28": -5.0,
            "atr14": 1.0,
            "er20": 0.2,
            "ma20": 100.0,
            "ma28": 100.0,
        }
    )
    return frame


def test_cash_distribution_adjustment_removes_ex_date_price_gap() -> None:
    market = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
            "open": [10.0, 9.0],
            "high": [10.0, 9.0],
            "low": [10.0, 9.0],
            "close": [10.0, 9.0],
        }
    )
    dividends = pd.DataFrame(
        {
            "record_date": pd.to_datetime(["2026-01-05"]),
            "ex_date": pd.to_datetime(["2026-01-06"]),
            "payment_date": pd.to_datetime(["2026-01-07"]),
            "cash_dividend_per_share": [1.0],
        }
    )
    adjusted = add_point_in_time_adjusted_ohlc(market, dividends)
    assert adjusted.loc[0, "adjusted_close"] == pytest.approx(9.0)
    assert adjusted.loc[1, "adjusted_close"] == pytest.approx(9.0)


def test_frozen_history_divergences_are_confirmed_and_pivots_are_not_reused() -> None:
    market, _, dividends = load_inputs(ROOT, CONFIG)
    _, events = build_features(market, dividends, CONFIG)
    assert len(events) == 15
    assert events["event_id"].tolist() == [f"D{number:04d}" for number in range(1, 16)]
    assert events["confirmation_date"].is_monotonic_increasing
    assert events["confirmation_lag_from_price_bars"].ge(3).all()
    price_pivots = pd.concat(
        [events["previous_price_pivot_index"], events["current_price_pivot_index"]]
    )
    dif_pivots = pd.concat(
        [events["previous_dif_pivot_index"], events["current_dif_pivot_index"]]
    )
    assert price_pivots.is_unique
    assert dif_pivots.is_unique


def test_capped_martingale_has_three_next_open_layers_and_never_exceeds_87_5_percent_target() -> None:
    result = simulate_variant(
        _synthetic_features(),
        _empty_dividends(),
        CONFIG,
        "M0_CAPPED_MARTINGALE_ONLY",
    )
    executions = result["executions"]
    assert executions["reason"].tolist() == [
        "MARTINGALE_LAYER_1",
        "MARTINGALE_LAYER_2",
        "MARTINGALE_LAYER_3",
    ]
    assert executions["target_exposure"].tolist() == pytest.approx([0.125, 0.375, 0.875])
    assert executions["target_exposure"].max() <= 0.875
    assert (executions["execution_date"] > executions["signal_date"]).all()
    assert result["ledger"]["martin_layer"].max() == 3


def test_turtle_entry_and_exit_execute_on_following_trading_day() -> None:
    features = _synthetic_features()
    features.loc[:, "range_eligible"] = False
    features.loc[:, "bottom_divergence_active"] = False
    features.loc[:, "bottom_divergence_active_id"] = None
    features.loc[0, "turtle_breakout"] = True
    features.loc[1, "turtle_regular_exit"] = True
    result = simulate_variant(
        features,
        _empty_dividends(),
        CONFIG,
        "T0_DONCHIAN_20_10",
    )
    executions = result["executions"]
    assert executions["reason"].tolist() == ["TURTLE_ENTRY", "TURTLE_REGULAR_EXIT"]
    assert (executions["execution_date"] > executions["signal_date"]).all()
    assert len(result["cycles"]) == 1


def test_bias28_branch_enters_once_without_averaging_down() -> None:
    features = _synthetic_features()
    features.loc[0, "bias28"] = -11.0
    result = simulate_variant(
        features,
        _empty_dividends(),
        CONFIG,
        "P0_BIAS28_SWING_ONLY",
    )
    executions = result["executions"]
    assert executions["reason"].tolist() == ["PRING_ENTRY"]
    assert executions["target_exposure"].tolist() == pytest.approx([0.50])


def test_martingale_to_turtle_transition_stays_in_one_auditable_cycle() -> None:
    features = _synthetic_features()
    features.loc[1, "turtle_breakout"] = True
    features.loc[2, "turtle_regular_exit"] = True
    result = simulate_variant(
        features,
        _empty_dividends(),
        CONFIG,
        "S1_MARTINGALE_TURTLE_SWITCH",
    )
    executions = result["executions"]
    assert executions["reason"].tolist() == [
        "MARTINGALE_LAYER_1",
        "MARTIN_TO_TURTLE",
        "TURTLE_REGULAR_EXIT",
    ]
    cycles = result["cycles"]
    assert len(cycles) == 1
    assert cycles.loc[0, "switch_reason"] == "MARTIN_TO_TURTLE"
    assert pd.notna(cycles.loc[0, "pnl_to_switch"])
    assert pd.notna(cycles.loc[0, "pnl_after_switch"])


def test_every_execution_charges_commission_and_slippage() -> None:
    result = simulate_variant(
        _synthetic_features(),
        _empty_dividends(),
        CONFIG,
        "M0_CAPPED_MARTINGALE_ONLY",
    )
    executions = result["executions"]
    assert executions["commission"].ge(5.0).all()
    assert executions["slippage_cost"].gt(0.0).all()


def test_safety_switches_remain_disabled() -> None:
    governance = CONFIG["governance"]
    assert governance["live_position_mapping_enabled"] is False
    assert governance["order_generation_enabled"] is False
    assert governance["broker_connection_enabled"] is False
    assert CONFIG["protocol"]["true_forward_start"] is None


def test_markdown_report_renders_unavailable_sharpe_as_na() -> None:
    report = {
        "data_cutoff": "2026-08-14",
        "evidence_label": "HISTORICALLY_CONTAMINATED",
        "registered_pass_count": 0,
        "variants": {
            "P0_BIAS28_SWING_ONLY": {
                "decision": "INSUFFICIENT_EVIDENCE",
                "closed_trades": 0,
                "base": {
                    "cagr": 0.01,
                    "maximum_drawdown": 0.0,
                    "sharpe": None,
                    "average_exposure": 0.0,
                },
                "timing_contribution_after_cost": 0.0,
            }
        },
        "buy_hold": {"cagr": 0.05, "maximum_drawdown": -0.40, "sharpe": 0.3},
        "h00300": {"cagr": 0.05, "sharpe": 0.3},
    }
    rendered = _markdown_report(report, {"bottom": 8, "top": 7})
    assert "N/A" in rendered
