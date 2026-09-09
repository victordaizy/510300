from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.extreme_two_level_martin_v1 import load_config, simulate_variant
from research.graph_regime_martin_turtle_v2 import load_config as load_source_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config()
SOURCE = load_source_config()


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_date": pd.Series(dtype="datetime64[ns]"),
            "ex_date": pd.Series(dtype="datetime64[ns]"),
            "payment_date": pd.Series(dtype="datetime64[ns]"),
            "cash_dividend_per_share": pd.Series(dtype=float),
        }
    )


def _features(second_close: float = 8.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-05", periods=6),
            "open": [10.0, 10.0, 8.0, 6.9, 7.0, 7.1],
            "close": [10.0, second_close, 6.9, 6.9, 7.0, 7.1],
            "adjusted_close": [10.0, second_close, 6.9, 6.9, 7.0, 7.1],
            "adjustment_factor": 1.0,
            "chart_state": "RANGE",
            "range_eligible": True,
            "trend_down": False,
            "turtle_breakout": False,
            "turtle_regular_exit": False,
            "bottom_divergence_active": [True, False, False, False, False, False],
            "bottom_divergence_active_id": ["D0001", None, None, None, None, None],
            "top_divergence_event": None,
            "atr14": 1.0,
            "ma20": 100.0,
            "ma28": 100.0,
        }
    )


def test_two_levels_are_50_and_100_percent_with_two_atr_spacing() -> None:
    result = simulate_variant(
        _features(),
        _empty_dividends(),
        SOURCE,
        "XM0_EXTREME_TWO_LEVEL_ONLY",
    )
    executions = result["executions"]
    assert executions["reason"].tolist() == [
        "EXTREME_LAYER_1",
        "EXTREME_LAYER_2",
        "EXTREME_HARD_INVALIDATION_EXIT",
    ]
    assert executions["target_exposure"].tolist() == pytest.approx([0.50, 1.00, 0.0])
    assert (executions["execution_date"] > executions["signal_date"]).all()
    assert result["ledger"]["layer"].max() == 2


def test_drop_smaller_than_two_atr_does_not_fill_second_layer() -> None:
    result = simulate_variant(
        _features(second_close=8.2),
        _empty_dividends(),
        SOURCE,
        "XM0_EXTREME_TWO_LEVEL_ONLY",
    )
    reasons = result["executions"]["reason"].tolist()
    assert "EXTREME_LAYER_1" in reasons
    assert "EXTREME_LAYER_2" not in reasons


def test_first_layer_has_economic_order_size_and_no_leverage() -> None:
    result = simulate_variant(
        _features(),
        _empty_dividends(),
        SOURCE,
        "XM0_EXTREME_TWO_LEVEL_ONLY",
    )
    first = result["executions"].iloc[0]
    one_way_bps = 10_000.0 * (first["commission"] + first["slippage_cost"]) / first[
        "raw_notional"
    ]
    assert first["raw_notional"] == pytest.approx(10_000.0)
    assert one_way_bps == pytest.approx(10.0)
    assert result["ledger"]["exposure"].dropna().max() <= 1.0


def test_full_position_cycle_records_pre_and_post_full_pnl() -> None:
    result = simulate_variant(
        _features(),
        _empty_dividends(),
        SOURCE,
        "XM0_EXTREME_TWO_LEVEL_ONLY",
    )
    cycles = result["cycles"]
    assert len(cycles) == 1
    assert cycles.loc[0, "full_position_reason"] == "EXTREME_LAYER_2"
    assert pd.notna(cycles.loc[0, "pnl_before_full"])
    assert pd.notna(cycles.loc[0, "pnl_after_full"])


def test_governance_disables_live_actions() -> None:
    governance = CONFIG["governance"]
    assert governance["live_position_mapping_enabled"] is False
    assert governance["order_generation_enabled"] is False
    assert governance["broker_connection_enabled"] is False
    assert CONFIG["protocol"]["true_forward_start"] is None
