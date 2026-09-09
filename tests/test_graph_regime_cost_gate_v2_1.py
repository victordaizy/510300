from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from research.graph_regime_cost_gate_v2_1 import (
    load_cost_gate_config,
    load_source_config,
    preview_buy_cost,
    simulate_variant_cost_gated,
)
from research.graph_regime_martin_turtle_v2 import _cost_model


COST_GATE = load_cost_gate_config()
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


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-05", periods=5),
            "open": [10.0, 10.0, 4.0, 4.1, 4.2],
            "close": [10.0, 4.0, 4.0, 4.1, 4.2],
            "adjusted_close": [10.0, 4.0, 4.0, 4.1, 4.2],
            "adjustment_factor": 1.0,
            "chart_state": ["RANGE", "TREND_DOWN", "RANGE", "RANGE", "RANGE"],
            "range_eligible": [True, False, True, True, True],
            "trend_down": [False, True, False, False, False],
            "turtle_breakout": False,
            "turtle_regular_exit": False,
            "bottom_divergence_event": ["D0001", None, None, None, None],
            "bottom_divergence_active": [True, False, False, False, False],
            "bottom_divergence_active_id": ["D0001", None, None, None, None],
            "bottom_divergence_invalidation": [2.0, np.nan, np.nan, np.nan, np.nan],
            "top_divergence_event": None,
            "bias28": -5.0,
            "atr14": 1.0,
            "er20": 0.2,
            "ma20": 20.0,
            "ma28": 20.0,
        }
    )


def test_ten_bp_gate_algebra_implies_ten_thousand_yuan_minimum_buy() -> None:
    costs = _cost_model(SOURCE, 1.0)
    below = preview_buy_cost(
        target_exposure=0.50,
        raw_open=10.0,
        cash=19_999.0,
        shares=0,
        receivable=0.0,
        lot_size=100,
        costs=costs,
    )
    at_threshold = 10_000.0 * (5.0 / 10_000.0 + 0.0005)
    assert below is not None
    assert below["raw_notional"] == pytest.approx(9_000.0)
    assert below["one_way_explicit_cost_bps"] > 10.0
    assert at_threshold == pytest.approx(10.0)


def test_small_martingale_entry_is_blocked_and_event_is_not_retried() -> None:
    result = simulate_variant_cost_gated(
        _features(),
        _empty_dividends(),
        SOURCE,
        COST_GATE,
        "M0_CAPPED_MARTINGALE_ONLY",
    )
    assert result["executions"].empty
    blocked = result["blocked_orders"]
    assert len(blocked) == 1
    assert blocked.loc[0, "reason"] == "MARTINGALE_LAYER_1"
    assert blocked.loc[0, "intended_raw_notional"] < 10_000.0
    assert blocked.loc[0, "one_way_explicit_cost_bps"] > 10.0
    assert result["ledger"]["shares"].eq(0).all()


def test_full_size_turtle_entry_is_allowed_and_remains_next_open() -> None:
    features = _features()
    features.loc[:, "range_eligible"] = False
    features.loc[:, "bottom_divergence_active"] = False
    features.loc[:, "bottom_divergence_active_id"] = None
    features.loc[0, "turtle_breakout"] = True
    features.loc[1, "turtle_regular_exit"] = True
    result = simulate_variant_cost_gated(
        features,
        _empty_dividends(),
        SOURCE,
        COST_GATE,
        "T0_DONCHIAN_20_10",
    )
    executions = result["executions"]
    assert executions["reason"].tolist() == ["TURTLE_ENTRY", "TURTLE_REGULAR_EXIT"]
    assert (executions["execution_date"] > executions["signal_date"]).all()
    assert result["blocked_orders"].empty


def test_small_risk_exit_is_never_blocked() -> None:
    source = deepcopy(SOURCE)
    source["price_and_execution"]["initial_capital_cny"] = 100_000.0
    result = simulate_variant_cost_gated(
        _features(),
        _empty_dividends(),
        source,
        COST_GATE,
        "M0_CAPPED_MARTINGALE_ONLY",
    )
    executions = result["executions"]
    assert executions["reason"].tolist() == [
        "MARTINGALE_LAYER_1",
        "MODULE_TREND_DOWN_EXIT",
    ]
    sell = executions.loc[executions["side"].eq("SELL")].iloc[0]
    sell_cost_bps = 10_000.0 * (sell["commission"] + sell["slippage_cost"]) / sell[
        "raw_notional"
    ]
    assert sell["raw_notional"] < 10_000.0
    assert sell_cost_bps > 10.0
    assert result["blocked_orders"].empty


def test_cost_gate_remains_research_only() -> None:
    governance = COST_GATE["governance"]
    assert governance["live_position_mapping_enabled"] is False
    assert governance["order_generation_enabled"] is False
    assert governance["broker_connection_enabled"] is False
    assert COST_GATE["protocol"]["true_forward_start"] is None
