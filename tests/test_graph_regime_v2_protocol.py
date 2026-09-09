from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.build_510300_complete_dividends import EVENTS
from scripts.freeze_graph_regime_v2_protocol import validate


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (ROOT / "config" / "graph_regime_martin_turtle_v2.yaml").read_text(
        encoding="utf-8"
    )
)


def test_v2_freezes_five_candidates_and_external_turtle_baseline() -> None:
    validate(CONFIG)
    assert CONFIG["protocol"]["external_frozen_baseline"] == "T0_DONCHIAN_20_10"
    assert [candidate["id"] for candidate in CONFIG["candidates"]] == [
        "T1_TURTLE_GRAPH_EXIT",
        "M0_CAPPED_MARTINGALE_ONLY",
        "P0_BIAS28_SWING_ONLY",
        "S1_MARTINGALE_TURTLE_SWITCH",
        "S2_BIAS28_TURTLE_SWITCH",
    ]


def test_martingale_is_capped_at_three_layers_and_87_5_percent() -> None:
    module = CONFIG["martingale_module"]
    assert module["maximum_layers"] == 3
    assert module["unlimited_doubling_allowed"] is False
    assert module["tranche_weights_of_current_equity"] == [0.125, 0.25, 0.50]
    assert module["cumulative_target_exposures"] == [0.125, 0.375, 0.875]


def test_pring_style_branch_freezes_bias28_without_averaging_down() -> None:
    module = CONFIG["pring_style_module"]
    assert module["bias_definition"] == "100*(close/MA28-1)"
    assert "BIAS28<-10" in module["entry_gate"]
    assert module["target_exposure"] == pytest.approx(0.50)
    assert module["averaging_down_allowed"] is False


def test_divergence_requires_independent_confirmed_non_reused_pivots() -> None:
    divergence = CONFIG["graph_divergence"]
    assert divergence["price_and_dif_pivots_identified_independently"] is True
    assert divergence["pivot_reuse_allowed"] is False
    assert divergence["confirmation_lag_bars"] == 3
    assert divergence["indicator_pair_tolerance_bars"] == 3
    assert divergence["price_swing_minimum_atr"] == pytest.approx(1.0)


def test_all_live_execution_paths_remain_disabled() -> None:
    governance = CONFIG["governance"]
    assert governance["live_position_mapping_enabled"] is False
    assert governance["order_generation_enabled"] is False
    assert governance["broker_connection_enabled"] is False
    assert governance["live_trading_authorized"] is False
    assert CONFIG["protocol"]["true_forward_start"] is None


def test_complete_dividend_registry_has_fourteen_unique_events() -> None:
    assert len(EVENTS) == 14
    ex_dates = [event[1] for event in EVENTS]
    assert ex_dates == sorted(ex_dates)
    assert len(set(ex_dates)) == len(ex_dates)
    assert ex_dates[0] == "2012-12-18"
    assert ex_dates[-1] == "2026-01-19"

