"""510300方向切换V1冻结协议与核心计算的回归测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.direction_switch_common_v1 import (
    execution_targets_from_intervals,
    holm_bonferroni,
    load_config,
    maximum_round_trips_per_year,
    normalize_dividends,
    normalize_market,
    prepare_total_return_market,
    rolling_prior_percentile,
    simulate_binary_execution_targets,
    verify_manifest,
)
from research.direction_switch_diagnostics_v1 import (
    _build_decision_blocks,
    _negative_episodes,
)
from research.if_forced_flow_state_v1 import build_state_machine, evaluate_mechanism


def _market(rows: int = 12) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=rows)
    close = np.asarray(
        [4.00, 4.10, 4.00, 3.90, 3.95, 4.05, 4.10, 4.00, 3.85, 3.90, 4.00, 4.10]
    )[:rows]
    open_price = close * np.asarray(
        [1.0, 0.99, 1.01, 1.0, 0.99, 1.0, 1.01, 1.0, 0.99, 1.0, 1.0, 1.0]
    )[:rows]
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_price,
            "high": np.maximum(open_price, close) + 0.05,
            "low": np.minimum(open_price, close) - 0.05,
            "close": close,
            "volume": np.arange(rows, dtype=float) + 1000.0,
            "amount": (np.arange(rows, dtype=float) + 1000.0) * close,
            "symbol": "510300.SH",
        }
    )


def _dividends() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["510300.SH"],
            "record_date": [pd.Timestamp("2025-01-03")],
            "ex_date": [pd.Timestamp("2025-01-06")],
            "payment_date": [pd.Timestamp("2025-01-08")],
            "cash_dividend_per_share": [0.10],
        }
    )


def test_frozen_protocol_and_manifest() -> None:
    config = load_config()
    manifest = verify_manifest(config)
    assert manifest["historical_runs_consumed"] == 0
    assert config["protocol"]["trade_assets"] == ["510300.SH", "CASH_CNY"]
    assert config["execution"]["initial_capital_cny"] == 200000.0
    assert config["execution"]["commission_rate_per_leg"] == 0.0002
    assert config["if_signal"]["entry_basis_percentile_max"] == 0.05
    assert config["if_signal"]["entry_open_interest_percentile_min"] == 0.70
    assert config["if_signal"]["exhaustion_basis_percentile_min"] == 0.25
    assert config["if_signal"]["exhaustion_open_interest_percentile_max"] == 0.50
    assert config["mechanism_tests"]["primary_horizon_trading_days"] == 10
    assert config["portfolio_gate"]["hard_gates"]["net_sharpe_min"] == 1.20
    assert config["protocol"]["live_trading_authorized"] is False


def test_total_return_contribution_identity_includes_dividend() -> None:
    market = normalize_market(_market(), "510300.SH")
    dividends = normalize_dividends(_dividends())
    prepared = prepare_total_return_market(market, dividends)
    valid = prepared["total_return"].notna()
    error = (
        prepared.loc[valid, "total_return"]
        - prepared.loc[valid, "overnight_contribution"]
        - prepared.loc[valid, "intraday_contribution"]
    )
    assert error.abs().max() < 1e-12
    ex_row = prepared.loc[prepared["date"].eq(pd.Timestamp("2025-01-06"))].iloc[0]
    assert ex_row["cash_dividend_per_share"] == 0.10


def test_rolling_percentile_uses_only_prior_values() -> None:
    values = pd.Series([1.0, 2.0, 100.0, 3.0])
    result = rolling_prior_percentile(values, window=3, minimum_observations=2)
    assert np.isnan(result.iloc[0])
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == 1.0
    assert result.iloc[3] == 2.0 / 3.0


def test_binary_simulator_enforces_lots_and_cash_state() -> None:
    config = load_config()
    market = prepare_total_return_market(
        normalize_market(_market(), "510300.SH"), normalize_dividends(_dividends())
    )
    targets = np.ones(len(market), dtype=float)
    targets[0] = 0.0
    targets[4:7] = 0.0
    metrics, daily = simulate_binary_execution_targets(
        market,
        normalize_dividends(_dividends()),
        targets,
        config["execution"],
        metric_start_index=1,
        return_daily=True,
    )
    assert len(metrics) == 1
    assert daily is not None and daily.shape == (1, len(market) - 1)
    assert metrics.iloc[0]["trade_count"] >= 3
    assert metrics.iloc[0]["cash_entry_count"] == 1
    assert np.isfinite(metrics.iloc[0]["ending_equity_cny"])


def test_intervals_and_round_trip_count() -> None:
    market = _market()
    intervals = [
        (market.iloc[3]["date"], market.iloc[5]["date"]),
        (market.iloc[8]["date"], market.iloc[10]["date"]),
    ]
    target = execution_targets_from_intervals(
        market["date"], intervals, metric_start_index=1
    )
    counts = maximum_round_trips_per_year(market["date"], target)
    assert counts == {2025: 2}
    assert target[3] == 0.0 and target[5] == 1.0


def test_state_machine_requires_two_day_exhaustion_confirmation() -> None:
    config = load_config()
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=7),
            "data_eligible": True,
            "etf_past_3d_total_return": [-0.01] * 7,
            "basis_residual_percentile": [0.50, 0.04, 0.04, 0.20, 0.30, 0.31, 0.50],
            "open_interest_shock_percentile": [0.50, 0.80, 0.80, 0.60, 0.40, 0.40, 0.50],
        }
    )
    states = build_state_machine(config, frame)
    assert states.loc[1, "pressure_start_event"]
    assert states.loc[4, "state"] == "FORCED_SELLING_ACTIVE"
    assert states.loc[5, "pressure_exhaustion_event"]
    assert states.loc[5, "state"] == "FORCED_SELLING_EXHAUSTED"
    assert states.loc[6, "state"] == "NORMAL"


def test_decision_blocks_merge_adjacent_negative_blocks() -> None:
    market = prepare_total_return_market(
        normalize_market(_market(), "510300.SH"), normalize_dividends(_dividends())
    )
    blocks = _build_decision_blocks(market, metric_start_index=1, horizon=2)
    episodes = _negative_episodes(blocks)
    assert not blocks.empty
    assert set(blocks["negative_block"].unique()).issubset({True, False})
    assert (episodes["avoided_gross_log_loss_score"] > 0.0).all()


def test_holm_adjustment_is_monotone_and_bounded() -> None:
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.03, "c": None, "d": 0.20})
    assert adjusted["a"] == 0.03
    assert adjusted["b"] >= adjusted["a"]
    assert adjusted["d"] <= 1.0
    assert adjusted["c"] is None


def test_if_mechanism_stops_before_future_returns_when_event_count_fails() -> None:
    config = load_config()
    dates = pd.bdate_range("2015-01-05", periods=100)
    states = pd.DataFrame(
        {
            "date": dates,
            "pressure_start_event": False,
            "pressure_exhaustion_event": False,
        }
    )
    states.loc[[10, 50], "pressure_start_event"] = True
    states.loc[[30, 70], "pressure_exhaustion_event"] = True
    report, events, controls = evaluate_mechanism(config, states)
    assert report["return_evaluation"] == "NOT_ALLOWED"
    assert report["future_return_columns_read"] == 0
    assert report["matching_run"] is False
    assert events.empty and controls.empty
