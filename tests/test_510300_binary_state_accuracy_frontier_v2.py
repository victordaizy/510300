from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.binary_state_accuracy_frontier_v2 import (  # noqa: E402
    build_benchmark_equity,
    build_exact_count_predictions,
    build_severity_ranked_predictions,
    expand_block_predictions,
    load_config,
    simulate_binary_batch,
    summarize_batch,
    validate_config,
)
from research.binary_state_feasibility_v1 import (  # noqa: E402
    CostModel,
    build_benchmark_ledger,
    simulate_binary_path,
    summarize_path,
)


def _market(prices: list[float], benchmark: list[float] | None = None) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=len(prices))
    values = np.asarray(prices, dtype=float)
    benchmark_values = values if benchmark is None else np.asarray(benchmark, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "etf_open": values,
            "etf_high": values,
            "etf_low": values,
            "etf_close": values,
            "benchmark_close": benchmark_values,
        }
    )


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "record_date",
            "ex_date",
            "payment_date",
            "cash_dividend_per_share",
            "source",
        ]
    )


def _costs(*, slippage_bps: float = 10.0, cash_rate: float = 0.015) -> CostModel:
    return CostModel(
        commission_rate=0.0001,
        minimum_commission=0.0,
        slippage_bps=slippage_bps,
        cash_annual_rate=cash_rate,
        trading_days_per_year=242,
        lot_size=100,
    )


def test_config_keeps_exact_binary_scope_and_short_horizons() -> None:
    config = load_config()
    validate_config(config)
    assert config["scope"]["allowed_target_states"] == [0, 1]
    assert config["scope"]["allowed_holdings"] == ["510300.SH", "CASH_CNY"]
    assert config["objective"]["minimum_annualized_excess"] == 0.20
    assert config["accuracy_frontier"]["block_horizons_trading_days"] == [1, 5]


def test_exact_count_predictions_fix_error_counts() -> None:
    true_states = np.asarray([0] * 10 + [1] * 20, dtype=np.int8)
    predictions = build_exact_count_predictions(
        true_states,
        bad_block_recall=0.65,
        false_exit_rate=0.125,
        repetitions=25,
        rng=np.random.default_rng(7),
    )
    assert predictions.shape == (25, 30)
    assert np.all((predictions[:, :10] == 0).sum(axis=1) == 7)
    assert np.all((predictions[:, 10:] == 0).sum(axis=1) == 3)


def test_expand_block_predictions_keeps_first_row_cash() -> None:
    predictions = np.asarray([[1, 0], [0, 1]], dtype=np.int8)
    states = expand_block_predictions(predictions, horizon=3)
    assert states.shape == (7, 2)
    assert states[:, 0].tolist() == [0, 1, 1, 1, 0, 0, 0]
    assert states[:, 1].tolist() == [0, 0, 0, 0, 1, 1, 1]


def test_batch_simulator_matches_scalar_engine_for_multiple_paths() -> None:
    market = _market([10.0, 10.1, 9.9, 10.3, 10.0, 10.5, 10.2, 10.7])
    states = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )
    costs = _costs()
    batch = simulate_binary_batch(
        market,
        _empty_dividends(),
        states,
        costs=costs,
        initial_capital=500000.0,
        reinvest_paid_dividends=True,
    )
    for path_index in range(states.shape[1]):
        ledger, trades = simulate_binary_path(
            market,
            _empty_dividends(),
            states[:, path_index],
            costs=costs,
            initial_capital=500000.0,
            reinvest_paid_dividends=True,
        )
        assert np.allclose(batch.equity[:, path_index], ledger["equity"].to_numpy())
        expected_cost = 0.0 if trades.empty else float(
            trades["commission"].sum() + trades["slippage_cost"].sum()
        )
        assert np.isclose(batch.total_execution_cost[path_index], expected_cost)
        assert batch.trade_leg_count[path_index] == len(trades)


def test_batch_simulator_matches_scalar_dividend_entitlement_and_reinvestment() -> None:
    market = _market([10.0] * 7)
    dates = market["date"].tolist()
    dividends = pd.DataFrame(
        [
            {
                "symbol": "510300.SH",
                "record_date": dates[1],
                "ex_date": dates[2],
                "payment_date": dates[4],
                "cash_dividend_per_share": 1.5,
                "source": "合成测试",
            }
        ]
    )
    states = np.asarray(
        [
            [0, 0],
            [1, 1],
            [0, 1],
            [0, 1],
            [0, 1],
            [1, 1],
            [1, 0],
        ],
        dtype=np.int8,
    )
    costs = _costs(slippage_bps=0.0, cash_rate=0.0)
    batch = simulate_binary_batch(
        market,
        dividends,
        states,
        costs=costs,
        initial_capital=100000.0,
        reinvest_paid_dividends=True,
    )
    for path_index in range(states.shape[1]):
        ledger, _ = simulate_binary_path(
            market,
            dividends,
            states[:, path_index],
            costs=costs,
            initial_capital=100000.0,
            reinvest_paid_dividends=True,
        )
        assert np.allclose(batch.equity[:, path_index], ledger["equity"].to_numpy())


def test_batch_summary_matches_scalar_metrics() -> None:
    prices = list(np.linspace(10.0, 13.0, 300))
    benchmark = list(np.linspace(1000.0, 1120.0, 300))
    market = _market(prices, benchmark)
    states = np.zeros((300, 2), dtype=np.int8)
    states[1:101, 0] = 1
    states[150:260, 0] = 1
    states[20:200, 1] = 1
    costs = _costs()
    batch = simulate_binary_batch(
        market,
        _empty_dividends(),
        states,
        costs=costs,
        initial_capital=500000.0,
        reinvest_paid_dividends=True,
    )
    objective = {
        "annualization_trading_days": 242,
        "rolling_window_trading_days": 242,
        "minimum_annualized_excess": 0.20,
        "minimum_rolling_excess_median": 0.20,
    }
    metrics = summarize_batch(
        batch,
        states,
        build_benchmark_equity(market, 500000.0),
        objective=objective,
    )
    benchmark_ledger = build_benchmark_ledger(market, 500000.0)
    for path_index in range(states.shape[1]):
        ledger, trades = simulate_binary_path(
            market,
            _empty_dividends(),
            states[:, path_index],
            costs=costs,
            initial_capital=500000.0,
            reinvest_paid_dividends=True,
        )
        scalar = summarize_path(ledger, trades, benchmark_ledger, objective=objective)
        for column in (
            "strategy_cagr",
            "benchmark_cagr",
            "annualized_excess",
            "rolling_242d_excess_median",
            "rolling_242d_excess_minimum",
            "maximum_drawdown",
            "average_policy_state",
            "total_execution_cost_cny",
        ):
            assert np.isclose(metrics.loc[path_index, column], scalar[column])
        assert metrics.loc[path_index, "state_change_count"] == scalar["state_change_count"]
        assert metrics.loc[path_index, "trade_leg_count"] == scalar["trade_leg_count"]


def test_severity_ranked_predictions_take_largest_future_losses_first() -> None:
    blocks = pd.DataFrame(
        {
            "block_index": [0, 1, 2, 3, 4],
            "oracle_state": [0, 1, 0, 0, 1],
            "cash_gross_factor": [1.0] * 5,
            "full_gross_factor": [0.90, 1.02, 0.70, 0.80, 1.01],
        }
    )
    predictions, capture, selected_count = build_severity_ranked_predictions(
        blocks,
        bad_block_coverage=2.0 / 3.0,
    )
    assert selected_count == 2
    assert predictions.tolist() == [1, 1, 0, 0, 1]
    assert np.isclose(capture, 0.5 / 0.6)
