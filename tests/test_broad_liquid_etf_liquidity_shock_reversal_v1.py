from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import (
    _pair_correlation,
    build_liquidity_shock_signals,
    etf_cost_rate,
    evaluate_historical_returns,
    load_contract,
    run_portfolio_backtest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "broad_liquid_etf_liquidity_shock_reversal_v1.yaml"


def contract() -> dict:
    return load_contract(CONFIG)


def test_contract_keeps_user_target_cost_and_research_boundary() -> None:
    value = contract()
    assert value["account"]["initial_capital_cny"] == 500_000.0
    assert value["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert value["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert value["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert value["signal"]["maximum_shock_score"] == -1.5
    assert value["portfolio"]["maximum_positions"] == 5
    assert value["portfolio"]["target_fraction_per_position"] == 0.20
    assert value["portfolio"]["maximum_pairwise_correlation"] == 0.95
    assert value["risk"]["leverage_allowed"] is False
    assert value["safety"]["order_generation"] is False
    assert value["safety"]["live_trading_authorized"] is False


def test_cost_rate_includes_user_fee_exchange_slippage_and_impact() -> None:
    value = contract()
    assert np.isclose(etf_cost_rate(value, scenario="base"), 0.00074)
    assert np.isclose(etf_cost_rate(value, scenario="stress"), 0.00194)


def test_signal_uses_prior_volatility_and_maps_to_next_open() -> None:
    dates = pd.bdate_range("2018-01-02", periods=275)
    log_returns = np.resize(np.array([0.001, -0.001]), len(dates))
    shock_index = 260
    log_returns[shock_index] = -0.02
    close = 10.0 * np.exp(np.cumsum(log_returns))
    panel_one = pd.DataFrame(
        {
            "date": dates,
            "con_code": "510001.SH",
            "total_return_open": close,
            "total_return_close": close,
            "raw_open": close,
            "raw_close": close,
            "is_suspended": False,
            "amount": 200_000_000.0,
        }
    )
    panel_two = panel_one.copy()
    panel_two["con_code"] = "510002.SH"
    panel = pd.concat([panel_one, panel_two], ignore_index=True)
    master = pd.DataFrame(
        {
            "ts_code": ["510001.SH", "510002.SH"],
            "name": ["测试ETF一", "测试ETF二"],
            "status": ["L", "L"],
            "list_date": [pd.Timestamp("2017-01-01"), pd.Timestamp("2017-01-01")],
            "delist_date": [pd.NaT, pd.NaT],
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "benchmark_open": 100.0,
            "benchmark_close": 100.0,
        }
    )
    signals, _, _, audit = build_liquidity_shock_signals(
        panel,
        master,
        benchmark,
        contract(),
        start=dates[252],
        end=dates[-5],
    )
    target = signals.loc[
        signals["signal_date"].eq(dates[shock_index])
        & signals["con_code"].eq("510001.SH")
    ]
    assert len(target) == 1
    assert target.iloc[0]["entry_date"] == dates[shock_index + 1]
    assert target.iloc[0]["scheduled_exit_date"] == dates[shock_index + 6]
    assert target.iloc[0]["shock_score"] <= -1.5
    assert audit["future_return_used_in_signal_construction"] is False


def test_pair_correlation_uses_only_history_through_signal_date() -> None:
    dates = pd.bdate_range("2020-01-02", periods=80)
    values = np.linspace(-0.02, 0.02, len(dates))
    returns = pd.DataFrame(
        {
            "ETF_A": values,
            "ETF_B": values * 2.0,
            "ETF_C": -values,
        },
        index=dates,
    )
    same = _pair_correlation(
        returns,
        "ETF_A",
        "ETF_B",
        dates[69],
        lookback=60,
        minimum_observations=50,
    )
    opposite = _pair_correlation(
        returns,
        "ETF_A",
        "ETF_C",
        dates[69],
        lookback=60,
        minimum_observations=50,
    )
    assert np.isclose(same, 1.0)
    assert np.isclose(opposite, -1.0)


def _small_path(*, block_first_exit: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=10)
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "benchmark_open": np.linspace(100.0, 101.0, len(dates)),
            "benchmark_close": np.linspace(100.1, 101.1, len(dates)),
        }
    )
    entry_date = dates[1]
    scheduled_exit = dates[6]
    signals = pd.DataFrame(
        [
            {
                "signal_date": dates[0],
                "entry_date": entry_date,
                "scheduled_exit_date": scheduled_exit,
                "con_code": "510001.SH",
                "shock_score": -2.0,
                "prior_median_amount_20": 200_000_000.0,
                "amount_ratio": 1.5,
                "entry_gap": 0.0,
            }
        ]
    )
    raw_open = np.full(len(dates), 10.0)
    prior_raw_close = np.full(len(dates), 10.0)
    if block_first_exit:
        raw_open[6] = 9.0
        prior_raw_close[6] = 10.0
        raw_open[7] = 9.1
        prior_raw_close[7] = 9.0
    market = pd.DataFrame(
        {
            "date": dates,
            "con_code": "510001.SH",
            "total_return_open": raw_open,
            "total_return_close": raw_open * 1.001,
            "raw_open": raw_open,
            "raw_close": raw_open * 1.001,
            "prior_raw_close": prior_raw_close,
            "amount": 200_000_000.0,
            "is_suspended": False,
        }
    )
    return_wide = pd.DataFrame({"510001.SH": np.zeros(len(dates))}, index=dates)
    return signals, market, benchmark, return_wide


def test_backtest_uses_board_lot_cost_capacity_and_scheduled_exit() -> None:
    signals, market, benchmark, return_wide = _small_path(block_first_exit=False)
    daily, trades, audit = run_portfolio_backtest(
        signals,
        market,
        benchmark,
        return_wide,
        contract(),
        start=benchmark["date"].iloc[0],
        end=benchmark["date"].iloc[-1],
    )
    assert len(daily) == len(benchmark)
    assert trades["side"].tolist() == ["BUY", "SELL"]
    assert int(trades.loc[trades["side"].eq("BUY"), "shares"].iloc[0]) % 100 == 0
    assert audit["entry_transaction_count"] == 1
    assert audit["exit_transaction_count"] == 1
    assert audit["blocked_exit_attempt_count"] == 0
    assert audit["maximum_capacity_fraction"] <= 0.01
    assert audit["maximum_gross_exposure"] <= 1.0


def test_blocked_exit_is_carried_to_first_eligible_open() -> None:
    signals, market, benchmark, return_wide = _small_path(block_first_exit=True)
    _, trades, audit = run_portfolio_backtest(
        signals,
        market,
        benchmark,
        return_wide,
        contract(),
        start=benchmark["date"].iloc[0],
        end=benchmark["date"].iloc[-1],
    )
    sell = trades.loc[trades["side"].eq("SELL")].iloc[0]
    assert sell["trade_date"] == benchmark["date"].iloc[7]
    assert sell["delayed_exit_days"] == 1
    assert audit["blocked_exit_attempt_count"] == 1


def _synthetic_daily(days: int, *, noise: float) -> pd.DataFrame:
    dates = pd.bdate_range("2016-01-04", periods=days)
    strategy_daily = (1.0 + 0.75) ** (1.0 / 242.0) - 1.0
    benchmark_daily = (1.0 + 0.05) ** (1.0 / 242.0) - 1.0
    alternating = np.resize(np.array([noise, -noise]), days)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "strategy_base_net_return": strategy_daily + alternating,
            "strategy_stress_net_return": strategy_daily * 0.96 + alternating,
            "benchmark_total_return": benchmark_daily,
            "gross_exposure": 0.8,
            "net_exposure": 0.8,
            "quality_complete": True,
            "capacity_pass": True,
        }
    )


def test_high_return_high_sharpe_history_passes_numeric_gates() -> None:
    result = evaluate_historical_returns(
        _synthetic_daily(968, noise=0.002),
        contract(),
        bootstrap_repetitions_override=80,
    )
    assert result["metrics"]["stress_annualized_excess"] > 0.40
    assert result["metrics"]["stress_strategy_net_sharpe"] > 1.50
    assert result["all_visible_gates_pass"] is True


def test_high_return_low_sharpe_history_fails() -> None:
    result = evaluate_historical_returns(
        _synthetic_daily(968, noise=0.035),
        contract(),
        bootstrap_repetitions_override=60,
    )
    assert result["metrics"]["stress_annualized_excess"] > 0.40
    assert result["metrics"]["stress_strategy_net_sharpe"] < 1.50
    assert result["all_visible_gates_pass"] is False
