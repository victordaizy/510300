from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from research.digital_asset_current_quarter_basis_factor_v11 import (
    FIXED_SYMBOLS,
    build_daily_targets,
    delivery_days,
    load_contract,
    run_portfolio_backtest,
    validate_contract,
)


def _basis_frames(
    signal_times: pd.DatetimeIndex,
    btc_basis: list[float],
    eth_basis: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spot_rows: list[dict[str, object]] = []
    futures_rows: list[dict[str, object]] = []
    basis_by_symbol = {"BTCUSDT": btc_basis, "ETHUSDT": eth_basis}
    for symbol in FIXED_SYMBOLS:
        for timestamp, basis in zip(signal_times, basis_by_symbol[symbol], strict=True):
            futures_price = 100.0
            spot_price = futures_price * float(np.exp(basis))
            common = {
                "symbol": symbol,
                "open_time": timestamp,
                "quote_volume": 1_000_000_000.0,
            }
            spot_rows.append(
                {
                    **common,
                    "open": spot_price,
                    "high": spot_price,
                    "low": spot_price,
                    "close": spot_price,
                }
            )
            futures_rows.append(
                {
                    **common,
                    "open": futures_price,
                    "high": futures_price,
                    "low": futures_price,
                    "close": futures_price,
                }
            )
    return pd.DataFrame(spot_rows), pd.DataFrame(futures_rows)


def _hourly_futures_frame(times: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp in times:
        for symbol in FIXED_SYMBOLS:
            price = 100.0
            if symbol == "BTCUSDT" and timestamp >= pd.Timestamp("2021-09-07 00:00:00"):
                price = 101.0
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": timestamp,
                    "open": price,
                    "high": price * 1.002,
                    "low": price * 0.998,
                    "close": price,
                    "quote_volume": 20_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_contract_keeps_objective_basis_scale_cost_and_research_boundary() -> None:
    contract = load_contract()
    assert contract["account"]["initial_capital_cny"] == 500000.0
    assert contract["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["signal"]["basis_lookback_calendar_days"] == 5
    assert contract["signal"]["long_side_target_gross"] == 0.45
    assert contract["signal"]["short_side_target_gross"] == 0.45
    assert contract["signal"]["spot_position_allowed"] is False
    assert not any(contract["safety"].values())


def test_five_day_basis_ranking_is_strictly_lagged_and_deterministic() -> None:
    contract = copy.deepcopy(load_contract())
    contract["historical_partition"]["visible_start"] = "2021-09-06"
    contract["historical_partition"]["visible_end"] = "2021-09-06"
    contract["portfolio"]["terminal_winddown_start"] = "2021-09-07 00:00:00"
    times = pd.date_range("2021-09-01 23:00:00", periods=6, freq="1D")
    spot, futures = _basis_frames(
        times,
        [0.02, 0.02, 0.02, 0.02, 0.02, -1.0],
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.0],
    )
    targets, features, audit = build_daily_targets(spot, futures, contract)
    assert targets.loc[targets["target_weight"].gt(0.0), "symbol"].tolist() == [
        "BTCUSDT"
    ]
    assert targets.loc[targets["target_weight"].lt(0.0), "symbol"].tolist() == [
        "ETHUSDT"
    ]
    assert targets["target_weight"].abs().sum() == pytest.approx(0.90)
    assert targets["target_weight"].sum() == pytest.approx(0.0)
    assert targets["signal_time"].eq(pd.Timestamp("2021-09-05 23:00:00")).all()
    assert features.sort_values("mean_basis_rank_desc").iloc[0]["symbol"] == "BTCUSDT"
    assert audit["strict_execution_lag_hours"] == 1
    assert audit["future_price_or_return_used_in_signal"] is False


def test_last_quarter_friday_is_delivery_blackout() -> None:
    contract = copy.deepcopy(load_contract())
    contract["historical_partition"]["visible_start"] = "2021-09-24"
    contract["historical_partition"]["visible_end"] = "2021-09-24"
    contract["portfolio"]["terminal_winddown_start"] = "2021-09-25 00:00:00"
    times = pd.date_range("2021-09-19 23:00:00", periods=5, freq="1D")
    spot, futures = _basis_frames(
        times,
        [0.02] * 5,
        [0.00] * 5,
    )
    targets, _, audit = build_daily_targets(spot, futures, contract)
    assert list(delivery_days(contract)) == [pd.Timestamp("2021-09-24")]
    assert targets["delivery_blackout"].all()
    assert targets["target_weight"].eq(0.0).all()
    assert audit["delivery_blackout_day_count"] == 1


def test_full_engine_continues_capacity_and_reaches_terminal_zero() -> None:
    contract = copy.deepcopy(load_contract())
    contract["historical_partition"]["visible_start"] = "2021-09-06"
    contract["historical_partition"]["visible_end"] = "2021-09-08"
    contract["portfolio"]["terminal_winddown_start"] = "2021-09-08 00:00:00"
    contract["universe"]["turnover_lookback_observed_venue_hours"] = 2
    contract["portfolio"]["minimum_trade_notional_usdt"] = 1.0
    targets = pd.DataFrame(
        [
            {
                "signal_time": pd.Timestamp("2021-09-05 23:00:00"),
                "execution_time": pd.Timestamp("2021-09-06 00:00:00"),
                "symbol": "BTCUSDT",
                "target_weight": 0.45,
                "mean_log_basis_5d": 0.02,
                "delivery_blackout": False,
            },
            {
                "signal_time": pd.Timestamp("2021-09-05 23:00:00"),
                "execution_time": pd.Timestamp("2021-09-06 00:00:00"),
                "symbol": "ETHUSDT",
                "target_weight": -0.45,
                "mean_log_basis_5d": 0.00,
                "delivery_blackout": False,
            },
            {
                "signal_time": pd.Timestamp("2021-09-06 23:00:00"),
                "execution_time": pd.Timestamp("2021-09-07 00:00:00"),
                "symbol": "BTCUSDT",
                "target_weight": 0.45,
                "mean_log_basis_5d": 0.02,
                "delivery_blackout": False,
            },
            {
                "signal_time": pd.Timestamp("2021-09-06 23:00:00"),
                "execution_time": pd.Timestamp("2021-09-07 00:00:00"),
                "symbol": "ETHUSDT",
                "target_weight": -0.45,
                "mean_log_basis_5d": 0.00,
                "delivery_blackout": False,
            },
        ]
    )
    market_times = pd.date_range("2021-09-05 21:00:00", "2021-09-08 23:00:00", freq="1h")
    futures = _hourly_futures_frame(market_times)
    context_dates = pd.date_range("2021-09-01", "2021-09-09", freq="D")
    fx = pd.DataFrame(
        {
            "date": context_dates.to_numpy(dtype="datetime64[us]"),
            "cny_per_usd": 7.0,
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": context_dates.to_numpy(dtype="datetime64[ns]"),
            "close": 100.0,
        }
    )
    daily, trades, portfolio, context = run_portfolio_backtest(
        targets, futures, fx, benchmark, contract
    )
    assert len(daily) == 3
    assert not trades.empty
    assert portfolio["global_capacity_sliced_hour_count"] > 0
    assert portfolio["capacity_continuation_trade_hour_count"] > 0
    assert portfolio["maximum_order_capacity_fraction"] <= 0.001 + 1e-10
    assert portfolio["maximum_gross_exposure"] <= 1.05 + 1e-10
    assert portfolio["maximum_absolute_net_exposure"] <= 0.10 + 1e-10
    assert portfolio["terminal_position_count"] == 0
    assert portfolio["spot_position_count"] == 0
    assert portfolio["stress_total_transaction_cost_usdt"] > portfolio[
        "base_total_transaction_cost_usdt"
    ]
    assert context["strictly_lagged_fx"] is True
    assert np.isfinite(
        daily[["strategy_base_net_return", "strategy_stress_net_return"]].to_numpy(
            dtype=float
        )
    ).all()


def test_parameter_drift_is_rejected() -> None:
    contract = copy.deepcopy(load_contract())
    contract["signal"]["basis_lookback_calendar_days"] = 7
    with pytest.raises(ValueError, match="配置被弱化或损坏"):
        validate_contract(contract)
