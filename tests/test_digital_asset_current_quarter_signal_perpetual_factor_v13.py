from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from research.digital_asset_current_quarter_signal_perpetual_factor_v13 import (
    FIXED_SYMBOLS,
    build_daily_targets,
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
        for timestamp, basis in zip(
            signal_times, basis_by_symbol[symbol], strict=True
        ):
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


def _perpetual_frame(times: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp in times:
        for symbol in FIXED_SYMBOLS:
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": timestamp,
                    "open": 100.0,
                    "high": 100.1,
                    "low": 99.9,
                    "close": 100.0,
                    "quote_volume": 1_000_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_contract_keeps_objective_signal_cost_funding_and_boundary() -> None:
    contract = load_contract()
    assert contract["account"]["initial_capital_cny"] == 500000.0
    assert contract["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["signal"]["basis_lookback_calendar_days"] == 5
    assert contract["signal"]["long_side_target_gross"] == 0.45
    assert contract["signal"]["short_side_target_gross"] == 0.45
    assert contract["costs"]["base_total_perpetual_cost_bps_per_leg"] == 4.0
    assert contract["costs"]["stress_total_perpetual_cost_bps_per_leg"] == 16.0
    assert contract["universe"]["actual_funding_history_required"] is True
    assert not any(contract["safety"].values())


def test_five_day_current_quarter_basis_is_lagged_and_trades_perpetual_targets() -> None:
    contract = copy.deepcopy(load_contract())
    contract["historical_partition"]["visible_start"] = "2021-09-06"
    contract["historical_partition"]["visible_end"] = "2021-09-06"
    contract["portfolio"]["terminal_winddown_start"] = "2021-09-07 00:00:00"
    times = pd.date_range("2021-09-01 23:00:00", periods=6, freq="1D")
    spot, current_quarter = _basis_frames(
        times,
        [0.02, 0.02, 0.02, 0.02, 0.02, -1.0],
        [0.00, 0.00, 0.00, 0.00, 0.00, 1.0],
    )
    targets, features, audit = build_daily_targets(
        spot, current_quarter, contract
    )
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
    assert audit["execution_instrument"] == "USDS_MARGINED_PERPETUAL"
    assert audit["future_price_or_return_used_in_signal"] is False


def test_perpetual_engine_applies_funding_stress_and_terminal_zero() -> None:
    contract = copy.deepcopy(load_contract())
    contract["historical_partition"]["visible_start"] = "2021-09-06"
    contract["historical_partition"]["visible_end"] = "2021-09-08"
    contract["portfolio"]["terminal_winddown_start"] = "2021-09-08 00:00:00"
    contract["universe"]["turnover_lookback_hours"] = 2
    contract["portfolio"]["minimum_trade_notional_usdt"] = 1.0
    signal_times = pd.date_range("2021-09-01 23:00:00", periods=7, freq="1D")
    spot, current_quarter = _basis_frames(
        signal_times,
        [0.02] * len(signal_times),
        [0.00] * len(signal_times),
    )
    targets, _, _ = build_daily_targets(spot, current_quarter, contract)
    market_times = pd.date_range(
        "2021-09-05 21:00:00", "2021-09-08 23:00:00", freq="1h"
    )
    perpetual = _perpetual_frame(market_times)
    funding_rows: list[dict[str, object]] = []
    for timestamp in pd.date_range(
        "2021-09-06 08:00:00", "2021-09-08 16:00:00", freq="8h"
    ):
        for symbol in FIXED_SYMBOLS:
            funding_rows.append(
                {
                    "symbol": symbol,
                    "funding_time": timestamp,
                    "funding_rate": 0.001,
                    "mark_price": 100.0,
                }
            )
    funding = pd.DataFrame(funding_rows)
    dates = pd.date_range("2021-08-29", "2021-09-09", freq="D")
    fx = pd.DataFrame(
        {"date": dates.to_numpy(dtype="datetime64[us]"), "cny_per_usd": 7.0}
    )
    benchmark = pd.DataFrame(
        {"date": dates.to_numpy(dtype="datetime64[ns]"), "close": 100.0}
    )
    daily, trades, portfolio, context = run_portfolio_backtest(
        targets, perpetual, funding, fx, benchmark, contract
    )
    assert len(daily) == 3
    assert not trades.empty
    assert portfolio["maximum_order_capacity_fraction"] <= 0.001 + 1e-10
    assert portfolio["maximum_gross_exposure"] <= 1.05 + 1e-10
    assert portfolio["maximum_absolute_net_exposure"] <= 0.10 + 1e-10
    assert portfolio["terminal_position_count"] == 0
    assert portfolio["spot_position_count"] == 0
    assert portfolio["current_quarter_position_count"] == 0
    assert portfolio["actual_funding_history_applied"] is True
    assert portfolio["base_total_funding_usdt"] == pytest.approx(0.0, abs=1e-9)
    assert portfolio["stress_total_funding_usdt"] < 0.0
    assert portfolio["stress_total_transaction_cost_usdt"] > portfolio[
        "base_total_transaction_cost_usdt"
    ]
    assert context["strictly_lagged_fx"] is True
    assert np.isfinite(
        daily[
            ["strategy_base_net_return", "strategy_stress_net_return"]
        ].to_numpy(dtype=float)
    ).all()


def test_parameter_drift_is_rejected() -> None:
    contract = copy.deepcopy(load_contract())
    contract["costs"]["stress_funding_payment_multiplier"] = 1.0
    with pytest.raises(ValueError, match="配置被弱化或损坏"):
        validate_contract(contract)
