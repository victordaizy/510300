from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from research.digital_asset_cross_sectional_hourly_reversal_v8 import FIXED_SYMBOLS
from research.digital_asset_cross_sectional_perpetual_basis_v9 import (
    build_daily_targets,
    load_contract,
    run_portfolio_backtest,
    validate_contract,
)


def _market_frame(
    times: pd.DatetimeIndex,
    *,
    quote_volume: float = 1_000_000_000.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp in times:
        price = 102.0 if timestamp == pd.Timestamp("2020-01-03 01:00:00") else 100.0
        for symbol in FIXED_SYMBOLS:
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": timestamp,
                    "open": price,
                    "high": price * 1.001,
                    "low": price * 0.999,
                    "close": price,
                    "quote_volume": quote_volume,
                }
            )
    return pd.DataFrame(rows)


def _spot_signal_frame(
    signal_times: pd.DatetimeIndex,
    basis_by_symbol: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp in signal_times:
        for symbol in FIXED_SYMBOLS:
            price = 100.0 * (1.0 + basis_by_symbol[symbol])
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": timestamp,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "quote_volume": 1_000_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_contract_keeps_objective_factor_scale_and_research_boundary() -> None:
    contract = load_contract()
    assert contract["account"]["initial_capital_cny"] == 500000.0
    assert contract["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["signal"]["long_side_target_gross"] == 0.5
    assert contract["signal"]["short_side_target_gross"] == 0.5
    assert contract["signal"]["spot_position_allowed"] is False
    assert not any(contract["safety"].values())


def test_basis_ranking_is_deterministic_disjoint_and_next_day_executed() -> None:
    contract = copy.deepcopy(load_contract())
    contract["historical_partition"]["visible_start"] = "2020-01-03"
    contract["historical_partition"]["visible_end"] = "2020-01-05"
    contract["portfolio"]["terminal_winddown_start"] = "2020-01-05 00:00:00"
    alphabetical = sorted(FIXED_SYMBOLS)
    basis = {symbol: 0.0 for symbol in FIXED_SYMBOLS}
    for symbol in alphabetical[:4]:
        basis[symbol] = -0.02
    for symbol in ("XRPUSDT", "XTZUSDT", "ZECUSDT"):
        basis[symbol] = 0.02
    signal_times = pd.date_range("2020-01-02 23:00:00", "2020-01-03 23:00:00", freq="1D")
    spot = _spot_signal_frame(signal_times, basis)
    perpetual = _market_frame(signal_times)
    targets, features, audit = build_daily_targets(spot, perpetual, contract)
    first = targets.loc[targets["execution_time"].eq(pd.Timestamp("2020-01-03 00:00:00"))]
    longs = sorted(first.loc[first["target_weight"].gt(0.0), "symbol"])
    shorts = sorted(first.loc[first["target_weight"].lt(0.0), "symbol"])
    assert longs == ["XRPUSDT", "XTZUSDT", "ZECUSDT"]
    assert shorts == alphabetical[:3]
    assert set(longs).isdisjoint(shorts)
    assert first["target_weight"].clip(lower=0.0).sum() == pytest.approx(0.5)
    assert first["target_weight"].clip(upper=0.0).sum() == pytest.approx(-0.5)
    assert (first["execution_time"] - first["signal_time"]).eq(pd.Timedelta(hours=1)).all()
    assert audit["strict_execution_lag_hours"] == 1
    assert audit["spot_position_generated"] is False
    assert len(features) == 2 * len(FIXED_SYMBOLS)


def test_full_engine_applies_risk_deleveraging_funding_stress_and_terminal_zero() -> None:
    contract = copy.deepcopy(load_contract())
    contract["historical_partition"]["visible_start"] = "2020-01-03"
    contract["historical_partition"]["visible_end"] = "2020-01-05"
    contract["portfolio"]["terminal_winddown_start"] = "2020-01-05 00:00:00"
    contract["universe"]["turnover_lookback_hours"] = 2
    contract["portfolio"]["minimum_trade_notional_usdt"] = 1.0
    contract["portfolio"]["quantity_increment"] = {
        symbol: 0.001 for symbol in FIXED_SYMBOLS
    }
    alphabetical = sorted(FIXED_SYMBOLS)
    basis = {symbol: (rank - 7) / 10000.0 for rank, symbol in enumerate(alphabetical)}
    market_times = pd.date_range("2020-01-02 20:00:00", "2020-01-05 23:00:00", freq="1h")
    perpetual = _market_frame(market_times)
    signal_times = pd.date_range("2020-01-02 23:00:00", "2020-01-03 23:00:00", freq="1D")
    spot = _spot_signal_frame(signal_times, basis)
    signal_perpetual = perpetual.loc[perpetual["open_time"].isin(signal_times)].copy()
    signal_perpetual[["open", "high", "low", "close"]] = 100.0
    targets, _, _ = build_daily_targets(spot, signal_perpetual, contract)
    funding_rows: list[dict[str, object]] = []
    for timestamp in pd.date_range("2020-01-03 00:00:00", "2020-01-06 00:00:00", freq="8h"):
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
    dates = pd.date_range("2019-12-29", "2020-01-06", freq="D")
    fx = pd.DataFrame(
        {"date": dates.to_numpy(dtype="datetime64[us]"), "cny_per_usd": 7.0}
    )
    benchmark = pd.DataFrame(
        {"date": dates.to_numpy(dtype="datetime64[ns]"), "close": 100.0}
    )
    daily, trades, portfolio, context = run_portfolio_backtest(
        targets,
        perpetual,
        funding,
        fx,
        benchmark,
        contract,
    )
    assert len(daily) == 3
    assert not trades.empty
    assert portfolio["intraday_risk_deleveraging_hour_count"] > 0
    assert portfolio["maximum_gross_exposure"] <= 1.0 + 1e-10
    assert portfolio["maximum_absolute_net_exposure"] <= 0.03 + 1e-10
    assert portfolio["terminal_position_count"] == 0
    assert portfolio["spot_position_count"] == 0
    assert portfolio["stress_total_transaction_cost_usdt"] > portfolio[
        "base_total_transaction_cost_usdt"
    ]
    assert portfolio["base_total_funding_usdt"] == pytest.approx(0.0, abs=1e-9)
    assert portfolio["stress_total_funding_usdt"] < 0.0
    assert context["strictly_lagged_fx"] is True
    assert np.isfinite(
        daily[["strategy_base_net_return", "strategy_stress_net_return"]].to_numpy(dtype=float)
    ).all()


def test_parameter_drift_is_rejected() -> None:
    contract = copy.deepcopy(load_contract())
    contract["signal"]["long_side_target_gross"] = 0.6
    with pytest.raises(ValueError, match="配置被弱化或损坏"):
        validate_contract(contract)
