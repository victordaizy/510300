from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from research.digital_asset_cross_sectional_hourly_reversal_v8 import (
    FIXED_SYMBOLS,
    build_hourly_targets,
    load_contract,
    run_portfolio_backtest,
    validate_contract,
)


def _hourly_frame(
    times: pd.DatetimeIndex,
    returns_by_symbol: dict[str, float],
    *,
    quote_volume: float = 1_000_000.0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp in times:
        for symbol in FIXED_SYMBOLS:
            prior_return = returns_by_symbol[symbol]
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": timestamp,
                    "open": 100.0,
                    "high": max(100.1, 100.0 * (1.0 + prior_return)),
                    "low": min(99.9, 100.0 * (1.0 + prior_return)),
                    "close": 100.0 * (1.0 + prior_return),
                    "quote_volume": quote_volume,
                }
            )
    return pd.DataFrame(rows)


def test_contract_keeps_500k_40pct_high_sharpe_and_research_only() -> None:
    contract = load_contract()
    assert contract["account"]["initial_capital_cny"] == 500000.0
    assert contract["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["universe"]["fixed_symbols"] == FIXED_SYMBOLS
    assert contract["signal"]["maximum_target_gross_exposure"] == 2.0
    assert not any(contract["safety"].values())


def test_ranking_is_deterministic_disjoint_and_uses_strict_prior_hour() -> None:
    contract = copy.deepcopy(load_contract())
    contract["historical_partition"]["visible_start"] = "2020-02-18"
    contract["historical_partition"]["visible_end"] = "2020-02-18"
    contract["portfolio"]["terminal_winddown_start"] = "2020-02-18 03:00:00"
    alphabetical = sorted(FIXED_SYMBOLS)
    returns = {symbol: 0.0 for symbol in FIXED_SYMBOLS}
    for symbol in alphabetical[:4]:
        returns[symbol] = -0.02
    for symbol in ("XRPUSDT", "XTZUSDT", "ZECUSDT"):
        returns[symbol] = 0.02
    signal_times = pd.date_range("2020-02-17 23:00:00", "2020-02-18 01:00:00", freq="1h")
    frame = _hourly_frame(signal_times, returns, quote_volume=1_000_000_000.0)
    current_mask = frame["open_time"].eq(pd.Timestamp("2020-02-18 00:00:00"))
    frame.loc[current_mask & frame["symbol"].eq("BTCUSDT"), "close"] = 50.0
    all_equal_mask = frame["open_time"].eq(pd.Timestamp("2020-02-18 01:00:00"))
    frame.loc[all_equal_mask, "close"] = 100.0
    targets, features, audit = build_hourly_targets(frame, contract)
    first = targets.loc[targets["execution_time"].eq(pd.Timestamp("2020-02-18 00:00:00"))]
    longs = sorted(first.loc[first["target_weight"].gt(0.0), "symbol"])
    shorts = sorted(first.loc[first["target_weight"].lt(0.0), "symbol"])
    assert longs == alphabetical[:3]
    assert shorts == ["XRPUSDT", "XTZUSDT", "ZECUSDT"]
    assert set(longs).isdisjoint(shorts)
    assert first["target_weight"].clip(lower=0.0).sum() == pytest.approx(1.0)
    assert first["target_weight"].clip(upper=0.0).sum() == pytest.approx(-1.0)
    assert (first["execution_time"] - first["signal_time"]).eq(pd.Timedelta(hours=1)).all()
    assert audit["strict_execution_lag_hours"] == 1
    assert audit["current_execution_bar_return_used"] is False
    assert len(features) == 3 * len(FIXED_SYMBOLS)
    all_equal = targets.loc[
        targets["execution_time"].eq(pd.Timestamp("2020-02-18 02:00:00"))
    ]
    equal_longs = sorted(all_equal.loc[all_equal["target_weight"].gt(0.0), "symbol"])
    equal_shorts = sorted(all_equal.loc[all_equal["target_weight"].lt(0.0), "symbol"])
    assert equal_longs == alphabetical[:3]
    assert equal_shorts == alphabetical[3:6]


def test_full_engine_handles_mixed_time_precision_capacity_funding_and_terminal_zero() -> None:
    contract = copy.deepcopy(load_contract())
    contract["historical_partition"]["visible_start"] = "2020-01-03"
    contract["historical_partition"]["visible_end"] = "2020-01-04"
    contract["portfolio"]["terminal_winddown_start"] = "2020-01-04 00:00:00"
    contract["universe"]["turnover_lookback_hours"] = 2
    contract["portfolio"]["minimum_trade_notional_usdt"] = 1.0
    contract["portfolio"]["quantity_increment"] = {
        symbol: 0.001 for symbol in FIXED_SYMBOLS
    }
    alphabetical = sorted(FIXED_SYMBOLS)
    returns = {symbol: 0.0 for symbol in FIXED_SYMBOLS}
    for rank, symbol in enumerate(alphabetical):
        returns[symbol] = (rank - 7) / 10000.0
    market_times = pd.date_range("2020-01-02 20:00:00", "2020-01-04 23:00:00", freq="1h")
    perpetual = _hourly_frame(market_times, returns, quote_volume=1_000_000.0)
    targets, _, _ = build_hourly_targets(perpetual, contract)
    funding_rows: list[dict[str, object]] = []
    for timestamp in pd.date_range("2020-01-03 00:00:00", "2020-01-05 00:00:00", freq="8h"):
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
    dates = pd.date_range("2019-12-29", "2020-01-05", freq="D")
    fx = pd.DataFrame(
        {
            "date": dates.to_numpy(dtype="datetime64[us]"),
            "cny_per_usd": 7.0,
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": dates.to_numpy(dtype="datetime64[ns]"),
            "close": 100.0,
        }
    )
    daily, trades, portfolio, context = run_portfolio_backtest(
        targets,
        perpetual,
        funding,
        fx,
        benchmark,
        contract,
    )
    assert len(daily) == 2
    assert not trades.empty
    assert portfolio["global_capacity_sliced_hour_count"] > 0
    assert portfolio["maximum_order_capacity_fraction"] <= 0.001 + 1e-10
    assert portfolio["terminal_position_count"] == 0
    assert portfolio["stress_total_transaction_cost_usdt"] > portfolio[
        "base_total_transaction_cost_usdt"
    ]
    assert portfolio["base_total_funding_usdt"] == pytest.approx(0.0, abs=1e-9)
    assert portfolio["stress_total_funding_usdt"] < 0.0
    assert context["strictly_lagged_fx"] is True
    assert str(daily["trade_date"].dtype) == "datetime64[ns]"
    assert np.isfinite(
        daily[["strategy_base_net_return", "strategy_stress_net_return"]].to_numpy(dtype=float)
    ).all()


def test_parameter_drift_is_rejected() -> None:
    contract = copy.deepcopy(load_contract())
    contract["costs"]["stress_total_perpetual_cost_bps_per_leg"] = 15.0
    with pytest.raises(ValueError, match="配置被弱化或损坏"):
        validate_contract(contract)
