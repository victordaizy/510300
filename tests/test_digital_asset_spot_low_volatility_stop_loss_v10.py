from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from research.digital_asset_cross_sectional_hourly_reversal_v8 import FIXED_SYMBOLS
from research.digital_asset_spot_low_volatility_stop_loss_v10 import (
    build_semiannual_selections,
    load_contract,
    run_portfolio_backtest,
    validate_contract,
)


def _selection_spot_frame() -> pd.DataFrame:
    signal_closes = pd.date_range("2020-02-26 23:00:00", periods=5, freq="1D")
    alphabetical = sorted(FIXED_SYMBOLS)
    rows: list[dict[str, object]] = []
    for rank, symbol in enumerate(alphabetical, start=1):
        scale = rank * 0.001
        log_offsets = np.asarray([0.0, scale, 0.0, scale, 100.0 * scale])
        prices = 100.0 * np.exp(log_offsets)
        for timestamp, price in zip(signal_closes, prices, strict=True):
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": timestamp,
                    "open": float(price),
                    "high": float(price),
                    "low": float(price),
                    "close": float(price),
                    "quote_volume": 15_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def _hourly_spot_frame(times: pd.DatetimeIndex) -> pd.DataFrame:
    stopped_symbol = sorted(FIXED_SYMBOLS)[0]
    rows: list[dict[str, object]] = []
    for timestamp in times:
        for symbol in FIXED_SYMBOLS:
            price = 100.0
            low = 99.0
            if symbol == stopped_symbol and timestamp >= pd.Timestamp("2020-03-03 23:00:00"):
                price = 94.0
                low = 93.0
            rows.append(
                {
                    "symbol": symbol,
                    "open_time": timestamp,
                    "open": price,
                    "high": max(price, 100.0),
                    "low": low,
                    "close": price,
                    "quote_volume": 15_000_000.0,
                }
            )
    return pd.DataFrame(rows)


def test_contract_keeps_objective_long_only_cost_and_research_boundary() -> None:
    contract = load_contract()
    assert contract["account"]["initial_capital_cny"] == 500000.0
    assert contract["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["signal"]["selected_count"] == 3
    assert contract["signal"]["target_long_gross"] == 1.0
    assert contract["signal"]["short_position_allowed"] is False
    assert contract["signal"]["derivative_position_allowed"] is False
    assert not any(contract["safety"].values())


def test_volatility_selection_is_strictly_lagged_and_deterministic() -> None:
    contract = copy.deepcopy(load_contract())
    contract["historical_partition"]["visible_start"] = "2020-03-01"
    contract["historical_partition"]["visible_end"] = "2020-03-03"
    contract["portfolio"]["terminal_winddown_start"] = "2020-03-04 00:00:00"
    contract["signal"]["volatility_lookback_calendar_days"] = 3
    targets, features, audit = build_semiannual_selections(
        _selection_spot_frame(), contract
    )
    alphabetical = sorted(FIXED_SYMBOLS)
    assert sorted(targets["symbol"].tolist()) == alphabetical[:3]
    assert targets["slot_weight"].sum() == pytest.approx(1.0)
    assert targets["execution_time"].eq(pd.Timestamp("2020-03-01 00:00:00")).all()
    assert targets["signal_time"].eq(pd.Timestamp("2020-02-29 23:00:00")).all()
    ranked = features.sort_values("volatility_rank")["symbol"].tolist()
    assert ranked == alphabetical
    assert audit["strict_execution_lag_hours"] == 1
    assert audit["future_price_or_return_used_in_selection"] is False


def test_full_engine_continues_capacity_slices_stops_and_reaches_terminal_zero() -> None:
    contract = copy.deepcopy(load_contract())
    contract["historical_partition"]["visible_start"] = "2020-03-01"
    contract["historical_partition"]["visible_end"] = "2020-03-08"
    contract["portfolio"]["terminal_winddown_start"] = "2020-03-07 00:00:00"
    contract["universe"]["turnover_lookback_observed_venue_hours"] = 2
    contract["portfolio"]["minimum_trade_notional_usdt"] = 1.0
    contract["portfolio"]["quantity_increment"] = {
        symbol: 0.001 for symbol in FIXED_SYMBOLS
    }
    selected = sorted(FIXED_SYMBOLS)[:3]
    selections = pd.DataFrame(
        {
            "signal_time": pd.Timestamp("2020-02-29 23:00:00"),
            "execution_time": pd.Timestamp("2020-03-01 00:00:00"),
            "symbol": selected,
            "slot_weight": 1.0 / 3.0,
            "realized_volatility": [0.01, 0.02, 0.03],
        }
    )
    market_times = pd.date_range("2020-02-29 21:00:00", "2020-03-08 23:00:00", freq="1h")
    spot = _hourly_spot_frame(market_times)
    context_dates = pd.date_range("2020-02-25", "2020-03-09", freq="D")
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
        selections, spot, fx, benchmark, contract
    )
    stopped_symbol = selected[0]
    stop_trade = trades.loc[
        trades["reason"].eq("STOP_LOSS") & trades["symbol"].eq(stopped_symbol)
    ]
    assert len(daily) == 8
    assert not stop_trade.empty
    assert stop_trade.iloc[0]["trade_time"] == pd.Timestamp("2020-03-04 00:00:00")
    assert stop_trade.iloc[0]["side"] == "SELL"
    assert portfolio["global_capacity_sliced_day_count"] > 0
    assert portfolio["capacity_continuation_trade_day_count"] > 0
    assert portfolio["stop_loss_event_count"] == 1
    assert portfolio["maximum_order_capacity_fraction"] <= 0.001 + 1e-10
    assert portfolio["maximum_gross_exposure"] <= 1.0 + 1e-10
    assert portfolio["terminal_position_count"] == 0
    assert portfolio["short_position_count"] == 0
    assert portfolio["derivative_position_count"] == 0
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
    contract["stop_loss"]["threshold"] = 0.06
    with pytest.raises(ValueError, match="配置被弱化或损坏"):
        validate_contract(contract)
