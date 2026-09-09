from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.tech_03_donchian_55_20_v1 import (
    CostModel,
    audit_and_load_inputs,
    build_signals,
    load_config,
    simulate_portfolio,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config()


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_date": pd.Series(dtype="datetime64[ns]"),
            "ex_date": pd.Series(dtype="datetime64[ns]"),
            "payment_date": pd.Series(dtype="datetime64[ns]"),
            "cash_dividend_per_share": pd.Series(dtype=float),
        }
    )


def _market(values: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=len(values))
    close = np.asarray(values, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "etf_open": close,
            "etf_high": close + 0.2,
            "etf_low": close - 0.2,
            "etf_close": close,
            "price_index_open": close * 100,
            "price_index_high": (close + 0.2) * 100,
            "price_index_low": (close - 0.2) * 100,
            "price_index_close": close * 100,
            "h00300_close": close * 110,
            "tr_scale_factor": 1.1,
            "signal_open": close * 110,
            "signal_high": (close + 0.2) * 110,
            "signal_low": (close - 0.2) * 110,
            "signal_close": close * 110,
        }
    )


def _costs(fractional: bool) -> CostModel:
    return CostModel(
        commission_rate=0.0003,
        minimum_commission=0.0 if fractional else 5.0,
        slippage_bps=5.0,
        cash_annual_rate=0.015,
        trading_days_per_year=242,
        lot_size=None if fractional else 100,
        minimum_trade_shares=0 if fractional else 1000,
    )


def test_frozen_registry_contains_only_55_20_and_100_50_return_candidates() -> None:
    variants = CONFIG["variants"]
    assert [(item["entry_lookback_trading_days"], item["exit_lookback_trading_days"]) for item in variants] == [(55, 20), (100, 50)]
    assert CONFIG["trial_registry"]["registered_trial_count"] == 3
    assert CONFIG["trial_registry"]["trials"][0]["return_calculated"] is False
    assert CONFIG["governance"]["parameter_search_allowed"] is False


def test_strict_breakout_excludes_current_bar_and_equality_does_not_trigger() -> None:
    market = _market([10.0, 10.5, 11.0, 11.2, 11.4, 11.3, 11.2, 10.5])
    market.loc[3, "signal_close"] = market.loc[0:2, "signal_high"].max()
    market.loc[4, "signal_close"] = market.loc[1:3, "signal_high"].max() + 0.01
    market.loc[7, "signal_close"] = market.loc[5:6, "signal_low"].min() - 0.01
    variant = {
        "id": "TEST_DONCHIAN_3_2",
        "entry_lookback_trading_days": 3,
        "exit_lookback_trading_days": 2,
    }
    signals = build_signals(market, variant)
    assert signals.loc[0, "signal_action"] == "HOLD"
    assert signals.loc[1, "signal_action"] == "ENTER"
    assert signals.iloc[-1]["signal_action"] == "EXIT"
    assert signals.loc[1, "channel_end_date"] < signals.loc[1, "date"]


def test_signal_executes_only_on_next_trading_day_and_t_plus_one_inventory_is_old() -> None:
    signals = _market([4.0, 4.1, 4.2, 4.0])
    signals["variant_id"] = "TEST"
    signals["target_position"] = [1.0, 1.0, 0.0, 0.0]
    signals["signal_action"] = ["ENTER", "HOLD", "EXIT", "HOLD"]
    ledger, trades = simulate_portfolio(
        signals, _empty_dividends(), _costs(fractional=False), 20000.0, fractional=False
    )
    assert trades["side"].tolist() == ["BUY", "SELL"]
    assert trades.iloc[0]["execution_date"] == signals.iloc[1]["date"]
    assert trades.iloc[1]["execution_date"] == signals.iloc[3]["date"]
    assert trades.iloc[1]["t_plus_one_sellable_before_trade"] == trades.iloc[0]["quantity"]
    assert (trades["quantity"] % 100 == 0).all()
    assert trades["commission"].ge(5.0).all()
    assert ledger.iloc[-1]["shares"] == 0


def test_dividend_entitlement_uses_start_of_ex_date_inventory_and_payment_date_cash() -> None:
    signals = _market([4.0, 4.0, 3.9, 4.0, 4.0])
    signals["variant_id"] = "TEST"
    signals["target_position"] = [1.0, 1.0, 1.0, 1.0, 1.0]
    signals["signal_action"] = ["ENTER", "HOLD", "HOLD", "HOLD", "HOLD"]
    dividends = pd.DataFrame(
        {
            "record_date": [signals.loc[1, "date"]],
            "ex_date": [signals.loc[2, "date"]],
            "payment_date": [signals.loc[4, "date"]],
            "cash_dividend_per_share": [0.1],
        }
    )
    ledger, trades = simulate_portfolio(
        signals, dividends, _costs(fractional=False), 20000.0, fractional=False
    )
    quantity = float(trades.iloc[0]["quantity"])
    assert ledger.loc[2, "dividend_entitlement_today"] == pytest.approx(quantity * 0.1)
    assert ledger.loc[4, "dividend_payment_today"] == pytest.approx(quantity * 0.1)
    assert ledger.loc[4, "dividend_receivable"] == pytest.approx(0.0)


def test_real_frozen_data_gate_and_synthetic_close_identity_pass() -> None:
    market, dividends, audit = audit_and_load_inputs(ROOT, CONFIG)
    assert audit["status"] == "PASS"
    assert audit["branches"]["synthetic_total_return_ohlc"]["official_h00300_ohlc_claimed"] is False
    assert audit["branches"]["synthetic_total_return_ohlc"]["close_identity_within_1e_10"] is True
    assert len(market) == 3456
    assert len(dividends) == 14


def test_live_safety_switches_are_all_disabled() -> None:
    assert CONFIG["protocol"]["true_forward_start"] is None
    assert CONFIG["protocol"]["live_trading_authorized"] is False
    assert CONFIG["governance"]["position_mapping_enabled"] is False
    assert CONFIG["governance"]["order_generation_enabled"] is False
    assert CONFIG["governance"]["broker_connection_enabled"] is False
