from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.binary_state_feasibility_v1 import (  # noqa: E402
    CostModel,
    build_perfect_block_states,
    load_config,
    rolling_annualized_excess,
    simulate_binary_path,
    validate_config,
)


def _market(prices: list[float]) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=len(prices))
    values = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "etf_open": values,
            "etf_high": values,
            "etf_low": values,
            "etf_close": values,
            "benchmark_close": values,
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


def _costs(*, slippage_bps: float = 0.0) -> CostModel:
    return CostModel(
        commission_rate=0.0,
        minimum_commission=0.0,
        slippage_bps=slippage_bps,
        cash_annual_rate=0.0,
        trading_days_per_year=242,
        lot_size=100,
    )


def test_frozen_config_has_exact_binary_scope_and_twenty_percent_target() -> None:
    config = load_config()
    validate_config(config)
    assert config["scope"]["allowed_target_states"] == [0, 1]
    assert config["scope"]["allowed_holdings"] == ["510300.SH", "CASH_CNY"]
    assert config["objective"]["minimum_annualized_excess"] == 0.20
    assert config["objective"]["minimum_rolling_excess_median"] == 0.20


def test_perfect_block_state_uses_only_complete_future_block() -> None:
    market = _market([10.0, 10.0, 11.0, 11.0, 9.0, 8.0])
    states, blocks, last_end = build_perfect_block_states(
        market,
        _empty_dividends(),
        horizon=2,
        cash_annual_rate=0.0,
        trading_days_per_year=242,
    )
    assert last_end == 4
    assert states.tolist() == [0, 1, 1, 0, 0]
    assert blocks["oracle_state"].tolist() == [1, 0]
    assert blocks["start_index"].tolist() == [1, 3]
    assert blocks["end_index"].tolist() == [2, 4]


def test_dividend_entitlement_is_kept_when_selling_on_ex_date() -> None:
    market = _market([10.0, 10.0, 10.0, 10.0])
    dates = market["date"].tolist()
    dividends = pd.DataFrame(
        [
            {
                "symbol": "510300.SH",
                "record_date": dates[1],
                "ex_date": dates[2],
                "payment_date": dates[3],
                "cash_dividend_per_share": 1.0,
                "source": "合成测试",
            }
        ]
    )
    states = np.asarray([0, 1, 0, 0], dtype=np.int8)
    ledger, trades = simulate_binary_path(
        market,
        dividends,
        states,
        costs=_costs(),
        initial_capital=1000.0,
        reinvest_paid_dividends=True,
    )
    assert trades["side"].tolist() == ["BUY", "SELL"]
    assert ledger.loc[2, "dividend_entitlement_today"] == 100.0
    assert ledger.loc[2, "dividend_receivable"] == 100.0
    assert ledger.loc[3, "dividend_payment_today"] == 100.0
    assert ledger.loc[3, "equity"] == 1100.0


def test_full_state_reinvests_paid_dividend_at_next_open() -> None:
    market = _market([10.0, 10.0, 10.0, 10.0, 10.0])
    dates = market["date"].tolist()
    dividends = pd.DataFrame(
        [
            {
                "symbol": "510300.SH",
                "record_date": dates[1],
                "ex_date": dates[2],
                "payment_date": dates[3],
                "cash_dividend_per_share": 10.0,
                "source": "合成测试",
            }
        ]
    )
    states = np.asarray([0, 1, 1, 1, 1], dtype=np.int8)
    ledger, trades = simulate_binary_path(
        market,
        dividends,
        states,
        costs=_costs(),
        initial_capital=1000.0,
        reinvest_paid_dividends=True,
    )
    assert trades["trade_reason"].tolist() == [
        "STATE_ENTRY_FULL",
        "FULL_STATE_CASH_REINVESTMENT",
    ]
    assert ledger.loc[4, "shares"] == 200.0
    assert ledger.loc[4, "equity"] == 2000.0


def test_slippage_reduces_round_trip_equity() -> None:
    market = _market([10.0, 10.0, 10.0])
    states = np.asarray([0, 1, 0], dtype=np.int8)
    ledger, _ = simulate_binary_path(
        market,
        _empty_dividends(),
        states,
        costs=_costs(slippage_bps=10.0),
        initial_capital=100000.0,
        reinvest_paid_dividends=True,
    )
    assert ledger["equity"].iloc[-1] < 100000.0


def test_rolling_excess_is_zero_for_identical_paths() -> None:
    returns = pd.Series([0.0] + [0.001] * 300)
    rolling = rolling_annualized_excess(
        returns,
        returns,
        window=242,
        trading_days_per_year=242,
    ).dropna()
    assert not rolling.empty
    assert np.allclose(rolling.to_numpy(), 0.0)
