"""510300日内二元状态筛选V1的合成数据回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.intraday_binary_livermore_screen_v1 import (  # noqa: E402
    CostModel,
    _maximum_affordable_quantity,
    build_candidate_states,
    simulate_candidate,
)


def _costs() -> CostModel:
    return CostModel(
        scenario="TEST",
        commission_rate=0.0001,
        minimum_commission=0.0,
        slippage_bps=5.0,
        cash_annual_rate=0.015,
        trading_days_per_year=242,
        lot_size=100,
    )


def _synthetic_bars() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"])
    rows: list[dict[str, object]] = []
    price = 5.0
    for date in dates:
        for hour, minute in [(10, 0), (10, 15)]:
            rows.append(
                {
                    "trade_date": date,
                    "bar_end": date + pd.Timedelta(hours=hour, minutes=minute),
                    "open": price,
                    "high": price + 0.02,
                    "low": price - 0.02,
                    "close": price + 0.01,
                    "adx_14bar": 25.0,
                    "plus_di_14bar": 30.0,
                    "minus_di_14bar": 10.0,
                }
            )
            price += 0.01
    return pd.DataFrame(rows)


def test_maximum_affordable_quantity_is_round_lot_and_cash_safe() -> None:
    quantity, price, commission = _maximum_affordable_quantity(500000.0, 5.0, _costs())
    assert quantity > 0
    assert quantity % 100 == 0
    assert quantity * price + commission <= 500000.0 + 1e-9
    assert (quantity + 100) * price > 500000.0 - commission


def test_donchian_state_uses_prior_bars_and_hysteresis() -> None:
    bars = _synthetic_bars()
    bars.loc[2, "close"] = 5.05
    bars.loc[2, "high"] = 5.06
    candidate = {
        "id": "TEST_DONCHIAN",
        "type": "DONCHIAN_HYSTERESIS",
        "entry_lookback_bars": 2,
        "exit_lookback_bars": 2,
    }
    states = build_candidate_states(bars, candidate)
    assert states.dtype == np.int8
    assert states[0] == 0
    assert states[1] == 0
    assert states[2] == 1
    assert set(np.unique(states)).issubset({0, 1})


def test_same_day_exit_after_buy_is_blocked_by_t_plus_one() -> None:
    bars = _synthetic_bars()
    targets = np.zeros(len(bars), dtype=np.int8)
    targets[1] = 1
    dividends = pd.DataFrame(
        columns=[
            "symbol",
            "record_date",
            "ex_date",
            "payment_date",
            "cash_dividend_per_share",
            "source",
        ]
    )
    ledger, trades, diagnostics = simulate_candidate(
        bars,
        dividends,
        targets,
        start_date=pd.Timestamp("2026-01-06"),
        end_date=pd.Timestamp("2026-01-08"),
        costs=_costs(),
        initial_capital=500000.0,
    )
    assert list(trades["side"]) == ["BUY", "SELL"]
    assert pd.Timestamp(trades.iloc[0]["date"]) == pd.Timestamp("2026-01-06")
    assert pd.Timestamp(trades.iloc[1]["date"]) == pd.Timestamp("2026-01-07")
    assert diagnostics["t_plus_one_blocked_exit_attempts"] == 1
    assert diagnostics["t_plus_one_violations"] == 0
    assert set(ledger["actual_state"].unique()).issubset({0, 1})
