from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.daily_02_tom_liquidity_drift_v1 import (
    active_interval_map,
    build_event_episodes,
    build_event_trade_table,
    build_strategy_targets,
    circular_block_bootstrap_mean,
    profit_factor,
)


def _market(start: str = "2026-01-02", end: str = "2026-04-10", price: float = 10.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    return pd.DataFrame(
        {
            "date": dates,
            "open": np.full(len(dates), price),
            "high": np.full(len(dates), price + 0.1),
            "low": np.full(len(dates), price - 0.1),
            "close": np.full(len(dates), price),
        }
    )


def _execution() -> dict:
    return {
        "active_sleeve_test_capital_cny": 9700.0,
        "lot_size_shares": 100,
        "commission_rate": 0.0003,
        "minimum_commission_cny": 5.0,
        "cash_annual_rate": 0.015,
        "trading_days_per_year": 242,
    }


def test_event_is_last_day_plus_next_month_first_three_intervals() -> None:
    market = _market()
    episodes = build_event_episodes(market)
    january = episodes.loc[episodes["month"].eq("2026-01")].iloc[0]

    assert january["entry_date"] == pd.Timestamp("2026-01-30")
    assert january["first_date"] == pd.Timestamp("2026-02-02")
    assert january["third_date"] == pd.Timestamp("2026-02-04")
    assert january["exit_date"] == pd.Timestamp("2026-02-05")
    assert len(active_interval_map(episodes)) == len(episodes) * 4


def test_target_is_written_previous_day_and_exit_executes_fourth_day_open() -> None:
    market = _market()
    episodes = build_event_episodes(market)
    targets = build_strategy_targets(market, episodes, 0.50, 0.985).set_index("date")

    assert targets.loc[pd.Timestamp("2026-01-29"), "target_position"] == pytest.approx(0.985)
    assert targets.loc[pd.Timestamp("2026-01-29"), "signal_reason"] == "TOM_ENTER"
    assert targets.loc[pd.Timestamp("2026-02-04"), "target_position"] == pytest.approx(0.50)
    assert targets.loc[pd.Timestamp("2026-02-04"), "signal_reason"] == "TOM_EXIT"


def test_event_trade_table_deducts_two_minimum_commissions() -> None:
    market = _market()
    episodes = build_event_episodes(market).iloc[[0]].copy()
    dividends = pd.DataFrame(columns=["record_date", "cash_dividend_per_share"])
    result = build_event_trade_table(
        market, dividends, episodes, _execution(), slippage_bps=0.0
    ).iloc[0]

    assert result["quantity"] == 900
    assert result["buy_commission"] == pytest.approx(5.0)
    assert result["sell_commission"] == pytest.approx(5.0)
    assert result["net_excess_pnl"] < -10.0


def test_dividend_requires_record_date_inside_holding_interval() -> None:
    market = _market()
    episode = build_event_episodes(market).iloc[[0]].copy()
    record_date = episode.iloc[0]["first_date"]
    dividends = pd.DataFrame(
        {"record_date": [record_date], "cash_dividend_per_share": [0.10]}
    )
    with_dividend = build_event_trade_table(
        market, dividends, episode, _execution(), slippage_bps=0.0
    ).iloc[0]
    without_dividend = build_event_trade_table(
        market,
        dividends.iloc[0:0].copy(),
        episode,
        _execution(),
        slippage_bps=0.0,
    ).iloc[0]

    assert with_dividend["dividend_cash"] == pytest.approx(90.0)
    assert with_dividend["end_wealth"] - without_dividend["end_wealth"] == pytest.approx(90.0)


def test_profit_factor_and_bootstrap_are_directionally_correct() -> None:
    assert profit_factor(pd.Series([3.0, -1.0, 2.0, -1.0])) == pytest.approx(2.5)
    result = circular_block_bootstrap_mean(
        np.full(120, 0.01), block_length=6, repetitions=200, random_seed=20260819
    )
    assert result["lower_95"] == pytest.approx(0.01)
