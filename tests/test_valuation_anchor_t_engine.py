"""估值库存固定锚日内做T引擎测试。"""

from __future__ import annotations

import pandas as pd
import numpy as np
import yaml

from backtest.valuation_anchor_t_engine import (
    build_fixed_anchor_features,
    run_anchor_t_overlay,
    summarize_anchor_t,
)


def _config() -> dict:
    with open("config/round5_valuation_anchor_t_overlay.yaml", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _minute_day(date: str, closes: list[float]) -> pd.DataFrame:
    times = pd.date_range(f"{date} 09:45", periods=len(closes), freq="1min")
    return pd.DataFrame(
        {
            "trade_time": times,
            "open": closes,
            "high": [value + 0.002 for value in closes],
            "low": [value - 0.002 for value in closes],
            "close": closes,
            "vol": 1_000_000,
            "amount": [round(value * 1_000_000) for value in closes],
        }
    )


def _base_ledger(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "shares": 1200,
            "cash": 14_000.0,
            "equity": 20_000.0,
            "actual_position": 0.30,
        }
    )


def test_固定锚严格使用前一交易日收盘() -> None:
    first = _minute_day("2025-01-06", [5.0] * 8)
    second = _minute_day("2025-01-07", [4.85, 4.86, 4.87, 4.88, 4.90, 4.95, 5.01, 5.00])
    minute = pd.concat([first, second], ignore_index=True)
    daily = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-06", "2025-01-07"]), "close": [5.0, 5.0]}
    )
    result = build_fixed_anchor_features(minute, daily)
    second_day = result.loc[result["trade_date"].eq(pd.Timestamp("2025-01-07"))]
    assert second_day["anchor_close"].eq(5.0).all()
    assert np.isclose(second_day.iloc[0]["anchor_deviation"], -0.03)


def test_买低做T下一条记录成交且不改变核心份额() -> None:
    first = _minute_day("2025-01-06", [5.0] * 8)
    second = _minute_day("2025-01-07", [4.85, 4.86, 4.87, 4.88, 4.90, 4.95, 5.01, 5.00])
    minute = pd.concat([first, second], ignore_index=True)
    dates = pd.to_datetime(["2025-01-06", "2025-01-07"])
    daily = pd.DataFrame({"date": dates, "close": [5.0, 5.0]})
    features = build_fixed_anchor_features(minute, daily)
    result = run_anchor_t_overlay(
        features, _base_ledger(dates), set(), _config(), "BASE_COST", 5.0, "单元测试"
    )
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["direction"] == "BUY_LOW"
    assert trade["sequence"] == "BUY_NEW_THEN_SELL_OLD"
    assert trade["entry_execution_time"] > trade["entry_signal_time"]
    assert trade["exit_execution_time"] > trade["exit_signal_time"]
    assert result.ledger["base_shares"].eq(1200).all()
    assert summarize_anchor_t(result, 20_000.0)["intraday_t"]["round_trip_count"] == 1


def test_核心调仓日必须跳过做T() -> None:
    first = _minute_day("2025-01-06", [5.0] * 8)
    second = _minute_day("2025-01-07", [4.85, 4.86, 4.87, 4.88, 4.90, 4.95, 5.01, 5.00])
    minute = pd.concat([first, second], ignore_index=True)
    dates = pd.to_datetime(["2025-01-06", "2025-01-07"])
    daily = pd.DataFrame({"date": dates, "close": [5.0, 5.0]})
    features = build_fixed_anchor_features(minute, daily)
    result = run_anchor_t_overlay(
        features,
        _base_ledger(dates),
        {pd.Timestamp("2025-01-07")},
        _config(),
        "BASE_COST",
        5.0,
        "单元测试",
    )
    assert result.trades.empty
    assert result.ledger.loc[result.ledger["date"].eq(pd.Timestamp("2025-01-07")), "skipped_for_core_trade"].iloc[0]
