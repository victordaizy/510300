from __future__ import annotations

import pandas as pd

from research.technical_timing_candidates import (
    events_from_close_pressure_forecasts,
    events_from_trade_records,
)


def test_既有交易方向可以转换为买卖事件() -> None:
    trades = pd.DataFrame(
        {
            "direction": ["BUY_LOW", "SELL_HIGH"],
            "entry_execution_time": ["2026-01-05 10:00", "2026-01-06 11:00"],
            "entry_raw_open_cny": [4.0, 4.1],
        }
    )
    buy, sell = events_from_trade_records(trades, "TEST")
    assert buy[pd.Timestamp("2026-01-05")]["raw_execution_price"] == 4.0
    assert sell[pd.Timestamp("2026-01-06")]["raw_execution_price"] == 4.1


def test_收盘压力只使用当时概率相对基准并在次日开盘执行() -> None:
    daily = pd.DataFrame(
        {"date": pd.bdate_range("2026-01-05", periods=3), "open": [4.0, 4.1, 4.2]}
    )
    forecasts = pd.DataFrame(
        {
            "signal_date": daily["date"].iloc[:2],
            "probability_positive_2d_net": [0.55, 0.40],
            "expanding_base_rate_probability": [0.50, 0.45],
        }
    )
    buy, sell = events_from_close_pressure_forecasts(forecasts, daily)
    assert buy[daily["date"].iloc[1]]["raw_execution_price"] == 4.1
    assert sell[daily["date"].iloc[2]]["raw_execution_price"] == 4.2
