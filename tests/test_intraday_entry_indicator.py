"""510300十五分钟状态切换进场指标测试。"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from research.intraday_entry_indicator import (
    build_intraday_entry_indicator,
    load_indicator_config,
)


def _factor_row(**changes: object) -> pd.DataFrame:
    row: dict[str, object] = {
        "signal_asof": pd.Timestamp("2026-08-12 10:15:00"),
        "trade_date": pd.Timestamp("2026-08-12"),
        "bar_slot": 3,
        "close": 4.05,
        "session_vwap": 4.00,
        "session_vwap_slope_3bar_bps": 5.0,
        "ema_8bar": 4.03,
        "ema_21bar": 4.01,
        "adx_14bar": 25.0,
        "opening_range_direct_eligible": True,
        "orb_breakout_up_bps": 10.0,
        "orb_breakout_down_bps": 10.0,
        "time_of_day_rvol_median_20d": 1.5,
        "atr_14bar": 0.02,
        "rsi_7bar": 60.0,
        "vwap_deviation_atr": 2.5,
    }
    row.update(changes)
    return pd.DataFrame([row])


def test_趋势突破在缺少广度时只形成核心候选() -> None:
    result = build_intraday_entry_indicator(_factor_row())
    assert result.iloc[0]["entry_signal"] == "TREND_BREAKOUT_ETF_ONLY"
    assert bool(result.iloc[0]["is_entry_candidate"])
    assert bool(result.iloc[0]["is_entry_event"])
    assert bool(result.iloc[0]["is_fully_confirmed"])


def test_盘中广度可以确认或否决趋势突破() -> None:
    strong = pd.DataFrame(
        {"signal_asof": [pd.Timestamp("2026-08-12 10:15:00")], "breadth_fraction": [0.68]}
    )
    weak = strong.assign(breadth_fraction=0.38)
    config = deepcopy(load_indicator_config())
    config["parameters"]["breadth_policy"] = "OPTIONAL_CONFIRMATION"
    confirmed = build_intraday_entry_indicator(
        _factor_row(), config=config, intraday_breadth=strong
    )
    rejected = build_intraday_entry_indicator(
        _factor_row(), config=config, intraday_breadth=weak
    )
    assert confirmed.iloc[0]["entry_signal"] == "TREND_BREAKOUT_CONFIRMED"
    assert bool(confirmed.iloc[0]["is_fully_confirmed"])
    assert rejected.iloc[0]["entry_signal"] == "TREND_BREAKOUT_BREADTH_REJECTED"
    assert not bool(rejected.iloc[0]["is_entry_candidate"])


def test_震荡低吸不依赖趋势条件或盘中广度() -> None:
    factors = _factor_row(
        close=3.98,
        session_vwap=4.00,
        session_vwap_slope_3bar_bps=1.0,
        ema_8bar=3.99,
        ema_21bar=4.01,
        adx_14bar=15.0,
        orb_breakout_up_bps=-40.0,
        time_of_day_rvol_median_20d=0.9,
        rsi_7bar=25.0,
        vwap_deviation_atr=-1.0,
    )
    result = build_intraday_entry_indicator(factors)
    assert result.iloc[0]["market_regime"] == "RANGE"
    assert result.iloc[0]["entry_signal"] == "MEAN_REVERSION_BUY"
    assert bool(result.iloc[0]["is_fully_confirmed"])


def test_严格广度策略不把缺失广度当作可执行候选() -> None:
    config = deepcopy(load_indicator_config())
    config["parameters"]["breadth_policy"] = "REQUIRED"
    result = build_intraday_entry_indicator(_factor_row(), config=config)
    assert result.iloc[0]["entry_signal"] == "TREND_BREAKOUT_CORE"
    assert not bool(result.iloc[0]["is_entry_candidate"])


def test_收盘K线禁止生成新进场候选() -> None:
    result = build_intraday_entry_indicator(_factor_row(bar_slot=16))
    assert result.iloc[0]["entry_signal"] == "ENTRY_WINDOW_CLOSED"
    assert not bool(result.iloc[0]["is_entry_candidate"])


def test_非法广度被拒绝() -> None:
    breadth = pd.DataFrame(
        {"signal_asof": [pd.Timestamp("2026-08-12 10:15:00")], "breadth_fraction": [1.2]}
    )
    try:
        build_intraday_entry_indicator(_factor_row(), intraday_breadth=breadth)
    except ValueError as error:
        assert "0到1" in str(error)
    else:
        raise AssertionError("非法广度必须触发ValueError")


def test_同一天只把第一根候选标记为进场事件() -> None:
    first = _factor_row()
    second = _factor_row(
        signal_asof=pd.Timestamp("2026-08-12 10:30:00"),
        bar_slot=4,
    )
    result = build_intraday_entry_indicator(pd.concat([first, second], ignore_index=True))
    assert result["is_entry_candidate"].tolist() == [True, True]
    assert result["is_entry_event"].tolist() == [True, False]


def test_趋势跌破生成减仓事件() -> None:
    factors = _factor_row(
        close=3.95,
        session_vwap=4.00,
        session_vwap_slope_3bar_bps=-5.0,
        ema_8bar=3.97,
        ema_21bar=3.99,
        orb_breakout_up_bps=-10.0,
        orb_breakout_down_bps=-10.0,
        rsi_7bar=40.0,
        vwap_deviation_atr=-2.5,
    )
    result = build_intraday_entry_indicator(factors)
    assert result.iloc[0]["exit_signal"] == "TREND_BREAKDOWN_CORE"
    assert bool(result.iloc[0]["is_exit_event"])


def test_震荡高位生成减仓事件() -> None:
    factors = _factor_row(
        close=4.02,
        session_vwap=4.00,
        session_vwap_slope_3bar_bps=1.0,
        ema_8bar=4.01,
        ema_21bar=4.00,
        adx_14bar=15.0,
        orb_breakout_down_bps=30.0,
        time_of_day_rvol_median_20d=0.9,
        rsi_7bar=75.0,
        vwap_deviation_atr=1.0,
    )
    result = build_intraday_entry_indicator(factors)
    assert result.iloc[0]["exit_signal"] == "MEAN_REVERSION_SELL"
    assert bool(result.iloc[0]["is_exit_event"])
