from __future__ import annotations

import pandas as pd

from backtest.intraday_entry_overlay_engine import (
    OverlayCosts,
    prepare_executable_targets,
    prepare_intraday_events,
    run_valuation_entry_policy,
)


def _costs() -> OverlayCosts:
    return OverlayCosts(
        commission_rate=0.0,
        minimum_commission_cny=0.0,
        slippage_bps_per_leg=0.0,
        lot_size=100,
        minimum_trade_shares=100,
        cash_annual_rate=0.0,
    )


def _daily() -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-05", periods=8)
    return pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 10.0, 9.8, 9.7, 9.6, 9.5, 9.4, 9.3],
            "close": [10.0, 9.9, 9.7, 9.6, 9.5, 9.4, 9.3, 9.2],
        }
    )


def _dividends() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["ex_date", "payment_date", "cash_dividend_per_share"]
    )


def test_估值信号严格推迟到下一交易日并量化为五档() -> None:
    dates = pd.bdate_range("2026-01-05", periods=3)
    valuation = pd.DataFrame(
        {"date": dates, "target_position": [0.30, 0.70, 0.40]}
    )
    result = prepare_executable_targets(valuation, pd.Series(dates), 0.25)
    assert result["executable_target_position"].tolist() == [0.25, 0.25, 0.75]


def test_技术叠加只延迟加仓并在信号后下一根开盘执行() -> None:
    daily = _daily()
    targets = pd.DataFrame(
        {
            "date": daily["date"],
            "executable_target_position": [0.25, 0.50, 0.50, 0.25, 0.25, 0.25, 0.25, 0.25],
        }
    )
    event_date = daily.loc[2, "date"]
    events = {
        event_date: {
            "signal_asof": event_date + pd.Timedelta(hours=10, minutes=30),
            "technical_signal": "TREND_BREAKOUT_CORE",
            "raw_execution_price": 9.75,
        }
    }
    sell_date = daily.loc[3, "date"]
    sell_events = {
        sell_date: {
            "signal_asof": sell_date + pd.Timedelta(hours=10, minutes=45),
            "technical_signal": "MEAN_REVERSION_SELL",
            "raw_execution_price": 9.65,
        }
    }
    _, trades, diagnostics = run_valuation_entry_policy(
        daily,
        targets,
        _dividends(),
        events,
        sell_events,
        _costs(),
        20_000.0,
        "VALUATION_TECHNICAL_OVERLAY",
        5,
    )
    technical_buy = trades.loc[trades["reason"].str.contains("技术信号确认")].iloc[0]
    assert technical_buy["date"] == event_date
    assert technical_buy["raw_price"] == 9.75
    sell = trades.loc[trades["side"].eq("卖出")].iloc[0]
    assert sell["date"] == sell_date
    assert diagnostics["technical_confirmed_buys"] == 1


def test_超过等待期限后在下一交易日开盘回退加仓() -> None:
    daily = _daily()
    targets = pd.DataFrame(
        {
            "date": daily["date"],
            "executable_target_position": [0.25] + [0.50] * 7,
        }
    )
    _, trades, diagnostics = run_valuation_entry_policy(
        daily,
        targets,
        _dividends(),
        {},
        {},
        _costs(),
        20_000.0,
        "VALUATION_TECHNICAL_OVERLAY",
        2,
    )
    fallback = trades.loc[trades["reason"].str.contains("等待超时")].iloc[0]
    assert fallback["date"] == daily.loc[3, "date"]
    assert diagnostics["fallback_buys"] == 1


def test_十五分钟事件使用下一根同日K线开盘() -> None:
    date = pd.Timestamp("2026-01-05")
    indicator = pd.DataFrame(
        {
            "signal_asof": [date + pd.Timedelta(hours=10), date + pd.Timedelta(hours=10, minutes=15)],
            "trade_date": [date, date],
            "bar_slot": [3, 4],
            "open": [10.0, 10.2],
            "entry_signal": ["TREND_BREAKOUT_CORE", "NO_ENTRY"],
            "is_entry_event": [True, False],
        }
    )
    events = prepare_intraday_events(
        indicator, "is_entry_event", "entry_signal", {"TREND_BREAKOUT_CORE"}
    )
    assert events[date]["raw_execution_price"] == 10.2


def test_普通减仓等待退出信号但危机减仓立即执行() -> None:
    daily = _daily()
    targets = pd.DataFrame(
        {
            "date": daily["date"],
            "executable_target_position": [0.75, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
            "executable_risk_off_override": [False, False, False, False, True, False, False, False],
        }
    )
    exit_date = daily.loc[2, "date"]
    sell_events = {
        exit_date: {
            "signal_asof": exit_date + pd.Timedelta(hours=11),
            "technical_signal": "TREND_BREAKDOWN_CORE",
            "raw_execution_price": 9.75,
        }
    }
    _, trades, diagnostics = run_valuation_entry_policy(
        daily,
        targets,
        _dividends(),
        {},
        sell_events,
        _costs(),
        20_000.0,
        "VALUATION_TECHNICAL_OVERLAY",
        5,
    )
    technical_sell = trades.loc[trades["reason"].str.contains("技术信号确认")].iloc[0]
    assert technical_sell["date"] == exit_date
    risk_sell = trades.loc[trades["reason"].str.contains("危机强制减仓")].iloc[0]
    assert risk_sell["date"] == daily.loc[4, "date"]
    assert diagnostics["technical_confirmed_sells"] == 1
    assert diagnostics["risk_off_immediate_sells"] == 1
