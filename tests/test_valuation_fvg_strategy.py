"""510300估值仓位、短波动、FVG与T+1约束测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.valuation_fvg_engine import (
    StrategyCosts,
    aggregate_minute_to_5m,
    audit_fvg_funnel,
    build_short_wave,
    build_tactical_positions,
    build_valuation_signals,
    detect_fvg_zones,
    run_inventory_backtest,
    run_integrated_backtest,
    schedule_continuous_positions,
)


def _valuation_config() -> dict:
    return {
        "trading_days_per_year": 2,
        "growth_median_window_days": 2,
        "multiple_window_days": 3,
        "minimum_growth_observations": 1,
        "minimum_multiple_observations": 1,
        "eps_growth_clip": [-0.30, 0.30],
        "bps_growth_clip": [-0.20, 0.20],
        "earnings_anchor_weight": 0.60,
        "book_anchor_weight": 0.40,
        "low_quantile": 0.30,
        "mid_quantile": 0.50,
        "high_quantile": 0.70,
    }


def _valuation_frame(prices: list[float]) -> pd.DataFrame:
    values = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=len(values)),
            "index_close_pe_source": values,
            "pe_ttm": values / 10.0,
            "pb": values / 50.0,
        }
    )


def test_估值特征不被未来新增数据改写() -> None:
    original = _valuation_frame([100, 102, 104, 106, 108, 110, 112, 114])
    extended = pd.concat(
        [
            original,
            pd.DataFrame(
                {
                    "date": [pd.Timestamp("2025-02-03")],
                    "index_close_pe_source": [500.0],
                    "pe_ttm": [50.0],
                    "pb": [10.0],
                }
            ),
        ],
        ignore_index=True,
    )
    first = build_valuation_signals(original, _valuation_config())
    second = build_valuation_signals(extended, _valuation_config()).iloc[: len(original)]
    columns = [
        "eps_growth_12m",
        "bps_growth_12m",
        "fair_low_12m",
        "fair_mid_12m",
        "fair_high_12m",
        "target_position",
    ]
    pd.testing.assert_frame_equal(first[columns], second[columns])


def test_低于估值带为满仓_高于估值带为空仓() -> None:
    cheap = build_valuation_signals(
        _valuation_frame([100, 110, 120, 50]), _valuation_config()
    )
    expensive = build_valuation_signals(
        _valuation_frame([100, 110, 120, 200]), _valuation_config()
    )
    assert cheap.iloc[-1]["target_position"] == 1.0
    assert expensive.iloc[-1]["target_position"] == 0.0


def test_短波动只使用已完成日线并按阈值给方向() -> None:
    daily = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=30),
            "close": [100.0 + index * 0.1 for index in range(25)]
            + [101.0, 103.0, 106.0, 110.0, 115.0],
        }
    )
    result = build_short_wave(
        daily, {"wave_days": 5, "volatility_days": 20, "z_threshold": 0.75}
    )
    assert result.iloc[-1]["t_direction"] == "SELL_FIRST"


def test_连续基础仓位按固定节奏更新且允许任意中间值() -> None:
    features = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=7),
            "raw_continuous_position": [0.13, 0.22, 0.31, 0.47, 0.58, 0.69, 0.82],
        }
    )
    result = schedule_continuous_positions(features, 3)
    assert result["target_position"].tolist() == [0.13, 0.13, 0.13, 0.47, 0.47, 0.47, 0.82]
    assert result["is_valuation_rebalance_day"].tolist() == [
        True, False, False, True, False, False, True
    ]


def test_短周期只偏移库存目标而不要求回到基础仓位() -> None:
    signals = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=3),
            "target_position": [0.4, 0.4, 0.4],
            "wave_z": [-2.0, 0.0, 2.0],
        }
    )
    result = build_tactical_positions(
        signals,
        {"deadband_z": 0.75, "full_strength_z": 2.0, "maximum_tactical_position": 0.1},
    )
    assert np.allclose(result["desired_position"], [0.5, 0.4, 0.3])
    assert result["inventory_intent"].tolist() == ["BUY", "HOLD", "SELL"]


def test_5分钟聚合绝不跨越午休() -> None:
    morning = pd.date_range("2025-01-06 11:21", periods=10, freq="1min")
    afternoon = pd.date_range("2025-01-06 13:01", periods=10, freq="1min")
    times = morning.append(afternoon)
    minute = pd.DataFrame(
        {
            "trade_time": times,
            "open": 10.0,
            "high": 10.1,
            "low": 9.9,
            "close": 10.0,
            "vol": 100,
            "amount": 1000,
        }
    )
    bars = aggregate_minute_to_5m(minute)
    assert len(bars) == 4
    assert set(bars["session"]) == {"AM", "PM"}
    assert (bars["source_records"] == 5).all()


def test_FVG只在同一时段识别标准三K缺口() -> None:
    bars = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-01-06")] * 3,
            "session": ["AM"] * 3,
            "session_bar": [0, 1, 2],
            "trade_time": pd.date_range("2025-01-06 09:35", periods=3, freq="5min"),
            "open": [10.0, 10.1, 10.4],
            "high": [10.2, 10.3, 10.6],
            "low": [9.9, 10.0, 10.4],
            "close": [10.1, 10.2, 10.5],
        }
    )
    zones = detect_fvg_zones(bars, minimum_gap_bps=0.0)
    assert len(zones) == 1
    assert zones.iloc[0]["fvg_kind"] == "BULLISH"
    assert zones.iloc[0]["zone_lower"] == 10.2
    assert zones.iloc[0]["zone_upper"] == 10.4


def test_FVG漏斗必须区分方向匹配_中点触发和经济成交() -> None:
    date = pd.Timestamp("2025-01-06")
    ledger = pd.DataFrame(
        {
            "date": [date],
            "target_position": [0.5],
            "t_direction": ["BUY_FIRST"],
            "shares_at_start": [500],
            "shares": [500],
            "cash": [5000.0],
        }
    )
    bars = pd.DataFrame(
        {
            "date": [date] * 5,
            "session": ["AM"] * 5,
            "session_bar": range(5),
            "trade_time": pd.date_range("2025-01-06 09:35", periods=5, freq="5min"),
            "open": [9.7, 9.8, 10.1, 9.9, 10.1],
            "high": [9.8, 10.0, 10.2, 10.0, 10.2],
            "low": [9.6, 9.7, 10.0, 9.85, 10.0],
            "close": [9.7, 9.9, 10.1, 9.9, 10.1],
        }
    )
    zones = pd.DataFrame(
        {
            "date": [date],
            "session": ["AM"],
            "formation_bar": [2],
            "formation_time": [pd.Timestamp("2025-01-06 09:45")],
            "fvg_kind": ["BULLISH"],
            "zone_lower": [9.8],
            "zone_upper": [10.0],
            "zone_mid": [9.9],
            "gap_bps": [200.0],
        }
    )
    funnel = audit_fvg_funnel(
        ledger,
        bars,
        zones,
        StrategyCosts(),
        {
            "maximum_zone_age_bars": 12,
            "minimum_target_room_cost_multiple": 2.0,
        },
    )
    assert funnel["raw_fvg_events"] == 1
    assert funnel["direction_matched_events"] == 1
    assert funnel["midpoint_retracement_events"] == 1
    assert funnel["next_open_inside_events"] == 0
    assert funnel["economic_gate_events"] == 0


def _day_bars(date: pd.Timestamp) -> pd.DataFrame:
    times = pd.date_range(date + pd.Timedelta(hours=9, minutes=35), periods=6, freq="5min")
    return pd.DataFrame(
        {
            "date": date,
            "session": "AM",
            "session_bar": range(6),
            "trade_time": times,
            "open": [9.7, 9.8, 10.1, 9.9, 9.9, 10.0],
            "high": [9.8, 10.0, 10.2, 10.0, 10.0, 10.1],
            "low": [9.6, 9.7, 10.0, 9.85, 9.9, 9.9],
            "close": [9.7, 9.9, 10.1, 9.9, 10.0, 10.0],
        }
    )


def test_FVG单边买入可以跨日持有且不会强制日终卖回() -> None:
    dates = pd.bdate_range("2025-01-06", periods=3)
    daily = pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
        }
    )
    signals = pd.DataFrame(
        {
            "date": dates,
            "target_position": 0.5,
            "base_position": 0.5,
            "wave_z": 0.0,
            "tactical_adjustment": 0.0,
            "desired_position": 0.5,
        }
    )
    bars = pd.concat([_day_bars(date) for date in dates], ignore_index=True)
    zones = pd.DataFrame(
        {
            "date": dates,
            "session": "AM",
            "formation_bar": 2,
            "formation_time": dates + pd.Timedelta(hours=9, minutes=45),
            "fvg_kind": "BULLISH",
            "zone_lower": 9.8,
            "zone_upper": 10.0,
            "zone_mid": 9.9,
            "gap_bps": 200.0,
        }
    )
    dividends = pd.DataFrame(
        columns=["ex_date", "payment_date", "cash_dividend_per_share"]
    )
    costs = StrategyCosts(
        commission_rate=0.0,
        minimum_commission_cny=0.0,
        slippage_bps_per_leg=0.0,
        cash_annual_rate=0.0,
    )
    ledger, trades = run_inventory_backtest(
        daily,
        dividends,
        signals,
        bars,
        zones,
        10_000.0,
        costs,
        {
            "maximum_zone_age_bars": 12,
            "minimum_trade_notional_cny": 0.0,
            "maximum_one_way_commission_fraction": 1.0,
        },
        dates.min(),
        dates.max(),
    )
    assert len(trades) == 1
    assert trades.iloc[0]["side"] == "BUY"
    assert trades.iloc[0]["held_overnight_allowed"]
    assert ledger.loc[ledger["date"].eq(dates[1]), "shares"].iloc[0] == 500
    assert ledger.loc[ledger["date"].eq(dates[2]), "shares"].iloc[0] == 500
    assert trades.iloc[0]["shares_after"] == 500


def test_只有反向FVG锁定净利润时才允许可选日内第二条腿() -> None:
    dates = pd.bdate_range("2025-01-06", periods=3)
    daily = pd.DataFrame(
        {"date": dates, "open": 10.0, "high": 10.3, "low": 9.7, "close": 10.0}
    )
    signals = pd.DataFrame(
        {
            "date": dates,
            "target_position": [0.5, 0.0, 0.0],
            "base_position": [0.5, 0.0, 0.0],
            "wave_z": [0.0, 2.0, 2.0],
            "tactical_adjustment": 0.0,
            "desired_position": [0.5, 0.0, 0.0],
        }
    )
    first_day_bars = _day_bars(dates[1])
    second_times = pd.date_range(dates[2] + pd.Timedelta(hours=9, minutes=35), periods=7, freq="5min")
    second_day_bars = pd.DataFrame(
        {
            "date": dates[2],
            "session": "AM",
            "session_bar": range(7),
            "trade_time": second_times,
            "open": [10.0, 10.0, 10.2, 10.0, 9.8, 9.8, 10.0],
            "high": [10.1, 10.1, 10.25, 10.1, 9.85, 9.85, 10.1],
            "low": [9.9, 9.9, 10.15, 9.9, 9.75, 9.75, 9.9],
            "close": [10.0, 10.0, 10.2, 10.0, 9.8, 9.8, 10.0],
        }
    )
    bars = pd.concat([first_day_bars, second_day_bars], ignore_index=True)
    zones = pd.DataFrame(
        [
            {
                "date": dates[1], "session": "AM", "formation_bar": 2,
                "formation_time": dates[1] + pd.Timedelta(hours=9, minutes=45),
                "fvg_kind": "BULLISH", "zone_lower": 9.8, "zone_upper": 10.0,
                "zone_mid": 9.9, "gap_bps": 200.0,
            },
            {
                "date": dates[2], "session": "AM", "formation_bar": 1,
                "formation_time": dates[2] + pd.Timedelta(hours=9, minutes=40),
                "fvg_kind": "BEARISH", "zone_lower": 10.1, "zone_upper": 10.3,
                "zone_mid": 10.2, "gap_bps": 200.0,
            },
            {
                "date": dates[2], "session": "AM", "formation_bar": 3,
                "formation_time": dates[2] + pd.Timedelta(hours=9, minutes=50),
                "fvg_kind": "BULLISH", "zone_lower": 9.7, "zone_upper": 9.9,
                "zone_mid": 9.8, "gap_bps": 200.0,
            },
        ]
    )
    dividends = pd.DataFrame(
        columns=["ex_date", "payment_date", "cash_dividend_per_share"]
    )
    costs = StrategyCosts(
        commission_rate=0.0,
        minimum_commission_cny=0.0,
        slippage_bps_per_leg=0.0,
        cash_annual_rate=0.0,
    )
    ledger, trades = run_inventory_backtest(
        daily,
        dividends,
        signals,
        bars,
        zones,
        10_000.0,
        costs,
        {
            "maximum_zone_age_bars": 12,
            "minimum_trade_notional_cny": 0.0,
            "maximum_one_way_commission_fraction": 1.0,
            "optional_intraday_roundtrip": {
                "enabled": True,
                "minimum_locked_net_profit_cny": 10.0,
            },
        },
        dates.min(),
        dates.max(),
    )
    assert len(trades) == 3
    close = trades.loc[trades["trade_role"].eq("OPTIONAL_INTRADAY_T_CLOSE")].iloc[0]
    assert close["side"] == "BUY"
    assert np.isclose(close["locked_pair_net_pnl_cny"], 200.0)
    assert ledger.loc[ledger["date"].eq(dates[2]), "trade_legs_today"].iloc[0] == 2
    assert ledger.iloc[-1]["shares"] == 500


def test_新建半仓当日不能做T_隔夜后只卖旧仓并恢复份额() -> None:
    dates = pd.bdate_range("2025-01-06", periods=3)
    daily = pd.DataFrame(
        {
            "date": dates,
            "open": 10.0,
            "high": 10.2,
            "low": 9.8,
            "close": 10.0,
        }
    )
    signals = pd.DataFrame(
        {
            "date": dates,
            "target_position": 0.5,
            "wave_z": -1.0,
            "t_direction": "BUY_FIRST",
        }
    )
    bars = pd.concat([_day_bars(date) for date in dates], ignore_index=True)
    zones = pd.DataFrame(
        {
            "date": dates,
            "session": "AM",
            "formation_bar": 2,
            "formation_time": dates + pd.Timedelta(hours=9, minutes=45),
            "fvg_kind": "BULLISH",
            "zone_lower": 9.8,
            "zone_upper": 10.0,
            "zone_mid": 9.9,
            "gap_bps": 200.0,
        }
    )
    dividends = pd.DataFrame(
        columns=["ex_date", "payment_date", "cash_dividend_per_share"]
    )
    costs = StrategyCosts(
        commission_rate=0.0,
        minimum_commission_cny=0.0,
        slippage_bps_per_leg=0.0,
        cash_annual_rate=0.0,
    )
    ledger, _, t_trades = run_integrated_backtest(
        daily,
        dividends,
        signals,
        bars,
        zones,
        10_000.0,
        costs,
        {
            "maximum_zone_age_bars": 12,
            "maximum_hold_bars": 8,
            "minimum_target_room_cost_multiple": 2.0,
        },
        dates.min(),
        dates.max(),
        True,
    )
    assert not ledger.loc[ledger["date"].eq(dates[1]), "t_executed"].iloc[0]
    assert ledger.loc[ledger["date"].eq(dates[2]), "t_executed"].iloc[0]
    assert len(t_trades) == 1
    trade = t_trades.iloc[0]
    assert trade["sequence"] == "BUY_NEW_THEN_SELL_OLD"
    assert trade["entry_time"] > trade["touch_time"]
    assert trade["shares_before"] == trade["shares_after"] == 500
    assert ledger.iloc[-1]["shares"] == 500
