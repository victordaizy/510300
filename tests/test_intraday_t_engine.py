"""510300底仓做T回测引擎测试。"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from backtest.intraday_t_engine import (
    build_intraday_features,
    commission_for_notional,
    cost_model_from_config,
    load_strategy_config,
    maximum_affordable_lot,
    maximum_sell_first_lot_with_repurchase_reserve,
    simulate_period,
    unfavorable_execution_price,
)


def _make_day(
    date: str,
    closes: list[float],
    volumes: list[int],
    opens: list[float] | None = None,
) -> pd.DataFrame:
    times = pd.date_range(f"{date} 09:45", periods=len(closes), freq="1min")
    actual_opens = opens or closes
    rows: list[dict] = []
    for timestamp, open_price, close, volume in zip(times, actual_opens, closes, volumes):
        rows.append(
            {
                "ts_code": "510300.SH",
                "trade_time": timestamp,
                "open": open_price,
                "high": max(open_price, close) + 0.001,
                "low": min(open_price, close) - 0.001,
                "close": close,
                "vol": volume,
                "amount": int(round(close * volume)),
            }
        )
    return pd.DataFrame(rows)


def _two_day_buy_low_data() -> pd.DataFrame:
    first = _make_day(
        "2026-01-05",
        [5.0] * 9,
        [10_000_000] * 9,
    )
    closes = [5.00, 4.90, 4.91, 4.92, 4.93, 4.95, 5.00, 5.00, 5.00]
    opens = [5.00, 4.90, 4.91, 4.92, 4.93, 4.93, 4.99, 5.00, 5.00]
    second = _make_day(
        "2026-01-06",
        closes,
        [10_000_000] + [1_000_000] * 8,
        opens,
    )
    return pd.concat([first, second], ignore_index=True)


def _two_day_sell_high_data() -> pd.DataFrame:
    first = _make_day(
        "2026-01-05",
        [5.0] * 9,
        [10_000_000] * 9,
    )
    closes = [5.00, 5.10, 5.09, 5.08, 5.07, 5.05, 5.00, 5.00, 5.00]
    opens = [5.00, 5.10, 5.09, 5.08, 5.07, 5.07, 5.01, 5.00, 5.00]
    second = _make_day(
        "2026-01-06",
        closes,
        [10_000_000] + [1_000_000] * 8,
        opens,
    )
    return pd.concat([first, second], ignore_index=True)


def _empty_dividends() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["record_date", "payment_date", "cash_dividend_per_share"]
    )


def test_佣金最低收费与比例收费都正确() -> None:
    assert commission_for_notional(5_000.0, 0.0003, 5.0) == 5.0
    assert commission_for_notional(20_000.0, 0.0003, 5.0) == 6.0
    assert commission_for_notional(0.0, 0.0003, 5.0) == 0.0


def test_滑点和最小价位始终向不利方向取整() -> None:
    assert unfavorable_execution_price(5.000, "BUY", 5.0, 0.001) == 5.003
    assert unfavorable_execution_price(5.000, "SELL", 5.0, 0.001) == 4.997


def test_未来数据变化不会修改过去的分钟特征() -> None:
    original = _two_day_buy_low_data()
    changed = original.copy()
    cutoff = pd.Timestamp("2026-01-06 09:49")
    future = changed["trade_time"] > cutoff
    changed.loc[future, "close"] *= 1.20
    changed.loc[future, "amount"] = (
        changed.loc[future, "close"] * changed.loc[future, "vol"]
    ).round().astype(int)
    original_features = build_intraday_features(original)
    changed_features = build_intraday_features(changed)
    past = original_features["trade_time"] <= cutoff
    pd.testing.assert_frame_equal(
        original_features.loc[past, ["session_vwap", "vwap_deviation", "return_3_records"]],
        changed_features.loc[past, ["session_vwap", "vwap_deviation", "return_3_records"]],
    )


def test_买低做T只卖旧底仓并在日终恢复2500份() -> None:
    config = load_strategy_config()
    result = simulate_period(
        _two_day_buy_low_data(),
        _empty_dividends(),
        config,
        "2026-01-05",
        "2026-01-06",
        "单元测试",
        "BASE_COST",
        5.0,
    )
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["direction"] == "BUY_LOW"
    assert trade["sequence"] == "BUY_NEW_THEN_SELL_OLD"
    assert trade["sold_inventory_source"] == "OLD_CORE_INVENTORY"
    assert trade["entry_execution_time"] > trade["entry_signal_time"]
    assert trade["exit_execution_time"] > trade["exit_signal_time"]
    assert (result.daily_ledger["strategy_shares"] == 2500).all()
    assert result.daily_ledger.iloc[0]["round_trips_today"] == 0
    final_difference = (
        result.daily_ledger.iloc[-1]["strategy_nav_cny"]
        - result.daily_ledger.iloc[-1]["static_core_nav_cny"]
    )
    assert abs(final_difference - trade["net_pnl_cny"]) < 1e-7


def test_卖高做T只卖可卖旧底仓并按下一条记录成交() -> None:
    config = load_strategy_config()
    result = simulate_period(
        _two_day_sell_high_data(),
        _empty_dividends(),
        config,
        "2026-01-05",
        "2026-01-06",
        "单元测试",
        "BASE_COST",
        5.0,
    )
    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["direction"] == "SELL_HIGH"
    assert trade["sequence"] == "SELL_OLD_THEN_BUY_NEW"
    assert trade["sold_inventory_source"] == "OLD_CORE_INVENTORY_SOLD_AT_ENTRY"
    assert trade["shares_before"] == 2500
    assert trade["shares_after"] == 2500
    assert trade["entry_execution_time"] > trade["entry_signal_time"]
    assert trade["exit_execution_time"] > trade["exit_signal_time"]


def test_买入型交易不足1000份时直接跳过() -> None:
    config = deepcopy(load_strategy_config())
    model = cost_model_from_config(config)
    affordable = maximum_affordable_lot(
        cash_cny=1_000.0,
        raw_buy_price_cny=5.0,
        maximum_shares=1500,
        minimum_shares=1000,
        lot_size=100,
        cost_model=model,
        slippage_bps_per_leg=5.0,
    )
    assert affordable == 0


def test_先卖后买必须预留止损价回补资金() -> None:
    config = load_strategy_config()
    model = cost_model_from_config(config)
    no_capacity = maximum_sell_first_lot_with_repurchase_reserve(
        cash_cny=0.0,
        raw_sell_price_cny=5.0,
        maximum_shares=1500,
        minimum_shares=1000,
        lot_size=100,
        stop_loss_fraction=0.008,
        cost_model=model,
        slippage_bps_per_leg=15.0,
    )
    adequate_capacity = maximum_sell_first_lot_with_repurchase_reserve(
        cash_cny=100.0,
        raw_sell_price_cny=5.0,
        maximum_shares=1500,
        minimum_shares=1000,
        lot_size=100,
        stop_loss_fraction=0.008,
        cost_model=model,
        slippage_bps_per_leg=15.0,
    )
    assert no_capacity == 0
    assert adequate_capacity == 1500
