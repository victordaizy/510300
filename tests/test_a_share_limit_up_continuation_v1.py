"""A股涨停后可成交延续V1合成测试；不读取真实可见收益。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.a_share_limit_up_continuation_v1 import (
    CONFIG,
    build_limit_up_signals,
    limit_fraction,
    load_contract,
    run_portfolio_backtest,
    stock_cost_rate,
    theoretical_upper_limit_price,
)
from scripts.freeze_a_share_limit_up_continuation_v1 import (
    TRACKED_FILES,
    VISIBLE_SOURCE_DEPENDENCIES,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_signal_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2020-01-02", periods=150)
    signal_index = 130
    panel = pd.DataFrame(
        {
            "con_code": "600001.SH",
            "date": dates,
            "pre_close": 10.0,
            "raw_high": 10.0,
            "raw_close": 10.0,
            "pct_chg": 0.0,
            "amount": 300_000_000.0,
            "is_suspended": False,
        }
    )
    panel.loc[signal_index, ["raw_high", "raw_close", "pct_chg"]] = [
        11.0,
        11.0,
        10.0,
    ]
    master = pd.DataFrame(
        {
            "ts_code": ["600001.SH"],
            "symbol": ["600001"],
            "exchange": ["SSE"],
            "list_status": ["L"],
            "list_date": [pd.Timestamp("2010-01-01")],
            "delist_date": [pd.NaT],
            "split_bucket": [1],
            "split_group": ["TRAIN"],
        }
    )
    benchmark = pd.DataFrame({"date": dates})
    return panel, master, benchmark, dates


def _synthetic_backtest_inputs(
    *,
    entry_gap: float = 0.0,
    block_scheduled_exit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    dates = pd.bdate_range("2021-01-04", periods=11)
    start = dates[1]
    end = dates[-1]
    entry_date = dates[2]
    scheduled_exit_date = dates[7]
    signal = pd.DataFrame(
        {
            "con_code": ["600001.SH"],
            "signal_date": [dates[1]],
            "entry_date": [entry_date],
            "scheduled_exit_date": [scheduled_exit_date],
            "prior_median_amount_20": [300_000_000.0],
            "limit_fraction": [0.10],
        }
    )
    raw_open = np.full(len(dates), 10.0)
    total_return_open = np.full(len(dates), 10.0)
    total_return_close = np.full(len(dates), 10.0)
    raw_open[2] = 10.0 * (1.0 + entry_gap)
    total_return_open[2] = raw_open[2]
    total_return_close[2] = raw_open[2]
    if block_scheduled_exit:
        raw_open[7] = 9.01
        total_return_open[7] = 9.01
        total_return_close[7] = 9.01
        raw_open[8] = 10.0
        total_return_open[8] = 10.0
        total_return_close[8] = 10.0
    else:
        raw_open[7] = 11.0
        total_return_open[7] = 11.0
        total_return_close[7] = 11.0
    market = pd.DataFrame(
        {
            "con_code": "600001.SH",
            "date": dates,
            "pre_close": 10.0,
            "raw_open": raw_open,
            "amount": 300_000_000.0,
            "is_suspended": False,
            "total_return_open": total_return_open,
            "total_return_close": total_return_close,
        }
    )
    benchmark = pd.DataFrame(
        {
            "date": dates,
            "close": np.linspace(1000.0, 1010.0, len(dates)),
        }
    )
    return signal, market, benchmark, start, end


def test_contract_keeps_current_target_cost_and_safety_boundaries() -> None:
    contract = load_contract(CONFIG)
    assert contract["account"] == {
        "initial_capital_cny": 500000.0,
        "user_transaction_fee_rate_per_leg": 0.0001,
        "cash_annual_rate": 0.015,
    }
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["portfolio"]["maximum_positions"] == 10
    assert contract["signal"]["fixed_holding_period_trading_days_open_to_open"] == 5
    assert not any(contract["safety"].values())


def test_board_limit_fractions_and_round_half_up_are_frozen() -> None:
    assert limit_fraction("600001.SH", "2022-01-03") == 0.10
    assert limit_fraction("688001.SH", "2019-07-22") == 0.20
    assert limit_fraction("300001.SZ", "2020-08-21") == 0.10
    assert limit_fraction("300001.SZ", "2020-08-24") == 0.20
    assert limit_fraction("301001.SZ", "2022-01-03") == 0.20
    assert theoretical_upper_limit_price(10.05, 0.10) == pytest.approx(11.06)


def test_signal_uses_only_signal_day_and_prior_fields() -> None:
    contract = load_contract(CONFIG)
    panel, master, benchmark, dates = _synthetic_signal_inputs()
    panel_with_future = panel.copy()
    panel_with_future["raw_open"] = 10.0
    panel_with_future["total_return_open"] = 10.0
    panel_with_future["total_return_close"] = 10.0
    signals_before, audit_before = build_limit_up_signals(
        panel_with_future,
        master,
        benchmark,
        contract,
        start=dates[120],
        end=dates[-1],
    )
    panel_with_future.loc[131:, "raw_open"] = 9999.0
    panel_with_future.loc[131:, "total_return_open"] = 0.01
    panel_with_future.loc[131:, "total_return_close"] = 9999.0
    signals_after, audit_after = build_limit_up_signals(
        panel_with_future,
        master,
        benchmark,
        contract,
        start=dates[120],
        end=dates[-1],
    )
    pd.testing.assert_frame_equal(signals_before, signals_after)
    assert len(signals_before) == 1
    row = signals_before.iloc[0]
    assert row["signal_date"] == dates[130]
    assert row["entry_date"] == dates[131]
    assert row["scheduled_exit_date"] == dates[136]
    assert row["observed_history_days"] == 131
    assert audit_before["future_open_or_return_read"] is False
    assert audit_after["future_price_or_return_columns_used"] == []


def test_signal_requires_close_at_limit_and_high_equal_close() -> None:
    contract = load_contract(CONFIG)
    panel, master, benchmark, dates = _synthetic_signal_inputs()
    panel.loc[130, "raw_high"] = 11.10
    signals, _ = build_limit_up_signals(
        panel,
        master,
        benchmark,
        contract,
        start=dates[120],
        end=dates[-1],
    )
    assert signals.empty


def test_stock_cost_rates_include_user_fee_slippage_and_sell_stamp() -> None:
    contract = load_contract(CONFIG)
    assert stock_cost_rate(
        contract, scenario="base", side="BUY", date="2022-01-03"
    ) == pytest.approx(0.0011687)
    assert stock_cost_rate(
        contract, scenario="stress", side="BUY", date="2022-01-03"
    ) == pytest.approx(0.0031687)
    assert stock_cost_rate(
        contract, scenario="base", side="SELL", date="2022-01-03"
    ) == pytest.approx(0.0021687)
    assert stock_cost_rate(
        contract, scenario="stress", side="SELL", date="2024-01-03"
    ) == pytest.approx(0.0036687)


def test_portfolio_enters_next_open_exits_after_five_open_periods_without_leverage() -> None:
    contract = load_contract(CONFIG)
    signals, market, benchmark, start, end = _synthetic_backtest_inputs()
    daily, trades, audit = run_portfolio_backtest(
        signals,
        market,
        benchmark,
        contract,
        start=start,
        end=end,
    )
    assert audit["entry_transaction_count"] == 1
    assert audit["exit_transaction_count"] == 1
    assert audit["blocked_exit_attempt_count"] == 0
    assert trades["side"].tolist() == ["BUY", "SELL"]
    assert trades.iloc[0]["trade_date"] == signals.iloc[0]["entry_date"]
    assert trades.iloc[1]["trade_date"] == signals.iloc[0]["scheduled_exit_date"]
    assert trades.iloc[0]["gross_notional_cny"] <= 50_000.0
    assert trades.iloc[0]["shares"] % 100 == 0
    assert audit["maximum_position_count"] == 1
    assert audit["maximum_gross_exposure"] <= 1.0
    assert audit["maximum_net_exposure"] <= 1.0
    assert daily.iloc[-1]["position_count"] == 0
    assert daily[["quality_complete", "capacity_pass"]].all().all()


def test_entry_too_close_to_upper_limit_is_not_filled() -> None:
    contract = load_contract(CONFIG)
    signals, market, benchmark, start, end = _synthetic_backtest_inputs(
        entry_gap=0.099
    )
    daily, trades, audit = run_portfolio_backtest(
        signals,
        market,
        benchmark,
        contract,
        start=start,
        end=end,
    )
    assert trades.empty
    assert audit["upper_limit_entry_skip_count"] == 1
    assert audit["entry_transaction_count"] == 0
    assert daily["position_count"].eq(0).all()


def test_lower_limit_exit_is_carried_to_first_eligible_open() -> None:
    contract = load_contract(CONFIG)
    signals, market, benchmark, start, end = _synthetic_backtest_inputs(
        block_scheduled_exit=True
    )
    _, trades, audit = run_portfolio_backtest(
        signals,
        market,
        benchmark,
        contract,
        start=start,
        end=end,
    )
    sell = trades.loc[trades["side"].eq("SELL")].iloc[0]
    assert audit["blocked_exit_attempt_count"] == 1
    assert sell["trade_date"] > signals.iloc[0]["scheduled_exit_date"]
    assert sell["delayed_exit_days"] == 1


def test_freeze_scope_tracks_code_but_excludes_replication_inputs() -> None:
    assert "research/a_share_limit_up_continuation_v1.py" in TRACKED_FILES
    assert "scripts/run_a_share_limit_up_continuation_v1.py" in TRACKED_FILES
    assert all("holdout_panel" not in path for path in VISIBLE_SOURCE_DEPENDENCIES)
    assert all("bucket1_time_holdout" not in path for path in VISIBLE_SOURCE_DEPENDENCIES)
    assert (ROOT / "config" / "a_share_limit_up_continuation_v1.yaml").is_file()
