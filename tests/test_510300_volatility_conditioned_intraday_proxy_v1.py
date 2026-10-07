"""验证时点、跨日与分红会计，使用小型合成资料，不挑选历史参数。"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from research.volatility_conditioned_intraday_proxy_v1 import (
    affordable_quantity, build_features, detect_episodes, load_config, metrics, simulate_account,
)

CONFIG = load_config(Path(__file__).resolve().parents[1] / "config/510300_volatility_conditioned_intraday_proxy_v1.json")


def synthetic_bars(days=14):
    rng = np.random.default_rng(42)
    records = []
    for date in pd.bdate_range("2020-01-01", periods=days):
        clocks = pd.date_range(str(date.date()) + " 09:30", periods=40, freq="min")
        prices = 4 * np.exp(np.cumsum(rng.normal(0, .0005, len(clocks))))
        records.extend({"timestamp": clock, "close": price} for clock, price in zip(clocks, prices))
    return pd.DataFrame(records)


def short_config():
    config = deepcopy(CONFIG)
    config["volatility"]["recent_days"] = 2
    config["volatility"]["reference_days"] = 3
    config["signals"]["same_clock_sigma_days"] = 2
    return config


def test_current_and_future_price_changes_do_not_change_current_state_or_threshold():
    config = short_config()
    bars = synthetic_bars()
    target = pd.Timestamp("2020-01-15")
    original, _, thresholds = build_features(bars, config)
    changed = bars.copy()
    mask = changed.timestamp >= target
    changed.loc[mask, "close"] *= np.linspace(.5, 2, int(mask.sum()))
    revised, _, revised_thresholds = build_features(changed, config)
    pd.testing.assert_frame_equal(original.loc[original.date <= "2020-01-15"], revised.loc[revised.date <= "2020-01-15"])
    pd.testing.assert_frame_equal(thresholds.loc[:"2020-01-15"], revised_thresholds.loc[:"2020-01-15"])


def test_feature_prefix_and_event_prefix_reproduce_without_future_days():
    config = short_config()
    bars = synthetic_bars()
    states, prices, threshold = build_features(bars, config)
    cutoff = "2020-01-14"
    limited = bars.loc[bars.timestamp.dt.strftime("%Y-%m-%d") <= cutoff]
    s2, p2, t2 = build_features(limited, config)
    pd.testing.assert_frame_equal(states.loc[states.date <= cutoff].reset_index(drop=True), s2)
    events = detect_episodes(states, prices, threshold, config)
    e2 = detect_episodes(s2, p2, t2, config)
    pd.testing.assert_frame_equal(events.loc[events.date <= cutoff].reset_index(drop=True), e2)


def test_minimum_commission_and_lot_sizing_preserve_cash():
    cost = CONFIG["cost_scenarios"]["BASE"]
    assert affordable_quantity(404.99, 4, cost, 100) == 0
    assert affordable_quantity(405, 4, cost, 100) == 100
    assert affordable_quantity(20000, 4, cost, 100) == 4900


def account_fixture():
    dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
    states = pd.DataFrame({"date": dates, "regime": ["MEDIUM", "LOW", "LOW", "MEDIUM"]})
    prices = pd.DataFrame({"09:36": [4, 3.9, 3.9, 3.9], "15:00": [4, 3.9, 3.9, 3.9]}, index=dates)
    episodes = pd.DataFrame([{"date": dates[0], "direction": "RECOVERY", "status": "CONFIRMED", "entry_clock": "10:06", "entry_raw_proxy": 4.0, "event_id": "first"},
                             {"date": dates[0], "direction": "RECOVERY", "status": "CONFIRMED", "entry_clock": "11:06", "entry_raw_proxy": 3.8, "event_id": "later"},
                             {"date": dates[-1], "direction": "RECOVERY", "status": "CONFIRMED", "entry_clock": "10:06", "entry_raw_proxy": 3.9, "event_id": "terminal"}])
    dividends = pd.DataFrame([{"record_date": dates[0], "ex_date": dates[1], "payment_date": dates[2], "cash_dividend_per_share": .1}])
    return states, prices, episodes, dividends


def test_t_plus_one_first_candidate_cash_days_dividend_and_terminal_rule():
    states, prices, episodes, dividends = account_fixture()
    daily, trades = simulate_account(states, prices, episodes, dividends, CONFIG, "SWITCH", 20000, "BASE")
    assert len(daily) == 4 and len(trades) == 1
    trade = trades.iloc[0]
    assert trade.event_id == "first" and trade.exit_date == "2020-01-03"
    assert daily.iloc[0].receivable_cny == 0
    assert daily.iloc[1].receivable_cny == trade.quantity * .1
    assert daily.iloc[2].receivable_cny == 0
    assert daily.iloc[2].dividend_paid_cny == trade.quantity * .1
    assert abs(daily.iloc[2].equity_cny - daily.iloc[1].equity_cny) < 1e-8
    assert abs(trade.gross_pnl_cny) < 1e-8
    assert abs(daily.iloc[-1].equity_cny - 20000 - trade.net_pnl_cny) < 1e-8
    assert daily.no_order_cash_day.sum() == 2
    assert daily.iloc[-1].end_quantity == 0
    assert daily.iloc[-1].reason == "TERMINAL_DAY_NO_NEW_ENTRY"


def test_cash_policy_and_full_daily_denominator():
    states, prices, episodes, dividends = account_fixture()
    daily, trades = simulate_account(states, prices, episodes, dividends, CONFIG, "CASH", 20000, "BASE")
    result = metrics(daily, trades, 242, 20000)
    assert result["days"] == 4 and result["trade_count"] == 0
    assert result["total_return"] == 0 and result["sharpe"] is None
    assert result["no_order_cash_days"] == 4


def test_regime_cash_disables_entry_but_does_not_delay_existing_exit():
    states, prices, episodes, dividends = account_fixture()
    daily, trades = simulate_account(states, prices, episodes, dividends, CONFIG, "SWITCH", 20000, "BASE")
    assert daily.iloc[1].regime == "LOW"
    assert daily.iloc[1].sold_quantity > 0 and daily.iloc[1].bought_quantity == 0
    assert trades.iloc[0].exit_date == states.date.iloc[1]
