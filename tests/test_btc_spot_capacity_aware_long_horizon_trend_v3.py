"""BTC现货容量感知长周期趋势V3合成测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.btc_spot_capacity_aware_long_horizon_trend_v3 import (
    ALLOWED_CHANGED_PATHS,
    CONFIG,
    build_terminal_winddown_plan,
    changed_paths,
    load_contract,
    run_portfolio_backtest,
)
from research.btc_spot_long_horizon_trend_v2 import (
    build_weekly_targets,
    load_contract as load_v2_contract,
)
from scripts.freeze_btc_spot_capacity_aware_long_horizon_trend_v3 import (
    TRACKED_FILES,
    VISIBLE_SOURCE_DEPENDENCIES,
)


def _context(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "cny_per_usd": 7.0,
            "fx_source_date": dates - pd.Timedelta(days=1),
            "fx_age_calendar_days": 1,
            "benchmark_source_date": dates,
            "benchmark_close": 1000.0,
            "benchmark_total_return": 0.0,
        }
    )


def _market_panel(dates: pd.DatetimeIndex, prices: list[float]) -> pd.DataFrame:
    values = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {
            "product_id": "BTC-USD",
            "date": dates,
            "open": values,
            "high": values * 1.01,
            "low": values * 0.99,
            "close": values,
            "volume": 2_000_000.0,
            "dollar_turnover_usd": values * 2_000_000.0,
        }
    )


def _one_entry_inputs(
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_date = dates[0] - pd.Timedelta(days=1)
    targets = pd.DataFrame(
        [
            {
                "signal_date": signal_date,
                "execution_date": dates[0],
                "product_id": "BTC-USD",
                "sma_200": 80.0,
                "momentum_126": 0.5,
                "realized_volatility_60": 0.8,
                "prior_median_dollar_turnover_20": 1_000_000.0,
                "target_weight": 0.5,
            }
        ]
    )
    schedule = pd.DataFrame(
        {"signal_date": [signal_date], "execution_date": [dates[0]]}
    )
    features = pd.DataFrame(
        [
            {
                "signal_date": signal_date,
                "execution_date": dates[0],
                "product_id": "BTC-USD",
                "observed_bar_count": 300,
                "sma_200": 80.0,
                "momentum_126": 0.5,
                "realized_volatility_60": 0.8,
                "prior_median_dollar_turnover_20": 1_000_000.0,
                "eligible": True,
            }
        ]
    )
    return targets, schedule, features


def _signal_panel() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.date_range("2020-01-01", periods=430, freq="D")
    index = np.arange(len(dates), dtype=float)
    close = 100.0 * np.exp(0.0025 * index + 0.004 * np.sin(index / 8.0))
    return (
        pd.DataFrame(
            {
                "product_id": "BTC-USD",
                "date": dates,
                "open": close * 0.999,
                "close": close,
                "volume": 2_000_000.0,
                "dollar_turnover_usd": close * 2_000_000.0,
            }
        ),
        dates,
    )


def test_contract_changes_only_capacity_execution_and_keeps_hard_objective() -> None:
    contract = load_contract(CONFIG)
    base = load_v2_contract()
    assert set(changed_paths(contract)) == ALLOWED_CHANGED_PATHS
    for unchanged in (
        "historical_partition", "account", "universe", "signal", "costs",
        "risk", "visible_gates",
    ):
        assert contract[unchanged] == base[unchanged]
    assert contract["portfolio"]["terminal_winddown_calendar_days"] == 30
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert not any(contract["safety"].values())


def test_signal_targets_are_identical_to_v2() -> None:
    base = load_v2_contract()
    contract = load_contract(CONFIG)
    panel, dates = _signal_panel()
    left = build_weekly_targets(panel, base, start=dates[250], end=dates[-1])
    right = build_weekly_targets(panel, contract, start=dates[250], end=dates[-1])
    for first, second in zip(left[:3], right[:3], strict=True):
        pd.testing.assert_frame_equal(first, second)
    assert left[3] == right[3]


def test_winddown_starts_on_first_of_last_thirty_calendar_days() -> None:
    contract = load_contract(CONFIG)
    dates = pd.date_range("2022-01-01", periods=35, freq="D")
    targets, schedule, features = _one_entry_inputs(dates)
    planned = build_terminal_winddown_plan(
        targets, schedule, features, _context(dates), contract
    )
    expected = dates[-30]
    assert planned[3]["winddown_start_date"] == expected.date().isoformat()
    assert planned[1]["execution_date"].tolist() == [dates[0], expected]
    assert planned[3]["market_return_or_future_open_read"] is False


def test_capacity_target_and_two_day_sliced_exit_stay_within_limit() -> None:
    contract = load_contract(CONFIG)
    dates = pd.date_range("2022-01-01", periods=35, freq="D")
    targets, schedule, features = _one_entry_inputs(dates)
    planned = build_terminal_winddown_plan(
        targets, schedule, features, _context(dates), contract
    )
    prices = [100.0] * 5 + [200.0] * 30
    daily, trades, audit = run_portfolio_backtest(
        planned[0], planned[1], planned[2],
        _market_panel(dates, prices), _context(dates), contract,
        start=dates[0], end=dates[-1],
    )
    buys = trades.loc[trades["side"].eq("BUY")]
    sells = trades.loc[trades["side"].eq("SELL")]
    assert buys["quantity"].tolist() == [30.0]
    assert sells["quantity"].tolist() == [15.0, 15.0]
    assert sells["trade_date"].tolist() == [dates[-30], dates[-29]]
    assert trades["capacity_fraction"].le(0.003).all()
    assert audit["capacity_target_capped_count"] == 1
    assert audit["capacity_sliced_exit_transaction_count"] == 1
    assert audit["terminal_position_count"] == 0
    assert daily.iloc[-1]["position_count"] == 0


def test_freeze_scope_tracks_v3_and_excludes_sealed_inputs() -> None:
    assert "research/btc_spot_capacity_aware_long_horizon_trend_v3.py" in TRACKED_FILES
    assert "scripts/run_btc_spot_capacity_aware_long_horizon_trend_v3.py" in TRACKED_FILES
    assert all("sealed" not in path.lower() for path in VISIBLE_SOURCE_DEPENDENCIES)
