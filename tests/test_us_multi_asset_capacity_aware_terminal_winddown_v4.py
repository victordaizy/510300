"""美国多资产容量感知期末减仓V4合成测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.us_multi_asset_capacity_aware_absolute_momentum_v3 import (
    load_contract as load_v3_contract,
    run_portfolio_backtest,
)
from research.us_multi_asset_capacity_aware_terminal_winddown_v4 import (
    ALLOWED_CHANGED_PATHS,
    CONFIG,
    build_terminal_winddown_plan,
    changed_paths,
    load_contract,
)
from scripts.freeze_us_multi_asset_capacity_aware_terminal_winddown_v4 import (
    TRACKED_FILES,
    VISIBLE_SOURCE_DEPENDENCIES,
)


TICKERS = ["UPRO", "TQQQ", "TLT", "UGL"]


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


def _market_panel(
    dates: pd.DatetimeIndex,
    *,
    upro_prices: list[float] | None = None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for ticker in TICKERS:
        prices = np.full(len(dates), 100.0)
        if ticker == "UPRO" and upro_prices is not None:
            prices = np.asarray(upro_prices, dtype=float)
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "raw_open": prices,
                    "raw_high": prices * 1.01,
                    "raw_low": prices * 0.99,
                    "raw_close": prices,
                    "raw_volume": 2_000_000.0,
                    "raw_dollar_turnover_usd": prices * 2_000_000.0,
                    "qfq_factor": 1.0,
                    "adjust": 0.0,
                    "split_ratio_at_open": 1.0,
                    "cash_distribution_per_post_event_share_usd": 0.0,
                    "signal_total_return_index": 100.0,
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def _selection_row(
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    *,
    ticker: str = "UPRO",
    turnover: float = 10_000_000.0,
) -> dict[str, object]:
    return {
        "signal_date": signal_date,
        "execution_date": execution_date,
        "ticker": ticker,
        "selection_rank": 1,
        "momentum_252": 0.50,
        "realized_volatility_60": 0.30,
        "prior_median_dollar_turnover_20": turnover,
    }


def _feature_row(
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    *,
    ticker: str = "UPRO",
    turnover: float = 10_000_000.0,
) -> dict[str, object]:
    return {
        "signal_date": signal_date,
        "execution_date": execution_date,
        "ticker": ticker,
        "observed_bar_count": 500,
        "momentum_252": 0.50,
        "trend_sma_200": 80.0,
        "realized_volatility_60": 0.30,
        "prior_median_dollar_turnover_20": turnover,
        "eligible": True,
    }


def _one_entry_inputs(
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_date = dates[0] - pd.Timedelta(days=1)
    selections = pd.DataFrame([_selection_row(signal_date, dates[0])])
    schedule = pd.DataFrame(
        {"signal_date": [signal_date], "execution_date": [dates[0]]}
    )
    features = pd.DataFrame([_feature_row(signal_date, dates[0])])
    return selections, schedule, features


def test_contract_changes_only_terminal_winddown_and_keeps_hard_objective() -> None:
    contract = load_contract(CONFIG)
    base = load_v3_contract()
    assert set(changed_paths(contract)) == ALLOWED_CHANGED_PATHS
    for unchanged in (
        "historical_partition",
        "account",
        "universe",
        "signal",
        "costs",
        "risk",
        "visible_gates",
    ):
        assert contract[unchanged] == base[unchanged]
    assert contract["portfolio"]["terminal_winddown_common_trading_days"] == 20
    assert contract["portfolio"]["stop_new_entries_at_terminal_winddown_start"]
    assert contract["account"]["initial_capital_cny"] == 500000.0
    assert contract["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert not any(contract["safety"].values())


def test_winddown_starts_on_first_of_last_twenty_common_trading_days() -> None:
    contract = load_contract(CONFIG)
    dates = pd.bdate_range("2020-01-02", periods=25)
    early_signal = dates[0] - pd.Timedelta(days=1)
    late_signal = dates[9]
    selections = pd.DataFrame(
        [
            _selection_row(early_signal, dates[0]),
            _selection_row(late_signal, dates[10], ticker="TLT"),
        ]
    )
    schedule = selections[["signal_date", "execution_date"]].drop_duplicates()
    features = pd.DataFrame(
        [
            _feature_row(early_signal, dates[0]),
            _feature_row(late_signal, dates[10], ticker="TLT"),
        ]
    )
    planned_selections, planned_schedule, planned_features, audit = (
        build_terminal_winddown_plan(
            selections,
            schedule,
            features,
            _context(dates),
            contract,
        )
    )
    expected = dates[-20]
    assert audit["winddown_start_date"] == expected.date().isoformat()
    assert planned_selections["execution_date"].tolist() == [dates[0]]
    assert planned_features["execution_date"].tolist() == [dates[0]]
    assert planned_schedule["execution_date"].tolist() == [dates[0], expected]
    assert planned_schedule.iloc[-1]["signal_date"] == expected
    assert audit["new_entries_after_winddown"] == 0
    assert audit["market_return_or_future_open_read"] is False


def test_fixed_winddown_slices_exit_within_capacity_and_reaches_zero() -> None:
    contract = load_contract(CONFIG)
    dates = pd.bdate_range("2020-01-02", periods=25)
    selections, schedule, features = _one_entry_inputs(dates)
    planned = build_terminal_winddown_plan(
        selections,
        schedule,
        features,
        _context(dates),
        contract,
    )
    prices = [100.0] * 5 + [200.0] * 20
    daily, trades, audit = run_portfolio_backtest(
        planned[0],
        planned[1],
        planned[2],
        _market_panel(dates, upro_prices=prices),
        _context(dates),
        contract,
        start=dates[0],
        end=dates[-1],
    )
    buys = trades.loc[trades["side"].eq("BUY")]
    sells = trades.loc[trades["side"].eq("SELL")]
    assert buys["shares"].tolist() == [50]
    assert sells["shares"].tolist() == [25, 25]
    assert sells["trade_date"].tolist() == [dates[-20], dates[-19]]
    assert trades["capacity_fraction"].le(0.0005).all()
    assert audit["maximum_observed_order_capacity_fraction"] <= 0.0005
    assert audit["terminal_position_count"] == 0
    assert daily.iloc[-1]["position_count"] == 0


def test_winddown_date_does_not_depend_on_returns_or_open_prices() -> None:
    contract = load_contract(CONFIG)
    dates = pd.bdate_range("2020-01-02", periods=25)
    selections, schedule, features = _one_entry_inputs(dates)
    plain_context = _context(dates)
    altered_context = plain_context.copy()
    altered_context["unused_future_return"] = np.linspace(-0.9, 4.0, len(dates))
    altered_context["unused_future_open"] = np.linspace(1.0, 1_000_000.0, len(dates))
    first = build_terminal_winddown_plan(
        selections, schedule, features, plain_context, contract
    )
    second = build_terminal_winddown_plan(
        selections, schedule, features, altered_context, contract
    )
    for left, right in zip(first[:3], second[:3], strict=True):
        pd.testing.assert_frame_equal(left, right)
    assert first[3] == second[3]
    assert first[3]["market_return_or_future_open_read"] is False


def test_freeze_scope_tracks_v4_and_excludes_sealed_inputs() -> None:
    assert "research/us_multi_asset_capacity_aware_terminal_winddown_v4.py" in TRACKED_FILES
    assert "scripts/run_us_multi_asset_capacity_aware_terminal_winddown_v4.py" in TRACKED_FILES
    assert all("sealed" not in path.lower() for path in VISIBLE_SOURCE_DEPENDENCIES)
    assert all("daily_panel.parquet" not in path for path in VISIBLE_SOURCE_DEPENDENCIES)
