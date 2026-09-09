"""美国多资产容量感知绝对动量V3合成测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import DataContractError
from research.us_liquid_treasury_multi_asset_absolute_momentum_v2 import (
    load_contract as load_v2_contract,
)
from research.us_multi_asset_capacity_aware_absolute_momentum_v3 import (
    ALLOWED_CHANGED_PATHS,
    CONFIG,
    build_monthly_selections,
    changed_paths,
    load_contract,
    run_portfolio_backtest,
)
from scripts.freeze_us_multi_asset_capacity_aware_absolute_momentum_v3 import (
    TRACKED_FILES,
    VISIBLE_SOURCE_DEPENDENCIES,
)


TICKERS = ["UPRO", "TQQQ", "TLT", "UGL"]


def _signal_panel() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2017-01-03", periods=430)
    slopes = {"UPRO": 0.0020, "TQQQ": 0.0015, "TLT": -0.0004, "UGL": 0.0010}
    parts: list[pd.DataFrame] = []
    for ticker in TICKERS:
        index = np.arange(len(dates), dtype=float)
        tri = 100.0 * np.exp(slopes[ticker] * index + 0.002 * np.sin(index / 7.0))
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "raw_close": 100.0,
                    "raw_volume": 2_000_000.0,
                    "raw_dollar_turnover_usd": 200_000_000.0,
                    "signal_total_return_index": tri,
                }
            )
        )
    return pd.concat(parts, ignore_index=True), dates


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


def _selection_row(
    *,
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    ticker: str,
    rank: int,
    turnover: float,
) -> dict[str, object]:
    return {
        "signal_date": signal_date,
        "execution_date": execution_date,
        "ticker": ticker,
        "selection_rank": rank,
        "momentum_252": 0.50 - rank * 0.05,
        "realized_volatility_60": 0.30,
        "prior_median_dollar_turnover_20": turnover,
    }


def _feature_row(
    *,
    signal_date: pd.Timestamp,
    execution_date: pd.Timestamp,
    ticker: str,
    turnover: float,
    eligible: bool,
) -> dict[str, object]:
    return {
        "signal_date": signal_date,
        "execution_date": execution_date,
        "ticker": ticker,
        "observed_bar_count": 500,
        "momentum_252": 0.5 if eligible else -0.1,
        "trend_sma_200": 80.0,
        "realized_volatility_60": 0.3,
        "prior_median_dollar_turnover_20": turnover,
        "eligible": eligible,
    }


def test_contract_changes_only_capacity_execution_and_keeps_hard_objective() -> None:
    contract = load_contract(CONFIG)
    base = load_v2_contract()
    assert set(changed_paths(contract)) == ALLOWED_CHANGED_PATHS
    for unchanged in (
        "historical_partition",
        "universe",
        "signal",
        "costs",
        "visible_gates",
    ):
        assert contract[unchanged] == base[unchanged]
    assert contract["account"]["initial_capital_cny"] == 500000.0
    assert contract["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["portfolio"]["target_weight_per_selected_asset"] == 0.50
    assert not any(contract["safety"].values())


def test_selection_is_identical_to_v2() -> None:
    v2 = load_v2_contract()
    v3 = load_contract(CONFIG)
    panel, dates = _signal_panel()
    v2_result = build_monthly_selections(
        panel,
        v2,
        start=dates[300],
        end=dates[-1],
    )
    v3_result = build_monthly_selections(
        panel,
        v3,
        start=dates[300],
        end=dates[-1],
    )
    for left, right in zip(v2_result[:3], v3_result[:3], strict=True):
        pd.testing.assert_frame_equal(left, right)
    assert v2_result[3] == v3_result[3]


def test_target_is_capped_without_any_order_exceeding_capacity() -> None:
    contract = load_contract(CONFIG)
    dates = pd.bdate_range("2020-01-02", periods=5)
    signal_date = pd.Timestamp("2019-12-31")
    selections = pd.DataFrame(
        [
            _selection_row(
                signal_date=signal_date,
                execution_date=dates[0],
                ticker="UPRO",
                rank=1,
                turnover=20_000_000.0,
            ),
            _selection_row(
                signal_date=signal_date,
                execution_date=dates[0],
                ticker="TLT",
                rank=2,
                turnover=100_000_000.0,
            ),
        ]
    )
    schedule = pd.DataFrame(
        {"signal_date": [signal_date], "execution_date": [dates[0]]}
    )
    features = pd.DataFrame(
        [
            _feature_row(
                signal_date=signal_date,
                execution_date=dates[0],
                ticker=ticker,
                turnover=20_000_000.0 if ticker == "UPRO" else 100_000_000.0,
                eligible=ticker in {"UPRO", "TLT"},
            )
            for ticker in TICKERS
        ]
    )
    daily, trades, audit = run_portfolio_backtest(
        selections,
        schedule,
        features,
        _market_panel(dates),
        _context(dates),
        contract,
        start=dates[0],
        end=dates[-1],
    )
    assert audit["capacity_target_capped_count"] == 1
    assert audit["capacity_block_count"] == 0
    assert audit["maximum_observed_order_capacity_fraction"] <= 0.0005
    assert trades["capacity_fraction"].le(0.0005).all()
    assert daily["capacity_pass"].all()
    assert daily.iloc[-1]["position_count"] == 0
    assert daily.iloc[-1]["base_nav_cny"] > daily.iloc[-1]["stress_nav_cny"]


def test_exit_above_daily_capacity_is_sliced_across_days() -> None:
    contract = load_contract(CONFIG)
    dates = pd.bdate_range("2020-01-02", periods=4)
    first_signal = pd.Timestamp("2019-12-31")
    exit_signal = pd.Timestamp("2020-01-02")
    selections = pd.DataFrame(
        [
            _selection_row(
                signal_date=first_signal,
                execution_date=dates[0],
                ticker="UPRO",
                rank=1,
                turnover=10_000_000.0,
            )
        ]
    )
    schedule = pd.DataFrame(
        {
            "signal_date": [first_signal, exit_signal],
            "execution_date": [dates[0], dates[1]],
        }
    )
    features = pd.DataFrame(
        [
            _feature_row(
                signal_date=first_signal,
                execution_date=dates[0],
                ticker="UPRO",
                turnover=10_000_000.0,
                eligible=True,
            ),
            _feature_row(
                signal_date=exit_signal,
                execution_date=dates[1],
                ticker="UPRO",
                turnover=10_000_000.0,
                eligible=False,
            ),
        ]
    )
    daily, trades, audit = run_portfolio_backtest(
        selections,
        schedule,
        features,
        _market_panel(dates, upro_prices=[100.0, 200.0, 200.0, 200.0]),
        _context(dates),
        contract,
        start=dates[0],
        end=dates[-1],
    )
    sells = trades.loc[trades["side"].eq("SELL")]
    assert sells["shares"].tolist() == [25, 25]
    assert sells["trade_date"].tolist() == [dates[1], dates[2]]
    assert audit["capacity_sliced_exit_transaction_count"] == 1
    assert sells["capacity_fraction"].le(0.0005).all()
    assert daily.iloc[-1]["position_count"] == 0


def test_terminal_exit_cannot_exceed_daily_capacity() -> None:
    contract = load_contract(CONFIG)
    dates = pd.bdate_range("2020-01-02", periods=2)
    signal_date = pd.Timestamp("2019-12-31")
    selections = pd.DataFrame(
        [
            _selection_row(
                signal_date=signal_date,
                execution_date=dates[0],
                ticker="UPRO",
                rank=1,
                turnover=10_000_000.0,
            )
        ]
    )
    schedule = pd.DataFrame(
        {"signal_date": [signal_date], "execution_date": [dates[0]]}
    )
    features = pd.DataFrame(
        [
            _feature_row(
                signal_date=signal_date,
                execution_date=dates[0],
                ticker="UPRO",
                turnover=10_000_000.0,
                eligible=True,
            )
        ]
    )
    with pytest.raises(DataContractError, match="评价期末容量内无法完整退出"):
        run_portfolio_backtest(
            selections,
            schedule,
            features,
            _market_panel(dates, upro_prices=[100.0, 200.0]),
            _context(dates),
            contract,
            start=dates[0],
            end=dates[-1],
        )


def test_freeze_scope_tracks_implementation_and_excludes_sealed_inputs() -> None:
    assert "research/us_multi_asset_capacity_aware_absolute_momentum_v3.py" in TRACKED_FILES
    assert "scripts/run_us_multi_asset_capacity_aware_absolute_momentum_v3.py" in TRACKED_FILES
    assert all("sealed" not in path.lower() for path in VISIBLE_SOURCE_DEPENDENCIES)
    assert all("daily_panel.parquet" not in path for path in VISIBLE_SOURCE_DEPENDENCIES)
