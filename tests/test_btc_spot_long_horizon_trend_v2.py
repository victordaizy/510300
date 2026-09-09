"""BTC现货长周期趋势V2合成测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.btc_spot_long_horizon_trend_v2 import (
    CONFIG,
    build_weekly_targets,
    load_contract,
    run_portfolio_backtest,
)
from research.digital_asset_spot_volatility_scaled_trend_v1 import (
    fx_spread_rate,
    spot_cost_rate,
)
from scripts.freeze_btc_spot_long_horizon_trend_v2 import (
    TRACKED_FILES,
    VISIBLE_SOURCE_DEPENDENCIES,
)


def _trend_panel(*, daily_log_return: float = 0.003) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.date_range("2020-01-01", periods=430, freq="D")
    index = np.arange(len(dates), dtype=float)
    close = 100.0 * np.exp(daily_log_return * index + 0.004 * np.sin(index / 9.0))
    volume = np.full(len(dates), 2_000_000.0)
    return (
        pd.DataFrame(
            {
                "product_id": "BTC-USD",
                "date": dates,
                "open": close * 0.999,
                "close": close,
                "volume": volume,
                "dollar_turnover_usd": close * volume,
            }
        ),
        dates,
    )


def _portfolio_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2022-01-03", periods=5, freq="D")
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
                "prior_median_dollar_turnover_20": 100_000_000.0,
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
                "prior_median_dollar_turnover_20": 100_000_000.0,
                "eligible": True,
            }
        ]
    )
    panel = pd.DataFrame(
        {
            "product_id": "BTC-USD",
            "date": dates,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 2_000_000.0,
            "dollar_turnover_usd": 200_000_000.0,
        }
    )
    context = pd.DataFrame(
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
    return targets, schedule, features, panel, context


def test_contract_keeps_500k_40pct_high_sharpe_and_research_boundary() -> None:
    contract = load_contract(CONFIG)
    assert contract["account"]["initial_capital_cny"] == 500000.0
    assert contract["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["universe"]["fixed_products"] == ["BTC-USD"]
    assert contract["signal"]["trend_sma_calendar_days"] == 200
    assert contract["signal"]["momentum_lookback_calendar_days"] == 126
    assert contract["signal"]["realized_volatility_lookback_calendar_days"] == 60
    assert contract["signal"]["portfolio_target_annualized_volatility"] == 0.40
    assert not any(contract["safety"].values())


def test_costs_include_user_fee_exchange_fee_slippage_and_impact() -> None:
    contract = load_contract(CONFIG)
    assert spot_cost_rate(contract, "base") == pytest.approx(0.0076)
    assert spot_cost_rate(contract, "stress") == pytest.approx(0.0106)
    assert fx_spread_rate(contract, "base") == pytest.approx(0.0010)
    assert fx_spread_rate(contract, "stress") == pytest.approx(0.0030)


def test_signal_uses_completed_sunday_and_next_monday_without_leverage() -> None:
    contract = load_contract(CONFIG)
    panel, dates = _trend_panel()
    targets, schedule, features, audit = build_weekly_targets(
        panel,
        contract,
        start=dates[250],
        end=dates[-1],
    )
    assert not targets.empty
    assert set(targets["product_id"]) == {"BTC-USD"}
    assert targets["target_weight"].between(0.0, 1.0, inclusive="right").all()
    assert (schedule["signal_date"].dt.weekday == 6).all()
    assert (schedule["execution_date"].dt.weekday == 0).all()
    assert (
        schedule["execution_date"] - schedule["signal_date"]
        == pd.Timedelta(days=1)
    ).all()
    assert features["signal_date"].nunique() == len(schedule)
    assert audit["future_open_or_strategy_return_read"] is False


def test_future_execution_open_cannot_change_frozen_targets() -> None:
    contract = load_contract(CONFIG)
    panel, dates = _trend_panel()
    before = build_weekly_targets(
        panel,
        contract,
        start=dates[250],
        end=dates[-1],
    )[0]
    altered = panel.copy()
    altered.loc[altered["date"].gt(dates[300]), "open"] = 999_999_999.0
    after = build_weekly_targets(
        altered,
        contract,
        start=dates[250],
        end=dates[-1],
    )[0]
    pd.testing.assert_frame_equal(before, after)


def test_empty_trend_target_keeps_weekly_schedule_for_mandatory_exit() -> None:
    contract = load_contract(CONFIG)
    panel, dates = _trend_panel(daily_log_return=-0.003)
    targets, schedule, _features, audit = build_weekly_targets(
        panel,
        contract,
        start=dates[250],
        end=dates[-1],
    )
    assert targets.empty
    assert not schedule.empty
    assert audit["signal_dates_with_positive_target_weight"] == 0


def test_fractional_spot_ledger_applies_dual_costs_and_exits_at_terminal_open() -> None:
    contract = load_contract(CONFIG)
    targets, schedule, features, panel, context = _portfolio_inputs()
    daily, trades, audit = run_portfolio_backtest(
        targets,
        schedule,
        features,
        panel,
        context,
        contract,
        start=context["date"].iloc[0],
        end=context["date"].iloc[-1],
    )
    assert audit["entry_transaction_count"] == 1
    assert audit["exit_transaction_count"] == 1
    assert audit["terminal_position_count"] == 0
    assert trades["capacity_fraction"].le(0.003).all()
    assert daily.iloc[-1]["position_count"] == 0
    assert daily.iloc[-1]["base_nav_cny"] > daily.iloc[-1]["stress_nav_cny"]


def test_freeze_scope_tracks_candidate_and_excludes_sealed_inputs() -> None:
    assert "research/btc_spot_long_horizon_trend_v2.py" in TRACKED_FILES
    assert "scripts/run_btc_spot_long_horizon_trend_v2.py" in TRACKED_FILES
    assert all("sealed" not in path.lower() for path in VISIBLE_SOURCE_DEPENDENCIES)
    assert all("replication_panel" not in path for path in VISIBLE_SOURCE_DEPENDENCIES)
