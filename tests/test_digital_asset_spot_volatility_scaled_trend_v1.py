"""数字资产现货波动缩放趋势V1的纯合成数据测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import DataContractError
from research.digital_asset_spot_volatility_scaled_trend_v1 import (
    CONFIG,
    build_weekly_targets,
    fx_spread_rate,
    load_contract,
    prepare_daily_context,
    run_portfolio_backtest,
    spot_cost_rate,
)
from scripts.freeze_digital_asset_spot_volatility_scaled_trend_v1 import (
    TRACKED_FILES,
    VISIBLE_SOURCE_DEPENDENCIES,
)


PRODUCTS = ["BTC-USD", "ETH-USD"]


def _signal_panel() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.date_range("2021-01-01", periods=220, freq="D")
    index = np.arange(len(dates), dtype=float)
    returns = {
        "BTC-USD": 0.004 + 0.012 * np.sin(index * 0.41),
        "ETH-USD": -0.004 + 0.008 * np.cos(index * 0.37),
    }
    parts: list[pd.DataFrame] = []
    for product_id in PRODUCTS:
        close = 100.0 * np.cumprod(1.0 + returns[product_id])
        volume = np.full(len(dates), 2_000_000.0)
        parts.append(
            pd.DataFrame(
                {
                    "product_id": product_id,
                    "date": dates,
                    "open": close * 0.999,
                    "close": close,
                    "volume": volume,
                    "dollar_turnover_usd": close * volume,
                }
            )
        )
    return pd.concat(parts, ignore_index=True), dates


def _portfolio_inputs(
    *,
    block_terminal_btc: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2022-01-03", periods=5, freq="D")
    signal_date = dates[0] - pd.Timedelta(days=1)
    targets = pd.DataFrame(
        [
            {
                "signal_date": signal_date,
                "execution_date": dates[0],
                "product_id": product_id,
                "momentum_20": 0.25,
                "realized_volatility_56": 0.70,
                "prior_median_dollar_turnover_20": 100_000_000.0,
                "preliminary_weight": 0.50,
                "preliminary_portfolio_volatility": 0.70,
                "volatility_scale": 0.90,
                "target_weight": 0.45,
            }
            for product_id in PRODUCTS
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
                "product_id": product_id,
                "observed_bar_count": 200,
                "momentum_20": 0.25,
                "realized_volatility_56": 0.70,
                "prior_median_dollar_turnover_20": 100_000_000.0,
                "eligible": True,
            }
            for product_id in PRODUCTS
        ]
    )
    panel_parts: list[pd.DataFrame] = []
    for product_id in PRODUCTS:
        volume = np.full(len(dates), 2_000_000.0)
        if product_id == "BTC-USD" and block_terminal_btc:
            volume[-1] = 0.0
        panel_parts.append(
            pd.DataFrame(
                {
                    "product_id": product_id,
                    "date": dates,
                    "open": np.full(len(dates), 100.0),
                    "high": np.full(len(dates), 101.0),
                    "low": np.full(len(dates), 99.0),
                    "close": np.full(len(dates), 100.0),
                    "volume": volume,
                    "dollar_turnover_usd": 100.0 * volume,
                }
            )
        )
    panel = pd.concat(panel_parts, ignore_index=True)
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


def test_contract_keeps_500k_40pct_high_sharpe_and_research_boundaries() -> None:
    contract = load_contract(CONFIG)
    assert contract["account"]["initial_capital_cny"] == 500000.0
    assert contract["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["universe"]["fixed_products"] == PRODUCTS
    assert contract["portfolio"]["blocked_exit_reselection_policy"] == (
        "PENDING_EXIT_REMAINS_MANDATORY"
    )
    assert not any(contract["safety"].values())


def test_costs_include_user_fee_exchange_fee_and_conservative_friction() -> None:
    contract = load_contract(CONFIG)
    assert spot_cost_rate(contract, "base") == pytest.approx(0.0076)
    assert spot_cost_rate(contract, "stress") == pytest.approx(0.0106)
    assert fx_spread_rate(contract, "base") == pytest.approx(0.0010)
    assert fx_spread_rate(contract, "stress") == pytest.approx(0.0030)


def test_weekly_signal_uses_completed_sunday_and_executes_monday() -> None:
    contract = load_contract(CONFIG)
    panel, dates = _signal_panel()
    targets, schedule, features, audit = build_weekly_targets(
        panel,
        contract,
        start=dates[90],
        end=dates[-1],
    )
    assert not targets.empty
    assert set(targets["product_id"]) == {"BTC-USD"}
    assert targets["target_weight"].gt(0.0).all()
    assert targets.groupby("signal_date")["target_weight"].sum().le(1.0).all()
    assert (schedule["signal_date"].dt.weekday == 6).all()
    assert (schedule["execution_date"].dt.weekday == 0).all()
    assert (
        schedule["execution_date"] - schedule["signal_date"]
        == pd.Timedelta(days=1)
    ).all()
    assert audit["future_open_or_strategy_return_read"] is False
    assert features["signal_date"].nunique() == len(schedule)


def test_future_execution_open_perturbation_cannot_change_targets() -> None:
    contract = load_contract(CONFIG)
    panel, dates = _signal_panel()
    before, _, _, _ = build_weekly_targets(
        panel,
        contract,
        start=dates[90],
        end=dates[-1],
    )
    panel.loc[panel["date"].gt(dates[150]), "open"] = 999_999.0
    after, _, _, _ = build_weekly_targets(
        panel,
        contract,
        start=dates[90],
        end=dates[-1],
    )
    pd.testing.assert_frame_equal(before, after)


def test_daily_context_uses_fx_strictly_before_utc_date() -> None:
    contract = load_contract(CONFIG)
    dates = pd.date_range("2021-12-20", "2022-01-10", freq="D")
    panel = pd.concat(
        [pd.DataFrame({"product_id": product_id, "date": dates}) for product_id in PRODUCTS],
        ignore_index=True,
    )
    fx_dates = pd.date_range("2021-12-15", "2022-01-10", freq="D")
    fx = pd.DataFrame(
        {"date": fx_dates, "cny_per_usd": np.arange(len(fx_dates)) + 6.0}
    )
    benchmark = pd.DataFrame(
        {"date": dates, "close": np.arange(len(dates), dtype=float) + 1000.0}
    )
    context = prepare_daily_context(
        panel,
        fx,
        benchmark,
        contract,
        start=pd.Timestamp("2022-01-03"),
        end=pd.Timestamp("2022-01-10"),
    )
    assert (context["fx_source_date"] < context["date"]).all()
    first = context.iloc[0]
    expected = fx.loc[
        fx["date"].eq(pd.Timestamp("2022-01-02")), "cny_per_usd"
    ].iloc[0]
    assert first["cny_per_usd"] == expected


def test_portfolio_uses_fractional_units_dual_costs_and_terminal_open_exit() -> None:
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
    assert audit["entry_transaction_count"] == 2
    assert audit["exit_transaction_count"] == 2
    assert audit["terminal_position_count"] == 0
    assert audit["maximum_position_count"] == 2
    assert audit["maximum_gross_exposure"] <= 1.0
    increment = contract["portfolio"]["quantity_increment"]
    scaled = trades["quantity"] / increment
    assert np.allclose(scaled, np.round(scaled), atol=1e-5)
    assert (trades.loc[trades["side"].eq("SELL"), "trade_date"] == context["date"].iloc[-1]).all()
    assert daily.iloc[-1]["position_count"] == 0
    assert daily.iloc[-1]["base_nav_cny"] > daily.iloc[-1]["stress_nav_cny"]


def test_terminal_missing_volume_fails_instead_of_using_close() -> None:
    contract = load_contract(CONFIG)
    targets, schedule, features, panel, context = _portfolio_inputs(
        block_terminal_btc=True
    )
    with pytest.raises(DataContractError, match="评价期末仍有无法退出"):
        run_portfolio_backtest(
            targets,
            schedule,
            features,
            panel,
            context,
            contract,
            start=context["date"].iloc[0],
            end=context["date"].iloc[-1],
        )


def test_freeze_scope_tracks_implementation_and_excludes_sealed_inputs() -> None:
    assert "research/digital_asset_spot_volatility_scaled_trend_v1.py" in TRACKED_FILES
    assert "scripts/run_digital_asset_spot_volatility_scaled_trend_v1.py" in TRACKED_FILES
    assert all("sealed" not in path.lower() for path in VISIBLE_SOURCE_DEPENDENCIES)
    assert all("replication_panel" not in path for path in VISIBLE_SOURCE_DEPENDENCIES)
