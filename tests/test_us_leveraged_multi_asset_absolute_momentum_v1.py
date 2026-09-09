"""美国杠杆多资产绝对动量V1的纯合成数据测试。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import DataContractError
from research.us_leveraged_multi_asset_absolute_momentum_v1 import (
    CONFIG,
    build_monthly_selections,
    fx_spread_rate,
    load_contract,
    prepare_daily_context,
    run_portfolio_backtest,
    security_cost_rate,
)
from scripts.freeze_us_leveraged_multi_asset_absolute_momentum_v1 import (
    TRACKED_FILES,
    VISIBLE_SOURCE_DEPENDENCIES,
)


TICKERS = ["UPRO", "TQQQ", "TMF", "UGL"]


def _signal_panel() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    dates = pd.bdate_range("2018-01-02", periods=430)
    slopes = {"UPRO": 0.0020, "TQQQ": 0.0015, "TMF": -0.0005, "UGL": 0.0010}
    parts: list[pd.DataFrame] = []
    for ticker in TICKERS:
        index = np.arange(len(dates), dtype=float)
        tri = 100.0 * np.exp(slopes[ticker] * index)
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "raw_open": 100.0,
                    "raw_close": 100.0,
                    "raw_volume": 2_000_000.0,
                    "raw_dollar_turnover_usd": 200_000_000.0,
                    "signal_total_return_index": tri,
                }
            )
        )
    return pd.concat(parts, ignore_index=True), dates


def _portfolio_inputs(
    *,
    block_terminal_upro: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=5)
    signal_date = pd.Timestamp("2019-12-31")
    selections = pd.DataFrame(
        [
            {
                "signal_date": signal_date,
                "execution_date": dates[0],
                "ticker": "UPRO",
                "selection_rank": 1,
                "momentum_252": 0.50,
                "realized_volatility_60": 0.30,
                "prior_median_dollar_turnover_20": 100_000_000.0,
            },
            {
                "signal_date": signal_date,
                "execution_date": dates[0],
                "ticker": "TQQQ",
                "selection_rank": 2,
                "momentum_252": 0.40,
                "realized_volatility_60": 0.35,
                "prior_median_dollar_turnover_20": 100_000_000.0,
            },
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
                "ticker": ticker,
                "observed_bar_count": 500,
                "momentum_252": 0.5 if ticker == "UPRO" else 0.4,
                "trend_sma_200": 80.0,
                "realized_volatility_60": 0.3,
                "prior_median_dollar_turnover_20": 100_000_000.0,
                "eligible": ticker in {"UPRO", "TQQQ"},
            }
            for ticker in TICKERS
        ]
    )
    parts: list[pd.DataFrame] = []
    for ticker in TICKERS:
        raw_open = np.full(len(dates), 100.0)
        raw_close = np.full(len(dates), 100.0)
        split = np.ones(len(dates))
        distribution = np.zeros(len(dates))
        if ticker == "UPRO":
            raw_open[1:] = 50.0
            raw_close[1:] = 50.0
            split[1] = 2.0
            distribution[2] = 1.0
        volume = np.full(len(dates), 2_000_000.0)
        if ticker == "UPRO" and block_terminal_upro:
            volume[-1] = 0.0
        parts.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "raw_open": raw_open,
                    "raw_high": raw_open,
                    "raw_low": raw_open,
                    "raw_close": raw_close,
                    "raw_volume": volume,
                    "raw_dollar_turnover_usd": raw_close * volume,
                    "qfq_factor": 1.0,
                    "adjust": 0.0,
                    "split_ratio_at_open": split,
                    "cash_distribution_per_post_event_share_usd": distribution,
                    "signal_total_return_index": 100.0,
                }
            )
        )
    panel = pd.concat(parts, ignore_index=True)
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
    return selections, schedule, features, panel, context


def test_contract_keeps_500k_40pct_high_sharpe_and_research_boundaries() -> None:
    contract = load_contract(CONFIG)
    assert contract["account"] == {
        "initial_capital_cny": 500000.0,
        "user_transaction_fee_rate_per_leg": 0.0001,
        "cash_annual_rate": 0.015,
    }
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["universe"]["fixed_tickers"] == TICKERS
    assert contract["portfolio"]["maximum_positions"] == 2
    assert not any(contract["safety"].values())


def test_costs_include_user_fee_and_conservative_friction() -> None:
    contract = load_contract(CONFIG)
    assert security_cost_rate(contract, "base") == pytest.approx(0.0017)
    assert security_cost_rate(contract, "stress") == pytest.approx(0.0047)
    assert fx_spread_rate(contract, "base") == pytest.approx(0.0010)
    assert fx_spread_rate(contract, "stress") == pytest.approx(0.0030)


def test_monthly_signal_ranks_positive_trends_and_executes_next_common_day() -> None:
    contract = load_contract(CONFIG)
    panel, dates = _signal_panel()
    selections, schedule, features, audit = build_monthly_selections(
        panel,
        contract,
        start=dates[300],
        end=dates[-1],
    )
    assert not selections.empty
    assert set(selections["ticker"]) == {"UPRO", "TQQQ"}
    first = selections.loc[selections["execution_date"].eq(selections["execution_date"].min())]
    assert first.sort_values("selection_rank")["ticker"].tolist() == ["UPRO", "TQQQ"]
    common = pd.Series(dates)
    for row in schedule.itertuples(index=False):
        position = int(common[common.eq(row.signal_date)].index[0])
        assert row.execution_date == common.iloc[position + 1]
    assert audit["future_open_or_strategy_return_read"] is False
    assert features["signal_date"].nunique() == len(schedule)


def test_future_open_perturbation_does_not_change_frozen_selection() -> None:
    contract = load_contract(CONFIG)
    panel, dates = _signal_panel()
    before, _, _, _ = build_monthly_selections(
        panel,
        contract,
        start=dates[300],
        end=dates[-1],
    )
    panel.loc[panel["date"].gt(dates[350]), "raw_open"] = 9999.0
    after, _, _, _ = build_monthly_selections(
        panel,
        contract,
        start=dates[300],
        end=dates[-1],
    )
    pd.testing.assert_frame_equal(before, after)


def test_daily_context_uses_fx_strictly_before_us_date() -> None:
    contract = load_contract(CONFIG)
    dates = pd.bdate_range("2019-12-27", "2020-01-10")
    panel = pd.concat(
        [pd.DataFrame({"ticker": ticker, "date": dates}) for ticker in TICKERS],
        ignore_index=True,
    )
    fx_dates = pd.date_range("2019-12-20", "2020-01-10", freq="D")
    fx = pd.DataFrame({"date": fx_dates, "cny_per_usd": np.arange(len(fx_dates)) + 6.0})
    benchmark = pd.DataFrame(
        {"date": dates, "close": np.arange(len(dates), dtype=float) + 1000.0}
    )
    context = prepare_daily_context(
        panel,
        fx,
        benchmark,
        contract,
        start=pd.Timestamp("2020-01-02"),
        end=pd.Timestamp("2020-01-10"),
    )
    assert (context["fx_source_date"] < context["date"]).all()
    first = context.iloc[0]
    expected = fx.loc[fx["date"].eq(pd.Timestamp("2020-01-01")), "cny_per_usd"].iloc[0]
    assert first["cny_per_usd"] == expected


def test_portfolio_uses_integer_shares_split_distribution_and_terminal_open_exit() -> None:
    contract = load_contract(CONFIG)
    selections, schedule, features, panel, context = _portfolio_inputs()
    daily, trades, audit = run_portfolio_backtest(
        selections,
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
    assert audit["split_event_count_for_held_positions"] == 1
    assert audit["distribution_event_count_for_held_positions"] == 1
    assert audit["terminal_position_count"] == 0
    assert audit["maximum_position_count"] == 2
    assert audit["maximum_gross_exposure"] <= 1.0
    assert trades["shares"].astype(int).eq(trades["shares"]).all()
    upro_buy = trades.loc[(trades["ticker"] == "UPRO") & (trades["side"] == "BUY")].iloc[0]
    upro_sell = trades.loc[(trades["ticker"] == "UPRO") & (trades["side"] == "SELL")].iloc[0]
    assert upro_sell["shares"] == 2 * upro_buy["shares"]
    assert upro_sell["trade_date"] == context["date"].iloc[-1]
    assert daily.iloc[-1]["position_count"] == 0
    assert daily.iloc[-1]["base_nav_cny"] > daily.iloc[-1]["stress_nav_cny"]


def test_terminal_missing_volume_fails_instead_of_using_close() -> None:
    contract = load_contract(CONFIG)
    selections, schedule, features, panel, context = _portfolio_inputs(
        block_terminal_upro=True
    )
    with pytest.raises(DataContractError, match="评价期末仍有无法退出"):
        run_portfolio_backtest(
            selections,
            schedule,
            features,
            panel,
            context,
            contract,
            start=context["date"].iloc[0],
            end=context["date"].iloc[-1],
        )


def test_freeze_scope_tracks_implementation_and_excludes_sealed_inputs() -> None:
    assert "research/us_leveraged_multi_asset_absolute_momentum_v1.py" in TRACKED_FILES
    assert "scripts/run_us_leveraged_multi_asset_absolute_momentum_v1.py" in TRACKED_FILES
    assert all("sealed" not in path.lower() for path in VISIBLE_SOURCE_DEPENDENCIES)
    assert all("daily_panel.parquet" not in path for path in VISIBLE_SOURCE_DEPENDENCIES)
