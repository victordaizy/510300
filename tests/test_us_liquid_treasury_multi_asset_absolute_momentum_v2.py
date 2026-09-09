"""美国高流动性美债多资产绝对动量V2的合成测试。"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
import yaml

from research.us_liquid_treasury_multi_asset_absolute_momentum_v2 import (
    ALLOWED_CHANGED_PATHS,
    CONFIG,
    build_monthly_selections,
    changed_paths,
    fx_spread_rate,
    load_contract,
    run_portfolio_backtest,
    security_cost_rate,
)
from scripts.download_us_liquid_treasury_multi_asset_history_v2 import (
    normalize_tlt_yahoo_chart,
)
from scripts.freeze_us_liquid_treasury_multi_asset_absolute_momentum_v2 import (
    TRACKED_FILES,
    VISIBLE_SOURCE_DEPENDENCIES,
)


TICKERS = ["UPRO", "TQQQ", "TLT", "UGL"]


def _epoch(date: str) -> int:
    return int(
        datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp()
    )


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
                    "raw_open": 100.0,
                    "raw_close": 100.0,
                    "raw_volume": 2_000_000.0,
                    "raw_dollar_turnover_usd": 200_000_000.0,
                    "signal_total_return_index": tri,
                }
            )
        )
    return pd.concat(parts, ignore_index=True), dates


def _portfolio_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-02", periods=5)
    signal_date = pd.Timestamp("2019-12-31")
    selections = pd.DataFrame(
        [
            {
                "signal_date": signal_date,
                "execution_date": dates[0],
                "ticker": ticker,
                "selection_rank": rank,
                "momentum_252": 0.50 - rank * 0.05,
                "realized_volatility_60": 0.30,
                "prior_median_dollar_turnover_20": 100_000_000.0,
            }
            for rank, ticker in enumerate(["UPRO", "TLT"], start=1)
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
                "momentum_252": 0.5 if ticker in {"UPRO", "TLT"} else -0.1,
                "trend_sma_200": 80.0,
                "realized_volatility_60": 0.3,
                "prior_median_dollar_turnover_20": 100_000_000.0,
                "eligible": ticker in {"UPRO", "TLT"},
            }
            for ticker in TICKERS
        ]
    )
    panel = pd.concat(
        [
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "raw_open": 100.0,
                    "raw_high": 101.0,
                    "raw_low": 99.0,
                    "raw_close": 100.0,
                    "raw_volume": 2_000_000.0,
                    "raw_dollar_turnover_usd": 200_000_000.0,
                    "qfq_factor": 1.0,
                    "adjust": 0.0,
                    "split_ratio_at_open": 1.0,
                    "cash_distribution_per_post_event_share_usd": 0.0,
                    "signal_total_return_index": 100.0,
                }
            )
            for ticker in TICKERS
        ],
        ignore_index=True,
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
    return selections, schedule, features, panel, context


def test_contract_changes_only_authorized_reconstitution_paths() -> None:
    contract = load_contract(CONFIG)
    base = yaml.safe_load(
        open(contract_path := "config/us_leveraged_multi_asset_absolute_momentum_v1.yaml", encoding="utf-8")
    )
    assert set(changed_paths(contract)) == ALLOWED_CHANGED_PATHS
    assert contract["universe"]["fixed_tickers"] == TICKERS
    assert "TMF" not in contract["universe"]["products"]
    assert contract["universe"]["products"]["TLT"]["stated_daily_multiple"] == 1.0
    for unchanged in (
        "historical_partition",
        "signal",
        "portfolio",
        "costs",
        "visible_gates",
    ):
        assert contract[unchanged] == base[unchanged], (contract_path, unchanged)


def test_contract_keeps_500k_40pct_high_sharpe_costs_and_closed_boundaries() -> None:
    contract = load_contract(CONFIG)
    assert contract["account"]["initial_capital_cny"] == 500000.0
    assert contract["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert security_cost_rate(contract, "base") == pytest.approx(0.0017)
    assert security_cost_rate(contract, "stress") == pytest.approx(0.0047)
    assert fx_spread_rate(contract, "base") == pytest.approx(0.0010)
    assert fx_spread_rate(contract, "stress") == pytest.approx(0.0030)
    assert not any(contract["safety"].values())


def test_yahoo_chart_parser_preserves_raw_prices_splits_dividends_and_total_return() -> None:
    timestamps = [_epoch("2020-01-02"), _epoch("2020-01-03"), _epoch("2020-01-06")]
    payload = {
        "chart": {
            "error": None,
            "result": [
                {
                    "timestamp": timestamps,
                    "indicators": {
                        "quote": [
                            {
                                "open": [99.0, 50.0, 50.5],
                                "high": [101.0, 51.0, 52.0],
                                "low": [98.0, 49.0, 50.0],
                                "close": [100.0, 50.0, 51.0],
                                "volume": [1_000_000, 2_000_000, 2_000_000],
                            }
                        ]
                    },
                    "events": {
                        "splits": {
                            "s": {
                                "date": timestamps[1],
                                "numerator": 2.0,
                                "denominator": 1.0,
                            }
                        },
                        "dividends": {
                            "d": {"date": timestamps[2], "amount": 1.0}
                        },
                    },
                }
            ],
        }
    }
    panel = normalize_tlt_yahoo_chart(
        payload,
        start=pd.Timestamp("2020-01-02"),
        end=pd.Timestamp("2020-01-06"),
    )
    assert panel["qfq_factor"].tolist() == [1.0, 2.0, 2.0]
    assert panel["split_ratio_at_open"].tolist() == [1.0, 2.0, 1.0]
    assert panel["cash_distribution_per_post_event_share_usd"].tolist() == [0.0, 0.0, 1.0]
    assert panel.iloc[-1]["signal_total_return_index"] == pytest.approx(104.0)
    assert panel.iloc[-1]["raw_dollar_turnover_usd"] == 102_000_000.0


def test_monthly_signal_still_selects_top_two_without_reading_future_open() -> None:
    contract = load_contract(CONFIG)
    panel, dates = _signal_panel()
    before, schedule, features, audit = build_monthly_selections(
        panel,
        contract,
        start=dates[300],
        end=dates[-1],
    )
    assert not before.empty
    first = before.loc[before["execution_date"].eq(before["execution_date"].min())]
    assert first.sort_values("selection_rank")["ticker"].tolist() == ["UPRO", "TQQQ"]
    panel.loc[panel["date"].gt(dates[350]), "raw_open"] = 9999.0
    after, _, _, _ = build_monthly_selections(
        panel,
        contract,
        start=dates[300],
        end=dates[-1],
    )
    pd.testing.assert_frame_equal(before, after)
    assert audit["future_open_or_strategy_return_read"] is False
    assert features["signal_date"].nunique() == len(schedule)


def test_portfolio_can_hold_tlt_with_integer_shares_capacity_and_terminal_exit() -> None:
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
    assert set(trades["ticker"]) == {"UPRO", "TLT"}
    assert audit["entry_transaction_count"] == 2
    assert audit["exit_transaction_count"] == 2
    assert audit["terminal_position_count"] == 0
    assert audit["maximum_observed_order_capacity_fraction"] <= 0.0005
    assert trades["shares"].astype(int).eq(trades["shares"]).all()
    assert daily.iloc[-1]["base_nav_cny"] > daily.iloc[-1]["stress_nav_cny"]


def test_freeze_scope_tracks_implementation_and_excludes_sealed_inputs() -> None:
    assert "research/us_liquid_treasury_multi_asset_absolute_momentum_v2.py" in TRACKED_FILES
    assert "scripts/run_us_liquid_treasury_multi_asset_absolute_momentum_v2.py" in TRACKED_FILES
    assert all("sealed" not in path.lower() for path in VISIBLE_SOURCE_DEPENDENCIES)
    assert all("daily_panel.parquet" not in path for path in VISIBLE_SOURCE_DEPENDENCIES)
