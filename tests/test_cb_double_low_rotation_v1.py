"""可转债传统双低周频轮动V1合成测试；不读取真实可见收益。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.broad_liquid_etf_liquidity_shock_reversal_v1 import DataContractError
from research.cb_double_low_rotation_v1 import (
    CONFIG,
    build_weekly_selections,
    cb_cost_rate,
    load_contract,
    run_portfolio_backtest,
)
from scripts.freeze_cb_double_low_rotation_v1 import (
    TRACKED_FILES,
    VISIBLE_SOURCE_DEPENDENCIES,
)


ROOT = Path(__file__).resolve().parents[1]


def _synthetic_selection_inputs() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DatetimeIndex
]:
    dates = pd.bdate_range("2020-01-02", periods=100)
    panel_parts: list[pd.DataFrame] = []
    master_rows: list[dict[str, object]] = []
    for index in range(12):
        code = f"110{index + 1:03d}"
        close = 100.0 + index
        premium = 5.0 + index
        turnover = 60_000_000.0 + index * 1_000_000.0
        if index == 0:
            close = 100.0
            premium = 5.0
            turnover = 60_000_000.0
        elif index == 1:
            close = 101.0
            premium = 4.0
            turnover = 80_000_000.0
        panel_parts.append(
            pd.DataFrame(
                {
                    "bond_code": code,
                    "date": dates,
                    "close": close,
                    "volume": 500_000,
                    "turnover_notional_proxy_cny": turnover,
                    "conversion_premium_rate_pct": premium,
                    "value_table_close": close,
                    "close_cross_source_absolute_difference": 0.0,
                }
            )
        )
        master_rows.append(
            {
                "bond_code": code,
                "listing_date": pd.Timestamp("2015-01-01"),
                "exchange": "SSE",
            }
        )
    return (
        pd.concat(panel_parts, ignore_index=True),
        pd.DataFrame(master_rows),
        pd.DataFrame({"date": dates}),
        dates,
    )


def _synthetic_portfolio_inputs(
    *,
    blocked_exit: bool = False,
    blocked_new_entry: bool = False,
    block_terminal_exit: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    dates = pd.bdate_range("2021-01-04", periods=16)
    start = pd.Timestamp("2021-01-05")
    end = pd.Timestamp("2021-01-25")
    first_signal = pd.Timestamp("2021-01-08")
    first_execution = pd.Timestamp("2021-01-11")
    second_signal = pd.Timestamp("2021-01-15")
    second_execution = pd.Timestamp("2021-01-18")
    terminal_signal = pd.Timestamp("2021-01-22")
    terminal_execution = pd.Timestamp("2021-01-25")
    selections = pd.DataFrame(
        [
            {
                "signal_date": first_signal,
                "execution_date": first_execution,
                "bond_code": "110001",
                "selection_rank": 1,
                "double_low_score": 105.0,
                "prior_median_turnover_20": 60_000_000.0,
            },
            {
                "signal_date": first_signal,
                "execution_date": first_execution,
                "bond_code": "110002",
                "selection_rank": 2,
                "double_low_score": 106.0,
                "prior_median_turnover_20": 60_000_000.0,
            },
            {
                "signal_date": second_signal,
                "execution_date": second_execution,
                "bond_code": "110002",
                "selection_rank": 1,
                "double_low_score": 104.0,
                "prior_median_turnover_20": 60_000_000.0,
            },
            {
                "signal_date": second_signal,
                "execution_date": second_execution,
                "bond_code": "110003",
                "selection_rank": 2,
                "double_low_score": 107.0,
                "prior_median_turnover_20": 60_000_000.0,
            },
            {
                "signal_date": terminal_signal,
                "execution_date": terminal_execution,
                "bond_code": "110002",
                "selection_rank": 1,
                "double_low_score": 103.0,
                "prior_median_turnover_20": 60_000_000.0,
            },
            {
                "signal_date": terminal_signal,
                "execution_date": terminal_execution,
                "bond_code": "110003",
                "selection_rank": 2,
                "double_low_score": 104.0,
                "prior_median_turnover_20": 60_000_000.0,
            },
        ]
    )
    market_parts: list[pd.DataFrame] = []
    for code in ("110001", "110002", "110003"):
        market_parts.append(
            pd.DataFrame(
                {
                    "bond_code": code,
                    "date": dates,
                    "open": 100.0,
                    "close": 100.0,
                    "volume": 500_000,
                }
            )
        )
    market = pd.concat(market_parts, ignore_index=True)
    if blocked_exit:
        market.loc[
            market["bond_code"].eq("110001")
            & market["date"].eq(second_execution),
            "open",
        ] = 80.0
    if blocked_new_entry:
        market.loc[
            market["bond_code"].eq("110003")
            & market["date"].eq(second_execution),
            "open",
        ] = 120.0
    if block_terminal_exit:
        market.loc[
            market["bond_code"].isin(["110002", "110003"])
            & market["date"].eq(terminal_execution),
            "volume",
        ] = 0
    benchmark = pd.DataFrame(
        {"date": dates, "close": np.linspace(1000.0, 1015.0, len(dates))}
    )
    return selections, market, benchmark, start, end


def test_contract_keeps_target_cost_and_safety_boundaries() -> None:
    contract = load_contract(CONFIG)
    assert contract["account"] == {
        "initial_capital_cny": 500000.0,
        "user_transaction_fee_rate_per_leg": 0.0001,
        "cash_annual_rate": 0.015,
    }
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["portfolio"]["maximum_positions"] == 10
    assert contract["portfolio"]["board_lot_bonds"] == 10
    assert contract["signal"]["blocked_exit_reselection_policy"] == (
        "PENDING_EXIT_REMAINS_MANDATORY"
    )
    assert not any(contract["safety"].values())


def test_weekly_signal_uses_last_trading_day_and_next_open_mapping() -> None:
    contract = load_contract(CONFIG)
    panel, master, benchmark, dates = _synthetic_selection_inputs()
    selections, audit = build_weekly_selections(
        panel,
        master,
        benchmark,
        contract,
        start=dates[65],
        end=dates[-1],
    )
    assert not selections.empty
    mapping = selections[["signal_date", "execution_date"]].drop_duplicates()
    calendar = pd.Series(dates)
    for row in mapping.itertuples(index=False):
        signal_index = int(calendar[calendar.eq(row.signal_date)].index[0])
        assert row.execution_date == calendar.iloc[signal_index + 1]
        week = calendar[calendar.dt.isocalendar().week.eq(row.signal_date.isocalendar().week)]
        assert row.signal_date == week.max()
    assert audit["future_open_or_return_read"] is False
    assert audit["sealed_replication_read"] is False


def test_double_low_ranking_uses_liquidity_then_code_for_ties() -> None:
    contract = load_contract(CONFIG)
    panel, master, benchmark, dates = _synthetic_selection_inputs()
    selections, _ = build_weekly_selections(
        panel,
        master,
        benchmark,
        contract,
        start=dates[65],
        end=dates[-1],
    )
    first = selections.loc[
        selections["signal_date"].eq(selections["signal_date"].min())
    ]
    assert len(first) == 10
    assert first.iloc[0]["bond_code"] == "110002"
    assert first.iloc[1]["bond_code"] == "110001"
    assert first["double_low_score"].is_monotonic_increasing


def test_future_open_perturbation_does_not_change_selection() -> None:
    contract = load_contract(CONFIG)
    panel, master, benchmark, dates = _synthetic_selection_inputs()
    with_open = panel.copy()
    with_open["open"] = with_open["close"]
    before, audit_before = build_weekly_selections(
        with_open,
        master,
        benchmark,
        contract,
        start=dates[65],
        end=dates[-1],
    )
    with_open.loc[with_open["date"].gt(dates[70]), "open"] = 9999.0
    after, audit_after = build_weekly_selections(
        with_open,
        master,
        benchmark,
        contract,
        start=dates[65],
        end=dates[-1],
    )
    pd.testing.assert_frame_equal(before, after)
    assert audit_before["future_price_or_return_columns_used"] == []
    assert audit_after["future_price_or_return_columns_used"] == []


def test_cb_cost_rates_include_user_fee_handling_slippage_and_impact() -> None:
    contract = load_contract(CONFIG)
    assert cb_cost_rate(contract, scenario="base", side="BUY") == pytest.approx(
        0.00164
    )
    assert cb_cost_rate(contract, scenario="base", side="SELL") == pytest.approx(
        0.00164
    )
    assert cb_cost_rate(contract, scenario="stress", side="BUY") == pytest.approx(
        0.00464
    )
    assert cb_cost_rate(contract, scenario="stress", side="SELL") == pytest.approx(
        0.00464
    )


def test_portfolio_rebalances_retains_positions_and_liquidates_at_terminal_open() -> None:
    contract = load_contract(CONFIG)
    selections, market, benchmark, start, end = _synthetic_portfolio_inputs()
    daily, trades, audit = run_portfolio_backtest(
        selections,
        market,
        benchmark,
        contract,
        start=start,
        end=end,
    )
    assert audit["entry_transaction_count"] == 3
    assert audit["exit_transaction_count"] == 3
    assert audit["retained_position_observation_count"] >= 1
    assert audit["blocked_exit_attempt_count"] == 0
    assert trades.loc[trades["side"].eq("BUY"), "bond_code"].tolist() == [
        "110001",
        "110002",
        "110003",
    ]
    assert trades.loc[
        trades["side"].eq("SELL") & trades["bond_code"].eq("110001"),
        "trade_date",
    ].iloc[0] == pd.Timestamp("2021-01-18")
    assert not (
        trades["side"].eq("BUY") & trades["bond_code"].eq("110002")
    ).loc[trades["trade_date"].eq(pd.Timestamp("2021-01-18"))].any()
    assert trades["units"].mod(10).eq(0).all()
    assert trades.loc[trades["side"].eq("BUY"), "gross_notional_cny"].le(
        60_000.0
    ).all()
    assert audit["maximum_position_count"] == 2
    assert audit["maximum_gross_exposure"] <= 1.0
    assert audit["maximum_net_exposure"] <= 1.0
    assert daily.iloc[-1]["position_count"] == 0
    assert daily[["quality_complete", "capacity_pass"]].all().all()


def test_blocked_exit_is_carried_to_first_eligible_open() -> None:
    contract = load_contract(CONFIG)
    selections, market, benchmark, start, end = _synthetic_portfolio_inputs(
        blocked_exit=True
    )
    _, trades, audit = run_portfolio_backtest(
        selections,
        market,
        benchmark,
        contract,
        start=start,
        end=end,
    )
    sell = trades.loc[
        trades["side"].eq("SELL") & trades["bond_code"].eq("110001")
    ].iloc[0]
    assert audit["blocked_exit_attempt_count"] == 1
    assert sell["trade_date"] == pd.Timestamp("2021-01-19")
    assert sell["delayed_exit_days"] == 1


def test_entry_at_or_above_positive_19pct_gap_is_skipped() -> None:
    contract = load_contract(CONFIG)
    selections, market, benchmark, start, end = _synthetic_portfolio_inputs(
        blocked_new_entry=True
    )
    _, trades, audit = run_portfolio_backtest(
        selections,
        market,
        benchmark,
        contract,
        start=start,
        end=end,
    )
    assert audit["upper_gap_entry_skip_count"] == 1
    assert not (
        trades["side"].eq("BUY") & trades["bond_code"].eq("110003")
    ).any()


def test_unresolved_terminal_exit_fails_instead_of_fictitious_redemption() -> None:
    contract = load_contract(CONFIG)
    selections, market, benchmark, start, end = _synthetic_portfolio_inputs(
        block_terminal_exit=True
    )
    with pytest.raises(DataContractError, match="期末仍有无法退出"):
        run_portfolio_backtest(
            selections,
            market,
            benchmark,
            contract,
            start=start,
            end=end,
        )


def test_freeze_scope_tracks_code_but_excludes_sealed_inputs() -> None:
    assert "research/cb_double_low_rotation_v1.py" in TRACKED_FILES
    assert "scripts/run_cb_double_low_rotation_v1.py" in TRACKED_FILES
    assert all("sealed" not in path.lower() for path in VISIBLE_SOURCE_DEPENDENCIES)
    assert all("2023_2026" not in path for path in VISIBLE_SOURCE_DEPENDENCIES)
    assert all("bucket1_time_holdout" not in path for path in VISIBLE_SOURCE_DEPENDENCIES)
    assert (ROOT / "config" / "cb_double_low_rotation_v1.yaml").is_file()
