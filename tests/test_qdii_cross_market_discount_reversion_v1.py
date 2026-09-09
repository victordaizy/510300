from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from research.qdii_cross_market_discount_reversion_v1 import (
    _rolling_mad,
    build_discount_signals,
    calendar_successor_maps,
    etf_cost_rate,
    load_contract,
    load_research_inputs,
    run_portfolio_backtest,
    validate_contract,
)


def test_frozen_contract_keeps_user_target_and_high_sharpe() -> None:
    contract = load_contract()
    assert contract["account"]["initial_capital_cny"] == 500_000.0
    assert contract["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["information_timing"]["same_open_price_used_for_signal_and_execution"] is False


def test_contract_rejects_relaxed_excess_or_same_open_lookahead() -> None:
    contract = load_contract()
    relaxed = deepcopy(contract)
    relaxed["visible_gates"]["minimum_annualized_net_excess"] = 0.20
    try:
        validate_contract(relaxed)
    except ValueError as exc:
        assert "minimum_annualized_net_excess" in str(exc)
    else:
        raise AssertionError("降低40个百分点门槛必须失败")
    lookahead = deepcopy(contract)
    lookahead["information_timing"]["same_open_price_used_for_signal_and_execution"] = True
    try:
        validate_contract(lookahead)
    except ValueError as exc:
        assert "same_open_price_used_for_signal_and_execution" in str(exc)
    else:
        raise AssertionError("同一开盘信号和成交必须失败")


def test_cost_rates_include_user_fee_and_stress_is_higher() -> None:
    contract = load_contract()
    assert np.isclose(etf_cost_rate(contract, "base"), 0.00074)
    assert np.isclose(etf_cost_rate(contract, "stress"), 0.00194)
    assert etf_cost_rate(contract, "stress") > etf_cost_rate(contract, "base")


def test_rolling_mad_excludes_current_observation() -> None:
    values = pd.Series([0.0, 0.1, -0.1, 0.0, 99.0])
    mad = _rolling_mad(values, 4)
    expected = np.median(np.abs(np.array([0.0, 0.1, -0.1, 0.0]) - 0.0))
    assert np.isclose(mad.iloc[-1], expected)


def test_calendar_mapping_executes_next_open_and_exits_five_days_later() -> None:
    calendar = pd.date_range("2022-01-03", periods=10, freq="B")
    entries, exits = calendar_successor_maps(calendar, 5)
    assert entries[calendar[0]] == calendar[1]
    assert exits[calendar[0]] == calendar[6]


def test_real_input_signal_timing_and_future_perturbation_do_not_change_past_signals() -> None:
    contract = load_contract()
    panel, master, benchmark, fx, indices = load_research_inputs(contract)
    fixed_codes = {
        code
        for group in contract["tracked_groups"]
        for code in group["etf_codes"]
    }
    panel = panel.loc[panel["con_code"].isin(fixed_codes)].copy()
    start = pd.Timestamp(contract["historical_partition"]["visible_start"])
    end = pd.Timestamp(contract["historical_partition"]["visible_end"])
    baseline, _, _, audit = build_discount_signals(
        panel,
        master,
        benchmark,
        fx,
        indices,
        contract,
        start=start,
        end=end,
    )
    assert audit["foreign_date_strictly_before_signal"] is True
    assert audit["fx_date_not_after_signal"] is True
    assert audit["same_open_used_for_signal_and_execution"] is False
    assert audit["future_return_used_in_signal_construction"] is False
    cutoff = pd.Timestamp("2020-12-31")
    perturbed = panel.copy()
    future = perturbed["date"].gt(cutoff)
    for column in ("total_return_open", "total_return_close", "raw_open", "raw_close"):
        perturbed.loc[future, column] = perturbed.loc[future, column] * 7.0
    changed, _, _, _ = build_discount_signals(
        perturbed,
        master,
        benchmark,
        fx,
        indices,
        contract,
        start=start,
        end=end,
    )
    columns = ["signal_date", "con_code", "discount_zscore", "relative_log_deviation"]
    left = baseline.loc[baseline["signal_date"].le(cutoff), columns].reset_index(drop=True)
    right = changed.loc[changed["signal_date"].le(cutoff), columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_synthetic_portfolio_applies_two_sided_costs_without_leverage() -> None:
    contract = load_contract()
    dates = pd.date_range("2022-01-03", periods=7, freq="B")
    benchmark = pd.DataFrame(
        {"date": dates, "benchmark_open": 100.0, "benchmark_close": 100.0}
    )
    market = pd.DataFrame(
        {
            "date": dates,
            "con_code": "513500.SH",
            "total_return_open": 1.0,
            "total_return_close": 1.0,
            "raw_open": 1.0,
            "raw_close": 1.0,
            "prior_raw_close": 1.0,
            "amount": 200_000_000.0,
            "is_suspended": False,
        }
    )
    signals = pd.DataFrame(
        [
            {
                "signal_date": dates[0],
                "entry_date": dates[1],
                "scheduled_exit_date": dates[6],
                "con_code": "513500.SH",
                "group_id": "SP_500",
                "discount_zscore": -2.5,
                "relative_log_deviation": -0.02,
                "prior_median_amount_20": 200_000_000.0,
                "entry_gap": 0.0,
            }
        ]
    )
    returns = pd.DataFrame({"513500.SH": 0.0}, index=dates)
    daily, trades, audit = run_portfolio_backtest(
        signals,
        market,
        benchmark,
        returns,
        contract,
        start=dates[0],
        end=dates[-1],
    )
    assert len(trades) == 2
    assert audit["entry_transaction_count"] == 1
    assert audit["exit_transaction_count"] == 1
    assert audit["maximum_gross_exposure"] <= 1.0
    assert daily["stress_nav_cny"].iloc[-1] < daily["base_nav_cny"].iloc[-1]
    assert daily["base_nav_cny"].iloc[-1] < 500_000.0 * (1.015 ** (6 / 242))
