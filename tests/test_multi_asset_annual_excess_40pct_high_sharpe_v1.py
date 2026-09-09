from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.multi_asset_annual_excess_40pct_high_sharpe_v1 import (
    evaluate_forward_returns,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v1.yaml"


def contract() -> dict:
    return load_contract(CONFIG)


def make_returns(
    days: int,
    *,
    strategy_cagr: float,
    benchmark_cagr: float = 0.0,
    daily_noise: float = 0.003,
) -> pd.DataFrame:
    dates = pd.bdate_range("2026-08-27", periods=days)
    benchmark_daily = (1.0 + benchmark_cagr) ** (1.0 / 242.0) - 1.0
    strategy_daily = (1.0 + strategy_cagr) ** (1.0 / 242.0) - 1.0
    noise = np.resize(np.array([daily_noise, -daily_noise], dtype=float), days)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "strategy_base_net_return": strategy_daily + noise,
            "strategy_stress_net_return": strategy_daily * 0.94 + noise,
            "benchmark_total_return": np.full(days, benchmark_daily),
            "gross_exposure": np.full(days, 1.0),
            "net_exposure": np.full(days, 0.5),
            "quality_complete": True,
            "capacity_pass": True,
            "forward_observed": True,
        }
    )


def test_contract_keeps_500k_1bp_40pct_and_sharpe_1_5() -> None:
    value = contract()
    assert value["scope"]["initial_capital_cny"] == 500_000.0
    assert value["costs"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert value["objective"]["minimum_annualized_excess"] == 0.40
    assert value["objective"]["minimum_strategy_net_sharpe"] == 1.50
    assert value["scope"]["maximum_active_research_lanes"] == 2
    assert value["scope"]["risk_envelope"]["maximum_gross_exposure"] == 2.0
    assert value["safety"]["order_generation"] is False
    assert value["safety"]["live_trading_authorized"] is False


def test_returns_hidden_before_120_day_gate() -> None:
    result = evaluate_forward_returns(
        make_returns(119, strategy_cagr=0.70),
        contract(),
        bootstrap_repetitions_override=40,
    )
    assert result["status"] == "PENDING_FIRST_UNSEEN_GATE"
    assert result["metrics"] is None
    assert result["goal_achieved"] is False


def test_120_days_are_only_provisional() -> None:
    result = evaluate_forward_returns(
        make_returns(120, strategy_cagr=0.70),
        contract(),
        bootstrap_repetitions_override=40,
    )
    assert result["status"] == "PROVISIONAL_FIRST_UNSEEN_NOT_FULL_YEAR"
    assert result["metrics"]["stress_annualized_excess"] > 0.40
    assert result["metrics"]["stress_strategy_net_sharpe"] > 1.50
    assert result["goal_achieved"] is False


def test_one_year_cannot_verify_target() -> None:
    result = evaluate_forward_returns(
        make_returns(242, strategy_cagr=0.70),
        contract(),
        bootstrap_repetitions_override=60,
    )
    assert result["status"] == "ONE_YEAR_40PCT_HIGH_SHARPE_CANDIDATE_NOT_VERIFIED"
    assert result["gates"]["stress_annualized_excess_at_least_40pct"] is True
    assert result["gates"]["stress_strategy_sharpe_at_least_1_5"] is True
    assert result["goal_achieved"] is False


def test_three_strong_years_can_verify_target() -> None:
    result = evaluate_forward_returns(
        make_returns(726, strategy_cagr=0.70),
        contract(),
        bootstrap_repetitions_override=80,
    )
    assert result["status"] == "VERIFIED_40PCT_EXCESS_HIGH_SHARPE"
    assert all(result["gates"].values())
    assert result["goal_achieved"] is True


def test_high_return_with_low_sharpe_is_rejected() -> None:
    frame = make_returns(726, strategy_cagr=0.70, daily_noise=0.03)
    result = evaluate_forward_returns(
        frame,
        contract(),
        bootstrap_repetitions_override=60,
    )
    assert result["metrics"]["stress_annualized_excess"] > 0.40
    assert result["metrics"]["stress_strategy_net_sharpe"] < 1.50
    assert result["status"] == "REJECTED_40PCT_EXCESS_HIGH_SHARPE"
    assert result["goal_achieved"] is False


def test_incomplete_or_capacity_failed_rows_do_not_advance_maturity() -> None:
    frame = make_returns(150, strategy_cagr=0.70)
    frame.loc[:20, "quality_complete"] = False
    frame.loc[21:40, "capacity_pass"] = False
    result = evaluate_forward_returns(
        frame,
        contract(),
        bootstrap_repetitions_override=40,
    )
    assert result["eligible_forward_day_count"] == 109
    assert result["status"] == "PENDING_FIRST_UNSEEN_GATE"


def test_exposure_above_frozen_cap_is_rejected() -> None:
    frame = make_returns(120, strategy_cagr=0.70)
    frame.loc[0, "gross_exposure"] = 2.01
    with pytest.raises(ValueError, match="gross_exposure"):
        evaluate_forward_returns(frame, contract(), bootstrap_repetitions_override=40)


def test_pre_effective_rows_are_rejected() -> None:
    frame = make_returns(120, strategy_cagr=0.70)
    frame.loc[0, "trade_date"] = pd.Timestamp("2026-08-26")
    with pytest.raises(ValueError, match="合同生效日前"):
        evaluate_forward_returns(frame, contract(), bootstrap_repetitions_override=40)
