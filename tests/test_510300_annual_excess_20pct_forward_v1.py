from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.annual_excess_20pct_forward_v1 import (
    evaluate_forward_returns,
    load_contract,
    required_edge_budget,
    round_trip_cost_bps,
    summarize_iopv_gap_sample,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "510300_annual_excess_20pct_forward_v1.yaml"


def contract() -> dict:
    return load_contract(CONFIG)


def make_returns(days: int, *, base_cagr: float, stress_cagr: float, benchmark_cagr: float = 0.0) -> pd.DataFrame:
    dates = pd.bdate_range("2026-08-27", periods=days)
    return pd.DataFrame(
        {
            "trade_date": dates,
            "strategy_base_net_return": np.full(days, (1.0 + base_cagr) ** (1.0 / 242.0) - 1.0),
            "strategy_stress_net_return": np.full(days, (1.0 + stress_cagr) ** (1.0 / 242.0) - 1.0),
            "benchmark_total_return": np.full(days, (1.0 + benchmark_cagr) ** (1.0 / 242.0) - 1.0),
            "quality_complete": True,
            "forward_observed": True,
        }
    )


def test_contract_keeps_user_20pct_target_and_two_stream_cap() -> None:
    value = contract()
    assert value["objective"]["minimum_annualized_excess"] == 0.20
    assert value["objective"]["minimum_rolling_242d_excess_median"] == 0.20
    assert value["objective"]["base_cost_required"] is True
    assert value["objective"]["stress_cost_required"] is True
    assert value["scope"]["active_research_streams"] == [
        "PRIMARY_MARKET_PCF_IOPV",
        "INDUSTRY_EXPECTATION_GAP",
    ]
    assert value["scope"]["maximum_active_research_streams"] == 2
    assert value["scope"]["initial_capital_cny"] == 500_000.0
    assert value["costs"]["commission_rate_per_leg"] == 0.0001
    assert value["costs"]["minimum_commission_cny_per_leg"] == 0.0


def test_500k_account_cost_and_required_edge_budget_are_explicit() -> None:
    base = round_trip_cost_bps(
        250_000.0,
        commission_rate_per_leg=0.0001,
        minimum_commission_cny_per_leg=0.0,
        slippage_bps_per_leg=5.0,
    )
    stress = round_trip_cost_bps(
        250_000.0,
        commission_rate_per_leg=0.0001,
        minimum_commission_cny_per_leg=0.0,
        slippage_bps_per_leg=15.0,
    )
    assert base["total_cost_cny"] == pytest.approx(300.0)
    assert base["total_cost_bps"] == pytest.approx(12.0)
    assert stress["total_cost_bps"] == pytest.approx(32.0)

    budget = required_edge_budget(
        initial_capital_cny=500_000.0,
        annual_excess_target=0.20,
        event_notional_cny=250_000.0,
        events_per_year=242,
        round_trip_cost=base,
    )
    assert budget["required_net_edge_bps_per_event"] == pytest.approx(16.5289256198)
    assert budget["required_gross_edge_bps_per_event"] == pytest.approx(28.5289256198)


def test_iopv_summary_excludes_after_close_rows() -> None:
    frame = pd.DataFrame(
        {
            "exchange_timestamp": ["2026-08-27 10:00:00", "2026-08-27 16:00:00"],
            "trade_date": ["2026-08-27", "2026-08-27"],
            "premium_discount_bps": [-12.5, 200.0],
        }
    )
    summary = summarize_iopv_gap_sample(frame)
    assert summary["row_count"] == 1
    assert summary["absolute_maximum_bps"] == pytest.approx(12.5)
    assert summary["discovery_only"] is True


def test_returns_are_hidden_before_first_unseen_gate() -> None:
    result = evaluate_forward_returns(
        make_returns(79, base_cagr=0.60, stress_cagr=0.50),
        contract(),
        bootstrap_repetitions_override=40,
    )
    assert result["status"] == "PENDING_FIRST_UNSEEN_GATE"
    assert result["metrics"] is None
    assert result["goal_achieved"] is False


def test_eighty_days_only_produce_provisional_estimate() -> None:
    result = evaluate_forward_returns(
        make_returns(80, base_cagr=0.30, stress_cagr=0.25),
        contract(),
        bootstrap_repetitions_override=40,
    )
    assert result["status"] == "PROVISIONAL_FIRST_UNSEEN_NOT_REPLICATED"
    assert result["metrics"]["base_annualized_excess"] == pytest.approx(0.30)
    assert result["goal_achieved"] is False


def test_one_year_cannot_be_called_verified() -> None:
    result = evaluate_forward_returns(
        make_returns(242, base_cagr=0.30, stress_cagr=0.25),
        contract(),
        bootstrap_repetitions_override=40,
    )
    assert result["status"] == "ONE_YEAR_TARGET_CANDIDATE_NOT_MULTIYEAR"
    assert result["gates"]["base_annualized_excess_at_least_20pct"] is True
    assert result["gates"]["stress_annualized_excess_at_least_20pct"] is True
    assert result["goal_achieved"] is False


def test_three_positive_years_can_verify_target() -> None:
    value = deepcopy(contract())
    result = evaluate_forward_returns(
        make_returns(726, base_cagr=0.30, stress_cagr=0.25),
        value,
        bootstrap_repetitions_override=80,
    )
    assert result["status"] == "VERIFIED_ANNUAL_EXCESS_20PCT"
    assert all(result["gates"].values())
    assert result["goal_achieved"] is True


def test_incomplete_quality_rows_never_advance_maturity() -> None:
    frame = make_returns(100, base_cagr=0.30, stress_cagr=0.25)
    frame.loc[:30, "quality_complete"] = False
    result = evaluate_forward_returns(frame, contract(), bootstrap_repetitions_override=40)
    assert result["eligible_forward_day_count"] == 69
    assert result["status"] == "PENDING_FIRST_UNSEEN_GATE"


def test_pre_effective_rows_are_rejected() -> None:
    frame = make_returns(80, base_cagr=0.30, stress_cagr=0.25)
    frame.loc[0, "trade_date"] = pd.Timestamp("2026-08-26")
    with pytest.raises(ValueError, match="合同生效日前"):
        evaluate_forward_returns(frame, contract(), bootstrap_repetitions_override=40)
