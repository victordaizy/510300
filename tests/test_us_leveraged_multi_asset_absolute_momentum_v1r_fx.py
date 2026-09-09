"""美国杠杆多资产V1R汇率修正的差异与边界测试。"""

from __future__ import annotations

import json

from research.us_leveraged_multi_asset_absolute_momentum_v1r_fx import (
    ALLOWED_CHANGED_PATHS,
    changed_paths,
    load_contract,
    load_visible_inputs,
)
from scripts.freeze_us_leveraged_multi_asset_absolute_momentum_v1r_fx import (
    TRACKED_FILES,
    VISIBLE_SOURCE_DEPENDENCIES,
)


def test_v1r_actual_changes_are_exactly_the_v6_authorized_paths() -> None:
    contract = load_contract()
    assert set(changed_paths(contract)) == ALLOWED_CHANGED_PATHS
    assert contract["protocol"]["candidate_id"].endswith("V1R_FX_SOURCE_CORRECTION")
    assert contract["currency"]["maximum_fx_staleness_calendar_days"] == 14
    assert contract["data_sources"]["fx"] == "CHINAMONEY_CFETS_CC_PR_HISTORICAL_USD_CNY"


def test_v1r_keeps_goal_costs_dates_products_and_safety() -> None:
    contract = load_contract()
    assert contract["account"]["initial_capital_cny"] == 500000.0
    assert contract["account"]["user_transaction_fee_rate_per_leg"] == 0.0001
    assert contract["visible_gates"]["minimum_annualized_net_excess"] == 0.40
    assert contract["visible_gates"]["minimum_strategy_net_sharpe"] == 1.50
    assert contract["universe"]["fixed_tickers"] == ["UPRO", "TQQQ", "TMF", "UGL"]
    assert contract["historical_partition"]["visible_start"] == "2014-01-02"
    assert contract["historical_partition"]["visible_end"] == "2019-12-31"
    assert contract["costs"]["base_total_security_cost_bps_per_leg"] == 17.0
    assert contract["costs"]["stress_total_security_cost_bps_per_leg"] == 47.0
    assert not any(contract["safety"].values())


def test_v1r_signal_only_loader_does_not_read_fx_open_or_sealed_inputs() -> None:
    contract = load_contract()
    panel, master, fx, benchmark, audit = load_visible_inputs(
        contract, signal_only=True
    )
    assert len(panel) == 10349
    assert master["ticker"].nunique() == 4
    assert fx.empty and benchmark.empty
    assert audit["future_open_or_strategy_return_read"] is False
    assert audit["sealed_panel_or_benchmark_read"] is False
    assert "raw_open" not in audit["panel_columns_read"]


def test_corrected_status_confirms_no_strategy_result_and_ten_day_worst_gap() -> None:
    contract = load_contract()
    status_path = contract["inputs"]["source_status"]
    status = json.loads(open(status_path, encoding="utf-8").read())
    assert status["strategy_total_return_or_rank_computed"] is False
    assert status["portfolio_nav_or_target_gate_computed"] is False
    assert status["visible_strict_lag_coverage"]["maximum_fx_age_calendar_days"] == 10
    assert status["visible_strict_lag_coverage"]["strictly_lagged"] is True


def test_v1r_freeze_scope_tracks_code_and_excludes_sealed_inputs() -> None:
    assert "research/us_leveraged_multi_asset_absolute_momentum_v1r_fx.py" in TRACKED_FILES
    assert "scripts/run_us_leveraged_multi_asset_absolute_momentum_v1r_fx.py" in TRACKED_FILES
    assert all("sealed" not in path.lower() for path in VISIBLE_SOURCE_DEPENDENCIES)
