from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

import mechanism_atlas_v1 as atlas


@pytest.fixture(scope="module")
def config() -> dict:
    return atlas.load_config()


@pytest.fixture(scope="module")
def inputs(config: dict) -> dict:
    return atlas.load_inputs(config)


@pytest.fixture(scope="module")
def audit(config: dict, inputs: dict) -> dict:
    return atlas.audit_inputs(config, inputs)


@pytest.fixture(scope="module")
def panel(config: dict, inputs: dict) -> pd.DataFrame:
    return atlas.build_state_panel(config, inputs)


@pytest.fixture(scope="module")
def outcomes(panel: pd.DataFrame, config: dict) -> pd.DataFrame:
    return atlas.build_outcome_panel(panel, config)


@pytest.fixture(scope="module")
def matched(
    panel: pd.DataFrame,
    outcomes: pd.DataFrame,
    inputs: dict,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return atlas.select_cases_and_match_controls(
        panel, outcomes, inputs["source_events"], config
    )


def test_protocol_preserves_rejected_source_and_no_trade_boundary(config: dict) -> None:
    assert config["source_branch"]["required_status"] == (
        "REJECTED_FROZEN_MACD_BREADTH_DOWNSIDE_PREFLIGHT_V1_0_1_NO_RESCUE"
    )
    assert config["scope"]["allowed_holdings"] == ["510300.SH", "CASH_CNY"]
    assert config["scope"]["return_backtest_allowed"] is False
    assert config["scope"]["order_generation"] is False
    assert config["scope"]["broker_connection"] is False
    assert config["scope"]["live_trading_authorized"] is False
    assert config["adjudication"]["return_evaluation"] == "NOT_ALLOWED"
    assert config["adjudication"]["net_sharpe"] == "NOT_COMPUTED"


def test_input_audit_passes_and_all_anchor_dates_are_fixed(audit: dict) -> None:
    assert audit["status"] == "PASS_PHASE1_COMMON_DATA_CONTRACT"
    assert audit["passed"] is True
    assert audit["common_window"]["rows"] == 1211
    assert audit["source_branch"]["passed"] is False
    assert audit["source_branch"]["return_evaluation"] == "NOT_ALLOWED"
    assert audit["anchor_events"]["actual"] == [
        "2023-07-12",
        "2024-02-29",
        "2024-05-13",
        "2025-01-27",
        "2025-07-10",
        "2026-07-01",
    ]
    assert audit["anchor_events"]["all_valid_for_attribution"] is True
    assert audit["point_in_time_cross_section"]["future_weight_rows"] == 0


def test_state_panel_is_point_in_time_and_contains_no_future_outcomes(
    panel: pd.DataFrame,
) -> None:
    assert len(panel) == 1211
    assert panel["date"].is_monotonic_increasing
    assert not panel["date"].duplicated().any()
    assert not any(column.startswith("forward_") for column in panel.columns)
    assert "if_raw_annualized_near_basis" in panel.columns
    assert "if_fair_basis" not in panel.columns
    for prefix in ("fdr007", "pmi_new_orders", "tsf_stock_yoy", "usdcny"):
        available = panel[f"{prefix}_available_at"]
        assert (available.dropna() <= panel.loc[available.notna(), "signal_timestamp"]).all()
    assert panel["official_breadth_valid"].all()


def test_outcome_panel_is_separate_and_censors_incomplete_horizons(
    panel: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    assert list(outcomes.columns[:2]) == ["date", "forward_return_5d"]
    assert len(outcomes) == len(panel)
    assert outcomes["outcome_complete_5d"].sum() == len(panel) - 5
    assert outcomes["outcome_complete_60d"].sum() == len(panel) - 60
    assert outcomes.loc[
        outcomes["date"].eq(pd.Timestamp("2026-07-01")), "forward_return_10d"
    ].iloc[0] < 0


def test_global_matching_uses_three_unique_controls_per_case(
    matched: tuple[pd.DataFrame, pd.DataFrame, dict],
) -> None:
    cases, matches, balance = matched
    assert len(cases) == 6
    assert len(matches) == 18
    assert matches["control_date"].nunique() == 18
    assert matches.groupby("case_id")["control_date"].nunique().eq(3).all()
    assert matches["control_forward_return_10d_frozen"].le(0).all()
    assert balance["maximum_pair_distance"] <= 3.25
    assert balance["maximum_absolute_standardized_mean_difference"] <= 0.75
    assert all(balance["gates"].values())


def test_event_windows_and_chronology_are_complete(
    panel: pd.DataFrame,
    matched: tuple[pd.DataFrame, pd.DataFrame, dict],
    config: dict,
) -> None:
    cases, matches, _ = matched
    windows = atlas.build_event_window_panel(panel, cases, matches, config)
    assert len(windows) == 24 * 81
    assert windows.groupby("unit_id").size().eq(81).all()
    assert set(windows["relative_trading_day"].unique()) == set(range(-60, 21))
    chronology = atlas.build_chronology_table(windows, config)
    assert len(chronology) == 24
    assert chronology["unit_id"].nunique() == 24
    assert set(chronology["unit_type"]) == {"CASE", "FAILED_CONTROL"}


def test_local_projection_family_and_factor_states_are_frozen(
    panel: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: dict,
) -> None:
    results, rolling, audit = atlas.run_local_projections(panel, outcomes, config)
    family = results.loc[results["primary_interaction_family_member"]]
    assert len(family) == 8
    assert family["adjusted_p_value_holm"].notna().all()
    assert audit["primary_interaction_family_count"] == 8
    assert not rolling.empty
    matrix = atlas.build_factor_matrix(results)
    allowed = {
        "STRUCTURAL",
        "CONDITIONAL",
        "DECAYING",
        "DESCRIPTIVE_ONLY",
        "REJECTED",
        "DATA_BLOCKED",
    }
    assert set(matrix["current_status"]).issubset(allowed)
    h4 = matrix.loc[matrix["factor"].eq("INTRADAY_ETF_IF_IOPV_PRESSURE")]
    assert h4["current_status"].iloc[0] == "DATA_BLOCKED"
    assert h4["strategy_authorization"].iloc[0] == (
        "NONE_NO_VIEW_DATA_CONTRACT_FAILED"
    )

