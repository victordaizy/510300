from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

import stress_transmission_and_exhaustion_atlas_v1 as atlas


def test_protocol_is_discovery_only_and_preserves_upstream_rejections() -> None:
    config = atlas.load_config()
    assert config["protocol"]["research_stage"] == "MECHANISM_DISCOVERY_ONLY"
    assert config["protocol"]["strategy_backtest_allowed"] is False
    assert config["scope"]["portfolio_mapping"] == "disabled"
    assert config["scope"]["current_signal"] == "NOT_CREATED"
    assert config["scope"]["live_trading_authorized"] is False
    assert config["upstream_boundaries"]["rejected_router"]["required_status"] == (
        "REJECTED_FROZEN_STATE_IDENTIFIABILITY_GATE_FAILED_NO_PHASE_2_NO_RESCUE"
    )
    assert config["upstream_boundaries"]["systemic_sell_pressure"][
        "required_status"
    ] == "VISIBLE_REJECTED_FROZEN_REPLICATION_UNREAD"
    assert config["known_data_limits"][
        "point_in_time_fundamental_expectation_2015_onward"
    ] == "NO_VIEW_INCOMPLETE_CONTRACT"


def test_causal_percentile_excludes_current_observation() -> None:
    values = pd.Series([1.0, 2.0, 100.0, -100.0])
    result = atlas.causal_percentile(values, prior_rows=2, minimum_prior_rows=2)
    assert result.iloc[:2].isna().all()
    assert result.iloc[2] == 1.0
    assert result.iloc[3] == 0.0


def test_signed_timing_distinguishes_leading_synchronous_and_lagging() -> None:
    dates = pd.bdate_range("2020-01-01", periods=10)

    leading = pd.Series([0.1, 0.2, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], index=dates)
    crossing, timing, censored = atlas._crossing_timing(leading, 4, 4, 3, 0.8)
    assert crossing == dates[2]
    assert timing == 2.0
    assert censored is False

    synchronous = pd.Series(
        [0.1, 0.2, 0.3, 0.4, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3], index=dates
    )
    crossing, timing, censored = atlas._crossing_timing(
        synchronous, 4, 4, 3, 0.8
    )
    assert crossing == dates[4]
    assert timing == 0.0
    assert censored is False

    lagging = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 0.7, 0.6, 0.5], index=dates)
    crossing, timing, censored = atlas._crossing_timing(lagging, 4, 4, 3, 0.8)
    assert crossing == dates[6]
    assert timing == -2.0
    assert censored is False


def test_timing_window_excludes_left_censored_threshold_state() -> None:
    dates = pd.bdate_range("2020-01-01", periods=8)
    values = pd.Series([0.9, 0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2], index=dates)
    crossing, timing, censored = atlas._crossing_timing(values, 4, 4, 2, 0.8)
    assert pd.isna(crossing)
    assert timing is None
    assert censored is True


def test_independent_repair_anchor_does_not_depend_on_exhaustion_candidate() -> None:
    dates = pd.bdate_range("2020-01-01", periods=8)
    panel = pd.DataFrame(
        {
            "date": dates,
            "total_return_3d": [-0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "breadth_change5_rebuilt": [-0.2, -0.1, -0.1, 0.2, 0.2, 0.2, 0.2, 0.2],
            "total_return_index_rebuilt": [90, 91, 92, 101, 102, 103, 104, 105],
            "price_ma20_rebuilt": [100] * 8,
            "exhaustion_candidate": [False, False, False, False, False, False, False, False],
        }
    )
    anchor = atlas._first_independent_repair_anchor(panel, 0, 5, 2)
    assert anchor == dates[3]


def test_outcome_dividend_requires_entry_by_record_date_and_accrues_on_ex_date() -> None:
    dates = pd.bdate_range("2020-01-01", periods=30)
    market = pd.DataFrame(
        {
            "date": dates,
            "open": np.full(len(dates), 100.0),
            "high": np.full(len(dates), 100.0),
            "low": np.full(len(dates), 100.0),
            "close": np.full(len(dates), 100.0),
        }
    )
    dividends = pd.DataFrame(
        {
            "record_date": [dates[2]],
            "ex_date": [dates[3]],
            "cash_dividend_per_share": [1.0],
        }
    )
    mechanism_panel = pd.DataFrame({"date": [dates[0], dates[2]]})
    config = deepcopy(atlas.load_config())
    config["dates"]["outcome_market_cutoff"] = dates[-1].date().isoformat()
    outcomes = atlas.build_outcome_panel(config, mechanism_panel, market, dividends)
    assert outcomes.loc[0, "entry_date"] == dates[1]
    assert np.isclose(outcomes.loc[0, "future_return_10d"], 0.01)
    assert outcomes.loc[1, "entry_date"] == dates[3]
    assert outcomes.loc[1, "future_return_10d"] == 0.0


def test_direction_and_timing_classification_require_non_censored_sample() -> None:
    summary = atlas.direction_summary(
        pd.Series([2.0] * 8 + [0.0, -1.0]),
        expected="positive",
        minimum_observations=10,
        minimum_fraction=0.60,
        maximum_pvalue=0.10,
        minimum_median=1.0,
    )
    assert summary.passed is True
    assert atlas._timing_classification(pd.Series([2.0] * 8 + [0.0, -1.0]), True, 10) == (
        "LEADING"
    )
    assert atlas._timing_classification(pd.Series([0.0] * 10), False, 10) == (
        "SYNCHRONOUS"
    )
    assert atlas._timing_classification(pd.Series([-1.0] * 10), False, 10) == (
        "LAGGING"
    )


def test_insufficient_event_model_is_not_promoted() -> None:
    config = atlas.load_config()
    frame = pd.DataFrame(
        {
            "event_id": [f"E{index:02d}" for index in range(10)],
            "era": ["ERA_2015_2016"] * 10,
            "pressure_source_score": np.linspace(0.1, 0.9, 10),
            "transmission_score": np.linspace(0.2, 0.8, 10),
            "major_left_tail_10d": [0, 1] * 5,
        }
    )
    result, rows = atlas.evaluate_question_model(
        config,
        "QUESTION_A_LEFT_TAIL",
        frame,
        ["pressure_source_score", "transmission_score"],
        "major_left_tail_10d",
        "transmission_score",
    )
    assert result["status"] == "NOT_EVALUABLE_INSUFFICIENT_EVENT_OR_CLASS_SAMPLE"
    assert result["model_gate_passed"] is False
    assert rows.empty
