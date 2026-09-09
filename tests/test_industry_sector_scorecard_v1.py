from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.industry_sector_scorecard_v1 import (
    aggregate_forward_history,
    score_current_entities,
    summarize_base_rates,
    wilson_interval,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (ROOT / "config" / "industry_sector_scorecard_v1.yaml").read_text(
        encoding="utf-8"
    )
)


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_interval(48, 77, 0.95)
    assert low is not None and high is not None
    assert low < 48 / 77 < high
    assert low == pytest.approx(0.512, abs=0.002)
    assert high == pytest.approx(0.723, abs=0.002)


def test_forward_history_aggregates_contribution_without_reweighting_market() -> None:
    targets = pd.DataFrame(
        [
            {
                "date": "2025-01-31",
                "industry_l1": "电子",
                "sector_weight": 0.20,
                "sector_contribution_60d": 0.02,
                "snapshot_index_return_60d": 0.05,
                "target_output": "TARGET_READY",
            },
            {
                "date": "2025-01-31",
                "industry_l1": "计算机",
                "sector_weight": 0.05,
                "sector_contribution_60d": -0.0025,
                "snapshot_index_return_60d": 0.05,
                "target_output": "TARGET_READY",
            },
        ]
    )
    definitions = [
        {
            "entity_id": "TECH",
            "entity_name_cn": "科技",
            "industries": ["电子", "计算机"],
        }
    ]
    result = aggregate_forward_history(targets, definitions, "ECONOMIC_BUCKET", 1e-6)
    assert len(result) == 1
    assert result.iloc[0]["entity_weight"] == pytest.approx(0.25)
    assert result.iloc[0]["entity_forward_return_60d"] == pytest.approx(0.07)
    assert result.iloc[0]["excess_forward_return_60d"] == pytest.approx(0.02)
    assert bool(result.iloc[0]["relative_win"]) is True


def test_base_rate_frequency_is_win_count_divided_by_observations() -> None:
    history = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2025-01-31"),
                "entity_type": "ECONOMIC_BUCKET",
                "entity_id": "A",
                "entity_name_cn": "甲",
                "entity_forward_return_60d": 0.10,
                "market_forward_return_60d": 0.05,
                "excess_forward_return_60d": 0.05,
                "relative_win": True,
                "absolute_win": True,
            },
            {
                "date": pd.Timestamp("2025-02-28"),
                "entity_type": "ECONOMIC_BUCKET",
                "entity_id": "A",
                "entity_name_cn": "甲",
                "entity_forward_return_60d": -0.02,
                "market_forward_return_60d": 0.01,
                "excess_forward_return_60d": -0.03,
                "relative_win": False,
                "absolute_win": False,
            },
        ]
    )
    result = summarize_base_rates(history, 0.95, 1, False).iloc[0]
    assert result["relative_win_count"] == 1
    assert result["eligible_monthly_observations"] == 2
    assert result["relative_win_rate"] == pytest.approx(0.5)
    assert result["absolute_win_rate"] == pytest.approx(0.5)


def test_current_score_is_bounded_and_not_probability() -> None:
    inputs = pd.DataFrame(
        [
            {
                "entity_type": "ECONOMIC_BUCKET",
                "entity_id": "A",
                "entity_name_cn": "甲",
                "current_index_weight": 0.1,
                "fundamental_60d_score": 2.0,
                "expectation_gap_score": 1.0,
                "expectation_coverage_ratio": 1.0,
                "price_phase": "EARLY_RECOVERY_FROM_WEAK_BASE",
                "flow_intensity_5d": 0.02,
                "weighted_earnings_yield": 0.08,
            },
            {
                "entity_type": "ECONOMIC_BUCKET",
                "entity_id": "B",
                "entity_name_cn": "乙",
                "current_index_weight": 0.1,
                "fundamental_60d_score": -2.0,
                "expectation_gap_score": -1.0,
                "expectation_coverage_ratio": 1.0,
                "price_phase": "LATE_OR_EXTENDED",
                "flow_intensity_5d": -0.02,
                "weighted_earnings_yield": 0.02,
            },
        ]
    )
    result = score_current_entities(inputs, CONFIG["current_score"])
    assert result["current_score"].between(0, 100).all()
    assert result.iloc[0]["entity_id"] == "A"
    assert result.iloc[0]["grade"] == "A"
    assert result["score_is_probability"].eq(False).all()
    assert result["score_conditioned_win_rate"].isna().all()


def test_actual_scorecard_has_complete_counts_and_frozen_failure_boundary() -> None:
    report = json.loads(
        (ROOT / CONFIG["outputs"]["json"]).read_text(encoding="utf-8")
    )
    assert report["historical_sample"]["mature_monthly_observations"] == 77
    assert report["historical_sample"]["effective_nonoverlapping_blocks"] == 24
    assert report["failed_model_reference"]["status"] == "HISTORICAL_REJECTED_FROZEN"
    assert report["failed_model_reference"]["reuse_allowed"] is False
    assert len(report["economic_buckets"]) == 11
    assert len(report["themes"]) == 10
    for row in report["economic_buckets"] + report["themes"]:
        assert 0 <= row["current_score"] <= 100
        assert row["relative_win_count"] <= row["eligible_monthly_observations"]
        assert row["relative_win_rate"] == pytest.approx(
            row["relative_win_count"] / row["eligible_monthly_observations"]
        )
        assert row["score_conditioned_win_rate"] is None
        assert row["score_conditioned_win_rate_state"] == "UNAVAILABLE_AWAITING_TRUE_FORWARD"

