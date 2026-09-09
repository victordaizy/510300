from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from research.industry_sector_scorecard_v1_2 import (
    assign_research_priority,
    classify_current_axis,
    classify_historical_odds,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "industry_sector_scorecard_v1_2.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _odds_row(**overrides: float) -> dict[str, float]:
    row = {
        "mean_excess_return_60d": 0.02,
        "profit_factor": 1.4,
        "average_payoff_ratio": 1.5,
        "nonoverlap_cohort_win_rate_min": 0.55,
        "breakeven_win_rate": 0.40,
        "block_bootstrap_expectancy_high": 0.05,
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("score", "expected", "positive"),
    [
        (65.0, "CURRENT_FORWARD_POSITIVE", True),
        (64.999, "CURRENT_FORWARD_WATCH", False),
        (55.0, "CURRENT_FORWARD_WATCH", False),
        (54.999, "CURRENT_FORWARD_WEAK", False),
    ],
)
def test_current_axis_uses_frozen_grade_boundaries(
    score: float, expected: str, positive: bool
) -> None:
    actual, flag = classify_current_axis(score, CONFIG["current_forward_axis"])
    assert actual == expected
    assert flag is positive


def test_historical_axis_requires_all_natural_payoff_gates() -> None:
    axis, favorable = classify_historical_odds(
        _odds_row(), CONFIG["historical_odds_axis"]
    )
    assert axis == "FAVORABLE_POINT_ESTIMATE_UNCONFIRMED"
    assert favorable is True

    axis, favorable = classify_historical_odds(
        _odds_row(average_payoff_ratio=0.9), CONFIG["historical_odds_axis"]
    )
    assert axis == "MIXED_HISTORICAL_ODDS_UNCERTAIN"
    assert favorable is False

    axis, favorable = classify_historical_odds(
        _odds_row(block_bootstrap_expectancy_high=-0.001),
        CONFIG["historical_odds_axis"],
    )
    assert axis == "ADVERSE_IN_SAMPLE_NOT_OOS"
    assert favorable is False


def test_priority_matrix_never_emits_position_action() -> None:
    matrix = CONFIG["decision_matrix"]
    cases = [
        (True, True, "FAVORABLE_POINT_ESTIMATE_UNCONFIRMED"),
        (True, False, "MIXED_HISTORICAL_ODDS_UNCERTAIN"),
        (False, False, "UNFAVORABLE_POINT_ESTIMATE_UNCONFIRMED"),
        (False, False, "ADVERSE_IN_SAMPLE_NOT_OOS"),
    ]
    tiers = []
    for current, historical, axis in cases:
        tier, action = assign_research_priority(
            current, historical, axis, matrix
        )
        tiers.append(tier)
        assert action.endswith("NO_POSITION")
    assert tiers == [
        "P1_DUAL_AXIS_ALIGN_UNCONFIRMED",
        "P2_ONE_AXIS_ONLY",
        "P3_NO_POSITIVE_ALIGNMENT",
        "P4_HISTORICAL_ADVERSE_IN_SAMPLE",
    ]


def test_actual_v1_2_has_expected_dual_axis_sets_and_no_synthetic_score() -> None:
    report = json.loads(
        (ROOT / CONFIG["outputs"]["json"]).read_text(encoding="utf-8")
    )
    p1 = {
        (row["entity_type"], row["entity_id"])
        for row in report["p1_research_watchlist"]
    }
    assert p1 == {
        ("ECONOMIC_BUCKET", "ENERGY_RESOURCES"),
        ("ECONOMIC_BUCKET", "CORE_TECHNOLOGY"),
        ("ECONOMIC_BUCKET", "DIGITAL_COMMUNICATION"),
        ("THEME", "AI_DIGITAL_INFRASTRUCTURE"),
        ("THEME", "BROAD_TECHNOLOGY"),
        ("THEME", "NEW_QUALITY_PRODUCTIVITY"),
        ("THEME", "EXPORT_MANUFACTURING"),
    }
    all_rows = report["economic_buckets"] + report["themes"]
    adverse = {
        (row["entity_type"], row["entity_id"])
        for row in all_rows
        if row["research_priority_tier"]
        == "P4_HISTORICAL_ADVERSE_IN_SAMPLE"
    }
    assert adverse == {
        ("ECONOMIC_BUCKET", "HEALTHCARE"),
        ("THEME", "REAL_ESTATE_CHAIN"),
    }
    assert report["synthetic_combined_numeric_score"] is None
    assert report["current_score_is_probability"] is False
    assert report["historical_odds_are_current_conditional_probability"] is False
    assert report["score_conditioned_win_rate"] == "UNAVAILABLE_AWAITING_TRUE_FORWARD"
    assert report["may_call_predictive_edge"] is False
    assert report["position_mapping_enabled"] is False
    for entity_type in ("ECONOMIC_BUCKET", "THEME"):
        ranks = [
            row["research_priority_rank"]
            for row in all_rows
            if row["entity_type"] == entity_type
        ]
        assert ranks == list(range(1, len(ranks) + 1))
    for row in all_rows:
        assert row["synthetic_combined_numeric_score"] is None
        assert row["score_conditioned_win_rate"] is None
        assert row["position_mapping_enabled"] is False


def test_v1_1_parent_manifest_is_unchanged() -> None:
    expected = CONFIG["parent_integrity"][
        "industry_sector_scorecard_v1_1_manifest_sha256"
    ]
    assert _sha256(ROOT / CONFIG["parents"]["v1_1_manifest"]) == expected
