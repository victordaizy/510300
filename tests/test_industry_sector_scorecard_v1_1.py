from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from research.industry_sector_scorecard_v1_1 import (
    build_nonoverlap_cohorts,
    moving_block_bootstrap,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "industry_sector_scorecard_v1_1.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_moving_block_bootstrap_is_deterministic_and_contains_finite_bounds() -> None:
    values = np.array([0.03, 0.01, -0.02, 0.04, -0.01, 0.02], dtype=float)
    first = moving_block_bootstrap(values, 500, 3, 20260819)
    second = moving_block_bootstrap(values, 500, 3, 20260819)
    assert first == second
    assert 0.0 <= first["win_rate_low"] <= first["win_rate_high"] <= 1.0
    assert first["expectancy_low"] <= first["expectancy_high"]
    assert all(np.isfinite(value) for value in first.values())


def test_nonoverlap_cohorts_are_strictly_after_previous_maturity() -> None:
    dates = pd.date_range("2024-01-31", periods=12, freq="ME")
    history = pd.DataFrame(
        {
            "date": dates,
            "entry_date": dates + pd.offsets.BDay(1),
            "maturity_date": dates + pd.offsets.BDay(61),
            "entity_type": "ECONOMIC_BUCKET",
            "entity_id": "A",
            "entity_name_cn": "甲",
            "excess_forward_return_60d": np.linspace(-0.05, 0.06, len(dates)),
            "relative_win": np.arange(len(dates)) % 2 == 0,
        }
    )
    cohorts = build_nonoverlap_cohorts(history, [0, 1, 2])
    assert set(cohorts["cohort_offset"].unique()) == {0, 1, 2}
    for _, group in cohorts.groupby("cohort_offset"):
        ordered = group.sort_values("entry_date").reset_index(drop=True)
        assert (
            ordered["entry_date"].iloc[1:].to_numpy()
            > ordered["maturity_date"].iloc[:-1].to_numpy()
        ).all()


def test_actual_report_has_correct_odds_identity_and_dependence_boundaries() -> None:
    report = json.loads(
        (ROOT / CONFIG["outputs"]["json"]).read_text(encoding="utf-8")
    )
    assert report["sample"]["overlapping_monthly_observations"] == 77
    assert 20 <= report["sample"]["nonoverlap_observations_min"] <= 25
    assert 20 <= report["sample"]["nonoverlap_observations_max"] <= 25
    assert report["score_conditioned_win_rate"] == "UNAVAILABLE_AWAITING_TRUE_FORWARD"
    assert report["may_call_predictive_edge"] is False
    assert report["multiple_comparison_adjusted"] is False
    assert report["odds_implementation_scope"].startswith("gross_theoretical")
    assert report["expectancy_identity_max_error"] < 1.0e-12
    for row in report["economic_buckets"] + report["themes"]:
        assert row["eligible_monthly_observations"] == 77
        assert row["nonoverlap_cohort_observations_median"] < 77
        assert row["average_payoff_ratio"] > 0
        assert row["profit_factor"] > 0
        assert row["breakeven_win_rate"] == pytest.approx(
            1.0 / (1.0 + row["average_payoff_ratio"])
        )
        expected = (
            row["overlapping_relative_win_rate"] * row["average_excess_win"]
            - (1.0 - row["overlapping_relative_win_rate"])
            * row["average_excess_loss"]
        )
        assert row["mean_excess_return_60d"] == pytest.approx(expected, abs=1.0e-12)
        assert row["may_call_predictive_edge"] is False


def test_parent_frozen_manifests_are_unchanged() -> None:
    integrity = CONFIG["parent_integrity"]
    assert _sha256(ROOT / CONFIG["parents"]["v1_manifest"]) == integrity[
        "industry_sector_scorecard_v1_manifest_sha256"
    ]
    assert _sha256(ROOT / CONFIG["parents"]["taxonomy_manifest"]) == integrity[
        "industry_expectation_gap_etf_model_v2_manifest_sha256"
    ]
