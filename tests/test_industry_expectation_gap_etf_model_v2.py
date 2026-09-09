from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from research.industry_expectation_gap_etf_model_v2 import (
    SectorTaxonomyError,
    build_economic_bucket_daily,
    classify_bucket_forward_priorities,
    validate_official_cics_snapshot,
    validate_taxonomy,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load(
    (ROOT / "config" / "industry_expectation_gap_etf_model_v2.yaml").read_text(
        encoding="utf-8"
    )
)


def _forecast_industries() -> list[str]:
    parent = json.loads(
        (ROOT / CONFIG["parent_frozen_result"]).read_text(encoding="utf-8")
    )
    return [str(row["industry_l1"]) for row in parent["industry_rows"]]


def test_taxonomy_covers_current_25_exactly_once() -> None:
    industries = _forecast_industries()
    mapping = validate_taxonomy(industries, CONFIG["taxonomy"])
    assert len(industries) == 25
    assert set(industries).issubset(mapping)
    assert all(mapping[industry] for industry in industries)


def test_taxonomy_rejects_duplicate_economic_membership() -> None:
    taxonomy = {
        "economic_buckets": [
            {"id": "A", "name_cn": "甲", "industries": ["电子"]},
            {"id": "B", "name_cn": "乙", "industries": ["电子"]},
        ],
        "themes": [],
    }
    with pytest.raises(SectorTaxonomyError, match="同时进入"):
        validate_taxonomy(["电子"], taxonomy)


def test_official_cics_snapshot_is_current_and_sums_to_100() -> None:
    path = ROOT / CONFIG["acquired_inputs"]["cics_snapshot"]
    snapshot = pd.read_parquet(path)
    level_one = validate_official_cics_snapshot(snapshot, 0.05)
    assert level_one["date"].max() == pd.Timestamp(CONFIG["as_of_date"])
    assert level_one["industry_name_en"].nunique() == 11
    assert level_one["weight_pct"].sum() == pytest.approx(100.0, abs=0.05)


def test_economic_bucket_daily_preserves_weighted_contribution_identity() -> None:
    industry = pd.DataFrame(
        [
            {
                "date": "2026-08-18",
                "industry_l1": "电子",
                "industry_weight": 0.20,
                "return_coverage_weight": 0.20,
                "industry_return_1d": 0.01,
                "weighted_return_contribution_1d": 0.002,
            },
            {
                "date": "2026-08-18",
                "industry_l1": "计算机",
                "industry_weight": 0.05,
                "return_coverage_weight": 0.05,
                "industry_return_1d": -0.02,
                "weighted_return_contribution_1d": -0.001,
            },
        ]
    )
    taxonomy = {
        "economic_buckets": [
            {
                "id": "CORE_TECHNOLOGY",
                "name_cn": "核心科技",
                "industries": ["电子", "计算机"],
            }
        ],
        "themes": [],
    }
    result = build_economic_bucket_daily(industry, taxonomy)
    assert len(result) == 1
    assert result.iloc[0]["bucket_weight"] == pytest.approx(0.25)
    assert result.iloc[0]["weighted_return_contribution_1d"] == pytest.approx(0.001)
    assert result.iloc[0]["bucket_return_1d"] == pytest.approx(0.004)


def test_forward_priority_requires_gap_coverage_and_price_phase() -> None:
    buckets = [
        {
            "bucket_id": "A",
            "bucket_name_cn": "甲",
            "index_weight": 0.10,
            "return_5d": 0.01,
            "return_20d": 0.02,
            "return_60d_history_percentile": 0.20,
            "fundamental_60d": {"normalized_score": 1.0},
            "expectation_gap": {
                "observed_weight": 0.08,
                "label_weight": {"POSITIVE": 0.06, "NEGATIVE": 0.0},
            },
        },
        {
            "bucket_id": "B",
            "bucket_name_cn": "乙",
            "index_weight": 0.10,
            "return_5d": 0.02,
            "return_20d": 0.01,
            "return_60d_history_percentile": 0.20,
            "fundamental_60d": {"normalized_score": 1.0},
            "expectation_gap": {
                "observed_weight": 0.02,
                "label_weight": {"POSITIVE": 0.02, "NEGATIVE": 0.0},
            },
        },
    ]
    rules = CONFIG["rules"]["qualitative_forward_rules"]
    result = {row["bucket_id"]: row for row in classify_bucket_forward_priorities(buckets, rules)}
    assert result["A"]["research_priority"] == "PRIORITY_FISH_MIDDLE_CANDIDATE"
    assert result["B"]["research_priority"] == "NO_VIEW_EXPECTATION_DATA_GAP"
    assert result["A"]["not_a_trade_signal"] is True


def test_actual_v2_outputs_are_current_and_non_overlapping_at_economic_layer() -> None:
    report = json.loads(
        (ROOT / CONFIG["outputs"]["json"]).read_text(encoding="utf-8")
    )
    assert report["as_of_date"] == "2026-08-18"
    assert report["original_prediction_state"] == "NO_VIEW"
    assert report["original_prediction_changed"] is False
    current_bucket_weight = sum(row["index_weight"] for row in report["economic_buckets"])
    parent = json.loads(
        (ROOT / CONFIG["parent_frozen_result"]).read_text(encoding="utf-8")
    )
    assert current_bucket_weight == pytest.approx(
        parent["industry_aggregation"]["sector_snapshot"]["weight_coverage"]
    )
    assert all(row["effective_weight"] is None for row in report["themes"])
    assert all(row["cross_theme_sum_allowed"] is False for row in report["themes"])

