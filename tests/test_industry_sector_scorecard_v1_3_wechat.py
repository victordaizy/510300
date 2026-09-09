from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from research.industry_sector_scorecard_v1_3_wechat import (
    build_evidence_frame,
    validate_wechat_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "industry_sector_scorecard_v1_3_wechat_forward.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
REPORT_PATH = ROOT / CONFIG["outputs"]["json"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_report() -> dict[str, object]:
    return json.loads(REPORT_PATH.read_text(encoding="utf-8"))


def _entity_set(rows: list[dict[str, object]]) -> set[tuple[str, str]]:
    return {(str(row["entity_type"]), str(row["entity_id"])) for row in rows}


def test_wechat_snapshot_is_point_in_time_and_selected_articles_are_complete() -> None:
    selected, audit = validate_wechat_snapshot(CONFIG)

    assert audit["status"] == "PARTIAL_SUCCESS"
    assert audit["article_count"] == 180
    assert audit["archive_complete_count"] == 158
    assert audit["archive_partial_count"] == 22
    assert len(selected) == 15
    assert selected["source_name"].nunique() == 6
    assert selected["article_id"].is_unique
    assert selected["archive_status"].eq("SUCCESS").all()
    cutoff = pd.Timestamp(CONFIG["information_cutoff"]).tz_convert("UTC")
    assert selected["published_at"].le(cutoff).all()
    assert selected["markdown_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_high_confidence_requires_at_least_three_independent_sources() -> None:
    selected, _ = validate_wechat_snapshot(CONFIG)
    economic = build_evidence_frame(
        CONFIG["economic_bucket_assessments"],
        "ECONOMIC_BUCKET",
        selected,
        CONFIG["evidence_rules"],
    )
    themes = build_evidence_frame(
        CONFIG["theme_assessments"],
        "THEME",
        selected,
        CONFIG["evidence_rules"],
    )
    evidence = pd.concat([economic, themes], ignore_index=True)
    minimum = CONFIG["evidence_rules"][
        "minimum_independent_sources_for_high_confidence"
    ]

    assert CONFIG["evidence_rules"]["use_raw_article_count_as_vote"] is False
    assert evidence.loc[
        evidence["wechat_confidence"].eq("HIGH"), "independent_source_count"
    ].ge(minimum).all()


def test_integrated_forward_states_match_frozen_three_axis_decision() -> None:
    report = _load_report()

    assert _entity_set(report["fish_middle_convergence"]) == {
        ("ECONOMIC_BUCKET", "ENERGY_RESOURCES"),
        ("ECONOMIC_BUCKET", "CORE_TECHNOLOGY"),
        ("ECONOMIC_BUCKET", "DIGITAL_COMMUNICATION"),
        ("THEME", "AI_DIGITAL_INFRASTRUCTURE"),
        ("THEME", "BROAD_TECHNOLOGY"),
        ("THEME", "NEW_QUALITY_PRODUCTIVITY"),
        ("THEME", "EXPORT_MANUFACTURING"),
    }
    assert _entity_set(report["qualitative_leads_quant"]) == {
        ("ECONOMIC_BUCKET", "ADVANCED_MANUFACTURING"),
        ("THEME", "ADVANCED_MANUFACTURING_THEME"),
    }
    assert _entity_set(report["fish_tail_warnings"]) == {
        ("ECONOMIC_BUCKET", "HEALTHCARE")
    }
    assert _entity_set(report["negative_convergence"]) == {
        ("ECONOMIC_BUCKET", "MIDSTREAM_MATERIALS"),
        ("ECONOMIC_BUCKET", "CONSUMER_DISCRETIONARY"),
        ("ECONOMIC_BUCKET", "CONSUMER_STAPLES"),
        ("ECONOMIC_BUCKET", "REAL_ESTATE_INFRASTRUCTURE"),
        ("THEME", "DOMESTIC_DEMAND"),
        ("THEME", "REAL_ESTATE_CHAIN"),
    }


def test_report_preserves_corpus_bias_and_market_data_boundaries() -> None:
    report = _load_report()
    corpus = report["corpus_audit"]
    context = report["market_forward_context"]

    assert corpus["article_count"] == 180
    assert corpus["archive_complete_count"] == 158
    assert corpus["archive_partial_count"] == 22
    assert corpus["selected_article_count"] == 15
    assert corpus["selected_source_count"] == 6
    assert corpus["largest_source_article_share"] == 50 / 180
    assert corpus["raw_article_count_used_as_vote"] is False
    assert context["liquidity_state"] == "MIXED_INTERNAL_RECOVERY_EXTERNAL_OUTFLOW"
    assert (
        context["national_team_holdings_state"]
        == "NO_VIEW_NO_AUDITABLE_POSITION_LEVEL_DATA"
    )


def test_report_never_turns_wechat_judgment_into_probability_or_position() -> None:
    report = _load_report()
    rows = report["economic_buckets"] + report["themes"]

    assert report["synthetic_numeric_score"] is None
    assert report["wechat_direction_is_probability"] is False
    assert report["score_conditioned_win_rate"] == "UNAVAILABLE_AWAITING_TRUE_FORWARD"
    assert report["may_call_predictive_edge"] is False
    assert report["position_mapping_enabled"] is False
    for row in rows:
        assert row["synthetic_numeric_score"] is None
        assert row["synthetic_combined_numeric_score"] is None
        assert row["score_conditioned_win_rate"] is None
        assert row["wechat_direction_is_probability"] is False
        assert row["position_mapping_enabled"] is False


def test_markdown_escapes_pipe_characters_in_article_titles() -> None:
    report = _load_report()
    markdown = (ROOT / CONFIG["outputs"]["markdown"]).read_text(encoding="utf-8")

    titles_with_pipe = [
        row["title"] for row in report["selected_articles"] if "|" in row["title"]
    ]
    assert titles_with_pipe
    for title in titles_with_pipe:
        assert title.replace("|", r"\|") in markdown


def test_v1_2_parent_manifest_is_unchanged() -> None:
    expected = CONFIG["parent_integrity"][
        "industry_sector_scorecard_v1_2_manifest_sha256"
    ]
    assert _sha256(ROOT / CONFIG["parents"]["v1_2_manifest"]) == expected

