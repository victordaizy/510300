from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from research.predictable_fish_middle_decision_v1 import (
    PredictableDecisionError,
    aggregate_exclusive_sector_states,
    build_reliability_evidence,
    evaluate_entry_gates,
    validate_parent_hashes,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "510300_predictable_fish_middle_decision_v1.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _load(relative_path: str) -> dict[str, object]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parents() -> dict[str, dict[str, object]]:
    parents = CONFIG["parents"]
    return {
        key: _load(parents[key])
        for key in (
            "sector_scorecard_report",
            "expectation_gap_report",
            "sector_prediction_result",
            "driver_episode_result",
            "daily_quality",
            "forward_input_refresh",
            "etf_forward_readiness",
        )
    }


def test_all_parent_hashes_are_frozen_and_unchanged() -> None:
    actual = validate_parent_hashes(ROOT, CONFIG)

    assert actual == CONFIG["parent_integrity"]
    for key, relative_path in CONFIG["parents"].items():
        assert _sha256(ROOT / relative_path) == CONFIG["parent_integrity"][key]


def test_exclusive_sector_weights_close_the_index_structure_without_themes() -> None:
    parents = _parents()
    scorecard = parents["sector_scorecard_report"]
    structure = aggregate_exclusive_sector_states(
        scorecard["economic_buckets"], CONFIG["exclusive_sector_states"]
    )

    assert structure["mapped_index_weight"] == pytest.approx(0.99365)
    assert structure["fish_middle_weight"] == pytest.approx(0.39133)
    assert structure["qualitative_leads_weight"] == pytest.approx(0.15308)
    assert structure["positive_structure_weight"] == pytest.approx(0.54441)
    assert structure["fish_tail_watch_weight"] == pytest.approx(0.04354)
    assert structure["negative_convergence_weight"] == pytest.approx(0.14709)
    assert structure["mixed_no_view_weight"] == pytest.approx(0.25861)
    assert structure["unmapped_index_weight"] == pytest.approx(0.00635)
    assert structure["positive_structure_exceeds_negative_weight"] is True


def test_overlapping_theme_cannot_enter_exclusive_weight_aggregation() -> None:
    parents = _parents()
    theme = parents["sector_scorecard_report"]["themes"][0]

    with pytest.raises(PredictableDecisionError, match="互斥经济大类"):
        aggregate_exclusive_sector_states(
            [theme], CONFIG["exclusive_sector_states"]
        )


def test_current_gate_result_says_data_current_but_no_predictive_entry() -> None:
    p = _parents()
    result = evaluate_entry_gates(
        CONFIG,
        p["sector_scorecard_report"],
        p["expectation_gap_report"],
        p["sector_prediction_result"],
        p["driver_episode_result"],
        p["daily_quality"],
        p["forward_input_refresh"],
        p["etf_forward_readiness"],
    )

    assert result["gates"]["G1_DATA_FRESHNESS"]["passed"] is True
    assert result["gates"]["G2_INDEX_EXPECTATION_GAP"]["passed"] is False
    assert result["gates"]["G3_TRUE_FORWARD_PREDICTABILITY"]["passed"] is False
    assert result["gates"]["G4_MARKET_LIQUIDITY"]["passed"] is False
    assert result["gates"]["G5_TAIL_AND_NATIONAL_TEAM"]["passed"] is False
    assert result["gates"]["G6_ETF_EXECUTION_RESEARCH_READINESS"]["passed"] is False
    assert result["all_research_entry_gates_passed"] is False
    assert result["current_entry_state"] == "WAIT_NO_PREDICTIVE_ENTRY"
    assert result["current_action"] == "WAIT_NO_NEW_ENTRY"
    assert result["existing_holding_action"] == "NOT_EVALUATED_NO_POSITION_DATA_READ"


def test_rejected_s1_m1_t1_cannot_be_presented_as_predictive() -> None:
    p = _parents()
    evidence = build_reliability_evidence(
        p["sector_prediction_result"], p["driver_episode_result"]
    )

    assert [row["candidate_id"] for row in evidence] == [
        "S1_SECTOR_FUNDAMENTAL_RIDGE_V1",
        "M1",
        "T1",
    ]
    assert all(row["status"] == "HISTORICAL_REJECTED_FROZEN" for row in evidence)
    assert all(row["forecast_eligible"] is False for row in evidence)
    assert evidence[0]["primary_metric"] == pytest.approx(0.39473684210526316)
    assert evidence[1]["primary_metric"] == pytest.approx(0.9489664082687338)
    assert evidence[2]["primary_metric"] == pytest.approx(1.0492857142857142)


def test_generated_report_closes_goal_without_inventing_probability_or_position() -> None:
    report = _load(CONFIG["outputs"]["json"])

    assert report["completion_state"] == "MODEL_LOGIC_COMPLETE_CURRENT_NO_ENTRY"
    assert (
        report["final_decision"]["current_research_view"]
        == "SECTOR_STRUCTURE_POSITIVE_NOT_INDEX_FORECAST"
    )
    assert report["final_decision"]["current_entry_state"] == "WAIT_NO_PREDICTIVE_ENTRY"
    assert report["statistics_semantics"]["score_conditioned_win_rate"] == (
        "UNAVAILABLE_AWAITING_TRUE_FORWARD"
    )
    assert report["statistics_semantics"]["themes_are_index_weight_additive"] is False
    assert report["safety"]["synthetic_numeric_score"] is None
    assert report["safety"]["may_call_predictive_edge"] is False
    assert report["safety"]["current_holdings_read_enabled"] is False
    assert report["safety"]["position_mapping_enabled"] is False
    assert report["safety"]["order_generation_enabled"] is False
    assert len(report["economic_sectors"]) == 11
    assert len(report["themes"]) == 10
    for row in report["economic_sectors"] + report["themes"]:
        assert row["score_conditioned_win_rate"] is None
        assert row["conditional_odds"] is None
        assert row["position_mapping_enabled"] is False


def test_markdown_states_the_actual_current_action_and_statistical_boundary() -> None:
    markdown = (ROOT / CONFIG["outputs"]["markdown"]).read_text(encoding="utf-8")

    assert "WAIT_NO_PREDICTIVE_ENTRY" in markdown
    assert "WAIT_NO_NEW_ENTRY" in markdown
    assert "行业结构偏正，但指数预期差" in markdown
    assert "历史胜率为重叠60日窗口描述频率" in markdown
    assert "UNAVAILABLE_AWAITING_TRUE_FORWARD" in markdown
    assert "不自动改变已有持仓" in markdown

