"""生成510300“可预测才进入”最终决策报告。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.predictable_fish_middle_decision_v1 import (  # noqa: E402
    aggregate_exclusive_sector_states,
    build_reliability_evidence,
    build_sector_statistics,
    evaluate_entry_gates,
    render_markdown,
    validate_parent_hashes,
)


CONFIG_PATH = ROOT / "config" / "510300_predictable_fish_middle_decision_v1.yaml"


def load_json(relative_path: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    actual_parent_hashes = validate_parent_hashes(ROOT, config)
    parents = config["parents"]
    scorecard = load_json(parents["sector_scorecard_report"])
    expectation_gap = load_json(parents["expectation_gap_report"])
    sector_prediction = load_json(parents["sector_prediction_result"])
    driver_episode = load_json(parents["driver_episode_result"])
    daily_quality = load_json(parents["daily_quality"])
    forward_refresh = load_json(parents["forward_input_refresh"])
    etf_readiness = load_json(parents["etf_forward_readiness"])

    economic_rows = sorted(
        scorecard["economic_buckets"],
        key=lambda row: int(row["integrated_research_rank"]),
    )
    theme_rows = sorted(
        scorecard["themes"], key=lambda row: int(row["integrated_research_rank"])
    )
    structure = aggregate_exclusive_sector_states(
        economic_rows, config["exclusive_sector_states"]
    )
    decision = evaluate_entry_gates(
        config,
        scorecard,
        expectation_gap,
        sector_prediction,
        driver_episode,
        daily_quality,
        forward_refresh,
        etf_readiness,
    )
    reliability = build_reliability_evidence(sector_prediction, driver_episode)

    report: dict[str, Any] = {
        "version": config["version"],
        "status": "FINAL_SIMPLIFIED_DECISION_READY",
        "completion_state": "MODEL_LOGIC_COMPLETE_CURRENT_NO_ENTRY",
        "as_of_date": config["as_of_date"],
        "information_cutoff": config["information_cutoff"],
        "asset": config["asset"],
        "original_goal": config["objective"],
        "index_sector_structure": structure,
        "final_decision": decision,
        "economic_sectors": build_sector_statistics(
            economic_rows, include_weight=True
        ),
        "themes": build_sector_statistics(theme_rows, include_weight=False),
        "historical_predictability_evidence": reliability,
        "forward_completion_contract": config["forward_completion_contract"],
        "data_freshness": {
            "daily_quality_status": daily_quality["status"],
            "510300_latest": daily_quality["actual_last_date"],
            "forward_refresh_status": forward_refresh["status"],
            "official_weight_date": forward_refresh["steps"][
                "csindex_weights_and_membership"
            ]["official_weight_date"],
            "constituent_close_date": forward_refresh["steps"][
                "constituent_closes"
            ]["constituent_close_date"],
            "wechat_corpus_status": scorecard["corpus_audit"]["status"],
            "wechat_article_count": scorecard["corpus_audit"]["article_count"],
            "wechat_selected_article_count": scorecard["corpus_audit"][
                "selected_article_count"
            ],
        },
        "statistics_semantics": {
            "descriptive_overlapping_win_rate_is_probability": False,
            "historical_odds_are_current_conditional_odds": False,
            "score_conditioned_win_rate": "UNAVAILABLE_AWAITING_TRUE_FORWARD",
            "independent_frequency_unit": "NONOVERLAPPING_60D_CHAINS_PER_YEAR",
            "themes_are_index_weight_additive": False,
        },
        "safety": {
            "synthetic_numeric_score": None,
            "may_call_predictive_edge": False,
            "current_holdings_read_enabled": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_enabled": False,
        },
        "parent_integrity": actual_parent_hashes,
        "interpretation": (
            "行业结构偏正是研究事实，但当前没有形成指数级预期差，也没有任何鱼中或鱼尾模型取得预测资格。"
            "因此最终简化模型的当前动作是等待而不是进入。"
        ),
    }

    json_path = ROOT / config["outputs"]["json"]
    markdown_path = ROOT / config["outputs"]["markdown"]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), json_path)
    atomic_text(render_markdown(report), markdown_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "completion_state": report["completion_state"],
                "current_research_view": decision["current_research_view"],
                "current_entry_state": decision["current_entry_state"],
                "current_action": decision["current_action"],
                "positive_structure_weight": structure[
                    "positive_structure_weight"
                ],
                "failed_gates": [
                    blocker["gate_id"] for blocker in decision["blockers"]
                ],
                "position_mapping_enabled": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

