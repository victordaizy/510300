"""生成微信公众号前瞻证据与板块三轴整合报告。"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.industry_sector_scorecard_v1_3_wechat import (  # noqa: E402
    build_evidence_frame,
    integrate_forward_axes,
    render_markdown,
    validate_wechat_snapshot,
)


CONFIG_FILE = ROOT / "config" / "industry_sector_scorecard_v1_3_wechat_forward.yaml"


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _parent_rows(report: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(report["economic_buckets"] + report["themes"])


def _price_phases(report: dict[str, Any]) -> pd.DataFrame:
    rows = report["economic_buckets"] + report["themes"]
    return pd.DataFrame(rows)[
        ["entity_type", "entity_id", "entity_name_cn", "price_phase"]
    ].copy()


def _state_entities(frame: pd.DataFrame, state: str) -> list[dict[str, Any]]:
    return frame.loc[
        frame["integrated_forward_state"].eq(state),
        ["entity_type", "entity_id", "entity_name_cn"],
    ].to_dict(orient="records")


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    selected_articles, corpus_audit = validate_wechat_snapshot(config)
    economic_evidence = build_evidence_frame(
        config["economic_bucket_assessments"],
        "ECONOMIC_BUCKET",
        selected_articles,
        config["evidence_rules"],
    )
    theme_evidence = build_evidence_frame(
        config["theme_assessments"],
        "THEME",
        selected_articles,
        config["evidence_rules"],
    )
    evidence = pd.concat([economic_evidence, theme_evidence], ignore_index=True)

    parent_report = json.loads(
        (ROOT / config["parents"]["v1_2_report"]).read_text(encoding="utf-8")
    )
    v1_report = json.loads(
        (ROOT / config["parents"]["v1_scorecard_report"]).read_text(
            encoding="utf-8"
        )
    )
    integrated = integrate_forward_axes(
        _parent_rows(parent_report),
        _price_phases(v1_report),
        evidence,
        config["integration_rules"],
    )

    atomic_parquet(
        selected_articles, ROOT / config["outputs"]["selected_article_audit"]
    )
    atomic_parquet(evidence, ROOT / config["outputs"]["forward_evidence"])
    atomic_parquet(integrated, ROOT / config["outputs"]["integrated_scorecard"])

    source_counts = {
        str(row["account_name"]): int(row["article_count"])
        for row in corpus_audit["sources"]
    }
    largest_source_count = max(source_counts.values())
    report = safe_json(
        {
            "version": config["version"],
            "status": "WECHAT_FORWARD_THREE_AXIS_INTEGRATION_READY",
            "as_of_date": config["as_of_date"],
            "information_cutoff": config["information_cutoff"],
            "supersedes_decision_layer_of": "INDUSTRY_SECTOR_SCORECARD_V1_2",
            "parent_files_changed": False,
            "corpus_audit": {
                "root": config["wechat_snapshot"]["root"],
                "period": corpus_audit["period"],
                "status": corpus_audit["status"],
                "account_count": corpus_audit["observed_account_count"],
                "article_count": corpus_audit["article_count"],
                "archive_complete_count": corpus_audit["archive_complete_count"],
                "archive_partial_count": corpus_audit["archive_partial_count"],
                "selected_article_count": int(len(selected_articles)),
                "selected_source_count": int(selected_articles["source_name"].nunique()),
                "largest_source_article_share": largest_source_count
                / int(corpus_audit["article_count"]),
                "raw_article_count_used_as_vote": False,
            },
            "market_forward_context": config["market_forward_context"],
            "economic_buckets": integrated.loc[
                integrated["entity_type"].eq("ECONOMIC_BUCKET")
            ].to_dict(orient="records"),
            "themes": integrated.loc[
                integrated["entity_type"].eq("THEME")
            ].to_dict(orient="records"),
            "fish_middle_convergence": _state_entities(
                integrated, "FISH_MIDDLE_CONVERGENCE"
            ),
            "qualitative_leads_quant": _state_entities(
                integrated, "QUALITATIVE_LEADS_QUANT"
            ),
            "fish_tail_warnings": _state_entities(
                integrated, "FISH_TAIL_WARNING_NO_CHASE"
            ),
            "negative_convergence": _state_entities(
                integrated, "NEGATIVE_CONVERGENCE"
            ),
            "selected_articles": selected_articles.to_dict(orient="records"),
            "synthetic_numeric_score": None,
            "wechat_direction_is_probability": False,
            "score_conditioned_win_rate": config["governance"][
                "score_conditioned_win_rate"
            ],
            "may_call_predictive_edge": False,
            "position_mapping_enabled": False,
            "interpretation": (
                "V1.3允许经过点时审计的微信公众号证据形成定性前瞻轴，并显式识别定性领先定量和鱼尾风险。"
                "它不把文章数当投票，不生成综合数值分数，也不代理推断国家队持仓。"
            ),
        }
    )
    json_path = ROOT / config["outputs"]["json"]
    markdown_path = ROOT / config["outputs"]["markdown"]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), json_path)
    atomic_text(render_markdown(report), markdown_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "fish_middle_convergence": report["fish_middle_convergence"],
                "qualitative_leads_quant": report["qualitative_leads_quant"],
                "fish_tail_warnings": report["fish_tail_warnings"],
                "national_team_holdings_state": report["market_forward_context"][
                    "national_team_holdings_state"
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
