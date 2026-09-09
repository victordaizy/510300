"""生成板块评分卡V1.2双轴决策层。"""

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

from research.industry_sector_scorecard_v1_2 import (  # noqa: E402
    build_dual_axis_scorecard,
    render_markdown,
)


CONFIG_FILE = ROOT / "config" / "industry_sector_scorecard_v1_2.yaml"


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
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    parent = json.loads(
        (ROOT / config["parents"]["v1_1_report"]).read_text(encoding="utf-8")
    )
    source_rows = parent["economic_buckets"] + parent["themes"]
    table = build_dual_axis_scorecard(source_rows, config)
    atomic_parquet(table, ROOT / config["outputs"]["dual_axis_table"])

    economic = table.loc[table["entity_type"].eq("ECONOMIC_BUCKET")]
    themes = table.loc[table["entity_type"].eq("THEME")]
    p1 = table.loc[
        table["research_priority_tier"].eq("P1_DUAL_AXIS_ALIGN_UNCONFIRMED")
    ]
    report = safe_json(
        {
            "version": config["version"],
            "status": "DUAL_AXIS_RESEARCH_PRIORITY_READY",
            "as_of_date": config["as_of_date"],
            "information_cutoff": config["information_cutoff"],
            "supersedes_decision_layer_of": "INDUSTRY_SECTOR_SCORECARD_V1_1",
            "parent_files_changed": False,
            "economic_buckets": economic.to_dict(orient="records"),
            "themes": themes.to_dict(orient="records"),
            "p1_research_watchlist": p1[
                ["entity_type", "entity_id", "entity_name_cn"]
            ].to_dict(orient="records"),
            "tier_counts": table["research_priority_tier"]
            .value_counts()
            .sort_index()
            .to_dict(),
            "synthetic_combined_numeric_score": None,
            "current_score_is_probability": False,
            "historical_odds_are_current_conditional_probability": False,
            "score_conditioned_win_rate": config["governance"][
                "score_conditioned_win_rate"
            ],
            "may_call_predictive_edge": False,
            "position_mapping_enabled": False,
            "methodology": {
                "current_forward_axis": config["current_forward_axis"],
                "historical_odds_axis": config["historical_odds_axis"],
                "decision_matrix": config["decision_matrix"],
                "simplification_rules": config["simplification_rules"],
            },
            "interpretation": (
                "V1.2不再生成综合数值分数。P1仅表示当前前瞻分数与无条件历史赔率点估计方向一致，"
                "但历史期望区间下界尚未转正，因此仍不是预测Edge或交易信号。"
            ),
        }
    )
    atomic_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        ROOT / config["outputs"]["json"],
    )
    atomic_text(render_markdown(report), ROOT / config["outputs"]["markdown"])
    print(
        json.dumps(
            {
                "status": report["status"],
                "p1_research_watchlist": report["p1_research_watchlist"],
                "tier_counts": report["tier_counts"],
                "synthetic_combined_numeric_score": None,
                "position_mapping_enabled": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
