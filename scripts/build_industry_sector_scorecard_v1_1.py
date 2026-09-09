"""生成板块评分卡V1.1依赖修正胜率、赔率与期望值。"""

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

from research.industry_sector_scorecard_v1_1 import (  # noqa: E402
    aggregate_forward_history_with_maturity,
    build_nonoverlap_cohorts,
    render_markdown,
    summarize_odds_and_dependence,
)


CONFIG_FILE = ROOT / "config" / "industry_sector_scorecard_v1_1.yaml"


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _definitions(
    current_report: dict[str, Any], taxonomy: dict[str, Any], entity_type: str
) -> list[dict[str, Any]]:
    if entity_type == "ECONOMIC_BUCKET":
        source = {str(row["id"]): row for row in taxonomy["economic_buckets"]}
        return [
            {
                "entity_id": str(row["bucket_id"]),
                "entity_name_cn": str(row["bucket_name_cn"]),
                "industries": list(source[str(row["bucket_id"])]["industries"]),
            }
            for row in current_report["economic_buckets"]
        ]
    source = {str(row["id"]): row for row in taxonomy["themes"]}
    return [
        {
            "entity_id": str(row["theme_id"]),
            "entity_name_cn": str(row["theme_name_cn"]),
            "industries": list(source[str(row["theme_id"])]["core"]),
        }
        for row in current_report["themes"]
    ]


def _score_frame(v1_report: dict[str, Any], entity_type: str) -> pd.DataFrame:
    source = v1_report["economic_buckets" if entity_type == "ECONOMIC_BUCKET" else "themes"]
    return pd.DataFrame(source)[
        [
            "entity_type",
            "entity_id",
            "entity_name_cn",
            "current_score",
            "grade",
            "score_rank",
            "current_index_weight",
        ]
    ].copy()


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    parents = config["parents"]
    v1_report = json.loads((ROOT / parents["v1_report"]).read_text(encoding="utf-8"))
    current_report = json.loads(
        (ROOT / parents["current_sector_report"]).read_text(encoding="utf-8")
    )
    taxonomy_config = yaml.safe_load(
        (ROOT / parents["taxonomy_config"]).read_text(encoding="utf-8")
    )
    targets = pd.read_parquet(ROOT / parents["frozen_sector_target_60d"])
    definitions_bucket = _definitions(
        current_report, taxonomy_config["taxonomy"], "ECONOMIC_BUCKET"
    )
    definitions_theme = _definitions(
        current_report, taxonomy_config["taxonomy"], "THEME"
    )
    minimum_weight = float(
        yaml.safe_load((ROOT / parents["v1_config"]).read_text(encoding="utf-8"))[
            "historical_reference"
        ]["minimum_entity_weight"]
    )
    bucket_history = aggregate_forward_history_with_maturity(
        targets, definitions_bucket, "ECONOMIC_BUCKET", minimum_weight
    )
    theme_history = aggregate_forward_history_with_maturity(
        targets, definitions_theme, "THEME", minimum_weight
    )
    offsets = [int(value) for value in config["dependence_adjustment"]["nonoverlap_cohorts"]["starting_offsets"]]
    bucket_cohorts = build_nonoverlap_cohorts(bucket_history, offsets)
    theme_cohorts = build_nonoverlap_cohorts(theme_history, offsets)
    bucket_stats = summarize_odds_and_dependence(
        bucket_history,
        bucket_cohorts,
        config["dependence_adjustment"]["moving_block_bootstrap"],
        config["interpretation_gates"],
    )
    theme_stats = summarize_odds_and_dependence(
        theme_history,
        theme_cohorts,
        config["dependence_adjustment"]["moving_block_bootstrap"],
        config["interpretation_gates"],
    )
    bucket_scorecard = _score_frame(v1_report, "ECONOMIC_BUCKET").merge(
        bucket_stats,
        on=["entity_type", "entity_id", "entity_name_cn"],
        validate="one_to_one",
    ).sort_values("score_rank")
    theme_scorecard = _score_frame(v1_report, "THEME").merge(
        theme_stats,
        on=["entity_type", "entity_id", "entity_name_cn"],
        validate="one_to_one",
    ).sort_values("score_rank")

    corrected_stats = pd.concat([bucket_stats, theme_stats], ignore_index=True).drop(
        columns=["cohort_details"]
    )
    atomic_parquet(
        corrected_stats, ROOT / config["outputs"]["corrected_statistics"]
    )
    atomic_parquet(
        bucket_cohorts, ROOT / config["outputs"]["economic_nonoverlap_cohorts"]
    )
    atomic_parquet(
        theme_cohorts, ROOT / config["outputs"]["theme_nonoverlap_cohorts"]
    )

    expectation_identity_max_error = float(
        corrected_stats["expected_value_identity_error"].max()
    )
    if expectation_identity_max_error > 1.0e-12:
        raise RuntimeError(
            f"胜率赔率期望恒等式误差过大：{expectation_identity_max_error}"
        )
    robust_entities = corrected_stats.loc[
        corrected_stats["historical_robustness_state"].eq(
            "HISTORICAL_EXPECTANCY_ROBUST_TO_DEFINED_GATES"
        ),
        ["entity_type", "entity_id", "entity_name_cn"],
    ].to_dict(orient="records")
    report = {
        "version": config["version"],
        "status": "DEPENDENCE_CORRECTED_WIN_RATE_AND_ODDS_READY",
        "as_of_date": config["as_of_date"],
        "information_cutoff": config["information_cutoff"],
        "supersedes_interpretation_of": config["supersedes_interpretation_of"],
        "parent_files_changed": False,
        "economic_buckets": bucket_scorecard.to_dict(orient="records"),
        "themes": theme_scorecard.to_dict(orient="records"),
        "methodology_corrections": [
            "REPLACED_INDEPENDENT_BINOMIAL_WILSON_WITH_3_MONTH_MOVING_BLOCK_BOOTSTRAP",
            "ADDED_THREE_NONOVERLAPPING_ENTRY_TO_MATURITY_COHORTS",
            "ADDED_AVERAGE_AND_MEDIAN_PAYOFF_ODDS",
            "ADDED_BREAKEVEN_WIN_RATE_PROFIT_FACTOR_AND_EXPECTANCY",
            "ADDED_BLOCK_BOOTSTRAP_EXPECTANCY_INTERVAL",
            "RENAMED_MONTHLY_WIN_RATE_AS_OVERLAPPING_DESCRIPTIVE_FREQUENCY",
        ],
        "sample": {
            "snapshot_start": bucket_history["date"].min().date().isoformat(),
            "snapshot_end": bucket_history["date"].max().date().isoformat(),
            "entry_start": bucket_history["entry_date"].min().date().isoformat(),
            "entry_end": bucket_history["entry_date"].max().date().isoformat(),
            "maturity_start": bucket_history["maturity_date"].min().date().isoformat(),
            "maturity_end": bucket_history["maturity_date"].max().date().isoformat(),
            "overlapping_monthly_observations": int(
                corrected_stats["eligible_monthly_observations"].min()
            ),
            "bootstrap_repetitions": int(
                config["dependence_adjustment"]["moving_block_bootstrap"]["repetitions"]
            ),
            "bootstrap_block_length_months": int(
                config["dependence_adjustment"]["moving_block_bootstrap"]["block_length_months"]
            ),
            "nonoverlap_cohort_offsets": offsets,
            "nonoverlap_observations_min": int(
                min(
                    bucket_cohorts.groupby(
                        ["entity_id", "cohort_offset"]
                    ).size().min(),
                    theme_cohorts.groupby(["entity_id", "cohort_offset"])
                    .size()
                    .min(),
                )
            ),
            "nonoverlap_observations_max": int(
                max(
                    bucket_cohorts.groupby(
                        ["entity_id", "cohort_offset"]
                    ).size().max(),
                    theme_cohorts.groupby(["entity_id", "cohort_offset"])
                    .size()
                    .max(),
                )
            ),
            "effective_nonoverlapping_blocks_reference": int(
                config["dependence_adjustment"]["effective_nonoverlapping_blocks_reference"]
            ),
        },
        "robust_historical_entities_under_defined_gates": robust_entities,
        "expectancy_identity_max_error": expectation_identity_max_error,
        "current_score_trigger_frequency": config["frequency"]["current_score_trigger_frequency"],
        "score_conditioned_win_rate": "UNAVAILABLE_AWAITING_TRUE_FORWARD",
        "multiple_comparison_adjusted": False,
        "may_call_predictive_edge": False,
        "odds_implementation_scope": config["odds"]["implementation_scope"],
        "governance": config["governance"],
        "interpretation": (
            "V1.1修正了重叠样本导致的独立性错配，并把胜率与赔率合成为期望值。"
            "所有统计仍是历史污染的描述性基准，不是当前评分的条件概率，也没有经过多重比较校正。"
        ),
    }
    report = _safe(report)
    json_path = ROOT / config["outputs"]["json"]
    markdown_path = ROOT / config["outputs"]["markdown"]
    atomic_text(json.dumps(report, ensure_ascii=False, indent=2), json_path)
    atomic_text(render_markdown(report), markdown_path)
    print(
        json.dumps(
            {
                "status": report["status"],
                "robust_historical_entities_under_defined_gates": robust_entities,
                "top_economic_odds": [
                    {
                        "name": row["entity_name_cn"],
                        "score": row["current_score"],
                        "win_rate": row["overlapping_relative_win_rate"],
                        "block_win_interval": [
                            row["block_bootstrap_win_rate_low"],
                            row["block_bootstrap_win_rate_high"],
                        ],
                        "payoff_ratio": row["average_payoff_ratio"],
                        "breakeven_win_rate": row["breakeven_win_rate"],
                        "expectancy": row["mean_excess_return_60d"],
                        "expectancy_interval": [
                            row["block_bootstrap_expectancy_low"],
                            row["block_bootstrap_expectancy_high"],
                        ],
                    }
                    for row in report["economic_buckets"][:5]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
