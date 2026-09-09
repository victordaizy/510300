"""生成板块当前评分、历史无条件胜率和频率。"""

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

from research.industry_sector_scorecard_v1 import (  # noqa: E402
    aggregate_forward_history,
    build_current_entity_inputs,
    render_markdown,
    score_current_entities,
    summarize_base_rates,
)


CONFIG_FILE = ROOT / "config" / "industry_sector_scorecard_v1.yaml"


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


def _compound(values: pd.Series, window: int) -> pd.Series:
    return (1.0 + pd.to_numeric(values, errors="coerce")).rolling(
        window, min_periods=window
    ).apply(np.prod, raw=True) - 1.0


def _phase_from_history(
    industry_daily: pd.DataFrame,
    industries: list[str],
    as_of: pd.Timestamp,
    forward_rules: dict[str, Any],
) -> str:
    selected = industry_daily.loc[
        industry_daily["industry_l1"].astype(str).isin(industries)
    ].copy()
    rows: list[dict[str, Any]] = []
    for date, group in selected.groupby("date", sort=True):
        coverage = float(pd.to_numeric(group["return_coverage_weight"], errors="coerce").sum())
        contribution = pd.to_numeric(
            group["weighted_return_contribution_1d"], errors="coerce"
        ).sum(min_count=1)
        rows.append(
            {
                "date": pd.Timestamp(date),
                "return_1d": float(contribution / coverage)
                if coverage > 0 and pd.notna(contribution)
                else np.nan,
            }
        )
    history = pd.DataFrame(rows).sort_values("date")
    history = history.loc[history["date"].le(as_of)].copy()
    for window in (5, 20, 60):
        history[f"return_{window}d"] = _compound(history["return_1d"], window)
    current = history.iloc[-1]
    rolling60 = history["return_60d"].dropna()
    if rolling60.empty or pd.isna(current["return_60d"]):
        return "PRICE_PHASE_UNOBSERVED"
    percentile = float(rolling60.le(float(current["return_60d"])).mean())
    early_max = float(forward_rules["early_recovery_maximum_60d_history_percentile"])
    extended_min = float(forward_rules["extended_minimum_60d_history_percentile"])
    if percentile >= extended_min:
        return "LATE_OR_EXTENDED"
    if percentile <= early_max and float(current["return_5d"]) > 0:
        return "EARLY_RECOVERY_FROM_WEAK_BASE"
    if percentile <= early_max:
        return "WEAK_BASE_NOT_TURNED"
    if float(current["return_20d"]) > 0:
        return "MIDDLE_TREND"
    return "MIXED_OR_SOFT"


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


def main() -> int:
    config = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
    parent_paths = config["parents"]
    current_report = json.loads(
        (ROOT / parent_paths["current_sector_report"]).read_text(encoding="utf-8")
    )
    failed_model = json.loads(
        (ROOT / parent_paths["failed_sector_prediction_result"]).read_text(encoding="utf-8")
    )
    if failed_model.get("status") != "HISTORICAL_REJECTED_FROZEN":
        raise RuntimeError("既有S1模型状态变化，评分卡必须重新审计")
    v2_config = yaml.safe_load(
        (ROOT / parent_paths["sector_taxonomy_config"]).read_text(encoding="utf-8")
    )
    taxonomy = v2_config["taxonomy"]
    as_of = pd.Timestamp(config["as_of_date"])
    targets = pd.read_parquet(ROOT / config["inputs"]["frozen_sector_target_60d"])
    industry_daily = pd.read_parquet(ROOT / config["inputs"]["industry_daily"])
    industry_daily["date"] = pd.to_datetime(industry_daily["date"], errors="coerce")
    data_status = json.loads(
        (ROOT / config["inputs"]["sector_data_status"]).read_text(encoding="utf-8")
    )

    bucket_definitions = _definitions(current_report, taxonomy, "ECONOMIC_BUCKET")
    theme_definitions = _definitions(current_report, taxonomy, "THEME")
    historical_rules = config["historical_reference"]
    minimum_weight = float(historical_rules["minimum_entity_weight"])
    bucket_history = aggregate_forward_history(
        targets, bucket_definitions, "ECONOMIC_BUCKET", minimum_weight
    )
    theme_history = aggregate_forward_history(
        targets, theme_definitions, "THEME", minimum_weight
    )
    bucket_history_path = ROOT / config["outputs"]["economic_bucket_history"]
    theme_history_path = ROOT / config["outputs"]["theme_history"]
    atomic_parquet(bucket_history, bucket_history_path)
    atomic_parquet(theme_history, theme_history_path)

    confidence = float(historical_rules["wilson_interval_confidence"])
    minimum_observations = int(historical_rules["minimum_observations"])
    bucket_rates = summarize_base_rates(
        bucket_history, confidence, minimum_observations, True
    )
    theme_rates = summarize_base_rates(
        theme_history, confidence, minimum_observations, False
    )
    base_rates = pd.concat([bucket_rates, theme_rates], ignore_index=True)
    base_rates_path = ROOT / config["outputs"]["historical_base_rates"]
    atomic_parquet(base_rates, base_rates_path)

    bucket_phases = {
        str(row["bucket_id"]): str(row["price_phase"])
        for row in current_report["qualitative_forward_priorities"]
    }
    forward_rules = v2_config["rules"]["qualitative_forward_rules"]
    theme_phases = {
        definition["entity_id"]: _phase_from_history(
            industry_daily,
            definition["industries"],
            as_of,
            forward_rules,
        )
        for definition in theme_definitions
    }
    bucket_inputs = build_current_entity_inputs(
        current_report, bucket_definitions, "ECONOMIC_BUCKET", bucket_phases
    )
    theme_inputs = build_current_entity_inputs(
        current_report, theme_definitions, "THEME", theme_phases
    )
    bucket_scores = score_current_entities(bucket_inputs, config["current_score"])
    theme_scores = score_current_entities(theme_inputs, config["current_score"])
    bucket_scorecard = bucket_scores.merge(
        bucket_rates, on=["entity_type", "entity_id", "entity_name_cn"], validate="one_to_one"
    )
    theme_scorecard = theme_scores.merge(
        theme_rates, on=["entity_type", "entity_id", "entity_name_cn"], validate="one_to_one"
    )

    mature_months = int(targets.loc[targets["target_output"].eq("TARGET_READY"), "date"].nunique())
    if mature_months != int(data_status["snapshot_count"]) - 3:
        # 数据状态含3个尚未成熟的最近月末；只作为一致性提示，不改变目标文件事实。
        expected_note = "TARGET_FILE_MATURE_COUNT_IS_AUTHORITATIVE"
    else:
        expected_note = "MATCHES_DATA_STATUS_MINUS_UNMATURE_SNAPSHOTS"
    report = {
        "version": config["version"],
        "status": "CURRENT_SCORE_READY_HISTORICAL_BASE_RATE_READY_NOT_MODEL_PROBABILITY",
        "as_of_date": config["as_of_date"],
        "information_cutoff": config["information_cutoff"],
        "economic_buckets": bucket_scorecard.to_dict(orient="records"),
        "themes": theme_scorecard.to_dict(orient="records"),
        "historical_sample": {
            "mature_monthly_observations": mature_months,
            "first_signal_date": pd.to_datetime(targets["date"]).min().date().isoformat(),
            "last_mature_signal_date": pd.to_datetime(
                targets.loc[targets["target_output"].eq("TARGET_READY"), "date"]
            ).max().date().isoformat(),
            "effective_nonoverlapping_blocks": int(
                historical_rules["effective_nonoverlapping_60d_blocks"]
            ),
            "maturity_consistency_note": expected_note,
            "historical_label": historical_rules["historical_label"],
        },
        "definitions": {
            "score": config["current_score"],
            "historical_reference": historical_rules,
            "score_conditioned_win_rate": "UNAVAILABLE_AWAITING_TRUE_FORWARD",
        },
        "failed_model_reference": {
            "status": failed_model["status"],
            "direction_accuracy": failed_model["evaluation"]["direction_accuracy"],
            "spearman": failed_model["evaluation"]["spearman"],
            "reuse_allowed": False,
        },
        "governance": config["governance"],
        "interpretation": (
            "当前分数用于同一截面研究排序；历史胜率和频率是板块自身60日无条件基准率，"
            "不是分数的预测准确率。60日月末样本相互重叠，必须结合Wilson区间和24个等效非重叠时间块阅读。"
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
                "mature_monthly_observations": mature_months,
                "top_economic_buckets": [
                    {
                        "name": row["entity_name_cn"],
                        "score": row["current_score"],
                        "relative_win_rate": row["relative_win_rate"],
                        "wins": row["relative_win_count"],
                        "observations": row["eligible_monthly_observations"],
                    }
                    for row in report["economic_buckets"][:5]
                ],
                "top_themes": [
                    {
                        "name": row["entity_name_cn"],
                        "score": row["current_score"],
                        "relative_win_rate": row["relative_win_rate"],
                    }
                    for row in report["themes"][:5]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

