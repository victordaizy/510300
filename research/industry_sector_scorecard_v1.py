"""板块当前研究评分与历史无条件胜率/频率。

评分不是概率；历史基准率不是评分条件胜率。
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class ScorecardError(ValueError):
    """评分卡输入或定义不满足冻结契约。"""


def wilson_interval(wins: int, total: int, confidence: float = 0.95) -> tuple[float | None, float | None]:
    """二项比例Wilson区间。"""

    if total <= 0:
        return None, None
    if wins < 0 or wins > total:
        raise ScorecardError("胜出次数必须介于0和样本数之间")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    proportion = wins / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def aggregate_forward_history(
    sector_targets: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    entity_type: str,
    minimum_entity_weight: float,
) -> pd.DataFrame:
    """由行业贡献聚合经济大类或主题核心的60日前瞻收益。"""

    required = {
        "date",
        "industry_l1",
        "sector_weight",
        "sector_contribution_60d",
        "snapshot_index_return_60d",
        "target_output",
    }
    missing = sorted(required.difference(sector_targets.columns))
    if missing:
        raise ScorecardError(f"板块目标缺少字段：{missing}")
    data = sector_targets.loc[sector_targets["target_output"].eq("TARGET_READY")].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    numeric = ["sector_weight", "sector_contribution_60d", "snapshot_index_return_60d"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    if data[["date", "industry_l1", *numeric]].isna().any().any():
        raise ScorecardError("成熟板块目标存在关键空值")
    rows: list[dict[str, Any]] = []
    for definition in definitions:
        entity_id = str(definition["entity_id"])
        entity_name = str(definition["entity_name_cn"])
        industries = [str(value) for value in definition["industries"]]
        selected = data.loc[data["industry_l1"].astype(str).isin(industries)].copy()
        for date, group in selected.groupby("date", sort=True):
            market_values = data.loc[data["date"].eq(date), "snapshot_index_return_60d"].unique()
            if len(market_values) != 1:
                raise ScorecardError(f"{date.date()}快照指数收益不唯一")
            entity_weight = float(group["sector_weight"].sum())
            if entity_weight < minimum_entity_weight:
                continue
            contribution = float(group["sector_contribution_60d"].sum())
            entity_return = contribution / entity_weight
            market_return = float(market_values[0])
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "entity_name_cn": entity_name,
                    "entity_weight": entity_weight,
                    "entity_forward_return_60d": entity_return,
                    "market_forward_return_60d": market_return,
                    "excess_forward_return_60d": entity_return - market_return,
                    "relative_win": entity_return > market_return,
                    "absolute_win": entity_return > 0.0,
                    "industry_count": int(group["industry_l1"].nunique()),
                }
            )
    result = pd.DataFrame(rows).sort_values(["date", "entity_id"]).reset_index(drop=True)
    if result.empty:
        raise ScorecardError(f"{entity_type}没有可聚合的历史目标")
    if result[["date", "entity_id"]].duplicated().any():
        raise ScorecardError(f"{entity_type}历史目标存在重复实体日期")
    return result


def summarize_base_rates(
    history: pd.DataFrame,
    confidence: float,
    minimum_observations: int,
    include_top3_frequency: bool,
) -> pd.DataFrame:
    """汇总无条件相对/绝对胜率、次数、频率和收益分布。"""

    data = history.copy()
    if include_top3_frequency:
        data["excess_rank"] = data.groupby("date")["excess_forward_return_60d"].rank(
            method="min", ascending=False
        )
        data["top3"] = data["excess_rank"].le(3)
    else:
        data["top3"] = False
    rows: list[dict[str, Any]] = []
    for (entity_type, entity_id, entity_name), group in data.groupby(
        ["entity_type", "entity_id", "entity_name_cn"], sort=False
    ):
        observations = int(len(group))
        relative_wins = int(group["relative_win"].sum())
        absolute_wins = int(group["absolute_win"].sum())
        interval_low, interval_high = wilson_interval(relative_wins, observations, confidence)
        rows.append(
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "entity_name_cn": entity_name,
                "eligible_monthly_observations": observations,
                "relative_win_count": relative_wins,
                "relative_win_rate": relative_wins / observations if observations else np.nan,
                "relative_win_rate_wilson_low": interval_low,
                "relative_win_rate_wilson_high": interval_high,
                "absolute_win_count": absolute_wins,
                "absolute_win_rate": absolute_wins / observations if observations else np.nan,
                "top3_count": int(group["top3"].sum()) if include_top3_frequency else np.nan,
                "top3_frequency": float(group["top3"].mean()) if include_top3_frequency else np.nan,
                "median_forward_return_60d": float(group["entity_forward_return_60d"].median()),
                "median_excess_return_60d": float(group["excess_forward_return_60d"].median()),
                "mean_excess_return_60d": float(group["excess_forward_return_60d"].mean()),
                "excess_return_q25": float(group["excess_forward_return_60d"].quantile(0.25)),
                "excess_return_q75": float(group["excess_forward_return_60d"].quantile(0.75)),
                "minimum_sample_gate_pass": observations >= minimum_observations,
                "historical_label": "HISTORICALLY_CONTAMINATED_UNCONDITIONAL_BASE_RATE",
            }
        )
    return pd.DataFrame(rows)


def _weighted_average(
    frame: pd.DataFrame, value_column: str, weight_column: str = "sector_weight"
) -> float | None:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    weights = pd.to_numeric(frame[weight_column], errors="coerce")
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any() or float(weights.loc[valid].sum()) <= 0:
        return None
    return float(np.average(values.loc[valid], weights=weights.loc[valid]))


def build_current_entity_inputs(
    current_report: Mapping[str, Any],
    definitions: Sequence[Mapping[str, Any]],
    entity_type: str,
    phase_by_entity: Mapping[str, str],
) -> pd.DataFrame:
    """由当前行业证据聚合评分所需输入。"""

    industries = pd.DataFrame(current_report["industry_rows"]).copy()
    industries["sector_weight"] = pd.to_numeric(industries["sector_weight"], errors="coerce")
    rows: list[dict[str, Any]] = []
    report_entities = {
        str(row["bucket_id"] if entity_type == "ECONOMIC_BUCKET" else row["theme_id"]): row
        for row in current_report[
            "economic_buckets" if entity_type == "ECONOMIC_BUCKET" else "themes"
        ]
    }
    for definition in definitions:
        entity_id = str(definition["entity_id"])
        members = [str(value) for value in definition["industries"]]
        selected = industries.loc[industries["industry_l1"].astype(str).isin(members)].copy()
        if selected.empty:
            continue
        entity_weight = float(selected["sector_weight"].sum())
        gap_scores = pd.to_numeric(selected["expectation_gap_score"], errors="coerce")
        gap_valid = gap_scores.notna()
        gap_observed_weight = float(selected.loc[gap_valid, "sector_weight"].sum())
        gap_score = (
            float(
                np.average(
                    gap_scores.loc[gap_valid],
                    weights=selected.loc[gap_valid, "sector_weight"],
                )
            )
            if gap_valid.any() and gap_observed_weight > 0
            else 0.0
        )
        report_entity = report_entities[entity_id]
        rows.append(
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "entity_name_cn": str(definition["entity_name_cn"]),
                "current_index_weight": entity_weight,
                "fundamental_60d_score": _weighted_average(selected, "fundamental_60d_score"),
                "expectation_gap_score": gap_score,
                "expectation_coverage_ratio": gap_observed_weight / entity_weight if entity_weight > 0 else 0.0,
                "price_phase": phase_by_entity.get(entity_id, "PRICE_PHASE_UNOBSERVED"),
                "flow_intensity_5d": report_entity.get("flow_intensity_5d"),
                "weighted_earnings_yield": _weighted_average(selected, "weighted_earnings_yield"),
            }
        )
    return pd.DataFrame(rows)


def score_current_entities(
    inputs: pd.DataFrame, score_config: Mapping[str, Any]
) -> pd.DataFrame:
    """按冻结五维公式计算0–100当前研究评分。"""

    data = inputs.copy()
    component = score_config["components"]
    fundamental = pd.to_numeric(data["fundamental_60d_score"], errors="coerce").fillna(0.0)
    data["fundamental_points"] = (
        (fundamental.clip(-2.0, 2.0) + 2.0) / 4.0
        * float(component["fundamental"]["maximum_points"])
    )
    gap = pd.to_numeric(data["expectation_gap_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    coverage = pd.to_numeric(data["expectation_coverage_ratio"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    data["expectation_gap_points"] = 12.5 + 12.5 * gap * coverage
    phase_points = {
        str(name): float(value)
        for name, value in component["price_phase"]["phase_points"].items()
    }
    data["price_phase_points"] = data["price_phase"].map(phase_points).fillna(0.0)
    flow = pd.to_numeric(data["flow_intensity_5d"], errors="coerce")
    valuation = pd.to_numeric(data["weighted_earnings_yield"], errors="coerce")
    data["flow_points"] = (
        flow.rank(pct=True, method="average", na_option="bottom")
        * float(component["flow"]["maximum_points"])
    )
    data["valuation_points"] = (
        valuation.rank(pct=True, method="average", na_option="bottom")
        * float(component["valuation"]["maximum_points"])
    )
    point_columns = [
        "fundamental_points",
        "expectation_gap_points",
        "price_phase_points",
        "flow_points",
        "valuation_points",
    ]
    data["current_score"] = data[point_columns].sum(axis=1).clip(0.0, 100.0)

    grades = score_config["grades"]
    def grade(value: float) -> str:
        for name in ("A", "B", "C", "D", "E"):
            lower, upper = map(float, grades[name])
            if name == "A" and lower <= value <= upper:
                return name
            if lower <= value < upper:
                return name
        raise ScorecardError(f"评分{value}不能映射等级")
    data["grade"] = data["current_score"].map(grade)
    data["score_rank"] = data["current_score"].rank(method="min", ascending=False).astype(int)
    data["score_is_probability"] = False
    data["score_conditioned_win_rate"] = np.nan
    data["score_conditioned_win_rate_state"] = "UNAVAILABLE_AWAITING_TRUE_FORWARD"
    return data.sort_values(["current_score", "entity_id"], ascending=[False, True]).reset_index(drop=True)


def render_markdown(report: Mapping[str, Any]) -> str:
    """渲染评分、胜率与频率报告。"""

    lines = [
        "# 板块评分、胜率与频率 V1",
        "",
        f"- 截止日：`{report['as_of_date']}`",
        f"- 状态：`{report['status']}`",
        "- 当前评分不是概率；历史胜率是同板块无条件基准率。",
        f"- 历史成熟月末：`{report['historical_sample']['mature_monthly_observations']}`；"
        f"60日等效非重叠时间块：`{report['historical_sample']['effective_nonoverlapping_blocks']}`。",
        f"- 既有S1模型方向准确率：`{report['failed_model_reference']['direction_accuracy']:.2%}`，"
        "已冻结失败。",
        "",
        "## 互斥经济大类",
        "",
        "| 排名 | 板块 | 分数 | 等级 | 权重 | 历史相对胜率 | 95%区间 | 胜出频率 | 绝对胜率 | 前3频率 | 超额中位数 |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["economic_buckets"]:
        lines.append(
            f"| {row['score_rank']} | {row['entity_name_cn']} | {row['current_score']:.1f} | "
            f"{row['grade']} | {row['current_index_weight']:.2%} | {row['relative_win_rate']:.1%} | "
            f"{row['relative_win_rate_wilson_low']:.1%}–{row['relative_win_rate_wilson_high']:.1%} | "
            f"{row['relative_win_count']}/{row['eligible_monthly_observations']} | "
            f"{row['absolute_win_rate']:.1%} | {row['top3_frequency']:.1%} | "
            f"{row['median_excess_return_60d']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 重叠主题链（仅核心行业）",
            "",
            "| 排名 | 主题 | 分数 | 等级 | 核心权重 | 历史相对胜率 | 95%区间 | 胜出频率 | 绝对胜率 | 超额中位数 |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["themes"]:
        lines.append(
            f"| {row['score_rank']} | {row['entity_name_cn']} | {row['current_score']:.1f} | "
            f"{row['grade']} | {row['current_index_weight']:.2%} | {row['relative_win_rate']:.1%} | "
            f"{row['relative_win_rate_wilson_low']:.1%}–{row['relative_win_rate_wilson_high']:.1%} | "
            f"{row['relative_win_count']}/{row['eligible_monthly_observations']} | "
            f"{row['absolute_win_rate']:.1%} | {row['median_excess_return_60d']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 胜出频率按月末信号统计，60日窗口相互重叠，不等于独立重复试验。",
            "- 主题彼此重叠，主题胜率和频率不能相加，也不能与互斥经济大类直接比较容量。",
            "- 当前评分的条件胜率为`UNAVAILABLE_AWAITING_TRUE_FORWARD`；只有冻结后的真实前瞻样本成熟后才能计算。",
            "- 本报告不生成仓位、订单或交易授权。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "ScorecardError",
    "aggregate_forward_history",
    "build_current_entity_inputs",
    "render_markdown",
    "score_current_entities",
    "summarize_base_rates",
    "wilson_interval",
]

