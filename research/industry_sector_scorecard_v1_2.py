"""板块评分卡V1.2：当前前瞻与历史赔率双轴决策层。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


class DualAxisScorecardError(ValueError):
    """双轴评分卡输入或规则不满足冻结定义。"""


def classify_current_axis(
    current_score: float, config: Mapping[str, Any]
) -> tuple[str, bool]:
    """按冻结V1等级边界分类当前前瞻轴。"""

    score = float(current_score)
    if not np.isfinite(score) or not 0.0 <= score <= 100.0:
        raise DualAxisScorecardError(f"当前分数非法：{current_score}")
    labels = config["labels"]
    if score >= float(config["positive_minimum"]):
        return str(labels["positive"]), True
    if score >= float(config["watch_minimum"]):
        return str(labels["watch"]), False
    return str(labels["weak"]), False


def classify_historical_odds(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[str, bool]:
    """用自然盈亏平衡点分类历史赔率，不把它解释为当前概率。"""

    fields = [
        "mean_excess_return_60d",
        "profit_factor",
        "average_payoff_ratio",
        "nonoverlap_cohort_win_rate_min",
        "breakeven_win_rate",
        "block_bootstrap_expectancy_high",
    ]
    values = {field: float(row[field]) for field in fields}
    if not all(np.isfinite(value) for value in values.values()):
        raise DualAxisScorecardError("历史赔率轴存在非有限值")
    if values["block_bootstrap_expectancy_high"] < 0.0:
        return str(config["adverse_label"]), False
    favorable = (
        values["mean_excess_return_60d"] > 0.0
        and values["profit_factor"] > 1.0
        and values["average_payoff_ratio"] > 1.0
        and values["nonoverlap_cohort_win_rate_min"]
        > values["breakeven_win_rate"]
    )
    if favorable:
        return str(config["favorable_label"]), True
    positive_checks = [
        values["mean_excess_return_60d"] > 0.0,
        values["profit_factor"] > 1.0,
        values["average_payoff_ratio"] > 1.0,
        values["nonoverlap_cohort_win_rate_min"]
        > values["breakeven_win_rate"],
    ]
    if any(positive_checks):
        return str(config["mixed_label"]), False
    return str(config["unfavorable_label"]), False


def assign_research_priority(
    current_positive: bool,
    historical_favorable: bool,
    historical_axis: str,
    config: Mapping[str, Any],
) -> tuple[str, str]:
    """应用四级研究矩阵；所有结果均不映射仓位。"""

    if historical_axis == "ADVERSE_IN_SAMPLE_NOT_OOS":
        tier = "P4_HISTORICAL_ADVERSE_IN_SAMPLE"
    elif current_positive and historical_favorable:
        tier = "P1_DUAL_AXIS_ALIGN_UNCONFIRMED"
    elif bool(current_positive) != bool(historical_favorable):
        tier = "P2_ONE_AXIS_ONLY"
    else:
        tier = "P3_NO_POSITIVE_ALIGNMENT"
    return tier, str(config[tier]["action"])


def build_dual_axis_scorecard(
    v1_1_rows: list[Mapping[str, Any]], config: Mapping[str, Any]
) -> pd.DataFrame:
    """构建不含合成数值分数的双轴评分卡。"""

    output: list[dict[str, Any]] = []
    for source in v1_1_rows:
        current_axis, current_positive = classify_current_axis(
            float(source["current_score"]), config["current_forward_axis"]
        )
        historical_axis, historical_favorable = classify_historical_odds(
            source, config["historical_odds_axis"]
        )
        tier, action = assign_research_priority(
            current_positive,
            historical_favorable,
            historical_axis,
            config["decision_matrix"],
        )
        output.append(
            {
                "entity_type": str(source["entity_type"]),
                "entity_id": str(source["entity_id"]),
                "entity_name_cn": str(source["entity_name_cn"]),
                "current_score": float(source["current_score"]),
                "current_grade": str(source["grade"]),
                "source_score_rank": int(source["score_rank"]),
                "current_index_weight": float(source["current_index_weight"]),
                "current_forward_axis": current_axis,
                "current_forward_positive": current_positive,
                "historical_odds_axis": historical_axis,
                "historical_odds_favorable": historical_favorable,
                "research_priority_tier": tier,
                "action_state": action,
                "overlapping_relative_win_rate": float(
                    source["overlapping_relative_win_rate"]
                ),
                "block_bootstrap_win_rate_low": float(
                    source["block_bootstrap_win_rate_low"]
                ),
                "block_bootstrap_win_rate_high": float(
                    source["block_bootstrap_win_rate_high"]
                ),
                "nonoverlap_cohort_win_rate_min": float(
                    source["nonoverlap_cohort_win_rate_min"]
                ),
                "nonoverlap_cohort_win_rate_median": float(
                    source["nonoverlap_cohort_win_rate_median"]
                ),
                "average_payoff_ratio": float(source["average_payoff_ratio"]),
                "breakeven_win_rate": float(source["breakeven_win_rate"]),
                "profit_factor": float(source["profit_factor"]),
                "mean_excess_return_60d": float(
                    source["mean_excess_return_60d"]
                ),
                "block_bootstrap_expectancy_low": float(
                    source["block_bootstrap_expectancy_low"]
                ),
                "block_bootstrap_expectancy_high": float(
                    source["block_bootstrap_expectancy_high"]
                ),
                "independent_wins_per_year": float(
                    source["independent_wins_per_year"]
                ),
                "score_conditioned_win_rate": None,
                "synthetic_combined_numeric_score": None,
                "position_mapping_enabled": False,
            }
        )
    result = pd.DataFrame(output)
    if result.empty or result[["entity_type", "entity_id"]].duplicated().any():
        raise DualAxisScorecardError("双轴评分卡为空或实体重复")
    tier_order = {
        tier: index
        for index, tier in enumerate(config["decision_matrix"]["tier_order"], start=1)
    }
    result["tier_order"] = result["research_priority_tier"].map(tier_order)
    if result["tier_order"].isna().any():
        raise DualAxisScorecardError("出现未注册的研究优先级")
    result = result.sort_values(
        ["entity_type", "tier_order", "current_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
    result["research_priority_rank"] = (
        result.groupby("entity_type").cumcount() + 1
    )
    return result


def render_markdown(report: Mapping[str, Any]) -> str:
    """渲染V1.2双轴研究优先级报告。"""

    lines = [
        "# 板块评分卡 V1.2：当前前瞻与历史赔率双轴",
        "",
        f"- 截止日：`{report['as_of_date']}`",
        f"- 状态：`{report['status']}`",
        "- 不再把当前分数、历史胜率和赔率相加成单一分数。",
        "- P1只是双轴一致的研究观察名单，不是买入评级。",
        "- 高分条件胜率仍为`UNAVAILABLE_AWAITING_TRUE_FORWARD`。",
    ]
    for title, key in (("互斥经济大类", "economic_buckets"), ("重叠主题", "themes")):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| 研究序 | 板块 | 层级 | 当前分 | 当前轴 | 历史赔率轴 | 月胜率 | 块胜率95% | 赔率 | PF | 60日期望 | 期望95% | 状态 |",
                "|---:|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in report[key]:
            lines.append(
                f"| {row['research_priority_rank']} | {row['entity_name_cn']} | "
                f"{row['research_priority_tier']} | {row['current_score']:.1f} | "
                f"{row['current_forward_axis']} | {row['historical_odds_axis']} | "
                f"{row['overlapping_relative_win_rate']:.1%} | "
                f"{row['block_bootstrap_win_rate_low']:.1%}–{row['block_bootstrap_win_rate_high']:.1%} | "
                f"{row['average_payoff_ratio']:.2f} | {row['profit_factor']:.2f} | "
                f"{row['mean_excess_return_60d']:.2%} | "
                f"{row['block_bootstrap_expectancy_low']:.2%}–{row['block_bootstrap_expectancy_high']:.2%} | "
                f"{row['action_state']} |"
            )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 当前轴回答当前证据是否偏正；历史赔率轴只提供无条件历史盈亏结构。",
            "- 双轴一致不等于条件预测优势；历史赔率未经多重比较校正，期望区间下界也未转正。",
            "- 本报告没有合成数值总分、预测概率、Kelly仓位、交易映射或订单。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "DualAxisScorecardError",
    "assign_research_priority",
    "build_dual_axis_scorecard",
    "classify_current_axis",
    "classify_historical_odds",
    "render_markdown",
]
