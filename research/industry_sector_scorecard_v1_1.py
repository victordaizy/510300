"""板块评分卡V1.1：修正重叠样本依赖并补充赔率和期望值。"""

from __future__ import annotations

import math
import zlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class OddsAnalysisError(ValueError):
    """赔率或依赖修正输入不满足冻结定义。"""


def aggregate_forward_history_with_maturity(
    sector_targets: pd.DataFrame,
    definitions: Sequence[Mapping[str, Any]],
    entity_type: str,
    minimum_entity_weight: float,
) -> pd.DataFrame:
    """聚合行业贡献，同时保留真实进入日和成熟日。"""

    required = {
        "date",
        "entry_date",
        "maturity_date",
        "industry_l1",
        "sector_weight",
        "sector_contribution_60d",
        "snapshot_index_return_60d",
        "target_output",
    }
    missing = sorted(required.difference(sector_targets.columns))
    if missing:
        raise OddsAnalysisError(f"60日板块目标缺少字段：{missing}")
    data = sector_targets.loc[sector_targets["target_output"].eq("TARGET_READY")].copy()
    for column in ("date", "entry_date", "maturity_date"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    numeric = ["sector_weight", "sector_contribution_60d", "snapshot_index_return_60d"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    if data[["date", "entry_date", "maturity_date", "industry_l1", *numeric]].isna().any().any():
        raise OddsAnalysisError("成熟60日目标存在关键空值")
    rows: list[dict[str, Any]] = []
    for definition in definitions:
        industries = [str(value) for value in definition["industries"]]
        selected = data.loc[data["industry_l1"].astype(str).isin(industries)].copy()
        for date, group in selected.groupby("date", sort=True):
            entity_weight = float(group["sector_weight"].sum())
            if entity_weight < minimum_entity_weight:
                continue
            entry_values = group["entry_date"].unique()
            maturity_values = group["maturity_date"].unique()
            market_values = group["snapshot_index_return_60d"].unique()
            if len(entry_values) != 1 or len(maturity_values) != 1 or len(market_values) != 1:
                raise OddsAnalysisError(f"{date.date()}目标日期或市场收益不唯一")
            contribution = float(group["sector_contribution_60d"].sum())
            entity_return = contribution / entity_weight
            market_return = float(market_values[0])
            excess = entity_return - market_return
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "entry_date": pd.Timestamp(entry_values[0]),
                    "maturity_date": pd.Timestamp(maturity_values[0]),
                    "entity_type": entity_type,
                    "entity_id": str(definition["entity_id"]),
                    "entity_name_cn": str(definition["entity_name_cn"]),
                    "entity_weight": entity_weight,
                    "entity_forward_return_60d": entity_return,
                    "market_forward_return_60d": market_return,
                    "excess_forward_return_60d": excess,
                    "relative_win": excess > 0.0,
                    "absolute_win": entity_return > 0.0,
                }
            )
    result = pd.DataFrame(rows).sort_values(["entity_id", "date"]).reset_index(drop=True)
    if result.empty or result[["date", "entity_type", "entity_id"]].duplicated().any():
        raise OddsAnalysisError("聚合后的板块历史为空或存在重复实体日期")
    return result


def moving_block_bootstrap(
    excess_returns: np.ndarray,
    repetitions: int,
    block_length: int,
    random_seed: int,
) -> dict[str, float]:
    """对连续月末超额收益做固定长度移动块Bootstrap。"""

    values = np.asarray(excess_returns, dtype=float)
    if values.ndim != 1 or len(values) < block_length or not np.isfinite(values).all():
        raise OddsAnalysisError("移动块Bootstrap输入非法或样本不足")
    starts = np.arange(0, len(values) - block_length + 1, dtype=int)
    blocks_needed = math.ceil(len(values) / block_length)
    rng = np.random.default_rng(random_seed)
    win_rates = np.empty(repetitions, dtype=float)
    expectancies = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sampled = np.concatenate(
            [values[start : start + block_length] for start in sampled_starts]
        )[: len(values)]
        win_rates[index] = float(np.mean(sampled > 0.0))
        expectancies[index] = float(np.mean(sampled))
    return {
        "win_rate_low": float(np.quantile(win_rates, 0.025)),
        "win_rate_median": float(np.quantile(win_rates, 0.5)),
        "win_rate_high": float(np.quantile(win_rates, 0.975)),
        "expectancy_low": float(np.quantile(expectancies, 0.025)),
        "expectancy_median": float(np.quantile(expectancies, 0.5)),
        "expectancy_high": float(np.quantile(expectancies, 0.975)),
    }


def build_nonoverlap_cohorts(
    history: pd.DataFrame, starting_offsets: Sequence[int]
) -> pd.DataFrame:
    """从不同起点构造进入日严格晚于上次成熟日的非重叠链。"""

    rows: list[pd.DataFrame] = []
    for (entity_type, entity_id), group in history.groupby(
        ["entity_type", "entity_id"], sort=False
    ):
        ordered = group.sort_values("entry_date").reset_index(drop=True)
        for offset in starting_offsets:
            selected_indices: list[int] = []
            previous_maturity: pd.Timestamp | None = None
            for index, row in ordered.iloc[int(offset) :].iterrows():
                entry = pd.Timestamp(row["entry_date"])
                if previous_maturity is None or entry > previous_maturity:
                    selected_indices.append(index)
                    previous_maturity = pd.Timestamp(row["maturity_date"])
            if not selected_indices:
                continue
            selected = ordered.loc[selected_indices].copy()
            selected["cohort_offset"] = int(offset)
            selected["cohort_sequence"] = np.arange(1, len(selected) + 1)
            rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    for _, group in result.groupby(["entity_type", "entity_id", "cohort_offset"]):
        ordered = group.sort_values("entry_date")
        prior_maturity = ordered["maturity_date"].shift(1)
        if not ordered.loc[prior_maturity.notna(), "entry_date"].gt(
            prior_maturity.loc[prior_maturity.notna()]
        ).all():
            raise OddsAnalysisError("非重叠链仍存在持有期交叠")
    return result.sort_values(
        ["entity_type", "entity_id", "cohort_offset", "entry_date"]
    ).reset_index(drop=True)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def summarize_odds_and_dependence(
    history: pd.DataFrame,
    cohorts: pd.DataFrame,
    bootstrap_config: Mapping[str, Any],
    interpretation_gates: Mapping[str, Any],
) -> pd.DataFrame:
    """汇总依赖修正胜率、赔率、期望值和独立机会频率。"""

    repetitions = int(bootstrap_config["repetitions"])
    block_length = int(bootstrap_config["block_length_months"])
    base_seed = int(bootstrap_config["random_seed"])
    rows: list[dict[str, Any]] = []
    for (entity_type, entity_id, entity_name), group in history.groupby(
        ["entity_type", "entity_id", "entity_name_cn"], sort=False
    ):
        ordered = group.sort_values("date")
        excess = ordered["excess_forward_return_60d"].to_numpy(dtype=float)
        positive = excess[excess > 0.0]
        nonpositive = excess[excess <= 0.0]
        if len(positive) == 0 or len(nonpositive) == 0:
            raise OddsAnalysisError(f"{entity_id}无法同时形成盈利和亏损赔率样本")
        average_win = float(np.mean(positive))
        average_loss = float(abs(np.mean(nonpositive)))
        median_win = float(np.median(positive))
        median_loss = float(abs(np.median(nonpositive)))
        payoff_ratio = _safe_ratio(average_win, average_loss)
        median_payoff_ratio = _safe_ratio(median_win, median_loss)
        profit_factor = _safe_ratio(float(positive.sum()), float(abs(nonpositive.sum())))
        win_rate = float(np.mean(excess > 0.0))
        breakeven = 1.0 / (1.0 + payoff_ratio) if payoff_ratio is not None else None
        expectancy = float(np.mean(excess))
        identity_expectancy = win_rate * average_win - (1.0 - win_rate) * average_loss
        entity_seed = base_seed + zlib.crc32(f"{entity_type}|{entity_id}".encode("utf-8"))
        bootstrap = moving_block_bootstrap(
            excess, repetitions, block_length, entity_seed
        )

        entity_cohorts = cohorts.loc[
            cohorts["entity_type"].eq(entity_type)
            & cohorts["entity_id"].eq(entity_id)
        ]
        cohort_rows: list[dict[str, Any]] = []
        for offset, cohort in entity_cohorts.groupby("cohort_offset", sort=True):
            cohort_rows.append(
                {
                    "offset": int(offset),
                    "observations": int(len(cohort)),
                    "wins": int(cohort["relative_win"].sum()),
                    "win_rate": float(cohort["relative_win"].mean()),
                    "mean_excess": float(cohort["excess_forward_return_60d"].mean()),
                }
            )
        if len(cohort_rows) != 3:
            raise OddsAnalysisError(f"{entity_id}没有形成三条非重叠链")
        cohort_rates = np.array([row["win_rate"] for row in cohort_rows], dtype=float)
        cohort_counts = np.array([row["observations"] for row in cohort_rows], dtype=float)
        cohort_wins = np.array([row["wins"] for row in cohort_rows], dtype=float)
        elapsed_years = max(
            (ordered["maturity_date"].max() - ordered["entry_date"].min()).days
            / 365.25,
            1.0 / 365.25,
        )
        robust_gate = (
            bootstrap["expectancy_low"] > 0.0
            and breakeven is not None
            and float(cohort_rates.min()) > breakeven
        )
        rows.append(
            {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "entity_name_cn": entity_name,
                "eligible_monthly_observations": int(len(ordered)),
                "overlapping_relative_win_count": int((excess > 0.0).sum()),
                "overlapping_relative_win_rate": win_rate,
                "block_bootstrap_win_rate_low": bootstrap["win_rate_low"],
                "block_bootstrap_win_rate_median": bootstrap["win_rate_median"],
                "block_bootstrap_win_rate_high": bootstrap["win_rate_high"],
                "nonoverlap_cohort_win_rate_median": float(np.median(cohort_rates)),
                "nonoverlap_cohort_win_rate_min": float(cohort_rates.min()),
                "nonoverlap_cohort_win_rate_max": float(cohort_rates.max()),
                "nonoverlap_cohort_observations_median": float(np.median(cohort_counts)),
                "nonoverlap_cohort_wins_median": float(np.median(cohort_wins)),
                "independent_opportunities_per_year": float(np.median(cohort_counts) / elapsed_years),
                "independent_wins_per_year": float(np.median(cohort_wins) / elapsed_years),
                "average_excess_win": average_win,
                "average_excess_loss": average_loss,
                "average_payoff_ratio": payoff_ratio,
                "median_excess_win": median_win,
                "median_excess_loss": median_loss,
                "median_payoff_ratio": median_payoff_ratio,
                "profit_factor": profit_factor,
                "breakeven_win_rate": breakeven,
                "win_rate_minus_breakeven": win_rate - breakeven if breakeven is not None else None,
                "mean_excess_return_60d": expectancy,
                "expected_value_identity": identity_expectancy,
                "expected_value_identity_error": abs(expectancy - identity_expectancy),
                "block_bootstrap_expectancy_low": bootstrap["expectancy_low"],
                "block_bootstrap_expectancy_median": bootstrap["expectancy_median"],
                "block_bootstrap_expectancy_high": bootstrap["expectancy_high"],
                "excess_return_q10": float(np.quantile(excess, 0.10)),
                "excess_return_q90": float(np.quantile(excess, 0.90)),
                "worst_excess_return": float(np.min(excess)),
                "best_excess_return": float(np.max(excess)),
                "historical_robustness_state": (
                    "HISTORICAL_EXPECTANCY_ROBUST_TO_DEFINED_GATES"
                    if robust_gate
                    else "HISTORICAL_EXPECTANCY_NOT_CONFIRMED"
                ),
                "may_call_predictive_edge": bool(
                    interpretation_gates.get("may_call_predictive_edge", False)
                ),
                "cohort_details": cohort_rows,
            }
        )
    return pd.DataFrame(rows)


def render_markdown(report: Mapping[str, Any]) -> str:
    """渲染V1.1纠正版。"""

    lines = [
        "# 板块评分卡 V1.1：胜率修正与赔率",
        "",
        f"- 截止日：`{report['as_of_date']}`",
        f"- 历史月末：`{report['sample']['snapshot_start']}` 至 `{report['sample']['snapshot_end']}`；"
        f"真实进入日至成熟日：`{report['sample']['entry_start']}` 至 `{report['sample']['maturity_end']}`",
        f"- 状态：`{report['status']}`",
        "- V1文件未修改；V1.1替代V1的胜率区间和赔率解释。",
        "- 重叠月度胜率只作描述；主区间为3个月移动块Bootstrap。",
        "- 当前评分仍不是概率，高分条件胜率仍不可用。",
        "- 赔率 = 平均正超额 / 平均负超额绝对值；盈亏平衡胜率 = 1 / (1 + 赔率)。",
        "- 所有收益均为理论板块组合相对沪深300的60日毛超额，未扣ETF费率、冲击成本、跟踪误差和税费。",
        "",
        "## 互斥经济大类：修正胜率",
        "",
        "| 排名 | 板块 | 分数 | 重叠月胜率 | 块Bootstrap 95% | 非重叠胜率范围 | 独立机会/年 | 独立胜出/年 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["economic_buckets"]:
        lines.append(
            f"| {row['score_rank']} | {row['entity_name_cn']} | {row['current_score']:.1f} | "
            f"{row['overlapping_relative_win_rate']:.1%} | "
            f"{row['block_bootstrap_win_rate_low']:.1%}–{row['block_bootstrap_win_rate_high']:.1%} | "
            f"{row['nonoverlap_cohort_win_rate_min']:.1%}–{row['nonoverlap_cohort_win_rate_max']:.1%} | "
            f"{row['independent_opportunities_per_year']:.2f} | {row['independent_wins_per_year']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 互斥经济大类：赔率与期望",
            "",
            "| 板块 | 平均赚 | 平均亏 | 赔率 | 盈亏平衡胜率 | Profit Factor | 期望超额 | 期望Bootstrap 95% | 状态 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["economic_buckets"]:
        lines.append(
            f"| {row['entity_name_cn']} | {row['average_excess_win']:.2%} | "
            f"-{row['average_excess_loss']:.2%} | {row['average_payoff_ratio']:.2f} | "
            f"{row['breakeven_win_rate']:.1%} | {row['profit_factor']:.2f} | "
            f"{row['mean_excess_return_60d']:.2%} | "
            f"{row['block_bootstrap_expectancy_low']:.2%}–{row['block_bootstrap_expectancy_high']:.2%} | "
            f"{row['historical_robustness_state']} |"
        )
    lines.extend(
        [
            "",
            "## 互斥经济大类：尾部赔率",
            "",
            "| 板块 | 中位赔率 | 超额Q10 | 超额Q90 | 历史最差 | 历史最好 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["economic_buckets"]:
        lines.append(
            f"| {row['entity_name_cn']} | {row['median_payoff_ratio']:.2f} | "
            f"{row['excess_return_q10']:.2%} | {row['excess_return_q90']:.2%} | "
            f"{row['worst_excess_return']:.2%} | {row['best_excess_return']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 重叠主题：胜率、赔率与期望",
            "",
            "| 排名 | 主题 | 分数 | 块胜率95% | 非重叠胜率范围 | 赔率 | 盈亏平衡胜率 | PF | 期望超额 | 期望95% |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["themes"]:
        lines.append(
            f"| {row['score_rank']} | {row['entity_name_cn']} | {row['current_score']:.1f} | "
            f"{row['block_bootstrap_win_rate_low']:.1%}–{row['block_bootstrap_win_rate_high']:.1%} | "
            f"{row['nonoverlap_cohort_win_rate_min']:.1%}–{row['nonoverlap_cohort_win_rate_max']:.1%} | "
            f"{row['average_payoff_ratio']:.2f} | {row['breakeven_win_rate']:.1%} | "
            f"{row['profit_factor']:.2f} | {row['mean_excess_return_60d']:.2%} | "
            f"{row['block_bootstrap_expectancy_low']:.2%}–{row['block_bootstrap_expectancy_high']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            "- 月度胜出次数仍可描述频率，但不能当作77次独立试验。",
            "- 赔率基于相对沪深300的60日历史毛超额，不是ETF单次交易止盈/止损比，也不能直接用于仓位。",
            "- 块Bootstrap区间是依赖修正后的历史不确定性区间，不是当前评分触发后的预测概率区间。",
            "- 当前主题定义回看历史，且未做多重比较校正；即使定义门槛通过也不能称为样本外Edge。",
            "- 高分触发频率和高分条件胜率仍为`UNAVAILABLE`，需从冻结后真实前瞻记录积累。",
            "- 禁止Kelly和仓位映射。",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "OddsAnalysisError",
    "aggregate_forward_history_with_maturity",
    "build_nonoverlap_cohorts",
    "moving_block_bootstrap",
    "render_markdown",
    "summarize_odds_and_dependence",
]
