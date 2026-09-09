"""510300 驱动周期结构、鱼中候选与鱼尾候选的冻结研究实现。

本模块把周期构造、相位判断和历史否证集中在一个深模块内。调用方只需
提供已经审计的点时归因表与 510300 日线；模块不会产生仓位、订单或券商指令。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


NO_VIEW = "NO_VIEW"
NO_EPISODE = "NO_EPISODE"
EPISODE_OBSERVE_ONLY = "EPISODE_OBSERVE_ONLY"
FORECAST_ELIGIBLE = "FORECAST_ELIGIBLE"

PROPAGATION = "PROPAGATION"
MATURE_CONTINUATION = "MATURE_CONTINUATION"
EXHAUSTION = "EXHAUSTION"
BREAKDOWN = "BREAKDOWN"


@dataclass(frozen=True)
class EpisodeRules:
    """驱动周期确认、相位和终止的冻结规则。"""

    data_cutoff: pd.Timestamp
    confirmation_lookback_days: int
    maximum_core_industries: int
    minimum_core_positive_contribution_share: float
    minimum_core_positive_day_share: float
    minimum_internal_advancer_weight_share: float
    minimum_effective_positive_contributors: float
    maximum_largest_positive_contributor_share: float
    require_positive_index_contribution: bool
    short_lookback_days: int
    minimum_propagation_noncore_advancer_weight_share: float
    minimum_propagation_positive_noncore_industry_share: float
    exhaustion_minimum_hazard_count: int
    exhaustion_core_slowdown_ratio: float
    exhaustion_core_advancer_weight_share: float
    exhaustion_minimum_core_positive_days: int
    exhaustion_maximum_largest_positive_contributor_share: float
    breakdown_minimum_core_positive_contribution_share: float
    breakdown_confirmation_days: int
    maximum_episode_valid_days: int

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "EpisodeRules":
        protocol = contract["protocol"]
        confirmation = contract["episode_confirmation"]
        phase = contract["phase"]
        return cls(
            data_cutoff=pd.Timestamp(protocol["historical_contamination_cutoff"]),
            confirmation_lookback_days=int(confirmation["lookback_valid_days"]),
            maximum_core_industries=int(confirmation["maximum_core_industries"]),
            minimum_core_positive_contribution_share=float(
                confirmation["minimum_core_positive_contribution_share"]
            ),
            minimum_core_positive_day_share=float(
                confirmation["minimum_core_positive_day_share"]
            ),
            minimum_internal_advancer_weight_share=float(
                confirmation["minimum_internal_advancer_weight_share"]
            ),
            minimum_effective_positive_contributors=float(
                confirmation["minimum_effective_positive_contributors"]
            ),
            maximum_largest_positive_contributor_share=float(
                confirmation["maximum_largest_positive_contributor_share"]
            ),
            require_positive_index_contribution=bool(
                confirmation["require_positive_index_contribution"]
            ),
            short_lookback_days=int(phase["short_lookback_valid_days"]),
            minimum_propagation_noncore_advancer_weight_share=float(
                phase["minimum_propagation_noncore_advancer_weight_share"]
            ),
            minimum_propagation_positive_noncore_industry_share=float(
                phase["minimum_propagation_positive_noncore_industry_share"]
            ),
            exhaustion_minimum_hazard_count=int(
                phase["exhaustion_minimum_hazard_count"]
            ),
            exhaustion_core_slowdown_ratio=float(
                phase["exhaustion_core_slowdown_ratio"]
            ),
            exhaustion_core_advancer_weight_share=float(
                phase["exhaustion_core_advancer_weight_share"]
            ),
            exhaustion_minimum_core_positive_days=int(
                phase["exhaustion_minimum_core_positive_days"]
            ),
            exhaustion_maximum_largest_positive_contributor_share=float(
                phase["exhaustion_maximum_largest_positive_contributor_share"]
            ),
            breakdown_minimum_core_positive_contribution_share=float(
                phase["breakdown_minimum_core_positive_contribution_share"]
            ),
            breakdown_confirmation_days=int(phase["breakdown_confirmation_days"]),
            maximum_episode_valid_days=int(phase["maximum_episode_valid_days"]),
        )


@dataclass(frozen=True)
class OutcomeRules:
    """下一交易日开盘起算的冻结未来收益定义。"""

    data_cutoff: pd.Timestamp
    horizon_trading_days: int
    cash_annual_rate: float
    trading_days_per_year: int
    middle_minimum_relative_return: float
    bad20_terminal_threshold: float
    tail20_path_threshold: float

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "OutcomeRules":
        protocol = contract["protocol"]
        targets = contract["targets"]
        return cls(
            data_cutoff=pd.Timestamp(protocol["historical_contamination_cutoff"]),
            horizon_trading_days=int(targets["horizon_trading_days"]),
            cash_annual_rate=float(targets["cash_annual_rate"]),
            trading_days_per_year=int(targets["trading_days_per_year"]),
            middle_minimum_relative_return=float(
                targets["middle_minimum_relative_return"]
            ),
            bad20_terminal_threshold=float(targets["bad20_terminal_threshold"]),
            tail20_path_threshold=float(targets["tail20_path_threshold"]),
        )


@dataclass(frozen=True)
class EvaluationRules:
    """两个固定候选的历史否证规则。"""

    bootstrap_repetitions: int
    bootstrap_event_block_length: int
    random_seed: int
    signal_delay_days: int
    minimum_effect_retention: float
    fish_minimum_events: int
    fish_minimum_lift: float
    fish_bootstrap_lower: float
    fish_minimum_median_x20: float
    fish_minimum_positive_rate: float
    fish_maximum_tail_rate_ratio: float
    fish_split_minimum_lift: float
    tail_minimum_events: int
    tail_minimum_target_events: int
    tail_minimum_lift: float
    tail_bootstrap_lower: float
    tail_split_minimum_lift: float

    @classmethod
    def from_contract(cls, contract: Mapping[str, Any]) -> "EvaluationRules":
        evaluation = contract["evaluation"]
        fish = evaluation["fish_middle"]
        tail = evaluation["tail_warning"]
        return cls(
            bootstrap_repetitions=int(evaluation["bootstrap_repetitions"]),
            bootstrap_event_block_length=int(
                evaluation["bootstrap_event_block_length"]
            ),
            random_seed=int(evaluation["random_seed"]),
            signal_delay_days=int(evaluation["signal_delay_stress_trading_days"]),
            minimum_effect_retention=float(evaluation["minimum_effect_retention"]),
            fish_minimum_events=int(fish["minimum_mature_events"]),
            fish_minimum_lift=float(fish["minimum_middle20_lift"]),
            fish_bootstrap_lower=float(
                fish["bootstrap_lift_lower_bound_strictly_above"]
            ),
            fish_minimum_median_x20=float(fish["minimum_median_x20"]),
            fish_minimum_positive_rate=float(
                fish["minimum_positive_x20_rate_strictly_above"]
            ),
            fish_maximum_tail_rate_ratio=float(
                fish["maximum_tail20_rate_relative_to_baseline"]
            ),
            fish_split_minimum_lift=float(
                fish["split_minimum_lift_strictly_above"]
            ),
            tail_minimum_events=int(tail["minimum_mature_events"]),
            tail_minimum_target_events=int(tail["minimum_tail20_events"]),
            tail_minimum_lift=float(tail["minimum_tail20_lift"]),
            tail_bootstrap_lower=float(
                tail["bootstrap_lift_lower_bound_strictly_above"]
            ),
            tail_split_minimum_lift=float(
                tail["split_minimum_lift_strictly_above"]
            ),
        )


@dataclass(frozen=True)
class EpisodeStructure:
    """逐日状态、周期级记录与每周期一次的候选事件。"""

    daily_states: pd.DataFrame
    episodes: pd.DataFrame
    events: pd.DataFrame


def _require_columns(
    frame: pd.DataFrame, required: set[str], dataset_name: str
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name}缺少字段：{missing}")


def _prepare_inputs(
    cross_section: pd.DataFrame,
    industry_attribution: pd.DataFrame,
    daily_attribution: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _require_columns(
        cross_section,
        {
            "date",
            "con_code",
            "snapshot_weight",
            "constituent_return_1d",
            "industry_l1",
            "return_available",
            "industry_available",
            "component_return_contribution_1d",
        },
        "点时成分截面",
    )
    _require_columns(
        industry_attribution,
        {"date", "industry_l1", "weighted_return_contribution_1d"},
        "行业贡献",
    )
    _require_columns(
        daily_attribution,
        {
            "date",
            "snapshot_weighted_constituent_return_1d",
            "valid_for_attribution",
            "output",
            "failure_category",
        },
        "每日归因摘要",
    )
    cross = cross_section.copy()
    industry = industry_attribution.copy()
    daily = daily_attribution.copy()
    for frame in (cross, industry, daily):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].isna().any():
            raise ValueError("驱动周期输入存在无效日期")
        frame.drop(frame.loc[frame["date"].gt(cutoff)].index, inplace=True)
    if cross[["date", "con_code"]].duplicated().any():
        raise ValueError("点时成分截面存在重复证券日期")
    if industry[["date", "industry_l1"]].duplicated().any():
        raise ValueError("行业贡献存在重复行业日期")
    if daily["date"].duplicated().any():
        raise ValueError("每日归因摘要存在重复日期")
    cross["snapshot_weight"] = pd.to_numeric(
        cross["snapshot_weight"], errors="coerce"
    )
    cross["constituent_return_1d"] = pd.to_numeric(
        cross["constituent_return_1d"], errors="coerce"
    )
    cross["component_return_contribution_1d"] = pd.to_numeric(
        cross["component_return_contribution_1d"], errors="coerce"
    )
    industry["weighted_return_contribution_1d"] = pd.to_numeric(
        industry["weighted_return_contribution_1d"], errors="coerce"
    )
    return (
        cross.sort_values(["date", "con_code"]).reset_index(drop=True),
        industry.sort_values(["date", "industry_l1"]).reset_index(drop=True),
        daily.sort_values("date").reset_index(drop=True),
    )


def _select_candidate_core(
    industry_window: pd.DataFrame, rules: EpisodeRules
) -> tuple[tuple[str, ...], float]:
    cumulative = (
        industry_window.groupby("industry_l1", sort=True)[
            "weighted_return_contribution_1d"
        ]
        .sum(min_count=1)
        .dropna()
        .sort_values(ascending=False)
    )
    positive = cumulative.loc[cumulative.gt(0)]
    positive_total = float(positive.sum())
    if positive.empty or positive_total <= 0:
        return (), np.nan
    selected: list[str] = []
    selected_sum = 0.0
    for industry_name, value in positive.items():
        if len(selected) >= rules.maximum_core_industries:
            break
        selected.append(str(industry_name))
        selected_sum += float(value)
        if (
            selected_sum / positive_total
            >= rules.minimum_core_positive_contribution_share
        ):
            break
    return tuple(selected), selected_sum / positive_total


def _fixed_core_metrics(
    dates: Sequence[pd.Timestamp],
    core: Sequence[str],
    cross_by_date: Mapping[pd.Timestamp, pd.DataFrame],
    industry_by_date: Mapping[pd.Timestamp, pd.DataFrame],
) -> dict[str, float | int]:
    core_set = set(core)
    industry_frames = [industry_by_date[date] for date in dates]
    industry_window = pd.concat(industry_frames, ignore_index=True)
    contribution = industry_window["weighted_return_contribution_1d"]
    core_mask = industry_window["industry_l1"].astype(str).isin(core_set)
    core_contribution = float(contribution.loc[core_mask].sum(min_count=1))
    cumulative_industry = (
        industry_window.groupby("industry_l1", sort=True)[
            "weighted_return_contribution_1d"
        ]
        .sum(min_count=1)
        .dropna()
    )
    positive_total = float(cumulative_industry.clip(lower=0).sum())
    core_positive_share = (
        core_contribution / positive_total if positive_total > 0 else np.nan
    )
    daily_core = (
        industry_window.loc[core_mask]
        .groupby("date")["weighted_return_contribution_1d"]
        .sum(min_count=1)
        .reindex(pd.DatetimeIndex(dates))
    )
    positive_days = int(daily_core.gt(0).sum())

    cross_window = pd.concat([cross_by_date[date] for date in dates], ignore_index=True)
    eligible = (
        cross_window["industry_l1"].astype(str).isin(core_set)
        & cross_window["return_available"].fillna(False).astype(bool)
        & cross_window["industry_available"].fillna(False).astype(bool)
        & cross_window["snapshot_weight"].notna()
        & cross_window["constituent_return_1d"].notna()
    )
    core_cross = cross_window.loc[eligible].copy()
    coverage_weight = float(core_cross["snapshot_weight"].sum())
    advancer_weight = float(
        core_cross.loc[
            core_cross["constituent_return_1d"].gt(0), "snapshot_weight"
        ].sum()
    )
    advancer_share = (
        advancer_weight / coverage_weight if coverage_weight > 0 else np.nan
    )
    positive_contribution = (
        core_cross.assign(
            positive_contribution=core_cross[
                "component_return_contribution_1d"
            ].clip(lower=0)
        )
        .groupby("con_code")["positive_contribution"]
        .sum()
    )
    positive_contribution = positive_contribution.loc[
        positive_contribution.gt(0)
    ]
    positive_security_total = float(positive_contribution.sum())
    if positive_security_total > 0:
        shares = positive_contribution / positive_security_total
        effective_contributors = float(1.0 / np.square(shares).sum())
        largest_share = float(shares.max())
    else:
        effective_contributors = 0.0
        largest_share = np.nan
    return {
        "core_contribution": core_contribution,
        "core_positive_contribution_share": core_positive_share,
        "core_positive_days": positive_days,
        "core_positive_day_share": positive_days / len(dates),
        "core_internal_advancer_weight_share": advancer_share,
        "core_effective_positive_contributors": effective_contributors,
        "core_largest_positive_contributor_share": largest_share,
    }


def _noncore_metrics(
    dates: Sequence[pd.Timestamp],
    core: Sequence[str],
    cross_by_date: Mapping[pd.Timestamp, pd.DataFrame],
    industry_by_date: Mapping[pd.Timestamp, pd.DataFrame],
) -> dict[str, float]:
    core_set = set(core)
    industry_window = pd.concat(
        [industry_by_date[date] for date in dates], ignore_index=True
    )
    noncore_industry = industry_window.loc[
        ~industry_window["industry_l1"].astype(str).isin(core_set)
    ]
    cumulative = (
        noncore_industry.groupby("industry_l1")[
            "weighted_return_contribution_1d"
        ]
        .sum(min_count=1)
        .dropna()
    )
    noncore_contribution = float(cumulative.sum()) if not cumulative.empty else np.nan
    positive_industry_share = (
        float(cumulative.gt(0).mean()) if not cumulative.empty else np.nan
    )
    cross_window = pd.concat([cross_by_date[date] for date in dates], ignore_index=True)
    eligible = (
        ~cross_window["industry_l1"].astype(str).isin(core_set)
        & cross_window["return_available"].fillna(False).astype(bool)
        & cross_window["industry_available"].fillna(False).astype(bool)
        & cross_window["snapshot_weight"].notna()
        & cross_window["constituent_return_1d"].notna()
    )
    noncore_cross = cross_window.loc[eligible]
    coverage = float(noncore_cross["snapshot_weight"].sum())
    advancer = float(
        noncore_cross.loc[
            noncore_cross["constituent_return_1d"].gt(0), "snapshot_weight"
        ].sum()
    )
    return {
        "noncore_contribution": noncore_contribution,
        "noncore_positive_industry_share": positive_industry_share,
        "noncore_advancer_weight_share": advancer / coverage if coverage > 0 else np.nan,
    }


def _confirmation_metrics(
    dates: Sequence[pd.Timestamp],
    candidate_core: Sequence[str],
    daily_by_date: Mapping[pd.Timestamp, pd.Series],
    cross_by_date: Mapping[pd.Timestamp, pd.DataFrame],
    industry_by_date: Mapping[pd.Timestamp, pd.DataFrame],
    rules: EpisodeRules,
) -> tuple[dict[str, Any], list[str]]:
    metrics = _fixed_core_metrics(
        dates, candidate_core, cross_by_date, industry_by_date
    )
    index_contribution = float(
        sum(
            float(daily_by_date[date]["snapshot_weighted_constituent_return_1d"])
            for date in dates
        )
    )
    metrics["index_contribution"] = index_contribution
    failures: list[str] = []
    if not candidate_core:
        failures.append("NO_POSITIVE_CORE_CANDIDATE")
    if len(candidate_core) > rules.maximum_core_industries:
        failures.append("CORE_INDUSTRY_COUNT_ABOVE_GATE")
    if (
        pd.isna(metrics["core_positive_contribution_share"])
        or metrics["core_positive_contribution_share"]
        < rules.minimum_core_positive_contribution_share
    ):
        failures.append("CONTRIBUTION_CONCENTRATION_BELOW_GATE")
    if (
        metrics["core_positive_day_share"]
        < rules.minimum_core_positive_day_share
    ):
        failures.append("IDENTITY_PERSISTENCE_BELOW_GATE")
    if (
        pd.isna(metrics["core_internal_advancer_weight_share"])
        or metrics["core_internal_advancer_weight_share"]
        < rules.minimum_internal_advancer_weight_share
    ):
        failures.append("INTERNAL_BREADTH_BELOW_GATE")
    if (
        metrics["core_effective_positive_contributors"]
        < rules.minimum_effective_positive_contributors
    ):
        failures.append("EFFECTIVE_CONTRIBUTORS_BELOW_GATE")
    if (
        pd.isna(metrics["core_largest_positive_contributor_share"])
        or metrics["core_largest_positive_contributor_share"]
        > rules.maximum_largest_positive_contributor_share
    ):
        failures.append("SINGLE_CONTRIBUTOR_CONCENTRATION_ABOVE_GATE")
    if rules.require_positive_index_contribution and index_contribution <= 0:
        failures.append("INDEX_ACCEPTANCE_NOT_POSITIVE")
    return metrics, failures


def _formation_date(
    dates: Sequence[pd.Timestamp],
    core: Sequence[str],
    industry_by_date: Mapping[pd.Timestamp, pd.DataFrame],
) -> pd.Timestamp:
    core_set = set(core)
    for date in dates:
        frame = industry_by_date[date]
        value = frame.loc[
            frame["industry_l1"].astype(str).isin(core_set),
            "weighted_return_contribution_1d",
        ].sum(min_count=1)
        if pd.notna(value) and float(value) > 0:
            return pd.Timestamp(date)
    return pd.Timestamp(dates[0])


def _phase_and_hazards(
    valid_history: Sequence[pd.Timestamp],
    core: Sequence[str],
    cross_by_date: Mapping[pd.Timestamp, pd.DataFrame],
    industry_by_date: Mapping[pd.Timestamp, pd.DataFrame],
    rules: EpisodeRules,
) -> tuple[str, dict[str, Any], list[str]]:
    short_dates = list(valid_history[-rules.short_lookback_days :])
    short_core = _fixed_core_metrics(
        short_dates, core, cross_by_date, industry_by_date
    )
    noncore = _noncore_metrics(
        short_dates, core, cross_by_date, industry_by_date
    )
    hazards: list[str] = []
    previous_dates = list(
        valid_history[
            -2 * rules.short_lookback_days : -rules.short_lookback_days
        ]
    )
    previous_contribution = np.nan
    if len(previous_dates) == rules.short_lookback_days:
        previous_contribution = float(
            _fixed_core_metrics(
                previous_dates, core, cross_by_date, industry_by_date
            )["core_contribution"]
        )
        if (
            previous_contribution > 0
            and short_core["core_contribution"]
            < previous_contribution * rules.exhaustion_core_slowdown_ratio
        ):
            hazards.append("CORE_CONTRIBUTION_SLOWDOWN")
    if (
        pd.isna(short_core["core_internal_advancer_weight_share"])
        or short_core["core_internal_advancer_weight_share"]
        < rules.exhaustion_core_advancer_weight_share
    ):
        hazards.append("CORE_BREADTH_WEAK")
    if pd.isna(noncore["noncore_contribution"]) or noncore["noncore_contribution"] <= 0:
        hazards.append("NONCORE_CONTRIBUTION_NOT_POSITIVE")
    if short_core["core_positive_days"] < rules.exhaustion_minimum_core_positive_days:
        hazards.append("CORE_POSITIVE_DAYS_INSUFFICIENT")
    if (
        pd.isna(short_core["core_largest_positive_contributor_share"])
        or short_core["core_largest_positive_contributor_share"]
        > rules.exhaustion_maximum_largest_positive_contributor_share
    ):
        hazards.append("CORE_SINGLE_CONTRIBUTOR_CONCENTRATION")

    propagation = (
        short_core["core_contribution"] > 0
        and pd.notna(noncore["noncore_contribution"])
        and noncore["noncore_contribution"] > 0
        and pd.notna(noncore["noncore_advancer_weight_share"])
        and noncore["noncore_advancer_weight_share"]
        >= rules.minimum_propagation_noncore_advancer_weight_share
        and pd.notna(noncore["noncore_positive_industry_share"])
        and noncore["noncore_positive_industry_share"]
        >= rules.minimum_propagation_positive_noncore_industry_share
    )
    if len(hazards) >= rules.exhaustion_minimum_hazard_count:
        phase = EXHAUSTION
    elif propagation:
        phase = PROPAGATION
    else:
        phase = MATURE_CONTINUATION
    metrics = {**short_core, **noncore, "previous_core_contribution": previous_contribution}
    return phase, metrics, hazards


def build_driver_episode_structure(
    cross_section: pd.DataFrame,
    industry_attribution: pd.DataFrame,
    daily_attribution: pd.DataFrame,
    rules: EpisodeRules,
) -> EpisodeStructure:
    """只使用截至当日信息构造驱动周期、相位和每周期一次事件。"""

    cross, industry, daily = _prepare_inputs(
        cross_section, industry_attribution, daily_attribution, rules.data_cutoff
    )
    cross_by_date = {
        pd.Timestamp(date): frame.copy()
        for date, frame in cross.groupby("date", sort=False)
    }
    industry_by_date = {
        pd.Timestamp(date): frame.copy()
        for date, frame in industry.groupby("date", sort=False)
    }
    daily_by_date = {
        pd.Timestamp(row.date): row
        for row in daily.itertuples(index=False)
    }
    daily_series_by_date = {
        pd.Timestamp(date): frame.iloc[0]
        for date, frame in daily.groupby("date", sort=False)
    }

    state_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    valid_history: list[pd.Timestamp] = []
    active: dict[str, Any] | None = None
    episode_counter = 0

    def close_episode(end_date: pd.Timestamp, reason: str) -> None:
        nonlocal active
        if active is None:
            return
        episode_rows.append(
            {
                "episode_id": active["episode_id"],
                "formation_date": active["formation_date"],
                "confirmation_date": active["confirmation_date"],
                "end_date": pd.Timestamp(end_date),
                "core_industries": "|".join(active["core"]),
                "valid_day_count": int(active["age"]),
                "first_propagation_date": active["first_propagation_date"],
                "first_exhaustion_date": active["first_exhaustion_date"],
                "termination_reason": reason,
            }
        )
        active = None

    for date in daily["date"]:
        date = pd.Timestamp(date)
        daily_row = daily_by_date[date]
        is_valid = bool(daily_row.valid_for_attribution) and daily_row.output != NO_VIEW
        if (
            not is_valid
            or date not in cross_by_date
            or date not in industry_by_date
        ):
            if active is not None:
                prior_date = (
                    valid_history[-1]
                    if valid_history
                    else active["confirmation_date"]
                )
                close_episode(pd.Timestamp(prior_date), "NO_VIEW_INTERRUPTION")
            valid_history = []
            state_rows.append(
                {
                    "date": date,
                    "research_output": NO_VIEW,
                    "failure_category": str(daily_row.failure_category),
                    "episode_id": None,
                    "phase": None,
                    "candidate_role": None,
                    "candidate_event": False,
                }
            )
            continue

        valid_history.append(date)
        confirmation_metrics: dict[str, Any] = {}
        confirmation_failures: list[str] = []
        if active is None:
            if len(valid_history) < rules.confirmation_lookback_days:
                state_rows.append(
                    {
                        "date": date,
                        "research_output": NO_EPISODE,
                        "failure_category": "INSUFFICIENT_CONTIGUOUS_VALID_HISTORY",
                        "episode_id": None,
                        "phase": None,
                        "candidate_role": None,
                        "candidate_event": False,
                    }
                )
                continue
            confirmation_dates = valid_history[-rules.confirmation_lookback_days :]
            industry_window = pd.concat(
                [industry_by_date[value] for value in confirmation_dates],
                ignore_index=True,
            )
            candidate_core, candidate_share = _select_candidate_core(
                industry_window, rules
            )
            confirmation_metrics, confirmation_failures = _confirmation_metrics(
                confirmation_dates,
                candidate_core,
                daily_series_by_date,
                cross_by_date,
                industry_by_date,
                rules,
            )
            confirmation_metrics["candidate_selected_positive_share"] = candidate_share
            if confirmation_failures:
                state_rows.append(
                    {
                        "date": date,
                        "research_output": NO_EPISODE,
                        "failure_category": "|".join(confirmation_failures),
                        "episode_id": None,
                        "phase": None,
                        "candidate_core_industries": "|".join(candidate_core),
                        "candidate_role": None,
                        "candidate_event": False,
                        **confirmation_metrics,
                    }
                )
                continue
            episode_counter += 1
            active = {
                "episode_id": f"EP{episode_counter:04d}",
                "core": tuple(candidate_core),
                "formation_date": _formation_date(
                    confirmation_dates, candidate_core, industry_by_date
                ),
                "confirmation_date": date,
                "age": 0,
                "breakdown_count": 0,
                "first_propagation_date": pd.NaT,
                "first_exhaustion_date": pd.NaT,
            }

        active["age"] += 1
        monitoring_dates = valid_history[-rules.confirmation_lookback_days :]
        monitoring = _fixed_core_metrics(
            monitoring_dates, active["core"], cross_by_date, industry_by_date
        )
        short_dates = valid_history[-rules.short_lookback_days :]
        short_monitoring = _fixed_core_metrics(
            short_dates, active["core"], cross_by_date, industry_by_date
        )
        structural_breakdown = (
            short_monitoring["core_contribution"] <= 0
            or pd.isna(monitoring["core_positive_contribution_share"])
            or monitoring["core_positive_contribution_share"]
            < rules.breakdown_minimum_core_positive_contribution_share
        )
        active["breakdown_count"] = (
            active["breakdown_count"] + 1 if structural_breakdown else 0
        )
        termination_reason: str | None = None
        if active["breakdown_count"] >= rules.breakdown_confirmation_days:
            termination_reason = "STRUCTURAL_BREAKDOWN"
        elif active["age"] >= rules.maximum_episode_valid_days:
            termination_reason = "MAXIMUM_AGE"

        candidate_event = False
        candidate_role: str | None = None
        phase_metrics: dict[str, Any] = {}
        hazards: list[str] = []
        if termination_reason is not None:
            phase = BREAKDOWN
        else:
            phase, phase_metrics, hazards = _phase_and_hazards(
                valid_history,
                active["core"],
                cross_by_date,
                industry_by_date,
                rules,
            )
            if phase == PROPAGATION and pd.isna(active["first_propagation_date"]):
                active["first_propagation_date"] = date
                candidate_event = True
                candidate_role = "FISH_MIDDLE_CANDIDATE"
            elif phase == EXHAUSTION and pd.isna(active["first_exhaustion_date"]):
                active["first_exhaustion_date"] = date
                candidate_event = True
                candidate_role = "TAIL_WARNING_CANDIDATE"

        row = {
            "date": date,
            "research_output": EPISODE_OBSERVE_ONLY,
            "failure_category": "PASS_STRUCTURE_ONLY",
            "episode_id": active["episode_id"],
            "phase": phase,
            "formation_date": active["formation_date"],
            "confirmation_date": active["confirmation_date"],
            "core_industries": "|".join(active["core"]),
            "episode_age_valid_days": int(active["age"]),
            "candidate_role": candidate_role,
            "candidate_event": candidate_event,
            "hazard_count": len(hazards),
            "hazards": "|".join(hazards) if hazards else "NONE",
            "breakdown_evidence_consecutive_days": int(
                active["breakdown_count"]
            ),
            "termination_reason": termination_reason,
            **monitoring,
            **{f"phase_{key}": value for key, value in phase_metrics.items()},
        }
        if date == active["confirmation_date"]:
            row.update(
                {
                    f"confirmation_{key}": value
                    for key, value in confirmation_metrics.items()
                }
            )
        state_rows.append(row)
        if candidate_event:
            candidate_id = "M1" if candidate_role == "FISH_MIDDLE_CANDIDATE" else "T1"
            event_rows.append(
                {
                    "event_id": f"{active['episode_id']}-{candidate_id}",
                    "candidate_id": candidate_id,
                    "candidate_role": candidate_role,
                    "episode_id": active["episode_id"],
                    "event_date": date,
                    "phase": phase,
                    "formation_date": active["formation_date"],
                    "confirmation_date": active["confirmation_date"],
                    "core_industries": "|".join(active["core"]),
                    "historical_evidence_label": "HISTORICALLY_CONTAMINATED",
                }
            )
        if termination_reason is not None:
            close_episode(date, termination_reason)
            valid_history = []

    if active is not None:
        close_episode(valid_history[-1], "DATA_CUTOFF")

    states = pd.DataFrame(state_rows).sort_values("date").reset_index(drop=True)
    episodes = pd.DataFrame(episode_rows)
    if not episodes.empty:
        episodes.sort_values("confirmation_date", inplace=True)
        episodes.reset_index(drop=True, inplace=True)
    events = pd.DataFrame(event_rows)
    if not events.empty:
        if events[["episode_id", "candidate_id"]].duplicated().any():
            raise AssertionError("同一周期生成了重复候选事件")
        events.sort_values("event_date", inplace=True)
        events.reset_index(drop=True, inplace=True)
    return EpisodeStructure(daily_states=states, episodes=episodes, events=events)


def build_forward_outcomes(
    etf_daily: pd.DataFrame,
    dividends: pd.DataFrame,
    rules: OutcomeRules,
) -> pd.DataFrame:
    """构造所有成熟信号日从下一交易日开盘起算的 20 日结果。"""

    _require_columns(etf_daily, {"date", "open", "low", "close"}, "510300日线")
    _require_columns(
        dividends,
        {"symbol", "ex_date", "cash_dividend_per_share"},
        "510300现金分红",
    )
    market = etf_daily[["date", "open", "low", "close"]].copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce")
    for column in ("open", "low", "close"):
        market[column] = pd.to_numeric(market[column], errors="coerce")
    if market.isna().any(axis=None):
        raise ValueError("510300日线存在空日期或空价格")
    if market["date"].duplicated().any():
        raise ValueError("510300日线存在重复日期")
    if (market[["open", "low", "close"]] <= 0).any(axis=None):
        raise ValueError("510300日线存在非正价格")
    market = market.loc[market["date"].le(rules.data_cutoff)].sort_values("date")
    market.reset_index(drop=True, inplace=True)

    cash = dividends.loc[dividends["symbol"].astype(str).eq("510300.SH")].copy()
    cash["ex_date"] = pd.to_datetime(cash["ex_date"], errors="coerce")
    cash["cash_dividend_per_share"] = pd.to_numeric(
        cash["cash_dividend_per_share"], errors="coerce"
    )
    if cash[["ex_date", "cash_dividend_per_share"]].isna().any().any():
        raise ValueError("510300现金分红存在空除息日或空金额")
    if (cash["cash_dividend_per_share"] < 0).any():
        raise ValueError("510300现金分红存在负金额")
    dividend_by_date = cash.groupby("ex_date")["cash_dividend_per_share"].sum()

    rows: list[dict[str, Any]] = []
    horizon = rules.horizon_trading_days
    for signal_position in range(len(market)):
        entry_position = signal_position + 1
        maturity_position = entry_position + horizon - 1
        if maturity_position >= len(market):
            break
        entry = market.iloc[entry_position]
        entry_open = float(entry["open"])
        wealth_close = float(entry["close"]) / entry_open
        first_cash = (1.0 + rules.cash_annual_rate) ** (
            1.0 / rules.trading_days_per_year
        )
        minimum_relative_path = float(entry["low"]) / entry_open - first_cash
        previous_close = float(entry["close"])
        for position in range(entry_position + 1, maturity_position + 1):
            day = market.iloc[position]
            dividend = float(dividend_by_date.get(pd.Timestamp(day["date"]), 0.0))
            low_wealth = wealth_close * (float(day["low"]) + dividend) / previous_close
            close_wealth = (
                wealth_close * (float(day["close"]) + dividend) / previous_close
            )
            elapsed = position - entry_position + 1
            cash_wealth = (1.0 + rules.cash_annual_rate) ** (
                elapsed / rules.trading_days_per_year
            )
            minimum_relative_path = min(
                minimum_relative_path, low_wealth - cash_wealth
            )
            wealth_close = close_wealth
            previous_close = float(day["close"])
        cash_wealth = (1.0 + rules.cash_annual_rate) ** (
            horizon / rules.trading_days_per_year
        )
        x20 = wealth_close - cash_wealth
        rows.append(
            {
                "signal_date": pd.Timestamp(market.iloc[signal_position]["date"]),
                "entry_date": pd.Timestamp(entry["date"]),
                "maturity_date": pd.Timestamp(market.iloc[maturity_position]["date"]),
                "horizon_trading_days": horizon,
                "x20": float(x20),
                "min_path20": float(minimum_relative_path),
                "middle20": bool(
                    x20 > rules.middle_minimum_relative_return
                    and minimum_relative_path > rules.tail20_path_threshold
                ),
                "tail20": bool(minimum_relative_path <= rules.tail20_path_threshold),
                "bad20": bool(x20 <= rules.bad20_terminal_threshold),
            }
        )
    return pd.DataFrame(rows)


def _moving_event_block_bootstrap_lift(
    values: np.ndarray,
    baseline_rate: float,
    block_length: int,
    repetitions: int,
    seed: int,
) -> dict[str, float | int | list[float] | None]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if len(clean) == 0 or baseline_rate <= 0:
        return {
            "observations": int(len(clean)),
            "repetitions": 0,
            "lift_interval_95pct": None,
        }
    length = min(max(1, int(block_length)), len(clean))
    starts = np.arange(0, len(clean) - length + 1)
    blocks_needed = int(np.ceil(len(clean) / length))
    rng = np.random.default_rng(seed)
    lifts = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        selected = rng.choice(starts, size=blocks_needed, replace=True)
        indices = np.concatenate(
            [np.arange(start, start + length) for start in selected]
        )[: len(clean)]
        lifts[repetition] = float(clean[indices].mean() / baseline_rate)
    interval = np.quantile(lifts, [0.025, 0.975])
    return {
        "observations": int(len(clean)),
        "block_length_events": int(length),
        "repetitions": int(repetitions),
        "median_lift": float(np.median(lifts)),
        "lift_interval_95pct": [float(interval[0]), float(interval[1])],
    }


def _next_dates(
    dates: Iterable[pd.Timestamp], market_dates: pd.DatetimeIndex, delay: int
) -> list[pd.Timestamp | pd.NaT]:
    positions = pd.Series(np.arange(len(market_dates)), index=market_dates)
    shifted: list[pd.Timestamp | pd.NaT] = []
    for value in dates:
        position = positions.get(pd.Timestamp(value))
        target = None if position is None else int(position) + delay
        shifted.append(
            pd.NaT
            if target is None or target >= len(market_dates)
            else pd.Timestamp(market_dates[target])
        )
    return shifted


def _candidate_statistics(
    events: pd.DataFrame,
    outcomes: pd.DataFrame,
    candidate_id: str,
    target_column: str,
    baseline_rate: float,
    baseline_tail_rate: float,
    rules: EvaluationRules,
    seed_offset: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    selected = events.loc[events["candidate_id"].eq(candidate_id)].merge(
        outcomes, left_on="event_date", right_on="signal_date", how="inner"
    )
    selected.sort_values("event_date", inplace=True)
    selected.reset_index(drop=True, inplace=True)
    count = len(selected)
    target_rate = float(selected[target_column].mean()) if count else np.nan
    lift = target_rate / baseline_rate if count and baseline_rate > 0 else np.nan
    bootstrap = _moving_event_block_bootstrap_lift(
        selected[target_column].astype(float).to_numpy() if count else np.array([]),
        baseline_rate,
        rules.bootstrap_event_block_length,
        rules.bootstrap_repetitions,
        rules.random_seed + seed_offset,
    )
    midpoint = count // 2
    first_rate = (
        float(selected.iloc[:midpoint][target_column].mean()) if midpoint else np.nan
    )
    second_rate = (
        float(selected.iloc[midpoint:][target_column].mean())
        if count - midpoint
        else np.nan
    )
    market_dates = pd.DatetimeIndex(outcomes["signal_date"].sort_values().unique())
    delayed_dates = _next_dates(
        selected["event_date"] if count else [], market_dates, rules.signal_delay_days
    )
    delayed_lookup = outcomes.set_index("signal_date")
    delayed_values = [
        delayed_lookup.at[value, target_column]
        for value in delayed_dates
        if pd.notna(value) and value in delayed_lookup.index
    ]
    delayed_rate = float(np.mean(delayed_values)) if delayed_values else np.nan
    delayed_lift = (
        delayed_rate / baseline_rate
        if pd.notna(delayed_rate) and baseline_rate > 0
        else np.nan
    )
    effect_retention = (
        (delayed_lift - 1.0) / (lift - 1.0)
        if pd.notna(delayed_lift) and pd.notna(lift) and lift > 1.0
        else np.nan
    )
    statistics = {
        "candidate_id": candidate_id,
        "mature_event_count": int(count),
        "first_event_date": (
            str(pd.Timestamp(selected["event_date"].min()).date()) if count else None
        ),
        "last_event_date": (
            str(pd.Timestamp(selected["event_date"].max()).date()) if count else None
        ),
        "target": target_column.upper(),
        "target_event_count": int(selected[target_column].sum()) if count else 0,
        "target_rate": target_rate,
        "baseline_rate": baseline_rate,
        "lift": lift,
        "bootstrap": bootstrap,
        "first_half_lift": first_rate / baseline_rate if baseline_rate > 0 else np.nan,
        "second_half_lift": second_rate / baseline_rate if baseline_rate > 0 else np.nan,
        "delayed_target_rate": delayed_rate,
        "delayed_lift": delayed_lift,
        "excess_lift_retention": effect_retention,
        "median_x20": float(selected["x20"].median()) if count else np.nan,
        "positive_x20_rate": float(selected["x20"].gt(0).mean()) if count else np.nan,
        "tail20_rate": float(selected["tail20"].mean()) if count else np.nan,
        "baseline_tail20_rate": baseline_tail_rate,
    }
    return statistics, selected


def evaluate_episode_candidates(
    structure: EpisodeStructure,
    outcomes: pd.DataFrame,
    rules: EvaluationRules,
) -> dict[str, Any]:
    """按冻结门槛分别否证 M1 鱼中与 T1 鱼尾候选。"""

    _require_columns(
        outcomes,
        {"signal_date", "x20", "middle20", "tail20", "bad20"},
        "未来结果",
    )
    states = structure.daily_states
    baseline = states.loc[states["research_output"].ne(NO_VIEW), ["date"]].merge(
        outcomes, left_on="date", right_on="signal_date", how="inner"
    )
    if baseline.empty:
        raise ValueError("没有成熟的归因有效日可计算冻结基准")
    middle_rate = float(baseline["middle20"].mean())
    tail_rate = float(baseline["tail20"].mean())
    if middle_rate <= 0 or tail_rate <= 0:
        raise ValueError("冻结基准事件率为零，无法计算 Lift")

    fish, fish_events = _candidate_statistics(
        structure.events,
        outcomes,
        "M1",
        "middle20",
        middle_rate,
        tail_rate,
        rules,
        seed_offset=1,
    )
    fish_lower = (
        fish["bootstrap"]["lift_interval_95pct"][0]
        if fish["bootstrap"]["lift_interval_95pct"] is not None
        else np.nan
    )
    fish_gates = {
        "minimum_mature_events": fish["mature_event_count"]
        >= rules.fish_minimum_events,
        "minimum_middle20_lift": pd.notna(fish["lift"])
        and fish["lift"] >= rules.fish_minimum_lift,
        "bootstrap_lift_lower_bound": pd.notna(fish_lower)
        and fish_lower > rules.fish_bootstrap_lower,
        "median_x20_positive": pd.notna(fish["median_x20"])
        and fish["median_x20"] > rules.fish_minimum_median_x20,
        "positive_x20_rate": pd.notna(fish["positive_x20_rate"])
        and fish["positive_x20_rate"] > rules.fish_minimum_positive_rate,
        "tail20_not_worse_than_baseline": pd.notna(fish["tail20_rate"])
        and fish["tail20_rate"]
        <= fish["baseline_tail20_rate"]
        * rules.fish_maximum_tail_rate_ratio,
        "delay_retention": pd.notna(fish["excess_lift_retention"])
        and fish["excess_lift_retention"] >= rules.minimum_effect_retention,
        "first_half_direction": pd.notna(fish["first_half_lift"])
        and fish["first_half_lift"] > rules.fish_split_minimum_lift,
        "second_half_direction": pd.notna(fish["second_half_lift"])
        and fish["second_half_lift"] > rules.fish_split_minimum_lift,
    }
    fish["gates"] = fish_gates
    fish["status"] = (
        "HISTORICAL_PASS_AWAITING_TRUE_FORWARD"
        if all(fish_gates.values())
        else "HISTORICAL_REJECTED_FROZEN"
    )

    tail, tail_events = _candidate_statistics(
        structure.events,
        outcomes,
        "T1",
        "tail20",
        tail_rate,
        tail_rate,
        rules,
        seed_offset=2,
    )
    tail_lower = (
        tail["bootstrap"]["lift_interval_95pct"][0]
        if tail["bootstrap"]["lift_interval_95pct"] is not None
        else np.nan
    )
    tail_gates = {
        "minimum_mature_events": tail["mature_event_count"]
        >= rules.tail_minimum_events,
        "minimum_tail20_events": tail["target_event_count"]
        >= rules.tail_minimum_target_events,
        "minimum_tail20_lift": pd.notna(tail["lift"])
        and tail["lift"] >= rules.tail_minimum_lift,
        "bootstrap_lift_lower_bound": pd.notna(tail_lower)
        and tail_lower > rules.tail_bootstrap_lower,
        "delay_retention": pd.notna(tail["excess_lift_retention"])
        and tail["excess_lift_retention"] >= rules.minimum_effect_retention,
        "first_half_direction": pd.notna(tail["first_half_lift"])
        and tail["first_half_lift"] > rules.tail_split_minimum_lift,
        "second_half_direction": pd.notna(tail["second_half_lift"])
        and tail["second_half_lift"] > rules.tail_split_minimum_lift,
    }
    tail["gates"] = tail_gates
    tail["status"] = (
        "HISTORICAL_PASS_AWAITING_TRUE_FORWARD"
        if all(tail_gates.values())
        else "HISTORICAL_REJECTED_FROZEN"
    )

    overall = (
        "HISTORICAL_PASS_AWAITING_TRUE_FORWARD"
        if fish["status"] == "HISTORICAL_PASS_AWAITING_TRUE_FORWARD"
        and tail["status"] == "HISTORICAL_PASS_AWAITING_TRUE_FORWARD"
        else "PARTIAL_OR_REJECTED_NO_FORECAST_AUTHORIZATION"
    )
    return {
        "overall_status": overall,
        "historical_evidence_label": "HISTORICALLY_CONTAMINATED",
        "baseline": {
            "mature_valid_day_count": int(len(baseline)),
            "middle20_rate": middle_rate,
            "tail20_rate": tail_rate,
            "median_x20": float(baseline["x20"].median()),
        },
        "fish_middle": fish,
        "tail_warning": tail,
        "event_audit": {
            "fish_event_ids": fish_events["event_id"].tolist(),
            "tail_event_ids": tail_events["event_id"].tolist(),
        },
        "safety": {
            "forecast_eligible_emitted": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
        },
    }


__all__ = [
    "BREAKDOWN",
    "EPISODE_OBSERVE_ONLY",
    "EXHAUSTION",
    "EpisodeRules",
    "EpisodeStructure",
    "EvaluationRules",
    "FORECAST_ELIGIBLE",
    "MATURE_CONTINUATION",
    "NO_EPISODE",
    "NO_VIEW",
    "OutcomeRules",
    "PROPAGATION",
    "build_driver_episode_structure",
    "build_forward_outcomes",
    "evaluate_episode_candidates",
]
