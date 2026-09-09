"""V2 G1 停止后的历史覆盖审计；不重开门禁、不训练模型。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class HistoricalCoverageAuditError(ValueError):
    """历史覆盖审计契约不满足。"""


USABLE_RETURN_STATES = {"TRADED_VALID", "OFFICIAL_SUSPENSION"}
KNOWN_RETURN_STATES = {
    "TRADED_VALID",
    "OFFICIAL_SUSPENSION",
    "CORPORATE_ACTION_UNRESOLVED",
    "SUPPLIER_MISSING_OR_CONFLICT",
}
FORBIDDEN_INPUT_COLUMNS = {
    "minimum_path_return",
    "first_breach_date",
    "cash_dividend_per_share_in_horizon",
    "forward_return",
    "horizon_return",
}
FORBIDDEN_OUTPUT_TOKENS = {
    "auc",
    "pr_auc",
    "brier",
    "log_loss",
    "sharpe",
    "drawdown",
    "portfolio_return",
    "minimum_path_return",
    "first_breach_date",
}


@dataclass(frozen=True)
class HistoricalCoverageAuditArtifacts:
    """可持久化的三张诊断表及不带时钟的确定性元数据。"""

    prehistory_coverage_comparison: pd.DataFrame
    event_gap_ledger: pd.DataFrame
    member_window_gap_ledger: pd.DataFrame
    metrics: dict[str, Any]
    audit_state: dict[str, Any]


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise HistoricalCoverageAuditError(f"{label}缺少列：{missing}")


def _reject_forbidden_columns(frame: pd.DataFrame, label: str) -> None:
    present = sorted(FORBIDDEN_INPUT_COLUMNS.intersection(map(str, frame.columns)))
    if present:
        raise HistoricalCoverageAuditError(f"{label}包含禁止读取列：{present}")


def _normalize_dates(values: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(values, errors="coerce").dt.normalize()
    if result.isna().any():
        raise HistoricalCoverageAuditError(f"{label}存在非法日期")
    return result


def _strict_bool(values: pd.Series, label: str) -> pd.Series:
    if values.isna().any():
        raise HistoricalCoverageAuditError(f"{label}存在空布尔值")
    if not values.isin([True, False]).all():
        invalid = sorted(set(values.loc[~values.isin([True, False])].astype(str)))
        raise HistoricalCoverageAuditError(f"{label}存在非法布尔值：{invalid[:5]}")
    return values.astype(bool)


def _require_unique(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    duplicated = frame.duplicated(list(columns), keep=False)
    if duplicated.any():
        sample = frame.loc[duplicated, list(columns)].head(5).to_dict("records")
        raise HistoricalCoverageAuditError(f"{label}键不唯一：{sample}")


def _assert_cutoff(frame: pd.DataFrame, column: str, cutoff: pd.Timestamp, label: str) -> None:
    if frame.empty:
        raise HistoricalCoverageAuditError(f"{label}为空")
    later = frame[column].gt(cutoff)
    if later.any():
        latest = frame.loc[later, column].max().date().isoformat()
        raise HistoricalCoverageAuditError(
            f"{label}包含历史截止日之后日期：cutoff={cutoff.date()}, latest={latest}"
        )


def _required_count(member_count: pd.Series, minimum_ratio: float) -> pd.Series:
    return np.ceil(member_count.astype(float) * minimum_ratio - 1e-12).astype(int)


def _rolling_complete(usable: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 0:
        raise HistoricalCoverageAuditError("滚动窗口必须为正整数")
    counts = usable.astype("int16").rolling(window, min_periods=window).sum()
    return counts.eq(window)


def cumulative_tail_deficit_upper_bounds(
    event_gap_ledger: pd.DataFrame,
    deficit_caps: Sequence[int],
) -> dict[str, int]:
    """生成尾部缺口单维上界；结果永远不具备门禁准入效力。"""

    _require_columns(
        event_gap_ledger,
        ["current_b2_identifiable_event", "tail_member_deficit"],
        "事件缺口账本",
    )
    current = event_gap_ledger["current_b2_identifiable_event"].astype(bool)
    deficit = pd.to_numeric(event_gap_ledger["tail_member_deficit"], errors="raise")
    result: dict[str, int] = {}
    for raw_cap in sorted(set(int(value) for value in deficit_caps)):
        if raw_cap < 0:
            raise HistoricalCoverageAuditError("尾部缺口上限不得为负")
        potential = current | ((~current) & deficit.le(raw_cap))
        result[str(raw_cap)] = int(potential.sum())
    return result


def _first_cap_reaching(
    upper_bounds: Mapping[str, int], threshold: int
) -> int | None:
    candidates = sorted(
        int(cap) for cap, count in upper_bounds.items() if int(count) >= threshold
    )
    return candidates[0] if candidates else None


def _prepare_inputs(
    *,
    sample_eligibility: pd.DataFrame,
    event_eligibility: pd.DataFrame,
    mft_coverage: pd.DataFrame,
    classified_returns: pd.DataFrame,
    membership: pd.DataFrame,
    cutoff: pd.Timestamp,
    expected_members_per_day: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for label, frame in (
        ("G1 样本准入", sample_eligibility),
        ("G1 事件准入", event_eligibility),
        ("M/F/T 覆盖", mft_coverage),
        ("四态收益", classified_returns),
        ("点时成员", membership),
    ):
        _reject_forbidden_columns(frame, label)

    _require_columns(
        sample_eligibility,
        [
            "origin_date",
            "bad10",
            "event_id",
            "B1_state",
            "B2_feature_state",
            "B3_feature_state",
            "b1_eligible",
            "b2_vs_b1_eligible",
            "b3_vs_b2_eligible",
            "b2_vs_b1_no_view_reason",
            "b3_vs_b2_no_view_reason",
        ],
        "G1 样本准入",
    )
    _require_columns(
        event_eligibility,
        [
            "event_id",
            "total_positive_origin_count",
            "b2_eligible_positive_origin_count",
            "b2_identifiable_event",
            "b3_eligible_positive_origin_count",
            "b3_identifiable_event",
        ],
        "G1 事件准入",
    )
    _require_columns(
        mft_coverage,
        [
            "date",
            "point_in_time_member_count",
            "return20_scoreable_member_count",
            "tail_scoreable_member_count",
            "comovement_scoreable_member_ratio",
            "four_state_daily_coverage_state",
            "internal_formula_coverage_state",
            "internal_feature_state",
            "F_state",
            "M_state",
            "T_state",
            "B2_feature_state",
            "B3_feature_state",
            "NO_VIEW_REASON",
        ],
        "M/F/T 覆盖",
    )
    _require_columns(
        classified_returns,
        [
            "date",
            "symbol",
            "source_observed",
            "supplier_conflict",
            "official_suspension",
            "suspension_evidence_id",
            "corporate_action_status",
            "constituent_return_state",
            "daily_total_shareholder_return",
            "return_is_usable",
            "state_reason",
        ],
        "四态收益",
    )
    _require_columns(membership, ["membership_date", "index_code", "symbol"], "点时成员")

    samples = sample_eligibility.copy()
    samples["origin_date"] = _normalize_dates(samples["origin_date"], "样本原点")
    _assert_cutoff(samples, "origin_date", cutoff, "G1 样本准入")
    samples["bad10"] = pd.to_numeric(samples["bad10"], errors="coerce")
    if samples["bad10"].isna().any() or not samples["bad10"].isin([0, 1]).all():
        raise HistoricalCoverageAuditError("BAD10 必须且只能为 0/1")
    samples["event_id"] = samples["event_id"].astype("string").fillna("").str.strip()
    for column in ("b1_eligible", "b2_vs_b1_eligible", "b3_vs_b2_eligible"):
        samples[column] = _strict_bool(samples[column], column)

    events = event_eligibility.copy()
    events["event_id"] = events["event_id"].astype("string").str.strip()
    if events["event_id"].eq("").any():
        raise HistoricalCoverageAuditError("事件账本存在空 event_id")
    _require_unique(events, ["event_id"], "G1 事件准入")
    for column in ("b2_identifiable_event", "b3_identifiable_event"):
        events[column] = _strict_bool(events[column], column)

    coverage = mft_coverage.copy()
    coverage["date"] = _normalize_dates(coverage["date"], "M/F/T 覆盖日期")
    _assert_cutoff(coverage, "date", cutoff, "M/F/T 覆盖")
    _require_unique(coverage, ["date"], "M/F/T 覆盖")

    classified = classified_returns.copy()
    classified["date"] = _normalize_dates(classified["date"], "四态收益日期")
    _assert_cutoff(classified, "date", cutoff, "四态收益")
    classified["symbol"] = classified["symbol"].astype("string").str.strip().str.upper()
    _require_unique(classified, ["date", "symbol"], "四态收益")
    for column in ("source_observed", "supplier_conflict", "official_suspension", "return_is_usable"):
        classified[column] = _strict_bool(classified[column], column)
    states = set(classified["constituent_return_state"].dropna().astype(str))
    unknown = sorted(states.difference(KNOWN_RETURN_STATES))
    if unknown:
        raise HistoricalCoverageAuditError(f"四态收益存在未知状态：{unknown}")

    members = membership.copy()
    members["date"] = _normalize_dates(members["membership_date"], "成员日期")
    _assert_cutoff(members, "date", cutoff, "点时成员")
    members["symbol"] = members["symbol"].astype("string").str.strip().str.upper()
    members["index_code"] = members["index_code"].astype("string").str.strip()
    if not members["index_code"].eq("000300").all():
        raise HistoricalCoverageAuditError("点时成员只能是 000300 研究输入")
    _require_unique(members, ["date", "symbol"], "点时成员")
    daily_counts = members.groupby("date", sort=True).size()
    if not daily_counts.eq(expected_members_per_day).all():
        bad = daily_counts.loc[~daily_counts.eq(expected_members_per_day)].head(5).to_dict()
        raise HistoricalCoverageAuditError(
            f"点时成员日计数不等于 {expected_members_per_day}：{bad}"
        )

    positive_event_ids = set(samples.loc[samples["bad10"].eq(1), "event_id"])
    if "" in positive_event_ids:
        raise HistoricalCoverageAuditError("BAD10 正样本必须有 event_id")
    if positive_event_ids != set(events["event_id"]):
        raise HistoricalCoverageAuditError("正样本事件集合与事件准入账本不一致")
    return samples, events, coverage, classified, members


def _build_window_comparison(
    *,
    classified: pd.DataFrame,
    members: pd.DataFrame,
    coverage: pd.DataFrame,
    lookback_days: int,
    tail_days: int,
    member_coverage_minimum: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_dates = pd.DatetimeIndex(sorted(members["date"].unique()))
    extended_dates = pd.DatetimeIndex(
        sorted(set(classified["date"].unique()).union(feature_dates))
    )
    symbols = pd.Index(sorted(set(members["symbol"].astype(str))))

    numeric = pd.to_numeric(
        classified["daily_total_shareholder_return"], errors="coerce"
    )
    usable = (
        classified["return_is_usable"].eq(True)
        & classified["constituent_return_state"].isin(USABLE_RETURN_STATES)
        & numeric.notna()
        & np.isfinite(numeric)
    )
    usable_wide = (
        classified.assign(_usable=usable)
        .pivot(index="date", columns="symbol", values="_usable")
        .reindex(index=extended_dates, columns=symbols)
        .fillna(False)
        .astype(bool)
    )
    membership_mask = (
        members.assign(_member=True)
        .pivot(index="date", columns="symbol", values="_member")
        .reindex(index=feature_dates, columns=symbols)
        .fillna(False)
        .astype(bool)
    )

    current_usable = usable_wide.reindex(feature_dates).fillna(False).astype(bool)
    current_return20 = _rolling_complete(current_usable, lookback_days)
    current_tail60 = _rolling_complete(current_usable, tail_days)
    extended_return20 = _rolling_complete(usable_wide, lookback_days).reindex(feature_dates)
    extended_tail60 = _rolling_complete(usable_wide, tail_days).reindex(feature_dates)

    current_return20_count = (current_return20 & membership_mask).sum(axis=1).astype(int)
    current_tail60_count = (current_tail60 & membership_mask).sum(axis=1).astype(int)
    extended_return20_count = (extended_return20 & membership_mask).sum(axis=1).astype(int)
    extended_tail60_count = (extended_tail60 & membership_mask).sum(axis=1).astype(int)
    member_count = membership_mask.sum(axis=1).astype(int)
    required = _required_count(member_count, member_coverage_minimum)

    frozen = coverage.set_index("date").reindex(feature_dates)
    if frozen["point_in_time_member_count"].isna().any():
        raise HistoricalCoverageAuditError("M/F/T 覆盖缺少成员日期")
    frozen_return20 = pd.to_numeric(
        frozen["return20_scoreable_member_count"], errors="raise"
    ).astype(int)
    frozen_tail60 = pd.to_numeric(
        frozen["tail_scoreable_member_count"], errors="raise"
    ).astype(int)
    return_match = current_return20_count.eq(frozen_return20)
    tail_match = current_tail60_count.eq(frozen_tail60)
    if not return_match.all() or not tail_match.all():
        bad_dates = feature_dates[(~return_match | ~tail_match).to_numpy()][:5]
        raise HistoricalCoverageAuditError(
            "当前成员日历口径不能复现冻结覆盖计数："
            f"{[date.date().isoformat() for date in bad_dates]}"
        )

    comparison = pd.DataFrame(
        {
            "date": feature_dates,
            "point_in_time_member_count": member_count.to_numpy(),
            "required_scoreable_member_count": required.to_numpy(),
            "frozen_return20_scoreable_member_count": frozen_return20.to_numpy(),
            "recomputed_current_return20_scoreable_member_count": current_return20_count.to_numpy(),
            "current_return20_matches_frozen": return_match.to_numpy(),
            "prehistory_inclusive_return20_scoreable_member_count": extended_return20_count.to_numpy(),
            "prehistory_return20_member_delta": (extended_return20_count - current_return20_count).to_numpy(),
            "frozen_tail_scoreable_member_count": frozen_tail60.to_numpy(),
            "recomputed_current_tail_scoreable_member_count": current_tail60_count.to_numpy(),
            "current_tail_matches_frozen": tail_match.to_numpy(),
            "prehistory_inclusive_tail_scoreable_member_count": extended_tail60_count.to_numpy(),
            "prehistory_tail_member_delta": (extended_tail60_count - current_tail60_count).to_numpy(),
            "current_return20_gate_pass": current_return20_count.ge(required).to_numpy(),
            "prehistory_return20_gate_pass": extended_return20_count.ge(required).to_numpy(),
            "current_tail_gate_pass": current_tail60_count.ge(required).to_numpy(),
            "prehistory_tail_gate_pass": extended_tail60_count.ge(required).to_numpy(),
        }
    )
    context = {
        "feature_dates": feature_dates,
        "extended_dates": extended_dates,
        "symbols": symbols,
        "usable_wide": usable_wide,
        "membership_mask": membership_mask,
        "current_tail60": current_tail60,
        "extended_tail60": extended_tail60,
    }
    return comparison, context


def _build_event_gap_ledger(
    *,
    samples: pd.DataFrame,
    events: pd.DataFrame,
    coverage: pd.DataFrame,
    comparison: pd.DataFrame,
    member_coverage_minimum: float,
    comovement_minimum: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positives = samples.loc[samples["bad10"].eq(1)].copy()
    positives = positives.merge(
        coverage,
        left_on="origin_date",
        right_on="date",
        how="left",
        validate="many_to_one",
        suffixes=("_sample", "_coverage"),
    ).drop(columns=["date"])
    positives = positives.merge(
        comparison.drop(columns=["point_in_time_member_count"]),
        left_on="origin_date",
        right_on="date",
        how="left",
        validate="many_to_one",
    ).drop(columns=["date"])
    if positives["point_in_time_member_count"].isna().any():
        raise HistoricalCoverageAuditError("正样本原点缺少历史覆盖记录")

    member_count = pd.to_numeric(
        positives["point_in_time_member_count"], errors="raise"
    ).astype(int)
    required = _required_count(member_count, member_coverage_minimum)
    positives["return20_member_deficit"] = np.maximum(
        0,
        required
        - pd.to_numeric(
            positives["frozen_return20_scoreable_member_count"], errors="raise"
        ).astype(int),
    ).astype(int)
    positives["tail_member_deficit"] = np.maximum(
        0,
        required
        - pd.to_numeric(
            positives["frozen_tail_scoreable_member_count"], errors="raise"
        ).astype(int),
    ).astype(int)
    ratio = pd.to_numeric(
        positives["comovement_scoreable_member_ratio"], errors="coerce"
    )
    scoreable_estimate = np.rint(ratio.fillna(0.0) * member_count).astype(int)
    comovement_required = _required_count(member_count, comovement_minimum)
    positives["comovement_scoreable_member_count_estimate"] = scoreable_estimate
    positives["comovement_member_deficit"] = np.maximum(
        0, comovement_required - scoreable_estimate
    ).astype(int)
    positives["current_b2_blocker"] = (~positives["b2_vs_b1_eligible"]).astype(int)
    positives["b1_blocker"] = (~positives["b1_eligible"]).astype(int)
    positives["prehistory_price_and_frozen_comovement_gate_pass"] = (
        positives["b1_eligible"]
        & positives["prehistory_return20_gate_pass"]
        & positives["prehistory_tail_gate_pass"]
        & ratio.ge(comovement_minimum)
    )

    sort_columns = [
        "event_id",
        "current_b2_blocker",
        "b1_blocker",
        "tail_member_deficit",
        "return20_member_deficit",
        "comovement_member_deficit",
        "origin_date",
    ]
    best = positives.sort_values(sort_columns, kind="stable").groupby(
        "event_id", sort=True, as_index=False
    ).head(1)
    any_prehistory = (
        positives.groupby("event_id", sort=True)[
            "prehistory_price_and_frozen_comovement_gate_pass"
        ]
        .any()
        .rename("any_positive_origin_prehistory_price_and_frozen_comovement_gate_pass")
    )
    best = best.merge(any_prehistory, on="event_id", how="left", validate="one_to_one")
    best = best.merge(
        events,
        on="event_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_event"),
    )
    if best["total_positive_origin_count"].isna().any():
        raise HistoricalCoverageAuditError("代表原点缺少事件准入记录")

    columns = [
        "event_id",
        "total_positive_origin_count",
        "b2_eligible_positive_origin_count",
        "b2_identifiable_event",
        "b3_eligible_positive_origin_count",
        "b3_identifiable_event",
        "origin_date",
        "B1_state",
        "B2_feature_state_sample",
        "B3_feature_state_sample",
        "b1_eligible",
        "b2_vs_b1_eligible",
        "b3_vs_b2_eligible",
        "b2_vs_b1_no_view_reason",
        "b3_vs_b2_no_view_reason",
        "point_in_time_member_count",
        "required_scoreable_member_count",
        "frozen_return20_scoreable_member_count",
        "return20_member_deficit",
        "prehistory_inclusive_return20_scoreable_member_count",
        "prehistory_return20_member_delta",
        "frozen_tail_scoreable_member_count",
        "tail_member_deficit",
        "prehistory_inclusive_tail_scoreable_member_count",
        "prehistory_tail_member_delta",
        "comovement_scoreable_member_ratio",
        "comovement_scoreable_member_count_estimate",
        "comovement_member_deficit",
        "four_state_daily_coverage_state",
        "internal_formula_coverage_state",
        "internal_feature_state",
        "F_state",
        "M_state",
        "T_state",
        "NO_VIEW_REASON",
        "prehistory_price_and_frozen_comovement_gate_pass",
        "any_positive_origin_prehistory_price_and_frozen_comovement_gate_pass",
    ]
    best = best.loc[:, columns].rename(
        columns={
            "origin_date": "deterministic_best_origin_date",
            "b2_identifiable_event": "current_b2_identifiable_event",
            "b3_identifiable_event": "current_b3_identifiable_event",
            "B2_feature_state_sample": "current_B2_feature_state",
            "B3_feature_state_sample": "current_B3_feature_state",
        }
    )
    best["diagnostic_authority"] = "NON_ADMISSIBLE_DIAGNOSTIC_ONLY"
    best["g1_gate_promoted"] = False
    best = best.sort_values("event_id", kind="stable").reset_index(drop=True)
    return best, positives


def _primary_gap_reason(
    *,
    current_scoreable: bool,
    prehistory_scoreable: bool,
    state_counts: Mapping[str, int],
    prehistory_calendar_shortfall: int,
) -> str:
    if current_scoreable:
        return "CURRENT_TAIL_WINDOW_COMPLETE"
    if prehistory_scoreable:
        return "IMPLEMENTATION_DISCARDED_ADMITTED_PRE_OBSERVATION_HISTORY"
    if int(state_counts.get("SUPPLIER_MISSING_OR_CONFLICT", 0)) > 0:
        return "OFFICIAL_SUSPENSION_EVIDENCE_OR_SUPPLIER_GAP_REVIEW_REQUIRED"
    if int(state_counts.get("CORPORATE_ACTION_UNRESOLVED", 0)) > 0:
        return "CORPORATE_ACTION_REMEDIATION_REQUIRED"
    if int(state_counts.get("NO_CLASSIFIED_ROW", 0)) > 0:
        return "NO_CLASSIFIED_ROW_REVIEW_REQUIRED"
    if prehistory_calendar_shortfall > 0:
        return "CLASSIFIED_HISTORY_START_TOO_LATE_FOR_COMPLETE_WINDOW"
    return "OTHER_INCOMPLETE_WINDOW_NO_ZERO_FILL"


def _build_member_gap_ledger(
    *,
    event_gap: pd.DataFrame,
    classified: pd.DataFrame,
    members: pd.DataFrame,
    context: Mapping[str, Any],
    tail_days: int,
) -> pd.DataFrame:
    feature_dates: pd.DatetimeIndex = context["feature_dates"]
    extended_dates: pd.DatetimeIndex = context["extended_dates"]
    usable_wide: pd.DataFrame = context["usable_wide"]
    membership_mask: pd.DataFrame = context["membership_mask"]
    current_tail60: pd.DataFrame = context["current_tail60"]
    extended_tail60: pd.DataFrame = context["extended_tail60"]
    symbols: pd.Index = context["symbols"]

    state_wide = (
        classified.pivot(
            index="date", columns="symbol", values="constituent_return_state"
        )
        .reindex(index=extended_dates, columns=symbols)
        .fillna("NO_CLASSIFIED_ROW")
        .astype(str)
    )
    feature_positions = {date: index for index, date in enumerate(feature_dates)}
    extended_positions = {date: index for index, date in enumerate(extended_dates)}
    rows: list[dict[str, Any]] = []
    unidentifiable = event_gap.loc[~event_gap["current_b2_identifiable_event"].astype(bool)]
    for event in unidentifiable.itertuples(index=False):
        date = pd.Timestamp(event.deterministic_best_origin_date).normalize()
        if date not in feature_positions or date not in extended_positions:
            raise HistoricalCoverageAuditError(f"代表原点不在日历：{date.date()}")
        current_position = feature_positions[date]
        extended_position = extended_positions[date]
        current_window = feature_dates[
            max(0, current_position - tail_days + 1) : current_position + 1
        ]
        extended_window = extended_dates[
            max(0, extended_position - tail_days + 1) : extended_position + 1
        ]
        member_symbols = symbols[
            membership_mask.loc[date].to_numpy(dtype=bool)
        ]
        for symbol in member_symbols:
            current_scoreable = bool(current_tail60.loc[date, symbol])
            if current_scoreable:
                continue
            prehistory_scoreable = bool(extended_tail60.loc[date, symbol])
            current_usable_count = int(usable_wide.loc[current_window, symbol].sum())
            extended_usable_count = int(usable_wide.loc[extended_window, symbol].sum())
            state_counts = (
                state_wide.loc[extended_window, symbol].value_counts().to_dict()
            )
            state_counts = {str(key): int(value) for key, value in state_counts.items()}
            shortfall = int(max(0, tail_days - len(extended_window)))
            rows.append(
                {
                    "event_id": str(event.event_id),
                    "deterministic_best_origin_date": date,
                    "symbol": str(symbol),
                    "current_calendar_window_session_count": int(len(current_window)),
                    "prehistory_inclusive_window_session_count": int(len(extended_window)),
                    "prehistory_calendar_session_shortfall": shortfall,
                    "current_usable_observation_count_60": current_usable_count,
                    "prehistory_inclusive_usable_observation_count_60": extended_usable_count,
                    "current_tail_scoreable": current_scoreable,
                    "prehistory_inclusive_tail_scoreable": prehistory_scoreable,
                    "traded_valid_session_count": int(state_counts.get("TRADED_VALID", 0)),
                    "official_suspension_session_count": int(state_counts.get("OFFICIAL_SUSPENSION", 0)),
                    "corporate_action_unresolved_session_count": int(state_counts.get("CORPORATE_ACTION_UNRESOLVED", 0)),
                    "supplier_missing_or_conflict_session_count": int(state_counts.get("SUPPLIER_MISSING_OR_CONFLICT", 0)),
                    "no_classified_row_session_count": int(state_counts.get("NO_CLASSIFIED_ROW", 0)),
                    "primary_gap_reason": _primary_gap_reason(
                        current_scoreable=current_scoreable,
                        prehistory_scoreable=prehistory_scoreable,
                        state_counts=state_counts,
                        prehistory_calendar_shortfall=shortfall,
                    ),
                    "provider_suspend_flag_may_promote_state": False,
                    "missing_return_zero_filled": False,
                    "diagnostic_authority": "NON_ADMISSIBLE_DIAGNOSTIC_ONLY",
                }
            )
    columns = [
        "event_id",
        "deterministic_best_origin_date",
        "symbol",
        "current_calendar_window_session_count",
        "prehistory_inclusive_window_session_count",
        "prehistory_calendar_session_shortfall",
        "current_usable_observation_count_60",
        "prehistory_inclusive_usable_observation_count_60",
        "current_tail_scoreable",
        "prehistory_inclusive_tail_scoreable",
        "traded_valid_session_count",
        "official_suspension_session_count",
        "corporate_action_unresolved_session_count",
        "supplier_missing_or_conflict_session_count",
        "no_classified_row_session_count",
        "primary_gap_reason",
        "provider_suspend_flag_may_promote_state",
        "missing_return_zero_filled",
        "diagnostic_authority",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["event_id", "symbol"], kind="stable", ignore_index=True
    )


def _current_member_state_metrics(
    members: pd.DataFrame, classified: pd.DataFrame
) -> dict[str, Any]:
    columns = [
        "date",
        "symbol",
        "constituent_return_state",
        "source_observed",
    ]
    joined = members[["date", "symbol"]].merge(
        classified[columns],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    states = joined["constituent_return_state"].fillna("NO_CLASSIFIED_ROW")
    counts = {str(key): int(value) for key, value in states.value_counts().to_dict().items()}
    explicit_source_missing = joined["source_observed"].eq(False)
    return {
        "point_in_time_member_day_count": int(len(joined)),
        "member_day_state_counts": {
            state: int(counts.get(state, 0))
            for state in [
                "TRADED_VALID",
                "OFFICIAL_SUSPENSION",
                "CORPORATE_ACTION_UNRESOLVED",
                "SUPPLIER_MISSING_OR_CONFLICT",
                "NO_CLASSIFIED_ROW",
            ]
        },
        "explicit_source_observed_false_member_day_count": int(explicit_source_missing.sum()),
        "explicit_source_observed_false_symbol_count": int(
            joined.loc[explicit_source_missing, "symbol"].nunique()
        ),
    }


def _assert_output_boundary(artifacts: HistoricalCoverageAuditArtifacts) -> None:
    for name, frame in (
        ("逐日对照", artifacts.prehistory_coverage_comparison),
        ("事件缺口", artifacts.event_gap_ledger),
        ("成员窗口缺口", artifacts.member_window_gap_ledger),
    ):
        lowered = {str(column).casefold() for column in frame.columns}
        forbidden = sorted(FORBIDDEN_OUTPUT_TOKENS.intersection(lowered))
        if forbidden:
            raise HistoricalCoverageAuditError(f"{name}意外产生禁止输出列：{forbidden}")


def build_historical_coverage_audit(
    *,
    sample_eligibility: pd.DataFrame,
    event_eligibility: pd.DataFrame,
    mft_coverage: pd.DataFrame,
    classified_returns: pd.DataFrame,
    membership: pd.DataFrame,
    historical_cutoff: str | pd.Timestamp,
    lookback_days: int = 20,
    tail_days: int = 60,
    tail_return_days: int = 5,
    member_coverage_minimum: float = 0.98,
    comovement_member_ratio_minimum: float = 0.90,
    expected_members_per_day: int = 300,
    deficit_caps: Sequence[int] = (0, 1, 3, 4, 5, 6, 7, 10, 20, 60, 294),
    mechanism_event_minimum: int = 30,
    full_model_event_minimum: int = 40,
) -> HistoricalCoverageAuditArtifacts:
    """构建全事件、历史截止内的覆盖诊断，不重算任何 G1 门禁。"""

    cutoff = pd.Timestamp(historical_cutoff).normalize()
    if pd.isna(cutoff):
        raise HistoricalCoverageAuditError("历史截止日非法")
    if (lookback_days, tail_return_days, tail_days) != (20, 5, 60):
        raise HistoricalCoverageAuditError("窗口必须保持冻结的 20/5/60")
    if not math.isclose(member_coverage_minimum, 0.98, abs_tol=1e-12):
        raise HistoricalCoverageAuditError("成员覆盖门必须保持 98%")
    if not math.isclose(comovement_member_ratio_minimum, 0.90, abs_tol=1e-12):
        raise HistoricalCoverageAuditError("共同运动覆盖门必须保持 90%")
    if mechanism_event_minimum != 30 or full_model_event_minimum != 40:
        raise HistoricalCoverageAuditError("事件门必须保持 30/40")

    samples, events, coverage, classified, members = _prepare_inputs(
        sample_eligibility=sample_eligibility,
        event_eligibility=event_eligibility,
        mft_coverage=mft_coverage,
        classified_returns=classified_returns,
        membership=membership,
        cutoff=cutoff,
        expected_members_per_day=expected_members_per_day,
    )
    comparison, context = _build_window_comparison(
        classified=classified,
        members=members,
        coverage=coverage,
        lookback_days=lookback_days,
        tail_days=tail_days,
        member_coverage_minimum=member_coverage_minimum,
    )
    event_gap, positives = _build_event_gap_ledger(
        samples=samples,
        events=events,
        coverage=coverage,
        comparison=comparison,
        member_coverage_minimum=member_coverage_minimum,
        comovement_minimum=comovement_member_ratio_minimum,
    )
    member_gap = _build_member_gap_ledger(
        event_gap=event_gap,
        classified=classified,
        members=members,
        context=context,
        tail_days=tail_days,
    )
    upper_bounds = cumulative_tail_deficit_upper_bounds(event_gap, deficit_caps)
    member_state_metrics = _current_member_state_metrics(members, classified)
    prehistory_dates = context["extended_dates"][
        context["extended_dates"] < context["feature_dates"].min()
    ]
    return_delta = comparison["prehistory_return20_member_delta"]
    tail_delta = comparison["prehistory_tail_member_delta"]
    newly_tail_pass = (
        (~comparison["current_tail_gate_pass"])
        & comparison["prehistory_tail_gate_pass"]
    )
    current_b2 = int(event_gap["current_b2_identifiable_event"].sum())
    current_b3 = int(event_gap["current_b3_identifiable_event"].sum())
    prehistory_event_diagnostic = int(
        (
            event_gap["current_b2_identifiable_event"]
            | event_gap[
                "any_positive_origin_prehistory_price_and_frozen_comovement_gate_pass"
            ]
        ).sum()
    )
    metrics: dict[str, Any] = {
        "historical_cutoff": cutoff.date().isoformat(),
        "classified_history_start": context["extended_dates"].min().date().isoformat(),
        "observation_start": context["feature_dates"].min().date().isoformat(),
        "observation_end": context["feature_dates"].max().date().isoformat(),
        "pre_observation_market_session_count": int(len(prehistory_dates)),
        "coverage_date_count": int(len(comparison)),
        "current_return20_count_matches_frozen_on_all_dates": bool(
            comparison["current_return20_matches_frozen"].all()
        ),
        "current_tail_count_matches_frozen_on_all_dates": bool(
            comparison["current_tail_matches_frozen"].all()
        ),
        "prehistory_return20_positive_delta_date_count": int(return_delta.gt(0).sum()),
        "prehistory_return20_max_member_delta": int(return_delta.max()),
        "prehistory_tail_positive_delta_date_count": int(tail_delta.gt(0).sum()),
        "prehistory_tail_max_member_delta": int(tail_delta.max()),
        "prehistory_tail_newly_passing_date_count": int(newly_tail_pass.sum()),
        "implementation_protocol_mismatch_confirmed": bool(
            return_delta.gt(0).any() or tail_delta.gt(0).any()
        ),
        "total_independent_event_count": int(len(event_gap)),
        "total_positive_origin_count": int(samples["bad10"].eq(1).sum()),
        "total_non_event_risk_day_count": int(samples["bad10"].eq(0).sum()),
        "current_b2_identifiable_event_count": current_b2,
        "current_b3_identifiable_event_count": current_b3,
        "current_b2_unidentifiable_event_count": int(len(event_gap) - current_b2),
        "prehistory_price_and_frozen_comovement_diagnostic_event_upper_bound": prehistory_event_diagnostic,
        "tail_deficit_cap_event_count_upper_bounds": upper_bounds,
        "first_observed_tail_deficit_cap_reaching_30": _first_cap_reaching(
            upper_bounds, mechanism_event_minimum
        ),
        "first_observed_tail_deficit_cap_reaching_40": _first_cap_reaching(
            upper_bounds, full_model_event_minimum
        ),
        "member_window_gap_row_count": int(len(member_gap)),
        "member_window_gap_primary_reason_counts": {
            str(key): int(value)
            for key, value in member_gap["primary_gap_reason"].value_counts().to_dict().items()
        },
        **member_state_metrics,
        "diagnostic_authority": "NON_ADMISSIBLE_DIAGNOSTIC_ONLY",
        "g1_gate_promoted": False,
        "model_trained": False,
        "prediction_metric_generated": False,
        "performance_artifact_read": False,
        "position_impact": 0,
    }
    audit_state = {
        "audit_status": "PASS_HISTORICAL_COVERAGE_AUDIT_G1_REMAINS_NO_VIEW",
        "G1_DATA_AND_EVENTS": "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY",
        "G1_MECHANISM_DISCOVERY_PREREQUISITE": "FAIL_UNCHANGED",
        "G1_FULL_THREE_COEFFICIENT_MODEL_PREREQUISITE": "FAIL_UNCHANGED",
        "branch_state": "STOPPED_AT_G1_POST_G1_DIAGNOSTIC_ONLY",
        "next_allowed_step": "FREEZE_OFFICIAL_HISTORICAL_SUSPENSION_SOURCE_PROBE_AND_PREHISTORY_INDEX_CORRECTION_REVIEW",
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_evaluation": "NOT_ALLOWED",
        "paper_or_shadow": "NOT_AUTHORIZED",
        "broker_connection": "NOT_AUTHORIZED",
        "position_impact": 0,
    }
    artifacts = HistoricalCoverageAuditArtifacts(
        prehistory_coverage_comparison=comparison.reset_index(drop=True),
        event_gap_ledger=event_gap.reset_index(drop=True),
        member_window_gap_ledger=member_gap.reset_index(drop=True),
        metrics=metrics,
        audit_state=audit_state,
    )
    _assert_output_boundary(artifacts)
    if len(positives) != metrics["total_positive_origin_count"]:
        raise HistoricalCoverageAuditError("正样本诊断行数漂移")
    return artifacts
