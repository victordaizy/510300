"""510300 PIT 盈余信息扩散候选的 outcome-blind 数据准入工具。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GateResult:
    """单个硬门的结构化结果。"""

    passed: bool
    actual: Any
    required: Any
    note: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "actual": self.actual,
            "required": self.required,
            "note": self.note,
        }


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 对象。"""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return value


def assert_boolean_audit_flags_false(frame: pd.DataFrame, label: str) -> None:
    """保证输入自带的价格/未来收益读取审计标记均为假。"""

    for column in ("market_price_read", "future_return_read"):
        if column in frame and frame[column].fillna(False).astype(bool).any():
            raise ValueError(f"{label} 的 {column} 出现 true")


def validate_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    """校验冻结代码/协议文件与固定输入证据哈希。"""

    manifest = read_json(manifest_path)
    expected_status = (
        "FROZEN_OUTCOME_BLIND_PIT_EARNINGS_DATA_FEASIBILITY_"
        "BEFORE_ANY_POST_EVENT_OUTCOME"
    )
    if manifest.get("status") != expected_status:
        raise ValueError("备用分支数据可行性清单状态不正确")
    mismatches: list[str] = []
    for section in ("files", "fixed_input_evidence"):
        for item in manifest.get(section, []):
            path = root / str(item["path"])
            if not path.is_file() or sha256_file(path) != str(item["sha256"]):
                mismatches.append(str(item["path"]))
    if mismatches:
        raise ValueError(f"冻结文件或固定输入证据哈希漂移：{mismatches}")
    return manifest


def validate_membership_intervals(intervals: pd.DataFrame) -> dict[str, Any]:
    """验证历史成员区间的结构完整性。"""

    required = {"symbol", "opt_in", "opt_out", "source", "source_license"}
    missing = sorted(required.difference(intervals.columns))
    if missing:
        raise ValueError(f"历史成分区间缺少字段：{missing}")
    data = intervals.copy()
    data["opt_in"] = pd.to_datetime(data["opt_in"], errors="raise")
    data["opt_out"] = pd.to_datetime(data["opt_out"], errors="coerce")
    duplicate_count = int(data.duplicated(["symbol", "opt_in"]).sum())
    invalid_order_count = int(
        (
            data["opt_out"].notna()
            & data["opt_out"].le(data["opt_in"])
        ).sum()
    )
    overlap_count = 0
    for _, group in data.sort_values(["symbol", "opt_in"]).groupby("symbol"):
        previous_end: pd.Timestamp | None = None
        for row in group.itertuples(index=False):
            if previous_end is not None and pd.Timestamp(row.opt_in) < previous_end:
                overlap_count += 1
            if pd.notna(row.opt_out):
                previous_end = pd.Timestamp(row.opt_out)
            else:
                previous_end = pd.Timestamp.max
    return {
        "row_count": int(len(data)),
        "unique_symbol_count": int(data["symbol"].nunique()),
        "duplicate_symbol_opt_in_rows": duplicate_count,
        "invalid_interval_order_rows": invalid_order_count,
        "overlapping_interval_rows": overlap_count,
        "first_opt_in": str(data["opt_in"].min()),
        "last_opt_in": str(data["opt_in"].max()),
        "open_interval_count": int(data["opt_out"].isna().sum()),
        "sources": data["source"].value_counts(dropna=False).to_dict(),
        "source_licenses": data["source_license"].value_counts(dropna=False).to_dict(),
    }


def validate_weights(weights: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """验证月度历史权重快照。"""

    required = {
        "index_code",
        "con_code",
        "trade_date",
        "weight",
        "source",
        "retrieved_at",
    }
    missing = sorted(required.difference(weights.columns))
    if missing:
        raise ValueError(f"历史权重缺少字段：{missing}")
    data = weights.copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"], errors="raise")
    data["weight"] = pd.to_numeric(data["weight"], errors="raise")
    duplicate_count = int(data.duplicated(["trade_date", "con_code"]).sum())
    grouped = data.groupby("trade_date").agg(
        member_count=("con_code", "nunique"),
        weight_sum_percent=("weight", "sum"),
    )
    summary = {
        "row_count": int(len(data)),
        "snapshot_count": int(data["trade_date"].nunique()),
        "first_snapshot": str(data["trade_date"].min()),
        "last_snapshot": str(data["trade_date"].max()),
        "duplicate_date_security_rows": duplicate_count,
        "member_count_min": int(grouped["member_count"].min()),
        "member_count_median": float(grouped["member_count"].median()),
        "member_count_max": int(grouped["member_count"].max()),
        "weight_sum_min_percent": float(grouped["weight_sum_percent"].min()),
        "weight_sum_median_percent": float(grouped["weight_sum_percent"].median()),
        "weight_sum_max_percent": float(grouped["weight_sum_percent"].max()),
        "sources": data["source"].value_counts(dropna=False).to_dict(),
        "retrieved_at_min": str(data["retrieved_at"].min()),
        "retrieved_at_max": str(data["retrieved_at"].max()),
    }
    return data, summary


def active_member_counts_on_dates(
    intervals: pd.DataFrame,
    dates: Iterable[pd.Timestamp],
) -> pd.Series:
    """在公告日期集合上重建成员数量，不接触市场日线。"""

    data = intervals.copy()
    data["opt_in"] = pd.to_datetime(data["opt_in"])
    data["opt_out"] = pd.to_datetime(data["opt_out"])
    result: dict[pd.Timestamp, int] = {}
    for raw_date in sorted({pd.Timestamp(value).normalize() for value in dates}):
        active = data["opt_in"].le(raw_date) & (
            data["opt_out"].isna() | data["opt_out"].gt(raw_date)
        )
        result[raw_date] = int(active.sum())
    return pd.Series(result, name="active_member_count", dtype="int64")


def build_member_event_audit(
    metadata: pd.DataFrame,
    inventory: pd.DataFrame,
    intervals: pd.DataFrame,
    weights: pd.DataFrame,
    trading_calendar: pd.DataFrame,
    maximum_snapshot_age_days: int,
    maximum_availability_lag_days: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """构造不含盈利数值、价格或未来收益的成员公告覆盖表。"""

    metadata_required = {
        "announcement_id",
        "ts_code",
        "announcement_type",
        "target_title_type",
        "announcement_timestamp_at",
        "status",
        "market_price_read",
        "future_return_read",
    }
    missing = sorted(metadata_required.difference(metadata.columns))
    if missing:
        raise ValueError(f"官方公告元数据缺少字段：{missing}")
    assert_boolean_audit_flags_false(metadata, "官方公告元数据")
    assert_boolean_audit_flags_false(inventory, "官方PDF清单")

    target = metadata.loc[metadata["target_title_type"].fillna(False)].copy()
    target["announcement_id"] = target["announcement_id"].astype(str)
    target["announcement_timestamp_at"] = pd.to_datetime(
        target["announcement_timestamp_at"], errors="raise"
    )
    target["notice_date"] = (
        target["announcement_timestamp_at"].dt.tz_localize(None).dt.normalize()
    )
    target["notice_time_hour"] = target["announcement_timestamp_at"].dt.hour
    target["notice_time_precision"] = np.where(
        target["notice_time_hour"].eq(0), "DATE_ONLY_CONSERVATIVE", "RAW_TIME_PRESERVED"
    )
    duplicate_announcement_ids = int(target.duplicated("announcement_id").sum())
    if duplicate_announcement_ids:
        raise ValueError("目标公告ID不唯一")

    calendar_required = {"exchange", "date", "is_open", "available_at", "source"}
    calendar_missing = sorted(calendar_required.difference(trading_calendar.columns))
    if calendar_missing:
        raise ValueError(f"交易日历缺少字段：{calendar_missing}")
    calendar = trading_calendar.loc[trading_calendar["is_open"].fillna(False)].copy()
    calendar["date"] = pd.to_datetime(calendar["date"], errors="raise").dt.normalize()
    open_dates = pd.DataFrame(
        {"availability_date": sorted(calendar["date"].drop_duplicates().tolist())}
    )
    target = pd.merge_asof(
        target.sort_values("notice_date"),
        open_dates,
        left_on="notice_date",
        right_on="availability_date",
        direction="forward",
        allow_exact_matches=False,
    )
    target["notice_to_availability_lag_days"] = (
        target["availability_date"] - target["notice_date"]
    ).dt.days
    target["availability_calendar_covered"] = (
        target["availability_date"].notna()
        & target["notice_to_availability_lag_days"].between(
            1, maximum_availability_lag_days
        )
    )
    calendar_uncovered_target_count = int(
        (~target["availability_calendar_covered"]).sum()
    )
    eligible_target = target.loc[target["availability_calendar_covered"]].copy()

    member_intervals = intervals[
        ["symbol", "opt_in", "opt_out", "source", "source_license"]
    ].copy()
    member_intervals["opt_in"] = pd.to_datetime(member_intervals["opt_in"])
    member_intervals["opt_out"] = pd.to_datetime(member_intervals["opt_out"])
    joined = eligible_target.merge(
        member_intervals,
        left_on="ts_code",
        right_on="symbol",
        how="left",
        validate="many_to_many",
    )
    active = joined["opt_in"].notna() & joined["opt_in"].le(
        joined["availability_date"]
    ) & (
        joined["opt_out"].isna()
        | joined["availability_date"].lt(joined["opt_out"])
    )
    events = joined.loc[active].copy()
    if events["announcement_id"].duplicated().any():
        raise ValueError("历史成员区间使同一公告匹配多个活动区间")

    snapshots = pd.DataFrame(
        {"weight_snapshot_date": sorted(pd.to_datetime(weights["trade_date"]).unique())}
    )
    events = pd.merge_asof(
        events.sort_values("notice_date"),
        snapshots.sort_values("weight_snapshot_date"),
        left_on="notice_date",
        right_on="weight_snapshot_date",
        direction="backward",
        allow_exact_matches=False,
    )
    weight_values = weights[
        ["trade_date", "con_code", "weight", "source", "retrieved_at"]
    ].rename(
        columns={
            "trade_date": "weight_snapshot_date",
            "source": "weight_source",
            "retrieved_at": "weight_retrieved_at",
        }
    )
    events = events.merge(
        weight_values,
        left_on=["weight_snapshot_date", "ts_code"],
        right_on=["weight_snapshot_date", "con_code"],
        how="left",
        validate="many_to_one",
    )
    events["weight_snapshot_age_days"] = (
        events["notice_date"] - events["weight_snapshot_date"]
    ).dt.days
    events["valid_strict_prior_weight"] = (
        events["weight"].notna()
        & events["weight_snapshot_age_days"].between(1, maximum_snapshot_age_days)
    )

    pdf = inventory.copy()
    pdf["announcement_id"] = pdf["announcement_id"].astype(str)
    if pdf["announcement_id"].duplicated().any():
        raise ValueError("官方PDF清单公告ID不唯一")
    events = events.merge(
        pdf[["announcement_id", "pdf_status", "pdf_sha256"]],
        on="announcement_id",
        how="left",
        validate="one_to_one",
    )
    events["pdf_available"] = events["pdf_status"].isin(
        {"PASS_DOWNLOADED_VERIFIED_PDF", "PASS_REUSED_VERIFIED_PDF"}
    )
    events["earliest_allowed_feature_use"] = events["availability_date"]
    events["market_price_read"] = False
    events["future_return_read"] = False

    first_eligible = events.loc[events["valid_strict_prior_weight"], "notice_date"].min()
    after_start = (
        events.loc[events["notice_date"].ge(first_eligible)].copy()
        if pd.notna(first_eligible)
        else events.iloc[0:0].copy()
    )
    weight_coverage_after_start = (
        float(after_start["valid_strict_prior_weight"].mean())
        if len(after_start)
        else 0.0
    )
    summary = {
        "all_target_announcement_count": int(len(target)),
        "calendar_covered_target_announcement_count": int(len(eligible_target)),
        "calendar_uncovered_target_announcement_count": calendar_uncovered_target_count,
        "trading_calendar_first_open_date": str(open_dates["availability_date"].min()),
        "trading_calendar_last_open_date": str(open_dates["availability_date"].max()),
        "duplicate_target_announcement_ids": duplicate_announcement_ids,
        "member_target_announcement_count": int(len(events)),
        "member_unique_issuer_count": int(events["ts_code"].nunique()),
        "member_distinct_notice_date_count": int(events["notice_date"].nunique()),
        "member_announcement_count_by_year": (
            events["notice_date"].dt.year.value_counts().sort_index().to_dict()
        ),
        "member_announcement_count_by_type": events[
            "announcement_type"
        ].value_counts().to_dict(),
        "date_only_conservative_count": int(
            events["notice_time_precision"].eq("DATE_ONLY_CONSERVATIVE").sum()
        ),
        "date_only_conservative_ratio": float(
            events["notice_time_precision"].eq("DATE_ONLY_CONSERVATIVE").mean()
        ),
        "member_pdf_available_count": int(events["pdf_available"].sum()),
        "member_pdf_unavailable_count": int((~events["pdf_available"]).sum()),
        "valid_strict_prior_weight_count": int(
            events["valid_strict_prior_weight"].sum()
        ),
        "valid_strict_prior_weight_ratio_all_history": float(
            events["valid_strict_prior_weight"].mean()
        ),
        "first_valid_strict_prior_weight_event_date": (
            str(first_eligible) if pd.notna(first_eligible) else None
        ),
        "weight_coverage_after_first_eligible_date": weight_coverage_after_start,
        "weight_snapshot_age_days_min": (
            int(events.loc[events["weight"].notna(), "weight_snapshot_age_days"].min())
            if events["weight"].notna().any()
            else None
        ),
        "weight_snapshot_age_days_median": (
            float(events.loc[events["weight"].notna(), "weight_snapshot_age_days"].median())
            if events["weight"].notna().any()
            else None
        ),
        "weight_snapshot_age_days_max": (
            int(events.loc[events["weight"].notna(), "weight_snapshot_age_days"].max())
            if events["weight"].notna().any()
            else None
        ),
    }
    keep = [
        "announcement_id",
        "ts_code",
        "announcement_type",
        "announcement_timestamp_at",
        "notice_date",
        "availability_date",
        "notice_to_availability_lag_days",
        "notice_time_precision",
        "opt_in",
        "opt_out",
        "source",
        "source_license",
        "weight_snapshot_date",
        "weight",
        "weight_snapshot_age_days",
        "weight_source",
        "weight_retrieved_at",
        "valid_strict_prior_weight",
        "pdf_status",
        "pdf_sha256",
        "pdf_available",
        "earliest_allowed_feature_use",
        "market_price_read",
        "future_return_read",
    ]
    return events[keep].sort_values(["notice_date", "ts_code", "announcement_id"]), summary


def gate(
    passed: bool,
    actual: Any,
    required: Any,
    note: str,
) -> dict[str, Any]:
    return GateResult(bool(passed), actual, required, note).as_dict()


def choose_status(
    *,
    activation_passed: bool,
    metadata_passed: bool,
    membership_weights_passed: bool,
    facts_progress_status: str,
    facts_automation_status: str | None,
    facts_automation_passed: bool | None,
    human_review_completed: bool,
    facts_prevalence_passed: bool | None,
    statuses: dict[str, str],
) -> str:
    """按优先级选择权威状态，避免把运行中或待复核误写成失败。"""

    if not activation_passed:
        return statuses["activation_not_allowed"]
    if not metadata_passed:
        return statuses["blocked_metadata"]
    if not membership_weights_passed:
        return statuses["blocked_membership_or_weights"]
    if facts_progress_status == "RUNNING_MULTIPROCESS_FACT_CHECKPOINT_EXECUTION":
        return statuses["running_upstream"]
    if facts_progress_status == "PASS_MULTIPROCESS_FACT_TASKS_ATTEMPTED_WITH_EXPLICIT_RESULTS" and facts_automation_status is None:
        return statuses["ready_for_consolidation"]
    if facts_automation_passed is False or facts_prevalence_passed is False:
        return statuses["blocked_facts"]
    if facts_automation_passed and not human_review_completed:
        return statuses["pending_human_review"]
    if facts_automation_passed and human_review_completed and facts_prevalence_passed:
        return statuses["pass_data_only"]
    return statuses["blocked_facts"]
