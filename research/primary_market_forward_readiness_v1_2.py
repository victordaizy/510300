"""PCF/IOPV V1.2 成熟度：从 2026-08-26 起强制官方日端点复核。"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

import pandas as pd

from research.primary_market_forward_readiness import evaluate_forward_readiness


def _time(value: str) -> time:
    try:
        return datetime.strptime(value, "%H:%M:%S").time()
    except ValueError as exc:
        raise ValueError(f"无法识别最终快照时间门槛：{value}") from exc


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _stage(
    full_days: int,
    minimum_full_coverage_days: int,
    recommended_full_coverage_days: int,
    first_unseen_evaluation_full_coverage_days: int,
    replication_full_coverage_days: int,
) -> tuple[str, str]:
    if full_days >= replication_full_coverage_days:
        return "REPLICATION_EVALUATION_ELIGIBLE", "STAGE_120_REPLICATION"
    if full_days >= first_unseen_evaluation_full_coverage_days:
        return (
            "FIRST_UNSEEN_EVALUATION_ELIGIBLE",
            "STAGE_80_FIRST_UNSEEN_EVALUATION",
        )
    if full_days >= recommended_full_coverage_days:
        return (
            "FEATURE_FREEZE_ELIGIBLE_NOT_EVALUATION",
            "STAGE_40_FEATURE_FREEZE",
        )
    if full_days >= minimum_full_coverage_days:
        return (
            "QUALITY_AUDIT_ELIGIBLE_NOT_FEATURE_FREEZE",
            "STAGE_20_QUALITY_AUDIT",
        )
    return "COLLECTING_NOT_ELIGIBLE", "STAGE_0_19_COLLECTION"


def evaluate_forward_readiness_v1_2(
    pcf: pd.DataFrame,
    iopv: pd.DataFrame,
    daily_crosscheck: pd.DataFrame,
    *,
    minimum_full_coverage_days: int,
    recommended_full_coverage_days: int,
    minimum_snapshots_per_day: int,
    first_unseen_evaluation_full_coverage_days: int = 80,
    replication_full_coverage_days: int = 120,
    timezone: str = "Asia/Shanghai",
    maximum_iopv_staleness_seconds: float = 180.0,
    maximum_negative_clock_skew_seconds: float = 5.0,
    crosscheck_required_from: date,
    minimum_final_snapshot_time: str,
    maximum_ohl_price_difference_cny: float,
    maximum_close_price_difference_cny: float,
    fund_code: str = "510300",
) -> dict[str, Any]:
    """沿用原成熟度门槛，并对新日期增加可重放的日端点证据门。"""

    if maximum_ohl_price_difference_cny < 0:
        raise ValueError("OHLC 复核容差不能为负数")
    if maximum_close_price_difference_cny < 0:
        raise ValueError("收盘价复核容差不能为负数")
    final_time = _time(minimum_final_snapshot_time)
    result = evaluate_forward_readiness(
        pcf,
        iopv,
        minimum_full_coverage_days=minimum_full_coverage_days,
        recommended_full_coverage_days=recommended_full_coverage_days,
        minimum_snapshots_per_day=minimum_snapshots_per_day,
        first_unseen_evaluation_full_coverage_days=(
            first_unseen_evaluation_full_coverage_days
        ),
        replication_full_coverage_days=replication_full_coverage_days,
        timezone=timezone,
        maximum_iopv_staleness_seconds=maximum_iopv_staleness_seconds,
        maximum_negative_clock_skew_seconds=maximum_negative_clock_skew_seconds,
    )

    normalized_iopv = iopv.copy()
    normalized_iopv["trade_date_normalized"] = pd.to_datetime(
        normalized_iopv["trade_date"], errors="coerce"
    ).dt.date
    normalized_iopv["exchange_timestamp_normalized"] = pd.to_datetime(
        normalized_iopv["exchange_timestamp"], errors="coerce"
    )
    if normalized_iopv["exchange_timestamp_normalized"].dt.tz is not None:
        normalized_iopv["exchange_timestamp_normalized"] = normalized_iopv[
            "exchange_timestamp_normalized"
        ].dt.tz_convert(timezone).dt.tz_localize(None)

    if daily_crosscheck.empty:
        normalized_crosscheck = pd.DataFrame()
    else:
        normalized_crosscheck = daily_crosscheck.copy()
        required_columns = {
            "trade_date",
            "fund_code",
            "open",
            "high",
            "low",
            "close",
            "raw_hash_verified",
            "receipt_verified",
        }
        missing = required_columns - set(normalized_crosscheck.columns)
        if missing:
            raise ValueError(f"官方日端点复核缺少字段：{sorted(missing)}")
        normalized_crosscheck["trade_date_normalized"] = pd.to_datetime(
            normalized_crosscheck["trade_date"], errors="coerce"
        ).dt.date

    required_day_count = 0
    passed_crosscheck_day_count = 0
    legacy_complete_day_count = 0
    for quality in result["daily_quality"]:
        trade_date = date.fromisoformat(str(quality["trade_date"]))
        if trade_date < crosscheck_required_from:
            quality["checks"]["official_daily_crosscheck_not_required_legacy"] = True
            quality["crosscheck"] = {
                "required": False,
                "effective_from": crosscheck_required_from.isoformat(),
            }
            quality["complete_quality_day"] = all(quality["checks"].values())
            quality["failure_reasons"] = [
                name for name, passed in quality["checks"].items() if not passed
            ]
            if quality["complete_quality_day"]:
                legacy_complete_day_count += 1
            continue

        required_day_count += 1
        day_iopv = normalized_iopv.loc[
            normalized_iopv["trade_date_normalized"].eq(trade_date)
        ].sort_values("exchange_timestamp_normalized")
        if normalized_crosscheck.empty:
            day_crosscheck = normalized_crosscheck
        else:
            day_crosscheck = normalized_crosscheck.loc[
                normalized_crosscheck["trade_date_normalized"].eq(trade_date)
                & normalized_crosscheck["fund_code"].astype(str).eq(fund_code)
            ]
        present_once = len(day_crosscheck) == 1
        final_snapshot_present = False
        ohl_match = False
        close_match = False
        raw_hash_verified = False
        receipt_verified = False
        differences: dict[str, float | None] = {
            "open": None,
            "high": None,
            "low": None,
            "close": None,
        }
        if not day_iopv.empty:
            latest = day_iopv.iloc[-1]
            latest_timestamp = latest["exchange_timestamp_normalized"]
            final_snapshot_present = bool(
                pd.notna(latest_timestamp) and latest_timestamp.time() >= final_time
            )
        else:
            latest = None
        if present_once:
            summary = day_crosscheck.iloc[0]
            raw_hash_verified = _as_bool(summary["raw_hash_verified"])
            receipt_verified = _as_bool(summary["receipt_verified"])
            if latest is not None:
                for field in ("open", "high", "low"):
                    left = pd.to_numeric(pd.Series([latest.get(field)]), errors="coerce").iloc[0]
                    right = pd.to_numeric(
                        pd.Series([summary.get(field)]), errors="coerce"
                    ).iloc[0]
                    if pd.notna(left) and pd.notna(right):
                        differences[field] = abs(float(left) - float(right))
                last_price = pd.to_numeric(
                    pd.Series([latest.get("last_price")]), errors="coerce"
                ).iloc[0]
                close_price = pd.to_numeric(
                    pd.Series([summary.get("close")]), errors="coerce"
                ).iloc[0]
                if pd.notna(last_price) and pd.notna(close_price):
                    differences["close"] = abs(float(last_price) - float(close_price))
            ohl_match = all(
                differences[field] is not None
                and float(differences[field]) <= maximum_ohl_price_difference_cny
                for field in ("open", "high", "low")
            )
            close_match = bool(
                differences["close"] is not None
                and float(differences["close"])
                <= maximum_close_price_difference_cny
            )
        new_checks = {
            "official_daily_crosscheck_present_once": present_once,
            "official_daily_crosscheck_raw_hash_verified": raw_hash_verified,
            "official_daily_crosscheck_receipt_verified": receipt_verified,
            "official_final_snapshot_present": final_snapshot_present,
            "official_daily_ohl_match": ohl_match,
            "official_daily_close_match": close_match,
        }
        quality["checks"].update(new_checks)
        quality["crosscheck"] = {
            "required": True,
            "effective_from": crosscheck_required_from.isoformat(),
            "source_id": "SSE_ETF_DAILY_TURNOVER_OFFICIAL",
            "price_differences_cny": differences,
            "maximum_ohl_price_difference_cny": maximum_ohl_price_difference_cny,
            "maximum_close_price_difference_cny": maximum_close_price_difference_cny,
            "minimum_final_snapshot_time": minimum_final_snapshot_time,
        }
        quality["complete_quality_day"] = all(quality["checks"].values())
        quality["failure_reasons"] = [
            name for name, passed in quality["checks"].items() if not passed
        ]
        if all(new_checks.values()):
            passed_crosscheck_day_count += 1

    full_days = sum(
        bool(item["complete_quality_day"]) for item in result["daily_quality"]
    )
    status, stage = _stage(
        full_days,
        minimum_full_coverage_days,
        recommended_full_coverage_days,
        first_unseen_evaluation_full_coverage_days,
        replication_full_coverage_days,
    )
    result.update(
        {
            "status": status,
            "stage": stage,
            "full_coverage_days": full_days,
            "minimum_progress": min(1.0, full_days / minimum_full_coverage_days),
            "recommended_progress": min(
                1.0, full_days / recommended_full_coverage_days
            ),
            "first_unseen_evaluation_progress": min(
                1.0, full_days / first_unseen_evaluation_full_coverage_days
            ),
            "replication_progress": min(
                1.0, full_days / replication_full_coverage_days
            ),
            "eligible_for_quality_audit": full_days >= minimum_full_coverage_days,
            "eligible_for_feature_freeze": full_days
            >= recommended_full_coverage_days,
            "eligible_for_first_unseen_evaluation": full_days
            >= first_unseen_evaluation_full_coverage_days,
            "eligible_for_research_evaluation": full_days
            >= first_unseen_evaluation_full_coverage_days,
            "eligible_for_replication_evaluation": full_days
            >= replication_full_coverage_days,
            "feature_freeze_block_reason": (
                None
                if full_days >= recommended_full_coverage_days
                else "完整质量日不足40，只允许继续采集和质量审计"
            ),
            "evaluation_block_reason": (
                None
                if full_days >= first_unseen_evaluation_full_coverage_days
                else "完整质量日不足80，不允许读取未见段收益评价"
            ),
            "crosscheck_required_day_count": required_day_count,
            "crosscheck_passed_day_count": passed_crosscheck_day_count,
            "legacy_complete_quality_day_count": legacy_complete_day_count,
        }
    )
    result["quality_contract"].update(
        {
            "official_daily_crosscheck_source_id": (
                "SSE_ETF_DAILY_TURNOVER_OFFICIAL"
            ),
            "official_daily_crosscheck_required_from": (
                crosscheck_required_from.isoformat()
            ),
            "minimum_final_snapshot_time": minimum_final_snapshot_time,
            "maximum_ohl_price_difference_cny": (
                maximum_ohl_price_difference_cny
            ),
            "maximum_close_price_difference_cny": (
                maximum_close_price_difference_cny
            ),
            "raw_hash_and_receipt_must_verify": True,
        }
    )
    return result
