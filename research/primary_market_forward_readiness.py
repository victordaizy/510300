"""评估510300一级—二级市场前向样本是否达到研究门槛。"""

from __future__ import annotations

from typing import Any

import pandas as pd


def evaluate_forward_readiness(
    pcf: pd.DataFrame,
    iopv: pd.DataFrame,
    *,
    minimum_full_coverage_days: int,
    recommended_full_coverage_days: int,
    minimum_snapshots_per_day: int,
    first_unseen_evaluation_full_coverage_days: int = 80,
    replication_full_coverage_days: int = 120,
    timezone: str = "Asia/Shanghai",
    maximum_iopv_staleness_seconds: float = 180.0,
    maximum_negative_clock_skew_seconds: float = 5.0,
) -> dict[str, Any]:
    """按交易日聚类核验PCF/IOPV质量与20/40/80/120阶段。"""

    if minimum_full_coverage_days <= 0:
        raise ValueError("最低完整覆盖日数必须为正数")
    if recommended_full_coverage_days < minimum_full_coverage_days:
        raise ValueError("建议完整覆盖日数不能小于最低门槛")
    if minimum_snapshots_per_day <= 0:
        raise ValueError("每日最低快照数必须为正数")
    if first_unseen_evaluation_full_coverage_days < recommended_full_coverage_days:
        raise ValueError("首次未见评价日数不能小于特征冻结门槛")
    if replication_full_coverage_days < first_unseen_evaluation_full_coverage_days:
        raise ValueError("复制段门槛不能小于首次未见评价门槛")
    if maximum_iopv_staleness_seconds < 0 or maximum_negative_clock_skew_seconds < 0:
        raise ValueError("IOPV陈旧度与时钟负偏差容忍必须非负")

    required_pcf = {"trading_day", "retrieved_at"}
    required_iopv = {
        "trade_date",
        "exchange_timestamp",
        "retrieved_at",
        "iopv",
    }
    missing_pcf = required_pcf - set(pcf.columns)
    missing_iopv = required_iopv - set(iopv.columns)
    if missing_pcf:
        raise ValueError(f"PCF缺少质量字段：{sorted(missing_pcf)}")
    if missing_iopv:
        raise ValueError(f"IOPV缺少质量字段：{sorted(missing_iopv)}")

    normalized_pcf = pcf.copy()
    normalized_pcf["trading_day"] = pd.to_datetime(
        normalized_pcf["trading_day"], errors="coerce"
    ).dt.date
    if normalized_pcf["trading_day"].isna().any():
        raise ValueError("PCF包含非法trading_day")
    normalized_pcf["retrieved_at_utc"] = pd.to_datetime(
        normalized_pcf["retrieved_at"], errors="coerce", utc=True
    )
    normalized_pcf["retrieved_timezone_explicit"] = normalized_pcf[
        "retrieved_at"
    ].astype(str).str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True)

    normalized = iopv.copy()
    normalized["trade_date"] = pd.to_datetime(
        normalized["trade_date"], errors="coerce"
    ).dt.date
    normalized["exchange_timestamp_local"] = pd.to_datetime(
        normalized["exchange_timestamp"], errors="coerce"
    )
    if normalized["trade_date"].isna().any():
        raise ValueError("IOPV包含非法trade_date")
    if normalized["exchange_timestamp_local"].isna().any():
        raise ValueError("IOPV包含非法exchange_timestamp")
    if normalized["exchange_timestamp_local"].dt.tz is None:
        normalized["exchange_timestamp_local"] = normalized[
            "exchange_timestamp_local"
        ].dt.tz_localize(timezone)
    else:
        normalized["exchange_timestamp_local"] = normalized[
            "exchange_timestamp_local"
        ].dt.tz_convert(timezone)
    normalized["retrieved_at_utc"] = pd.to_datetime(
        normalized["retrieved_at"], errors="coerce", utc=True
    )
    normalized["retrieved_at_local"] = normalized["retrieved_at_utc"].dt.tz_convert(
        timezone
    )
    normalized["retrieved_timezone_explicit"] = normalized["retrieved_at"].astype(
        str
    ).str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", regex=True)
    normalized["clock_delay_seconds"] = (
        normalized["retrieved_at_local"] - normalized["exchange_timestamp_local"]
    ).dt.total_seconds()
    normalized["exchange_date_matches_trade_date"] = normalized[
        "exchange_timestamp_local"
    ].dt.date.eq(normalized["trade_date"])
    normalized["positive_iopv"] = pd.to_numeric(
        normalized["iopv"], errors="coerce"
    ).gt(0)

    deduplicated = normalized.drop_duplicates(
        ["trade_date", "exchange_timestamp_local"], keep="last"
    ).copy()
    raw_counts = normalized.groupby("trade_date").size()
    daily_counts = deduplicated.groupby("trade_date").size().sort_index()
    pcf_counts = normalized_pcf.groupby("trading_day").size()
    pcf_dates = set(normalized_pcf["trading_day"])
    daily_quality: list[dict[str, Any]] = []
    full_dates: list[Any] = []
    for trade_date, count in daily_counts.items():
        day = deduplicated.loc[deduplicated["trade_date"].eq(trade_date)]
        pcf_day = normalized_pcf.loc[
            normalized_pcf["trading_day"].eq(trade_date)
        ]
        clock_delay = pd.to_numeric(day["clock_delay_seconds"], errors="coerce")
        checks = {
            "minimum_deduplicated_snapshots": int(count) >= minimum_snapshots_per_day,
            "pcf_date_present_once": int(pcf_counts.get(trade_date, 0)) == 1,
            "pcf_retrieval_timestamp_valid": bool(
                len(pcf_day) == 1
                and pcf_day["retrieved_at_utc"].notna().all()
                and pcf_day["retrieved_timezone_explicit"].all()
            ),
            "exchange_timestamp_valid": bool(
                day["exchange_timestamp_local"].notna().all()
                and day["exchange_date_matches_trade_date"].all()
            ),
            "retrieval_timezone_valid": bool(
                day["retrieved_at_utc"].notna().all()
                and day["retrieved_timezone_explicit"].all()
            ),
            "iopv_staleness_within_limit": bool(
                clock_delay.notna().all()
                and clock_delay.le(maximum_iopv_staleness_seconds).all()
                and clock_delay.ge(-maximum_negative_clock_skew_seconds).all()
            ),
            "positive_iopv": bool(day["positive_iopv"].all()),
        }
        complete = all(checks.values())
        if complete:
            full_dates.append(trade_date)
        daily_quality.append(
            {
                "trade_date": str(trade_date),
                "raw_snapshot_count": int(raw_counts.get(trade_date, 0)),
                "deduplicated_snapshot_count": int(count),
                "duplicate_snapshot_count": int(raw_counts.get(trade_date, 0) - count),
                "pcf_record_count": int(pcf_counts.get(trade_date, 0)),
                "maximum_iopv_staleness_seconds": (
                    float(clock_delay.max()) if clock_delay.notna().any() else None
                ),
                "minimum_iopv_staleness_seconds": (
                    float(clock_delay.min()) if clock_delay.notna().any() else None
                ),
                "checks": checks,
                "complete_quality_day": complete,
                "failure_reasons": [name for name, passed in checks.items() if not passed],
            }
        )

    full_days = len(full_dates)
    if full_days >= replication_full_coverage_days:
        status = "REPLICATION_EVALUATION_ELIGIBLE"
        stage = "STAGE_120_REPLICATION"
    elif full_days >= first_unseen_evaluation_full_coverage_days:
        status = "FIRST_UNSEEN_EVALUATION_ELIGIBLE"
        stage = "STAGE_80_FIRST_UNSEEN_EVALUATION"
    elif full_days >= recommended_full_coverage_days:
        status = "FEATURE_FREEZE_ELIGIBLE_NOT_EVALUATION"
        stage = "STAGE_40_FEATURE_FREEZE"
    elif full_days >= minimum_full_coverage_days:
        status = "QUALITY_AUDIT_ELIGIBLE_NOT_FEATURE_FREEZE"
        stage = "STAGE_20_QUALITY_AUDIT"
    else:
        status = "COLLECTING_NOT_ELIGIBLE"
        stage = "STAGE_0_19_COLLECTION"

    observed_dates = list(daily_counts.index)
    return {
        "status": status,
        "stage": stage,
        "statistical_cluster_unit": "trade_date",
        "observed_trading_days": int(len(observed_dates)),
        "full_coverage_days": full_days,
        "minimum_full_coverage_days": minimum_full_coverage_days,
        "recommended_full_coverage_days": recommended_full_coverage_days,
        "first_unseen_evaluation_full_coverage_days": first_unseen_evaluation_full_coverage_days,
        "replication_full_coverage_days": replication_full_coverage_days,
        "minimum_snapshots_per_day": minimum_snapshots_per_day,
        "minimum_progress": min(1.0, full_days / minimum_full_coverage_days),
        "recommended_progress": min(1.0, full_days / recommended_full_coverage_days),
        "first_unseen_evaluation_progress": min(
            1.0, full_days / first_unseen_evaluation_full_coverage_days
        ),
        "replication_progress": min(1.0, full_days / replication_full_coverage_days),
        "first_observed_trade_date": str(observed_dates[0]) if observed_dates else None,
        "last_observed_trade_date": str(observed_dates[-1]) if observed_dates else None,
        "daily_snapshot_counts": {
            str(trade_date): int(count) for trade_date, count in daily_counts.items()
        },
        "daily_quality": daily_quality,
        "quality_contract": {
            "timezone": timezone,
            "maximum_iopv_staleness_seconds": maximum_iopv_staleness_seconds,
            "maximum_negative_clock_skew_seconds": maximum_negative_clock_skew_seconds,
            "pcf_date_must_equal_trade_date": True,
            "minimum_deduplicated_snapshots_per_day": minimum_snapshots_per_day,
        },
        "eligible_for_quality_audit": full_days >= minimum_full_coverage_days,
        "eligible_for_feature_freeze": full_days >= recommended_full_coverage_days,
        "eligible_for_first_unseen_evaluation": (
            full_days >= first_unseen_evaluation_full_coverage_days
        ),
        "eligible_for_research_evaluation": (
            full_days >= first_unseen_evaluation_full_coverage_days
        ),
        "eligible_for_replication_evaluation": (
            full_days >= replication_full_coverage_days
        ),
        "eligible_for_position_mapping": False,
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
        "position_mapping_block_reason": (
            "PCF/IOPV链只用于研究收集；任何阶段都不自动授权仓位映射"
        ),
    }
