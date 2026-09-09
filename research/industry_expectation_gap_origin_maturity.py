"""按预测日期簇统计行业预期差前瞻成熟度，不读取未成熟收益。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


class OriginMaturityError(ValueError):
    """预测原点簇输入违反冻结治理边界。"""


def _date(value: Any, field: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise OriginMaturityError(f"{field}不能为空")
    return parsed.normalize()


def _primary_horizon(report: Mapping[str, Any], horizon_days: int) -> Mapping[str, Any]:
    matches = [
        item
        for item in report.get("horizons", [])
        if int(item.get("horizon_trading_days", -1)) == horizon_days
    ]
    if len(matches) != 1:
        raise OriginMaturityError(f"每个报告必须且只能有一个{horizon_days}日期限")
    return matches[0]


def build_origin_cluster_maturity_status(
    reports: Sequence[Mapping[str, Any]],
    *,
    primary_horizon_trading_days: int = 60,
    calibration_minimum_origin_clusters: int = 20,
    calibration_minimum_non_overlapping_blocks: int = 4,
    model_comparison_minimum_origin_clusters: int = 40,
    model_comparison_minimum_non_overlapping_blocks: int = 8,
) -> dict[str, Any]:
    """同一预测日期无论有多少行业行，都只计一个时间原点簇。"""

    if calibration_minimum_origin_clusters <= 0:
        raise OriginMaturityError("校准原点簇门槛必须为正")
    if model_comparison_minimum_origin_clusters < calibration_minimum_origin_clusters:
        raise OriginMaturityError("模型比较原点簇门槛不能低于校准门槛")
    if model_comparison_minimum_non_overlapping_blocks < (
        calibration_minimum_non_overlapping_blocks
    ):
        raise OriginMaturityError("模型比较非重叠块门槛不能低于校准门槛")

    reports_by_cluster: dict[str, dict[str, dict[str, Any]]] = {}
    for report in reports:
        if report.get("original_prediction_state") != "NO_VIEW":
            raise OriginMaturityError("原指数观点必须保持NO_VIEW")
        if report.get("may_upgrade_original_no_view") is not False:
            raise OriginMaturityError("禁止改写原NO_VIEW")
        prediction = _date(report.get("prediction_date"), "prediction_date")
        origin_cluster = prediction.date().isoformat()
        observation_date = _date(report.get("observation_date"), "observation_date")
        if observation_date < prediction:
            raise OriginMaturityError("observation_date不能早于prediction_date")
        primary = _primary_horizon(report, primary_horizon_trading_days)
        state = str(primary.get("state"))
        mature = state == "MATURE_SCORED"
        entry_date = _date(report.get("entry_date"), "entry_date")
        maturity_raw = primary.get("maturity_date")
        maturity_date = (
            _date(maturity_raw, "maturity_date") if maturity_raw is not None else None
        )
        cluster = {
            "origin_cluster": origin_cluster,
            "prediction_date": origin_cluster,
            "observation_date": observation_date.date().isoformat(),
            "entry_date": entry_date.date().isoformat(),
            "maturity_date": (
                maturity_date.date().isoformat() if maturity_date is not None else None
            ),
            "primary_state": state,
            "mature": mature,
            "industry_row_count": int(
                primary.get("price_evaluation", {}).get("industry_rows", [])
                and len(primary["price_evaluation"]["industry_rows"])
                or 0
            ),
            "counts_as_independent_time_samples": 1,
        }
        by_observation = reports_by_cluster.setdefault(origin_cluster, {})
        observation_key = observation_date.date().isoformat()
        existing = by_observation.get(observation_key)
        if existing is not None and existing != cluster:
            raise OriginMaturityError(
                f"同一origin_cluster与observation_date出现冲突："
                f"{origin_cluster}/{observation_key}"
            )
        by_observation[observation_key] = cluster

    state_rank = {
        "WAITING_FOR_ENTRY": 0,
        "ACCUMULATING_NO_PEEK": 1,
        "UNRESOLVED_CALENDAR": 1,
        "MATURE_DATA_INCOMPLETE": 2,
        "MATURE_SCORED": 2,
    }
    clusters: dict[str, dict[str, Any]] = {}
    for origin_cluster, by_observation in reports_by_cluster.items():
        history = [by_observation[key] for key in sorted(by_observation)]
        identity_fields = ("entry_date", "maturity_date")
        for field in identity_fields:
            values = {item[field] for item in history}
            if len(values) != 1:
                raise OriginMaturityError(
                    f"同一origin_cluster的{field}发生变化：{origin_cluster}"
                )
        previous_rank = -1
        for item in history:
            state = item["primary_state"]
            if state not in state_rank:
                raise OriginMaturityError(f"未知的前瞻成熟状态：{state}")
            rank = state_rank[state]
            if rank < previous_rank:
                raise OriginMaturityError(
                    f"同一origin_cluster成熟状态倒退：{origin_cluster}"
                )
            previous_rank = rank
        latest = dict(history[-1])
        latest["first_observation_date"] = history[0]["observation_date"]
        latest["latest_observation_date"] = history[-1]["observation_date"]
        latest["append_only_report_count"] = len(history)
        clusters[origin_cluster] = latest

    ordered = [clusters[key] for key in sorted(clusters)]
    mature_clusters = [item for item in ordered if item["mature"]]
    intervals = sorted(
        (
            pd.Timestamp(item["entry_date"]),
            pd.Timestamp(item["maturity_date"]),
        )
        for item in mature_clusters
        if item["maturity_date"] is not None
    )
    non_overlapping_blocks = 0
    last_end: pd.Timestamp | None = None
    for start, end in intervals:
        if end < start:
            raise OriginMaturityError("成熟区间终点早于入场日")
        if last_end is None or start > last_end:
            non_overlapping_blocks += 1
            last_end = end

    mature_count = len(mature_clusters)
    calibration_eligible = bool(
        mature_count >= calibration_minimum_origin_clusters
        and non_overlapping_blocks >= calibration_minimum_non_overlapping_blocks
    )
    comparison_eligible = bool(
        mature_count >= model_comparison_minimum_origin_clusters
        and non_overlapping_blocks >= model_comparison_minimum_non_overlapping_blocks
    )
    if comparison_eligible:
        status = "MODEL_COMPARISON_ELIGIBLE"
    elif calibration_eligible:
        status = "CALIBRATION_ELIGIBLE_NOT_MODEL_COMPARISON"
    else:
        status = "COLLECTING_FORWARD"

    return {
        "status": status,
        "origin_cluster_definition": "prediction_date",
        "independent_time_sample_unit": "origin_cluster",
        "industry_rows_count_as_time_samples": False,
        "origin_cluster_count": len(ordered),
        "append_only_report_count": sum(
            item["append_only_report_count"] for item in ordered
        ),
        "mature_origin_cluster_count": mature_count,
        "non_overlapping_60d_block_count": non_overlapping_blocks,
        "calibration": {
            "minimum_origin_clusters": calibration_minimum_origin_clusters,
            "minimum_non_overlapping_60d_blocks": calibration_minimum_non_overlapping_blocks,
            "eligible": calibration_eligible,
        },
        "model_comparison": {
            "minimum_origin_clusters": model_comparison_minimum_origin_clusters,
            "minimum_non_overlapping_60d_blocks": model_comparison_minimum_non_overlapping_blocks,
            "eligible": comparison_eligible,
        },
        "origin_clusters": ordered,
        "original_prediction_state": "NO_VIEW",
        "may_upgrade_original_no_view": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }


__all__ = [
    "OriginMaturityError",
    "build_origin_cluster_maturity_status",
]
