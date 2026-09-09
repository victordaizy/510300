"""行业预期差前瞻原点与结果输入的运行闸门。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

import pandas as pd


class ForwardOperationsError(ValueError):
    """运行输入违反独立原点或前瞻治理约束。"""


def _timestamp(value: Any, field: str) -> pd.Timestamp:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise ForwardOperationsError(f"{field}不能为空")
    if parsed.tzinfo is None and "T" in str(value):
        raise ForwardOperationsError(f"{field}必须包含时区")
    return parsed


def _date(value: Any, field: str) -> pd.Timestamp:
    return _timestamp(value, field).tz_localize(None).normalize()


def validate_operations_config(config: Mapping[str, Any]) -> None:
    """拒绝任何会制造伪原点、回填或交易授权的配置。"""

    origin = config.get("origin_collection", {})
    outcome = config.get("outcome_collection", {})
    safety = config.get("safety", {})
    if origin.get("cadence") != "MONTHLY_DISTINCT_PREDICTION_DATE":
        raise ForwardOperationsError("预测原点频率必须是月度独立prediction_date")
    if int(origin.get("minimum_sources_per_industry", 0)) < 2:
        raise ForwardOperationsError("每个行业至少需要两个可核验来源")
    if int(origin.get("minimum_new_official_releases", 0)) < 1:
        raise ForwardOperationsError("新原点必须要求新的官方发布")
    required_true = (
        "require_source_registry_cutoff_advance",
        "require_sector_panel_month_advance",
        "require_manual_evidence_package",
    )
    for field in required_true:
        if origin.get(field) is not True:
            raise ForwardOperationsError(f"必须启用原点闸门：{field}")
    required_false = (
        "automatic_origin_generation_enabled",
        "historical_backfill_enabled",
        "duplicate_prediction_date_enabled",
    )
    for field in required_false:
        if origin.get(field) is not False:
            raise ForwardOperationsError(f"必须关闭：{field}")
    if outcome.get("partial_horizon_return_output_enabled") is not False:
        raise ForwardOperationsError("未成熟期限不得输出部分收益")
    if safety.get("research_only") is not True:
        raise ForwardOperationsError("本运行层只能是研究模式")
    for field in (
        "may_upgrade_original_no_view",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_enabled",
    ):
        if safety.get(field) is not False:
            raise ForwardOperationsError(f"安全开关必须关闭：{field}")


def validate_origin_ledgers(
    ledgers: Sequence[Mapping[str, Any]],
    source_registry: Mapping[str, Any],
    *,
    minimum_sources_per_industry: int = 2,
) -> list[dict[str, Any]]:
    """验证每个prediction_date只出现一次且来源当时已经可用。"""

    if not ledgers:
        raise ForwardOperationsError("至少需要一个已冻结预测原点")
    sources = source_registry.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ForwardOperationsError("来源注册表为空")
    seen_dates: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for ledger in ledgers:
        prediction = _date(ledger.get("as_of_date"), "as_of_date")
        prediction_date = prediction.date().isoformat()
        if prediction_date in seen_dates:
            raise ForwardOperationsError(f"同一prediction_date重复：{prediction_date}")
        seen_dates.add(prediction_date)
        cutoff = _timestamp(ledger.get("information_cutoff"), "information_cutoff")
        if cutoff.tz_localize(None).normalize() != prediction:
            raise ForwardOperationsError("information_cutoff必须属于prediction_date")
        weight_date = _date(ledger.get("weight_snapshot_date"), "weight_snapshot_date")
        if weight_date > prediction:
            raise ForwardOperationsError("权重截面不得晚于预测日期")
        if ledger.get("historical_return_read") is not False:
            raise ForwardOperationsError("冻结原点不得读取历史收益进行调参")
        if ledger.get("model_fitting_performed") is not False:
            raise ForwardOperationsError("冻结原点不得执行模型拟合")
        judgments = ledger.get("judgments")
        if not isinstance(judgments, list) or not judgments:
            raise ForwardOperationsError("预测原点缺少行业判断")
        industries: set[str] = set()
        used_sources: set[str] = set()
        for judgment in judgments:
            industry = str(judgment.get("industry_l1", "")).strip()
            if not industry or industry in industries:
                raise ForwardOperationsError(f"行业为空或重复：{industry}")
            industries.add(industry)
            source_ids = judgment.get("source_ids")
            if not isinstance(source_ids, list) or len(set(source_ids)) < minimum_sources_per_industry:
                raise ForwardOperationsError(f"行业来源不足：{industry}")
            for source_id in source_ids:
                source = sources.get(source_id)
                if not isinstance(source, Mapping):
                    raise ForwardOperationsError(f"来源未登记：{source_id}")
                available_at = _timestamp(source.get("available_at"), f"{source_id}.available_at")
                if available_at > cutoff:
                    raise ForwardOperationsError(f"来源在预测截止后才可用：{source_id}")
                used_sources.add(str(source_id))
        normalized.append(
            {
                "prediction_date": prediction_date,
                "information_cutoff": cutoff.isoformat(),
                "weight_snapshot_date": weight_date.date().isoformat(),
                "industry_count": len(industries),
                "source_ids": sorted(used_sources),
            }
        )
    return sorted(normalized, key=lambda item: item["prediction_date"])


def _first_day_next_month(value: pd.Timestamp) -> pd.Timestamp:
    if value.month == 12:
        return pd.Timestamp(date(value.year + 1, 1, 1))
    return pd.Timestamp(date(value.year, value.month + 1, 1))


def assess_next_origin_gate(
    origins: Sequence[Mapping[str, Any]],
    source_registry: Mapping[str, Any],
    *,
    sector_panel_latest_date: Any,
    as_of: Any,
    minimum_new_official_releases: int,
) -> dict[str, Any]:
    """只有月份、官方证据和点时面板都推进后才允许人工建新原点。"""

    if not origins:
        raise ForwardOperationsError("无法在没有基准原点时评估下一原点")
    last = sorted(origins, key=lambda item: str(item["prediction_date"]))[-1]
    last_prediction = _date(last["prediction_date"], "last_prediction_date")
    last_cutoff = _timestamp(last["information_cutoff"], "last_information_cutoff")
    last_weight = _date(last["weight_snapshot_date"], "last_weight_snapshot_date")
    current = _timestamp(as_of, "as_of")
    current_date = current.tz_localize(None).normalize()
    earliest = _first_day_next_month(last_prediction)
    registry_cutoff = _timestamp(source_registry.get("information_cutoff"), "registry_information_cutoff")
    sector_latest = _date(sector_panel_latest_date, "sector_panel_latest_date")

    last_source_ids = set(last.get("source_ids", []))
    new_official: list[str] = []
    for source_id, source in source_registry.get("sources", {}).items():
        if source_id in last_source_ids or not isinstance(source, Mapping):
            continue
        if not str(source.get("kind", "")).startswith("OFFICIAL_"):
            continue
        available_at = _timestamp(source.get("available_at"), f"{source_id}.available_at")
        if last_cutoff < available_at <= current:
            new_official.append(str(source_id))

    blockers: list[str] = []
    if current_date < earliest:
        blockers.append("CURRENT_MONTH_ALREADY_HAS_ORIGIN")
    if registry_cutoff <= last_cutoff:
        blockers.append("SOURCE_REGISTRY_CUTOFF_NOT_ADVANCED")
    if sector_latest <= last_weight:
        blockers.append("SECTOR_PANEL_MONTH_NOT_ADVANCED")
    if len(new_official) < minimum_new_official_releases:
        blockers.append("NEW_OFFICIAL_RELEASES_BELOW_MINIMUM")
    ready = not blockers
    return {
        "status": "READY_FOR_MANUAL_EVIDENCE_PACKAGE" if ready else "WAITING_FOR_NEXT_ORIGIN_EVIDENCE",
        "ready": ready,
        "last_prediction_date": last_prediction.date().isoformat(),
        "earliest_candidate_date": earliest.date().isoformat(),
        "candidate_date_is_authorization": False,
        "last_information_cutoff": last_cutoff.isoformat(),
        "source_registry_information_cutoff": registry_cutoff.isoformat(),
        "last_weight_snapshot_date": last_weight.date().isoformat(),
        "sector_panel_latest_date": sector_latest.date().isoformat(),
        "new_official_release_ids": sorted(new_official),
        "minimum_new_official_releases": minimum_new_official_releases,
        "blockers": blockers,
        "manual_evidence_package_required": True,
        "automatic_origin_generation_enabled": False,
        "historical_backfill_enabled": False,
    }


def assess_outcome_input_gate(
    *,
    entry_date: Any,
    constituent_latest_date: Any,
    etf_latest_date: Any,
    expected_latest_trading_date: Any,
) -> dict[str, Any]:
    """结果数据必须至少覆盖入场日，且达到最近已收盘交易日。"""

    entry = _date(entry_date, "entry_date")
    constituent = _date(constituent_latest_date, "constituent_latest_date")
    etf = _date(etf_latest_date, "etf_latest_date")
    expected = _date(expected_latest_trading_date, "expected_latest_trading_date")
    common = min(constituent, etf)
    blockers: list[str] = []
    if common < entry:
        blockers.append("OUTCOME_INPUT_BEFORE_ENTRY_DATE")
        status = "BLOCKED_OUTCOME_INPUT_DATE_MISMATCH"
    elif common < expected:
        blockers.append("OUTCOME_INPUT_BEHIND_LATEST_CLOSED_TRADING_DAY")
        status = "COLLECTING_OUTCOME_INPUT_STALE"
    else:
        status = "READY_TO_REFRESH_APPEND_ONLY_EVALUATION"
    return {
        "status": status,
        "ready": not blockers,
        "entry_date": entry.date().isoformat(),
        "constituent_latest_date": constituent.date().isoformat(),
        "etf_latest_date": etf.date().isoformat(),
        "common_latest_date": common.date().isoformat(),
        "expected_latest_closed_trading_date": expected.date().isoformat(),
        "blockers": blockers,
        "partial_horizon_return_output_enabled": False,
    }


__all__ = [
    "ForwardOperationsError",
    "assess_next_origin_gate",
    "assess_outcome_input_gate",
    "validate_operations_config",
    "validate_origin_ledgers",
]
