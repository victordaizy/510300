"""行业预期差—ETF 估值研究模型 V1。

该模块只负责点时输入校验、行业聚合和研究状态生成。它不读取未来收益，
不生成仓位，不连接券商，也不产生订单。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


FUNDAMENTAL_SCORES = {
    "STRONG_IMPROVEMENT": 2,
    "IMPROVEMENT": 1,
    "MIXED": 0,
    "DETERIORATION": -1,
    "STRONG_DETERIORATION": -2,
}

EXPECTATION_GAP_SCORES: dict[str, int | None] = {
    "POSITIVE": 1,
    "BALANCED": 0,
    "NEGATIVE": -1,
    "UNOBSERVED": None,
}

CONFIDENCE_LEVELS = {"LOW", "MEDIUM", "HIGH"}
LIQUIDITY_STATES = {"SUPPORTIVE", "NEUTRAL", "ADVERSE", "NO_VIEW"}
HOLDINGS_STATES = {"PRESENT", "ABSENT", "UNOBSERVED"}
ACTIVITY_PROXY_STATES = {"SUPPORTIVE", "NEUTRAL", "ADVERSE", "UNOBSERVED"}
PROHIBITED_OUTCOME_KEY_PARTS = (
    "future_return",
    "forward_return",
    "realized_return",
    "outcome",
    "target_label",
)


class ResearchInputError(ValueError):
    """输入违反冻结研究协议。"""


@dataclass(frozen=True)
class ModelRules:
    """从冻结配置读取的不可变规则。"""

    required_latest_weight_coverage: float
    maximum_sector_weight_age_calendar_days: int
    minimum_expectation_gap_weight_coverage: float
    minimum_abs_weighted_gap_for_direction: float
    minimum_sources_per_industry: int
    primary_horizon_trading_days: int
    tail_horizon_trading_days: int
    minimum_calibration_observations: int
    minimum_model_comparison_observations: int
    allow_historical_return_read: bool
    allow_model_fitting: bool
    allow_failed_family_reuse: bool
    allow_proxy_identity_inference: bool
    require_market_gate_for_directional_view: bool
    require_all_latest_industries: bool

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ModelRules":
        required = {field_name for field_name in cls.__dataclass_fields__}
        missing = required - set(values)
        if missing:
            raise ResearchInputError(f"冻结规则缺少字段：{sorted(missing)}")
        return cls(**{name: values[name] for name in required})


def _timestamp(value: Any, field_name: str) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:  # pragma: no cover - pandas 提供具体异常
        raise ResearchInputError(f"{field_name} 不是合法时间：{value}") from exc
    if pd.isna(timestamp):
        raise ResearchInputError(f"{field_name} 不能为空")
    return timestamp


def _date(value: Any, field_name: str) -> date:
    return _timestamp(value, field_name).date()


def _check_no_outcomes(value: Any, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(part in normalized for part in PROHIBITED_OUTCOME_KEY_PARTS):
                raise ResearchInputError(f"前瞻账本禁止包含结果字段：{path}.{key}")
            _check_no_outcomes(child, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            _check_no_outcomes(child, f"{path}[{index}]")


def validate_contract(contract: Mapping[str, Any]) -> tuple[ModelRules, dict[str, bool]]:
    """校验配置中的研究边界和安全开关。"""

    if contract.get("status") != "FROZEN_BEFORE_FORWARD_OUTCOME":
        raise ResearchInputError("配置状态必须为 FROZEN_BEFORE_FORWARD_OUTCOME")
    if contract.get("asset") != "510300" or contract.get("benchmark") != "000300":
        raise ResearchInputError("V1 只允许 510300/000300")

    rules = ModelRules.from_mapping(contract.get("rules", {}))
    forbidden_true = {
        "allow_historical_return_read": rules.allow_historical_return_read,
        "allow_model_fitting": rules.allow_model_fitting,
        "allow_failed_family_reuse": rules.allow_failed_family_reuse,
        "allow_proxy_identity_inference": rules.allow_proxy_identity_inference,
    }
    enabled = [name for name, state in forbidden_true.items() if state]
    if enabled:
        raise ResearchInputError(f"冻结前瞻版本禁止启用：{enabled}")

    safety = dict(contract.get("safety", {}))
    required_safety = {
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_enabled",
        "current_holdings_read_enabled",
    }
    if required_safety - set(safety):
        raise ResearchInputError(f"安全开关缺失：{sorted(required_safety - set(safety))}")
    unsafe = [name for name in required_safety if safety[name] is not False]
    if unsafe:
        raise ResearchInputError(f"V1 安全开关必须全部为 false：{unsafe}")
    return rules, safety


def validate_source_registry(
    registry: Mapping[str, Any],
    workspace_root: Path,
    information_cutoff: Any,
) -> dict[str, dict[str, Any]]:
    """验证来源在截止时点前可得，并检查所有本地文件存在。"""

    cutoff = _timestamp(information_cutoff, "information_cutoff")
    sources = registry.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ResearchInputError("来源注册表为空")

    normalized: dict[str, dict[str, Any]] = {}
    for source_id, raw_source in sources.items():
        source = dict(raw_source)
        available_at = _timestamp(source.get("available_at"), f"{source_id}.available_at")
        if available_at.tzinfo is None and cutoff.tzinfo is not None:
            available_at = available_at.tz_localize(cutoff.tzinfo)
        if cutoff.tzinfo is None and available_at.tzinfo is not None:
            cutoff_for_compare = cutoff.tz_localize(available_at.tzinfo)
        else:
            cutoff_for_compare = cutoff
        if available_at > cutoff_for_compare:
            raise ResearchInputError(f"来源晚于信息截止时间：{source_id}")
        if "path" in source:
            local_path = workspace_root / str(source["path"])
            if not local_path.is_file():
                raise ResearchInputError(f"本地来源不存在：{source_id} -> {local_path}")
        elif "url" not in source:
            raise ResearchInputError(f"来源既无 path 也无 url：{source_id}")
        normalized[str(source_id)] = source
    return normalized


def select_sector_snapshot(
    sector_panel: pd.DataFrame,
    as_of_date: Any,
    rules: ModelRules,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """选择不晚于预测日的最新行业权重截面。"""

    required_columns = {
        "date",
        "industry_l1",
        "sector_weight",
        "component_count",
        "core_feature_coverage_ratio",
        "weighted_earnings_yield",
        "weighted_book_yield",
        "weighted_ttm_roe",
        "weighted_ttm_profit_growth_yoy",
        "weighted_ttm_revenue_growth_yoy",
    }
    missing = required_columns - set(sector_panel.columns)
    if missing:
        raise ResearchInputError(f"行业点时面板缺少字段：{sorted(missing)}")

    frame = sector_panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ResearchInputError("行业点时面板包含非法日期")
    as_of = pd.Timestamp(_date(as_of_date, "as_of_date"))
    eligible = frame.loc[frame["date"] <= as_of]
    if eligible.empty:
        raise ResearchInputError("预测日前没有可用行业权重截面")
    snapshot_date = eligible["date"].max()
    snapshot = eligible.loc[eligible["date"] == snapshot_date].copy()
    if snapshot["industry_l1"].duplicated().any():
        duplicates = snapshot.loc[snapshot["industry_l1"].duplicated(), "industry_l1"].tolist()
        raise ResearchInputError(f"行业截面存在重复行业：{duplicates}")
    age_days = (as_of - snapshot_date).days
    if age_days > rules.maximum_sector_weight_age_calendar_days:
        raise ResearchInputError(
            f"行业权重已过期：{snapshot_date.date()}，距预测日 {age_days} 天"
        )
    weight_coverage = float(snapshot["sector_weight"].sum())
    if weight_coverage < rules.required_latest_weight_coverage:
        raise ResearchInputError(
            f"最新行业权重覆盖不足：{weight_coverage:.6f} < "
            f"{rules.required_latest_weight_coverage:.6f}"
        )
    if (snapshot["sector_weight"] < 0).any():
        raise ResearchInputError("行业权重不能为负")
    metadata = {
        "snapshot_date": snapshot_date.date().isoformat(),
        "age_calendar_days": age_days,
        "industry_count": int(len(snapshot)),
        "weight_coverage": weight_coverage,
    }
    return snapshot, metadata


def _validate_source_ids(
    source_ids: Any,
    sources: Mapping[str, Any],
    field_name: str,
    minimum_count: int = 1,
) -> list[str]:
    if not isinstance(source_ids, list):
        raise ResearchInputError(f"{field_name} 必须是来源ID列表")
    unique_ids = list(dict.fromkeys(str(item) for item in source_ids))
    if len(unique_ids) < minimum_count:
        raise ResearchInputError(f"{field_name} 至少需要 {minimum_count} 个不同来源")
    unknown = [source_id for source_id in unique_ids if source_id not in sources]
    if unknown:
        raise ResearchInputError(f"{field_name} 包含未注册来源：{unknown}")
    return unique_ids


def build_industry_map(
    ledger: Mapping[str, Any],
    sector_snapshot: pd.DataFrame,
    sector_metadata: Mapping[str, Any],
    sources: Mapping[str, Any],
    rules: ModelRules,
    expected_as_of_date: str,
    expected_cutoff: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """校验冻结行业判断并按点时权重聚合。"""

    _check_no_outcomes(ledger)
    if ledger.get("as_of_date") != expected_as_of_date:
        raise ResearchInputError("行业账本 as_of_date 与配置不一致")
    if ledger.get("information_cutoff") != expected_cutoff:
        raise ResearchInputError("行业账本 information_cutoff 与配置不一致")
    if ledger.get("historical_return_read") is not False:
        raise ResearchInputError("首期行业账本禁止读取历史收益")
    if ledger.get("model_fitting_performed") is not False:
        raise ResearchInputError("首期行业账本禁止模型拟合")
    if ledger.get("weight_snapshot_date") != sector_metadata["snapshot_date"]:
        raise ResearchInputError("行业账本的权重日期与实际点时截面不一致")

    judgments = ledger.get("judgments")
    if not isinstance(judgments, list) or not judgments:
        raise ResearchInputError("行业判断列表为空")
    names = [item.get("industry_l1") for item in judgments]
    if len(names) != len(set(names)):
        raise ResearchInputError("行业判断存在重复行业")
    snapshot_names = set(sector_snapshot["industry_l1"].astype(str))
    judgment_names = set(str(name) for name in names)
    if rules.require_all_latest_industries and judgment_names != snapshot_names:
        raise ResearchInputError(
            "行业账本必须精确覆盖最新截面；"
            f"缺少={sorted(snapshot_names - judgment_names)}，"
            f"多出={sorted(judgment_names - snapshot_names)}"
        )

    rows: list[dict[str, Any]] = []
    for item in judgments:
        industry = str(item.get("industry_l1"))
        f60 = item.get("fundamental_60d")
        f120 = item.get("fundamental_120d")
        gap = item.get("expectation_gap")
        confidence = item.get("confidence")
        if f60 not in FUNDAMENTAL_SCORES or f120 not in FUNDAMENTAL_SCORES:
            raise ResearchInputError(f"{industry} 的基本面方向非法")
        if gap not in EXPECTATION_GAP_SCORES:
            raise ResearchInputError(f"{industry} 的预期差状态非法：{gap}")
        if confidence not in CONFIDENCE_LEVELS:
            raise ResearchInputError(f"{industry} 的置信度非法：{confidence}")
        for text_field in ("valuation_method", "thesis", "invalidation"):
            if not str(item.get(text_field, "")).strip():
                raise ResearchInputError(f"{industry} 缺少 {text_field}")
        source_ids = _validate_source_ids(
            item.get("source_ids"),
            sources,
            f"{industry}.source_ids",
            rules.minimum_sources_per_industry,
        )
        rows.append(
            {
                "industry_l1": industry,
                "fundamental_60d": f60,
                "fundamental_120d": f120,
                "fundamental_60d_score": FUNDAMENTAL_SCORES[f60],
                "fundamental_120d_score": FUNDAMENTAL_SCORES[f120],
                "expectation_gap": gap,
                "expectation_gap_score": EXPECTATION_GAP_SCORES[gap],
                "confidence": confidence,
                "valuation_method": str(item["valuation_method"]),
                "thesis": str(item["thesis"]),
                "invalidation": str(item["invalidation"]),
                "source_ids": source_ids,
            }
        )

    judgment_frame = pd.DataFrame(rows)
    snapshot_columns = [
        "industry_l1",
        "sector_weight",
        "component_count",
        "core_feature_coverage_ratio",
        "weighted_earnings_yield",
        "weighted_book_yield",
        "weighted_ttm_roe",
        "weighted_ttm_profit_growth_yoy",
        "weighted_ttm_revenue_growth_yoy",
    ]
    merged = judgment_frame.merge(
        sector_snapshot[snapshot_columns],
        on="industry_l1",
        how="left",
        validate="one_to_one",
    )
    if merged["sector_weight"].isna().any():
        missing_weights = merged.loc[merged["sector_weight"].isna(), "industry_l1"].tolist()
        raise ResearchInputError(f"行业缺少点时权重：{missing_weights}")

    merged["weighted_fundamental_60d"] = (
        merged["sector_weight"] * merged["fundamental_60d_score"]
    )
    merged["weighted_fundamental_120d"] = (
        merged["sector_weight"] * merged["fundamental_120d_score"]
    )
    merged["weighted_gap_contribution"] = (
        merged["sector_weight"] * merged["expectation_gap_score"]
    )
    merged = merged.sort_values("sector_weight", ascending=False).reset_index(drop=True)

    observed = merged["expectation_gap_score"].notna()
    observed_weight = float(merged.loc[observed, "sector_weight"].sum())
    total_weight = float(merged["sector_weight"].sum())
    positive_gap = float(
        merged.loc[merged["weighted_gap_contribution"] > 0, "weighted_gap_contribution"].sum()
    )
    negative_gap = float(
        merged.loc[merged["weighted_gap_contribution"] < 0, "weighted_gap_contribution"].sum()
    )
    net_gap = positive_gap + negative_gap
    if observed_weight < rules.minimum_expectation_gap_weight_coverage:
        aggregation_state = "NO_VIEW"
        aggregation_reason = "EXPECTATION_GAP_WEIGHT_COVERAGE_BELOW_FROZEN_THRESHOLD"
    elif abs(net_gap) < rules.minimum_abs_weighted_gap_for_direction:
        aggregation_state = "BALANCED_NO_EDGE"
        aggregation_reason = "NET_EXPECTATION_GAP_BELOW_FROZEN_DIRECTION_THRESHOLD"
    elif net_gap > 0:
        aggregation_state = "POSITIVE_INDEX_GAP"
        aggregation_reason = "POSITIVE_NET_EXPECTATION_GAP_ABOVE_THRESHOLD"
    else:
        aggregation_state = "NEGATIVE_INDEX_GAP"
        aggregation_reason = "NEGATIVE_NET_EXPECTATION_GAP_ABOVE_THRESHOLD"

    summary = {
        "state": aggregation_state,
        "reason": aggregation_reason,
        "sector_snapshot": dict(sector_metadata),
        "expectation_gap_observed_industries": int(observed.sum()),
        "expectation_gap_unobserved_industries": int((~observed).sum()),
        "expectation_gap_observed_index_weight": observed_weight,
        "expectation_gap_observed_share_of_mapped_weight": (
            observed_weight / total_weight if total_weight else 0.0
        ),
        "positive_gap_weighted_contribution": positive_gap,
        "negative_gap_weighted_contribution": negative_gap,
        "net_gap_weighted_contribution": net_gap,
        "improving_60d_index_weight": float(
            merged.loc[merged["fundamental_60d_score"] > 0, "sector_weight"].sum()
        ),
        "deteriorating_60d_index_weight": float(
            merged.loc[merged["fundamental_60d_score"] < 0, "sector_weight"].sum()
        ),
        "weighted_fundamental_60d_score": float(merged["weighted_fundamental_60d"].sum()),
        "weighted_fundamental_120d_score": float(merged["weighted_fundamental_120d"].sum()),
        "direction_threshold": rules.minimum_abs_weighted_gap_for_direction,
        "coverage_threshold": rules.minimum_expectation_gap_weight_coverage,
    }
    return merged, summary


def evaluate_market_state(
    ledger: Mapping[str, Any],
    sources: Mapping[str, Any],
    expected_as_of_date: str,
    expected_cutoff: str,
) -> dict[str, Any]:
    """评价市场流动性与国家队披露尾部，禁止身份推断。"""

    _check_no_outcomes(ledger)
    if ledger.get("as_of_date") != expected_as_of_date:
        raise ResearchInputError("市场账本 as_of_date 与配置不一致")
    if ledger.get("information_cutoff") != expected_cutoff:
        raise ResearchInputError("市场账本 information_cutoff 与配置不一致")

    dimensions: dict[str, dict[str, Any]] = {}
    for name in ("funding_liquidity", "equity_liquidity", "risk_bearing"):
        raw = ledger.get(name)
        if not isinstance(raw, Mapping):
            raise ResearchInputError(f"市场账本缺少 {name}")
        state = raw.get("state")
        if state not in LIQUIDITY_STATES:
            raise ResearchInputError(f"{name}.state 非法：{state}")
        source_ids = _validate_source_ids(raw.get("source_ids"), sources, f"{name}.source_ids")
        if not str(raw.get("reason", "")).strip():
            raise ResearchInputError(f"{name} 缺少 reason")
        dimensions[name] = {
            "state": state,
            "data_as_of": str(raw.get("data_as_of")),
            "failure_category": raw.get("failure_category"),
            "reason": str(raw["reason"]),
            "source_ids": source_ids,
        }

    states = [item["state"] for item in dimensions.values()]
    if "NO_VIEW" in states:
        gate_state = "NO_VIEW"
        gate_reason = "ONE_OR_MORE_LIQUIDITY_DIMENSIONS_UNOBSERVED"
    elif "ADVERSE" in states:
        gate_state = "ADVERSE"
        gate_reason = "ONE_OR_MORE_LIQUIDITY_DIMENSIONS_ADVERSE"
    else:
        gate_state = "PASS"
        gate_reason = "ALL_LIQUIDITY_DIMENSIONS_OBSERVED_AND_NON_ADVERSE"

    raw_national = ledger.get("national_team")
    if not isinstance(raw_national, Mapping):
        raise ResearchInputError("市场账本缺少 national_team")
    holdings_state = raw_national.get("disclosed_holdings_state")
    proxy_state = raw_national.get("activity_proxy_state")
    if holdings_state not in HOLDINGS_STATES:
        raise ResearchInputError(f"国家队披露持仓状态非法：{holdings_state}")
    if proxy_state not in ACTIVITY_PROXY_STATES:
        raise ResearchInputError(f"国家队活动代理状态非法：{proxy_state}")
    national_sources = raw_national.get("source_ids")
    if not isinstance(national_sources, list):
        raise ResearchInputError("national_team.source_ids 必须是列表")
    if holdings_state == "PRESENT":
        _validate_source_ids(national_sources, sources, "national_team.source_ids")
        if not raw_national.get("holdings_report_period") or not raw_national.get(
            "holdings_publication_date"
        ):
            raise ResearchInputError("披露持仓为 PRESENT 时必须有报告期和公开日")
    elif holdings_state == "UNOBSERVED" and national_sources:
        _validate_source_ids(national_sources, sources, "national_team.source_ids")
    if not str(raw_national.get("reason", "")).strip():
        raise ResearchInputError("national_team 缺少 reason")

    national_team = {
        "disclosed_holdings_state": holdings_state,
        "holdings_report_period": raw_national.get("holdings_report_period"),
        "holdings_publication_date": raw_national.get("holdings_publication_date"),
        "activity_proxy_state": proxy_state,
        "failure_category": raw_national.get("failure_category"),
        "reason": str(raw_national["reason"]),
        "source_ids": list(national_sources),
        "identity_inference_performed": False,
        "can_change_direction": False,
    }
    return {
        "gate_state": gate_state,
        "gate_reason": gate_reason,
        "dimensions": dimensions,
        "national_team": national_team,
    }


def evaluate_etf_snapshot(
    snapshot: Mapping[str, Any],
    sources: Mapping[str, Any],
    expected_as_of_date: str,
    expected_cutoff: str,
    rules: ModelRules,
) -> dict[str, Any]:
    """区分当日 ETF 观察有效与预测样本成熟。"""

    _check_no_outcomes(snapshot)
    if snapshot.get("asset") != "510300":
        raise ResearchInputError("ETF 快照只能是 510300")
    if snapshot.get("as_of_date") != expected_as_of_date:
        raise ResearchInputError("ETF 快照 as_of_date 与配置不一致")
    if snapshot.get("information_cutoff") != expected_cutoff:
        raise ResearchInputError("ETF 快照 information_cutoff 与配置不一致")
    source_ids = _validate_source_ids(snapshot.get("source_ids"), sources, "ETF.source_ids", 2)

    pcf = snapshot.get("pcf", {})
    iopv = snapshot.get("iopv", {})
    readiness = snapshot.get("forward_readiness", {})
    iopv_date = _date(iopv.get("timestamp"), "iopv.timestamp").isoformat()
    observation_checks = {
        "collector_pass": snapshot.get("collector_status") == "PASS",
        "pcf_current": pcf.get("trading_date") == expected_as_of_date,
        "iopv_current": iopv_date == expected_as_of_date,
        "creation_status_known": isinstance(pcf.get("creation_allowed"), bool)
        and isinstance(pcf.get("redemption_allowed"), bool),
        "not_used_as_directional_signal": snapshot.get("use_as_directional_signal") is False,
    }
    observation_state = "PASS" if all(observation_checks.values()) else "INVALID"

    full_days = int(readiness.get("full_coverage_days", 0))
    predictive_ready = (
        readiness.get("eligible_for_research_evaluation") is True
        and full_days >= rules.minimum_calibration_observations
    )
    predictive_state = "READY" if predictive_ready else "INSUFFICIENT_FORWARD_HISTORY"
    return {
        "observation_state": observation_state,
        "observation_checks": observation_checks,
        "predictive_state": predictive_state,
        "collector_role": snapshot.get("collector_role"),
        "pcf": dict(pcf),
        "iopv": dict(iopv),
        "forward_readiness": dict(readiness),
        "small_account_primary_market_feasible": bool(
            snapshot.get("small_account_primary_market_feasible")
        ),
        "execution_mode": "OBSERVE_ONLY",
        "source_ids": source_ids,
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, Mapping):
        return {key: _json_ready(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_ready(child) for child in value]
    return value


def build_research_result(
    contract: Mapping[str, Any],
    industry_frame: pd.DataFrame,
    industry_summary: Mapping[str, Any],
    market_summary: Mapping[str, Any],
    etf_summary: Mapping[str, Any],
    safety: Mapping[str, bool],
) -> dict[str, Any]:
    """生成首期研究结果；多层失败原因同时保留。"""

    failure_categories: list[str] = []
    if industry_summary["state"] == "NO_VIEW":
        failure_categories.append(str(industry_summary["reason"]))
    if industry_summary["state"] == "BALANCED_NO_EDGE":
        failure_categories.append(str(industry_summary["reason"]))
    if market_summary["gate_state"] != "PASS":
        failure_categories.append(str(market_summary["gate_reason"]))
    if market_summary["national_team"]["disclosed_holdings_state"] == "UNOBSERVED":
        failure_categories.append("NATIONAL_TEAM_DISCLOSED_HOLDINGS_UNOBSERVED")
    if etf_summary["observation_state"] != "PASS":
        failure_categories.append("ETF_CURRENT_OBSERVATION_INVALID")
    if etf_summary["predictive_state"] != "READY":
        failure_categories.append("ETF_FORWARD_HISTORY_INSUFFICIENT")

    directional_industry_state = industry_summary["state"] in {
        "POSITIVE_INDEX_GAP",
        "NEGATIVE_INDEX_GAP",
    }
    directional_allowed = (
        directional_industry_state
        and market_summary["gate_state"] == "PASS"
        and etf_summary["observation_state"] == "PASS"
        and etf_summary["predictive_state"] == "READY"
    )
    final_state = "DIRECTIONAL_RESEARCH_VIEW" if directional_allowed else "NO_VIEW"

    rows = [_json_ready(record) for record in industry_frame.to_dict(orient="records")]
    return {
        "model_version": contract["version"],
        "model_status": contract["status"],
        "research_mode": contract["research_mode"],
        "signal_mode": contract["signal_mode"],
        "asset": contract["asset"],
        "benchmark": contract["benchmark"],
        "as_of_date": contract["as_of_date"],
        "information_cutoff": contract["information_cutoff"],
        "first_forward_observation_date": contract["first_forward_observation_date"],
        "current_simplification_stage": "M0_FULL_RECORD",
        "final_state": final_state,
        "failure_categories": list(dict.fromkeys(failure_categories)),
        "industry_aggregation": _json_ready(dict(industry_summary)),
        "market_liquidity_and_tail": _json_ready(dict(market_summary)),
        "etf_execution_wrapper": _json_ready(dict(etf_summary)),
        "tail_risk_watch": "NOT_AUTHORIZED_BEFORE_FORWARD_EVIDENCE",
        "industry_rows": rows,
        "safety": dict(safety),
        "interpretation": (
            "首期输出是冻结待验证的行业预期差底图；NO_VIEW 不等于看空，"
            "不构成仓位、订单或实盘建议。"
        ),
    }


def render_markdown(result: Mapping[str, Any]) -> str:
    """渲染便于人工审计的中文报告。"""

    industry = result["industry_aggregation"]
    market = result["market_liquidity_and_tail"]
    etf = result["etf_execution_wrapper"]
    lines = [
        "# 行业预期差—ETF 估值研究 V1 首期输出",
        "",
        f"- 截止日：`{result['as_of_date']}`",
        f"- 信息截止：`{result['information_cutoff']}`",
        f"- 最终状态：`{result['final_state']}`",
        f"- 当前阶段：`{result['current_simplification_stage']}`",
        "- 边界：`RESEARCH_ONLY / SHADOW_ONLY / NO_POSITION_CHANGE`",
        "",
        "## 结论",
        "",
        (
            f"行业预期差覆盖指数权重 `{industry['expectation_gap_observed_index_weight']:.3%}`，"
            f"净加权贡献 `{industry['net_gap_weighted_contribution']:.3%}`，"
            f"未越过冻结方向阈值 `{industry['direction_threshold']:.1%}`；"
            f"行业聚合状态为 `{industry['state']}`。"
        ),
        "",
        (
            f"市场流动性门控为 `{market['gate_state']}`；国家队披露持仓为 "
            f"`{market['national_team']['disclosed_holdings_state']}`。"
        ),
        "",
        (
            f"ETF 当日观察为 `{etf['observation_state']}`，但前瞻成熟度为 "
            f"`{etf['predictive_state']}`，因此只允许 `OBSERVE_ONLY`。"
        ),
        "",
        "失败/停止原因：" + "；".join(f"`{item}`" for item in result["failure_categories"]),
        "",
        "## 行业底图",
        "",
        "| 行业 | 权重 | 60日经营 | 120日经营 | 预期差 | 加权贡献 | 置信度 |",
        "|---|---:|---|---|---|---:|---|",
    ]
    for row in result["industry_rows"]:
        contribution = row["weighted_gap_contribution"]
        contribution_text = "未观察" if contribution is None else f"{contribution:.3%}"
        lines.append(
            "| {industry} | {weight:.3%} | {f60} | {f120} | {gap} | {contribution} | {confidence} |".format(
                industry=row["industry_l1"],
                weight=row["sector_weight"],
                f60=row["fundamental_60d"],
                f120=row["fundamental_120d"],
                gap=row["expectation_gap"],
                contribution=contribution_text,
                confidence=row["confidence"],
            )
        )

    lines.extend(["", "## 可审计的行业论点与失效条件", ""])
    for row in result["industry_rows"]:
        lines.extend(
            [
                f"### {row['industry_l1']}",
                "",
                f"- 论点：{row['thesis']}",
                f"- 失效条件：{row['invalidation']}",
                f"- 估值方法：{row['valuation_method']}",
                f"- 来源：{', '.join(row['source_ids'])}",
                "",
            ]
        )

    lines.extend(
        [
            "## 市场与尾部边界",
            "",
            *[
                f"- `{name}`：`{item['state']}`；{item['reason']}"
                for name, item in market["dimensions"].items()
            ],
            (
                "- `national_team`："
                f"`{market['national_team']['disclosed_holdings_state']}`；"
                f"{market['national_team']['reason']}"
            ),
            "",
            "## 安全开关",
            "",
            *[f"- `{name}`：`{str(value).lower()}`" for name, value in result["safety"].items()],
            "",
            result["interpretation"],
            "",
        ]
    )
    return "\n".join(lines)
