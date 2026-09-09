"""市场状态感知策略资格与风险配置器 V1。

本模块只产生研究层资格和风险预算。它不生成证券数量、目标仓位、交易信号、
订单、账户连接或实盘动作。
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "regime_aware_strategy_allocator_v1.yaml"

EXPECTED_STRATEGY_IDS = (
    "BTC_SPOT_CAPACITY_AWARE_LONG_HORIZON_TREND_V3",
    "DIGITAL_ASSET_DUAL_LONG_HORIZON_TREND_V4",
    "DIGITAL_ASSET_RELATIVE_STRENGTH_ROTATION_V5",
    "DIGITAL_ASSET_MULTI_HORIZON_TREND_VOTE_V6",
    "DIGITAL_ASSET_CROSS_SECTIONAL_PERPETUAL_BASIS_V15",
)

EXPECTED_ENVIRONMENTS = (
    "DEFENSIVE_HIGH_RISK",
    "CARRY_DISPERSION",
    "TREND_PERSISTENT_UP",
    "TREND_PERSISTENT_DOWN",
    "TREND_TRANSITION",
    "MEAN_REVERTING_LIQUID",
    "UNKNOWN",
)

ENVIRONMENT_STATUSES = {
    "PASS",
    "NO_VIEW",
    "RUNNING",
    "PARTIAL_SUCCESS",
    "FAILED",
    "PROGRAM_FAILED",
    "SKIPPED",
}
META_EVIDENCE_STATUSES = {
    "NOT_STARTED",
    "FORWARD_COLLECTING",
    "INSUFFICIENT_EVIDENCE",
    "CONDITIONAL_FORWARD_PASSED",
    "CONDITIONALLY_PASSING_CONFIRMATION_PENDING",
    "REJECTED_FROZEN",
    "NO_VIEW",
}

ENVIRONMENT_REQUIRED_FIELDS = {
    "schema_version",
    "evidence_id",
    "environment_scope",
    "observation_at",
    "available_at",
    "data_status",
    "source_path",
    "source_sha256",
    "features",
    "reason",
}

STRATEGY_REQUIRED_FIELDS = {
    "schema_version",
    "evidence_id",
    "strategy_id",
    "environment_id",
    "recorded_at",
    "available_at",
    "source_path",
    "source_sha256",
    "evidence_type",
    "meta_evidence_status",
    "evaluation_window_start",
    "evaluation_window_end",
    "latest_forward_observation_at",
    "latest_forward_data_status",
    "forward_observation_days",
    "recent_health_observation_days",
    "independent_environment_episodes",
    "consecutive_entry_confirmations",
    "historical_contamination",
    "base_net_conditional_edge_lcb_after_all_costs",
    "stress_net_conditional_edge_lcb_after_all_costs",
    "stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs",
    "recent_stress_net_edge_vs_cash_after_all_costs",
    "recent_stress_net_incremental_vs_static_equal_risk_after_all_costs",
    "deflated_sharpe_probability",
    "probability_backtest_overfit",
    "maximum_environment_profit_concentration",
    "forecast_annualized_volatility",
    "benchmark_set_complete",
    "benchmarks",
    "cost_model_complete",
    "included_cost_components",
    "paper_position_generation",
    "shadow_signal_generation",
    "order_generation",
    "live_trading_authorized",
}

EVALUATOR_REPORT_REQUIRED_FIELDS = {
    "schema_version",
    "evaluator_id",
    "candidate_id",
    "allocator_tracked_content_sha256",
    "allocator_manifest_path",
    "allocator_manifest_sha256",
    "generated_at",
    "decision_at",
    "ledger",
    "environment_evidence",
    "evidence_core",
    "evaluation",
    "performance_used_for_recent_winner_ranking",
    "historical_backfill_used",
    "paper_position_generation",
    "shadow_signal_generation",
    "order_generation",
    "live_trading_authorized",
}


class ContractError(ValueError):
    """冻结合同不满足 V1 不变量。"""


class EvidenceError(ValueError):
    """点时证据不满足追加、时间或来源约束。"""


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file_prefix(path: Path, byte_count: int) -> str:
    """计算追加式文件前缀哈希，使旧报告在文件追加后仍可复核。"""

    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise EvidenceError("追加式证据前缀字节数必须是非负整数")
    if path.stat().st_size < byte_count:
        raise EvidenceError(f"追加式证据文件被截断：{path}")
    remaining = byte_count
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise EvidenceError(f"追加式证据文件读取长度不足：{path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def load_contract(path: Path = CONFIG) -> dict[str, Any]:
    """读取并验证冻结合同。"""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractError("V1 配置必须是 YAML 对象")
    validate_contract(payload)
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(f"{label} 不得偏离冻结值：expected={expected!r}, actual={actual!r}")


def _require_probability(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{label} 必须是数值")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ContractError(f"{label} 必须位于 0 至 1")
    return number


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("来源路径必须是非空项目相对路径")
    if "\\" in value:
        raise ValueError(f"来源路径必须使用正斜杠：{value}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or ":" in value:
        raise ValueError(f"来源路径越出项目根目录：{value}")
    return pure.as_posix()


def validate_contract(contract: Mapping[str, Any]) -> None:
    """验证 V1 合同中的不可变身份、门槛、配置和安全边界。"""

    required_sections = {
        "protocol",
        "inputs",
        "forward_evaluation",
        "environment",
        "strategy_pool",
        "evidence_gates",
        "allocation",
        "governance",
        "safety",
        "outputs",
    }
    missing_sections = required_sections.difference(contract)
    if missing_sections:
        raise ContractError(f"V1 配置缺少章节：{sorted(missing_sections)}")

    protocol = contract["protocol"]
    _require_equal(protocol.get("candidate_id"), "REGIME_AWARE_STRATEGY_ALLOCATOR_V1", "candidate_id")
    _require_equal(protocol.get("version"), "1.0.0", "version")
    _require_equal(
        protocol.get("status"),
        "PREREGISTERED_BEFORE_FIRST_FORWARD_EVIDENCE",
        "protocol.status",
    )
    _require_equal(
        protocol.get("evidence_label"),
        "META_FORWARD_ONLY_HISTORICAL_COMPONENT_RESULTS_ALREADY_VIEWED",
        "protocol.evidence_label",
    )
    _require_equal(
        protocol.get("true_forward_start_policy"),
        "STRICTLY_AFTER_MANIFEST_FROZEN_AT",
        "true_forward_start_policy",
    )
    _require_equal(protocol.get("historical_component_result_cutoff"), "2026-08-14", "历史污染截止日")
    _require_equal(protocol.get("research_only"), True, "research_only")

    inputs = contract["inputs"]
    _require_equal(inputs.get("missing_evidence_policy"), "NO_VIEW_CASH_ONLY", "缺失证据政策")
    _require_equal(
        inputs.get("future_available_record_policy"),
        "IGNORE_FOR_CURRENT_DECISION_BUT_VALIDATE_SCHEMA",
        "未来可得记录政策",
    )
    _require_equal(inputs.get("future_decision_at_allowed"), False, "未来决策时点政策")
    _require_equal(inputs.get("stale_record_carry_forward_allowed"), False, "旧记录沿用政策")
    _require_equal(inputs.get("append_only_evidence_required"), True, "追加式证据政策")
    _require_equal(inputs.get("forward_environment_binding_required"), True, "前向环境绑定政策")
    _require_equal(
        inputs.get("environment_episode_ids_may_be_user_supplied"),
        False,
        "环境片段 ID 派生政策",
    )
    _require_equal(inputs.get("maximum_environment_age_hours"), 24, "环境证据最大年龄")
    _require_equal(inputs.get("source_hash_required_when_view_is_pass"), True, "来源哈希门")
    _safe_relative_path(inputs.get("environment_evidence_jsonl"))
    _safe_relative_path(inputs.get("strategy_evidence_jsonl"))
    _safe_relative_path(inputs.get("forward_observation_ledger_jsonl"))

    forward_evaluation = contract["forward_evaluation"]
    _require_equal(
        forward_evaluation.get("evaluator_id"),
        "REGIME_AWARE_STRATEGY_FORWARD_EVIDENCE_V1",
        "forward_evaluation.evaluator_id",
    )
    _safe_relative_path(forward_evaluation.get("immutable_report_directory"))
    _safe_relative_path(forward_evaluation.get("latest_build_status_json"))
    _safe_relative_path(forward_evaluation.get("immutable_build_receipt_directory"))
    _require_equal(forward_evaluation.get("annualization_periods"), 365, "前向年化周期")
    _require_equal(
        forward_evaluation.get("observation_day_timezone"),
        "Asia/Shanghai",
        "前向观察日时区",
    )
    _require_equal(
        forward_evaluation.get("observation_return_horizon_hours"),
        24,
        "前向收益观察周期",
    )
    _require_equal(
        forward_evaluation.get("maximum_outcome_availability_lag_hours"),
        24,
        "结果最大可得延迟",
    )
    _require_equal(
        forward_evaluation.get("rolling_qualification_window_observation_days"),
        504,
        "滚动资格窗口",
    )
    _require_equal(
        forward_evaluation.get("recent_health_window_observation_days"),
        90,
        "近期健康窗口",
    )
    _require_equal(
        tuple(forward_evaluation.get("allowed_observation_statuses", [])),
        (
            "PASS",
            "NO_VIEW",
            "RUNNING",
            "PARTIAL_SUCCESS",
            "FAILED",
            "PROGRAM_FAILED",
            "SKIPPED",
            "CENSORED_NO_OUTCOME",
        ),
        "前向观察状态集合",
    )
    bootstrap = forward_evaluation.get("bootstrap", {})
    frozen_bootstrap = {
        "method": "PAIRED_CIRCULAR_MOVING_BLOCK",
        "repetitions": 2000,
        "block_length_observations": 20,
        "lower_quantile": 0.05,
        "random_seed": 20260827,
    }
    for name, expected in frozen_bootstrap.items():
        _require_equal(bootstrap.get(name), expected, f"forward_evaluation.bootstrap.{name}")
    deflated = forward_evaluation.get("deflated_sharpe", {})
    _require_equal(deflated.get("independent_trials"), 35, "Deflated Sharpe 冻结试验数")
    _require_equal(
        float(deflated.get("euler_mascheroni", math.nan)),
        0.5772156649015329,
        "Euler-Mascheroni 常数",
    )
    pbo = forward_evaluation.get("probability_backtest_overfit", {})
    frozen_pbo = {
        "method": "COMBINATORIALLY_SYMMETRIC_CONTIGUOUS_PARTITIONS",
        "candidate_universe": "PERMITTED_STRATEGIES_PLUS_STATIC_EQUAL_RISK_BASELINE",
        "baseline_candidate_id": "STATIC_EQUAL_RISK_FROZEN_POOL",
        "partition_count": 8,
        "training_partition_count": 4,
        "minimum_complete_observation_days": 504,
    }
    for name, expected in frozen_pbo.items():
        _require_equal(pbo.get(name), expected, f"forward_evaluation.pbo.{name}")
    cost_reconciliation = forward_evaluation.get("cost_reconciliation", {})
    frozen_cost_reconciliation = {
        "net_return_formula": "STRATEGY_NET_RETURN_EQUALS_GROSS_RETURN_MINUS_SUM_COMPONENT_COSTS",
        "numeric_tolerance": 1.0e-12,
        "allocator_switch_cost_source": "FROZEN_STRATEGY_POOL_BPS",
        "stress_component_cost_must_not_be_below_base": True,
        "pass_requires_exact_required_component_set": True,
        "incomplete_statuses_require_null_returns_and_empty_costs": True,
    }
    for name, expected in frozen_cost_reconciliation.items():
        _require_equal(
            cost_reconciliation.get(name),
            expected,
            f"forward_evaluation.cost_reconciliation.{name}",
        )
    confirmation = forward_evaluation.get("confirmation", {})
    frozen_confirmation = {
        "minimum_elapsed_hours": 168,
        "minimum_new_complete_observation_days": 7,
        "requires_later_evaluation_window_end": True,
        "failure_resets_count": True,
    }
    for name, expected in frozen_confirmation.items():
        _require_equal(
            confirmation.get(name),
            expected,
            f"forward_evaluation.confirmation.{name}",
        )
    _require_equal(
        forward_evaluation.get("first_statistical_pass_status"),
        "CONDITIONALLY_PASSING_CONFIRMATION_PENDING",
        "首次统计通过状态",
    )
    _require_equal(
        forward_evaluation.get("confirmed_pass_status"),
        "CONDITIONAL_FORWARD_PASSED",
        "确认通过状态",
    )
    _require_equal(
        forward_evaluation.get("insufficient_status"),
        "INSUFFICIENT_EVIDENCE",
        "证据不足状态",
    )

    environment = contract["environment"]
    _require_equal(environment.get("scope"), "DIGITAL_ASSET", "环境范围")
    _require_equal(
        tuple(environment.get("allowed_data_statuses", [])),
        (
            "PASS",
            "NO_VIEW",
            "RUNNING",
            "PARTIAL_SUCCESS",
            "FAILED",
            "PROGRAM_FAILED",
            "SKIPPED",
        ),
        "环境证据状态集合",
    )
    _require_equal(
        environment.get("minimum_same_environment_episode_separation_hours"),
        168,
        "同类环境独立片段最短分隔",
    )
    feature_names = tuple(environment.get("feature_names", []))
    expected_features = (
        "slow_trend_score",
        "fast_trend_score",
        "breadth_score",
        "risk_score",
        "liquidity_score",
        "carry_opportunity_score",
    )
    _require_equal(feature_names, expected_features, "环境特征顺序")
    _require_equal(
        tuple(environment.get("classification_precedence", [])),
        EXPECTED_ENVIRONMENTS,
        "环境分类优先级",
    )
    feature_ranges = environment.get("feature_ranges", {})
    for name in expected_features:
        if name not in feature_ranges or len(feature_ranges[name]) != 2:
            raise ContractError(f"环境特征 {name} 缺少冻结范围")
        lower, upper = (float(value) for value in feature_ranges[name])
        if not lower < upper:
            raise ContractError(f"环境特征 {name} 范围无效")

    thresholds = environment["thresholds"]
    frozen_thresholds = {
        "high_risk_minimum": 0.70,
        "carry_opportunity_minimum": 0.70,
        "liquid_minimum": 0.60,
        "persistent_trend_absolute_minimum": 0.20,
        "persistent_up_breadth_minimum": 0.50,
        "persistent_down_breadth_maximum": 0.50,
        "transition_fast_slow_gap_minimum": 0.80,
        "mean_reverting_trend_absolute_maximum": 0.20,
    }
    for name, expected in frozen_thresholds.items():
        _require_equal(float(thresholds.get(name, math.nan)), expected, f"environment.thresholds.{name}")

    pool = contract["strategy_pool"]
    if not isinstance(pool, list):
        raise ContractError("strategy_pool 必须是数组")
    strategy_ids = tuple(item.get("strategy_id") for item in pool)
    _require_equal(strategy_ids, EXPECTED_STRATEGY_IDS, "冻结策略池")
    for item in pool:
        if not isinstance(item.get("family"), str) or not item["family"]:
            raise ContractError(f"{item.get('strategy_id')} 缺少策略族")
        environments = item.get("permitted_environments")
        if not isinstance(environments, list) or not environments:
            raise ContractError(f"{item['strategy_id']} 缺少允许环境")
        unknown = set(environments).difference(EXPECTED_ENVIRONMENTS)
        if unknown:
            raise ContractError(f"{item['strategy_id']} 包含未知环境：{sorted(unknown)}")
        _safe_relative_path(item.get("component_manifest"))
        _safe_relative_path(item.get("component_status_source"))
        costs = item.get("required_cost_components")
        if (
            not isinstance(costs, list)
            or any(not isinstance(cost, str) or not cost for cost in costs)
            or "allocator_switching" not in costs
        ):
            raise ContractError(f"{item['strategy_id']} 成本项不完整")
        if len(costs) != len(set(costs)):
            raise ContractError(f"{item['strategy_id']} 成本项重复")
        if float(item.get("base_allocator_switch_cost_bps", -1.0)) != 1.0:
            raise ContractError(f"{item['strategy_id']} 基础分配成本不得偏离 1bp")
        if float(item.get("stress_allocator_switch_cost_bps", -1.0)) != 10.0:
            raise ContractError(f"{item['strategy_id']} 压力分配成本不得偏离 10bp")

    gates = contract["evidence_gates"]
    frozen_gates = {
        "required_evidence_type": "TRUE_FORWARD_CONDITIONAL",
        "required_meta_evidence_status": "CONDITIONAL_FORWARD_PASSED",
        "minimum_forward_observation_days": 504,
        "minimum_independent_environment_episodes": 12,
        "minimum_consecutive_entry_confirmations": 2,
        "maximum_strategy_evidence_age_hours": 192,
        "edge_lower_bound_operator": "STRICTLY_GREATER_THAN",
        "minimum_deflated_sharpe_probability": 0.95,
        "maximum_probability_backtest_overfit": 0.10,
        "maximum_environment_profit_concentration": 0.50,
        "maximum_forecast_annualized_volatility": 3.0,
        "historical_contamination_allowed": False,
        "benchmark_set_complete_required": True,
        "cost_model_complete_required": True,
    }
    for name, expected in frozen_gates.items():
        _require_equal(gates.get(name), expected, f"evidence_gates.{name}")
    for edge_name in (
        "minimum_base_net_conditional_edge_lcb_after_all_costs",
        "minimum_stress_net_conditional_edge_lcb_after_all_costs",
        "minimum_stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs",
        "minimum_recent_stress_net_edge_vs_cash_after_all_costs",
        "minimum_recent_stress_net_incremental_vs_static_equal_risk_after_all_costs",
    ):
        _require_equal(float(gates.get(edge_name, math.nan)), 0.0, f"evidence_gates.{edge_name}")
    required_benchmarks = gates.get("required_benchmarks")
    if (
        not isinstance(required_benchmarks, list)
        or any(not isinstance(item, str) or not item for item in required_benchmarks)
        or len(required_benchmarks) != len(set(required_benchmarks))
    ):
        raise ContractError("required_benchmarks 必须是非重复数组")
    forbidden_fields = gates.get("forbidden_selection_fields")
    expected_forbidden = {
        "trailing_return",
        "trailing_sharpe",
        "recent_rank",
        "winner_rank",
        "best_strategy",
        "oracle_weight",
    }
    _require_equal(set(forbidden_fields or []), expected_forbidden, "禁止赢家追逐字段")

    allocation = contract["allocation"]
    frozen_allocation = {
        "score_numerator": "stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs",
        "score_denominator": "forecast_annualized_volatility",
        "maximum_total_strategy_weight": 0.80,
        "minimum_cash_weight": 0.20,
        "maximum_single_strategy_weight": 0.35,
        "maximum_single_family_weight": 0.50,
        "maximum_positive_risk_increase_per_decision": 0.25,
        "risk_reduction_is_never_blocked": True,
        "negative_weights_allowed": False,
        "leverage_allowed": False,
        "cash_asset_id": "CASH_CNY",
        "empty_eligible_set_result": "CASH_ONLY",
    }
    for name, expected in frozen_allocation.items():
        _require_equal(allocation.get(name), expected, f"allocation.{name}")
    if not math.isclose(
        float(allocation["maximum_total_strategy_weight"]) + float(allocation["minimum_cash_weight"]),
        1.0,
        abs_tol=1e-12,
    ):
        raise ContractError("最大策略权重与最低现金权重必须合计为 1")

    governance_expected = {
        "rejected_component_status_is_relabelled": False,
        "component_parameters_may_change": False,
        "strategy_pool_may_expand_after_freeze": False,
        "failed_environment_may_be_removed": False,
        "threshold_rescue_after_result": False,
        "historical_backfill_as_forward_evidence": False,
        "latest_winner_selection_used": False,
    }
    for name, expected in governance_expected.items():
        _require_equal(contract["governance"].get(name), expected, f"governance.{name}")

    for name, value in contract["safety"].items():
        _require_equal(value, False, f"safety.{name}")
    expected_safety = {
        "paper_position_generation",
        "shadow_signal_generation",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_or_exchange_connection_enabled",
        "live_trading_authorized",
    }
    _require_equal(set(contract["safety"]), expected_safety, "安全边界字段")

    outputs = contract["outputs"]
    for name in ("latest_status_json", "latest_status_markdown", "immutable_receipt_directory"):
        _safe_relative_path(outputs.get(name))


def parse_zoned_datetime(value: Any, label: str) -> datetime:
    """解析必须带时区的 ISO 时间。"""

    if not isinstance(value, str):
        raise EvidenceError(f"{label} 必须是带时区的 ISO 时间")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise EvidenceError(f"{label} 不是合法 ISO 时间：{value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise EvidenceError(f"{label} 必须包含时区")
    return parsed


def load_jsonl(path: Path, *, allow_missing: bool = True) -> list[dict[str, Any]]:
    """读取追加式 JSONL；缺失文件按合同返回空证据。"""

    if not path.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(path)
    if not path.is_file():
        raise EvidenceError(f"证据路径不是文件：{path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceError(f"{path.name} 第 {line_number} 行不是合法 JSON：{exc}") from exc
            if not isinstance(record, dict):
                raise EvidenceError(f"{path.name} 第 {line_number} 行必须是 JSON 对象")
            record = dict(record)
            record["_line_number"] = line_number
            records.append(record)
    return records


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise EvidenceError(f"{label} 必须是 64 位小写 SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise EvidenceError(f"{label} 必须是 64 位小写 SHA-256")
    return value


def _validate_source_binding(
    record: Mapping[str, Any],
    *,
    root: Path,
    required: bool,
    label: str,
) -> dict[str, Any]:
    source_path = record.get("source_path")
    source_sha = record.get("source_sha256")
    if source_path is None or source_sha is None:
        if source_path is not None or source_sha is not None:
            raise EvidenceError(f"{label} 的来源路径和哈希必须同时存在或同时缺失")
        if required:
            raise EvidenceError(f"{label} 缺少必需来源绑定")
        return {"path": None, "sha256": None, "verified": False}
    try:
        relative = _safe_relative_path(source_path)
    except ValueError as exc:
        raise EvidenceError(f"{label}：{exc}") from exc
    expected = _validate_sha256(source_sha, f"{label}.source_sha256")
    absolute = root / PurePosixPath(relative)
    if not absolute.is_file():
        raise EvidenceError(f"{label} 来源文件不存在：{relative}")
    actual = sha256_file(absolute)
    if actual != expected:
        raise EvidenceError(
            f"{label} 来源 SHA-256 不一致：{relative}，expected={expected}，actual={actual}"
        )
    return {"path": relative, "sha256": expected, "verified": True}


def _validate_append_only_report_input(
    value: Any,
    *,
    expected_path: str,
    count_field: str,
    root: Path,
    label: str,
) -> dict[str, Any]:
    expected_fields = {"path", "bytes", "sha256", count_field}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise EvidenceError(f"{label} 追加式输入绑定字段不完整")
    if value["path"] != expected_path:
        raise EvidenceError(f"{label} 追加式输入路径与冻结合同不一致")
    relative = _safe_relative_path(value["path"])
    byte_count = value["bytes"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise EvidenceError(f"{label}.bytes 必须是正整数")
    record_count = value[count_field]
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count <= 0:
        raise EvidenceError(f"{label}.{count_field} 必须是正整数")
    expected_sha = _validate_sha256(value["sha256"], f"{label}.sha256")
    path = root / PurePosixPath(relative)
    if not path.is_file():
        raise EvidenceError(f"{label} 追加式输入文件不存在：{relative}")
    actual_sha = sha256_file_prefix(path, byte_count)
    if actual_sha != expected_sha:
        raise EvidenceError(
            f"{label} 追加式输入前缀 SHA-256 不一致：expected={expected_sha}，actual={actual_sha}"
        )
    with path.open("rb") as handle:
        prefix_bytes = handle.read(byte_count)
    try:
        prefix_text = prefix_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvidenceError(f"{label} 追加式输入前缀不是 UTF-8") from exc
    parsed_count = 0
    for line_number, line in enumerate(prefix_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"{label} 前缀第 {line_number} 行不是合法 JSON") from exc
        if not isinstance(parsed, dict):
            raise EvidenceError(f"{label} 前缀第 {line_number} 行必须是 JSON 对象")
        parsed_count += 1
    if parsed_count != record_count:
        raise EvidenceError(
            f"{label} 声明记录数与前缀实际记录数不一致：declared={record_count}，actual={parsed_count}"
        )
    return {
        "path": relative,
        "bytes": byte_count,
        "sha256": expected_sha,
        count_field: record_count,
        "prefix_verified": True,
    }


def _validate_evaluator_report(
    payload: Any,
    *,
    raw_record: Mapping[str, Any],
    source: Mapping[str, Any],
    contract: Mapping[str, Any],
    manifest_tracked_content_sha256: str,
    recorded_at: datetime,
    available_at: datetime,
    root: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != EVALUATOR_REPORT_REQUIRED_FIELDS:
        raise EvidenceError(f"{label} 评价报告字段不完整或含未冻结字段")
    if payload["schema_version"] != "1.0.0":
        raise EvidenceError(f"{label} 评价报告 schema_version 必须为 1.0.0")
    if payload["evaluator_id"] != contract["forward_evaluation"]["evaluator_id"]:
        raise EvidenceError(f"{label} 不是冻结评价器生成的报告")
    if payload["candidate_id"] != contract["protocol"]["candidate_id"]:
        raise EvidenceError(f"{label} 评价报告 candidate_id 不匹配")
    if payload["allocator_tracked_content_sha256"] != manifest_tracked_content_sha256:
        raise EvidenceError(f"{label} 评价报告未绑定当前冻结受控实现")

    expected_report_directory = PurePosixPath(
        contract["forward_evaluation"]["immutable_report_directory"]
    )
    report_path = PurePosixPath(str(source["path"]))
    if (
        report_path.parent != expected_report_directory
        or report_path.name != f"{raw_record['evidence_id']}.json"
    ):
        raise EvidenceError(f"{label} 评价报告不在冻结不可变目录或文件名不匹配")

    manifest_relative = _safe_relative_path(payload["allocator_manifest_path"])
    if manifest_relative != "config/regime_aware_strategy_allocator_v1_manifest.json":
        raise EvidenceError(f"{label} 评价报告清单路径不匹配")
    manifest_path = root / PurePosixPath(manifest_relative)
    if not manifest_path.is_file():
        raise EvidenceError(f"{label} 评价报告绑定的清单不存在")
    expected_manifest_sha = _validate_sha256(
        payload["allocator_manifest_sha256"], f"{label}.allocator_manifest_sha256"
    )
    if sha256_file(manifest_path) != expected_manifest_sha:
        raise EvidenceError(f"{label} 评价报告清单 SHA-256 不一致")

    generated_at = parse_zoned_datetime(payload["generated_at"], f"{label}.generated_at")
    report_decision_at = parse_zoned_datetime(payload["decision_at"], f"{label}.decision_at")
    if generated_at != recorded_at or available_at != recorded_at:
        raise EvidenceError(f"{label} 评价报告生成、记录和可得时点必须完全一致")
    if report_decision_at > generated_at:
        raise EvidenceError(f"{label} 评价决策时点晚于报告生成时点")
    if raw_record.get("evaluation_window_end") is not None:
        report_window_end = parse_zoned_datetime(
            raw_record["evaluation_window_end"], f"{label}.evaluation_window_end"
        )
        if report_window_end > report_decision_at:
            raise EvidenceError(f"{label} 评价窗口终点晚于报告决策时点")

    expected_core = {
        key: value
        for key, value in raw_record.items()
        if key not in {"_line_number", "source_path", "source_sha256"}
    }
    if payload["evidence_core"] != expected_core:
        raise EvidenceError(f"{label} 与不可变评价报告 evidence_core 不一致")
    evaluation = payload["evaluation"]
    if not isinstance(evaluation, dict):
        raise EvidenceError(f"{label} 评价报告 evaluation 必须是对象")
    if evaluation.get("strategy_id") != raw_record["strategy_id"]:
        raise EvidenceError(f"{label} evaluation.strategy_id 不一致")
    if evaluation.get("environment_id") != raw_record["environment_id"]:
        raise EvidenceError(f"{label} evaluation.environment_id 不一致")
    if evaluation.get("decision_at") != payload["decision_at"]:
        raise EvidenceError(f"{label} evaluation.decision_at 不一致")
    mirrored_fields = (
        "meta_evidence_status",
        "evaluation_window_start",
        "evaluation_window_end",
        "latest_forward_observation_at",
        "latest_forward_data_status",
        "forward_observation_days",
        "recent_health_observation_days",
        "independent_environment_episodes",
        "consecutive_entry_confirmations",
        "base_net_conditional_edge_lcb_after_all_costs",
        "stress_net_conditional_edge_lcb_after_all_costs",
        "stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs",
        "recent_stress_net_edge_vs_cash_after_all_costs",
        "recent_stress_net_incremental_vs_static_equal_risk_after_all_costs",
        "deflated_sharpe_probability",
        "probability_backtest_overfit",
        "maximum_environment_profit_concentration",
        "forecast_annualized_volatility",
        "benchmark_set_complete",
        "benchmarks",
        "cost_model_complete",
        "included_cost_components",
    )
    for field in mirrored_fields:
        if evaluation.get(field) != raw_record[field]:
            raise EvidenceError(f"{label} evaluation.{field} 与资格记录不一致")

    ledger = _validate_append_only_report_input(
        payload["ledger"],
        expected_path=contract["inputs"]["forward_observation_ledger_jsonl"],
        count_field="validated_observation_count",
        root=root,
        label=f"{label}.ledger",
    )
    environment_evidence = _validate_append_only_report_input(
        payload["environment_evidence"],
        expected_path=contract["inputs"]["environment_evidence_jsonl"],
        count_field="validated_record_count",
        root=root,
        label=f"{label}.environment_evidence",
    )
    forward_days = raw_record["forward_observation_days"]
    if isinstance(forward_days, int) and ledger["validated_observation_count"] < forward_days:
        raise EvidenceError(f"{label} 账本观察数小于资格记录的前向日数")

    false_fields = (
        "performance_used_for_recent_winner_ranking",
        "historical_backfill_used",
        "paper_position_generation",
        "shadow_signal_generation",
        "order_generation",
        "live_trading_authorized",
    )
    for field in false_fields:
        if payload[field] is not False:
            raise EvidenceError(f"{label} 评价报告 {field} 必须为 false")
    return {
        "report_path": str(source["path"]),
        "report_sha256": str(source["sha256"]),
        "manifest_path": manifest_relative,
        "manifest_sha256": expected_manifest_sha,
        "ledger": ledger,
        "environment_evidence": environment_evidence,
        "verified": True,
    }


def _ensure_append_order(
    records: Iterable[Mapping[str, Any]],
    *,
    field: str,
    label: str,
    allow_duplicate_times: bool = False,
) -> None:
    previous: datetime | None = None
    seen: set[datetime] = set()
    for record in records:
        value = record[field]
        if value in seen and not allow_duplicate_times:
            raise EvidenceError(f"{label} 出现重复 {field}：{value.isoformat()}")
        if previous is not None and value < previous:
            raise EvidenceError(f"{label} 未按 {field} 追加排序")
        seen.add(value)
        previous = value


def validate_environment_records(
    records: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> list[dict[str, Any]]:
    """严格验证环境证据的模式、点时顺序、特征范围和来源哈希。"""

    environment = contract["environment"]
    feature_names = tuple(environment["feature_names"])
    feature_ranges = environment["feature_ranges"]
    allowed_statuses = set(environment["allowed_data_statuses"])
    if allowed_statuses != ENVIRONMENT_STATUSES:
        raise EvidenceError("环境证据状态集合与冻结实现不一致")
    normalized: list[dict[str, Any]] = []
    evidence_ids: set[str] = set()
    for ordinal, raw in enumerate(records, start=1):
        line = raw.get("_line_number", ordinal)
        label = f"环境证据第 {line} 行"
        fields = set(raw).difference({"_line_number"})
        if fields != ENVIRONMENT_REQUIRED_FIELDS:
            missing = sorted(ENVIRONMENT_REQUIRED_FIELDS - fields)
            extra = sorted(fields - ENVIRONMENT_REQUIRED_FIELDS)
            raise EvidenceError(f"{label} 字段不匹配：missing={missing}，extra={extra}")
        if raw["schema_version"] != "1.0.0":
            raise EvidenceError(f"{label} schema_version 必须为 1.0.0")
        evidence_id = raw["evidence_id"]
        if not isinstance(evidence_id, str) or not evidence_id:
            raise EvidenceError(f"{label} evidence_id 无效")
        if evidence_id in evidence_ids:
            raise EvidenceError(f"环境 evidence_id 重复：{evidence_id}")
        evidence_ids.add(evidence_id)
        if raw["environment_scope"] != environment["scope"]:
            raise EvidenceError(f"{label} environment_scope 不匹配")
        data_status = raw["data_status"]
        if data_status not in allowed_statuses:
            raise EvidenceError(f"{label} data_status 无效：{data_status!r}")
        observation_at = parse_zoned_datetime(raw["observation_at"], f"{label}.observation_at")
        available_at = parse_zoned_datetime(raw["available_at"], f"{label}.available_at")
        if observation_at > available_at:
            raise EvidenceError(f"{label} observation_at 晚于 available_at")
        reason = raw["reason"]
        if not isinstance(reason, str) or not reason:
            raise EvidenceError(f"{label} reason 必须是非空字符串")
        source = _validate_source_binding(
            raw,
            root=root,
            required=data_status == "PASS",
            label=label,
        )
        features = raw["features"]
        normalized_features: dict[str, float] | None = None
        if data_status == "PASS":
            if not isinstance(features, dict) or set(features) != set(feature_names):
                raise EvidenceError(f"{label} PASS 时必须提供完整冻结环境特征")
            normalized_features = {}
            for name in feature_names:
                value = features[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise EvidenceError(f"{label}.{name} 必须是数值")
                number = float(value)
                lower, upper = (float(item) for item in feature_ranges[name])
                if not math.isfinite(number) or not lower <= number <= upper:
                    raise EvidenceError(f"{label}.{name} 超出冻结范围 [{lower}, {upper}]")
                normalized_features[name] = number
        elif features is not None:
            raise EvidenceError(f"{label} 非 PASS 状态时 features 必须为 null")
        normalized.append(
            {
                **{key: value for key, value in raw.items() if key != "_line_number"},
                "observation_at": observation_at,
                "available_at": available_at,
                "features": normalized_features,
                "source_binding": source,
                "_line_number": line,
            }
        )
    _ensure_append_order(normalized, field="available_at", label="环境证据")
    _ensure_append_order(
        normalized,
        field="observation_at",
        label="环境证据",
        allow_duplicate_times=True,
    )
    return normalized


def classify_environment(
    records: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    decision_at: datetime,
) -> dict[str, Any]:
    """只用决策时点已经可得的最新环境记录进行固定分类。"""

    available = [record for record in records if record["available_at"] <= decision_at]
    if not available:
        return {
            "view_status": "NO_VIEW",
            "environment_id": None,
            "evidence_id": None,
            "available_at": None,
            "reason": "决策时点没有可得环境证据，禁止沿用旧状态",
            "features": None,
        }
    latest = available[-1]
    maximum_age_seconds = float(contract["inputs"]["maximum_environment_age_hours"]) * 3600.0
    age_seconds = (decision_at - latest["observation_at"]).total_seconds()
    if age_seconds < 0.0:
        raise EvidenceError("最新环境 observation_at 晚于决策时点")
    if age_seconds > maximum_age_seconds:
        return {
            "view_status": "NO_VIEW",
            "environment_id": None,
            "evidence_id": latest["evidence_id"],
            "available_at": latest["available_at"].isoformat(),
            "reason": "最新环境证据超过 24 小时，禁止沿用旧状态",
            "features": None,
        }
    if latest["data_status"] != "PASS":
        return {
            "view_status": "NO_VIEW",
            "environment_id": None,
            "evidence_id": latest["evidence_id"],
            "available_at": latest["available_at"].isoformat(),
            "reason": latest["reason"],
            "features": None,
        }

    features = latest["features"]
    thresholds = contract["environment"]["thresholds"]
    slow = features["slow_trend_score"]
    fast = features["fast_trend_score"]
    breadth = features["breadth_score"]
    risk = features["risk_score"]
    liquidity = features["liquidity_score"]
    carry = features["carry_opportunity_score"]
    trend_min = float(thresholds["persistent_trend_absolute_minimum"])

    if risk >= float(thresholds["high_risk_minimum"]):
        environment_id = "DEFENSIVE_HIGH_RISK"
    elif (
        carry >= float(thresholds["carry_opportunity_minimum"])
        and liquidity >= float(thresholds["liquid_minimum"])
    ):
        environment_id = "CARRY_DISPERSION"
    elif slow >= trend_min and fast >= trend_min and breadth >= float(
        thresholds["persistent_up_breadth_minimum"]
    ):
        environment_id = "TREND_PERSISTENT_UP"
    elif slow <= -trend_min and fast <= -trend_min and breadth <= float(
        thresholds["persistent_down_breadth_maximum"]
    ):
        environment_id = "TREND_PERSISTENT_DOWN"
    elif slow * fast < 0.0 or abs(fast - slow) >= float(
        thresholds["transition_fast_slow_gap_minimum"]
    ):
        environment_id = "TREND_TRANSITION"
    elif (
        abs(slow) <= float(thresholds["mean_reverting_trend_absolute_maximum"])
        and abs(fast) <= float(thresholds["mean_reverting_trend_absolute_maximum"])
        and liquidity >= float(thresholds["liquid_minimum"])
    ):
        environment_id = "MEAN_REVERTING_LIQUID"
    else:
        environment_id = "UNKNOWN"

    return {
        "view_status": "VIEW",
        "environment_id": environment_id,
        "evidence_id": latest["evidence_id"],
        "available_at": latest["available_at"].isoformat(),
        "reason": "环境记录与固定分类门均通过；环境本身不构成收益预测",
        "features": dict(features),
    }


def _nullable_integer(value: Any, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(f"{label} 必须是非负整数或 null")
    return value


def _nullable_number(value: Any, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} 必须是有限数值或 null")
    number = float(value)
    if not math.isfinite(number):
        raise EvidenceError(f"{label} 必须是有限数值或 null")
    return number


def validate_strategy_records(
    records: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    manifest_frozen_at: datetime,
    manifest_tracked_content_sha256: str | None = None,
    root: Path = ROOT,
) -> list[dict[str, Any]]:
    """验证策略条件证据及其真正前向、终态和来源约束。"""

    pool_by_id = {item["strategy_id"]: item for item in contract["strategy_pool"]}
    pool_ids = set(pool_by_id)
    required_benchmarks = set(contract["evidence_gates"]["required_benchmarks"])
    forbidden = set(contract["evidence_gates"]["forbidden_selection_fields"])
    required_safety = {
        "paper_position_generation",
        "shadow_signal_generation",
        "order_generation",
        "live_trading_authorized",
    }
    normalized: list[dict[str, Any]] = []
    evidence_ids: set[str] = set()
    terminal_groups: set[tuple[str, str]] = set()
    group_available_times: set[tuple[str, str, datetime]] = set()

    for ordinal, raw in enumerate(records, start=1):
        line = raw.get("_line_number", ordinal)
        label = f"策略证据第 {line} 行"
        fields = set(raw).difference({"_line_number"})
        forbidden_present = sorted(fields.intersection(forbidden))
        if forbidden_present:
            raise EvidenceError(f"{label} 含禁止的赢家追逐字段：{forbidden_present}")
        if fields != STRATEGY_REQUIRED_FIELDS:
            missing = sorted(STRATEGY_REQUIRED_FIELDS - fields)
            extra = sorted(fields - STRATEGY_REQUIRED_FIELDS)
            raise EvidenceError(f"{label} 字段不匹配：missing={missing}，extra={extra}")
        if raw["schema_version"] != "1.0.0":
            raise EvidenceError(f"{label} schema_version 必须为 1.0.0")
        evidence_id = raw["evidence_id"]
        if not isinstance(evidence_id, str) or not evidence_id:
            raise EvidenceError(f"{label} evidence_id 无效")
        if evidence_id in evidence_ids:
            raise EvidenceError(f"策略 evidence_id 重复：{evidence_id}")
        evidence_ids.add(evidence_id)
        strategy_id = raw["strategy_id"]
        environment_id = raw["environment_id"]
        if strategy_id not in pool_ids:
            raise EvidenceError(f"{label} 策略不在冻结池：{strategy_id!r}")
        if environment_id not in EXPECTED_ENVIRONMENTS:
            raise EvidenceError(f"{label} 环境不在冻结分类：{environment_id!r}")
        group = (strategy_id, environment_id)
        available_at = parse_zoned_datetime(raw["available_at"], f"{label}.available_at")
        group_time = (strategy_id, environment_id, available_at)
        if group_time in group_available_times:
            raise EvidenceError(f"{label} 同一策略和环境出现重复 available_at")
        group_available_times.add(group_time)
        if group in terminal_groups and raw["meta_evidence_status"] != "REJECTED_FROZEN":
            raise EvidenceError(f"{label} 试图恢复同一 V1 内的 REJECTED_FROZEN 终态")
        meta_status = raw["meta_evidence_status"]
        if meta_status not in META_EVIDENCE_STATUSES:
            raise EvidenceError(f"{label} meta_evidence_status 无效：{meta_status!r}")
        if meta_status == "REJECTED_FROZEN":
            terminal_groups.add(group)

        recorded_at = parse_zoned_datetime(raw["recorded_at"], f"{label}.recorded_at")
        if recorded_at > available_at:
            raise EvidenceError(f"{label} recorded_at 晚于 available_at")
        evaluation_start = (
            None
            if raw["evaluation_window_start"] is None
            else parse_zoned_datetime(raw["evaluation_window_start"], f"{label}.evaluation_window_start")
        )
        evaluation_end = (
            None
            if raw["evaluation_window_end"] is None
            else parse_zoned_datetime(raw["evaluation_window_end"], f"{label}.evaluation_window_end")
        )
        latest_forward_observation_at = (
            None
            if raw["latest_forward_observation_at"] is None
            else parse_zoned_datetime(
                raw["latest_forward_observation_at"],
                f"{label}.latest_forward_observation_at",
            )
        )
        latest_forward_data_status = raw["latest_forward_data_status"]
        if (latest_forward_observation_at is None) != (latest_forward_data_status is None):
            raise EvidenceError(f"{label} 最新前向观察时点和状态必须同时存在或同时为 null")
        allowed_forward_statuses = set(
            contract["forward_evaluation"]["allowed_observation_statuses"]
        )
        if (
            latest_forward_data_status is not None
            and latest_forward_data_status not in allowed_forward_statuses
        ):
            raise EvidenceError(f"{label} latest_forward_data_status 不在冻结状态集合")
        if latest_forward_observation_at is not None and latest_forward_observation_at > recorded_at:
            raise EvidenceError(f"{label} 最新前向观察晚于记录时间")
        if (evaluation_start is None) != (evaluation_end is None):
            raise EvidenceError(f"{label} 评价窗口起止必须同时存在或同时为 null")
        if evaluation_start is not None:
            if evaluation_start > evaluation_end:
                raise EvidenceError(f"{label} 评价窗口起点晚于终点")
            if evaluation_start <= manifest_frozen_at:
                raise EvidenceError(f"{label} 使用清单冻结时点之前的历史区间取得资格")
            if evaluation_end > recorded_at:
                raise EvidenceError(f"{label} 评价窗口终点晚于记录时间")

        source = _validate_source_binding(raw, root=root, required=True, label=label)
        if manifest_tracked_content_sha256 is None:
            raise EvidenceError(f"{label} 缺少受控实现总哈希，不能验证评价器来源")
        source_payload = json.loads(
            (root / PurePosixPath(source["path"])).read_text(encoding="utf-8")
        )
        evaluator_report_binding = _validate_evaluator_report(
            source_payload,
            raw_record=raw,
            source=source,
            contract=contract,
            manifest_tracked_content_sha256=manifest_tracked_content_sha256,
            recorded_at=recorded_at,
            available_at=available_at,
            root=root,
            label=label,
        )
        if not isinstance(raw["historical_contamination"], bool):
            raise EvidenceError(f"{label} historical_contamination 必须是布尔值")
        integers = {
            name: _nullable_integer(raw[name], f"{label}.{name}")
            for name in (
                "forward_observation_days",
                "recent_health_observation_days",
                "independent_environment_episodes",
                "consecutive_entry_confirmations",
            )
        }
        numbers = {
            name: _nullable_number(raw[name], f"{label}.{name}")
            for name in (
                "base_net_conditional_edge_lcb_after_all_costs",
                "stress_net_conditional_edge_lcb_after_all_costs",
                "stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs",
                "recent_stress_net_edge_vs_cash_after_all_costs",
                "recent_stress_net_incremental_vs_static_equal_risk_after_all_costs",
                "deflated_sharpe_probability",
                "probability_backtest_overfit",
                "maximum_environment_profit_concentration",
                "forecast_annualized_volatility",
            )
        }
        for probability_name in (
            "deflated_sharpe_probability",
            "probability_backtest_overfit",
            "maximum_environment_profit_concentration",
        ):
            value = numbers[probability_name]
            if value is not None and not 0.0 <= value <= 1.0:
                raise EvidenceError(f"{label}.{probability_name} 必须位于 0 至 1")
        if numbers["forecast_annualized_volatility"] is not None and numbers[
            "forecast_annualized_volatility"
        ] <= 0.0:
            raise EvidenceError(f"{label}.forecast_annualized_volatility 必须大于 0")

        if not isinstance(raw["benchmark_set_complete"], bool):
            raise EvidenceError(f"{label}.benchmark_set_complete 必须是布尔值")
        if not isinstance(raw["cost_model_complete"], bool):
            raise EvidenceError(f"{label}.cost_model_complete 必须是布尔值")
        if (
            not isinstance(raw["benchmarks"], list)
            or any(not isinstance(item, str) or not item for item in raw["benchmarks"])
            or len(raw["benchmarks"]) != len(set(raw["benchmarks"]))
        ):
            raise EvidenceError(f"{label}.benchmarks 必须是非重复数组")
        if (
            not isinstance(raw["included_cost_components"], list)
            or any(
                not isinstance(item, str) or not item
                for item in raw["included_cost_components"]
            )
            or len(raw["included_cost_components"])
            != len(set(raw["included_cost_components"]))
        ):
            raise EvidenceError(f"{label}.included_cost_components 必须是非重复数组")
        if raw["benchmark_set_complete"]:
            if set(raw["benchmarks"]) != required_benchmarks:
                raise EvidenceError(f"{label} 完整基准集合必须与冻结合同完全一致")
        elif raw["benchmarks"]:
            raise EvidenceError(f"{label} 基准不完整时 benchmarks 必须为空数组")
        strategy_required_costs = set(pool_by_id[strategy_id]["required_cost_components"])
        if raw["cost_model_complete"]:
            if set(raw["included_cost_components"]) != strategy_required_costs:
                raise EvidenceError(f"{label} 完整成本集合必须与冻结策略完全一致")
        elif raw["included_cost_components"]:
            raise EvidenceError(
                f"{label} 成本模型不完整时 included_cost_components 必须为空数组"
            )
        if set(fields).intersection(required_safety) != required_safety:
            raise EvidenceError(f"{label} 缺少安全字段")
        for safety_name in required_safety:
            if raw[safety_name] is not False:
                raise EvidenceError(f"{label}.{safety_name} 必须为 false")

        if meta_status == "CONDITIONAL_FORWARD_PASSED":
            if evaluation_start is None:
                raise EvidenceError(f"{label} 通过记录必须包含评价窗口")
            required_non_null = [*integers.values(), *numbers.values()]
            if any(value is None for value in required_non_null):
                raise EvidenceError(f"{label} 通过记录的成熟度和统计指标不得为 null")
            if latest_forward_data_status != "PASS":
                raise EvidenceError(f"{label} 通过记录的最新前向观察状态必须为 PASS")

        normalized.append(
            {
                **{key: value for key, value in raw.items() if key != "_line_number"},
                "recorded_at": recorded_at,
                "available_at": available_at,
                "evaluation_window_start": evaluation_start,
                "evaluation_window_end": evaluation_end,
                "latest_forward_observation_at": latest_forward_observation_at,
                **integers,
                **numbers,
                "source_binding": source,
                "evaluator_report_binding": evaluator_report_binding,
                "_line_number": line,
            }
        )
    _ensure_append_order(
        normalized,
        field="available_at",
        label="策略证据",
        allow_duplicate_times=True,
    )
    return normalized


def _latest_matching_strategy_record(
    records: Iterable[Mapping[str, Any]],
    *,
    strategy_id: str,
    environment_id: str,
    decision_at: datetime,
) -> Mapping[str, Any] | None:
    matching = [
        record
        for record in records
        if record["strategy_id"] == strategy_id
        and record["environment_id"] == environment_id
        and record["available_at"] <= decision_at
    ]
    return matching[-1] if matching else None


def evaluate_strategy_eligibility(
    strategy: Mapping[str, Any],
    record: Mapping[str, Any] | None,
    environment_result: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    manifest_frozen_at: datetime,
    decision_at: datetime,
) -> dict[str, Any]:
    """逐门评价单条策略在当前环境下是否取得研究配置资格。"""

    gates_contract = contract["evidence_gates"]
    strategy_id = strategy["strategy_id"]
    environment_id = environment_result.get("environment_id")
    permitted = set(strategy["permitted_environments"])
    gates: dict[str, bool] = {
        "environment_view_pass": environment_result["view_status"] == "VIEW",
        "environment_is_permitted": environment_id in permitted,
        "conditional_evidence_exists": record is not None,
    }
    if record is None:
        return {
            "strategy_id": strategy_id,
            "family": strategy["family"],
            "component_status_does_not_grant_eligibility": True,
            "eligible": False,
            "score": 0.0,
            "evidence_id": None,
            "gates": gates,
            "failed_gates": [name for name, passed in gates.items() if not passed],
            "reason": "当前环境没有当时可得的真正前向条件证据",
        }

    required_benchmarks = set(gates_contract["required_benchmarks"])
    required_costs = set(strategy["required_cost_components"])
    evaluation_start = record["evaluation_window_start"]
    evaluation_end = record["evaluation_window_end"]
    base_edge = record["base_net_conditional_edge_lcb_after_all_costs"]
    stress_edge = record["stress_net_conditional_edge_lcb_after_all_costs"]
    incremental_edge = record[
        "stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs"
    ]
    recent_stress_edge = record["recent_stress_net_edge_vs_cash_after_all_costs"]
    recent_incremental_edge = record[
        "recent_stress_net_incremental_vs_static_equal_risk_after_all_costs"
    ]
    dsr = record["deflated_sharpe_probability"]
    pbo = record["probability_backtest_overfit"]
    concentration = record["maximum_environment_profit_concentration"]
    forecast_volatility = record["forecast_annualized_volatility"]

    gates.update(
        {
            "evidence_type_is_true_forward": record["evidence_type"]
            == gates_contract["required_evidence_type"],
            "meta_status_passed": record["meta_evidence_status"]
            == gates_contract["required_meta_evidence_status"],
            "latest_forward_data_status_pass": record["latest_forward_data_status"]
            == "PASS",
            "evaluation_window_exists": evaluation_start is not None and evaluation_end is not None,
            "evaluation_starts_after_manifest_freeze": evaluation_start is not None
            and evaluation_start > manifest_frozen_at,
            "evaluation_ends_before_decision": evaluation_end is not None
            and evaluation_end <= decision_at,
            "record_available_at_decision": record["available_at"] <= decision_at,
            "strategy_evidence_fresh_at_decision": 0.0
            <= (decision_at - record["available_at"]).total_seconds()
            <= float(gates_contract["maximum_strategy_evidence_age_hours"]) * 3600.0,
            "minimum_forward_days": record["forward_observation_days"] is not None
            and record["forward_observation_days"]
            >= int(gates_contract["minimum_forward_observation_days"]),
            "recent_health_window_complete": record["recent_health_observation_days"]
            is not None
            and record["recent_health_observation_days"]
            >= int(contract["forward_evaluation"]["recent_health_window_observation_days"]),
            "minimum_environment_episodes": record["independent_environment_episodes"] is not None
            and record["independent_environment_episodes"]
            >= int(gates_contract["minimum_independent_environment_episodes"]),
            "minimum_entry_confirmations": record["consecutive_entry_confirmations"] is not None
            and record["consecutive_entry_confirmations"]
            >= int(gates_contract["minimum_consecutive_entry_confirmations"]),
            "historical_contamination_absent": record["historical_contamination"] is False,
            "base_edge_lcb_positive_after_all_costs": base_edge is not None
            and base_edge
            > float(
                gates_contract[
                    "minimum_base_net_conditional_edge_lcb_after_all_costs"
                ]
            ),
            "stress_edge_lcb_positive_after_all_costs": stress_edge is not None
            and stress_edge
            > float(
                gates_contract[
                    "minimum_stress_net_conditional_edge_lcb_after_all_costs"
                ]
            ),
            "incremental_vs_static_equal_risk_lcb_positive_after_all_costs": incremental_edge
            is not None
            and incremental_edge
            > float(
                gates_contract[
                    "minimum_stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs"
                ]
            ),
            "recent_stress_edge_vs_cash_positive_after_all_costs": recent_stress_edge
            is not None
            and recent_stress_edge
            > float(
                gates_contract["minimum_recent_stress_net_edge_vs_cash_after_all_costs"]
            ),
            "recent_stress_incremental_vs_static_positive_after_all_costs": recent_incremental_edge
            is not None
            and recent_incremental_edge
            > float(
                gates_contract[
                    "minimum_recent_stress_net_incremental_vs_static_equal_risk_after_all_costs"
                ]
            ),
            "deflated_sharpe_probability_pass": dsr is not None
            and dsr >= float(gates_contract["minimum_deflated_sharpe_probability"]),
            "probability_backtest_overfit_pass": pbo is not None
            and pbo <= float(gates_contract["maximum_probability_backtest_overfit"]),
            "environment_profit_concentration_pass": concentration is not None
            and concentration
            <= float(gates_contract["maximum_environment_profit_concentration"]),
            "forecast_volatility_pass": forecast_volatility is not None
            and 0.0 < forecast_volatility
            <= float(gates_contract["maximum_forecast_annualized_volatility"]),
            "benchmark_set_complete": record["benchmark_set_complete"] is True
            and set(record["benchmarks"]) == required_benchmarks,
            "cost_model_complete": record["cost_model_complete"] is True
            and set(record["included_cost_components"]) == required_costs,
            "research_only_safety_pass": all(
                record[name] is False
                for name in (
                    "paper_position_generation",
                    "shadow_signal_generation",
                    "order_generation",
                    "live_trading_authorized",
                )
            ),
        }
    )
    failed = [name for name, passed in gates.items() if not passed]
    eligible = not failed
    score = 0.0
    if eligible:
        score = float(incremental_edge / forecast_volatility)
        if not math.isfinite(score) or score <= 0.0:
            eligible = False
            gates["positive_finite_allocation_score"] = False
            failed.append("positive_finite_allocation_score")
            score = 0.0
        else:
            gates["positive_finite_allocation_score"] = True
    return {
        "strategy_id": strategy_id,
        "family": strategy["family"],
        "component_status_does_not_grant_eligibility": True,
        "eligible": eligible,
        "score": score,
        "evidence_id": record["evidence_id"],
        "evidence_available_at": record["available_at"].isoformat(),
        "meta_evidence_status": record["meta_evidence_status"],
        "gates": gates,
        "failed_gates": failed,
        "reason": "全部冻结条件门通过" if eligible else "至少一个冻结条件门未通过",
    }


def _capped_proportional(
    scores: Mapping[str, float],
    *,
    budget: float,
    cap: float,
) -> dict[str, float]:
    """按正分数进行确定性带上限分配；无法使用的预算留给现金。"""

    result = {key: 0.0 for key in sorted(scores)}
    remaining = {key for key, value in scores.items() if value > 0.0}
    remaining_budget = max(0.0, float(budget))
    tolerance = 1e-15
    while remaining and remaining_budget > tolerance:
        total_score = sum(float(scores[key]) for key in remaining)
        if total_score <= 0.0:
            break
        proposals = {
            key: remaining_budget * float(scores[key]) / total_score for key in remaining
        }
        capped = [key for key, proposal in proposals.items() if proposal > cap + tolerance]
        if not capped:
            for key, proposal in proposals.items():
                result[key] += proposal
            remaining_budget = 0.0
            break
        for key in sorted(capped):
            room = max(0.0, cap - result[key])
            result[key] += room
            remaining_budget -= room
            remaining.remove(key)
    return result


def _validate_previous_weights(
    previous_weights: Mapping[str, Any] | None,
    contract: Mapping[str, Any],
) -> dict[str, float]:
    pool = contract["strategy_pool"]
    strategy_ids = {item["strategy_id"] for item in pool}
    values = {strategy_id: 0.0 for strategy_id in strategy_ids}
    if previous_weights is None:
        return values
    cash_asset_id = contract["allocation"]["cash_asset_id"]
    unknown = set(previous_weights).difference(strategy_ids | {cash_asset_id})
    if unknown:
        raise EvidenceError(f"前次权重包含冻结池之外的资产：{sorted(unknown)}")
    if cash_asset_id not in previous_weights:
        raise EvidenceError("前次权重必须显式包含现金权重")
    for strategy_id in strategy_ids:
        raw = previous_weights.get(strategy_id, 0.0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise EvidenceError(f"前次权重 {strategy_id} 必须是数值")
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise EvidenceError(f"前次权重 {strategy_id} 必须是非负有限数值")
        values[strategy_id] = value
    allocation = contract["allocation"]
    if sum(values.values()) > float(allocation["maximum_total_strategy_weight"]) + 1e-12:
        raise EvidenceError("前次策略总权重超过冻结上限")
    by_family: dict[str, float] = defaultdict(float)
    family_by_strategy = {item["strategy_id"]: item["family"] for item in pool}
    for strategy_id, value in values.items():
        if value > float(allocation["maximum_single_strategy_weight"]) + 1e-12:
            raise EvidenceError(f"前次权重 {strategy_id} 超过单策略上限")
        by_family[family_by_strategy[strategy_id]] += value
    if any(
        value > float(allocation["maximum_single_family_weight"]) + 1e-12
        for value in by_family.values()
    ):
        raise EvidenceError("前次权重超过单策略族上限")
    raw_cash = previous_weights[cash_asset_id]
    if isinstance(raw_cash, bool) or not isinstance(raw_cash, (int, float)):
        raise EvidenceError("前次现金权重必须是数值")
    cash_weight = float(raw_cash)
    if not math.isfinite(cash_weight) or cash_weight < 0.0:
        raise EvidenceError("前次现金权重必须是非负有限数值")
    if cash_weight + 1e-12 < float(allocation["minimum_cash_weight"]):
        raise EvidenceError("前次现金权重低于冻结最低现金权重")
    if not math.isclose(sum(values.values()) + cash_weight, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise EvidenceError("前次策略与现金权重合计不等于 1")
    return values


def allocate_research_risk(
    eligibility: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    previous_weights: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """对合格轨道分配研究风险预算；空集合机械回退现金。"""

    pool = contract["strategy_pool"]
    allocation = contract["allocation"]
    pool_by_id = {item["strategy_id"]: item for item in pool}
    eligibility_by_id = {item["strategy_id"]: item for item in eligibility}
    scores = {
        strategy_id: (
            float(eligibility_by_id[strategy_id]["score"])
            if strategy_id in eligibility_by_id and eligibility_by_id[strategy_id]["eligible"]
            else 0.0
        )
        for strategy_id in pool_by_id
    }
    family_scores: dict[str, float] = defaultdict(float)
    for strategy_id, score in scores.items():
        family_scores[pool_by_id[strategy_id]["family"]] += score
    family_weights = _capped_proportional(
        family_scores,
        budget=float(allocation["maximum_total_strategy_weight"]),
        cap=float(allocation["maximum_single_family_weight"]),
    )
    desired = {strategy_id: 0.0 for strategy_id in pool_by_id}
    for family, family_budget in family_weights.items():
        family_members = {
            strategy_id: score
            for strategy_id, score in scores.items()
            if pool_by_id[strategy_id]["family"] == family
        }
        within = _capped_proportional(
            family_members,
            budget=family_budget,
            cap=float(allocation["maximum_single_strategy_weight"]),
        )
        desired.update(within)

    previous = _validate_previous_weights(previous_weights, contract)
    final = {
        strategy_id: min(previous[strategy_id], desired[strategy_id])
        for strategy_id in desired
    }
    increases = {
        strategy_id: max(0.0, desired[strategy_id] - previous[strategy_id])
        for strategy_id in desired
    }
    requested_increase = sum(increases.values())
    maximum_increase = float(allocation["maximum_positive_risk_increase_per_decision"])
    increase_scale = 1.0 if requested_increase <= maximum_increase or requested_increase == 0.0 else maximum_increase / requested_increase
    for strategy_id, increase in increases.items():
        final[strategy_id] = previous[strategy_id] + increase * increase_scale
        if desired[strategy_id] < previous[strategy_id]:
            final[strategy_id] = desired[strategy_id]

    strategy_total = sum(final.values())
    cash_weight = max(0.0, 1.0 - strategy_total)
    minimum_cash = float(allocation["minimum_cash_weight"])
    if cash_weight + 1e-12 < minimum_cash:
        raise EvidenceError("风险配置违反最低现金权重")

    absolute_changes = {
        strategy_id: abs(final[strategy_id] - previous[strategy_id])
        for strategy_id in final
    }
    base_cost_fraction = sum(
        absolute_changes[strategy_id]
        * float(pool_by_id[strategy_id]["base_allocator_switch_cost_bps"])
        / 10_000.0
        for strategy_id in final
    )
    stress_cost_fraction = sum(
        absolute_changes[strategy_id]
        * float(pool_by_id[strategy_id]["stress_allocator_switch_cost_bps"])
        / 10_000.0
        for strategy_id in final
    )
    weights = {key: float(final[key]) for key in sorted(final)}
    weights[allocation["cash_asset_id"]] = float(cash_weight)
    desired_weights = {key: float(desired[key]) for key in sorted(desired)}
    desired_weights[allocation["cash_asset_id"]] = float(1.0 - sum(desired.values()))
    return {
        "method": "CAPPED_FAMILY_THEN_STRATEGY_STRESS_EDGE_OVER_VOLATILITY",
        "desired_weights_before_increase_cap": desired_weights,
        "research_risk_weights": weights,
        "strategy_weight_total": float(strategy_total),
        "cash_weight": float(cash_weight),
        "requested_positive_risk_increase": float(requested_increase),
        "applied_positive_risk_increase": float(sum(max(0.0, final[key] - previous[key]) for key in final)),
        "risk_increase_scale": float(increase_scale),
        "absolute_strategy_weight_turnover": float(sum(absolute_changes.values())),
        "estimated_base_allocator_switch_cost_fraction": float(base_cost_fraction),
        "estimated_stress_allocator_switch_cost_fraction": float(stress_cost_fraction),
        "negative_weights_used": False,
        "leverage_used": False,
        "cash_is_valid_state": True,
    }


def evaluate_allocator(
    contract: Mapping[str, Any],
    environment_records: Iterable[Mapping[str, Any]],
    strategy_records: Iterable[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    decision_at: datetime,
    previous_weights: Mapping[str, Any] | None = None,
    root: Path = ROOT,
) -> dict[str, Any]:
    """执行一次点时研究资格与风险配置决策。"""

    validate_contract(contract)
    if decision_at.tzinfo is None or decision_at.utcoffset() is None:
        raise EvidenceError("decision_at 必须包含时区")
    if manifest.get("candidate_id") != contract["protocol"]["candidate_id"]:
        raise ContractError("清单 candidate_id 与合同不一致")
    manifest_frozen_at = parse_zoned_datetime(manifest.get("frozen_at"), "manifest.frozen_at")
    validated_environment = validate_environment_records(environment_records, contract, root=root)
    validated_strategy = validate_strategy_records(
        strategy_records,
        contract,
        manifest_frozen_at=manifest_frozen_at,
        manifest_tracked_content_sha256=manifest.get("tracked_content_sha256"),
        root=root,
    )
    environment_result = classify_environment(
        validated_environment,
        contract,
        decision_at=decision_at,
    )

    before_forward_start = decision_at <= manifest_frozen_at
    eligibility: list[dict[str, Any]] = []
    for strategy in contract["strategy_pool"]:
        current_environment = environment_result.get("environment_id")
        record = None
        if current_environment is not None:
            record = _latest_matching_strategy_record(
                validated_strategy,
                strategy_id=strategy["strategy_id"],
                environment_id=current_environment,
                decision_at=decision_at,
            )
        result = evaluate_strategy_eligibility(
            strategy,
            record,
            environment_result,
            contract,
            manifest_frozen_at=manifest_frozen_at,
            decision_at=decision_at,
        )
        if before_forward_start:
            result["eligible"] = False
            result["score"] = 0.0
            result["gates"]["decision_strictly_after_manifest_freeze"] = False
            if "decision_strictly_after_manifest_freeze" not in result["failed_gates"]:
                result["failed_gates"].append("decision_strictly_after_manifest_freeze")
            result["reason"] = "决策时点尚未严格晚于清单冻结时点"
        else:
            result["gates"]["decision_strictly_after_manifest_freeze"] = True
        eligibility.append(result)

    eligible_count = sum(1 for item in eligibility if item["eligible"])
    if before_forward_start:
        status = "PROTOCOL_FROZEN_FORWARD_NOT_STARTED"
    elif environment_result["view_status"] != "VIEW":
        status = "NO_VIEW_CASH_ONLY"
    elif eligible_count == 0:
        status = "NO_ELIGIBLE_STRATEGY_CASH_ONLY"
    else:
        status = "RESEARCH_RISK_ALLOCATION_ELIGIBLE"

    allocation_result = allocate_research_risk(
        eligibility,
        contract,
        previous_weights=previous_weights,
    )
    return {
        "schema_version": "1.0.0",
        "candidate_id": contract["protocol"]["candidate_id"],
        "version": contract["protocol"]["version"],
        "decision_at": decision_at.isoformat(),
        "manifest_frozen_at": manifest_frozen_at.isoformat(),
        "status": status,
        "view_status": environment_result["view_status"],
        "evidence_label": contract["protocol"]["evidence_label"],
        "environment": environment_result,
        "eligible_strategy_count": int(eligible_count),
        "strategy_eligibility": eligibility,
        "allocation": allocation_result,
        "evidence_inventory": {
            "environment_record_count": int(len(validated_environment)),
            "strategy_record_count": int(len(validated_strategy)),
            "future_available_environment_records_ignored": int(
                sum(1 for item in validated_environment if item["available_at"] > decision_at)
            ),
            "future_available_strategy_records_ignored": int(
                sum(1 for item in validated_strategy if item["available_at"] > decision_at)
            ),
        },
        "governance": {
            **dict(contract["governance"]),
            "historical_returns_used_for_current_ranking": False,
            "winner_chasing_input_used": False,
            "component_rejection_relabelled": False,
            "cash_is_failure": False,
        },
        "safety": dict(contract["safety"]),
        "actionability": {
            "research_risk_budget_only": True,
            "paper_position": False,
            "shadow_signal": False,
            "position_mapping": False,
            "order": False,
            "broker_or_exchange_connection": False,
            "live_trading": False,
        },
    }


def audit_component_statuses(
    contract: Mapping[str, Any],
    dependency_snapshot: Mapping[str, Mapping[str, Any]],
    *,
    root: Path = ROOT,
) -> list[dict[str, Any]]:
    """只提取组件身份、终态和授权边界，不读取其收益用于排序。"""

    result: list[dict[str, Any]] = []
    for strategy in contract["strategy_pool"]:
        strategy_id = strategy["strategy_id"]
        source_path = strategy["component_status_source"]
        manifest_path = strategy["component_manifest"]
        source_record = dependency_snapshot.get(source_path)
        manifest_record = dependency_snapshot.get(manifest_path)
        if source_record is None or manifest_record is None:
            raise ContractError(f"清单缺少组件依赖快照：{strategy_id}")
        source = root / PurePosixPath(source_path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
        authorization = decision.get(
            "paper_or_live_authorized",
            payload.get("paper_or_live_authorized", False),
        )
        result.append(
            {
                "strategy_id": strategy_id,
                "family": strategy["family"],
                "component_status": payload.get("status", "UNKNOWN"),
                "component_status_source": source_path,
                "component_status_sha256": source_record["sha256"],
                "component_manifest": manifest_path,
                "component_manifest_sha256": manifest_record["sha256"],
                "paper_or_live_authorized": bool(authorization),
                "status_relabelled_by_allocator": False,
                "component_performance_used_for_ranking": False,
            }
        )
    return result


def _format_percentage(value: Any) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2%}"


def render_markdown(report: Mapping[str, Any]) -> str:
    """渲染中文、可审计且不含交易指令的最新状态报告。"""

    environment = report["environment"]
    allocation = report["allocation"]
    lines = [
        "# 市场状态感知策略资格与风险配置器 V1 状态",
        "",
        f"- 决策时点：`{report['decision_at']}`",
        f"- 清单冻结时点：`{report['manifest_frozen_at']}`",
        f"- 主状态：`{report['status']}`",
        f"- 研究视图：`{report['view_status']}`",
        f"- 当前环境：`{environment.get('environment_id') or 'NO_VIEW'}`",
        f"- 合格策略数：{report['eligible_strategy_count']}",
        f"- 现金权重：{_format_percentage(allocation['cash_weight'])}",
        "- 研究边界：只输出资格与风险预算；不生成 Paper、Shadow、仓位、订单、账户连接或实盘动作。",
        "",
        "## 当前环境证据",
        "",
        f"- 环境证据 ID：`{environment.get('evidence_id') or 'MISSING'}`",
        f"- 可得时间：`{environment.get('available_at') or 'MISSING'}`",
        f"- 判定原因：{environment['reason']}",
        "",
        "## 组件冻结状态",
        "",
        "| 策略 | 组件状态 | 原组件 Paper/实盘授权 | 是否被元策略改写 |",
        "|---|---|---:|---:|",
    ]
    for item in report.get("component_statuses", []):
        lines.append(
            "| {strategy_id} | `{component_status}` | {authorized} | {relabelled} |".format(
                strategy_id=item["strategy_id"],
                component_status=item["component_status"],
                authorized="是" if item["paper_or_live_authorized"] else "否",
                relabelled="是" if item["status_relabelled_by_allocator"] else "否",
            )
        )
    lines.extend(
        [
            "",
            "## 策略资格",
            "",
            "| 策略 | 当前环境是否允许 | 条件证据 | 通过门数/总门数 | 合格 | 研究权重 |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    weights = allocation["research_risk_weights"]
    for item in report["strategy_eligibility"]:
        gate_count = len(item["gates"])
        passed_count = sum(1 for value in item["gates"].values() if value)
        lines.append(
            "| {strategy_id} | {environment} | `{evidence}` | {passed}/{total} | {eligible} | {weight} |".format(
                strategy_id=item["strategy_id"],
                environment="是" if item["gates"].get("environment_is_permitted", False) else "否",
                evidence=item.get("evidence_id") or "MISSING",
                passed=passed_count,
                total=gate_count,
                eligible="是" if item["eligible"] else "否",
                weight=_format_percentage(weights[item["strategy_id"]]),
            )
        )
    lines.extend(
        [
            "",
            "## 成本后风险预算",
            "",
            f"- 策略总权重：{_format_percentage(allocation['strategy_weight_total'])}",
            f"- 现金权重：{_format_percentage(allocation['cash_weight'])}",
            f"- 绝对策略权重换手：{_format_percentage(allocation['absolute_strategy_weight_turnover'])}",
            f"- 基础分配切换成本估计：{_format_percentage(allocation['estimated_base_allocator_switch_cost_fraction'])}",
            f"- 压力分配切换成本估计：{_format_percentage(allocation['estimated_stress_allocator_switch_cost_fraction'])}",
            "",
            "## 治理核对",
            "",
            f"- 使用最近赢家排名：`{str(report['governance']['winner_chasing_input_used']).lower()}`",
            f"- 使用历史收益决定当前排名：`{str(report['governance']['historical_returns_used_for_current_ranking']).lower()}`",
            f"- 改写组件拒绝状态：`{str(report['governance']['component_rejection_relabelled']).lower()}`",
            f"- 将现金视为失败：`{str(report['governance']['cash_is_failure']).lower()}`",
            "",
            "本报告不是交易建议或交易授权。缺少合格证据时，100% 现金是系统的正确输出。",
            "",
        ]
    )
    return "\n".join(lines)
