"""从真正前向观察账本重算 V1 条件优势证据。"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo

from research.regime_aware_strategy_allocator_v1 import (
    EXPECTED_ENVIRONMENTS,
    EXPECTED_STRATEGY_IDS,
    EvidenceError,
    classify_environment,
    parse_zoned_datetime,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]

FORWARD_DATA_STATUSES = {
    "PASS",
    "NO_VIEW",
    "RUNNING",
    "PARTIAL_SUCCESS",
    "FAILED",
    "PROGRAM_FAILED",
    "SKIPPED",
    "CENSORED_NO_OUTCOME",
}

OBSERVATION_REQUIRED_FIELDS = {
    "schema_version",
    "observation_id",
    "strategy_id",
    "environment_id",
    "environment_evidence_id",
    "environment_episode_id",
    "signal_at",
    "outcome_at",
    "available_at",
    "data_status",
    "quality_complete",
    "forward_observed",
    "strategy_gross_return",
    "strategy_base_net_return",
    "strategy_stress_net_return",
    "cash_net_return",
    "static_equal_risk_stress_net_return",
    "benchmark_set_complete",
    "benchmarks",
    "cost_model_complete",
    "included_cost_components",
    "allocator_switch_event",
    "base_costs_fraction_by_component",
    "stress_costs_fraction_by_component",
    "source_path",
    "source_sha256",
    "paper_position_generation",
    "shadow_signal_generation",
    "order_generation",
    "live_trading_authorized",
}


def _safe_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise EvidenceError("观察来源路径必须是非空项目相对路径")
    if "\\" in value:
        raise EvidenceError(f"观察来源路径必须使用正斜杠：{value}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or ":" in value:
        raise EvidenceError(f"观察来源路径越出项目根目录：{value}")
    return pure.as_posix()


def _validate_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise EvidenceError(f"{label} 必须是 64 位小写 SHA-256")
    if any(character not in "0123456789abcdef" for character in value):
        raise EvidenceError(f"{label} 必须是 64 位小写 SHA-256")
    return value


def _finite_return(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} 必须是有限收益数值")
    number = float(value)
    if not math.isfinite(number) or number <= -1.0:
        raise EvidenceError(f"{label} 必须有限且严格大于 -1")
    return number


def _finite_cost_fraction(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} 必须是有限非负成本比例")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or number >= 1.0:
        raise EvidenceError(f"{label} 必须有限且位于 [0, 1) 区间")
    return number


def derive_environment_episode_map(
    environment_records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    manifest_frozen_at: datetime,
) -> dict[str, dict[str, str]]:
    """用冻结分类器从追加式环境证据派生不可手填的连续环境片段。"""

    result: dict[str, dict[str, str]] = {}
    previous_environment: str | None = None
    previous_observation_at: datetime | None = None
    current_episode_id: str | None = None
    minimum_separation_seconds = float(
        contract["environment"]["minimum_same_environment_episode_separation_hours"]
    ) * 3600.0
    for record in environment_records:
        if record["available_at"] <= manifest_frozen_at:
            previous_environment = None
            previous_observation_at = None
            current_episode_id = None
            continue
        classification = classify_environment(
            [record],
            contract,
            decision_at=record["available_at"],
        )
        if classification["view_status"] != "VIEW":
            continue
        environment_id = str(classification["environment_id"])
        gap_seconds = (
            None
            if previous_observation_at is None
            else (record["observation_at"] - previous_observation_at).total_seconds()
        )
        starts_new_episode = bool(
            previous_environment != environment_id
            or gap_seconds is None
            or gap_seconds > minimum_separation_seconds
        )
        if starts_new_episode:
            current_episode_id = f"{environment_id}::{record['evidence_id']}"
        if current_episode_id is None:
            raise EvidenceError("环境片段派生状态不完整")
        result[str(record["evidence_id"])] = {
            "environment_id": environment_id,
            "environment_episode_id": current_episode_id,
        }
        previous_environment = environment_id
        previous_observation_at = record["observation_at"]
    return result


def validate_forward_observations(
    records: Iterable[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    manifest_frozen_at: datetime,
    environment_records: Sequence[Mapping[str, Any]],
    root: Path = ROOT,
) -> list[dict[str, Any]]:
    """验证逐观察日真正前向账本、成本、基准、来源和安全边界。"""

    pool = {item["strategy_id"]: item for item in contract["strategy_pool"]}
    required_benchmarks = set(contract["evidence_gates"]["required_benchmarks"])
    forbidden = set(contract["evidence_gates"]["forbidden_selection_fields"])
    allowed_data_statuses = set(contract["forward_evaluation"]["allowed_observation_statuses"])
    if allowed_data_statuses != FORWARD_DATA_STATUSES:
        raise EvidenceError("前向观察状态集合与冻结评价器实现不一致")
    environment_episode_map = derive_environment_episode_map(
        environment_records,
        contract,
        manifest_frozen_at=manifest_frozen_at,
    )
    environment_by_id = {
        str(record["evidence_id"]): record for record in environment_records
    }
    normalized: list[dict[str, Any]] = []
    observation_ids: set[str] = set()
    group_outcome_days: set[tuple[str, str, Any]] = set()
    latest_outcome_by_group: dict[tuple[str, str], datetime] = {}
    previous_available: datetime | None = None

    for ordinal, raw in enumerate(records, start=1):
        line = raw.get("_line_number", ordinal)
        label = f"前向观察第 {line} 行"
        fields = set(raw).difference({"_line_number"})
        forbidden_present = sorted(fields.intersection(forbidden))
        if forbidden_present:
            raise EvidenceError(f"{label} 含禁止的赢家追逐字段：{forbidden_present}")
        if fields != OBSERVATION_REQUIRED_FIELDS:
            missing = sorted(OBSERVATION_REQUIRED_FIELDS - fields)
            extra = sorted(fields - OBSERVATION_REQUIRED_FIELDS)
            raise EvidenceError(f"{label} 字段不匹配：missing={missing}，extra={extra}")
        if raw["schema_version"] != "1.0.0":
            raise EvidenceError(f"{label} schema_version 必须为 1.0.0")
        observation_id = raw["observation_id"]
        if not isinstance(observation_id, str) or not observation_id:
            raise EvidenceError(f"{label} observation_id 无效")
        if observation_id in observation_ids:
            raise EvidenceError(f"前向 observation_id 重复：{observation_id}")
        observation_ids.add(observation_id)
        strategy_id = raw["strategy_id"]
        environment_id = raw["environment_id"]
        if strategy_id not in EXPECTED_STRATEGY_IDS or strategy_id not in pool:
            raise EvidenceError(f"{label} 策略不在冻结池：{strategy_id!r}")
        if environment_id not in EXPECTED_ENVIRONMENTS:
            raise EvidenceError(f"{label} 环境不在冻结分类：{environment_id!r}")
        if environment_id not in pool[strategy_id]["permitted_environments"]:
            raise EvidenceError(f"{label} 策略与环境不属于冻结组合")
        for name in ("environment_evidence_id", "environment_episode_id"):
            if not isinstance(raw[name], str) or not raw[name]:
                raise EvidenceError(f"{label}.{name} 必须是非空字符串")

        signal_at = parse_zoned_datetime(raw["signal_at"], f"{label}.signal_at")
        outcome_at = parse_zoned_datetime(raw["outcome_at"], f"{label}.outcome_at")
        available_at = parse_zoned_datetime(raw["available_at"], f"{label}.available_at")
        if signal_at <= manifest_frozen_at:
            raise EvidenceError(f"{label} signal_at 未严格晚于清单冻结时点")
        if not signal_at < outcome_at <= available_at:
            raise EvidenceError(f"{label} 必须满足 signal_at < outcome_at <= available_at")
        required_horizon = timedelta(
            hours=float(contract["forward_evaluation"]["observation_return_horizon_hours"])
        )
        if outcome_at - signal_at != required_horizon:
            raise EvidenceError(f"{label} 收益观察周期必须精确为冻结的 24 小时")
        maximum_lag = timedelta(
            hours=float(
                contract["forward_evaluation"]["maximum_outcome_availability_lag_hours"]
            )
        )
        if available_at - outcome_at > maximum_lag:
            raise EvidenceError(f"{label} 结果可得延迟超过冻结的 24 小时")
        group_key = (strategy_id, environment_id)
        previous_group_outcome = latest_outcome_by_group.get(group_key)
        if previous_group_outcome is not None and outcome_at <= previous_group_outcome:
            raise EvidenceError(f"{label} 同一策略与环境不得晚到回填或倒序追加结果")
        latest_outcome_by_group[group_key] = outcome_at
        point_in_time_environment = classify_environment(
            environment_records,
            contract,
            decision_at=signal_at,
        )
        if point_in_time_environment["view_status"] != "VIEW":
            raise EvidenceError(f"{label} signal_at 没有新鲜且通过的点时环境证据")
        if point_in_time_environment["evidence_id"] != raw["environment_evidence_id"]:
            raise EvidenceError(f"{label} 未绑定 signal_at 最新可得环境证据")
        environment_evidence = environment_by_id.get(str(raw["environment_evidence_id"]))
        if environment_evidence is None:
            raise EvidenceError(f"{label} 引用了不存在的环境 evidence_id")
        if (
            environment_evidence["observation_at"] <= manifest_frozen_at
            or environment_evidence["available_at"] <= manifest_frozen_at
        ):
            raise EvidenceError(f"{label} 使用了清单冻结前的环境证据")
        derived_environment = environment_episode_map.get(str(raw["environment_evidence_id"]))
        if derived_environment is None:
            raise EvidenceError(f"{label} 环境证据不能派生有效环境片段")
        if derived_environment["environment_id"] != environment_id:
            raise EvidenceError(f"{label} 环境标签与冻结分类器不一致")
        if derived_environment["environment_episode_id"] != raw["environment_episode_id"]:
            raise EvidenceError(f"{label} 环境片段 ID 不是由冻结环境序列派生")
        if previous_available is not None and available_at < previous_available:
            raise EvidenceError("前向观察账本未按 available_at 追加排序")
        previous_available = available_at
        observation_day = outcome_at.astimezone(
            ZoneInfo(contract["forward_evaluation"]["observation_day_timezone"])
        ).date()
        group_outcome_day = (strategy_id, environment_id, observation_day)
        if group_outcome_day in group_outcome_days:
            raise EvidenceError(f"{label} 同一策略、环境和上海观察日重复")
        group_outcome_days.add(group_outcome_day)

        data_status = raw["data_status"]
        if data_status not in allowed_data_statuses:
            raise EvidenceError(f"{label} data_status 无效：{data_status!r}")
        for name in (
            "quality_complete",
            "forward_observed",
            "benchmark_set_complete",
            "cost_model_complete",
            "allocator_switch_event",
        ):
            if not isinstance(raw[name], bool):
                raise EvidenceError(f"{label}.{name} 必须是布尔值")
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
        required_costs = set(pool[strategy_id]["required_cost_components"])
        if data_status == "PASS":
            completion_fields = (
                "quality_complete",
                "forward_observed",
                "benchmark_set_complete",
                "cost_model_complete",
            )
            incomplete = [name for name in completion_fields if raw[name] is not True]
            if incomplete:
                raise EvidenceError(f"{label} PASS 时完整性字段必须全部为 true：{incomplete}")
            if set(raw["benchmarks"]) != required_benchmarks:
                raise EvidenceError(f"{label} PASS 时必须且只能包含冻结基准集合")
            if set(raw["included_cost_components"]) != required_costs:
                raise EvidenceError(f"{label} PASS 时必须且只能包含冻结成本项集合")
        else:
            completion_fields = (
                "quality_complete",
                "forward_observed",
                "benchmark_set_complete",
                "cost_model_complete",
                "allocator_switch_event",
            )
            incorrectly_true = [name for name in completion_fields if raw[name] is not False]
            if incorrectly_true:
                raise EvidenceError(
                    f"{label} 非 PASS 状态不得声称完整、已观察或切换：{incorrectly_true}"
                )
            if raw["benchmarks"] or raw["included_cost_components"]:
                raise EvidenceError(f"{label} 非 PASS 状态的基准和成本项必须为空数组")
        for safety_name in (
            "paper_position_generation",
            "shadow_signal_generation",
            "order_generation",
            "live_trading_authorized",
        ):
            if raw[safety_name] is not False:
                raise EvidenceError(f"{label}.{safety_name} 必须为 false")

        relative = _safe_relative_path(raw["source_path"])
        expected_sha = _validate_sha256(raw["source_sha256"], f"{label}.source_sha256")
        source = root / PurePosixPath(relative)
        if not source.is_file():
            raise EvidenceError(f"{label} 来源文件不存在：{relative}")
        actual_sha = sha256_file(source)
        if actual_sha != expected_sha:
            raise EvidenceError(
                f"{label} 来源 SHA-256 不一致：{relative}，expected={expected_sha}，actual={actual_sha}"
            )

        returns: dict[str, float | None] = {}
        for name in (
            "strategy_gross_return",
            "strategy_base_net_return",
            "strategy_stress_net_return",
            "cash_net_return",
            "static_equal_risk_stress_net_return",
        ):
            value = raw[name]
            if data_status == "PASS":
                returns[name] = _finite_return(value, f"{label}.{name}")
            elif value is not None:
                raise EvidenceError(f"{label} NO_VIEW 时 {name} 必须为 null")
            else:
                returns[name] = None

        normalized_base_costs: dict[str, float] = {}
        normalized_stress_costs: dict[str, float] = {}
        base_costs = raw["base_costs_fraction_by_component"]
        stress_costs = raw["stress_costs_fraction_by_component"]
        if not isinstance(base_costs, dict) or not isinstance(stress_costs, dict):
            raise EvidenceError(f"{label} 成本明细必须是 JSON 对象")
        if data_status == "PASS":
            if set(base_costs) != required_costs or set(stress_costs) != required_costs:
                raise EvidenceError(f"{label} 成本明细键必须与冻结成本项集合完全一致")
            tolerance = float(
                contract["forward_evaluation"]["cost_reconciliation"]["numeric_tolerance"]
            )
            for component in sorted(required_costs):
                base_value = _finite_cost_fraction(
                    base_costs[component], f"{label}.base_costs_fraction_by_component.{component}"
                )
                stress_value = _finite_cost_fraction(
                    stress_costs[component],
                    f"{label}.stress_costs_fraction_by_component.{component}",
                )
                if stress_value + tolerance < base_value:
                    raise EvidenceError(f"{label} 压力成本不得低于基础成本：{component}")
                normalized_base_costs[component] = base_value
                normalized_stress_costs[component] = stress_value
            switch_multiplier = 1.0 if raw["allocator_switch_event"] else 0.0
            expected_base_switch = (
                float(pool[strategy_id]["base_allocator_switch_cost_bps"])
                / 10_000.0
                * switch_multiplier
            )
            expected_stress_switch = (
                float(pool[strategy_id]["stress_allocator_switch_cost_bps"])
                / 10_000.0
                * switch_multiplier
            )
            if not math.isclose(
                normalized_base_costs["allocator_switching"],
                expected_base_switch,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise EvidenceError(f"{label} 基础分配切换成本未使用冻结策略池费率")
            if not math.isclose(
                normalized_stress_costs["allocator_switching"],
                expected_stress_switch,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise EvidenceError(f"{label} 压力分配切换成本未使用冻结策略池费率")
            gross = float(returns["strategy_gross_return"])
            expected_base_net = gross - sum(normalized_base_costs.values())
            expected_stress_net = gross - sum(normalized_stress_costs.values())
            if not math.isclose(
                float(returns["strategy_base_net_return"]),
                expected_base_net,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise EvidenceError(f"{label} 基础净收益无法由毛收益减逐项成本复算")
            if not math.isclose(
                float(returns["strategy_stress_net_return"]),
                expected_stress_net,
                rel_tol=0.0,
                abs_tol=tolerance,
            ):
                raise EvidenceError(f"{label} 压力净收益无法由毛收益减逐项成本复算")
        elif base_costs or stress_costs:
            raise EvidenceError(f"{label} 非 PASS 状态的成本明细必须为空对象")

        normalized.append(
            {
                **{key: value for key, value in raw.items() if key != "_line_number"},
                "signal_at": signal_at,
                "outcome_at": outcome_at,
                "available_at": available_at,
                "observation_day": observation_day,
                **returns,
                "base_costs_fraction_by_component": normalized_base_costs,
                "stress_costs_fraction_by_component": normalized_stress_costs,
                "source_binding": {
                    "path": relative,
                    "sha256": expected_sha,
                    "verified": True,
                },
                "_line_number": line,
            }
        )
    return normalized


def geometric_annualized_return(values: Sequence[float], periods_per_year: int) -> float:
    if not values:
        raise ValueError("年化收益至少需要一个观察")
    log_growth = 0.0
    for value in values:
        if not math.isfinite(value) or value <= -1.0:
            raise ValueError("年化收益输入必须有限且严格大于 -1")
        log_growth += math.log1p(value)
    return math.expm1(log_growth * float(periods_per_year) / float(len(values)))


def _circular_block_indices(
    count: int,
    block_length: int,
    rng: random.Random,
) -> list[int]:
    result: list[int] = []
    while len(result) < count:
        start = rng.randrange(count)
        result.extend((start + offset) % count for offset in range(block_length))
    return result[:count]


def paired_annualized_edge_lower_bound(
    strategy_returns: Sequence[float],
    benchmark_returns: Sequence[float],
    *,
    periods_per_year: int,
    repetitions: int,
    block_length: int,
    lower_quantile: float,
    random_seed: int,
) -> dict[str, Any]:
    """配对循环移动区块 Bootstrap 的年化净优势下界。"""

    if len(strategy_returns) != len(benchmark_returns):
        raise ValueError("策略和基准收益长度必须一致")
    count = len(strategy_returns)
    if count < 2:
        return {
            "status": "INSUFFICIENT_OBSERVATIONS",
            "point_estimate": None,
            "lower_bound": None,
            "repetitions": repetitions,
            "block_length_observations": block_length,
            "lower_quantile": lower_quantile,
        }
    point = geometric_annualized_return(strategy_returns, periods_per_year) - geometric_annualized_return(
        benchmark_returns, periods_per_year
    )
    rng = random.Random(random_seed)
    samples: list[float] = []
    for _ in range(repetitions):
        indices = _circular_block_indices(count, block_length, rng)
        strategy_sample = [strategy_returns[index] for index in indices]
        benchmark_sample = [benchmark_returns[index] for index in indices]
        samples.append(
            geometric_annualized_return(strategy_sample, periods_per_year)
            - geometric_annualized_return(benchmark_sample, periods_per_year)
        )
    samples.sort()
    quantile_index = min(
        len(samples) - 1,
        max(0, int(math.floor(lower_quantile * (len(samples) - 1)))),
    )
    return {
        "status": "COMPLETE",
        "point_estimate": float(point),
        "lower_bound": float(samples[quantile_index]),
        "repetitions": int(repetitions),
        "block_length_observations": int(block_length),
        "lower_quantile": float(lower_quantile),
        "random_seed": int(random_seed),
    }


def _sample_sharpe(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("-inf")
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values)
    if deviation == 0.0:
        if mean > 0.0:
            return float("inf")
        if mean < 0.0:
            return float("-inf")
        return 0.0
    return mean / deviation


def deflated_sharpe_probability(
    values: Sequence[float],
    *,
    independent_trials: int,
    euler_mascheroni: float,
) -> dict[str, Any]:
    """按偏度、峰度和冻结试验数计算 Deflated Sharpe 概率。"""

    count = len(values)
    if count < 3 or independent_trials < 2:
        return {"status": "INSUFFICIENT_OBSERVATIONS", "probability": None}
    mean = statistics.fmean(values)
    deviation = statistics.stdev(values)
    if deviation == 0.0:
        probability = 1.0 if mean > 0.0 else 0.0
        return {
            "status": "COMPLETE_DEGENERATE_VARIANCE",
            "probability": probability,
            "observed_sharpe_per_period": math.inf if mean > 0.0 else 0.0,
            "expected_maximum_sharpe_per_period": 0.0,
            "independent_trials": int(independent_trials),
        }
    centered = [value - mean for value in values]
    second = statistics.fmean(value**2 for value in centered)
    third = statistics.fmean(value**3 for value in centered)
    fourth = statistics.fmean(value**4 for value in centered)
    skewness = third / (second ** 1.5) if second > 0.0 else 0.0
    kurtosis = fourth / (second**2) if second > 0.0 else 3.0
    observed_sharpe = mean / deviation
    variance_numerator = max(
        1e-15,
        1.0
        - skewness * observed_sharpe
        + ((kurtosis - 1.0) / 4.0) * observed_sharpe**2,
    )
    sharpe_standard_error = math.sqrt(variance_numerator / float(count - 1))
    normal = statistics.NormalDist()
    first_quantile = normal.inv_cdf(1.0 - 1.0 / float(independent_trials))
    second_quantile = normal.inv_cdf(
        1.0 - 1.0 / (float(independent_trials) * math.e)
    )
    expected_maximum = sharpe_standard_error * (
        (1.0 - euler_mascheroni) * first_quantile
        + euler_mascheroni * second_quantile
    )
    z_score = (observed_sharpe - expected_maximum) / sharpe_standard_error
    probability = normal.cdf(z_score)
    return {
        "status": "COMPLETE",
        "probability": float(probability),
        "observed_sharpe_per_period": float(observed_sharpe),
        "expected_maximum_sharpe_per_period": float(expected_maximum),
        "sharpe_standard_error": float(sharpe_standard_error),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "independent_trials": int(independent_trials),
    }


def _contiguous_partitions(count: int, partition_count: int) -> list[list[int]]:
    base, remainder = divmod(count, partition_count)
    result: list[list[int]] = []
    start = 0
    for partition in range(partition_count):
        size = base + (1 if partition < remainder else 0)
        result.append(list(range(start, start + size)))
        start += size
    return result


def probability_backtest_overfit(
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    environment_id: str,
    decision_at: datetime,
) -> dict[str, Any]:
    """按冻结连续分区执行组合对称交叉验证 PBO。"""

    pbo_contract = contract["forward_evaluation"]["probability_backtest_overfit"]
    partition_count = int(pbo_contract["partition_count"])
    training_partition_count = int(pbo_contract["training_partition_count"])
    minimum_days = int(pbo_contract["minimum_complete_observation_days"])
    baseline_candidate_id = str(pbo_contract["baseline_candidate_id"])
    candidate_ids = [
        item["strategy_id"]
        for item in contract["strategy_pool"]
        if environment_id in item["permitted_environments"]
    ]
    series_by_strategy: dict[str, dict[Any, float]] = {
        strategy_id: {} for strategy_id in candidate_ids
    }
    for record in records:
        if (
            record["strategy_id"] in series_by_strategy
            and record["environment_id"] == environment_id
            and record["available_at"] <= decision_at
            and record["data_status"] == "PASS"
            and record["quality_complete"]
            and record["forward_observed"]
        ):
            series_by_strategy[record["strategy_id"]][record["observation_day"]] = float(
                record["strategy_stress_net_return"]
                - record["static_equal_risk_stress_net_return"]
            )
    missing_candidates = sorted(
        strategy_id for strategy_id, values in series_by_strategy.items() if not values
    )
    if missing_candidates:
        return {
            "status": "INCOMPLETE_FROZEN_CANDIDATE_UNIVERSE",
            "probability": None,
            "candidate_count": int(len(candidate_ids) + 1),
            "complete_day_count": 0,
            "baseline_candidate_id": baseline_candidate_id,
            "missing_strategy_candidates": missing_candidates,
        }
    common_dates = set.intersection(*(set(values) for values in series_by_strategy.values()))
    dates = sorted(common_dates)
    if len(dates) < minimum_days or len(dates) < partition_count:
        return {
            "status": "INSUFFICIENT_COMPLETE_DAYS",
            "probability": None,
            "candidate_count": int(len(candidate_ids) + 1),
            "complete_day_count": int(len(dates)),
            "baseline_candidate_id": baseline_candidate_id,
            "missing_strategy_candidates": [],
        }
    rolling_window_days = int(
        contract["forward_evaluation"]["rolling_qualification_window_observation_days"]
    )
    dates = dates[-rolling_window_days:]
    candidates = sorted([*candidate_ids, baseline_candidate_id])
    matrix = {
        strategy_id: [series_by_strategy[strategy_id][date] for date in dates]
        for strategy_id in candidate_ids
    }
    matrix[baseline_candidate_id] = [0.0 for _ in dates]
    partitions = _contiguous_partitions(len(dates), partition_count)
    combinations = list(itertools.combinations(range(partition_count), training_partition_count))
    overfit_count = 0
    selections: dict[str, int] = defaultdict(int)
    for training_partitions in combinations:
        training_set = set(training_partitions)
        train_indices = [
            index
            for partition_id, partition in enumerate(partitions)
            if partition_id in training_set
            for index in partition
        ]
        test_indices = [
            index
            for partition_id, partition in enumerate(partitions)
            if partition_id not in training_set
            for index in partition
        ]
        train_sharpes = {
            strategy_id: _sample_sharpe([matrix[strategy_id][index] for index in train_indices])
            for strategy_id in candidates
        }
        selected = max(candidates, key=lambda strategy_id: (train_sharpes[strategy_id], strategy_id))
        selections[selected] += 1
        test_sharpes = {
            strategy_id: _sample_sharpe([matrix[strategy_id][index] for index in test_indices])
            for strategy_id in candidates
        }
        ordered = sorted(candidates, key=lambda strategy_id: (test_sharpes[strategy_id], strategy_id))
        rank_from_worst = ordered.index(selected) + 1
        relative_rank = rank_from_worst / float(len(candidates) + 1)
        if relative_rank <= 0.5:
            overfit_count += 1
    probability = overfit_count / float(len(combinations))
    return {
        "status": "COMPLETE",
        "probability": float(probability),
        "candidate_count": int(len(candidates)),
        "complete_day_count": int(len(dates)),
        "baseline_candidate_id": baseline_candidate_id,
        "missing_strategy_candidates": [],
        "partition_count": partition_count,
        "training_partition_count": training_partition_count,
        "combination_count": int(len(combinations)),
        "overfit_combination_count": int(overfit_count),
        "training_selection_counts": dict(sorted(selections.items())),
    }


def maximum_environment_profit_concentration(
    records: Sequence[Mapping[str, Any]],
) -> float:
    """计算单一环境片段占全部正增量利润的最大比例。"""

    by_episode: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_episode[record["environment_episode_id"]].append(
            float(
                record["strategy_stress_net_return"]
                - record["static_equal_risk_stress_net_return"]
            )
        )
    positive: list[float] = []
    for returns in by_episode.values():
        growth = 1.0
        for value in returns:
            growth *= 1.0 + value
        profit = growth - 1.0
        if profit > 0.0:
            positive.append(profit)
    if not positive:
        return 1.0
    return float(max(positive) / sum(positive))


def _previous_confirmation_state(
    previous_evidence: Iterable[Mapping[str, Any]],
    *,
    strategy_id: str,
    environment_id: str,
    decision_at: datetime,
    pending_status: str,
    passed_status: str,
) -> dict[str, Any]:
    def as_datetime(value: Any, label: str) -> datetime:
        if isinstance(value, datetime):
            if value.tzinfo is None or value.utcoffset() is None:
                raise EvidenceError(f"{label} 必须包含时区")
            return value
        return parse_zoned_datetime(value, label)

    matching = [
        record
        for record in previous_evidence
        if record.get("strategy_id") == strategy_id
        and record.get("environment_id") == environment_id
        and record.get("available_at") is not None
        and as_datetime(record["available_at"], "previous_evidence.available_at") <= decision_at
    ]
    if not matching:
        return {
            "status": None,
            "count": 0,
            "available_at": None,
            "evaluation_window_end": None,
            "forward_observation_days": 0,
        }
    matching.sort(
        key=lambda record: as_datetime(
            record["available_at"], "previous_evidence.available_at"
        )
    )
    latest = matching[-1]
    latest_available = as_datetime(
        latest["available_at"], "previous_evidence.available_at"
    )
    latest_window_end = (
        as_datetime(
            latest["evaluation_window_end"], "previous_evidence.evaluation_window_end"
        )
        if latest.get("evaluation_window_end") is not None
        else None
    )
    latest_days = latest.get("forward_observation_days")
    forward_days = latest_days if isinstance(latest_days, int) and latest_days >= 0 else 0
    if latest.get("meta_evidence_status") not in {pending_status, passed_status}:
        return {
            "status": latest.get("meta_evidence_status"),
            "count": 0,
            "available_at": latest_available,
            "evaluation_window_end": latest_window_end,
            "forward_observation_days": forward_days,
        }
    value = latest.get("consecutive_entry_confirmations")
    return {
        "status": latest.get("meta_evidence_status"),
        "count": value if isinstance(value, int) and value >= 0 else 0,
        "available_at": latest_available,
        "evaluation_window_end": latest_window_end,
        "forward_observation_days": forward_days,
    }


def evaluate_strategy_environment_forward_evidence(
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    *,
    strategy_id: str,
    environment_id: str,
    decision_at: datetime,
    previous_evidence: Iterable[Mapping[str, Any]] = (),
    bootstrap_repetitions_override: int | None = None,
) -> dict[str, Any]:
    """重算单一策略在单一环境下的全部资格统计与确认状态。"""

    evaluation = contract["forward_evaluation"]
    bootstrap = evaluation["bootstrap"]
    gates = contract["evidence_gates"]
    group_records = [
        record
        for record in records
        if record["strategy_id"] == strategy_id
        and record["environment_id"] == environment_id
        and record["available_at"] <= decision_at
    ]
    group_records.sort(key=lambda record: (record["available_at"], record["observation_id"]))
    latest_group_record = group_records[-1] if group_records else None
    latest_forward_status_pass = bool(
        latest_group_record is not None
        and latest_group_record["data_status"] == "PASS"
        and latest_group_record["quality_complete"]
        and latest_group_record["forward_observed"]
    )
    complete_records = [
        record
        for record in group_records
        if record["data_status"] == "PASS"
        and record["quality_complete"]
        and record["forward_observed"]
    ]
    complete_records.sort(key=lambda record: (record["outcome_at"], record["observation_id"]))
    cumulative_count = len(complete_records)
    rolling_window_days = int(evaluation["rolling_qualification_window_observation_days"])
    recent_window_days = int(evaluation["recent_health_window_observation_days"])
    qualification_records = complete_records[-rolling_window_days:]
    recent_records = complete_records[-recent_window_days:]
    qualification_count = len(qualification_records)
    recent_count = len(recent_records)
    episode_count = len(
        {record["environment_episode_id"] for record in qualification_records}
    )
    repetitions = (
        int(bootstrap_repetitions_override)
        if bootstrap_repetitions_override is not None
        else int(bootstrap["repetitions"])
    )
    periods = int(evaluation["annualization_periods"])

    base = [float(record["strategy_base_net_return"]) for record in qualification_records]
    stress = [float(record["strategy_stress_net_return"]) for record in qualification_records]
    cash = [float(record["cash_net_return"]) for record in qualification_records]
    static = [
        float(record["static_equal_risk_stress_net_return"])
        for record in qualification_records
    ]
    recent_stress = [
        float(record["strategy_stress_net_return"]) for record in recent_records
    ]
    recent_cash = [float(record["cash_net_return"]) for record in recent_records]
    recent_static = [
        float(record["static_equal_risk_stress_net_return"]) for record in recent_records
    ]
    recent_stress_edge = (
        geometric_annualized_return(recent_stress, periods)
        - geometric_annualized_return(recent_cash, periods)
        if recent_count >= recent_window_days
        else None
    )
    recent_incremental_edge = (
        geometric_annualized_return(recent_stress, periods)
        - geometric_annualized_return(recent_static, periods)
        if recent_count >= recent_window_days
        else None
    )
    base_edge = paired_annualized_edge_lower_bound(
        base,
        cash,
        periods_per_year=periods,
        repetitions=repetitions,
        block_length=int(bootstrap["block_length_observations"]),
        lower_quantile=float(bootstrap["lower_quantile"]),
        random_seed=int(bootstrap["random_seed"]),
    )
    stress_edge = paired_annualized_edge_lower_bound(
        stress,
        cash,
        periods_per_year=periods,
        repetitions=repetitions,
        block_length=int(bootstrap["block_length_observations"]),
        lower_quantile=float(bootstrap["lower_quantile"]),
        random_seed=int(bootstrap["random_seed"]) + 1,
    )
    incremental = paired_annualized_edge_lower_bound(
        stress,
        static,
        periods_per_year=periods,
        repetitions=repetitions,
        block_length=int(bootstrap["block_length_observations"]),
        lower_quantile=float(bootstrap["lower_quantile"]),
        random_seed=int(bootstrap["random_seed"]) + 2,
    )
    incremental_daily = [strategy - benchmark for strategy, benchmark in zip(stress, static)]
    dsr_contract = evaluation["deflated_sharpe"]
    dsr = deflated_sharpe_probability(
        incremental_daily,
        independent_trials=int(dsr_contract["independent_trials"]),
        euler_mascheroni=float(dsr_contract["euler_mascheroni"]),
    )
    pbo = probability_backtest_overfit(
        records,
        contract,
        environment_id=environment_id,
        decision_at=decision_at,
    )
    concentration = (
        maximum_environment_profit_concentration(qualification_records)
        if qualification_records
        else 1.0
    )
    forecast_volatility = (
        statistics.stdev(stress) * math.sqrt(float(periods)) if len(stress) >= 2 else None
    )
    pool_item = next(item for item in contract["strategy_pool"] if item["strategy_id"] == strategy_id)
    required_costs = set(pool_item["required_cost_components"])
    required_benchmarks = set(gates["required_benchmarks"])
    cost_complete = bool(qualification_records) and all(
        record["cost_model_complete"]
        and required_costs.issubset(set(record["included_cost_components"]))
        for record in qualification_records
    )
    benchmark_complete = bool(qualification_records) and all(
        record["benchmark_set_complete"]
        and required_benchmarks.issubset(set(record["benchmarks"]))
        for record in qualification_records
    )
    statistical_gates = {
        "minimum_forward_days": cumulative_count
        >= int(gates["minimum_forward_observation_days"]),
        "rolling_qualification_window_complete": qualification_count
        >= rolling_window_days,
        "recent_health_window_complete": recent_count >= recent_window_days,
        "minimum_environment_episodes": episode_count
        >= int(gates["minimum_independent_environment_episodes"]),
        "latest_forward_observation_status_pass": latest_forward_status_pass,
        "base_edge_lcb_positive_after_all_costs": base_edge["lower_bound"] is not None
        and base_edge["lower_bound"]
        > float(gates["minimum_base_net_conditional_edge_lcb_after_all_costs"]),
        "stress_edge_lcb_positive_after_all_costs": stress_edge["lower_bound"] is not None
        and stress_edge["lower_bound"]
        > float(gates["minimum_stress_net_conditional_edge_lcb_after_all_costs"]),
        "incremental_vs_static_equal_risk_lcb_positive_after_all_costs": incremental[
            "lower_bound"
        ]
        is not None
        and incremental["lower_bound"]
        > float(
            gates[
                "minimum_stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs"
            ]
        ),
        "recent_stress_edge_vs_cash_positive_after_all_costs": recent_stress_edge
        is not None
        and recent_stress_edge
        > float(gates["minimum_recent_stress_net_edge_vs_cash_after_all_costs"]),
        "recent_stress_incremental_vs_static_positive_after_all_costs": recent_incremental_edge
        is not None
        and recent_incremental_edge
        > float(
            gates[
                "minimum_recent_stress_net_incremental_vs_static_equal_risk_after_all_costs"
            ]
        ),
        "deflated_sharpe_probability_pass": dsr["probability"] is not None
        and dsr["probability"] >= float(gates["minimum_deflated_sharpe_probability"]),
        "probability_backtest_overfit_pass": pbo["probability"] is not None
        and pbo["probability"] <= float(gates["maximum_probability_backtest_overfit"]),
        "environment_profit_concentration_pass": concentration
        <= float(gates["maximum_environment_profit_concentration"]),
        "forecast_volatility_pass": forecast_volatility is not None
        and 0.0 < forecast_volatility
        <= float(gates["maximum_forecast_annualized_volatility"]),
        "benchmark_set_complete": benchmark_complete,
        "cost_model_complete": cost_complete,
    }
    preliminary_pass = all(statistical_gates.values())
    pending_status = evaluation["first_statistical_pass_status"]
    passed_status = evaluation["confirmed_pass_status"]
    previous_state = _previous_confirmation_state(
        previous_evidence,
        strategy_id=strategy_id,
        environment_id=environment_id,
        decision_at=decision_at,
        pending_status=pending_status,
        passed_status=passed_status,
    )
    confirmation_contract = evaluation["confirmation"]
    current_window_end = (
        qualification_records[-1]["outcome_at"] if qualification_records else None
    )
    previous_available = previous_state["available_at"]
    previous_window_end = previous_state["evaluation_window_end"]
    elapsed_hours = (
        None
        if previous_available is None
        else (decision_at - previous_available).total_seconds() / 3600.0
    )
    new_complete_days = cumulative_count - int(previous_state["forward_observation_days"])
    enough_elapsed = elapsed_hours is None or elapsed_hours >= float(
        confirmation_contract["minimum_elapsed_hours"]
    )
    enough_new_days = new_complete_days >= int(
        confirmation_contract["minimum_new_complete_observation_days"]
    )
    later_window_end = (
        previous_window_end is None
        or (current_window_end is not None and current_window_end > previous_window_end)
    )
    confirmation_increment_allowed = bool(
        previous_state["count"] > 0
        and enough_elapsed
        and enough_new_days
        and later_window_end
    )
    if preliminary_pass:
        if previous_state["count"] <= 0:
            confirmation_count = 1
        elif confirmation_increment_allowed:
            confirmation_count = int(previous_state["count"]) + 1
        else:
            confirmation_count = int(previous_state["count"])
    else:
        confirmation_count = 0
    if preliminary_pass and confirmation_count >= int(gates["minimum_consecutive_entry_confirmations"]):
        meta_status = passed_status
    elif preliminary_pass:
        meta_status = pending_status
    elif latest_group_record is not None and latest_group_record["data_status"] == "NO_VIEW":
        meta_status = "NO_VIEW"
    elif latest_group_record is not None and latest_group_record["data_status"] == "RUNNING":
        meta_status = "FORWARD_COLLECTING"
    elif latest_group_record is not None and not latest_forward_status_pass:
        meta_status = evaluation["insufficient_status"]
    elif cumulative_count < int(gates["minimum_forward_observation_days"]):
        meta_status = "FORWARD_COLLECTING"
    else:
        meta_status = evaluation["insufficient_status"]

    return {
        "strategy_id": strategy_id,
        "environment_id": environment_id,
        "decision_at": decision_at.isoformat(),
        "meta_evidence_status": meta_status,
        "preliminary_statistical_pass": preliminary_pass,
        "consecutive_entry_confirmations": int(confirmation_count),
        "forward_observation_days": int(cumulative_count),
        "recent_health_observation_days": int(recent_count),
        "independent_environment_episodes": int(episode_count),
        "evaluation_window_start": (
            qualification_records[0]["signal_at"].isoformat()
            if qualification_records
            else None
        ),
        "evaluation_window_end": (
            qualification_records[-1]["outcome_at"].isoformat()
            if qualification_records
            else None
        ),
        "latest_forward_observation_at": (
            latest_group_record["outcome_at"].isoformat()
            if latest_group_record is not None
            else None
        ),
        "latest_forward_data_status": (
            latest_group_record["data_status"] if latest_group_record is not None else None
        ),
        "base_net_conditional_edge_lcb_after_all_costs": base_edge["lower_bound"],
        "stress_net_conditional_edge_lcb_after_all_costs": stress_edge["lower_bound"],
        "stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs": incremental[
            "lower_bound"
        ],
        "recent_stress_net_edge_vs_cash_after_all_costs": recent_stress_edge,
        "recent_stress_net_incremental_vs_static_equal_risk_after_all_costs": recent_incremental_edge,
        "deflated_sharpe_probability": dsr["probability"],
        "probability_backtest_overfit": pbo["probability"],
        "maximum_environment_profit_concentration": float(concentration),
        "forecast_annualized_volatility": (
            None if forecast_volatility is None else float(forecast_volatility)
        ),
        "benchmark_set_complete": benchmark_complete,
        "benchmarks": list(gates["required_benchmarks"]) if benchmark_complete else [],
        "cost_model_complete": cost_complete,
        "included_cost_components": (
            list(pool_item["required_cost_components"]) if cost_complete else []
        ),
        "statistical_gates": statistical_gates,
        "failed_statistical_gates": [
            name for name, passed in statistical_gates.items() if not passed
        ],
        "statistics": {
            "base_edge_bootstrap": base_edge,
            "stress_edge_bootstrap": stress_edge,
            "incremental_edge_bootstrap": incremental,
            "deflated_sharpe": dsr,
            "probability_backtest_overfit": pbo,
            "rolling_qualification_window_observation_days": int(
                qualification_count
            ),
            "recent_health_window_observation_days": int(recent_count),
            "recent_stress_net_edge_vs_cash_after_all_costs": recent_stress_edge,
            "recent_stress_net_incremental_vs_static_equal_risk_after_all_costs": recent_incremental_edge,
        },
        "confirmation_progress": {
            "previous_status": previous_state["status"],
            "previous_confirmation_count": int(previous_state["count"]),
            "elapsed_hours_since_previous_evidence": elapsed_hours,
            "new_complete_observation_days": int(new_complete_days),
            "later_evaluation_window_end": bool(later_window_end),
            "minimum_elapsed_hours": int(confirmation_contract["minimum_elapsed_hours"]),
            "minimum_new_complete_observation_days": int(
                confirmation_contract["minimum_new_complete_observation_days"]
            ),
            "increment_allowed": confirmation_increment_allowed,
        },
    }


def build_evidence_core(
    evaluation: Mapping[str, Any],
    *,
    evidence_id: str,
    recorded_at: datetime,
    available_at: datetime,
) -> dict[str, Any]:
    """构造将由不可变评价报告逐字段绑定的资格记录核心。"""

    return {
        "schema_version": "1.0.0",
        "evidence_id": evidence_id,
        "strategy_id": evaluation["strategy_id"],
        "environment_id": evaluation["environment_id"],
        "recorded_at": recorded_at.isoformat(),
        "available_at": available_at.isoformat(),
        "evidence_type": "TRUE_FORWARD_CONDITIONAL",
        "meta_evidence_status": evaluation["meta_evidence_status"],
        "evaluation_window_start": evaluation["evaluation_window_start"],
        "evaluation_window_end": evaluation["evaluation_window_end"],
        "latest_forward_observation_at": evaluation["latest_forward_observation_at"],
        "latest_forward_data_status": evaluation["latest_forward_data_status"],
        "forward_observation_days": evaluation["forward_observation_days"],
        "recent_health_observation_days": evaluation["recent_health_observation_days"],
        "independent_environment_episodes": evaluation[
            "independent_environment_episodes"
        ],
        "consecutive_entry_confirmations": evaluation[
            "consecutive_entry_confirmations"
        ],
        "historical_contamination": False,
        "base_net_conditional_edge_lcb_after_all_costs": evaluation[
            "base_net_conditional_edge_lcb_after_all_costs"
        ],
        "stress_net_conditional_edge_lcb_after_all_costs": evaluation[
            "stress_net_conditional_edge_lcb_after_all_costs"
        ],
        "stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs": evaluation[
            "stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs"
        ],
        "recent_stress_net_edge_vs_cash_after_all_costs": evaluation[
            "recent_stress_net_edge_vs_cash_after_all_costs"
        ],
        "recent_stress_net_incremental_vs_static_equal_risk_after_all_costs": evaluation[
            "recent_stress_net_incremental_vs_static_equal_risk_after_all_costs"
        ],
        "deflated_sharpe_probability": evaluation["deflated_sharpe_probability"],
        "probability_backtest_overfit": evaluation["probability_backtest_overfit"],
        "maximum_environment_profit_concentration": evaluation[
            "maximum_environment_profit_concentration"
        ],
        "forecast_annualized_volatility": evaluation["forecast_annualized_volatility"],
        "benchmark_set_complete": evaluation["benchmark_set_complete"],
        "benchmarks": evaluation["benchmarks"],
        "cost_model_complete": evaluation["cost_model_complete"],
        "included_cost_components": evaluation["included_cost_components"],
        "paper_position_generation": False,
        "shadow_signal_generation": False,
        "order_generation": False,
        "live_trading_authorized": False,
    }
