from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import scripts.build_regime_aware_strategy_forward_evidence_v1 as build_module
import scripts.freeze_regime_aware_strategy_allocator_v1 as freeze_module
from research.regime_aware_strategy_allocator_v1 import (
    CONFIG,
    EXPECTED_STRATEGY_IDS,
    EvidenceError,
    load_contract,
    validate_environment_records,
)
from research.regime_aware_strategy_forward_evidence_v1 import (
    deflated_sharpe_probability,
    derive_environment_episode_map,
    evaluate_strategy_environment_forward_evidence,
    maximum_environment_profit_concentration,
    paired_annualized_edge_lower_bound,
    probability_backtest_overfit,
    validate_forward_observations,
)


FROZEN_AT = datetime(2030, 1, 1, 0, 0, tzinfo=timezone.utc)
V15 = "DIGITAL_ASSET_CROSS_SECTIONAL_PERPETUAL_BASIS_V15"
CARRY = "CARRY_DISPERSION"
REQUIRED_BENCHMARKS = [
    "CASH_CNY",
    "STATIC_EQUAL_RISK_FROZEN_POOL",
    "H00300_TOTAL_RETURN_CAPITAL_ALTERNATIVE",
    "EACH_STATIC_COMPONENT",
]


def contract() -> dict:
    return copy.deepcopy(load_contract(CONFIG))


def source_binding(tmp_path: Path, name: str, content: str) -> tuple[str, str]:
    relative = f"evidence/{name}"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return relative, hashlib.sha256(path.read_bytes()).hexdigest()


def environment_features(environment_id: str) -> dict[str, float]:
    if environment_id == CARRY:
        return {
            "slow_trend_score": 0.0,
            "fast_trend_score": 0.0,
            "breadth_score": 0.5,
            "risk_score": 0.3,
            "liquidity_score": 0.8,
            "carry_opportunity_score": 0.8,
        }
    if environment_id == "TREND_PERSISTENT_UP":
        return {
            "slow_trend_score": 0.4,
            "fast_trend_score": 0.3,
            "breadth_score": 0.7,
            "risk_score": 0.3,
            "liquidity_score": 0.8,
            "carry_opportunity_score": 0.2,
        }
    if environment_id == "DEFENSIVE_HIGH_RISK":
        return {
            "slow_trend_score": 0.0,
            "fast_trend_score": 0.0,
            "breadth_score": 0.5,
            "risk_score": 0.8,
            "liquidity_score": 0.8,
            "carry_opportunity_score": 0.2,
        }
    raise AssertionError(f"测试未定义环境特征：{environment_id}")


def make_environment_record(
    *,
    evidence_id: str,
    observation_at: datetime,
    available_at: datetime,
    environment_id: str,
    source_path: str | None,
    source_sha256: str | None,
    data_status: str = "PASS",
) -> dict:
    return {
        "schema_version": "1.0.0",
        "evidence_id": evidence_id,
        "environment_scope": "DIGITAL_ASSET",
        "observation_at": observation_at.isoformat(),
        "available_at": available_at.isoformat(),
        "data_status": data_status,
        "source_path": source_path if data_status == "PASS" else None,
        "source_sha256": source_sha256 if data_status == "PASS" else None,
        "features": environment_features(environment_id) if data_status == "PASS" else None,
        "reason": "合成前向测试环境记录",
    }


def component_costs(
    value: dict,
    *,
    strategy_id: str,
    allocator_switch_event: bool,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    item = next(entry for entry in value["strategy_pool"] if entry["strategy_id"] == strategy_id)
    components = list(item["required_cost_components"])
    base = {component: 0.00001 for component in components}
    stress = {component: 0.00002 for component in components}
    base["allocator_switching"] = (
        float(item["base_allocator_switch_cost_bps"]) / 10_000.0
        if allocator_switch_event
        else 0.0
    )
    stress["allocator_switching"] = (
        float(item["stress_allocator_switch_cost_bps"]) / 10_000.0
        if allocator_switch_event
        else 0.0
    )
    return components, base, stress


def make_observation(
    value: dict,
    *,
    index: int,
    strategy_id: str,
    environment_id: str,
    environment_evidence_id: str,
    environment_episode_id: str,
    signal_at: datetime,
    source_path: str,
    source_sha256: str,
    data_status: str = "PASS",
) -> dict:
    outcome_at = signal_at + timedelta(hours=24)
    available_at = outcome_at + timedelta(minutes=5)
    if data_status == "PASS":
        allocator_switch_event = index % 21 == 0
        components, base_costs, stress_costs = component_costs(
            value,
            strategy_id=strategy_id,
            allocator_switch_event=allocator_switch_event,
        )
        gross_return = 0.004 + ((index % 9) - 4) * 0.00004
        base_net = gross_return - sum(base_costs.values())
        stress_net = gross_return - sum(stress_costs.values())
        complete = True
        benchmarks = list(REQUIRED_BENCHMARKS)
    else:
        allocator_switch_event = False
        components, base_costs, stress_costs = [], {}, {}
        gross_return = None
        base_net = None
        stress_net = None
        complete = False
        benchmarks = []
    return {
        "schema_version": "1.0.0",
        "observation_id": f"OBS_{strategy_id}_{index:04d}",
        "strategy_id": strategy_id,
        "environment_id": environment_id,
        "environment_evidence_id": environment_evidence_id,
        "environment_episode_id": environment_episode_id,
        "signal_at": signal_at.isoformat(),
        "outcome_at": outcome_at.isoformat(),
        "available_at": available_at.isoformat(),
        "data_status": data_status,
        "quality_complete": complete,
        "forward_observed": complete,
        "strategy_gross_return": gross_return,
        "strategy_base_net_return": base_net,
        "strategy_stress_net_return": stress_net,
        "cash_net_return": 0.00005 if complete else None,
        "static_equal_risk_stress_net_return": 0.00020 if complete else None,
        "benchmark_set_complete": complete,
        "benchmarks": benchmarks,
        "cost_model_complete": complete,
        "included_cost_components": components,
        "allocator_switch_event": allocator_switch_event,
        "base_costs_fraction_by_component": base_costs,
        "stress_costs_fraction_by_component": stress_costs,
        "source_path": source_path,
        "source_sha256": source_sha256,
        "paper_position_generation": False,
        "shadow_signal_generation": False,
        "order_generation": False,
        "live_trading_authorized": False,
    }


def make_forward_dataset(
    tmp_path: Path,
    count: int,
    *,
    strategy_id: str = V15,
    environment_id: str = CARRY,
) -> tuple[list[dict], list[dict]]:
    value = contract()
    environment_path, environment_sha = source_binding(
        tmp_path, "environment_source.json", "冻结环境特征原始来源"
    )
    observation_path, observation_sha = source_binding(
        tmp_path, "observation_source.json", "冻结策略收益原始来源"
    )
    raw_environment: list[dict] = []
    raw_observations: list[dict] = []
    cursor = FROZEN_AT + timedelta(days=1)
    current_episode_id: str | None = None
    for index in range(count):
        if index > 0 and index % 42 == 0:
            raw_environment.append(
                make_environment_record(
                    evidence_id=f"ENV_BREAK_{index:04d}",
                    observation_at=cursor,
                    available_at=cursor + timedelta(minutes=5),
                    environment_id="DEFENSIVE_HIGH_RISK",
                    source_path=environment_path,
                    source_sha256=environment_sha,
                )
            )
            cursor += timedelta(days=1)
            current_episode_id = None
        environment_evidence_id = f"ENV_{index:04d}"
        environment_available = cursor + timedelta(minutes=5)
        raw_environment.append(
            make_environment_record(
                evidence_id=environment_evidence_id,
                observation_at=cursor,
                available_at=environment_available,
                environment_id=environment_id,
                source_path=environment_path,
                source_sha256=environment_sha,
            )
        )
        if current_episode_id is None:
            current_episode_id = f"{environment_id}::{environment_evidence_id}"
        raw_observations.append(
            make_observation(
                value,
                index=index,
                strategy_id=strategy_id,
                environment_id=environment_id,
                environment_evidence_id=environment_evidence_id,
                environment_episode_id=current_episode_id,
                signal_at=environment_available + timedelta(minutes=1),
                source_path=observation_path,
                source_sha256=observation_sha,
            )
        )
        cursor += timedelta(days=1)
    normalized_environment = validate_environment_records(
        raw_environment,
        value,
        root=tmp_path,
    )
    return normalized_environment, raw_observations


def validate_dataset(
    tmp_path: Path,
    count: int,
    *,
    strategy_id: str = V15,
    environment_id: str = CARRY,
) -> tuple[dict, list[dict], list[dict]]:
    value = contract()
    environment, raw_observations = make_forward_dataset(
        tmp_path,
        count,
        strategy_id=strategy_id,
        environment_id=environment_id,
    )
    observations = validate_forward_observations(
        raw_observations,
        value,
        manifest_frozen_at=FROZEN_AT,
        environment_records=environment,
        root=tmp_path,
    )
    return value, environment, observations


def test_valid_observation_recomputes_all_component_costs(tmp_path: Path) -> None:
    _, _, observations = validate_dataset(tmp_path, 1)
    record = observations[0]
    assert record["strategy_base_net_return"] == pytest.approx(
        record["strategy_gross_return"]
        - sum(record["base_costs_fraction_by_component"].values())
    )
    assert record["strategy_stress_net_return"] == pytest.approx(
        record["strategy_gross_return"]
        - sum(record["stress_costs_fraction_by_component"].values())
    )
    assert record["source_binding"]["verified"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda record: record.__setitem__(
                "strategy_base_net_return", record["strategy_base_net_return"] + 0.0001
            ),
            "基础净收益无法",
        ),
        (
            lambda record: record["stress_costs_fraction_by_component"].__setitem__(
                "commission", 0.0
            ),
            "压力成本不得低于基础成本",
        ),
        (
            lambda record: record["base_costs_fraction_by_component"].__setitem__(
                "allocator_switching", 0.0
            ),
            "基础分配切换成本",
        ),
    ],
)
def test_cost_tampering_is_rejected(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    value = contract()
    environment, raw_observations = make_forward_dataset(tmp_path, 1)
    mutation(raw_observations[0])
    with pytest.raises(EvidenceError, match=message):
        validate_forward_observations(
            raw_observations,
            value,
            manifest_frozen_at=FROZEN_AT,
            environment_records=environment,
            root=tmp_path,
        )


def test_non_pass_status_is_preserved_without_fabricated_returns(tmp_path: Path) -> None:
    value = contract()
    environment, raw_observations = make_forward_dataset(tmp_path, 1)
    original = raw_observations[0]
    raw_observations[0] = make_observation(
        value,
        index=0,
        strategy_id=V15,
        environment_id=CARRY,
        environment_evidence_id=original["environment_evidence_id"],
        environment_episode_id=original["environment_episode_id"],
        signal_at=datetime.fromisoformat(original["signal_at"]),
        source_path=original["source_path"],
        source_sha256=original["source_sha256"],
        data_status="PROGRAM_FAILED",
    )
    validated = validate_forward_observations(
        raw_observations,
        value,
        manifest_frozen_at=FROZEN_AT,
        environment_records=environment,
        root=tmp_path,
    )
    assert validated[0]["data_status"] == "PROGRAM_FAILED"
    assert validated[0]["strategy_gross_return"] is None
    assert validated[0]["base_costs_fraction_by_component"] == {}


def test_latest_point_in_time_environment_record_is_mandatory(tmp_path: Path) -> None:
    value = contract()
    environment, raw_observations = make_forward_dataset(tmp_path, 2)
    raw_observations[1]["environment_evidence_id"] = "ENV_0000"
    with pytest.raises(EvidenceError, match="最新可得环境证据"):
        validate_forward_observations(
            raw_observations,
            value,
            manifest_frozen_at=FROZEN_AT,
            environment_records=environment,
            root=tmp_path,
        )


def test_environment_episode_id_cannot_be_split_by_hand(tmp_path: Path) -> None:
    value = contract()
    environment, raw_observations = make_forward_dataset(tmp_path, 2)
    raw_observations[1]["environment_episode_id"] = "手工切片"
    with pytest.raises(EvidenceError, match="不是由冻结环境序列派生"):
        validate_forward_observations(
            raw_observations,
            value,
            manifest_frozen_at=FROZEN_AT,
            environment_records=environment,
            root=tmp_path,
        )


def test_distinct_regime_break_starts_a_new_derived_episode(tmp_path: Path) -> None:
    value = contract()
    environment, _ = make_forward_dataset(tmp_path, 43)
    mapping = derive_environment_episode_map(
        environment,
        value,
        manifest_frozen_at=FROZEN_AT,
    )
    assert mapping["ENV_0000"]["environment_episode_id"] == f"{CARRY}::ENV_0000"
    assert mapping["ENV_0041"]["environment_episode_id"] == f"{CARRY}::ENV_0000"
    assert mapping["ENV_0042"]["environment_episode_id"] == f"{CARRY}::ENV_0042"


def test_short_no_view_gap_cannot_manufacture_a_new_same_regime_episode(
    tmp_path: Path,
) -> None:
    value = contract()
    source_path, source_sha = source_binding(tmp_path, "episode_environment.json", "环境来源")
    first_at = FROZEN_AT + timedelta(days=1)
    raw = [
        make_environment_record(
            evidence_id="ENV_FIRST",
            observation_at=first_at,
            available_at=first_at + timedelta(minutes=5),
            environment_id=CARRY,
            source_path=source_path,
            source_sha256=source_sha,
        ),
        make_environment_record(
            evidence_id="ENV_SHORT_NO_VIEW",
            observation_at=first_at + timedelta(hours=12),
            available_at=first_at + timedelta(hours=12, minutes=5),
            environment_id=CARRY,
            source_path=None,
            source_sha256=None,
            data_status="NO_VIEW",
        ),
        make_environment_record(
            evidence_id="ENV_SECOND",
            observation_at=first_at + timedelta(days=1),
            available_at=first_at + timedelta(days=1, minutes=5),
            environment_id=CARRY,
            source_path=source_path,
            source_sha256=source_sha,
        ),
    ]
    environment = validate_environment_records(raw, value, root=tmp_path)
    mapping = derive_environment_episode_map(
        environment,
        value,
        manifest_frozen_at=FROZEN_AT,
    )
    assert mapping["ENV_SECOND"]["environment_episode_id"] == f"{CARRY}::ENV_FIRST"


def test_prefreeze_signal_is_never_accepted(tmp_path: Path) -> None:
    value = contract()
    environment, raw_observations = make_forward_dataset(tmp_path, 1)
    raw_observations[0]["signal_at"] = FROZEN_AT.isoformat()
    with pytest.raises(EvidenceError, match="未严格晚于清单冻结时点"):
        validate_forward_observations(
            raw_observations,
            value,
            manifest_frozen_at=FROZEN_AT,
            environment_records=environment,
            root=tmp_path,
        )


def test_observation_horizon_must_be_exactly_twenty_four_hours(tmp_path: Path) -> None:
    value = contract()
    environment, raw_observations = make_forward_dataset(tmp_path, 1)
    signal_at = datetime.fromisoformat(raw_observations[0]["signal_at"])
    raw_observations[0]["outcome_at"] = (signal_at + timedelta(hours=23)).isoformat()
    raw_observations[0]["available_at"] = (signal_at + timedelta(hours=23, minutes=5)).isoformat()
    with pytest.raises(EvidenceError, match="精确为冻结的 24 小时"):
        validate_forward_observations(
            raw_observations,
            value,
            manifest_frozen_at=FROZEN_AT,
            environment_records=environment,
            root=tmp_path,
        )


def test_same_strategy_environment_cannot_count_twice_on_one_observation_day(
    tmp_path: Path,
) -> None:
    value = contract()
    environment, raw_observations = make_forward_dataset(tmp_path, 1)
    duplicate = copy.deepcopy(raw_observations[0])
    duplicate["observation_id"] = "OBS_DUPLICATE_SAME_DAY"
    for field in ("signal_at", "outcome_at", "available_at"):
        duplicate[field] = (datetime.fromisoformat(duplicate[field]) + timedelta(hours=1)).isoformat()
    with pytest.raises(EvidenceError, match="上海观察日重复"):
        validate_forward_observations(
            [raw_observations[0], duplicate],
            value,
            manifest_frozen_at=FROZEN_AT,
            environment_records=environment,
            root=tmp_path,
        )


def test_future_observations_do_not_enter_current_evaluation(tmp_path: Path) -> None:
    value, _, observations = validate_dataset(tmp_path, 10)
    decision_at = observations[4]["available_at"]
    result = evaluate_strategy_environment_forward_evidence(
        observations,
        value,
        strategy_id=V15,
        environment_id=CARRY,
        decision_at=decision_at,
        bootstrap_repetitions_override=20,
    )
    assert result["forward_observation_days"] == 5
    assert result["meta_evidence_status"] == "FORWARD_COLLECTING"


def test_latest_failed_forward_observation_immediately_removes_statistical_pass(
    tmp_path: Path,
) -> None:
    value = contract()
    environment, raw_observations = make_forward_dataset(tmp_path, 505)
    latest = raw_observations[-1]
    raw_observations[-1] = make_observation(
        value,
        index=504,
        strategy_id=V15,
        environment_id=CARRY,
        environment_evidence_id=latest["environment_evidence_id"],
        environment_episode_id=latest["environment_episode_id"],
        signal_at=datetime.fromisoformat(latest["signal_at"]),
        source_path=latest["source_path"],
        source_sha256=latest["source_sha256"],
        data_status="PROGRAM_FAILED",
    )
    observations = validate_forward_observations(
        raw_observations,
        value,
        manifest_frozen_at=FROZEN_AT,
        environment_records=environment,
        root=tmp_path,
    )
    result = evaluate_strategy_environment_forward_evidence(
        observations,
        value,
        strategy_id=V15,
        environment_id=CARRY,
        decision_at=observations[-1]["available_at"],
        bootstrap_repetitions_override=40,
    )
    assert result["forward_observation_days"] == 504
    assert result["latest_forward_data_status"] == "PROGRAM_FAILED"
    assert result["statistical_gates"]["latest_forward_observation_status_pass"] is False
    assert result["preliminary_statistical_pass"] is False
    assert result["meta_evidence_status"] == "INSUFFICIENT_EVIDENCE"
    assert result["consecutive_entry_confirmations"] == 0


def test_recent_ninety_day_weakening_forces_exit_even_when_long_window_stays_positive(
    tmp_path: Path,
) -> None:
    value = contract()
    environment, raw_observations = make_forward_dataset(tmp_path, 504)
    for record in raw_observations[-90:]:
        gross_return = 0.00010
        record["strategy_gross_return"] = gross_return
        record["strategy_base_net_return"] = gross_return - sum(
            record["base_costs_fraction_by_component"].values()
        )
        record["strategy_stress_net_return"] = gross_return - sum(
            record["stress_costs_fraction_by_component"].values()
        )
    observations = validate_forward_observations(
        raw_observations,
        value,
        manifest_frozen_at=FROZEN_AT,
        environment_records=environment,
        root=tmp_path,
    )
    result = evaluate_strategy_environment_forward_evidence(
        observations,
        value,
        strategy_id=V15,
        environment_id=CARRY,
        decision_at=observations[-1]["available_at"],
        bootstrap_repetitions_override=40,
    )
    assert result["forward_observation_days"] == 504
    assert result["recent_health_observation_days"] == 90
    assert result["stress_net_conditional_edge_lcb_after_all_costs"] > 0.0
    assert result["stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs"] > 0.0
    assert result["recent_stress_net_edge_vs_cash_after_all_costs"] < 0.0
    assert (
        result[
            "recent_stress_net_incremental_vs_static_equal_risk_after_all_costs"
        ]
        < 0.0
    )
    assert (
        result["statistical_gates"][
            "recent_stress_edge_vs_cash_positive_after_all_costs"
        ]
        is False
    )
    assert (
        result["statistical_gates"][
            "recent_stress_incremental_vs_static_positive_after_all_costs"
        ]
        is False
    )
    assert result["preliminary_statistical_pass"] is False
    assert result["meta_evidence_status"] == "INSUFFICIENT_EVIDENCE"


def test_qualification_statistics_ignore_observations_older_than_latest_504(
    tmp_path: Path,
) -> None:
    value = contract()
    environment, raw_observations = make_forward_dataset(tmp_path, 594)
    for record in raw_observations[:90]:
        gross_return = -0.20
        record["strategy_gross_return"] = gross_return
        record["strategy_base_net_return"] = gross_return - sum(
            record["base_costs_fraction_by_component"].values()
        )
        record["strategy_stress_net_return"] = gross_return - sum(
            record["stress_costs_fraction_by_component"].values()
        )
    observations = validate_forward_observations(
        raw_observations,
        value,
        manifest_frozen_at=FROZEN_AT,
        environment_records=environment,
        root=tmp_path,
    )
    decision_at = observations[-1]["available_at"]
    result = evaluate_strategy_environment_forward_evidence(
        observations,
        value,
        strategy_id=V15,
        environment_id=CARRY,
        decision_at=decision_at,
        bootstrap_repetitions_override=40,
    )
    pbo = probability_backtest_overfit(
        observations,
        value,
        environment_id=CARRY,
        decision_at=decision_at,
    )
    assert result["forward_observation_days"] == 594
    assert result["evaluation_window_start"] == observations[-504]["signal_at"].isoformat()
    assert result["statistics"]["rolling_qualification_window_observation_days"] == 504
    assert result["stress_net_conditional_edge_lcb_after_all_costs"] > 0.0
    assert result["preliminary_statistical_pass"] is True
    assert pbo["status"] == "COMPLETE"
    assert pbo["complete_day_count"] == 504


def test_bootstrap_and_deflated_sharpe_are_deterministic() -> None:
    strategy = [0.002 + ((index % 5) - 2) * 0.0001 for index in range(100)]
    benchmark = [0.0001 for _ in strategy]
    first = paired_annualized_edge_lower_bound(
        strategy,
        benchmark,
        periods_per_year=365,
        repetitions=100,
        block_length=20,
        lower_quantile=0.05,
        random_seed=20260827,
    )
    second = paired_annualized_edge_lower_bound(
        strategy,
        benchmark,
        periods_per_year=365,
        repetitions=100,
        block_length=20,
        lower_quantile=0.05,
        random_seed=20260827,
    )
    assert first == second
    assert first["lower_bound"] > 0.0
    dsr = deflated_sharpe_probability(
        [strategy_value - benchmark_value for strategy_value, benchmark_value in zip(strategy, benchmark)],
        independent_trials=35,
        euler_mascheroni=0.5772156649015329,
    )
    assert dsr["probability"] > 0.95


def test_single_permitted_strategy_uses_frozen_static_baseline_for_pbo(tmp_path: Path) -> None:
    value, _, observations = validate_dataset(tmp_path, 504)
    decision_at = observations[-1]["available_at"]
    result = probability_backtest_overfit(
        observations,
        value,
        environment_id=CARRY,
        decision_at=decision_at,
    )
    assert result["status"] == "COMPLETE"
    assert result["candidate_count"] == 2
    assert result["baseline_candidate_id"] == "STATIC_EQUAL_RISK_FROZEN_POOL"
    assert result["probability"] == pytest.approx(0.0)


def test_missing_frozen_strategy_candidate_prevents_trend_pbo(tmp_path: Path) -> None:
    value, _, observations = validate_dataset(
        tmp_path,
        10,
        strategy_id=EXPECTED_STRATEGY_IDS[0],
        environment_id="TREND_PERSISTENT_UP",
    )
    result = probability_backtest_overfit(
        observations,
        value,
        environment_id="TREND_PERSISTENT_UP",
        decision_at=observations[-1]["available_at"],
    )
    assert result["status"] == "INCOMPLETE_FROZEN_CANDIDATE_UNIVERSE"
    assert set(result["missing_strategy_candidates"]) == set(EXPECTED_STRATEGY_IDS[1:4])


def test_statistical_pass_requires_two_separated_forward_confirmations(tmp_path: Path) -> None:
    value, _, observations = validate_dataset(tmp_path, 511)
    first_decision = observations[503]["available_at"]
    first = evaluate_strategy_environment_forward_evidence(
        observations,
        value,
        strategy_id=V15,
        environment_id=CARRY,
        decision_at=first_decision,
        bootstrap_repetitions_override=60,
    )
    assert first["preliminary_statistical_pass"] is True
    assert first["meta_evidence_status"] == "CONDITIONALLY_PASSING_CONFIRMATION_PENDING"
    assert first["consecutive_entry_confirmations"] == 1
    previous = {
        "strategy_id": V15,
        "environment_id": CARRY,
        "available_at": first_decision.isoformat(),
        "meta_evidence_status": first["meta_evidence_status"],
        "consecutive_entry_confirmations": 1,
        "evaluation_window_end": first["evaluation_window_end"],
        "forward_observation_days": 504,
    }
    too_early = evaluate_strategy_environment_forward_evidence(
        observations,
        value,
        strategy_id=V15,
        environment_id=CARRY,
        decision_at=observations[509]["available_at"],
        previous_evidence=[previous],
        bootstrap_repetitions_override=60,
    )
    assert too_early["consecutive_entry_confirmations"] == 1
    assert too_early["confirmation_progress"]["increment_allowed"] is False
    confirmed = evaluate_strategy_environment_forward_evidence(
        observations,
        value,
        strategy_id=V15,
        environment_id=CARRY,
        decision_at=observations[510]["available_at"],
        previous_evidence=[previous],
        bootstrap_repetitions_override=60,
    )
    assert confirmed["meta_evidence_status"] == "CONDITIONAL_FORWARD_PASSED"
    assert confirmed["consecutive_entry_confirmations"] == 2
    assert confirmed["confirmation_progress"]["increment_allowed"] is True


def test_profit_concentration_penalizes_one_episode_dominance() -> None:
    records = [
        {
            "environment_episode_id": "A",
            "strategy_stress_net_return": 0.10,
            "static_equal_risk_stress_net_return": 0.0,
        },
        {
            "environment_episode_id": "B",
            "strategy_stress_net_return": 0.01,
            "static_equal_risk_stress_net_return": 0.0,
        },
    ]
    assert maximum_environment_profit_concentration(records) > 0.50


def test_prefreeze_audit_rejects_existing_forward_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = contract()
    monkeypatch.setattr(freeze_module, "ROOT", tmp_path)
    ledger = tmp_path / value["inputs"]["forward_observation_ledger_jsonl"]
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="冻结前已经存在资格证据"):
        freeze_module._assert_prefreeze_outputs_and_evidence_absent(value)


def test_forward_builder_persists_no_view_status_and_immutable_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = contract()
    manifest_path = tmp_path / "config" / "regime_aware_strategy_allocator_v1_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text('{"candidate_id":"REGIME_AWARE_STRATEGY_ALLOCATOR_V1"}\n', encoding="utf-8")
    monkeypatch.setattr(build_module, "ROOT", tmp_path)
    monkeypatch.setattr(build_module, "MANIFEST", manifest_path)
    payload = {
        "schema_version": "1.0.0",
        "evaluator_id": "REGIME_AWARE_STRATEGY_FORWARD_EVIDENCE_V1",
        "status": "FORWARD_OBSERVATION_LEDGER_MISSING_NO_VIEW",
        "decision_at": (FROZEN_AT + timedelta(days=1)).isoformat(),
        "validated_observation_count": 0,
        "generated_evidence_count": 0,
        "strategy_evidence_appended": False,
        "cash_only_remains_required": True,
        "live_trading_authorized": False,
    }
    artifacts = build_module.write_build_artifacts(payload)
    status_path = tmp_path / value["forward_evaluation"]["latest_build_status_json"]
    receipt_path = tmp_path / artifacts["receipt"]
    assert status_path.is_file()
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["run_status"] == "SUCCESS"
    assert receipt["build_status"] == "FORWARD_OBSERVATION_LEDGER_MISSING_NO_VIEW"
    assert receipt["strategy_evidence_appended"] is False
    assert receipt["live_trading_authorized"] is False


def test_forward_builder_rejects_future_decision_time_before_any_manifest_read() -> None:
    with pytest.raises(EvidenceError, match="不得晚于当前可得时间"):
        build_module.build(decision_at=datetime.now(timezone.utc) + timedelta(days=1))
