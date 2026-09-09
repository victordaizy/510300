from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import scripts.run_regime_aware_strategy_allocator_v1 as run_module

from research.regime_aware_strategy_allocator_v1 import (
    CONFIG,
    EXPECTED_STRATEGY_IDS,
    EvidenceError,
    allocate_research_risk,
    audit_component_statuses,
    classify_environment,
    evaluate_allocator,
    load_contract,
    render_markdown,
    validate_contract,
    validate_environment_records,
    validate_strategy_records,
)


ROOT = Path(__file__).resolve().parents[1]
FROZEN_AT = datetime(2030, 1, 1, 0, 0, tzinfo=timezone.utc)
DECISION_AT = datetime(2032, 1, 10, 8, 0, tzinfo=timezone.utc)


def contract() -> dict:
    return copy.deepcopy(load_contract(CONFIG))


def manifest() -> dict:
    return {
        "candidate_id": "REGIME_AWARE_STRATEGY_ALLOCATOR_V1",
        "frozen_at": FROZEN_AT.isoformat(),
        "tracked_content_sha256": "test-controlled-content-sha256",
    }


def source_binding(tmp_path: Path, name: str, content: str = "合成证据") -> tuple[str, str]:
    relative = f"evidence/{name}"
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return relative, digest


def relative_source_binding(tmp_path: Path, relative: str, content: str) -> tuple[str, str]:
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return relative, hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_strategy_source_binding(tmp_path: Path, record: dict) -> dict:
    value = contract()
    core = {
        key: value
        for key, value in record.items()
        if key not in {"_line_number", "source_path", "source_sha256"}
    }
    manifest_relative = "config/regime_aware_strategy_allocator_v1_manifest.json"
    _, manifest_sha = relative_source_binding(
        tmp_path,
        manifest_relative,
        json.dumps({"tracked_content_sha256": "test-controlled-content-sha256"}),
    )
    ledger_relative = value["inputs"]["forward_observation_ledger_jsonl"]
    ledger_count = max(int(core.get("forward_observation_days") or 0), 1)
    ledger_path = tmp_path / ledger_relative
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    existing_ledger_count = (
        len([line for line in ledger_path.read_text(encoding="utf-8").splitlines() if line])
        if ledger_path.is_file()
        else 0
    )
    if existing_ledger_count < ledger_count:
        ledger_path.write_bytes(b"{}\n" * ledger_count)
    ledger_prefix = ("{}\n" * ledger_count).encode("utf-8")
    ledger_sha = hashlib.sha256(ledger_prefix).hexdigest()
    environment_relative = value["inputs"]["environment_evidence_jsonl"]
    _, environment_sha = relative_source_binding(
        tmp_path,
        environment_relative,
        "{}\n",
    )
    environment_path = tmp_path / environment_relative
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
    evaluation = {
        "strategy_id": core["strategy_id"],
        "environment_id": core["environment_id"],
        "decision_at": core["recorded_at"],
        **{field: core[field] for field in mirrored_fields},
    }
    report = {
        "schema_version": "1.0.0",
        "evaluator_id": "REGIME_AWARE_STRATEGY_FORWARD_EVIDENCE_V1",
        "candidate_id": "REGIME_AWARE_STRATEGY_ALLOCATOR_V1",
        "allocator_tracked_content_sha256": "test-controlled-content-sha256",
        "allocator_manifest_path": manifest_relative,
        "allocator_manifest_sha256": manifest_sha,
        "generated_at": core["recorded_at"],
        "decision_at": core["recorded_at"],
        "ledger": {
            "path": ledger_relative,
            "bytes": len(ledger_prefix),
            "sha256": ledger_sha,
            "validated_observation_count": ledger_count,
        },
        "environment_evidence": {
            "path": environment_relative,
            "bytes": environment_path.stat().st_size,
            "sha256": environment_sha,
            "validated_record_count": 1,
        },
        "evidence_core": core,
        "evaluation": evaluation,
        "performance_used_for_recent_winner_ranking": False,
        "historical_backfill_used": False,
        "paper_position_generation": False,
        "shadow_signal_generation": False,
        "order_generation": False,
        "live_trading_authorized": False,
    }
    report_relative = (
        f"{value['forward_evaluation']['immutable_report_directory']}/"
        f"{record['evidence_id']}.json"
    )
    relative, digest = relative_source_binding(
        tmp_path,
        report_relative,
        json.dumps(report, ensure_ascii=False, allow_nan=False),
    )
    return {**core, "source_path": relative, "source_sha256": digest}


def environment_record(
    tmp_path: Path,
    *,
    evidence_id: str = "ENV_1",
    observation_at: datetime | None = None,
    available_at: datetime | None = None,
    data_status: str = "PASS",
    features: dict | None = None,
) -> dict:
    observation = observation_at or (DECISION_AT - timedelta(hours=2))
    available = available_at or (DECISION_AT - timedelta(hours=1))
    if data_status == "PASS":
        relative, digest = source_binding(tmp_path, f"{evidence_id}.json")
        feature_payload = features or {
            "slow_trend_score": 0.40,
            "fast_trend_score": 0.35,
            "breadth_score": 0.70,
            "risk_score": 0.30,
            "liquidity_score": 0.80,
            "carry_opportunity_score": 0.20,
        }
    else:
        relative, digest, feature_payload = None, None, None
    return {
        "schema_version": "1.0.0",
        "evidence_id": evidence_id,
        "environment_scope": "DIGITAL_ASSET",
        "observation_at": observation.isoformat(),
        "available_at": available.isoformat(),
        "data_status": data_status,
        "source_path": relative,
        "source_sha256": digest,
        "features": feature_payload,
        "reason": "仅用于合成测试",
    }


def strategy_record(
    tmp_path: Path,
    *,
    evidence_id: str = "STRATEGY_1",
    strategy_id: str = EXPECTED_STRATEGY_IDS[0],
    environment_id: str = "TREND_PERSISTENT_UP",
    available_at: datetime | None = None,
    meta_status: str = "CONDITIONAL_FORWARD_PASSED",
    evidence_type: str = "TRUE_FORWARD_CONDITIONAL",
) -> dict:
    available = available_at or (DECISION_AT - timedelta(hours=1))
    recorded = available
    passed = meta_status == "CONDITIONAL_FORWARD_PASSED"
    latest_forward_status = (
        "PASS"
        if passed
        else "RUNNING"
        if meta_status == "FORWARD_COLLECTING"
        else "FAILED"
    )
    core = {
        "schema_version": "1.0.0",
        "evidence_id": evidence_id,
        "strategy_id": strategy_id,
        "environment_id": environment_id,
        "recorded_at": recorded.isoformat(),
        "available_at": available.isoformat(),
        "evidence_type": evidence_type,
        "meta_evidence_status": meta_status,
        "evaluation_window_start": (
            (FROZEN_AT + timedelta(days=1)).isoformat() if passed else None
        ),
        "evaluation_window_end": (
            (recorded - timedelta(days=1)).isoformat() if passed else None
        ),
        "latest_forward_observation_at": (recorded - timedelta(days=1)).isoformat(),
        "latest_forward_data_status": latest_forward_status,
        "forward_observation_days": 504 if passed else None,
        "recent_health_observation_days": 90 if passed else None,
        "independent_environment_episodes": 12 if passed else None,
        "consecutive_entry_confirmations": 2 if passed else None,
        "historical_contamination": False,
        "base_net_conditional_edge_lcb_after_all_costs": 0.010 if passed else None,
        "stress_net_conditional_edge_lcb_after_all_costs": 0.005 if passed else None,
        "stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs": (
            0.002 if passed else None
        ),
        "recent_stress_net_edge_vs_cash_after_all_costs": 0.004 if passed else None,
        "recent_stress_net_incremental_vs_static_equal_risk_after_all_costs": (
            0.002 if passed else None
        ),
        "deflated_sharpe_probability": 0.96 if passed else None,
        "probability_backtest_overfit": 0.08 if passed else None,
        "maximum_environment_profit_concentration": 0.45 if passed else None,
        "forecast_annualized_volatility": 0.25 if passed else None,
        "benchmark_set_complete": passed,
        "benchmarks": (
            [
                "CASH_CNY",
                "STATIC_EQUAL_RISK_FROZEN_POOL",
                "H00300_TOTAL_RETURN_CAPITAL_ALTERNATIVE",
                "EACH_STATIC_COMPONENT",
            ]
            if passed
            else []
        ),
        "cost_model_complete": passed,
        "included_cost_components": (
            [
                "commission",
                "exchange_fee",
                "bid_ask_spread",
                "slippage",
                "market_impact",
                "fx",
                "allocator_switching",
            ]
            if passed
            else []
        ),
        "paper_position_generation": False,
        "shadow_signal_generation": False,
        "order_generation": False,
        "live_trading_authorized": False,
    }
    return refresh_strategy_source_binding(tmp_path, core)


def test_actual_contract_and_component_dependencies_are_complete() -> None:
    value = contract()
    validate_contract(value)
    assert tuple(item["strategy_id"] for item in value["strategy_pool"]) == EXPECTED_STRATEGY_IDS
    for item in value["strategy_pool"]:
        assert (ROOT / item["component_manifest"]).is_file()
        assert (ROOT / item["component_status_source"]).is_file()
    assert value["allocation"]["minimum_cash_weight"] == 0.20
    assert value["evidence_gates"]["minimum_forward_observation_days"] == 504
    assert value["forward_evaluation"]["rolling_qualification_window_observation_days"] == 504
    assert value["forward_evaluation"]["recent_health_window_observation_days"] == 90
    assert value["evidence_gates"]["minimum_independent_environment_episodes"] == 12
    assert value["inputs"]["stale_record_carry_forward_allowed"] is False
    assert value["inputs"]["append_only_evidence_required"] is True
    assert value["inputs"]["forward_environment_binding_required"] is True
    assert value["inputs"]["environment_episode_ids_may_be_user_supplied"] is False


def test_missing_all_evidence_is_no_view_and_one_hundred_percent_cash(tmp_path: Path) -> None:
    result = evaluate_allocator(
        contract(),
        [],
        [],
        manifest(),
        decision_at=DECISION_AT,
        root=tmp_path,
    )
    assert result["status"] == "NO_VIEW_CASH_ONLY"
    assert result["view_status"] == "NO_VIEW"
    assert result["eligible_strategy_count"] == 0
    assert result["allocation"]["cash_weight"] == pytest.approx(1.0)
    assert all(weight == 0.0 for key, weight in result["allocation"]["research_risk_weights"].items() if key != "CASH_CNY")
    assert result["governance"]["winner_chasing_input_used"] is False
    assert result["actionability"]["live_trading"] is False


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (
            {"slow_trend_score": 0.5, "fast_trend_score": 0.5, "breadth_score": 0.8, "risk_score": 0.8, "liquidity_score": 0.9, "carry_opportunity_score": 0.9},
            "DEFENSIVE_HIGH_RISK",
        ),
        (
            {"slow_trend_score": 0.0, "fast_trend_score": 0.0, "breadth_score": 0.5, "risk_score": 0.3, "liquidity_score": 0.8, "carry_opportunity_score": 0.8},
            "CARRY_DISPERSION",
        ),
        (
            {"slow_trend_score": 0.4, "fast_trend_score": 0.3, "breadth_score": 0.7, "risk_score": 0.3, "liquidity_score": 0.8, "carry_opportunity_score": 0.2},
            "TREND_PERSISTENT_UP",
        ),
        (
            {"slow_trend_score": -0.4, "fast_trend_score": -0.3, "breadth_score": 0.3, "risk_score": 0.3, "liquidity_score": 0.8, "carry_opportunity_score": 0.2},
            "TREND_PERSISTENT_DOWN",
        ),
        (
            {"slow_trend_score": 0.4, "fast_trend_score": -0.4, "breadth_score": 0.5, "risk_score": 0.3, "liquidity_score": 0.8, "carry_opportunity_score": 0.2},
            "TREND_TRANSITION",
        ),
        (
            {"slow_trend_score": 0.1, "fast_trend_score": -0.1, "breadth_score": 0.5, "risk_score": 0.3, "liquidity_score": 0.8, "carry_opportunity_score": 0.2},
            "TREND_TRANSITION",
        ),
        (
            {"slow_trend_score": 0.1, "fast_trend_score": 0.1, "breadth_score": 0.5, "risk_score": 0.3, "liquidity_score": 0.8, "carry_opportunity_score": 0.2},
            "MEAN_REVERTING_LIQUID",
        ),
        (
            {"slow_trend_score": 0.1, "fast_trend_score": 0.1, "breadth_score": 0.5, "risk_score": 0.3, "liquidity_score": 0.4, "carry_opportunity_score": 0.2},
            "UNKNOWN",
        ),
    ],
)
def test_fixed_environment_classification_precedence(
    tmp_path: Path, features: dict, expected: str
) -> None:
    value = contract()
    records = validate_environment_records(
        [environment_record(tmp_path, features=features)], value, root=tmp_path
    )
    result = classify_environment(records, value, decision_at=DECISION_AT)
    assert result["view_status"] == "VIEW"
    assert result["environment_id"] == expected


def test_stale_environment_is_no_view_not_a_carried_state(tmp_path: Path) -> None:
    value = contract()
    stale = environment_record(
        tmp_path,
        observation_at=DECISION_AT - timedelta(hours=25),
        available_at=DECISION_AT - timedelta(hours=24, minutes=30),
    )
    records = validate_environment_records([stale], value, root=tmp_path)
    result = classify_environment(records, value, decision_at=DECISION_AT)
    assert result["view_status"] == "NO_VIEW"
    assert "超过 24 小时" in result["reason"]


def test_future_environment_record_is_not_used_early(tmp_path: Path) -> None:
    value = contract()
    future = environment_record(
        tmp_path,
        observation_at=DECISION_AT + timedelta(minutes=1),
        available_at=DECISION_AT + timedelta(minutes=2),
    )
    records = validate_environment_records([future], value, root=tmp_path)
    result = classify_environment(records, value, decision_at=DECISION_AT)
    assert result["view_status"] == "NO_VIEW"
    assert result["evidence_id"] is None


def test_matching_true_forward_evidence_activates_only_research_risk(tmp_path: Path) -> None:
    env = environment_record(tmp_path)
    evidence = strategy_record(tmp_path)
    result = evaluate_allocator(
        contract(),
        [env],
        [evidence],
        manifest(),
        decision_at=DECISION_AT,
        root=tmp_path,
    )
    assert result["status"] == "RESEARCH_RISK_ALLOCATION_ELIGIBLE"
    assert result["eligible_strategy_count"] == 1
    assert result["strategy_eligibility"][0]["eligible"] is True
    weights = result["allocation"]["research_risk_weights"]
    assert weights[EXPECTED_STRATEGY_IDS[0]] == pytest.approx(0.25)
    assert weights["CASH_CNY"] == pytest.approx(0.75)
    assert result["allocation"]["estimated_stress_allocator_switch_cost_fraction"] == pytest.approx(0.00025)
    assert all(value is False for value in result["safety"].values())
    assert result["actionability"]["research_risk_budget_only"] is True


def test_environment_mismatch_keeps_strategy_at_zero(tmp_path: Path) -> None:
    carry_features = {
        "slow_trend_score": 0.0,
        "fast_trend_score": 0.0,
        "breadth_score": 0.5,
        "risk_score": 0.3,
        "liquidity_score": 0.8,
        "carry_opportunity_score": 0.8,
    }
    result = evaluate_allocator(
        contract(),
        [environment_record(tmp_path, features=carry_features)],
        [strategy_record(tmp_path)],
        manifest(),
        decision_at=DECISION_AT,
        root=tmp_path,
    )
    assert result["status"] == "NO_ELIGIBLE_STRATEGY_CASH_ONLY"
    assert result["allocation"]["cash_weight"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value", "failed_gate"),
    [
        ("forward_observation_days", 503, "minimum_forward_days"),
        ("recent_health_observation_days", 89, "recent_health_window_complete"),
        ("independent_environment_episodes", 11, "minimum_environment_episodes"),
        ("consecutive_entry_confirmations", 1, "minimum_entry_confirmations"),
        ("base_net_conditional_edge_lcb_after_all_costs", 0.0, "base_edge_lcb_positive_after_all_costs"),
        ("stress_net_conditional_edge_lcb_after_all_costs", 0.0, "stress_edge_lcb_positive_after_all_costs"),
        (
            "stress_net_incremental_vs_static_equal_risk_lcb_after_all_costs",
            0.0,
            "incremental_vs_static_equal_risk_lcb_positive_after_all_costs",
        ),
        (
            "recent_stress_net_edge_vs_cash_after_all_costs",
            0.0,
            "recent_stress_edge_vs_cash_positive_after_all_costs",
        ),
        (
            "recent_stress_net_incremental_vs_static_equal_risk_after_all_costs",
            0.0,
            "recent_stress_incremental_vs_static_positive_after_all_costs",
        ),
        ("deflated_sharpe_probability", 0.949, "deflated_sharpe_probability_pass"),
        ("probability_backtest_overfit", 0.101, "probability_backtest_overfit_pass"),
        ("maximum_environment_profit_concentration", 0.501, "environment_profit_concentration_pass"),
        ("forecast_annualized_volatility", 3.01, "forecast_volatility_pass"),
        ("historical_contamination", True, "historical_contamination_absent"),
        ("benchmark_set_complete", False, "benchmark_set_complete"),
        ("cost_model_complete", False, "cost_model_complete"),
    ],
)
def test_every_frozen_hard_gate_can_independently_force_cash(
    tmp_path: Path, field: str, value: object, failed_gate: str
) -> None:
    record = strategy_record(tmp_path)
    record[field] = value
    if field == "benchmark_set_complete" and value is False:
        record["benchmarks"] = []
    if field == "cost_model_complete" and value is False:
        record["included_cost_components"] = []
    record = refresh_strategy_source_binding(tmp_path, record)
    result = evaluate_allocator(
        contract(),
        [environment_record(tmp_path, evidence_id=f"ENV_{field}")],
        [record],
        manifest(),
        decision_at=DECISION_AT,
        root=tmp_path,
    )
    first = result["strategy_eligibility"][0]
    assert first["eligible"] is False
    assert failed_gate in first["failed_gates"]
    assert result["allocation"]["cash_weight"] == pytest.approx(1.0)


def test_claimed_complete_cost_set_missing_component_is_input_failure(tmp_path: Path) -> None:
    record = strategy_record(tmp_path)
    record["included_cost_components"].remove("market_impact")
    record = refresh_strategy_source_binding(tmp_path, record)
    with pytest.raises(EvidenceError, match="完整成本集合"):
        validate_strategy_records(
            [record],
            contract(),
            manifest_frozen_at=FROZEN_AT,
            manifest_tracked_content_sha256="test-controlled-content-sha256",
            root=tmp_path,
        )


def test_stale_strategy_evidence_expires_after_192_hours(tmp_path: Path) -> None:
    record = strategy_record(
        tmp_path,
        available_at=DECISION_AT - timedelta(hours=193),
    )
    result = evaluate_allocator(
        contract(),
        [environment_record(tmp_path)],
        [record],
        manifest(),
        decision_at=DECISION_AT,
        root=tmp_path,
    )
    first = result["strategy_eligibility"][0]
    assert first["gates"]["strategy_evidence_fresh_at_decision"] is False
    assert first["eligible"] is False


def test_latest_collecting_record_overrides_old_pass_instead_of_best_record_selection(
    tmp_path: Path,
) -> None:
    old = strategy_record(
        tmp_path,
        evidence_id="OLD_PASS",
        available_at=DECISION_AT - timedelta(hours=3),
    )
    latest = strategy_record(
        tmp_path,
        evidence_id="LATEST_COLLECTING",
        available_at=DECISION_AT - timedelta(hours=1),
        meta_status="FORWARD_COLLECTING",
        evidence_type="TRUE_FORWARD_CONDITIONAL",
    )
    result = evaluate_allocator(
        contract(),
        [environment_record(tmp_path)],
        [old, latest],
        manifest(),
        decision_at=DECISION_AT,
        root=tmp_path,
    )
    first = result["strategy_eligibility"][0]
    assert first["evidence_id"] == "LATEST_COLLECTING"
    assert first["eligible"] is False
    assert result["allocation"]["cash_weight"] == pytest.approx(1.0)


def test_future_strategy_record_is_ignored_at_current_decision(tmp_path: Path) -> None:
    future = strategy_record(
        tmp_path,
        available_at=DECISION_AT + timedelta(hours=1),
    )
    result = evaluate_allocator(
        contract(),
        [environment_record(tmp_path)],
        [future],
        manifest(),
        decision_at=DECISION_AT,
        root=tmp_path,
    )
    assert result["strategy_eligibility"][0]["evidence_id"] is None
    assert result["eligible_strategy_count"] == 0


def test_winner_chasing_field_is_rejected_at_schema_boundary(tmp_path: Path) -> None:
    value = contract()
    record = strategy_record(tmp_path)
    record["recent_rank"] = 1
    with pytest.raises(EvidenceError, match="赢家追逐字段"):
        validate_strategy_records(
            [record],
            value,
            manifest_frozen_at=FROZEN_AT,
            manifest_tracked_content_sha256="test-controlled-content-sha256",
            root=tmp_path,
        )


def test_prefreeze_evaluation_window_can_never_claim_qualification(tmp_path: Path) -> None:
    value = contract()
    record = strategy_record(tmp_path)
    record["evaluation_window_start"] = FROZEN_AT.isoformat()
    record = refresh_strategy_source_binding(tmp_path, record)
    with pytest.raises(EvidenceError, match="冻结时点之前"):
        validate_strategy_records(
            [record],
            value,
            manifest_frozen_at=FROZEN_AT,
            manifest_tracked_content_sha256="test-controlled-content-sha256",
            root=tmp_path,
        )


def test_source_hash_drift_fails_closed(tmp_path: Path) -> None:
    value = contract()
    record = environment_record(tmp_path)
    (tmp_path / record["source_path"]).write_text("来源已经漂移", encoding="utf-8")
    with pytest.raises(EvidenceError, match="SHA-256 不一致"):
        validate_environment_records([record], value, root=tmp_path)


def test_strategy_record_cannot_diverge_from_immutable_evaluator_report(tmp_path: Path) -> None:
    value = contract()
    record = strategy_record(tmp_path)
    record["forward_observation_days"] = 999
    with pytest.raises(EvidenceError, match="evidence_core 不一致"):
        validate_strategy_records(
            [record],
            value,
            manifest_frozen_at=FROZEN_AT,
            manifest_tracked_content_sha256="test-controlled-content-sha256",
            root=tmp_path,
        )


def test_bound_forward_ledger_prefix_cannot_be_truncated(tmp_path: Path) -> None:
    value = contract()
    record = strategy_record(tmp_path)
    ledger = tmp_path / value["inputs"]["forward_observation_ledger_jsonl"]
    ledger.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EvidenceError, match="文件被截断"):
        validate_strategy_records(
            [record],
            value,
            manifest_frozen_at=FROZEN_AT,
            manifest_tracked_content_sha256="test-controlled-content-sha256",
            root=tmp_path,
        )


def test_rejected_frozen_meta_evidence_cannot_be_recovered(tmp_path: Path) -> None:
    value = contract()
    rejected = strategy_record(
        tmp_path,
        evidence_id="REJECTED",
        available_at=DECISION_AT - timedelta(hours=2),
        meta_status="REJECTED_FROZEN",
    )
    recovered = strategy_record(
        tmp_path,
        evidence_id="ILLEGAL_RECOVERY",
        available_at=DECISION_AT - timedelta(hours=1),
    )
    with pytest.raises(EvidenceError, match="恢复同一 V1"):
        validate_strategy_records(
            [rejected, recovered],
            value,
            manifest_frozen_at=FROZEN_AT,
            manifest_tracked_content_sha256="test-controlled-content-sha256",
            root=tmp_path,
        )


def test_batch_evidence_may_share_available_time_across_different_strategies(
    tmp_path: Path,
) -> None:
    value = contract()
    first = strategy_record(tmp_path, evidence_id="BATCH_1")
    second = strategy_record(
        tmp_path,
        evidence_id="BATCH_2",
        strategy_id=EXPECTED_STRATEGY_IDS[1],
    )
    validated = validate_strategy_records(
        [first, second],
        value,
        manifest_frozen_at=FROZEN_AT,
        manifest_tracked_content_sha256="test-controlled-content-sha256",
        root=tmp_path,
    )
    assert len(validated) == 2


def test_duplicate_available_time_for_same_strategy_environment_is_rejected(
    tmp_path: Path,
) -> None:
    value = contract()
    first = strategy_record(tmp_path, evidence_id="DUPLICATE_1")
    second = strategy_record(tmp_path, evidence_id="DUPLICATE_2")
    with pytest.raises(EvidenceError, match="重复 available_at"):
        validate_strategy_records(
            [first, second],
            value,
            manifest_frozen_at=FROZEN_AT,
            manifest_tracked_content_sha256="test-controlled-content-sha256",
            root=tmp_path,
        )


def test_environment_no_view_record_needs_no_fabricated_source_hash(tmp_path: Path) -> None:
    value = contract()
    record = environment_record(tmp_path, data_status="NO_VIEW")
    validated = validate_environment_records([record], value, root=tmp_path)
    result = classify_environment(validated, value, decision_at=DECISION_AT)
    assert result["view_status"] == "NO_VIEW"
    assert result["evidence_id"] == "ENV_1"


def test_risk_reduction_is_immediate_and_not_blocked_by_increase_cap() -> None:
    value = contract()
    previous = {EXPECTED_STRATEGY_IDS[0]: 0.35, "CASH_CNY": 0.65}
    eligibility = [
        {
            "strategy_id": strategy_id,
            "family": next(item["family"] for item in value["strategy_pool"] if item["strategy_id"] == strategy_id),
            "eligible": False,
            "score": 0.0,
        }
        for strategy_id in EXPECTED_STRATEGY_IDS
    ]
    result = allocate_research_risk(eligibility, value, previous_weights=previous)
    assert result["research_risk_weights"][EXPECTED_STRATEGY_IDS[0]] == 0.0
    assert result["cash_weight"] == pytest.approx(1.0)
    assert result["applied_positive_risk_increase"] == 0.0


def test_family_strategy_and_total_caps_hold_for_multiple_eligible_tracks() -> None:
    value = contract()
    family_by_id = {item["strategy_id"]: item["family"] for item in value["strategy_pool"]}
    eligibility = [
        {
            "strategy_id": strategy_id,
            "family": family_by_id[strategy_id],
            "eligible": True,
            "score": 1.0,
        }
        for strategy_id in EXPECTED_STRATEGY_IDS
    ]
    result = allocate_research_risk(eligibility, value)
    weights = result["research_risk_weights"]
    assert result["strategy_weight_total"] <= 0.25 + 1e-12
    assert all(weights[strategy_id] <= 0.35 + 1e-12 for strategy_id in EXPECTED_STRATEGY_IDS)
    family_weights: dict[str, float] = {}
    for strategy_id in EXPECTED_STRATEGY_IDS:
        family = family_by_id[strategy_id]
        family_weights[family] = family_weights.get(family, 0.0) + weights[strategy_id]
    assert all(weight <= 0.50 + 1e-12 for weight in family_weights.values())
    assert weights["CASH_CNY"] >= 0.20 - 1e-12


def test_previous_risk_weights_only_load_from_bound_immutable_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = contract()
    monkeypatch.setattr(run_module, "ROOT", tmp_path)
    receipt_directory = tmp_path / value["outputs"]["immutable_receipt_directory"]
    receipt_directory.mkdir(parents=True, exist_ok=True)
    receipt_id = "PRIOR_RECEIPT"
    receipt_path = receipt_directory / f"{receipt_id}.json"
    weights = {strategy_id: 0.0 for strategy_id in EXPECTED_STRATEGY_IDS}
    weights[EXPECTED_STRATEGY_IDS[0]] = 0.25
    weights["CASH_CNY"] = 0.75
    payload = {
        "receipt_id": receipt_id,
        "immutable_receipt": True,
        "candidate_id": "REGIME_AWARE_STRATEGY_ALLOCATOR_V1",
        "run_status": "SUCCESS",
        "manifest_sha256": "a" * 64,
        "decision_at": (DECISION_AT - timedelta(days=7)).isoformat(),
        "generated_at": (DECISION_AT - timedelta(days=7) + timedelta(minutes=1)).isoformat(),
        "safety": {
            "paper_position_generation": False,
            "shadow_signal_generation": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_or_exchange_connection_enabled": False,
            "live_trading_authorized": False,
        },
        "research_risk_weights": weights,
        "cash_weight": 0.75,
    }
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = run_module.load_previous_receipt_weights(
        receipt_path,
        contract=value,
        manifest_verification={"manifest_sha256": "a" * 64},
        decision_at=DECISION_AT,
    )
    assert loaded == weights
    outside = tmp_path / "arbitrary_status.json"
    outside.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvidenceError, match="不可变回执目录"):
        run_module.load_previous_receipt_weights(
            outside,
            contract=value,
            manifest_verification={"manifest_sha256": "a" * 64},
            decision_at=DECISION_AT,
        )


def test_previous_weights_must_sum_to_one_with_explicit_cash() -> None:
    value = contract()
    family_by_id = {item["strategy_id"]: item["family"] for item in value["strategy_pool"]}
    eligibility = [
        {
            "strategy_id": strategy_id,
            "family": family_by_id[strategy_id],
            "eligible": False,
            "score": 0.0,
        }
        for strategy_id in EXPECTED_STRATEGY_IDS
    ]
    with pytest.raises(EvidenceError, match="合计不等于 1"):
        allocate_research_risk(
            eligibility,
            value,
            previous_weights={EXPECTED_STRATEGY_IDS[0]: 0.25, "CASH_CNY": 0.50},
        )


def test_runner_rejects_future_decision_time_before_any_manifest_read() -> None:
    with pytest.raises(EvidenceError, match="不得晚于当前可得时间"):
        run_module.run(decision_at=datetime.now(timezone.utc) + timedelta(days=1), no_write=True)


def test_current_component_sources_remain_rejected_and_never_authorized() -> None:
    value = contract()
    dependencies = {}
    for item in value["strategy_pool"]:
        for relative in (item["component_manifest"], item["component_status_source"]):
            path = ROOT / relative
            dependencies[relative] = {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    statuses = audit_component_statuses(value, dependencies, root=ROOT)
    assert len(statuses) == 5
    assert all(item["component_status"].startswith("REJECTED_") for item in statuses)
    assert all(item["paper_or_live_authorized"] is False for item in statuses)
    assert all(item["status_relabelled_by_allocator"] is False for item in statuses)
    assert all(item["component_performance_used_for_ranking"] is False for item in statuses)


def test_markdown_makes_cash_and_non_actionability_explicit(tmp_path: Path) -> None:
    result = evaluate_allocator(
        contract(),
        [],
        [],
        manifest(),
        decision_at=DECISION_AT,
        root=tmp_path,
    )
    result["component_statuses"] = []
    markdown = render_markdown(result)
    assert "现金权重：100.00%" in markdown
    assert "不是交易建议或交易授权" in markdown
    assert "使用最近赢家排名：`false`" in markdown
