"""冻结 510300 压力传导危险率 V2 协议与 V1 保留清单。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    canonical_sha256,
    file_evidence,
    git_head,
    normalize_project_relative_path,
    read_json_strict,
    sha256_file,
    validate_authority_state,
    verify_manifest_payload,
)
from research.stress_transmission_hazard_v2 import (
    ACTION_NONE_CONFIRMED,
    ACTION_RESOLVED,
    ACTION_UNRESOLVED,
    MODEL_FEATURES,
    PROGRAM_ID,
    assert_predictive_names_are_industry_free,
)


DEFAULT_CONFIG = ROOT / "config/510300_stress_transmission_hazard_v2.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("V2 协议配置必须是对象")
    return payload


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _validate_file_identity(path: Path, contract: Mapping[str, Any], label: str) -> None:
    if not path.is_file():
        raise EvidenceContractError(f"{label}文件缺失：{path}")
    actual_bytes = path.stat().st_size
    expected_bytes = int(contract["bytes"])
    if actual_bytes != expected_bytes:
        raise EvidenceContractError(
            f"{label}字节数漂移：expected={expected_bytes}, actual={actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    expected_sha256 = str(contract["sha256"]).lower()
    if actual_sha256 != expected_sha256:
        raise EvidenceContractError(
            f"{label}哈希漂移：expected={expected_sha256}, actual={actual_sha256}"
        )


def _validate_exact(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise EvidenceContractError(
            f"{label}发生漂移：expected={expected!r}, actual={actual!r}"
        )


def validate_config(config: Mapping[str, Any]) -> None:
    program = config["program"]
    exact_program = {
        "program_id": PROGRAM_ID,
        "version": "2.0.0",
        "protocol_revision": "FULL_REVIEW_AUTHORITY_20260903_PRE_FEATURE_FREEZE",
        "predecessor_program_id": "510300_ASYMMETRIC_STRESS_HAZARD_V1",
        "predecessor_g2_authoritative_result": "INVALIDATED",
        "predecessor_economic_hypothesis": "NOT_CLEANLY_ADJUDICATED",
        "current_stage": "PROTOCOL_FREEZE_AND_DATA_CONTRACT_REPAIR_ONLY",
        "research_state": "DISCOVERY_ONLY",
        "return_evaluation": "NOT_ALLOWED",
        "position_impact": 0,
        "model_training_allowed_at_current_stage": False,
        "probability_threshold_selection_allowed_at_current_stage": False,
        "portfolio_evaluation_allowed_at_current_stage": False,
        "paper_or_shadow_allowed": False,
        "broker_connection_allowed": False,
        "live_trading_allowed": False,
    }
    for key, expected in exact_program.items():
        _validate_exact(program.get(key), expected, f"program.{key}")
    _validate_exact(
        program.get("executable_assets"),
        ["510300.SH", "CASH_CNY"],
        "可执行资产范围",
    )

    label = config["bad10_label"]
    exact_label = {
        "label_id": "BAD10",
        "execution_clock": "NEXT_TRADING_DAY_OPEN",
        "entry_asset": "510300.SH",
        "horizon_market_days": 10,
        "loss_threshold": -0.04,
        "transaction_cost_in_label": 0.0,
        "same_day_close_execution_allowed": False,
        "future_low_execution_allowed": False,
        "h00300_execution_proxy_allowed": False,
        "threshold_search_allowed": False,
        "horizon_search_allowed": False,
    }
    for key, expected in exact_label.items():
        _validate_exact(label.get(key), expected, f"bad10_label.{key}")

    event = config["event_contract"]
    _validate_exact(
        event.get("merge_rule"),
        "TRANSITIVE_MERGE_WHEN_POSITIVE_HOLDING_INTERVALS_OVERLAP",
        "事件合并规则",
    )
    _validate_exact(
        event.get("positive_origin_training_weight_per_event"),
        1.0,
        "事件正样本权重",
    )
    _validate_exact(
        event.get("event_may_cross_train_calibration_evaluation"),
        False,
        "事件样本隔离",
    )

    return_contract = config["constituent_return_contract"]
    _validate_exact(
        set(return_contract["states"]),
        {
            "TRADED_VALID",
            "OFFICIAL_SUSPENSION",
            "CORPORATE_ACTION_UNRESOLVED",
            "SUPPLIER_MISSING_OR_CONFLICT",
        },
        "成分收益四态",
    )
    _validate_exact(
        return_contract.get("blanket_fill_missing_return_with_zero_allowed"),
        False,
        "缺失收益补零权限",
    )
    _validate_exact(
        return_contract.get("post_adjusted_price_with_future_action_information_allowed"),
        False,
        "后复权未来信息权限",
    )

    coverage = config["coverage_contract"]
    _validate_exact(
        coverage["breadth_and_equal_weight"]["minimum_usable_member_ratio"],
        0.98,
        "成员覆盖门",
    )
    _validate_exact(
        coverage["point_in_time_weights"][
            "when_reliable_weights_exist_minimum_usable_weight_ratio"
        ],
        0.99,
        "点时权重覆盖门",
    )
    _validate_exact(
        coverage["comovement"]["lookback_market_days"],
        20,
        "共同运动窗口",
    )
    _validate_exact(
        coverage["comovement"]["minimum_valid_observations_per_member"],
        15,
        "共同运动有效观察门",
    )
    _validate_exact(
        coverage["comovement"]["minimum_scoreable_point_in_time_member_ratio"],
        0.90,
        "共同运动成员比例门",
    )

    industry = config["industry_contract"]
    _validate_exact(industry.get("predictive_use"), "NOT_ALLOWED", "行业预测用途")
    _validate_exact(industry.get("core_input_present"), False, "行业核心输入")
    _validate_exact(
        industry.get("current_or_effective_interval_backfill_allowed"),
        False,
        "行业历史回填权限",
    )
    serialized_features = json.dumps(
        config["feature_contract"], ensure_ascii=False, sort_keys=True
    ).casefold()
    for token in ("industry", "sector", "shenwan", "申万", "行业"):
        if token.casefold() in serialized_features:
            raise EvidenceContractError(f"feature_contract 含行业依赖：{token}")
    assert_predictive_names_are_industry_free(
        [name for fields in MODEL_FEATURES.values() for name in fields]
    )

    model = config["model_contract"]
    _validate_exact(model.get("l2_penalty"), 1.0, "模型 L2")
    _validate_exact(model.get("hyperparameter_grid_allowed"), False, "超参数网格")
    _validate_exact(model.get("sign_reversal_allowed"), False, "信号反转")
    _validate_exact(
        model.get("primary_incremental_metric"),
        "EVENT_WEIGHTED_LOG_LOSS",
        "主增量指标",
    )
    _validate_exact(model.get("auc_role"), "DESCRIPTION_ONLY", "AUC 角色")
    expected_models = {
        "B0": "UNCONDITIONAL_BAD10_BASE_RATE",
        "B1": "ETF_REALIZED_VOL20_DRAWDOWN20_AND_DOWNSIDE_RETURN5",
        "B2": "T_PLUS_T_TIMES_F",
        "B3": "T_PLUS_T_TIMES_F_PLUS_T_TIMES_M",
    }
    _validate_exact(model.get("fixed_models"), expected_models, "固定模型集合")
    _validate_exact(
        set(model["coefficient_constraints"].values()),
        {"NONNEGATIVE"},
        "模型符号约束",
    )

    gates = config["gates"]
    g1 = gates["G1_DATA_AND_EVENTS"]
    _validate_exact(
        g1.get("minimum_independent_events_for_mechanism_discovery"),
        30,
        "G1 机制事件数",
    )
    _validate_exact(
        g1.get("minimum_independent_events_for_full_three_coefficient_model"),
        40,
        "G1 完整模型事件数",
    )
    _validate_exact(g1.get("minimum_non_event_risk_days"), 750, "G1 非事件日")
    g2 = gates["G2_STRUCTURAL_INCREMENT"]
    _validate_exact(g2.get("comparison"), "B2_MINUS_B1", "G2 嵌套比较")
    _validate_exact(
        g2.get("metric"), "EVENT_WEIGHTED_LOG_LOSS_IMPROVEMENT", "G2 指标"
    )
    _validate_exact(g2.get("same_direction_eras_required"), 3, "G2 时代门")
    _validate_exact(g2.get("one_sided_confidence"), 0.90, "G2 Bootstrap 置信度")
    g3 = gates["G3_MACRO_INCREMENT"]
    _validate_exact(g3.get("comparison"), "B3_MINUS_B2", "G3 嵌套比较")
    _validate_exact(g3.get("beta3_may_reverse"), False, "G3 宏观系数反转权限")
    g4 = gates["G4_DECISION_CAPABILITY"]
    expected_g4 = {
        "event_recall_minimum": 0.80,
        "median_lead_full_market_days_minimum": 1,
        "alert_day_share_maximum": 0.15,
        "independent_alerts_per_year_maximum": 8,
        "median_false_alert_run_market_days_maximum": 3,
    }
    for key, expected in expected_g4.items():
        _validate_exact(g4.get(key), expected, f"G4.{key}")
    g5 = gates["G5_PORTFOLIO_VALUE"]
    _validate_exact(g5.get("baseline_cost_net_sharpe_minimum"), 1.20, "G5 夏普门")
    _validate_exact(g5.get("annual_round_trips_maximum"), 8, "G5 年往返门")
    g7 = gates["G7_STRICT_FORWARD"]
    _validate_exact(g7.get("first_review_minimum_market_days"), 252, "G7 初审日")
    _validate_exact(g7.get("first_review_minimum_independent_events"), 5, "G7 初审事件")
    _validate_exact(g7.get("formal_qualification_minimum_market_days"), 504, "G7 正式日")
    _validate_exact(g7.get("formal_qualification_minimum_independent_events"), 10, "G7 正式事件")

    position = config["mechanical_position_experiment_reserved_for_post_G4"]
    _validate_exact(position.get("probability_threshold"), None, "当前概率阈值")
    _validate_exact(position.get("defensive_period_market_days"), 10, "固定现金期")
    _validate_exact(
        position.get("trained_recovery_model_allowed_in_first_experiment"),
        False,
        "首次恢复模型权限",
    )
    _validate_exact(
        config["forward_contract"].get("position_impact"), 0, "严格前向仓位影响"
    )

    forbidden = set(config["forbidden_actions"])
    required_forbidden = {
        "RERUN_V1_G2_WITH_INDUSTRY_TEMPORAL_GATE_DISABLED",
        "FILL_MISSING_CONSTITUENT_RETURNS_WITH_ZERO",
        "REVERSE_LOW_AUC_SIGNAL",
        "SEARCH_BAD10_THRESHOLD_OR_HORIZON",
        "SELECT_PROBABILITY_THRESHOLD_BY_FINAL_SHARPE",
        "GENERATE_PORTFOLIO_CURVE_BEFORE_G0_THROUGH_G4_PASS",
        "GENERATE_POSITION_ORDER_BROKER_OR_LIVE_ACTION",
    }
    if not required_forbidden.issubset(forbidden):
        missing = sorted(required_forbidden.difference(forbidden))
        raise EvidenceContractError(f"禁止动作清单缺项：{missing}")


def verify_external_and_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    authority_contract = config["authority_contract"]
    authority_path = _project_path(str(authority_contract["path"]))
    _validate_file_identity(authority_path, authority_contract, "研究权威状态")
    authority = read_json_strict(authority_path)
    if not isinstance(authority, dict):
        raise EvidenceContractError("研究权威状态必须是对象")
    validate_authority_state(authority)
    _validate_exact(
        authority.get("authority_id"),
        authority_contract["authority_id"],
        "研究权威 identity",
    )

    adjudication = config["external_adjudication"]
    source_path = Path(str(adjudication["source_path"]))
    _validate_file_identity(
        source_path,
        {
            "bytes": adjudication["source_bytes"],
            "sha256": adjudication["source_sha256"],
        },
        "外部复核来源",
    )
    decision_path = _project_path(str(adjudication["machine_decision_path"]))
    decision = read_json_strict(decision_path)
    if not isinstance(decision, dict):
        raise EvidenceContractError("外部复核裁决 JSON 必须是对象")
    _validate_exact(
        decision.get("adjudication_id"),
        adjudication["decision_id"],
        "外部裁决 identity",
    )
    _validate_exact(
        decision["predecessor"].get("g2_authoritative_result"),
        "INVALIDATED",
        "V1 G2 权威裁决",
    )
    _validate_exact(
        decision["successor"].get("authorization"),
        "PROTOCOL_FREEZE_AND_DISCOVERY_ONLY",
        "V2 授权",
    )
    return {
        "authority": file_evidence(authority_path, project_root=ROOT),
        "external_source": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
        },
        "machine_decision": file_evidence(decision_path, project_root=ROOT),
        "human_decision": file_evidence(
            _project_path(str(adjudication["human_decision_path"])),
            project_root=ROOT,
        ),
    }


def verify_input_identities(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for input_id, contract in config["input_identity_registry"].items():
        if not isinstance(contract, Mapping):
            raise EvidenceContractError(f"输入 {input_id} 的契约不是对象")
        path = _project_path(str(contract["path"]))
        _validate_file_identity(path, contract, f"输入 {input_id}")
        evidence[str(input_id)] = file_evidence(path, project_root=ROOT)
        evidence[str(input_id)]["status"] = str(contract["status"])
    for input_id, contract in config["quarantined_legacy_inputs"].items():
        if not isinstance(contract, Mapping) or "path" not in contract:
            continue
        path = _project_path(str(contract["path"]))
        _validate_file_identity(path, contract, f"隔离输入 {input_id}")
        evidence[f"quarantined::{input_id}"] = file_evidence(path, project_root=ROOT)
        evidence[f"quarantined::{input_id}"]["status"] = str(contract["status"])
    return evidence


def _hash_file_group(paths: list[str]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for raw in paths:
        relative = normalize_project_relative_path(str(raw))
        path = _project_path(relative)
        if not path.is_file():
            raise EvidenceContractError(f"冻结文件缺失：{relative}")
        evidence[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return evidence


def collect_v1_preservation_files(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    contract = config["v1_preservation_contract"]
    excluded = [str(value).casefold() for value in contract["exclude_name_fragments"]]
    paths: set[Path] = set()
    for pattern in contract["include_patterns"]:
        normalized_pattern = str(pattern).replace("\\", "/")
        for path in ROOT.glob(normalized_pattern):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if any(fragment in relative.casefold() for fragment in excluded):
                continue
            paths.add(path)
    package_path = _project_path(str(contract["package_path"]))
    _validate_file_identity(
        package_path,
        {
            "bytes": contract["package_bytes"],
            "sha256": contract["package_sha256"],
        },
        "V1 原始交付包",
    )
    paths.add(package_path)
    if not paths:
        raise EvidenceContractError("V1 保留清单为空")
    evidence: dict[str, dict[str, Any]] = {}
    for path in sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix().casefold()):
        relative = path.relative_to(ROOT).as_posix()
        evidence[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    required_suffixes = {
        "config/510300_asymmetric_stress_hazard_v1.yaml",
        "research/asymmetric_stress_hazard_g2_mechanism_v1.py",
        "reports/research/510300_asymmetric_stress_hazard_v1_g2_mechanism.json",
        "reports/audit/510300_asymmetric_stress_hazard_v1_g2_mechanism_receipt.json",
        "deliverables/510300_ASYMMETRIC_STRESS_HAZARD_V1_GPT_PRO_FULL_REVIEW_20260903.zip",
    }
    missing = sorted(required_suffixes.difference(evidence))
    if missing:
        raise EvidenceContractError(f"V1 保留清单缺少关键文件：{missing}")
    return evidence


def _output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    freeze_contract = config["freeze_contract"]
    return {
        "manifest": _project_path(str(freeze_contract["manifest_output"])),
        "freeze_receipt": _project_path(str(freeze_contract["freeze_receipt_output"])),
        "v1_preservation_manifest": _project_path(
            str(config["v1_preservation_contract"]["output_manifest"])
        ),
    }


def freeze(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    outputs = _output_paths(config)
    existing = [str(path) for path in outputs.values() if path.exists()]
    if existing:
        raise EvidenceContractError(f"冻结输出已存在，禁止覆盖：{existing}")

    authority_and_decision = verify_external_and_authority(config)
    input_evidence = verify_input_identities(config)
    freeze_contract = config["freeze_contract"]
    implementation_files = _hash_file_group(
        [str(value) for value in freeze_contract["implementation_files"]]
    )
    governance_files = _hash_file_group(
        [str(value) for value in freeze_contract["governance_files"]]
    )
    v1_files = collect_v1_preservation_files(config)
    frozen_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()

    preservation_payload: dict[str, Any] = {
        "preservation_id": "510300_ASH_V1_IMMUTABLE_REFERENCE_20260903",
        "created_at": frozen_at,
        "predecessor_program_id": "510300_ASYMMETRIC_STRESS_HAZARD_V1",
        "original_output": "PRESERVE_IMMUTABLE",
        "g2_authoritative_result": "INVALIDATED",
        "economic_hypothesis": "NOT_CLEANLY_ADJUDICATED",
        "preservation_method": "APPEND_ONLY_SHA256_DRIFT_DETECTION_NO_SOURCE_MUTATION",
        "source_files_modified_by_preservation": False,
        "file_count": len(v1_files),
        "files": v1_files,
        "external_adjudication": authority_and_decision,
        "does_not_authorize_prediction_or_trading": True,
    }
    preservation_payload["manifest_payload_sha256"] = canonical_sha256(
        preservation_payload
    )
    atomic_write_json_new(outputs["v1_preservation_manifest"], preservation_payload)

    config_relative = config_path.relative_to(ROOT).as_posix()
    manifest: dict[str, Any] = {
        "program_id": PROGRAM_ID,
        "version": config["program"]["version"],
        "protocol_revision": config["program"]["protocol_revision"],
        "status": "FROZEN_BEFORE_V2_FEATURE_VALUE_CONSTRUCTION",
        "frozen_at": frozen_at,
        "config_file": config_relative,
        "config_bytes": config_path.stat().st_size,
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "implementation_files": implementation_files,
        "governance_files": governance_files,
        "authority_and_external_decision": authority_and_decision,
        "input_identities": input_evidence,
        "v1_preservation_manifest": file_evidence(
            outputs["v1_preservation_manifest"], project_root=ROOT
        ),
        "git_state": {
            "head_at_freeze": git_head(ROOT),
            "clean_replay_claimed": False,
            "g0_status": "PENDING_COMMITTED_CLEAN_ENVIRONMENT_REPLAY",
        },
        "bad10_contract": config["bad10_label"],
        "event_contract": config["event_contract"],
        "constituent_return_contract": config["constituent_return_contract"],
        "coverage_contract": config["coverage_contract"],
        "industry_contract": config["industry_contract"],
        "feature_contract": config["feature_contract"],
        "model_contract": config["model_contract"],
        "gates": config["gates"],
        "forbidden_actions": config["forbidden_actions"],
        "research_state": "DISCOVERY_ONLY",
        "return_evaluation": "NOT_ALLOWED",
        "model_position_target": "UNSET",
        "order_authorization": "NOT_AUTHORIZED",
        "live_trading_authorized": False,
        "position_impact": 0,
        "feature_values_constructed": False,
        "model_trained": False,
        "probability_threshold_selected": False,
        "portfolio_results_read": False,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(outputs["manifest"], manifest)

    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_PROTOCOL_FREEZE_20260903",
        "created_at": frozen_at,
        "status": "PASS_PROTOCOL_CONFIG_CODE_AND_HASH_FREEZE_G0_NOT_YET_PASSED",
        "program_id": PROGRAM_ID,
        "manifest": file_evidence(outputs["manifest"], project_root=ROOT),
        "v1_preservation_manifest": file_evidence(
            outputs["v1_preservation_manifest"], project_root=ROOT
        ),
        "frozen_file_count": len(implementation_files) + len(governance_files),
        "preserved_v1_file_count": len(v1_files),
        "input_identity_count": len(input_evidence),
        "checks": [
            "CONFIG_INVARIANTS",
            "AUTHORITY_SEPARATION",
            "EXTERNAL_ADJUDICATION_IDENTITY",
            "INPUT_BYTES_AND_SHA256",
            "V1_APPEND_ONLY_PRESERVATION_HASHES",
            "INDUSTRY_CORE_INPUT_ABSENT",
            "FOUR_STATE_CONTRACT_FROZEN",
            "PORTFOLIO_AND_TRADING_PROHIBITIONS_FROZEN",
        ],
        "not_claimed": [
            "G0_CLEAN_ENVIRONMENT_REPLAY_PASS",
            "HISTORICAL_FOUR_STATE_DATA_COVERAGE_PASS",
            "G1_THROUGH_G7_PASS",
            "PREDICTION_VALIDITY",
            "PORTFOLIO_VALUE",
            "TRADING_AUTHORIZATION",
        ],
        "security_audit_performed": False,
        "privacy_scan_performed": False,
        "full_repository_regression_performed": False,
        "research_state": "DISCOVERY_ONLY",
        "return_evaluation": "NOT_ALLOWED",
        "position_impact": 0,
    }
    atomic_write_json_new(outputs["freeze_receipt"], receipt)
    print(
        "V2 协议已冻结；"
        f"manifest={outputs['manifest'].relative_to(ROOT).as_posix()}；"
        "G0 仍待提交后的干净环境重放。"
    )
    return manifest


def verify_frozen_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    outputs = _output_paths(config)
    manifest = read_json_strict(outputs["manifest"])
    if not isinstance(manifest, dict):
        raise EvidenceContractError("V2 manifest 必须是对象")
    verify_manifest_payload(manifest)
    if manifest.get("program_id") != PROGRAM_ID:
        raise EvidenceContractError("V2 manifest program_id 不一致")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise EvidenceContractError("V2 配置在冻结后漂移")
    for group_name in ("implementation_files", "governance_files"):
        group = manifest.get(group_name)
        if not isinstance(group, Mapping):
            raise EvidenceContractError(f"V2 manifest 缺少 {group_name}")
        for relative, evidence in group.items():
            path = _project_path(str(relative))
            if not isinstance(evidence, Mapping):
                raise EvidenceContractError(f"{relative} 的文件证据非法")
            _validate_file_identity(path, evidence, f"冻结文件 {relative}")
    preservation_path = outputs["v1_preservation_manifest"]
    preservation = read_json_strict(preservation_path)
    if not isinstance(preservation, dict):
        raise EvidenceContractError("V1 保留 manifest 必须是对象")
    verify_manifest_payload(preservation)
    for relative, evidence in preservation["files"].items():
        _validate_file_identity(
            _project_path(str(relative)), evidence, f"V1 保留文件 {relative}"
        )
    verify_external_and_authority(config)
    verify_input_identities(config)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="冻结 510300 压力传导危险率 V2 协议"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="只验证既有冻结 manifest，不创建文件",
    )
    args = parser.parse_args()
    try:
        if args.verify_only:
            verify_frozen_manifest(args.config)
            print("V2 冻结 manifest、文件哈希与 V1 保留清单验证通过。")
        else:
            freeze(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError) as exc:
        print(f"V2 协议冻结失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
