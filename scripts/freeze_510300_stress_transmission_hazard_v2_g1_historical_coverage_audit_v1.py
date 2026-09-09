"""冻结 V2 G1 历史覆盖审计 V1；冻结动作不读取标签表或收益路径。"""

from __future__ import annotations

import argparse
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
    normalize_project_relative_path,
    read_json_strict,
    sha256_file,
    verify_manifest_payload,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_g1_event_prediction_admission_v1_0_1 import (
    verify_frozen_manifest as verify_g1_parent_manifest,
)


EXECUTION_ID = (
    "510300_STRESS_TRANSMISSION_HAZARD_V2_"
    "G1_HISTORICAL_COVERAGE_AUDIT_V1"
)
DEFAULT_CONFIG = (
    ROOT
    / "config/510300_stress_transmission_hazard_v2_g1_historical_coverage_audit_v1.yaml"
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("历史覆盖审计配置必须是对象")
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise EvidenceContractError(
            f"{label}发生漂移：expected={expected!r}, actual={actual!r}"
        )


def _validate_file_identity(contract: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = _project_path(str(contract["path"]))
    if not path.is_file():
        raise EvidenceContractError(f"{label}不存在：{path}")
    actual = file_evidence(path, project_root=ROOT)
    expected = {
        "path": normalize_project_relative_path(str(contract["path"])).replace("\\", "/"),
        "bytes": int(contract["bytes"]),
        "sha256": str(contract["sha256"]),
    }
    if actual != expected:
        raise EvidenceContractError(f"{label}文件身份漂移：{path}")
    return actual


def _verify_self_digest(payload: Mapping[str, Any], field: str, label: str) -> None:
    expected = str(payload.get(field, ""))
    body = dict(payload)
    body.pop(field, None)
    if canonical_sha256(body) != expected:
        raise EvidenceContractError(f"{label}自身摘要失败")


def validate_config(config: Mapping[str, Any]) -> None:
    program = config["program"]
    _require_equal(program.get("program_id"), "510300_STRESS_TRANSMISSION_HAZARD_V2", "项目 identity")
    _require_equal(program.get("execution_id"), EXECUTION_ID, "执行 identity")
    _require_equal(program.get("version"), "1.0.0", "版本")
    _require_equal(program.get("stage_id"), "G1_POST_STOP_HISTORICAL_COVERAGE_AUDIT", "阶段")
    _require_equal(program.get("lifecycle"), "POST_G1_DIAGNOSTIC_ONLY", "生命周期")
    _require_equal(program.get("research_state"), "DISCOVERY_ONLY", "研究状态")
    _require_equal(program.get("parent_g1_gate_may_be_promoted"), False, "G1 提升权限")
    _require_equal(program.get("return_evaluation"), "NOT_ALLOWED", "收益权限")
    _require_equal(program.get("model_training_allowed"), False, "模型训练权限")
    _require_equal(program.get("probability_generation_allowed"), False, "概率权限")
    _require_equal(program.get("portfolio_evaluation_allowed"), False, "组合权限")
    _require_equal(program.get("paper_or_shadow_allowed"), False, "Paper/Shadow 权限")
    _require_equal(program.get("broker_connection_allowed"), False, "经纪商权限")
    _require_equal(program.get("position_impact"), 0, "仓位影响")
    _require_equal(program.get("executable_assets"), ["510300.SH", "CASH_CNY"], "资产范围")
    _require_equal(program.get("observation_start"), "2015-01-05", "观察起点")
    _require_equal(program.get("historical_cutoff"), "2026-08-14", "历史截止日")
    _require_equal(program.get("data_after_historical_cutoff_allowed"), False, "截止日后数据权限")

    audit = config["historical_audit_contract"]
    checks = {
        "individual_return_lookback_market_days": 20,
        "tail_volatility_lookback_market_days": 60,
        "tail_return_lookback_market_days": 5,
        "member_coverage_minimum": 0.98,
        "comovement_member_ratio_minimum": 0.90,
        "expected_members_per_day": 300,
        "mechanism_minimum_independent_events": 30,
        "full_model_minimum_independent_events": 40,
        "minimum_non_event_risk_days": 750,
        "public_price_history_before_index_entry_allowed": True,
        "membership_identity_backfill_allowed": False,
        "comovement_may_count_nonmember_days": False,
        "all_events_required": True,
        "provider_suspend_flag_may_promote_state": False,
        "missing_return_zero_fill_allowed": False,
        "source_reclassification_in_this_stage_allowed": False,
        "feature_rebuild_in_this_stage_allowed": False,
        "g1_recalculation_in_this_stage_allowed": False,
    }
    for key, expected in checks.items():
        _require_equal(audit.get(key), expected, key)
    _require_equal(
        audit.get("prehistory_comparison_scope"),
        "INDIVIDUAL_RETURN20_AND_TAIL60_SCOREABILITY_ONLY",
        "入指前历史比较范围",
    )
    _require_equal(
        audit.get("diagnostic_label"),
        "NON_ADMISSIBLE_DIAGNOSTIC_ONLY",
        "诊断权限标签",
    )
    _require_equal(
        audit.get("official_suspension_acceptance_rule"),
        "EXCHANGE_OFFICIAL_INTERVAL_OR_ANNOUNCEMENT_EVIDENCE_REQUIRED",
        "官方停牌接受规则",
    )
    _require_equal(
        audit.get("next_allowed_step"),
        "FREEZE_OFFICIAL_HISTORICAL_SUSPENSION_SOURCE_PROBE_AND_PREHISTORY_INDEX_CORRECTION_REVIEW",
        "下一允许步骤",
    )
    _require_equal(
        audit.get("deterministic_best_origin_order"),
        [
            "current_b2_blocker",
            "b1_blocker",
            "tail_member_deficit",
            "return20_member_deficit",
            "comovement_member_deficit",
            "origin_date",
        ],
        "代表原点顺序",
    )

    boundary = config["label_and_performance_boundary"]
    _require_equal(boundary.get("parent_actual_label_artifact_was_read"), True, "父标签状态")
    _require_equal(
        boundary.get("this_stage_may_read_only_parent_minimal_label_columns"),
        ["origin_date", "bad10", "event_id"],
        "本阶段最小标签列",
    )
    for key in (
        "minimum_path_return_read_allowed",
        "first_breach_date_read_allowed",
        "horizon_return_read_allowed",
        "label_definition_change_allowed",
        "label_search_or_rescue_allowed",
        "prediction_metric_generation_allowed",
        "performance_read_allowed",
    ):
        _require_equal(boundary.get(key), False, key)

    expected_parent = config["expected_parent_state"]
    _require_equal(
        expected_parent.get("G1_DATA_AND_EVENTS"),
        "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY",
        "父 G1 状态",
    )
    for key, value in (
        ("total_independent_event_count", 58),
        ("total_positive_origin_count", 463),
        ("total_non_event_risk_day_count", 2350),
        ("b2_identifiable_event_count", 23),
        ("b3_identifiable_event_count", 23),
    ):
        _require_equal(expected_parent.get(key), value, key)
    _require_equal(expected_parent.get("mechanism_discovery_prerequisite_passed"), False, "父机制门")
    _require_equal(expected_parent.get("full_three_coefficient_model_prerequisite_passed"), False, "父完整模型门")

    allowed_sample_columns = set(config["inputs"]["g1_sample_eligibility"]["read_columns"])
    forbidden_read = {
        "minimum_path_return",
        "first_breach_date",
        "cash_dividend_per_share_in_horizon",
        "forward_return",
        "horizon_return",
    }
    if allowed_sample_columns.intersection(forbidden_read):
        raise EvidenceContractError("样本输入列越过最小标签读取边界")

    required_forbidden = {
        "USE_ANY_OBSERVATION_AFTER_2026_08_14",
        "REOPEN_OR_PROMOTE_G1_FROM_A_DIAGNOSTIC_UPPER_BOUND",
        "TREAT_MISSING_ROW_OR_PROVIDER_SUSPEND_FLAG_AS_OFFICIAL_SUSPENSION",
        "FILL_MISSING_OR_UNRESOLVED_RETURN_WITH_ZERO",
        "READ_MINIMUM_PATH_RETURN_FIRST_BREACH_DATE_OR_HORIZON_RETURN",
        "TRAIN_OR_PREDICT_B0_B1_B2_OR_B3",
        "READ_OR_GENERATE_PORTFOLIO_NAV_RETURN_SHARPE_OR_DRAWDOWN",
        "CREATE_POSITION_ORDER_BROKER_PAPER_SHADOW_OR_LIVE_ACTION",
    }
    forbidden = set(config["forbidden_actions"])
    if not required_forbidden.issubset(forbidden):
        raise EvidenceContractError(
            f"禁止动作缺项：{sorted(required_forbidden.difference(forbidden))}"
        )
    outputs = list(config["outputs"].values())
    if len(outputs) != len(set(outputs)):
        raise EvidenceContractError("历史覆盖审计输出路径重复")


def verify_immutable_parent(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    verify_g1_parent_manifest()
    evidence = {
        name: _validate_file_identity(contract, f"不可变父链 {name}")
        for name, contract in config["immutable_parent"].items()
    }
    parent = config["immutable_parent"]
    status = read_json_strict(_project_path(parent["g1_terminal_status"]["path"]))
    if not isinstance(status, dict):
        raise EvidenceContractError("父 G1 状态必须是对象")
    _verify_self_digest(status, "status_payload_sha256", "父 G1 状态")
    _require_equal(status.get("G1_DATA_AND_EVENTS"), parent["g1_terminal_status"]["required_g1_state"], "父 G1 状态")
    _require_equal(status.get("branch_state"), parent["g1_terminal_status"]["required_branch_state"], "父分支状态")
    _require_equal(status.get("next_allowed_step"), parent["g1_terminal_status"]["required_next_step"], "父下一步")
    _require_equal(status.get("model_trained"), False, "父模型训练")
    _require_equal(status.get("return_evaluation"), "NOT_ALLOWED", "父收益权限")
    _require_equal(status.get("position_impact"), 0, "父仓位影响")

    replay = read_json_strict(_project_path(parent["g1_clean_replay_receipt"]["path"]))
    if not isinstance(replay, dict):
        raise EvidenceContractError("父 G1 重放收据必须是对象")
    _verify_self_digest(replay, "receipt_payload_sha256", "父 G1 重放收据")
    _require_equal(replay.get("status"), parent["g1_clean_replay_receipt"]["required_status"], "父重放状态")
    _require_equal(replay.get("model_trained"), False, "父重放模型训练")
    _require_equal(replay.get("portfolio_or_performance_artifact_read"), False, "父重放绩效读取")

    mft_path = _project_path(parent["mft_feature_config"]["path"])
    with mft_path.open("r", encoding="utf-8") as handle:
        mft = yaml.safe_load(handle)
    history = mft["new_member_history_contract"]
    _require_equal(history["price_history_before_index_entry"]["allowed"], True, "入指前公开价格历史权限")
    _require_equal(history["price_history_before_index_entry"]["future_price_allowed"], False, "未来价格权限")
    _require_equal(history["price_history_before_index_entry"]["synthetic_history_allowed"], False, "合成历史权限")
    _require_equal(history["comovement_history"]["scoring_days"], "POINT_IN_TIME_MEMBER_DAYS_WITH_VALID_MEMBER_AND_LEAVE_ONE_OUT_PEER_RETURNS_ONLY", "共同运动成员日规则")
    return evidence


def verify_registered_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: _validate_file_identity(contract, f"冻结输入 {name}")
        for name, contract in config["inputs"].items()
    }


def _hash_group(paths: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in paths:
        relative = normalize_project_relative_path(raw)
        path = _project_path(relative)
        if not path.is_file():
            raise EvidenceContractError(f"待冻结文件不存在：{relative}")
        evidence = file_evidence(path, project_root=ROOT)
        evidence.pop("path")
        result[relative] = evidence
    return result


def _freeze_outputs(config: Mapping[str, Any]) -> tuple[Path, Path]:
    freeze_contract = config["freeze_contract"]
    return (
        _project_path(str(freeze_contract["manifest_output"])),
        _project_path(str(freeze_contract["freeze_receipt_output"])),
    )


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, receipt_path = _freeze_outputs(config)
    existing = [str(path) for path in (manifest_path, receipt_path) if path.exists()]
    if existing:
        raise EvidenceContractError(f"历史覆盖审计冻结输出已存在：{existing}")
    output_paths = [_project_path(str(path)) for path in config["outputs"].values()]
    existing_outputs = [str(path) for path in output_paths if path.exists()]
    if existing_outputs:
        raise EvidenceContractError(f"历史覆盖审计业务输出已存在：{existing_outputs}")

    parent = verify_immutable_parent(config)
    inputs = verify_registered_inputs(config)
    freeze_contract = config["freeze_contract"]
    implementation = _hash_group([str(path) for path in freeze_contract["implementation_files"]])
    governance = _hash_group([str(path) for path in freeze_contract["governance_files"]])
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest: dict[str, Any] = {
        "execution_id": EXECUTION_ID,
        "version": "1.0.0",
        "status": "FROZEN_POST_G1_HISTORICAL_COVERAGE_AUDIT_ONLY",
        "frozen_at": now,
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_bytes": config_path.stat().st_size,
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "immutable_parent": parent,
        "inputs": inputs,
        "historical_audit_contract": config["historical_audit_contract"],
        "label_and_performance_boundary": config["label_and_performance_boundary"],
        "expected_parent_state": config["expected_parent_state"],
        "implementation_files": implementation,
        "governance_files": governance,
        "outputs": config["outputs"],
        "forbidden_actions": config["forbidden_actions"],
        "parent_actual_label_artifact_read": True,
        "this_freeze_actual_label_artifact_parsed": False,
        "actual_future_path_value_read": False,
        "observation_after_historical_cutoff_read": False,
        "model_trained": False,
        "performance_artifact_read": False,
        "g1_gate_promoted": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_G1_HISTORICAL_COVERAGE_AUDIT_V1_FREEZE",
        "created_at": now,
        "status": "PASS_POST_G1_HISTORICAL_ONLY_AUDIT_FROZEN",
        "manifest": file_evidence(manifest_path, project_root=ROOT),
        "immutable_parent_verified": True,
        "registered_inputs_verified": True,
        "this_freeze_actual_label_artifact_parsed": False,
        "observation_after_historical_cutoff_read": False,
        "model_trained": False,
        "performance_artifact_read": False,
        "g1_gate_promoted": False,
        "position_impact": 0,
    }
    atomic_write_json_new(receipt_path, receipt)
    print("G1 历史覆盖审计 V1 已冻结；父 G1 仍保持停止和 NO_VIEW。")
    return manifest


def verify_frozen_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, _ = _freeze_outputs(config)
    manifest = read_json_strict(manifest_path)
    if not isinstance(manifest, dict):
        raise EvidenceContractError("历史覆盖审计 manifest 必须是对象")
    verify_manifest_payload(manifest)
    _require_equal(manifest.get("execution_id"), EXECUTION_ID, "执行 identity")
    _require_equal(manifest.get("config_sha256"), sha256_file(config_path), "配置哈希")
    _require_equal(manifest.get("g1_gate_promoted"), False, "G1 提升状态")
    verify_immutable_parent(config)
    verify_registered_inputs(config)
    for group_name in ("implementation_files", "governance_files"):
        group = manifest.get(group_name)
        if not isinstance(group, Mapping):
            raise EvidenceContractError(f"manifest 缺少 {group_name}")
        for relative, metadata in group.items():
            _validate_file_identity(
                {
                    "path": relative,
                    "bytes": metadata["bytes"],
                    "sha256": metadata["sha256"],
                },
                f"冻结文件 {relative}",
            )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_only:
            verify_frozen_manifest(args.config)
            print("历史覆盖审计 manifest、父链和输入身份验证通过。")
        else:
            freeze(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"历史覆盖审计冻结失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
