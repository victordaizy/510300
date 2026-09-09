"""冻结 V2 G1 准入 V1.0.1 的文件身份比较修正。"""

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
from scripts.freeze_510300_stress_transmission_hazard_v2_g1_event_prediction_admission_v1 import (
    verify_frozen_manifest as verify_original_manifest,
)


EXECUTION_ID = (
    "510300_STRESS_TRANSMISSION_HAZARD_V2_"
    "G1_EVENT_PREDICTION_ADMISSION_V1_0_1"
)
DEFAULT_CONFIG = (
    ROOT
    / "config/510300_stress_transmission_hazard_v2_g1_event_prediction_admission_v1_0_1.yaml"
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("G1 V1.0.1 配置必须是对象")
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
        "path": normalize_project_relative_path(str(contract["path"])) .replace("\\", "/"),
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
    _require_equal(program.get("version"), "1.0.1", "版本")
    _require_equal(program.get("stage_id"), "G1_EVENT_PREDICTION_ADMISSION_CLEAN_REPLAY_CORRECTION", "阶段")
    _require_equal(program.get("research_state"), "DISCOVERY_ONLY", "研究状态")
    _require_equal(program.get("return_evaluation"), "NOT_ALLOWED", "收益权限")
    _require_equal(program.get("model_training_allowed"), False, "模型训练权限")
    _require_equal(program.get("probability_generation_allowed"), False, "概率生成权限")
    _require_equal(program.get("portfolio_evaluation_allowed"), False, "组合权限")
    _require_equal(program.get("position_impact"), 0, "仓位影响")
    _require_equal(program.get("executable_assets"), ["510300.SH", "CASH_CNY"], "资产范围")

    correction = config["correction_contract"]
    _require_equal(correction.get("scope"), "FILE_EVIDENCE_COMPARISON_SUBSET_ONLY", "修正范围")
    _require_equal(correction.get("exact_file_identity_fields"), ["path", "bytes", "sha256"], "身份字段")
    _require_equal(correction.get("non_identity_fields_verified_separately"), ["row_count", "persisted_semantic_sha256"], "独立元数据字段")
    for key in (
        "table_values_changed",
        "table_files_rewritten",
        "feature_formula_changed",
        "b1_formula_changed",
        "label_definition_changed",
        "common_sample_rule_changed",
        "event_weight_rule_changed",
        "g1_thresholds_changed",
        "actual_future_path_value_read_allowed",
        "model_training_allowed",
        "prediction_metric_generation_allowed",
        "performance_read_allowed",
    ):
        _require_equal(correction.get(key), False, key)

    expected = config["expected_gate_result"]
    _require_equal(expected.get("G1_DATA_AND_EVENTS"), "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY", "预期 G1 状态")
    _require_equal(expected.get("mechanism_discovery_prerequisite_passed"), False, "预期机制门")
    _require_equal(expected.get("full_three_coefficient_model_prerequisite_passed"), False, "预期完整模型门")
    _require_equal(expected.get("next_allowed_step"), "STOP_V2_NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY", "预期下一步")
    _require_equal(expected.get("b2_identifiable_event_count"), 23, "预期 B2 事件数")
    _require_equal(expected.get("b3_identifiable_event_count"), 23, "预期 B3 事件数")

    forbidden = set(config["forbidden_actions"])
    required = {
        "MODIFY_OR_OVERWRITE_V1_CONFIG_MANIFEST_CODE_TEST_OUTPUT_OR_FAILURE_RECEIPT",
        "CHANGE_ANY_G1_INPUT_TABLE_OR_VALUE",
        "CHANGE_B1_LABEL_COMMON_SAMPLE_EVENT_WEIGHT_OR_GATE_RULE",
        "READ_MINIMUM_PATH_RETURN_OR_OTHER_ACTUAL_FUTURE_PATH_VALUE",
        "TRAIN_OR_PREDICT_B0_B1_B2_OR_B3",
        "GENERATE_PREDICTION_OR_PORTFOLIO_METRICS",
        "CREATE_POSITION_ORDER_BROKER_PAPER_SHADOW_OR_LIVE_ACTION",
    }
    if not required.issubset(forbidden):
        raise EvidenceContractError(
            f"V1.0.1 禁止动作缺项：{sorted(required.difference(forbidden))}"
        )
    outputs = list(config["outputs"].values())
    if len(outputs) != len(set(outputs)):
        raise EvidenceContractError("V1.0.1 输出路径重复")


def verify_immutable_parent(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    verify_original_manifest()
    parent = config["immutable_parent"]
    evidence = {
        name: _validate_file_identity(contract, f"不可变父链 {name}")
        for name, contract in parent.items()
    }
    build = read_json_strict(_project_path(parent["original_build_receipt"]["path"]))
    if not isinstance(build, dict):
        raise EvidenceContractError("V1 构建收据必须是对象")
    _verify_self_digest(build, "receipt_payload_sha256", "V1 构建收据")
    _require_equal(build.get("status"), parent["original_build_receipt"]["required_status"], "V1 构建状态")
    _require_equal(build.get("model_trained"), False, "V1 模型训练")
    _require_equal(build.get("prediction_metric_generated"), False, "V1 预测指标")
    _require_equal(build.get("portfolio_or_performance_artifact_read"), False, "V1 绩效读取")

    failure = read_json_strict(
        _project_path(parent["original_clean_replay_failure"]["path"])
    )
    if not isinstance(failure, dict):
        raise EvidenceContractError("V1 重放失败收据必须是对象")
    _verify_self_digest(failure, "receipt_payload_sha256", "V1 重放失败收据")
    _require_equal(failure.get("status"), parent["original_clean_replay_failure"]["required_status"], "V1 失败状态")
    _require_equal(failure.get("exception_type"), parent["original_clean_replay_failure"]["required_exception_type"], "V1 异常类型")
    fragment = str(parent["original_clean_replay_failure"]["required_message_fragment"])
    if fragment not in str(failure.get("message", "")):
        raise EvidenceContractError("V1 失败消息不符合修正对象")

    for name, metadata in build["outputs"].items():
        if name == "report":
            continue
        identity = {key: metadata[key] for key in ("path", "bytes", "sha256")}
        _validate_file_identity(identity, f"V1 构建输出 {name}")
        if "row_count" not in metadata or "persisted_semantic_sha256" not in metadata:
            raise EvidenceContractError(f"V1 构建输出缺少独立元数据：{name}")
    return evidence


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
    freeze = config["freeze_contract"]
    return (
        _project_path(str(freeze["manifest_output"])),
        _project_path(str(freeze["freeze_receipt_output"])),
    )


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, receipt_path = _freeze_outputs(config)
    existing = [str(path) for path in (manifest_path, receipt_path) if path.exists()]
    if existing:
        raise EvidenceContractError(f"G1 V1.0.1 冻结输出已存在：{existing}")
    parent = verify_immutable_parent(config)
    freeze_contract = config["freeze_contract"]
    implementation = _hash_group([str(x) for x in freeze_contract["implementation_files"]])
    governance = _hash_group([str(x) for x in freeze_contract["governance_files"]])
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest: dict[str, Any] = {
        "execution_id": EXECUTION_ID,
        "version": "1.0.1",
        "status": "FROZEN_APPEND_ONLY_FILE_EVIDENCE_COMPARISON_CORRECTION",
        "frozen_at": now,
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_bytes": config_path.stat().st_size,
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "immutable_parent": parent,
        "correction_contract": config["correction_contract"],
        "expected_gate_result": config["expected_gate_result"],
        "implementation_files": implementation,
        "governance_files": governance,
        "outputs": config["outputs"],
        "forbidden_actions": config["forbidden_actions"],
        "parent_actual_label_artifact_read": True,
        "this_freeze_actual_label_artifact_parsed": False,
        "actual_future_path_value_read": False,
        "model_trained": False,
        "performance_artifact_read": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_G1_EVENT_PREDICTION_ADMISSION_V1_0_1_FREEZE",
        "created_at": now,
        "status": "PASS_APPEND_ONLY_FILE_IDENTITY_COMPARISON_CORRECTION_FROZEN",
        "manifest": file_evidence(manifest_path, project_root=ROOT),
        "immutable_parent_verified": True,
        "original_failure_retained": True,
        "table_files_rewritten": False,
        "this_freeze_actual_label_artifact_parsed": False,
        "model_trained": False,
        "performance_artifact_read": False,
        "position_impact": 0,
    }
    atomic_write_json_new(receipt_path, receipt)
    print("G1 V1.0.1 文件身份比较修正已追加冻结，原 V1 与失败收据保持不变。")
    return manifest


def verify_frozen_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, _ = _freeze_outputs(config)
    manifest = read_json_strict(manifest_path)
    if not isinstance(manifest, dict):
        raise EvidenceContractError("G1 V1.0.1 manifest 必须是对象")
    verify_manifest_payload(manifest)
    _require_equal(manifest.get("execution_id"), EXECUTION_ID, "执行 identity")
    _require_equal(manifest.get("config_sha256"), sha256_file(config_path), "配置哈希")
    verify_immutable_parent(config)
    for group_name in ("implementation_files", "governance_files"):
        group = manifest.get(group_name)
        if not isinstance(group, Mapping):
            raise EvidenceContractError(f"V1.0.1 manifest 缺少 {group_name}")
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
            print("G1 V1.0.1 manifest、不可变父链和输出身份验证通过。")
        else:
            freeze(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError) as exc:
        print(f"G1 V1.0.1 冻结失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
