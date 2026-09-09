"""冻结 V2 M/F/T 重放的 V1.0.1 持久化 schema 规范化修正。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
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
from scripts.build_510300_stress_transmission_hazard_v2_mft_features_v1 import (
    frame_semantic_sha256,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_mft_feature_execution_v1 import (
    verify_frozen_manifest as verify_v1_mft_manifest,
)


CORRECTION_ID = "510300_STRESS_TRANSMISSION_HAZARD_V2_MFT_FEATURE_EXECUTION_V1_0_1"
DEFAULT_CONFIG = (
    ROOT / "config/510300_stress_transmission_hazard_v2_mft_feature_execution_v1_0_1.yaml"
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("M/F/T V1.0.1 修正配置必须是对象")
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise EvidenceContractError(
            f"{label}发生漂移：expected={expected!r}, actual={actual!r}"
        )


def _validate_identity(contract: Mapping[str, Any], label: str) -> Path:
    path = _project_path(str(contract["path"]))
    if not path.is_file():
        raise EvidenceContractError(f"{label}不存在：{path}")
    if path.stat().st_size != int(contract["bytes"]) or sha256_file(path) != str(
        contract["sha256"]
    ):
        raise EvidenceContractError(f"{label}字节或哈希漂移：{path}")
    return path


def validate_config(config: Mapping[str, Any]) -> None:
    program = config["program"]
    _require_equal(program.get("program_id"), "510300_STRESS_TRANSMISSION_HAZARD_V2", "项目")
    _require_equal(program.get("correction_id"), CORRECTION_ID, "修正 identity")
    _require_equal(program.get("version"), "1.0.1", "修正版本")
    _require_equal(
        program.get("correction_scope"),
        "PERSISTED_SCHEMA_CANONICALIZATION_BEFORE_REPLAY_COMPARISON_ONLY",
        "修正范围",
    )
    _require_equal(program.get("return_evaluation"), "NOT_ALLOWED", "收益评估")
    _require_equal(program.get("position_impact"), 0, "仓位影响")
    _require_equal(config["immutable_parent"].get("mutation_allowed"), False, "V1 修改权限")

    failure = config["observed_failure"]
    _require_equal(
        failure.get("status"),
        "BLOCKED_G0_REPLAY_PERSISTED_DTYPE_COMPARISON_MISMATCH",
        "失败状态",
    )
    _require_equal(failure.get("exit_code"), 1, "失败退出码")
    _require_equal(failure.get("comparison_stopped_at_output"), "internal_features", "失败表")
    _require_equal(failure.get("comparison_stopped_at_column"), "tail_diffusion5", "失败列")
    _require_equal(failure.get("stored_dtype"), "float64", "存储 dtype")
    _require_equal(failure.get("rebuilt_dtype"), "object", "重建 dtype")
    _require_equal(failure.get("replay_receipt_written"), False, "原重放收据")
    _require_equal(failure.get("g0_status_written"), False, "原 G0 状态")

    rule = config["canonicalization_rule"]
    _require_equal(rule.get("canonical_schema_source"), "IMMUTABLE_PERSISTED_PARQUET_SCHEMA", "规范 schema")
    _require_equal(
        rule.get("allowed_conversions"),
        {
            "internal_features": {"tail_diffusion5": {"from": "object", "to": "float64"}},
            "mft_feature_panel": {"tail_diffusion5": {"from": "object", "to": "float64"}},
        },
        "允许转换",
    )
    for key in (
        "any_other_dtype_difference_allowed",
        "any_value_difference_allowed",
        "row_order_difference_allowed",
        "column_order_difference_allowed",
        "original_output_overwrite_allowed",
        "original_receipt_overwrite_allowed",
        "feature_formula_change_allowed",
        "information_clock_change_allowed",
        "input_change_allowed",
        "coverage_gate_change_allowed",
        "no_view_change_allowed",
    ):
        _require_equal(rule.get(key), False, key)

    forbidden = set(config["forbidden_actions"])
    required = {
        "MODIFY_OR_DELETE_V1_MFT_FREEZE_OUTPUTS_OR_RECEIPTS",
        "CHANGE_FEATURE_FORMULA_CLOCK_INPUT_COVERAGE_OR_NO_VIEW_RULE",
        "CANONICALIZE_ANY_COLUMN_EXCEPT_THE_TWO_DECLARED_TAIL_DIFFUSION5_COLUMNS",
        "ACCEPT_ANY_VALUE_ROW_ORDER_COLUMN_ORDER_OR_UNDECLARED_DTYPE_DIFFERENCE",
        "READ_ACTUAL_BAD10_FUTURE_RETURN_OR_PERFORMANCE_ARTIFACT",
    }
    if not required.issubset(forbidden):
        raise EvidenceContractError(
            f"V1.0.1 禁止动作缺项：{sorted(required.difference(forbidden))}"
        )


def verify_parent_and_affected_outputs(config: Mapping[str, Any]) -> dict[str, Any]:
    verify_v1_mft_manifest()
    parent = config["immutable_parent"]
    evidence: dict[str, Any] = {}
    for name in ("manifest", "build_receipt", "feature_status"):
        path = _validate_identity(parent[name], f"V1 {name}")
        evidence[name] = file_evidence(path, project_root=ROOT)
    receipt = read_json_strict(_project_path(parent["build_receipt"]["path"]))
    _require_equal(receipt.get("status"), parent["build_receipt"]["required_status"], "V1 构建状态")
    status = read_json_strict(_project_path(parent["feature_status"]["path"]))
    _require_equal(status.get("next_allowed_step"), parent["feature_status"]["required_next_step"], "V1 下一步骤")
    affected_evidence: dict[str, Any] = {}
    for name, contract in config["affected_outputs"].items():
        path = _validate_identity(contract, f"受影响输出 {name}")
        frame = pd.read_parquet(path)
        persisted = frame_semantic_sha256(frame)
        _require_equal(persisted, contract["persisted_semantic_sha256"], f"{name} 写后语义摘要")
        if persisted == contract["original_prewrite_semantic_sha256"]:
            raise EvidenceContractError(f"{name} 未复现写前/写后摘要差异")
        if str(frame["tail_diffusion5"].dtype) != "float64":
            raise EvidenceContractError(f"{name}.tail_diffusion5 存储 dtype 不是 float64")
        affected_evidence[name] = {
            **file_evidence(path, project_root=ROOT),
            "persisted_semantic_sha256": persisted,
            "tail_diffusion5_dtype": str(frame["tail_diffusion5"].dtype),
        }
    evidence["affected_outputs"] = affected_evidence
    return evidence


def _hash_group(paths: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in paths:
        relative = normalize_project_relative_path(raw)
        path = _project_path(relative)
        if not path.is_file():
            raise EvidenceContractError(f"待冻结文件不存在：{relative}")
        item = file_evidence(path, project_root=ROOT)
        item.pop("path")
        result[relative] = item
    return result


def _freeze_paths(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    contract = config["freeze_contract"]
    failure = config["observed_failure"]
    return (
        _project_path(contract["manifest_output"]),
        _project_path(contract["receipt_output"]),
        _project_path(failure["failure_receipt_output"]),
    )


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, receipt_path, failure_path = _freeze_paths(config)
    corrected_outputs = [
        _project_path(path) for path in config["corrected_outputs"].values()
    ]
    existing = [
        str(path)
        for path in (manifest_path, receipt_path, failure_path, *corrected_outputs)
        if path.exists()
    ]
    if existing:
        raise EvidenceContractError(f"V1.0.1 修正输出已存在，禁止覆盖：{existing}")
    parent = verify_parent_and_affected_outputs(config)
    contract = config["freeze_contract"]
    implementation = _hash_group([str(x) for x in contract["implementation_files"]])
    governance = _hash_group([str(x) for x in contract["governance_files"]])
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    failure_receipt = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_MFT_CLEAN_REPLAY_V1_FAILURE",
        "recorded_at": now,
        **config["observed_failure"],
        "next_action": "APPEND_ONLY_PERSISTED_SCHEMA_CANONICALIZATION_V1_0_1",
        "feature_values_changed": False,
        "position_impact": 0,
    }
    failure_receipt.pop("failure_receipt_output", None)
    atomic_write_json_new(failure_path, failure_receipt)
    manifest: dict[str, Any] = {
        "correction_id": CORRECTION_ID,
        "version": "1.0.1",
        "status": "FROZEN_BEFORE_CORRECTED_FULL_REPLAY",
        "frozen_at": now,
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_bytes": config_path.stat().st_size,
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "immutable_parent_and_affected_outputs": parent,
        "failure_receipt": file_evidence(failure_path, project_root=ROOT),
        "implementation_files": implementation,
        "governance_files": governance,
        "canonicalization_rule": config["canonicalization_rule"],
        "corrected_outputs": config["corrected_outputs"],
        "targeted_test_files": config["targeted_test_files"],
        "forbidden_actions": config["forbidden_actions"],
        "actual_label_artifact_read": False,
        "actual_performance_artifact_read": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_MFT_FEATURE_EXECUTION_V1_0_1_FREEZE",
        "created_at": now,
        "status": "PASS_APPEND_ONLY_PERSISTED_SCHEMA_CANONICALIZATION_FREEZE",
        "manifest": file_evidence(manifest_path, project_root=ROOT),
        "failure_receipt": file_evidence(failure_path, project_root=ROOT),
        "feature_formula_changed": False,
        "information_clock_changed": False,
        "feature_values_changed": False,
        "actual_label_artifact_read": False,
        "actual_performance_artifact_read": False,
        "position_impact": 0,
    }
    atomic_write_json_new(receipt_path, receipt)
    print("V1.0.1 持久化 schema 规范化修正已冻结；原 V1 保持不变。")
    return manifest


def verify_frozen_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, _, failure_path = _freeze_paths(config)
    manifest = read_json_strict(manifest_path)
    if not isinstance(manifest, dict):
        raise EvidenceContractError("V1.0.1 修正 manifest 必须是对象")
    verify_manifest_payload(manifest)
    _require_equal(manifest.get("correction_id"), CORRECTION_ID, "修正 identity")
    _require_equal(manifest.get("config_sha256"), sha256_file(config_path), "修正配置哈希")
    verify_parent_and_affected_outputs(config)
    _validate_identity(manifest["failure_receipt"], "V1 失败收据")
    _require_equal(
        failure_path,
        _project_path(manifest["failure_receipt"]["path"]),
        "V1 失败收据路径",
    )
    for group_name in ("implementation_files", "governance_files"):
        group = manifest.get(group_name)
        if not isinstance(group, Mapping):
            raise EvidenceContractError(f"修正 manifest 缺少 {group_name}")
        for relative, evidence in group.items():
            _validate_identity(
                {"path": relative, "bytes": evidence["bytes"], "sha256": evidence["sha256"]},
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
            print("V1.0.1 修正 manifest、失败证据与父链验证通过。")
        else:
            freeze(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError) as exc:
        print(f"V1.0.1 修正冻结失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
