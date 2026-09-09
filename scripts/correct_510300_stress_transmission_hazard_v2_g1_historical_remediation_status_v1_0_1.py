"""冻结并执行 510300 V2 G1 历史修复状态映射纠正 V1.0.1。"""

from __future__ import annotations

import argparse
import hashlib
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
    normalize_project_relative_path,
)


EXECUTION_ID = (
    "510300_STRESS_TRANSMISSION_HAZARD_V2_"
    "G1_HISTORICAL_REMEDIATION_STATUS_CORRECTION_V1_0_1"
)
DEFAULT_CONFIG = (
    ROOT
    / "config"
    / "510300_stress_transmission_hazard_v2_"
    "g1_historical_remediation_status_correction_v1_0_1.yaml"
)


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise EvidenceContractError(f"YAML 顶层必须是对象：{path}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise EvidenceContractError(f"JSON 顶层必须是对象：{path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_identity(contract: Mapping[str, Any], *, label: str) -> None:
    path = _project_path(str(contract["path"]))
    if not path.is_file():
        raise EvidenceContractError(f"{label}不存在：{path}")
    actual = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    expected = {"bytes": int(contract["bytes"]), "sha256": str(contract["sha256"])}
    if actual != expected:
        raise EvidenceContractError(
            f"{label}身份漂移：actual={actual}, expected={expected}"
        )


def _verify_payload(payload: Mapping[str, Any], *, label: str) -> None:
    expected = str(payload.get("receipt_payload_sha256") or "")
    body = {
        key: value
        for key, value in payload.items()
        if key != "receipt_payload_sha256"
    }
    if not expected or canonical_sha256(body) != expected:
        raise EvidenceContractError(f"{label} payload 摘要不一致")


def derive_corrected_branch_state(
    gate_result: Mapping[str, Any],
) -> dict[str, Any]:
    """按两层冻结门生成不互相覆盖的 B2/B3 状态。"""

    mechanism = bool(gate_result["mechanism_discovery_prerequisite_passed"])
    full = bool(gate_result["full_three_coefficient_model_prerequisite_passed"])
    if full and not mechanism:
        raise EvidenceContractError("完整模型门通过但机制门未通过，门结果自相矛盾")
    if full:
        overall = "PASS_FULL_B2_AND_B3_EVENT_IDENTIFIABILITY"
        b2 = "PASS_MECHANISM_DISCOVERY_PREREQUISITE"
        b3 = "PASS_FULL_MODEL_PREREQUISITE"
    elif mechanism:
        overall = "PASS_B2_MECHANISM_ONLY_B3_NOT_IDENTIFIABLE"
        b2 = "PASS_MECHANISM_DISCOVERY_PREREQUISITE"
        b3 = "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY"
    else:
        overall = str(gate_result["G1_DATA_AND_EVENTS"])
        b2 = "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY"
        b3 = "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY"
    return {
        "G1_DATA_AND_EVENTS": overall,
        "B2_MECHANISM_G1": b2,
        "B3_FULL_MODEL_G1": b3,
        "mechanism_discovery_prerequisite_passed": mechanism,
        "full_three_coefficient_model_prerequisite_passed": full,
        "next_allowed_step": str(gate_result["next_allowed_step"]),
    }


def _frozen_paths(config: Mapping[str, Any]) -> list[Path]:
    freeze_contract = config["freeze_contract"]
    values = [
        *freeze_contract["implementation_files"],
        *freeze_contract["governance_files"],
    ]
    paths = [_project_path(str(value)) for value in values]
    if len(paths) != len(set(paths)):
        raise EvidenceContractError("纠正冻结文件列表存在重复项")
    return paths


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if str(config["program"]["execution_id"]) != EXECUTION_ID:
        raise EvidenceContractError("纠正 execution_id 漂移")
    if bool(config["program"]["g2_authorized_in_this_execution"]):
        raise EvidenceContractError("状态纠正不得授权 G2")
    for name, contract in config["immutable_parent"].items():
        _verify_identity(contract, label=f"immutable_parent.{name}")
    frozen_files: dict[str, Any] = {}
    for path in _frozen_paths(config):
        if not path.is_file():
            raise EvidenceContractError(f"纠正冻结文件不存在：{path}")
        frozen_files[path.relative_to(ROOT).as_posix()] = file_evidence(
            path, project_root=ROOT
        )
    manifest_path = _project_path(config["freeze_contract"]["manifest_output"])
    receipt_path = _project_path(config["freeze_contract"]["freeze_receipt_output"])
    if manifest_path.exists() or receipt_path.exists():
        raise EvidenceContractError("纠正 manifest 或冻结收据已存在，禁止覆盖")
    manifest: dict[str, Any] = {
        "execution_id": EXECUTION_ID,
        "version": str(config["program"]["version"]),
        "frozen_at": _now(),
        "status": "FROZEN_STATUS_MAPPING_CORRECTION_ONLY",
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": _sha256_file(config_path),
        "frozen_files": frozen_files,
        "immutable_parent": config["immutable_parent"],
        "mapping_contract": config["mapping_contract"],
        "g2_authorized": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt: dict[str, Any] = {
        "receipt_id": f"{EXECUTION_ID}_FREEZE_RECEIPT",
        "created_at": _now(),
        "status": "PASS_STATUS_MAPPING_CORRECTION_FROZEN_BEFORE_V1_REPLAY_RESULT",
        "manifest": file_evidence(manifest_path, project_root=ROOT),
        "g2_run": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(receipt_path, receipt)
    print("V1.0.1 状态映射纠正规则已冻结；尚未读取 V1 重放结果。", flush=True)
    return receipt


def verify_frozen(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = _load_yaml(config_path)
    manifest_path = _project_path(config["freeze_contract"]["manifest_output"])
    manifest = _read_json(manifest_path)
    expected = str(manifest.get("manifest_payload_sha256") or "")
    body = {
        key: value
        for key, value in manifest.items()
        if key != "manifest_payload_sha256"
    }
    if not expected or canonical_sha256(body) != expected:
        raise EvidenceContractError("纠正冻结 manifest payload 摘要不一致")
    if _sha256_file(config_path) != str(manifest["config_sha256"]):
        raise EvidenceContractError("纠正配置在冻结后发生漂移")
    for relative, expected_evidence in manifest["frozen_files"].items():
        actual = file_evidence(_project_path(relative), project_root=ROOT)
        if actual != expected_evidence:
            raise EvidenceContractError(f"纠正冻结文件发生漂移：{relative}")
    for name, contract in config["immutable_parent"].items():
        _verify_identity(contract, label=f"immutable_parent.{name}")
    return config


def correct(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = verify_frozen(config_path)
    output_status = _project_path(config["outputs"]["corrected_status"])
    output_receipt = _project_path(config["outputs"]["correction_receipt"])
    if output_status.exists() or output_receipt.exists():
        raise EvidenceContractError("V1.0.1 纠正输出已存在，禁止覆盖")

    build = _read_json(_project_path(config["immutable_parent"]["v1_build_receipt"]["path"]))
    replay_path = _project_path(config["post_replay_inputs"]["v1_replay_receipt"])
    original_status_path = _project_path(config["post_replay_inputs"]["v1_status"])
    replay = _read_json(replay_path)
    original = _read_json(original_status_path)
    _verify_payload(build, label="V1 构建收据")
    _verify_payload(replay, label="V1 重放收据")
    if replay.get("status") != "PASS_FRESH_PROCESS_FULL_SCOPE_REMEDIATION_AND_G1_REPLAY":
        raise EvidenceContractError("V1 独立新进程重放未通过")
    if canonical_sha256(build["metrics"]) != canonical_sha256(replay["metrics"]):
        raise EvidenceContractError("V1 构建与重放指标不一致")
    if canonical_sha256(build["g1_gate_result"]) != canonical_sha256(
        replay["g1_gate_result"]
    ):
        raise EvidenceContractError("V1 构建与重放门结果不一致")
    for payload, label in ((build, "V1 构建"), (replay, "V1 重放")):
        if bool(payload.get("g2_run")) or bool(payload.get("model_trained")):
            raise EvidenceContractError(f"{label}越权运行 G2 或训练模型")

    gate = replay["g1_gate_result"]
    branches = derive_corrected_branch_state(gate)
    if str(original.get("G1_DATA_AND_EVENTS")) != branches["G1_DATA_AND_EVENTS"]:
        raise EvidenceContractError("V1 原状态的总 G1 状态与重放门结果不一致")
    if original.get("fresh_process_replay") != "PASS":
        raise EvidenceContractError("V1 原状态未记录重放通过")
    metrics = replay["metrics"]["g1"]
    corrected: dict[str, Any] = {
        "program_id": config["program"]["program_id"],
        "execution_id": EXECUTION_ID,
        "updated_at": _now(),
        "status_correction": "PASS_APPEND_ONLY_V1_0_1_BRANCH_MAPPING_CORRECTION",
        "correction_reason": "V1_NEXT_ALLOWED_STEP_WAS_KEYED_ONLY_TO_FULL_B3_GATE",
        "supersedes_status_path": original_status_path.relative_to(ROOT).as_posix(),
        "original_v1_status_preserved": True,
        **branches,
        "b2_identifiable_event_count": int(metrics["b2_identifiable_event_count"]),
        "b3_identifiable_event_count": int(metrics["b3_identifiable_event_count"]),
        "b2_eligible_non_event_risk_day_count": int(
            metrics["b2_eligible_non_event_risk_day_count"]
        ),
        "b3_eligible_non_event_risk_day_count": int(
            metrics["b3_eligible_non_event_risk_day_count"]
        ),
        "next_step_authorized_in_this_execution": False,
        "G2": "NOT_RUN_REQUIRES_NEW_EXPLICIT_AUTHORIZATION",
        "model_trained": False,
        "current_market_probability": "NOT_GENERATED",
        "position_target": "UNSET",
        "position_impact": 0,
        "v1_build_receipt": file_evidence(
            _project_path(config["immutable_parent"]["v1_build_receipt"]["path"]),
            project_root=ROOT,
        ),
        "v1_replay_receipt": file_evidence(replay_path, project_root=ROOT),
        "v1_original_status": file_evidence(original_status_path, project_root=ROOT),
    }
    atomic_write_json_new(output_status, corrected)
    receipt: dict[str, Any] = {
        "receipt_id": f"{EXECUTION_ID}_RECEIPT",
        "created_at": _now(),
        "status": "PASS_APPEND_ONLY_STATUS_CORRECTION_WITH_NO_DATA_OR_GATE_CHANGE",
        "corrected_status": file_evidence(output_status, project_root=ROOT),
        "source_gate_result": gate,
        "source_metrics_sha256": canonical_sha256(replay["metrics"]),
        "data_feature_sample_event_or_gate_changed": False,
        "g2_run": False,
        "model_trained": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(output_receipt, receipt)
    print(
        f"V1.0.1 状态纠正完成：{branches['G1_DATA_AND_EVENTS']}；G2 未运行。",
        flush=True,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "correct"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        if args.command == "freeze":
            freeze(args.config)
        else:
            correct(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"执行失败：{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
