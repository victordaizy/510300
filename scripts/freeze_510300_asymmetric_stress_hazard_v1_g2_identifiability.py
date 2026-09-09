"""冻结并验证 510300 非对称压力风险 V1 的 G2 时间可识别性预检。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.asymmetric_stress_hazard_g2_identifiability_v1 import (
    load_config,
    validate_protocol,
)
from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    canonical_sha256,
    ensure_paths_committed_at_head,
    git_head,
    normalize_project_relative_path,
    read_json_strict,
    sha256_file,
    validate_authority_state,
    validate_registered_inputs,
    verify_manifest_files,
    verify_manifest_payload,
)


DEFAULT_CONFIG = (
    ROOT / "config/510300_asymmetric_stress_hazard_v1_g2_identifiability.yaml"
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _read_mapping(path: Path, *, name: str) -> dict[str, Any]:
    payload = read_json_strict(path)
    if not isinstance(payload, dict):
        raise EvidenceContractError(f"{name} 必须是 JSON 对象")
    return payload


def _verify_file_pin(item: Mapping[str, Any], *, name: str) -> Path:
    path = _project_path(str(item["path"]))
    if not path.is_file():
        raise EvidenceContractError(f"{name} 文件不存在：{path}")
    actual = sha256_file(path)
    expected = str(item["sha256"])
    if actual != expected:
        raise EvidenceContractError(
            f"{name} 冻结哈希漂移：expected={expected}, actual={actual}"
        )
    return path


def _verify_self_hash(
    payload: Mapping[str, Any], *, field: str, expected: str, name: str
) -> None:
    actual_field = str(payload.get(field, ""))
    if actual_field != expected:
        raise EvidenceContractError(
            f"{name} 的 {field} 不一致：expected={expected}, actual={actual_field}"
        )


def verify_parent_evidence(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = config.get("immutable_parent")
    if not isinstance(parent, Mapping):
        raise EvidenceContractError("配置缺少 immutable_parent")

    protocol_path = _verify_file_pin(parent["protocol_config"], name="父协议")
    manifest_path = _verify_file_pin(parent["protocol_manifest"], name="父 manifest")
    parent_manifest = _read_mapping(manifest_path, name="父 manifest")
    verify_manifest_payload(parent_manifest)
    _verify_self_hash(
        parent_manifest,
        field="manifest_payload_sha256",
        expected=str(parent["protocol_manifest"]["required_payload_sha256"]),
        name="父 manifest",
    )

    g1_status_path = _verify_file_pin(parent["g1_status"], name="G1 状态")
    g1_status = _read_mapping(g1_status_path, name="G1 状态")
    if g1_status.get("status") != parent["g1_status"]["required_status"]:
        raise EvidenceContractError("G1 权威状态不是冻结的通过状态")
    if g1_status.get("g1", {}).get("passed") is not True:
        raise EvidenceContractError("G1 passed 字段不是 true")

    g1_receipt_path = _verify_file_pin(parent["g1_receipt"], name="G1 回执")
    g1_receipt = _read_mapping(g1_receipt_path, name="G1 回执")
    _verify_self_hash(
        g1_receipt,
        field="receipt_payload_sha256",
        expected=str(parent["g1_receipt"]["required_payload_sha256"]),
        name="G1 回执",
    )

    remediation_contract_path = _verify_file_pin(
        parent["source_remediation_contract"], name="来源修复契约"
    )
    remediation_status_path = _verify_file_pin(
        parent["source_remediation_status"], name="来源修复状态"
    )
    remediation_status = _read_mapping(remediation_status_path, name="来源修复状态")
    expected_status = parent["source_remediation_status"]
    if remediation_status.get("status") != expected_status["required_status"]:
        raise EvidenceContractError("来源修复状态不是全部来源已准入")
    if remediation_status.get("all_required_sources_admitted") is not True:
        raise EvidenceContractError("来源修复状态未确认全部来源准入")
    if remediation_status.get("g2_allowed") is not True:
        raise EvidenceContractError("来源修复状态未允许 G2")
    if remediation_status.get("feature_values_constructed") is not False:
        raise EvidenceContractError("来源修复阶段越界构造了特征")
    if remediation_status.get("portfolio_metrics_read") is not False:
        raise EvidenceContractError("来源修复阶段越界读取了组合指标")

    remediation_receipt_path = _verify_file_pin(
        parent["source_remediation_receipt"], name="来源修复回执"
    )
    remediation_receipt = _read_mapping(
        remediation_receipt_path, name="来源修复回执"
    )
    _verify_self_hash(
        remediation_receipt,
        field="receipt_payload_sha256",
        expected=str(parent["source_remediation_receipt"]["required_payload_sha256"]),
        name="来源修复回执",
    )
    if remediation_receipt.get("source_remediation_status") != expected_status[
        "required_status"
    ]:
        raise EvidenceContractError("来源修复回执与状态不一致")

    return {
        "protocol_config": {
            "path": protocol_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(protocol_path),
        },
        "protocol_manifest": {
            "path": manifest_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(manifest_path),
            "manifest_payload_sha256": parent_manifest["manifest_payload_sha256"],
        },
        "g1_status": {
            "path": g1_status_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(g1_status_path),
            "status": g1_status["status"],
        },
        "g1_receipt": {
            "path": g1_receipt_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(g1_receipt_path),
            "receipt_payload_sha256": g1_receipt["receipt_payload_sha256"],
        },
        "source_remediation_contract": {
            "path": remediation_contract_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(remediation_contract_path),
        },
        "source_remediation_status": {
            "path": remediation_status_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(remediation_status_path),
            "status": remediation_status["status"],
        },
        "source_remediation_receipt": {
            "path": remediation_receipt_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(remediation_receipt_path),
            "receipt_payload_sha256": remediation_receipt[
                "receipt_payload_sha256"
            ],
        },
    }


def _registered_record(evidence: Any) -> dict[str, Any]:
    return {
        "input_id": evidence.input_id,
        "logical_path": evidence.logical_path,
        "physical_root_id": evidence.physical_root_id,
        "content_sha256": evidence.content_sha256,
        "bytes": evidence.bytes,
        "data_contract_version": evidence.data_contract_version,
    }


def build_manifest(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    validate_protocol(config)
    parent_evidence = verify_parent_evidence(config)

    authority_path = _project_path(str(config["authority_contract"]["path"]))
    if sha256_file(authority_path) != str(config["authority_contract"]["sha256"]):
        raise EvidenceContractError("权威状态文件哈希漂移")
    authority = _read_mapping(authority_path, name="权威状态")
    validate_authority_state(authority)

    registered = validate_registered_inputs(config, project_root=ROOT)
    freeze = config["freeze_contract"]
    committed = ensure_paths_committed_at_head(
        ROOT, [str(item) for item in freeze["required_committed_before_manifest"]]
    )
    implementation_files = {
        str(path): sha256_file(_project_path(str(path)))
        for path in freeze["implementation_files"]
    }
    governance_files = {
        str(path): sha256_file(_project_path(str(path)))
        for path in freeze["governance_files"]
    }
    manifest: dict[str, Any] = {
        "program_id": config["program"]["program_id"],
        "stage_id": config["program"]["stage_id"],
        "version": config["program"]["version"],
        "status": "FROZEN_BEFORE_G2_COVERAGE_VALUES_OR_LABEL_READ",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "git_head_before_manifest": git_head(ROOT),
        "config_path": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "authority": {
            "path": authority_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(authority_path),
            "research_state": authority["research_state"],
            "model_position_target": authority["model_position_target"],
            "order_authorization": authority["order_authorization"],
            "actual_holdings_state": authority["actual_holdings_state"],
            "position_impact": authority["position_impact"],
        },
        "parent_evidence": parent_evidence,
        "registered_inputs": {
            key: _registered_record(value) for key, value in registered.items()
        },
        "implementation_files": implementation_files,
        "governance_files": governance_files,
        "committed_pre_manifest_paths": committed,
        "fixed_subperiod_ids": [
            str(item["id"]) for item in config["fixed_subperiods"]
        ],
        "feature_values_constructed": False,
        "bad10_label_artifacts_read": False,
        "return_values_read": False,
        "model_trained": False,
        "portfolio_metrics_read": False,
        "position_impact": 0,
        "rescue_allowed": False,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    return manifest


def verify_frozen_contract(
    config_path: Path = DEFAULT_CONFIG,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_config(config_path)
    validate_protocol(config)
    manifest_path = _project_path(str(config["outputs"]["manifest"]))
    manifest = _read_mapping(manifest_path, name="G2 预检 manifest")
    verify_manifest_payload(manifest)
    verify_manifest_files(manifest, project_root=ROOT)
    if manifest.get("status") != "FROZEN_BEFORE_G2_COVERAGE_VALUES_OR_LABEL_READ":
        raise EvidenceContractError("G2 预检 manifest 状态不允许执行")
    if sha256_file(config_path) != manifest.get("config_sha256"):
        raise EvidenceContractError("G2 预检配置字节哈希漂移")
    if canonical_sha256(config) != manifest.get("config_canonical_sha256"):
        raise EvidenceContractError("G2 预检配置语义哈希漂移")
    parent_evidence = verify_parent_evidence(config)
    if parent_evidence != manifest.get("parent_evidence"):
        raise EvidenceContractError("父证据与 G2 预检 manifest 不一致")

    authority_path = _project_path(str(config["authority_contract"]["path"]))
    authority = _read_mapping(authority_path, name="权威状态")
    validate_authority_state(authority)
    if sha256_file(authority_path) != manifest["authority"]["sha256"]:
        raise EvidenceContractError("权威状态与 G2 预检 manifest 不一致")

    registered = validate_registered_inputs(config, project_root=ROOT)
    actual_registered = {
        key: _registered_record(value) for key, value in registered.items()
    }
    if actual_registered != manifest.get("registered_inputs"):
        raise EvidenceContractError("注册输入与 G2 预检 manifest 不一致")

    committed = ensure_paths_committed_at_head(
        ROOT,
        [
            str(item)
            for item in config["freeze_contract"][
                "required_committed_before_preflight"
            ]
        ],
    )
    return config, manifest, {
        "authority": authority,
        "registered_inputs": registered,
        "committed_paths": committed,
        "git_head": git_head(ROOT),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="冻结或验证 510300 非对称压力风险 V1 的 G2 时间可识别性预检"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    try:
        if args.write:
            manifest = build_manifest(args.config)
            output = _project_path(
                str(load_config(args.config)["outputs"]["manifest"])
            )
            atomic_write_json_new(output, manifest)
            print(f"已冻结 G2 时间可识别性 manifest：{output}")
            print(f"manifest_payload_sha256={manifest['manifest_payload_sha256']}")
        else:
            _, manifest, verification = verify_frozen_contract(args.config)
            print("G2 时间可识别性冻结契约验证通过")
            print(f"git_head={verification['git_head']}")
            print(f"manifest_payload_sha256={manifest['manifest_payload_sha256']}")
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"G2 时间可识别性冻结失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
