"""冻结 V2 四态来源 V1.0.1 可选字段修正。"""

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
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_sources_v1 import (
    verify_frozen_manifest as verify_v1_manifest,
)


CORRECTION_ID = "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_SOURCES_V1_0_1"
DEFAULT_CONFIG = (
    ROOT
    / "config/510300_stress_transmission_hazard_v2_four_state_sources_v1_0_1.yaml"
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("V1.0.1 修正配置必须是对象")
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise EvidenceContractError(
            f"{label}发生漂移：expected={expected!r}, actual={actual!r}"
        )


def _validate_identity(relative: str, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    path = _project_path(relative)
    if not path.is_file():
        raise EvidenceContractError(f"依赖文件不存在：{relative}")
    if path.stat().st_size != int(expected_bytes) or sha256_file(path) != expected_sha256:
        raise EvidenceContractError(f"依赖文件字节或哈希漂移：{relative}")
    return file_evidence(path, project_root=ROOT)


def validate_config(config: Mapping[str, Any]) -> None:
    program = config["program"]
    _require_equal(program.get("program_id"), "510300_STRESS_TRANSMISSION_HAZARD_V2", "项目")
    _require_equal(program.get("correction_id"), CORRECTION_ID, "修正 identity")
    _require_equal(program.get("version"), "1.0.1", "修正版本")
    _require_equal(program.get("correction_scope"), "DIVIDEND_OPTIONAL_FIELD_SCHEMA_ONLY", "修正范围")
    _require_equal(program.get("return_evaluation"), "NOT_ALLOWED", "收益评估权限")
    _require_equal(program.get("position_impact"), 0, "仓位影响")
    _require_equal(config["predecessor_freeze"].get("mutation_allowed"), False, "V1 修改权限")

    failure = config["predecessor_probe_failure"]
    _require_equal(
        failure.get("status"),
        "BLOCKED_SCHEMA_OVERCONSTRAINED_OPTIONAL_DIVIDEND_FIELDS",
        "V1 探测失败状态",
    )
    _require_equal(failure.get("data_snapshots_written"), False, "V1 数据快照状态")
    _require_equal(failure.get("performance_values_read"), False, "V1 绩效读取状态")

    correction = config["schema_correction"]
    _require_equal(correction.get("source"), "dividend", "修正来源")
    required = correction["required_fields_after_correction"]
    optional = correction["optional_fields"]
    if "base_date" in required or "base_share" in required:
        raise EvidenceContractError("base_date/base_share 不得继续作为必填字段")
    _require_equal(optional, ["base_date", "base_share"], "可选字段")
    for field in ("stk_div", "cash_div_tax", "ex_date", "imp_ann_date", "div_proc"):
        if field not in required:
            raise EvidenceContractError(f"经济行动必填字段缺失：{field}")
    _require_equal(correction.get("adjusted_return_prohibition_unchanged"), True, "复权收益禁令")
    _require_equal(correction.get("missing_return_zero_fill_prohibition_unchanged"), True, "缺失补零禁令")

    probe = config["corrected_probe_contract"]
    _require_equal(probe.get("scope_identical_to_predecessor"), True, "探测范围")
    _require_equal(probe.get("full_history_or_bulk_member_download_allowed"), False, "批量权限")

    forbidden = set(config["forbidden_actions"])
    required_forbidden = {
        "MODIFY_OR_DELETE_V1_SOURCE_ADDENDUM",
        "EXPAND_PROBE_DATE_SYMBOL_OR_ENDPOINT_SCOPE",
        "TREAT_PROVIDER_SUSPEND_D_AS_EXCHANGE_OFFICIAL_EVIDENCE",
        "FILL_MISSING_CONSTITUENT_RETURN_WITH_ZERO",
        "RUN_BULK_COLLECTION_WITHOUT_SEPARATE_EXECUTION_FREEZE",
        "READ_NEW_AUC_RETURN_SHARPE_DRAWDOWN_NAV_POSITION_OR_ORDER_OUTPUT",
    }
    if not required_forbidden.issubset(forbidden):
        raise EvidenceContractError(
            f"V1.0.1 禁止动作缺项：{sorted(required_forbidden.difference(forbidden))}"
        )


def _hash_group(paths: list[str]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for raw in paths:
        relative = normalize_project_relative_path(raw)
        path = _project_path(relative)
        if not path.is_file():
            raise EvidenceContractError(f"待冻结文件不存在：{relative}")
        evidence[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return evidence


def verify_predecessor(config: Mapping[str, Any]) -> dict[str, Any]:
    verify_v1_manifest()
    predecessor = config["predecessor_freeze"]
    return _validate_identity(
        str(predecessor["manifest_path"]),
        int(predecessor["manifest_bytes"]),
        str(predecessor["manifest_sha256"]),
    )


def _outputs(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    freeze_contract = config["freeze_contract"]
    failure = config["predecessor_probe_failure"]
    return (
        _project_path(str(freeze_contract["manifest_output"])),
        _project_path(str(freeze_contract["receipt_output"])),
        _project_path(str(failure["failure_receipt_output"])),
    )


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, receipt_path, failure_path = _outputs(config)
    existing = [str(path) for path in (manifest_path, receipt_path, failure_path) if path.exists()]
    if existing:
        raise EvidenceContractError(f"V1.0.1 输出已存在，禁止覆盖：{existing}")
    predecessor = verify_predecessor(config)
    contract = config["freeze_contract"]
    implementation = _hash_group([str(x) for x in contract["implementation_files"]])
    governance = _hash_group([str(x) for x in contract["governance_files"]])
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()

    failure_receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_SOURCE_PROBE_V1_FAILURE",
        "recorded_at": now,
        "observed_on": config["predecessor_probe_failure"]["observed_on"],
        "status": config["predecessor_probe_failure"]["status"],
        "endpoint_results": config["predecessor_probe_failure"]["endpoint_results"],
        "data_snapshots_written": False,
        "performance_values_read": False,
        "credential_persisted": False,
        "security_audit_performed": False,
        "position_impact": 0,
        "next_action": "APPEND_ONLY_V1_0_1_OPTIONAL_FIELD_CORRECTION",
    }
    atomic_write_json_new(failure_path, failure_receipt)

    manifest: dict[str, Any] = {
        "correction_id": CORRECTION_ID,
        "version": "1.0.1",
        "status": "FROZEN_BEFORE_CORRECTED_BOUNDED_PROBE",
        "frozen_at": now,
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_bytes": config_path.stat().st_size,
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "predecessor_manifest": predecessor,
        "predecessor_probe_failure_receipt": file_evidence(failure_path, project_root=ROOT),
        "implementation_files": implementation,
        "governance_files": governance,
        "schema_correction": config["schema_correction"],
        "corrected_probe_contract": config["corrected_probe_contract"],
        "forbidden_actions": config["forbidden_actions"],
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_results_read": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_SOURCES_V1_0_1_FREEZE",
        "created_at": now,
        "status": "PASS_APPEND_ONLY_OPTIONAL_FIELD_CORRECTION_FREEZE",
        "manifest": file_evidence(manifest_path, project_root=ROOT),
        "predecessor_probe_failure_receipt": file_evidence(failure_path, project_root=ROOT),
        "scope_expanded": False,
        "network_probe_run_by_freeze": False,
        "performance_values_read": False,
        "security_audit_performed": False,
        "position_impact": 0,
    }
    atomic_write_json_new(receipt_path, receipt)
    print("V1.0.1 可选字段修正已追加冻结；允许重做同规模有界探测。")
    return manifest


def verify_frozen_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, _, failure_path = _outputs(config)
    manifest = read_json_strict(manifest_path)
    if not isinstance(manifest, dict):
        raise EvidenceContractError("V1.0.1 manifest 必须是对象")
    verify_manifest_payload(manifest)
    _require_equal(manifest.get("correction_id"), CORRECTION_ID, "修正 identity")
    _require_equal(manifest.get("config_sha256"), sha256_file(config_path), "修正配置哈希")
    verify_predecessor(config)
    expected_failure = manifest["predecessor_probe_failure_receipt"]
    _validate_identity(
        str(expected_failure["path"]),
        int(expected_failure["bytes"]),
        str(expected_failure["sha256"]),
    )
    _require_equal(failure_path, _project_path(str(expected_failure["path"])), "失败收据路径")
    for group_name in ("implementation_files", "governance_files"):
        group = manifest.get(group_name)
        if not isinstance(group, Mapping):
            raise EvidenceContractError(f"V1.0.1 manifest 缺少 {group_name}")
        for relative, evidence in group.items():
            _validate_identity(
                str(relative), int(evidence["bytes"]), str(evidence["sha256"])
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
            print("V1.0.1 修正 manifest 与前序证据验证通过。")
        else:
            freeze(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError) as exc:
        print(f"V1.0.1 修正冻结失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
