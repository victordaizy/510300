"""冻结 V2 成分收益四态来源准入附录。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
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


ADDENDUM_ID = "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_SOURCES_V1"
DEFAULT_CONFIG = (
    ROOT / "config/510300_stress_transmission_hazard_v2_four_state_sources_v1.yaml"
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("四态来源附录配置必须是对象")
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise EvidenceContractError(
            f"{label}发生漂移：expected={expected!r}, actual={actual!r}"
        )


def _validate_identity(relative: str, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    path = _project_path(relative)
    if not path.is_file():
        raise EvidenceContractError(f"冻结依赖文件不存在：{relative}")
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != int(expected_bytes) or actual_sha256 != expected_sha256.lower():
        raise EvidenceContractError(
            f"冻结依赖漂移：{relative}，expected_bytes={expected_bytes}，"
            f"actual_bytes={actual_bytes}，expected_sha256={expected_sha256}，"
            f"actual_sha256={actual_sha256}"
        )
    return file_evidence(path, project_root=ROOT)


def validate_config(config: Mapping[str, Any]) -> None:
    program = config["program"]
    expected_program = {
        "program_id": "510300_STRESS_TRANSMISSION_HAZARD_V2",
        "addendum_id": ADDENDUM_ID,
        "version": "1.0.0",
        "stage": "SOURCE_CONTRACT_FREEZE_AND_BOUNDED_PROBE_ONLY",
        "research_state": "DISCOVERY_ONLY",
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_evaluation": "NOT_ALLOWED",
        "position_impact": 0,
        "model_training_allowed": False,
        "probability_threshold_selection_allowed": False,
        "paper_or_shadow_allowed": False,
        "broker_connection_allowed": False,
        "position_or_order_generation_allowed": False,
    }
    for key, expected in expected_program.items():
        _require_equal(program.get(key), expected, f"program.{key}")
    _require_equal(
        program.get("executable_assets"),
        ["510300.SH", "CASH_CNY"],
        "可执行资产范围",
    )

    parent = config["parent_freeze"]
    _require_equal(parent.get("mutation_allowed"), False, "父冻结修改权限")

    membership = config["point_in_time_membership"]
    _require_equal(membership.get("index_code"), "000300", "点时指数代码")
    _require_equal(
        membership.get("current_member_backfill_allowed"), False, "当前成员历史回填权限"
    )
    _require_equal(
        membership.get("expected_members_per_session"), 300, "每日点时成员数"
    )

    legacy = config["legacy_daily_observation_seed"]
    prohibited = set(legacy["prohibited_columns"])
    required_prohibited = {
        "linked_adjustment_factor",
        "total_return_open",
        "total_return_high",
        "total_return_low",
        "total_return_close",
        "is_suspended",
        "adjustment_source",
    }
    if not required_prohibited.issubset(prohibited):
        raise EvidenceContractError("旧链禁止字段清单不完整")
    _require_equal(legacy.get("direct_predictive_use_allowed"), False, "旧链直接用途")
    _require_equal(legacy.get("forward_filled_values_allowed"), False, "旧链前向填充值")

    provider = config["provider_contract"]
    _require_equal(provider.get("credential_source"), "EPHEMERAL_ENV_ONLY", "凭据来源")
    _require_equal(provider.get("credential_persistence_allowed"), False, "凭据持久化")
    allowed_hosts = set(provider["allowed_endpoint_hosts"])
    expected_hosts = {"api.tushare.pro", "fast.xiaodefa.cn", "tt.xiaodefa.cn"}
    _require_equal(allowed_hosts, expected_hosts, "供应商允许节点")
    for source_name in ("daily", "suspend_d", "dividend", "adj_factor"):
        source = provider[source_name]
        parsed = urlparse(str(source["documentation_url"]))
        if parsed.scheme != "https" or parsed.netloc != "tushare.pro":
            raise EvidenceContractError(f"{source_name} 文档必须指向 Tushare HTTPS 官方页")
        if not source.get("required_fields"):
            raise EvidenceContractError(f"{source_name} 必填字段不能为空")
    _require_equal(
        provider["suspend_d"].get("official_exchange_evidence_eligible"),
        False,
        "供应商停牌表官方证据资格",
    )
    _require_equal(
        provider["suspend_d"].get("zero_return_authorization"),
        False,
        "供应商停牌表补零权限",
    )
    _require_equal(
        provider["adj_factor"].get("direct_return_construction_allowed"),
        False,
        "复权因子直接收益权限",
    )
    _require_equal(
        provider["dividend"].get("availability_rule"),
        "IMPLEMENTATION_ANNOUNCEMENT_DATE_STRICTLY_BEFORE_EX_DATE",
        "公司行动可得时钟",
    )

    mapping = config["four_state_mapping"]
    _require_equal(
        set(mapping).intersection(
            {
                "TRADED_VALID",
                "OFFICIAL_SUSPENSION",
                "CORPORATE_ACTION_UNRESOLVED",
                "SUPPLIER_MISSING_OR_CONFLICT",
            }
        ),
        {
            "TRADED_VALID",
            "OFFICIAL_SUSPENSION",
            "CORPORATE_ACTION_UNRESOLVED",
            "SUPPLIER_MISSING_OR_CONFLICT",
        },
        "成分收益四态",
    )
    _require_equal(mapping.get("blanket_zero_fill_allowed"), False, "缺失补零权限")
    _require_equal(
        mapping["OFFICIAL_SUSPENSION"].get("provider_suspend_d_sufficient"),
        False,
        "官方停牌证据门",
    )

    coverage = config["coverage_and_new_member_contract"]
    _require_equal(
        coverage.get("breadth_equal_weight_minimum_usable_member_ratio"),
        0.98,
        "广度覆盖门",
    )
    _require_equal(
        coverage.get("comovement_minimum_scoreable_member_ratio"),
        0.90,
        "共同运动覆盖门",
    )
    _require_equal(
        coverage.get("comovement_minimum_valid_observations_in_20"),
        15,
        "共同运动观察门",
    )
    _require_equal(
        coverage.get("ordinary_missing_member_may_be_zero_filled"),
        False,
        "普通缺失补零权限",
    )

    probe = config["probe_contract"]
    _require_equal(
        probe.get("must_verify_frozen_addendum_before_network_call"),
        True,
        "探测前冻结验证",
    )
    _require_equal(
        probe.get("full_history_or_bulk_member_download_allowed"),
        False,
        "探测阶段批量下载权限",
    )

    forbidden = set(config["forbidden_actions"])
    required_forbidden = {
        "MUTATE_PARENT_V2_FREEZE",
        "TREAT_PROVIDER_SUSPEND_D_AS_EXCHANGE_OFFICIAL_EVIDENCE",
        "USE_ADJ_FACTOR_TO_BUILD_DIRECT_RETURN_SERIES",
        "FILL_MISSING_CONSTITUENT_RETURN_WITH_ZERO",
        "RUN_BULK_COLLECTION_BEFORE_PROBE_PASS",
        "READ_NEW_AUC_RETURN_SHARPE_DRAWDOWN_NAV_POSITION_OR_ORDER_OUTPUT",
        "GENERATE_PORTFOLIO_POSITION_ORDER_BROKER_OR_LIVE_ACTION",
    }
    if not required_forbidden.issubset(forbidden):
        raise EvidenceContractError(
            f"来源附录禁止动作缺项：{sorted(required_forbidden.difference(forbidden))}"
        )


def verify_dependencies(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = config["parent_freeze"]
    membership = config["point_in_time_membership"]
    legacy = config["legacy_daily_observation_seed"]
    return {
        "parent_manifest": _validate_identity(
            str(parent["manifest_path"]),
            int(parent["manifest_bytes"]),
            str(parent["manifest_sha256"]),
        ),
        "parent_config": _validate_identity(
            str(parent["config_path"]),
            int(parent["config_bytes"]),
            str(parent["config_sha256"]),
        ),
        "point_in_time_membership": _validate_identity(
            str(membership["path"]),
            int(membership["bytes"]),
            str(membership["sha256"]),
        ),
        "legacy_daily_observation_seed": _validate_identity(
            str(legacy["path"]),
            int(legacy["bytes"]),
            str(legacy["sha256"]),
        ),
    }


def _hash_group(paths: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in paths:
        relative = normalize_project_relative_path(raw)
        path = _project_path(relative)
        if not path.is_file():
            raise EvidenceContractError(f"待冻结文件不存在：{relative}")
        result[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def _output_paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    freeze_contract = config["freeze_contract"]
    return (
        _project_path(str(freeze_contract["manifest_output"])),
        _project_path(str(freeze_contract["receipt_output"])),
    )


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, receipt_path = _output_paths(config)
    existing = [str(path) for path in (manifest_path, receipt_path) if path.exists()]
    if existing:
        raise EvidenceContractError(f"冻结输出已存在，禁止覆盖：{existing}")
    dependencies = verify_dependencies(config)
    contract = config["freeze_contract"]
    implementation = _hash_group([str(x) for x in contract["implementation_files"]])
    governance = _hash_group([str(x) for x in contract["governance_files"]])
    frozen_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest: dict[str, Any] = {
        "addendum_id": ADDENDUM_ID,
        "version": config["program"]["version"],
        "status": "FROZEN_BEFORE_ANY_FOUR_STATE_SOURCE_NETWORK_PROBE",
        "frozen_at": frozen_at,
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_bytes": config_path.stat().st_size,
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "implementation_files": implementation,
        "governance_files": governance,
        "dependencies": dependencies,
        "provider_contract": config["provider_contract"],
        "corporate_action_reconciliation": config["corporate_action_reconciliation"],
        "four_state_mapping": config["four_state_mapping"],
        "coverage_and_new_member_contract": config["coverage_and_new_member_contract"],
        "probe_contract": config["probe_contract"],
        "forbidden_actions": config["forbidden_actions"],
        "research_state": "DISCOVERY_ONLY",
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_results_read": False,
        "model_trained": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_SOURCE_FREEZE_V1",
        "created_at": frozen_at,
        "status": "PASS_SOURCE_CONTRACT_FREEZE_PROBE_NOT_YET_RUN",
        "addendum_id": ADDENDUM_ID,
        "manifest": file_evidence(manifest_path, project_root=ROOT),
        "frozen_file_count": len(implementation) + len(governance),
        "checks": [
            "PARENT_V2_FREEZE_IDENTITY",
            "PIT_MEMBERSHIP_IDENTITY",
            "LEGACY_RAW_COLUMN_QUARANTINE",
            "SOURCE_DOCUMENT_CONTRACT",
            "FOUR_STATE_FAIL_CLOSED_MAPPING",
            "NO_BULK_BEFORE_PROBE",
            "NO_PERFORMANCE_OR_TRADING_AUTHORITY",
        ],
        "not_claimed": [
            "PROVIDER_ENDPOINT_AVAILABILITY",
            "HISTORICAL_FOUR_STATE_COVERAGE",
            "G0_OR_LATER_GATE_PASS",
            "PREDICTION_OR_PORTFOLIO_VALUE",
        ],
        "security_audit_performed": False,
        "full_repository_scan_performed": False,
        "return_evaluation": "NOT_ALLOWED",
        "position_impact": 0,
    }
    atomic_write_json_new(receipt_path, receipt)
    print("四态来源准入附录已冻结；仅允许下一步有界接口探测。")
    return manifest


def verify_frozen_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, _ = _output_paths(config)
    manifest = read_json_strict(manifest_path)
    if not isinstance(manifest, dict):
        raise EvidenceContractError("四态来源 manifest 必须是对象")
    verify_manifest_payload(manifest)
    _require_equal(manifest.get("addendum_id"), ADDENDUM_ID, "附录 identity")
    _require_equal(manifest.get("config_sha256"), sha256_file(config_path), "附录配置哈希")
    for group_name in ("implementation_files", "governance_files"):
        group = manifest.get(group_name)
        if not isinstance(group, Mapping):
            raise EvidenceContractError(f"manifest 缺少 {group_name}")
        for relative, evidence in group.items():
            if not isinstance(evidence, Mapping):
                raise EvidenceContractError(f"冻结证据非法：{relative}")
            _validate_identity(
                str(relative), int(evidence["bytes"]), str(evidence["sha256"])
            )
    verify_dependencies(config)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_only:
            verify_frozen_manifest(args.config)
            print("四态来源准入附录 manifest 与依赖哈希验证通过。")
        else:
            freeze(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"四态来源附录冻结失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
