"""冻结 V2 四态历史采集与账本执行包。"""

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
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_sources_v1_0_1 import (
    verify_frozen_manifest as verify_source_correction_manifest,
)


EXECUTION_ID = "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_LEDGER_EXECUTION_V1"
DEFAULT_CONFIG = (
    ROOT
    / "config/510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1.yaml"
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("四态账本执行配置必须是对象")
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise EvidenceContractError(
            f"{label}发生漂移：expected={expected!r}, actual={actual!r}"
        )


def _validate_identity(contract: Mapping[str, Any], label: str) -> dict[str, Any]:
    relative = normalize_project_relative_path(str(contract["path"]))
    path = _project_path(relative)
    if not path.is_file():
        raise EvidenceContractError(f"{label}不存在：{relative}")
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != int(contract["bytes"]) or actual_sha256 != str(
        contract["sha256"]
    ).lower():
        raise EvidenceContractError(f"{label}字节或哈希漂移：{relative}")
    return file_evidence(path, project_root=ROOT)


def validate_config(config: Mapping[str, Any]) -> None:
    program = config["program"]
    expected = {
        "program_id": "510300_STRESS_TRANSMISSION_HAZARD_V2",
        "execution_id": EXECUTION_ID,
        "version": "1.0.0",
        "stage": "BOUNDED_HISTORICAL_ACQUISITION_AND_FOUR_STATE_LEDGER_ONLY",
        "research_state": "DISCOVERY_ONLY",
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_evaluation": "NOT_ALLOWED",
        "model_training_allowed": False,
        "probability_threshold_selection_allowed": False,
        "position_impact": 0,
    }
    for key, value in expected.items():
        _require_equal(program.get(key), value, f"program.{key}")
    _require_equal(
        program.get("executable_assets"),
        ["510300.SH", "CASH_CNY"],
        "可执行资产范围",
    )

    source = config["source_admission_dependencies"]
    _require_equal(
        source.get("selected_endpoint"),
        "https://fast.xiaodefa.cn",
        "已探测节点",
    )
    _require_equal(source.get("alternate_endpoint_fallback_allowed"), False, "节点回退权限")
    _require_equal(source.get("credential_persistence_allowed"), False, "凭据持久化")

    membership = config["inputs"]["point_in_time_membership"]
    _require_equal(membership.get("unique_symbol_count"), 668, "历史证券数")
    _require_equal(membership.get("post_fresh_start_symbol_count"), 493, "新日线证券数")
    _require_equal(membership.get("expected_members_per_session"), 300, "每日成员数")
    _require_equal(membership.get("current_member_backfill_allowed"), False, "当前成员回填权限")

    legacy = config["inputs"]["legacy_unadjusted_daily_seed"]
    prohibited = set(legacy["forbidden_columns"])
    if not {
        "linked_adjustment_factor",
        "total_return_open",
        "total_return_high",
        "total_return_low",
        "total_return_close",
        "is_suspended",
        "adjustment_source",
    }.issubset(prohibited):
        raise EvidenceContractError("旧链禁止字段缺项")

    acquisition = config["acquisition"]
    control = acquisition["request_control"]
    _require_equal(control.get("maximum_workers"), 2, "采集线程数")
    if float(control.get("minimum_delay_seconds_per_request")) < 0.5:
        raise EvidenceContractError("请求最小间隔不得低于 0.5 秒")
    _require_equal(control.get("checkpoint_overwrite_allowed"), False, "断点覆盖权限")
    _require_equal(control.get("output_overwrite_allowed"), False, "输出覆盖权限")
    _require_equal(acquisition["fresh_daily"].get("expected_symbol_count"), 493, "日线请求数")
    _require_equal(
        acquisition["excluded_bulk_sources"]["suspend_d"].get("missing_daily_row_state"),
        "SUPPLIER_MISSING_OR_CONFLICT",
        "停牌候选缺行状态",
    )

    validation = config["daily_validation"]
    _require_equal(validation.get("pct_chg_absolute_tolerance_percentage_points"), 0.011, "涨跌幅容差")
    _require_equal(
        validation["cross_source_overlap"].get("conflict_state"),
        "SUPPLIER_MISSING_OR_CONFLICT",
        "跨来源冲突态",
    )

    action = config["corporate_action_reconciliation"]
    _require_equal(action.get("availability_rule"), "IMP_ANN_DATE_STRICTLY_BEFORE_EX_DATE", "行动可得时钟")
    _require_equal(action.get("subscription_cash_outflow"), 0.0, "认购现金流默认值")
    _require_equal(action.get("rights_issue_terms_available"), False, "配股条款可用性")
    _require_equal(action.get("unresolved_return_value"), None, "未解决行动收益")

    four_state = config["four_state_execution"]
    _require_equal(four_state.get("official_exchange_suspension_source_admitted"), False, "官方停牌源")
    _require_equal(four_state.get("official_suspension_rows_expected"), 0, "官方停牌行数")
    _require_equal(four_state.get("missing_member_day_supplier_conflict"), True, "缺行冲突态")
    _require_equal(four_state.get("blanket_zero_fill_allowed"), False, "缺失补零权限")
    _require_equal(four_state.get("daily_member_coverage_minimum"), 0.98, "成员覆盖门")

    forbidden = set(config["forbidden_actions"])
    required_forbidden = {
        "USE_LEGACY_TOTAL_RETURN_LINKED_ADJUSTMENT_OR_SUSPENSION_COLUMNS",
        "USE_CURRENT_CONSTITUENTS_TO_BACKFILL_HISTORY",
        "TREAT_MISSING_DAILY_ROW_OR_PROVIDER_SUSPEND_D_AS_OFFICIAL_SUSPENSION",
        "FILL_MISSING_OR_UNRESOLVED_RETURN_WITH_ZERO",
        "USE_ADJ_FACTOR_OR_POST_ADJUSTED_PRICE_TO_BUILD_RETURN",
        "RUN_MODEL_TRAINING_OR_THRESHOLD_SELECTION",
        "READ_NEW_AUC_RETURN_SHARPE_DRAWDOWN_NAV_POSITION_OR_ORDER_OUTPUT",
    }
    if not required_forbidden.issubset(forbidden):
        raise EvidenceContractError(
            f"执行禁止动作缺项：{sorted(required_forbidden.difference(forbidden))}"
        )


def verify_dependencies(config: Mapping[str, Any]) -> dict[str, Any]:
    verify_source_correction_manifest()
    dependencies = config["source_admission_dependencies"]
    inputs = config["inputs"]
    correction = _validate_identity(dependencies["correction_manifest"], "来源修正 manifest")
    probe = _validate_identity(dependencies["passed_probe_receipt"], "通过的来源探测收据")
    probe_payload = read_json_strict(_project_path(probe["path"]))
    if not isinstance(probe_payload, dict) or not str(probe_payload.get("status", "")).startswith("PASS_"):
        raise EvidenceContractError("来源探测收据不是通过状态")
    if probe_payload.get("performance_values_read") is not False:
        raise EvidenceContractError("来源探测收据的绩效读取状态非法")
    return {
        "correction_manifest": correction,
        "probe_receipt": probe,
        "membership": _validate_identity(inputs["point_in_time_membership"], "点时成员"),
        "legacy_seed": _validate_identity(inputs["legacy_unadjusted_daily_seed"], "旧未复权日线种子"),
    }


def _hash_group(paths: list[str]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for raw in paths:
        relative = normalize_project_relative_path(raw)
        path = _project_path(relative)
        if not path.is_file():
            raise EvidenceContractError(f"待冻结文件不存在：{relative}")
        evidence[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return evidence


def _outputs(config: Mapping[str, Any]) -> tuple[Path, Path]:
    contract = config["freeze_contract"]
    return (
        _project_path(str(contract["manifest_output"])),
        _project_path(str(contract["receipt_output"])),
    )


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, receipt_path = _outputs(config)
    existing = [str(path) for path in (manifest_path, receipt_path) if path.exists()]
    if existing:
        raise EvidenceContractError(f"执行冻结输出已存在，禁止覆盖：{existing}")
    dependencies = verify_dependencies(config)
    contract = config["freeze_contract"]
    implementation = _hash_group([str(x) for x in contract["implementation_files"]])
    governance = _hash_group([str(x) for x in contract["governance_files"]])
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest: dict[str, Any] = {
        "execution_id": EXECUTION_ID,
        "version": "1.0.0",
        "status": "FROZEN_BEFORE_HISTORICAL_NETWORK_ACQUISITION",
        "frozen_at": now,
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_bytes": config_path.stat().st_size,
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "dependencies": dependencies,
        "implementation_files": implementation,
        "governance_files": governance,
        "acquisition_contract": config["acquisition"],
        "daily_validation": config["daily_validation"],
        "corporate_action_reconciliation": config["corporate_action_reconciliation"],
        "four_state_execution": config["four_state_execution"],
        "forbidden_actions": config["forbidden_actions"],
        "return_evaluation": "NOT_ALLOWED",
        "model_trained": False,
        "portfolio_results_read": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_LEDGER_EXECUTION_V1_FREEZE",
        "created_at": now,
        "status": "PASS_EXECUTION_CONTRACT_FREEZE_NO_NETWORK_RUN_YET",
        "manifest": file_evidence(manifest_path, project_root=ROOT),
        "frozen_file_count": len(implementation) + len(governance),
        "network_acquisition_run": False,
        "four_state_ledger_built": False,
        "performance_values_read": False,
        "security_audit_performed": False,
        "position_impact": 0,
    }
    atomic_write_json_new(receipt_path, receipt)
    print("四态历史采集与账本执行包已冻结；允许按断点契约采集。")
    return manifest


def verify_frozen_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, _ = _outputs(config)
    manifest = read_json_strict(manifest_path)
    if not isinstance(manifest, dict):
        raise EvidenceContractError("执行 manifest 必须是对象")
    verify_manifest_payload(manifest)
    _require_equal(manifest.get("execution_id"), EXECUTION_ID, "执行 identity")
    _require_equal(manifest.get("config_sha256"), sha256_file(config_path), "执行配置哈希")
    verify_dependencies(config)
    for group_name in ("implementation_files", "governance_files"):
        group = manifest.get(group_name)
        if not isinstance(group, Mapping):
            raise EvidenceContractError(f"执行 manifest 缺少 {group_name}")
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
            print("四态历史执行 manifest 与依赖哈希验证通过。")
        else:
            freeze(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError) as exc:
        print(f"四态历史执行冻结失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
