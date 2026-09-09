"""冻结 V2 四态账本执行 V1.0.1 证券选择器修正。"""

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
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1 import (
    verify_frozen_manifest as verify_v1_execution_manifest,
)


CORRECTION_ID = "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_LEDGER_EXECUTION_V1_0_1"
DEFAULT_CONFIG = (
    ROOT
    / "config/510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1_0_1.yaml"
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("V1.0.1 执行修正配置必须是对象")
    return payload


def _require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise EvidenceContractError(
            f"{label}发生漂移：expected={expected!r}, actual={actual!r}"
        )


def _validate_identity(contract: Mapping[str, Any], label: str) -> dict[str, Any]:
    path = _project_path(str(contract["path"]))
    if not path.is_file():
        raise EvidenceContractError(f"{label}不存在：{path}")
    if path.stat().st_size != int(contract["bytes"]) or sha256_file(path) != str(
        contract["sha256"]
    ):
        raise EvidenceContractError(f"{label}字节或哈希漂移：{path}")
    return file_evidence(path, project_root=ROOT)


def validate_config(config: Mapping[str, Any]) -> None:
    program = config["program"]
    _require_equal(program.get("program_id"), "510300_STRESS_TRANSMISSION_HAZARD_V2", "项目")
    _require_equal(program.get("correction_id"), CORRECTION_ID, "修正 identity")
    _require_equal(program.get("version"), "1.0.1", "修正版本")
    _require_equal(program.get("correction_scope"), "FRESH_DAILY_SYMBOL_UNIVERSE_SELECTOR_ONLY", "修正范围")
    _require_equal(program.get("return_evaluation"), "NOT_ALLOWED", "收益评估权限")
    _require_equal(program.get("position_impact"), 0, "仓位影响")
    _require_equal(config["predecessor_execution_freeze"].get("mutation_allowed"), False, "V1 修改权限")

    failure = config["predecessor_execution_failure"]
    _require_equal(failure.get("status"), "BLOCKED_BEFORE_NETWORK_FRESH_DAILY_SYMBOL_COUNT_MISMATCH", "V1 失败状态")
    _require_equal(failure.get("network_requests_sent"), 0, "V1 网络请求数")
    _require_equal(failure.get("checkpoints_written"), 0, "V1 断点数")
    _require_equal(failure.get("performance_values_read"), False, "V1 绩效读取")

    selector = config["selector_correction"]
    _require_equal(selector.get("legacy_seed_last_date"), "2020-02-28", "旧种子截止日")
    _require_equal(selector.get("fresh_query_start_date"), "2019-12-01", "新查询起点")
    _require_equal(
        selector.get("corrected_symbol_universe_rule"),
        "ANY_PIT_MEMBERSHIP_STRICTLY_AFTER_LEGACY_SEED_LAST_DATE",
        "修正选择器",
    )
    _require_equal(selector.get("corrected_expected_symbol_count"), 493, "修正证券数")
    _require_equal(selector.get("excluded_overlap_only_symbol_count"), 12, "重叠专属证券数")
    if len(selector.get("excluded_overlap_only_symbols", [])) != 12:
        raise EvidenceContractError("重叠专属证券清单不是 12 个")
    _require_equal(selector.get("excluded_symbols_fully_covered_by_legacy_seed"), True, "旧种子覆盖")
    _require_equal(selector.get("current_constituent_panel_used_to_select_universe"), False, "当前成分选择权限")
    _require_equal(selector.get("point_in_time_membership_used_to_select_universe"), True, "点时成员选择")

    unchanged = config["unchanged_contracts"]
    _require_equal(unchanged.get("selected_endpoint"), "https://fast.xiaodefa.cn", "端点")
    _require_equal(unchanged.get("maximum_workers"), 2, "线程")
    _require_equal(unchanged.get("checkpoint_overwrite_allowed"), False, "断点覆盖")
    _require_equal(unchanged.get("missing_return_zero_fill_allowed"), False, "缺失补零")
    _require_equal(unchanged.get("model_training_allowed"), False, "模型训练")
    _require_equal(unchanged.get("performance_read_allowed"), False, "绩效读取")
    _require_equal(unchanged.get("position_impact"), 0, "仓位影响")

    forbidden = set(config["forbidden_actions"])
    required_forbidden = {
        "MODIFY_OR_DELETE_V1_EXECUTION_FREEZE",
        "CHANGE_ANY_RULE_OTHER_THAN_FRESH_DAILY_SYMBOL_SELECTOR_AND_OUTPUT_NAMESPACE",
        "USE_CURRENT_CONSTITUENT_PANEL_TO_SELECT_HISTORY",
        "FILL_MISSING_OR_UNRESOLVED_RETURN_WITH_ZERO",
        "RUN_MODEL_TRAINING_OR_THRESHOLD_SELECTION",
        "READ_NEW_AUC_RETURN_SHARPE_DRAWDOWN_NAV_POSITION_OR_ORDER_OUTPUT",
    }
    if not required_forbidden.issubset(forbidden):
        raise EvidenceContractError(
            f"V1.0.1 执行禁止动作缺项：{sorted(required_forbidden.difference(forbidden))}"
        )


def verify_predecessor(config: Mapping[str, Any]) -> dict[str, Any]:
    verify_v1_execution_manifest()
    predecessor = config["predecessor_execution_freeze"]
    return _validate_identity(
        {
            "path": predecessor["manifest_path"],
            "bytes": predecessor["manifest_bytes"],
            "sha256": predecessor["manifest_sha256"],
        },
        "V1 执行 manifest",
    )


def _hash_group(paths: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in paths:
        relative = normalize_project_relative_path(raw)
        path = _project_path(relative)
        if not path.is_file():
            raise EvidenceContractError(f"待冻结文件不存在：{relative}")
        result[relative] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def _outputs(config: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    freeze_contract = config["freeze_contract"]
    failure = config["predecessor_execution_failure"]
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
        raise EvidenceContractError(f"V1.0.1 执行输出已存在，禁止覆盖：{existing}")
    predecessor = verify_predecessor(config)
    contract = config["freeze_contract"]
    implementation = _hash_group([str(x) for x in contract["implementation_files"]])
    governance = _hash_group([str(x) for x in contract["governance_files"]])
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()

    failure_receipt = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_INPUT_ACQUISITION_V1_FAILURE",
        "recorded_at": now,
        "observed_on": config["predecessor_execution_failure"]["observed_on"],
        "status": config["predecessor_execution_failure"]["status"],
        "expected_symbol_count": config["predecessor_execution_failure"]["expected_by_v1"],
        "actual_symbol_count": config["predecessor_execution_failure"]["actual_under_v1_selector"],
        "network_requests_sent": 0,
        "checkpoints_written": 0,
        "performance_values_read": False,
        "position_impact": 0,
        "next_action": "APPEND_ONLY_SELECTOR_CORRECTION_V1_0_1",
    }
    atomic_write_json_new(failure_path, failure_receipt)

    manifest: dict[str, Any] = {
        "correction_id": CORRECTION_ID,
        "version": "1.0.1",
        "status": "FROZEN_BEFORE_CORRECTED_HISTORICAL_ACQUISITION",
        "frozen_at": now,
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_bytes": config_path.stat().st_size,
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "predecessor_execution_manifest": predecessor,
        "predecessor_failure_receipt": file_evidence(failure_path, project_root=ROOT),
        "implementation_files": implementation,
        "governance_files": governance,
        "selector_correction": config["selector_correction"],
        "corrected_outputs": config["corrected_outputs"],
        "unchanged_contracts": config["unchanged_contracts"],
        "forbidden_actions": config["forbidden_actions"],
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_results_read": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_FOUR_STATE_LEDGER_EXECUTION_V1_0_1_FREEZE",
        "created_at": now,
        "status": "PASS_APPEND_ONLY_FRESH_DAILY_SELECTOR_CORRECTION_FREEZE",
        "manifest": file_evidence(manifest_path, project_root=ROOT),
        "predecessor_failure_receipt": file_evidence(failure_path, project_root=ROOT),
        "scope_expanded": False,
        "network_acquisition_run": False,
        "performance_values_read": False,
        "position_impact": 0,
    }
    atomic_write_json_new(receipt_path, receipt)
    print("V1.0.1 新日线证券选择器修正已冻结；允许重新启动断点采集。")
    return manifest


def verify_frozen_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, _, failure_path = _outputs(config)
    manifest = read_json_strict(manifest_path)
    if not isinstance(manifest, dict):
        raise EvidenceContractError("V1.0.1 执行 manifest 必须是对象")
    verify_manifest_payload(manifest)
    _require_equal(manifest.get("correction_id"), CORRECTION_ID, "修正 identity")
    _require_equal(manifest.get("config_sha256"), sha256_file(config_path), "修正配置哈希")
    verify_predecessor(config)
    failure_evidence = manifest["predecessor_failure_receipt"]
    _validate_identity(failure_evidence, "V1 失败收据")
    _require_equal(failure_path, _project_path(str(failure_evidence["path"])), "失败收据路径")
    for group_name in ("implementation_files", "governance_files"):
        group = manifest.get(group_name)
        if not isinstance(group, Mapping):
            raise EvidenceContractError(f"V1.0.1 执行 manifest 缺少 {group_name}")
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
            print("V1.0.1 执行修正 manifest 与前序证据验证通过。")
        else:
            freeze(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError) as exc:
        print(f"V1.0.1 执行修正冻结失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
