"""冻结 510300 压力传导危险率 V2 的 M/F/T 点时特征执行 V1。"""

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
from scripts.freeze_510300_stress_transmission_hazard_v2 import (
    verify_frozen_manifest as verify_v2_protocol_manifest,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_four_state_ledger_execution_v1_0_1 import (
    verify_frozen_manifest as verify_four_state_execution_manifest,
)


EXECUTION_ID = "510300_STRESS_TRANSMISSION_HAZARD_V2_MFT_FEATURE_EXECUTION_V1"
DEFAULT_CONFIG = (
    ROOT / "config/510300_stress_transmission_hazard_v2_mft_feature_execution_v1.yaml"
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("M/F/T 执行配置必须是对象")
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
    expected_bytes = int(contract["bytes"])
    expected_sha = str(contract["sha256"])
    if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
        raise EvidenceContractError(f"{label}字节或哈希漂移：{path}")
    return file_evidence(path, project_root=ROOT)


def validate_config(config: Mapping[str, Any]) -> None:
    program = config["program"]
    _require_equal(
        program.get("program_id"),
        "510300_STRESS_TRANSMISSION_HAZARD_V2",
        "项目 identity",
    )
    _require_equal(program.get("execution_id"), EXECUTION_ID, "执行 identity")
    _require_equal(program.get("version"), "1.0.0", "执行版本")
    _require_equal(program.get("stage_id"), "POINT_IN_TIME_M_F_T_FEATURE_BUILD", "阶段")
    _require_equal(program.get("research_state"), "DISCOVERY_ONLY", "研究状态")
    _require_equal(program.get("return_evaluation"), "NOT_ALLOWED", "收益评估权限")
    _require_equal(program.get("label_read_allowed"), False, "标签读取权限")
    _require_equal(program.get("model_training_allowed"), False, "模型训练权限")
    _require_equal(program.get("portfolio_evaluation_allowed"), False, "组合评估权限")
    _require_equal(program.get("position_impact"), 0, "仓位影响")
    _require_equal(program.get("executable_assets"), ["510300.SH", "CASH_CNY"], "可执行资产")
    _require_equal(program.get("observation_start"), "2015-01-05", "观察起点")
    _require_equal(program.get("observation_cutoff"), "2026-08-14", "观察截止")

    clock = config["information_clock"]
    _require_equal(clock.get("decision_clock"), "T_CLOSE_15_00_ASIA_SHANGHAI", "决策时钟")
    _require_equal(
        clock["csi300_official_pe"].get("rule"),
        "NEXT_PIT_MARKET_SESSION_OPEN_AFTER_OBSERVATION_DATE",
        "PE 可得时钟",
    )
    _require_equal(
        clock["china_10y_yield"].get("rule"),
        "NEXT_PIT_MARKET_SESSION_OPEN_AFTER_OBSERVATION_DATE",
        "国债可得时钟",
    )
    _require_equal(
        clock["dr007"].get("rule"),
        "NEXT_PIT_MARKET_SESSION_OPEN_AFTER_RATE_DATE",
        "DR007 可得时钟",
    )
    _require_equal(
        clock["reverse_repo_7d_policy_rate"].get("rule"),
        "EXACT_ARCHIVED_PUBLISHED_AT_ASIA_SHANGHAI",
        "逆回购可得时钟",
    )
    _require_equal(
        clock.get("missing_release_rule"),
        "NO_VIEW_NO_INTERPOLATION_NO_ZERO_FILL",
        "宏观缺失规则",
    )

    feature = config["feature_contract"]
    _require_equal(feature.get("member_coverage_minimum"), 0.98, "成员覆盖门")
    _require_equal(feature.get("comovement_member_ratio_minimum"), 0.90, "共同运动成员门")
    _require_equal(feature.get("comovement_minimum_valid_observations"), 15, "共同运动观察门")
    _require_equal(feature.get("lookback_market_days"), 20, "主回看窗口")
    _require_equal(feature.get("change_market_days"), 5, "变化窗口")
    _require_equal(feature.get("tail_volatility_market_days"), 60, "左尾波动窗口")
    _require_equal(feature.get("tail_sigma_multiple"), 1.5, "左尾倍数")
    _require_equal(feature.get("reliable_point_in_time_weight_used"), False, "点时权重使用")

    history = config["new_member_history_contract"]
    _require_equal(history["price_history_before_index_entry"].get("allowed"), True, "入指前价格历史")
    _require_equal(history["price_history_before_index_entry"].get("future_price_allowed"), False, "未来价格")
    _require_equal(history["price_history_before_index_entry"].get("synthetic_history_allowed"), False, "合成历史")
    _require_equal(history["comovement_history"].get("trailing_market_day_window"), 20, "成员日窗口")
    _require_equal(history["comovement_history"].get("minimum_valid_member_days"), 15, "最少成员日")
    _require_equal(history["comovement_history"].get("nonmember_days_count_as_valid"), False, "非成员日")
    _require_equal(history["comovement_history"].get("current_constituent_backfill_allowed"), False, "当前成员回填")

    no_view = config["no_view_chain"]
    _require_equal(no_view.get("final_model_state_requires"), ["M", "F", "T"], "最终视图要求")
    _require_equal(no_view.get("abstain_implies_position"), False, "NO_VIEW 仓位语义")

    forbidden = set(config["forbidden_actions"])
    required_forbidden = {
        "USE_CURRENT_CONSTITUENTS_TO_BACKFILL_HISTORY",
        "USE_SAME_DAY_PE_BOND_OR_DR007_OBSERVATION",
        "USE_RELEASE_PUBLISHED_AFTER_T_CLOSE",
        "INTERPOLATE_OR_ZERO_FILL_MISSING_FEATURE_CHANNELS",
        "INCLUDE_INDUSTRY_SECTOR_OR_SHENWAN_FIELDS_IN_CORE",
        "READ_BAD10_LABEL_OR_ANY_FUTURE_RETURN_ARTIFACT",
        "TRAIN_B0_B1_B2_OR_B3",
        "READ_OR_GENERATE_SHARPE_DRAWDOWN_NAV_POSITION_OR_ORDER",
    }
    if not required_forbidden.issubset(forbidden):
        raise EvidenceContractError(
            f"M/F/T 禁止动作缺项：{sorted(required_forbidden.difference(forbidden))}"
        )

    output_values = list(config["outputs"].values())
    if len(output_values) != len(set(output_values)):
        raise EvidenceContractError("M/F/T 输出路径存在重复")


def verify_parent_chain(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    verify_v2_protocol_manifest()
    verify_four_state_execution_manifest()
    parent = config["parent_chain"]
    evidence = {
        "v2_protocol_manifest": _validate_file_identity(
            parent["v2_protocol_manifest"], "V2 协议 manifest"
        ),
        "four_state_execution_manifest": _validate_file_identity(
            parent["four_state_execution_manifest"], "四态执行 manifest"
        ),
        "four_state_ledger_receipt": _validate_file_identity(
            parent["four_state_ledger_receipt"], "四态账本收据"
        ),
        "four_state_status": _validate_file_identity(
            parent["four_state_status"], "四态状态"
        ),
    }
    receipt = read_json_strict(_project_path(parent["four_state_ledger_receipt"]["path"]))
    _require_equal(
        receipt.get("status"),
        parent["four_state_ledger_receipt"]["required_status"],
        "四态账本状态",
    )
    _require_equal(receipt.get("return_evaluation"), "NOT_ALLOWED", "四态收益评估权限")
    status = read_json_strict(_project_path(parent["four_state_status"]["path"]))
    _require_equal(
        status.get("next_allowed_step"),
        parent["four_state_status"]["required_next_step"],
        "四态下一允许步骤",
    )
    _require_equal(status.get("g1_through_g7_status"), "NOT_RUN", "四态后续门状态")
    return evidence


def verify_input_identities(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: _validate_file_identity(contract, f"输入 {name}")
        for name, contract in config["inputs"].items()
    }


def _hash_group(paths: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in paths:
        relative = normalize_project_relative_path(raw)
        path = _project_path(relative)
        if not path.is_file():
            raise EvidenceContractError(f"待冻结文件不存在：{relative}")
        result[relative] = file_evidence(path, project_root=ROOT)
        result[relative].pop("path")
    return result


def _freeze_outputs(config: Mapping[str, Any]) -> tuple[Path, Path]:
    contract = config["freeze_contract"]
    return (
        _project_path(str(contract["manifest_output"])),
        _project_path(str(contract["freeze_receipt_output"])),
    )


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, receipt_path = _freeze_outputs(config)
    existing = [str(path) for path in (manifest_path, receipt_path) if path.exists()]
    if existing:
        raise EvidenceContractError(f"M/F/T 冻结输出已存在，禁止覆盖：{existing}")
    parent_evidence = verify_parent_chain(config)
    input_evidence = verify_input_identities(config)
    contract = config["freeze_contract"]
    implementation = _hash_group([str(x) for x in contract["implementation_files"]])
    governance = _hash_group([str(x) for x in contract["governance_files"]])
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest: dict[str, Any] = {
        "execution_id": EXECUTION_ID,
        "version": "1.0.0",
        "status": "FROZEN_BEFORE_POINT_IN_TIME_FEATURE_VALUES",
        "frozen_at": now,
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_bytes": config_path.stat().st_size,
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "parent_chain": parent_evidence,
        "input_files": input_evidence,
        "implementation_files": implementation,
        "governance_files": governance,
        "information_clock": config["information_clock"],
        "feature_contract": config["feature_contract"],
        "new_member_history_contract": config["new_member_history_contract"],
        "no_view_chain": config["no_view_chain"],
        "outputs": config["outputs"],
        "forbidden_actions": config["forbidden_actions"],
        "actual_label_artifact_read": False,
        "actual_performance_artifact_read": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_MFT_FEATURE_EXECUTION_V1_FREEZE",
        "created_at": now,
        "status": "PASS_MFT_RULES_CLOCKS_INPUTS_AND_NO_VIEW_CHAIN_FROZEN_BEFORE_VALUES",
        "manifest": file_evidence(manifest_path, project_root=ROOT),
        "parent_chain_verified": True,
        "actual_feature_values_built": False,
        "actual_label_artifact_read": False,
        "actual_performance_artifact_read": False,
        "position_impact": 0,
    }
    atomic_write_json_new(receipt_path, receipt)
    print("M/F/T 特征规则、时钟、输入与 NO_VIEW 链已在数值构建前冻结。")
    return manifest


def verify_frozen_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, _ = _freeze_outputs(config)
    manifest = read_json_strict(manifest_path)
    if not isinstance(manifest, dict):
        raise EvidenceContractError("M/F/T manifest 必须是对象")
    verify_manifest_payload(manifest)
    _require_equal(manifest.get("execution_id"), EXECUTION_ID, "执行 identity")
    _require_equal(manifest.get("config_sha256"), sha256_file(config_path), "配置哈希")
    verify_parent_chain(config)
    verify_input_identities(config)
    for group_name in ("implementation_files", "governance_files"):
        group = manifest.get(group_name)
        if not isinstance(group, Mapping):
            raise EvidenceContractError(f"M/F/T manifest 缺少 {group_name}")
        for relative, evidence in group.items():
            _validate_file_identity(
                {
                    "path": relative,
                    "bytes": evidence["bytes"],
                    "sha256": evidence["sha256"],
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
            print("M/F/T 冻结 manifest、父链与输入身份验证通过。")
        else:
            freeze(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError) as exc:
        print(f"M/F/T 执行冻结失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
