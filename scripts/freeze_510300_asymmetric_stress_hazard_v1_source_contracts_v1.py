"""在构造任何 M/F/T 特征值前冻结点时来源准入协议。"""

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
    git_head,
    normalize_project_relative_path,
    read_json_strict,
    sha256_file,
    validate_authority_state,
    validate_registered_inputs,
    verify_manifest_payload,
)


DEFAULT_CONFIG = (
    ROOT / "config/510300_asymmetric_stress_hazard_v1_source_contracts_v1.yaml"
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("来源准入配置必须是对象")
    return payload


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _mapping_json(path: Path, *, name: str) -> dict[str, Any]:
    payload = read_json_strict(path)
    if not isinstance(payload, dict):
        raise EvidenceContractError(f"{name} 必须是 JSON 对象")
    return payload


def _verify_receipt_self_hash(receipt: Mapping[str, Any], *, name: str) -> str:
    expected = str(receipt.get("receipt_payload_sha256", ""))
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256", None)
    actual = canonical_sha256(payload)
    if expected != actual:
        raise EvidenceContractError(
            f"{name} 自身哈希失败：expected={expected}, actual={actual}"
        )
    return expected


def validate_source_contract(
    config: Mapping[str, Any], parent_config: Mapping[str, Any]
) -> None:
    program = config["program"]
    expected_program = {
        "program_id": "510300_ASYMMETRIC_STRESS_HAZARD_V1",
        "stage_id": "POINT_IN_TIME_M_F_T_SOURCE_ADMISSION_V1",
        "research_state": "DISCOVERY_ONLY",
        "position_impact": 0,
        "observation_start": "2015-01-05",
        "observation_cutoff": "2026-08-14",
        "feature_construction_allowed_before_source_admission": False,
        "model_training_allowed": False,
        "portfolio_evaluation_allowed": False,
        "paper_or_shadow_signal_allowed": False,
        "broker_connection_allowed": False,
        "order_generation_allowed": False,
        "live_trading_allowed": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": program.get(key)}
        for key, expected in expected_program.items()
        if str(program.get(key)) != expected
        if key in {"observation_start", "observation_cutoff"}
    }
    mismatches.update(
        {
            key: {"expected": expected, "actual": program.get(key)}
            for key, expected in expected_program.items()
            if key not in {"observation_start", "observation_cutoff"}
            and program.get(key) != expected
        }
    )
    if mismatches:
        raise EvidenceContractError(f"来源准入程序边界漂移：{mismatches}")

    prerequisite = config["prerequisite"]
    if prerequisite.get("required_g1_status") != (
        "PASS_G1_LABEL_IDENTIFIABILITY_ONLY_NO_FEATURE_CONSTRUCTION"
    ):
        raise EvidenceContractError("来源准入的 G1 前置状态发生漂移")
    if prerequisite.get("required_next_action") != (
        "FREEZE_AND_ADMIT_POINT_IN_TIME_M_F_T_SOURCE_CONTRACTS"
    ):
        raise EvidenceContractError("来源准入的前置允许动作发生漂移")
    if prerequisite.get("label_artifacts_may_be_read") is not False:
        raise EvidenceContractError("来源准入不得读取 BAD10 标签明细")
    if prerequisite.get("portfolio_artifacts_may_be_read") is not False:
        raise EvidenceContractError("来源准入不得读取组合结果")

    parent_program = parent_config["program"]
    if parent_program.get("program_id") != program["program_id"]:
        raise EvidenceContractError("来源准入与父协议 program_id 不一致")
    if str(parent_program.get("observation_start")) != str(
        program["observation_start"]
    ) or str(parent_program.get("observation_cutoff")) != str(
        program["observation_cutoff"]
    ):
        raise EvidenceContractError("来源准入观察窗与父协议不一致")
    label = parent_config["bad10_label"]
    frozen_label = {
        "execution_clock": "NEXT_TRADING_DAY_OPEN",
        "entry_asset": "510300.SH",
        "horizon_market_days": 10,
        "loss_threshold": -0.04,
        "same_day_close_execution_allowed": False,
        "future_low_execution_allowed": False,
        "index_proxy_allowed": False,
    }
    label_drift = {
        key: {"expected": expected, "actual": label.get(key)}
        for key, expected in frozen_label.items()
        if label.get(key) != expected
    }
    if label_drift:
        raise EvidenceContractError(f"父协议 BAD10 标签发生漂移：{label_drift}")

    channels = parent_config["feature_charter"]
    macro = channels["macro_vulnerability_M"]["channels"]
    internal = channels["internal_vulnerability_F"]["channels"]
    transmission = channels["transmission_T"]["channels"]
    expected_formulas = {
        "M1": (
            macro["M1_ERP_BUFFER"].get("formula"),
            "CSI300_OFFICIAL_EARNINGS_YIELD_MINUS_CHINA_10Y_GOVERNMENT_BOND_YIELD",
        ),
        "M2": (
            macro["M2_FUNDING_STRESS"].get("formula"),
            "DR007_MINUS_7D_REVERSE_REPO_POLICY_RATE",
        ),
        "M3": (
            macro["M3_CREDIT_IMPULSE"].get("formula"),
            "THREE_MONTH_CHANGE_IN_SOCIAL_FINANCING_STOCK_YOY_GROWTH",
        ),
        "F1": (
            internal["F1_BREADTH20"].get("formula"),
            "SHARE_OF_POINT_IN_TIME_CSI300_MEMBERS_WITH_POSITIVE_20D_TOTAL_RETURN",
        ),
        "F2": (
            internal["F2_LEADERSHIP_GAP20"].get("formula"),
            "510300_20D_TOTAL_RETURN_MINUS_POINT_IN_TIME_MEMBER_EQUAL_WEIGHT_20D_TOTAL_RETURN",
        ),
        "F3": (
            internal["F3_SECTOR_CORR20"].get("formula"),
            "MEDIAN_PAIRWISE_CORRELATION_OF_POINT_IN_TIME_SECTOR_DAILY_RETURNS_OVER_20D",
        ),
        "T1": (
            transmission["T1_NEGATIVE_DELTA_BREADTH5"].get("formula"),
            "NEGATIVE_OF_BREADTH20_T_MINUS_BREADTH20_T_MINUS_5",
        ),
        "T2": (
            transmission["T2_DELTA_CORR5"].get("formula"),
            "SECTOR_CORR20_T_MINUS_SECTOR_CORR20_T_MINUS_5",
        ),
        "T3": (
            transmission["T3_NEGATIVE_SECTOR_DIFFUSION5"].get("formula"),
            "SHARE_OF_POINT_IN_TIME_SECTORS_WITH_NEGATIVE_5D_RETURN",
        ),
    }
    formula_drift = {
        channel: {"expected": expected, "actual": actual}
        for channel, (actual, expected) in expected_formulas.items()
        if actual != expected
    }
    if formula_drift:
        raise EvidenceContractError(f"父协议 M/F/T 公式发生漂移：{formula_drift}")
    funding = macro["M2_FUNDING_STRESS"]
    if funding.get("selected_rate") != "DR007":
        raise EvidenceContractError("父协议 M2 必须冻结为 DR007")
    if funding.get("alternative_rates_forbidden") != ["FDR007", "R007"]:
        raise EvidenceContractError("父协议 M2 禁止替代清单发生漂移")

    missing = config["missing_required_sources"]
    dr007 = missing["dr007_daily"]
    policy = missing["reverse_repo_policy_rate_7d"]
    if dr007.get("selected_series") != "DR007" or dr007.get(
        "frozen_input_present"
    ) is not False:
        raise EvidenceContractError("DR007 缺失来源契约不一致")
    if dr007.get("forbidden_substitutes") != [
        "FDR007",
        "R007",
        "FR007",
        "EXCHANGE_REPO_R_007",
    ]:
        raise EvidenceContractError("DR007 禁止替代清单发生漂移")
    if policy.get("selected_series") != "PBOC_7D_REVERSE_REPO_OPERATION_RATE":
        raise EvidenceContractError("央行 7 天逆回购政策利率身份发生漂移")
    if policy.get("frozen_input_present") is not False:
        raise EvidenceContractError("央行 7 天逆回购政策利率缺失状态不一致")
    if policy.get("interpolation_allowed") is not False or policy.get(
        "inferred_change_dates_allowed"
    ) is not False:
        raise EvidenceContractError("央行政策利率不得插值或推断变更日")

    gates = config["admission_gates"]
    if gates.get("all_required_sources_must_be_admitted") is not True:
        raise EvidenceContractError("来源准入必须要求全部必需来源通过")
    if gates.get("feature_construction_on_blocked_status_allowed") is not False:
        raise EvidenceContractError("来源阻断后不得构造特征")
    if gates.get("model_training_on_source_admission_allowed") is not False:
        raise EvidenceContractError("来源准入阶段不得训练模型")
    if gates.get("portfolio_evaluation_on_source_admission_allowed") is not False:
        raise EvidenceContractError("来源准入阶段不得评价组合")
    if gates.get("rescue_with_proxy_or_alternative_series_allowed") is not False:
        raise EvidenceContractError("来源准入失败不得用代理救援")
    if config["availability_clock"].get("missing_value_rule") != (
        "NO_VIEW_NO_INTERPOLATION"
    ):
        raise EvidenceContractError("缺失值规则必须为 NO_VIEW_NO_INTERPOLATION")


def _hash_files(paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in paths:
        relative = normalize_project_relative_path(value)
        path = _project_path(relative)
        if not path.is_file():
            raise EvidenceContractError(f"来源冻结文件缺失：{relative}")
        result[relative] = sha256_file(path)
    return result


def _registered_without_resolved_path(record: Any) -> dict[str, Any]:
    payload = record.to_dict()
    payload.pop("resolved_path", None)
    return payload


def freeze(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    prerequisite = config["prerequisite"]
    parent_config_path = _project_path(str(prerequisite["parent_protocol_config"]))
    parent_config = load_config(parent_config_path)
    validate_source_contract(config, parent_config)

    authority_path = _project_path(str(config["authority_contract"]["path"]))
    authority = _mapping_json(authority_path, name="权威状态")
    validate_authority_state(authority)
    if authority.get("authority_id") != config["authority_contract"]["authority_id"]:
        raise EvidenceContractError("权威状态 identity 不一致")

    parent_manifest_path = _project_path(
        str(prerequisite["parent_protocol_manifest"])
    )
    parent_manifest = _mapping_json(parent_manifest_path, name="父协议 manifest")
    verify_manifest_payload(parent_manifest)
    if sha256_file(parent_config_path) != parent_manifest.get("config_sha256"):
        raise EvidenceContractError("父协议配置与父 manifest 不一致")

    parent_status_path = _project_path(str(prerequisite["census_status"]))
    parent_receipt_path = _project_path(str(prerequisite["census_receipt"]))
    parent_status = _mapping_json(parent_status_path, name="G1 权威状态")
    parent_receipt = _mapping_json(parent_receipt_path, name="G1 不可变收据")
    parent_receipt_hash = _verify_receipt_self_hash(
        parent_receipt, name="G1 不可变收据"
    )
    if parent_status.get("status") != prerequisite["required_g1_status"]:
        raise EvidenceContractError("G1 权威状态不允许来源准入")
    if parent_status.get("next_allowed_action") != prerequisite[
        "required_next_action"
    ]:
        raise EvidenceContractError("G1 权威状态的 next_allowed_action 不一致")
    if parent_status.get("feature_values_constructed") is not False:
        raise EvidenceContractError("G1 权威状态已越界构造特征")
    if parent_receipt.get("g1_passed") is not True:
        raise EvidenceContractError("G1 不可变收据没有记录通过")
    if parent_receipt.get("g1_status") != prerequisite["required_g1_status"]:
        raise EvidenceContractError("G1 不可变收据状态不一致")
    if parent_receipt.get("feature_values_constructed") is not False:
        raise EvidenceContractError("G1 不可变收据已越界构造特征")
    parent_manifest_hash = str(parent_manifest["manifest_payload_sha256"])
    if parent_status.get("protocol_manifest_payload_sha256") != parent_manifest_hash:
        raise EvidenceContractError("G1 权威状态与父 manifest 不一致")
    if parent_receipt.get("protocol_manifest_payload_sha256") != parent_manifest_hash:
        raise EvidenceContractError("G1 不可变收据与父 manifest 不一致")
    status_evidence = parent_receipt.get("outputs", {}).get(
        "authoritative_status"
    )
    if not isinstance(status_evidence, Mapping):
        raise EvidenceContractError("G1 收据缺少权威状态证据")
    if sha256_file(parent_status_path) != status_evidence.get("sha256"):
        raise EvidenceContractError("G1 权威状态哈希与不可变收据不一致")
    if parent_status_path.stat().st_size != int(status_evidence.get("bytes", -1)):
        raise EvidenceContractError("G1 权威状态字节数与不可变收据不一致")

    registered = validate_registered_inputs(config, project_root=ROOT)
    freeze_contract = config["freeze_contract"]
    implementation_files = _hash_files(
        [str(value) for value in freeze_contract["implementation_files"]]
    )
    governance_files = _hash_files(
        [str(value) for value in freeze_contract["governance_files"]]
    )

    manifest_relative = normalize_project_relative_path(
        str(config["artifacts"]["manifest"])
    )
    manifest_path = _project_path(manifest_relative)
    if manifest_path.exists():
        raise EvidenceContractError(f"来源冻结 manifest 已存在，禁止覆盖：{manifest_path}")

    payload: dict[str, Any] = {
        "program_id": config["program"]["program_id"],
        "stage_id": config["program"]["stage_id"],
        "version": config["program"]["version"],
        "status": "FROZEN_BEFORE_SOURCE_ADMISSION_NO_FEATURE_VALUES",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "authority": {
            "path": authority_path.relative_to(ROOT).as_posix(),
            "authority_id": authority["authority_id"],
            "sha256": sha256_file(authority_path),
            "research_state": authority["research_state"],
            "model_position_target": authority["model_position_target"],
            "order_authorization": authority["order_authorization"],
            "actual_holdings_state": authority["actual_holdings_state"],
            "position_impact": authority["position_impact"],
        },
        "parent_evidence": {
            "parent_protocol_config": file_evidence(
                parent_config_path, project_root=ROOT
            ),
            "parent_protocol_manifest": file_evidence(
                parent_manifest_path, project_root=ROOT
            ),
            "parent_manifest_payload_sha256": parent_manifest_hash,
            "g1_authoritative_status": file_evidence(
                parent_status_path, project_root=ROOT
            ),
            "g1_immutable_receipt": file_evidence(
                parent_receipt_path, project_root=ROOT
            ),
            "g1_receipt_payload_sha256": parent_receipt_hash,
            "g1_status": parent_status["status"],
            "g1_next_allowed_action": parent_status["next_allowed_action"],
            "label_artifacts_read": False,
        },
        "data_root_contracts": config["data_roots"],
        "registered_inputs": {
            input_id: _registered_without_resolved_path(evidence)
            for input_id, evidence in registered.items()
        },
        "missing_required_sources": config["missing_required_sources"],
        "availability_clock": config["availability_clock"],
        "admission_gates": config["admission_gates"],
        "implementation_files": implementation_files,
        "governance_files": governance_files,
        "git_state": {
            "head_before_protocol_commit": git_head(ROOT),
            "protocol_commit_required_before_admission": True,
            "required_committed_paths": list(
                freeze_contract["required_committed_before_admission"]
            ),
        },
        "research_state": "DISCOVERY_ONLY",
        "model_position_target": "UNSET",
        "order_authorization": "NOT_AUTHORIZED",
        "actual_holdings_state": "UNKNOWN_OUT_OF_SCOPE",
        "position_impact": 0,
        "label_artifacts_read": False,
        "feature_values_constructed": False,
        "model_trained": False,
        "portfolio_metrics_read": False,
        "rescue_allowed": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    atomic_write_json_new(manifest_path, payload)
    print(
        "来源准入协议已在特征构造前冻结："
        f"{manifest_relative}；"
        f"manifest_payload_sha256={payload['manifest_payload_sha256']}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="冻结 510300 非对称压力风险 V1 的点时来源准入协议"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        freeze(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"来源协议冻结失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

