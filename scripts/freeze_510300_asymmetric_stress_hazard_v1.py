"""在读取任何 BAD10 标签之前冻结 510300 非对称压力风险 V1。"""

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
    git_head,
    normalize_project_relative_path,
    read_json_strict,
    sha256_file,
    validate_authority_state,
    validate_registered_inputs,
)


DEFAULT_CONFIG = ROOT / "config/510300_asymmetric_stress_hazard_v1.yaml"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("协议配置必须是对象")
    return payload


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def validate_pre_label_contract(config: Mapping[str, Any]) -> None:
    program = config["program"]
    if program["program_id"] != "510300_ASYMMETRIC_STRESS_HAZARD_V1":
        raise EvidenceContractError("program_id 不一致")
    if program["research_stage"] != "BAD10_LABEL_IDENTIFIABILITY_CENSUS_ONLY":
        raise EvidenceContractError("冻结阶段必须仅允许 BAD10 普查")
    if program["feature_construction_allowed_before_g1"] is not False:
        raise EvidenceContractError("G1 前不得构造特征")
    if program["model_training_allowed_before_g1"] is not False:
        raise EvidenceContractError("G1 前不得训练模型")
    if program["portfolio_evaluation_allowed_before_g3"] is not False:
        raise EvidenceContractError("G3 前不得评价组合")

    label = config["bad10_label"]
    required_label = {
        "execution_clock": "NEXT_TRADING_DAY_OPEN",
        "entry_asset": "510300.SH",
        "horizon_market_days": 10,
        "loss_threshold": -0.04,
        "transaction_cost_in_label": 0.0,
        "same_day_close_execution_allowed": False,
        "future_low_execution_allowed": False,
        "index_proxy_allowed": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": label.get(key)}
        for key, expected in required_label.items()
        if label.get(key) != expected
    }
    if mismatches:
        raise EvidenceContractError(f"BAD10 标签契约漂移：{mismatches}")

    funding = config["feature_charter"]["macro_vulnerability_M"]["channels"][
        "M2_FUNDING_STRESS"
    ]
    if funding.get("selected_rate") != "DR007":
        raise EvidenceContractError("资金压力口径必须唯一冻结为 DR007")
    if funding.get("alternative_rates_forbidden") != ["FDR007", "R007"]:
        raise EvidenceContractError("FDR007/R007 禁止救援清单发生漂移")

    gate = config["gates"]["G1_LABEL_IDENTIFIABILITY"]
    if gate.get("minimum_independent_bad10_events") != 30:
        raise EvidenceContractError("G1 独立 BAD10 事件门必须为 30")
    if gate.get("minimum_independent_non_event_10d_blocks") != 120:
        raise EvidenceContractError("G1 独立非事件块门必须为 120")
    if gate.get("label_change_after_failure_allowed") is not False:
        raise EvidenceContractError("G1 失败后不得修改标签")


def _hash_files(paths: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in paths:
        normalized = normalize_project_relative_path(str(relative))
        path = _project_path(normalized)
        if not path.is_file():
            raise EvidenceContractError(f"冻结文件缺失：{normalized}")
        result[normalized] = sha256_file(path)
    return result


def freeze(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_pre_label_contract(config)

    authority_relative = normalize_project_relative_path(
        str(config["authority_contract"]["path"])
    )
    authority = read_json_strict(_project_path(authority_relative))
    if not isinstance(authority, dict):
        raise EvidenceContractError("权威状态必须是对象")
    validate_authority_state(authority)
    if authority.get("authority_id") != config["authority_contract"]["authority_id"]:
        raise EvidenceContractError("权威状态 identity 不一致")

    registered = validate_registered_inputs(config, project_root=ROOT)
    freeze_contract = config["freeze_contract"]
    implementation_files = _hash_files(
        [str(value) for value in freeze_contract["implementation_files"]]
    )
    governance_files = _hash_files(
        [str(value) for value in freeze_contract["governance_files"]]
    )

    manifest_relative = normalize_project_relative_path(
        str(config["artifacts"]["protocol_manifest"])
    )
    manifest_path = _project_path(manifest_relative)
    if manifest_path.exists():
        raise EvidenceContractError(f"冻结 manifest 已存在，禁止覆盖：{manifest_path}")

    registered_inputs = {}
    for input_id, evidence in registered.items():
        record = evidence.to_dict()
        record.pop("resolved_path", None)
        registered_inputs[input_id] = record

    config_relative = config_path.relative_to(ROOT).as_posix()
    payload: dict[str, Any] = {
        "program_id": config["program"]["program_id"],
        "version": config["program"]["version"],
        "protocol_revision": config["program"]["protocol_revision"],
        "status": "FROZEN_BEFORE_BAD10_LABEL_CENSUS",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "config_file": config_relative,
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "authority": {
            "path": authority_relative,
            "authority_id": authority["authority_id"],
            "sha256": sha256_file(_project_path(authority_relative)),
            "research_state": authority["research_state"],
            "model_position_target": authority["model_position_target"],
            "order_authorization": authority["order_authorization"],
            "actual_holdings_state": authority["actual_holdings_state"],
            "position_impact": authority["position_impact"],
        },
        "data_root_contracts": config["data_roots"],
        "registered_inputs": registered_inputs,
        "implementation_files": implementation_files,
        "governance_files": governance_files,
        "git_state": {
            "head_before_protocol_commit": git_head(ROOT),
            "protocol_commit_required_before_census": True,
            "required_committed_paths": list(
                freeze_contract["required_committed_before_census"]
            ),
        },
        "bad10_contract": config["bad10_label"],
        "independent_event_contract": config["independent_event_contract"],
        "g1_contract": config["gates"]["G1_LABEL_IDENTIFIABILITY"],
        "forbidden_before_g1": list(config["forbidden_before_g1"]),
        "research_state": "DISCOVERY_ONLY",
        "model_position_target": "UNSET",
        "order_authorization": "NOT_AUTHORIZED",
        "actual_holdings_state": "UNKNOWN_OUT_OF_SCOPE",
        "position_impact": 0,
        "label_values_read": False,
        "feature_values_constructed": False,
        "model_trained": False,
        "portfolio_results_read": False,
        "rescue_allowed": False,
    }
    payload["manifest_payload_sha256"] = canonical_sha256(payload)
    atomic_write_json_new(manifest_path, payload)
    print(
        f"协议已在读取 BAD10 标签前冻结：{manifest_relative}；"
        f"manifest_payload_sha256={payload['manifest_payload_sha256']}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="冻结 510300 非对称压力风险 V1，禁止读取标签"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        freeze(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError) as exc:
        print(f"冻结失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
