"""一次性运行 510300 非对称压力风险 V1 的无标签 G2 时间可识别性预检。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.asymmetric_stress_hazard_g2_identifiability_v1 import (
    adjudicate_temporal_identifiability,
    build_daily_necessary_coverage,
    build_potential_origin_ceiling,
    build_preflight_result,
    parse_fixed_subperiods,
    prepare_constituent_presence,
    prepare_industry_presence,
    prepare_membership_presence,
    summarize_subperiod_coverage,
)
from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    atomic_write_text_new,
    canonical_sha256,
    create_exclusive_claim,
    file_evidence,
    normalize_project_relative_path,
    read_json_strict,
    sha256_file,
    strict_json_text,
    validate_authority_state,
)
from scripts.freeze_510300_asymmetric_stress_hazard_v1_g2_identifiability import (
    DEFAULT_CONFIG,
    verify_frozen_contract,
)


class OneShotConsumedError(EvidenceContractError):
    """一次性预检已经被 claim 消耗。"""


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        str(name): _project_path(str(relative))
        for name, relative in config["outputs"].items()
    }


def _assert_unconsumed(artifacts: Mapping[str, Path]) -> None:
    checked = {
        key: path
        for key, path in artifacts.items()
        if key not in {"manifest"}
    }
    existing = {key: str(path) for key, path in checked.items() if path.exists()}
    if not existing:
        return
    if "one_shot_claim" in existing:
        raise OneShotConsumedError(
            f"G2 时间可识别性预检已经被 claim 消耗：{existing['one_shot_claim']}"
        )
    raise EvidenceContractError(f"G2 预检输出已存在，禁止覆盖：{existing}")


def _registered_record(evidence: Any) -> dict[str, Any]:
    return {
        "input_id": evidence.input_id,
        "logical_path": evidence.logical_path,
        "physical_root_id": evidence.physical_root_id,
        "content_sha256": evidence.content_sha256,
        "bytes": evidence.bytes,
        "data_contract_version": evidence.data_contract_version,
    }


def _human_report(result: Mapping[str, Any]) -> str:
    gate = result["gate"]
    coverage = result["coverage_upper_bound"]
    period_lines = [
        (
            f"| `{item['subperiod_id']}` | {item['start']} | {item['end']} | "
            f"{item['potential_origin_upper_bound_count']} | "
            f"{'是' if item['evaluable_upper_bound'] else '否'} |"
        )
        for item in coverage["subperiods"]
    ]
    decision = (
        "覆盖上界达到门槛；下一步仅允许另行冻结完整 G2 特征构造与机制检验。"
        if gate["passed"]
        else (
            "即使采用偏乐观的覆盖上界，可评价子期仍少于三期。V1 在 G2 数据可识别性门停止；"
            "不得重切子期、回填当前行业、替换成分面板或缩短窗口。"
        )
    )
    return "\n".join(
        [
            "# 510300_ASYMMETRIC_STRESS_HAZARD_V1 G2 时间可识别性预检",
            "",
            "## 裁决",
            "",
            f"- 状态：`{gate['status']}`",
            (
                "- 可评价子期上界："
                f"`{gate['evaluable_subperiod_count_upper_bound']}` / "
                f"要求 `{gate['required_evaluable_subperiods']}`（总计 "
                f"`{gate['total_subperiods']}` 期）。"
            ),
            (
                "- 完整必要来源日："
                f"`{coverage['necessary_day_valid_count']}` / "
                f"市场日 `{coverage['market_session_count']}`。"
            ),
            (
                "- 30 日必要联合窗口的潜在原点上界："
                f"`{coverage['potential_origin_upper_bound_count']}`。"
            ),
            "",
            decision,
            "",
            "## 固定子期覆盖",
            "",
            "| 子期 | 起点 | 终点 | 潜在原点上界 | 达到20个 |",
            "|---|---:|---:|---:|---:|",
            *period_lines,
            "",
            "## 上界含义",
            "",
            "预检只核对点时成员、成分总收益价格是否存在以及点时申万一级行业是否可用。它故意忽略首个日收益所需的更早价格、宏观缺失、因果分位数暖机和数值退化，因此实际可评价原点只能更少，不能更多。",
            "",
            "## 未越过的研究边界",
            "",
            "- 没有构造 M、F、T 或价格基准数值。",
            "- 没有读取 BAD10 原点、事件或非事件块明细。",
            "- 没有训练模型、选择阈值或读取组合收益、夏普与回撤。",
            "- `RESEARCH_STATE=DISCOVERY_ONLY`。",
            "- `MODEL_POSITION_TARGET=UNSET`。",
            "- `ORDER_AUTHORIZATION=NOT_AUTHORIZED`。",
            "- `ACTUAL_HOLDINGS_STATE=UNKNOWN_OUT_OF_SCOPE`。",
            "- `POSITION_IMPACT=0`。",
            "",
        ]
    )


def _write_failure_receipt(
    *, artifacts: Mapping[str, Path], manifest: Mapping[str, Any], exc: Exception
) -> None:
    path = artifacts["failure_receipt"]
    if path.exists():
        return
    claim = artifacts["one_shot_claim"]
    payload = {
        "program_id": "510300_ASYMMETRIC_STRESS_HAZARD_V1",
        "stage_id": "G2_TEMPORAL_IDENTIFIABILITY_PREFLIGHT_V1",
        "status": "PROGRAM_FAILED_AFTER_G2_PREFLIGHT_CLAIM_NO_RERUN",
        "failed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "manifest_payload_sha256": manifest.get("manifest_payload_sha256"),
        "claim": file_evidence(claim, project_root=ROOT) if claim.is_file() else None,
            "feature_values_constructed": False,
            "bad10_label_artifacts_read": False,
            "return_values_read": False,
            "return_evaluation": "NOT_ALLOWED",
            "g2_mechanism_evaluated": False,
            "g3_allowed": False,
            "portfolio_metrics_read": False,
        "position_impact": 0,
        "order_authorization": "NOT_AUTHORIZED",
        "rerun_allowed": False,
    }
    atomic_write_json_new(path, payload)


def run_preflight(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, manifest, verification = verify_frozen_contract(config_path)
    artifacts = _artifact_paths(config)
    _assert_unconsumed(artifacts)
    claimed_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    claim_payload = {
        "claim_id": "510300_STRESS_HAZARD_V1_G2_IDENTIFIABILITY_ONE_SHOT",
        "claimed_at": claimed_at,
        "git_head": verification["git_head"],
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "feature_values_constructed": False,
        "bad10_label_artifacts_read": False,
        "return_values_read": False,
        "portfolio_metrics_read": False,
        "position_impact": 0,
        "claim_consumption_rule": "ONE_ATTEMPT_ONLY_NO_OVERWRITE_NO_RERUN",
    }
    create_exclusive_claim(artifacts["one_shot_claim"], claim_payload)

    try:
        sources = config["source_contract"]
        registered = verification["registered_inputs"]
        membership_contract = sources["pit_csi300_membership"]
        constituent_contract = sources["constituent_total_return_presence"]
        industry_contract = sources["sw_pit_industry_presence"]

        membership_raw = pd.read_parquet(
            Path(registered["pit_csi300_membership"].resolved_path),
            columns=list(membership_contract["required_columns"]),
        )
        constituent_raw = pd.read_parquet(
            Path(registered["constituent_total_return_presence"].resolved_path),
            columns=list(constituent_contract["required_columns"]),
        )
        industry_raw = pd.read_parquet(
            Path(registered["sw_pit_industry_presence"].resolved_path),
            columns=list(industry_contract["required_columns"]),
        )
        start = pd.Timestamp(str(config["program"]["observation_start"]))
        cutoff = pd.Timestamp(str(config["program"]["observation_cutoff"]))
        membership = prepare_membership_presence(
            membership_raw,
            expected_index_code=str(membership_contract["expected_index_code"]),
            start=start,
            cutoff=cutoff,
        )
        constituent = prepare_constituent_presence(constituent_raw)
        industry = prepare_industry_presence(
            industry_raw,
            required_mapping_status=str(industry_contract["required_mapping_status"]),
        )
        expected_members = int(membership_contract["expected_members_per_session"])
        daily = build_daily_necessary_coverage(
            membership,
            constituent,
            industry,
            expected_members_per_session=expected_members,
        )
        preflight = config["temporal_identifiability_preflight"]
        origins = build_potential_origin_ceiling(
            daily,
            history_sessions_inclusive=int(
                preflight["internal_history_market_sessions_inclusive"]
            ),
            forward_sessions=int(preflight["macro_lead_forward_market_sessions"]),
        )
        periods = parse_fixed_subperiods(config)
        summaries = summarize_subperiod_coverage(
            origins,
            periods,
            minimum_potential_origins=int(
                preflight["minimum_potential_daily_origins_per_subperiod"]
            ),
        )
        gate = adjudicate_temporal_identifiability(
            summaries,
            required_evaluable_subperiods=int(
                preflight["required_evaluable_subperiods"]
            ),
            total_subperiods=int(preflight["total_subperiods"]),
        )
        body = build_preflight_result(
            subperiod_summaries=summaries,
            gate=gate,
            daily=daily,
            origins=origins,
        )
        completed_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        authority = verification["authority"]
        result = {
            "program_id": config["program"]["program_id"],
            "stage_id": config["program"]["stage_id"],
            "version": config["program"]["version"],
            "status": gate["status"],
            "completed_at": completed_at,
            "git_head": verification["git_head"],
            "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            "one_shot_claim": file_evidence(
                artifacts["one_shot_claim"], project_root=ROOT
            ),
            "authority": {
                "research_state": authority["research_state"],
                "model_position_target": authority["model_position_target"],
                "order_authorization": authority["order_authorization"],
                "actual_holdings_state": authority["actual_holdings_state"],
                "position_impact": authority["position_impact"],
            },
            **body,
        }
        report = _human_report(result)
        atomic_write_json_new(artifacts["status"], result)
        atomic_write_text_new(artifacts["human_report"], report)
        receipt = {
            "program_id": result["program_id"],
            "stage_id": result["stage_id"],
            "version": result["version"],
            "status": "IMMUTABLE_G2_IDENTIFIABILITY_PREFLIGHT_RECEIPT_COMPLETE",
            "research_status": result["status"],
            "completed_at": completed_at,
            "git_head": verification["git_head"],
            "manifest_payload_sha256": manifest["manifest_payload_sha256"],
            "one_shot_claim": file_evidence(
                artifacts["one_shot_claim"], project_root=ROOT
            ),
            "registered_inputs": {
                key: _registered_record(value)
                for key, value in registered.items()
            },
            "outputs": {
                "status": file_evidence(artifacts["status"], project_root=ROOT),
                "human_report": file_evidence(
                    artifacts["human_report"], project_root=ROOT
                ),
            },
            "gate_passed": gate["passed"],
            "evaluable_subperiod_count_upper_bound": gate[
                "evaluable_subperiod_count_upper_bound"
            ],
            "feature_values_constructed": False,
            "bad10_label_artifacts_read": False,
            "return_values_read": False,
            "return_evaluation": "NOT_ALLOWED",
            "g2_mechanism_evaluated": False,
            "g3_allowed": False,
            "model_trained": False,
            "portfolio_metrics_read": False,
            "position_impact": 0,
            "order_authorization": "NOT_AUTHORIZED",
            "rescue_allowed": False,
        }
        receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
        atomic_write_json_new(artifacts["immutable_receipt"], receipt)
        print(strict_json_text(result, pretty=True))
        return result
    except Exception as exc:
        _write_failure_receipt(artifacts=artifacts, manifest=manifest, exc=exc)
        raise


def verify_results(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config, manifest, verification = verify_frozen_contract(config_path)
    artifacts = _artifact_paths(config)
    receipt = read_json_strict(artifacts["immutable_receipt"])
    if not isinstance(receipt, dict):
        raise EvidenceContractError("G2 预检回执必须是对象")
    expected_hash = str(receipt.get("receipt_payload_sha256", ""))
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256", None)
    if canonical_sha256(payload) != expected_hash:
        raise EvidenceContractError("G2 预检回执自身哈希失败")
    if receipt.get("manifest_payload_sha256") != manifest.get(
        "manifest_payload_sha256"
    ):
        raise EvidenceContractError("G2 预检回执与 manifest 不一致")
    claim_evidence = receipt.get("one_shot_claim")
    if not isinstance(claim_evidence, Mapping):
        raise EvidenceContractError("G2 预检回执缺少一次性 claim 证据")
    claim_path = _project_path(str(claim_evidence["path"]))
    if claim_path != artifacts["one_shot_claim"]:
        raise EvidenceContractError("G2 预检 claim 路径不一致")
    if claim_path.stat().st_size != int(claim_evidence["bytes"]):
        raise EvidenceContractError("G2 预检 claim 字节数不一致")
    if sha256_file(claim_path) != claim_evidence["sha256"]:
        raise EvidenceContractError("G2 预检 claim 哈希不一致")
    for name, evidence in receipt["outputs"].items():
        path = _project_path(str(evidence["path"]))
        if path != artifacts[name]:
            raise EvidenceContractError(f"G2 预检输出路径不一致：{name}")
        if path.stat().st_size != int(evidence["bytes"]):
            raise EvidenceContractError(f"G2 预检输出字节数不一致：{name}")
        if sha256_file(path) != evidence["sha256"]:
            raise EvidenceContractError(f"G2 预检输出哈希不一致：{name}")
    result = read_json_strict(artifacts["status"])
    if not isinstance(result, dict):
        raise EvidenceContractError("G2 预检状态必须是对象")
    forbidden_true = [
        "feature_values_constructed",
        "bad10_label_artifacts_read",
        "return_values_read",
        "g2_mechanism_evaluated",
        "g3_allowed",
        "model_trained",
        "probability_threshold_selected",
        "portfolio_metrics_read",
        "position_generated",
        "paper_shadow_generated",
        "order_generated",
        "broker_action_performed",
        "live_trading_authorized",
    ]
    for field in forbidden_true:
        if result.get(field) is not False:
            raise EvidenceContractError(f"G2 预检越界字段不是 false：{field}")
    validate_authority_state(verification["authority"])
    return {
        "status": "PASS_G2_IDENTIFIABILITY_RESULTS_AND_RECEIPT_VERIFIED",
        "research_status": result["status"],
        "gate_passed": result["gate"]["passed"],
        "git_head": verification["git_head"],
        "manifest_payload_sha256": manifest["manifest_payload_sha256"],
        "receipt_payload_sha256": expected_hash,
        "feature_values_constructed": False,
        "bad10_label_artifacts_read": False,
        "portfolio_metrics_read": False,
        "position_impact": 0,
        "order_authorization": "NOT_AUTHORIZED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行 510300 非对称压力风险 V1 的一次性 G2 时间可识别性预检"
    )
    parser.add_argument(
        "--phase", choices=["verify-contract", "preflight", "verify-results"], required=True
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        if args.phase == "verify-contract":
            _, manifest, verification = verify_frozen_contract(args.config)
            output = {
                "status": "PASS_G2_IDENTIFIABILITY_CONTRACT_VERIFIED",
                "git_head": verification["git_head"],
                "manifest_payload_sha256": manifest["manifest_payload_sha256"],
                "coverage_values_read": False,
                "feature_values_constructed": False,
                "bad10_label_artifacts_read": False,
                "portfolio_metrics_read": False,
                "position_impact": 0,
            }
            print(strict_json_text(output, pretty=True))
        elif args.phase == "preflight":
            run_preflight(args.config)
        else:
            print(strict_json_text(verify_results(args.config), pretty=True))
        return 0
    except OneShotConsumedError as exc:
        print(f"G2 时间可识别性预检已消耗，拒绝重跑：{exc}")
        return 4
    except (EvidenceContractError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"G2 时间可识别性程序或契约失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
