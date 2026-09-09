"""运行并核验 510300_ASYMMETRIC_STRESS_HAZARD_V1 来源准入。"""

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

from research.asymmetric_stress_hazard_source_admission_v1 import (
    evaluate_source_admission,
)
from research.project_evidence_contract_v1 import (
    EvidenceContractError,
    atomic_write_json_new,
    atomic_write_text_new,
    canonical_sha256,
    ensure_paths_committed_at_head,
    file_evidence,
    git_head,
    normalize_project_relative_path,
    read_json_strict,
    sha256_file,
    strict_json_text,
    validate_authority_state,
    validate_registered_inputs,
    verify_manifest_files,
    verify_manifest_payload,
)


DEFAULT_CONFIG = (
    ROOT / "config/510300_asymmetric_stress_hazard_v1_source_contracts_v1.yaml"
)
BLOCKED_NEXT_ACTION = (
    "REMEDIATE_M2_DR007_AND_7D_REVERSE_REPO_POLICY_RATE_AND_F3_"
    "PIT_INDUSTRY_PROVENANCE_VIA_VERSIONED_SOURCE_CONTRACT"
)
PASSED_NEXT_ACTION = "CONSTRUCT_FROZEN_M_F_T_FEATURE_PANEL_AND_EVALUATE_G2"


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


def _artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        str(name): _project_path(str(relative))
        for name, relative in config["artifacts"].items()
    }


def _verify_file_evidence(
    evidence: Mapping[str, Any],
    *,
    expected_path: Path | None = None,
    name: str,
) -> Path:
    path = _project_path(str(evidence["path"]))
    if expected_path is not None and path != expected_path:
        raise EvidenceContractError(f"{name} 路径不一致")
    if not path.is_file():
        raise EvidenceContractError(f"{name} 文件缺失：{path}")
    if path.stat().st_size != int(evidence["bytes"]):
        raise EvidenceContractError(f"{name} 字节数不一致")
    if sha256_file(path) != str(evidence["sha256"]):
        raise EvidenceContractError(f"{name} 内容哈希不一致")
    return path


def _verify_receipt_self_hash(receipt: Mapping[str, Any], *, name: str) -> str:
    expected = str(receipt.get("receipt_payload_sha256", ""))
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256", None)
    actual = canonical_sha256(payload)
    if actual != expected:
        raise EvidenceContractError(
            f"{name} 自身哈希失败：expected={expected}, actual={actual}"
        )
    return expected


def _registered_without_resolved_path(record: Any) -> dict[str, Any]:
    payload = record.to_dict()
    payload.pop("resolved_path", None)
    return payload


def verify_frozen_contract(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """校验本阶段冻结字节、上游聚合终态与来源哈希，不读取标签明细。"""

    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    artifacts = _artifact_paths(config)
    manifest = _mapping_json(artifacts["manifest"], name="来源冻结 manifest")
    if manifest.get("status") != "FROZEN_BEFORE_SOURCE_ADMISSION_NO_FEATURE_VALUES":
        raise EvidenceContractError("来源冻结 manifest 状态不允许准入")
    verify_manifest_files(manifest, project_root=ROOT)
    if sha256_file(config_path) != manifest.get("config_sha256"):
        raise EvidenceContractError("来源配置字节与 manifest 不一致")
    if canonical_sha256(config) != manifest.get("config_canonical_sha256"):
        raise EvidenceContractError("来源配置语义与 manifest 不一致")

    authority_path = _project_path(str(config["authority_contract"]["path"]))
    authority = _mapping_json(authority_path, name="权威状态")
    validate_authority_state(authority)
    if authority.get("authority_id") != config["authority_contract"]["authority_id"]:
        raise EvidenceContractError("权威状态 identity 不一致")
    if sha256_file(authority_path) != manifest["authority"]["sha256"]:
        raise EvidenceContractError("权威状态与来源冻结 manifest 不一致")

    prerequisite = config["prerequisite"]
    parent = manifest.get("parent_evidence")
    if not isinstance(parent, Mapping):
        raise EvidenceContractError("来源 manifest 缺少 parent_evidence")
    parent_config_path = _verify_file_evidence(
        parent["parent_protocol_config"],
        expected_path=_project_path(str(prerequisite["parent_protocol_config"])),
        name="父协议配置",
    )
    parent_manifest_path = _verify_file_evidence(
        parent["parent_protocol_manifest"],
        expected_path=_project_path(str(prerequisite["parent_protocol_manifest"])),
        name="父协议 manifest",
    )
    parent_manifest = _mapping_json(parent_manifest_path, name="父协议 manifest")
    verify_manifest_payload(parent_manifest)
    if sha256_file(parent_config_path) != parent_manifest.get("config_sha256"):
        raise EvidenceContractError("父协议配置与父 manifest 不一致")
    if parent_manifest.get("manifest_payload_sha256") != parent.get(
        "parent_manifest_payload_sha256"
    ):
        raise EvidenceContractError("父 manifest payload 与来源 manifest 不一致")

    parent_status_path = _verify_file_evidence(
        parent["g1_authoritative_status"],
        expected_path=_project_path(str(prerequisite["census_status"])),
        name="G1 权威状态",
    )
    parent_receipt_path = _verify_file_evidence(
        parent["g1_immutable_receipt"],
        expected_path=_project_path(str(prerequisite["census_receipt"])),
        name="G1 不可变收据",
    )
    parent_status = _mapping_json(parent_status_path, name="G1 权威状态")
    parent_receipt = _mapping_json(parent_receipt_path, name="G1 不可变收据")
    parent_receipt_hash = _verify_receipt_self_hash(
        parent_receipt, name="G1 不可变收据"
    )
    if parent_receipt_hash != parent.get("g1_receipt_payload_sha256"):
        raise EvidenceContractError("G1 收据 payload 与来源 manifest 不一致")
    if parent_status.get("status") != prerequisite["required_g1_status"]:
        raise EvidenceContractError("G1 权威状态不允许来源准入")
    if parent_status.get("next_allowed_action") != prerequisite[
        "required_next_action"
    ]:
        raise EvidenceContractError("G1 next_allowed_action 不允许来源准入")
    if parent_status.get("feature_values_constructed") is not False:
        raise EvidenceContractError("G1 权威状态已越界构造特征")
    if parent_receipt.get("g1_passed") is not True:
        raise EvidenceContractError("G1 收据没有记录通过")
    if parent_receipt.get("feature_values_constructed") is not False:
        raise EvidenceContractError("G1 收据已越界构造特征")

    registered = validate_registered_inputs(config, project_root=ROOT)
    manifest_inputs = manifest.get("registered_inputs")
    if not isinstance(manifest_inputs, Mapping):
        raise EvidenceContractError("来源 manifest 缺少 registered_inputs")
    for input_id, record in registered.items():
        if manifest_inputs.get(input_id) != _registered_without_resolved_path(record):
            raise EvidenceContractError(f"{input_id} 来源身份与 manifest 不一致")

    required_paths = [
        str(value)
        for value in config["freeze_contract"][
            "required_committed_before_admission"
        ]
    ]
    committed = ensure_paths_committed_at_head(ROOT, required_paths)
    return config, manifest, {
        "authority": authority,
        "registered_inputs": registered,
        "committed_paths": committed,
        "git_head": git_head(ROOT),
        "parent_status": parent_status,
        "parent_receipt": parent_receipt,
    }


def _assert_outputs_absent(artifacts: Mapping[str, Path]) -> None:
    existing = {
        name: str(path)
        for name, path in artifacts.items()
        if name != "manifest" and path.exists()
    }
    if existing:
        raise EvidenceContractError(f"来源准入输出已存在，禁止覆盖或重跑：{existing}")


def _status_payload(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    completed_at: str,
) -> dict[str, Any]:
    decision = evaluation["decision"]
    passed = bool(decision["all_required_sources_admitted"])
    return {
        "program_id": config["program"]["program_id"],
        "stage_id": config["program"]["stage_id"],
        "version": config["program"]["version"],
        "status": decision["status"],
        "completed_at": completed_at,
        "source_contract_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "parent_g1_status": manifest["parent_evidence"]["g1_status"],
        "authority": {
            "research_state": authority["research_state"],
            "model_position_target": authority["model_position_target"],
            "order_authorization": authority["order_authorization"],
            "actual_holdings_state": authority["actual_holdings_state"],
            "position_impact": authority["position_impact"],
        },
        "source_results": evaluation["source_results"],
        "channel_results": evaluation["channel_results"],
        "all_required_sources_admitted": passed,
        "blocked_channels": decision["blocked_channels"],
        "research_disposition": "CONTINUE_TO_FEATURE_CONSTRUCTION" if passed else "NO_VIEW",
        "return_evaluation": "NOT_ALLOWED",
        "label_artifacts_read": False,
        "feature_construction_allowed": bool(decision["feature_construction_allowed"]),
        "feature_values_constructed": False,
        "g2_allowed": bool(decision["g2_allowed"]),
        "model_trained": False,
        "probability_threshold_selected": False,
        "portfolio_metrics_read": False,
        "position_generated": False,
        "paper_shadow_generated": False,
        "broker_action_performed": False,
        "order_generated": False,
        "live_trading_authorized": False,
        "rescue_allowed": False,
        "next_allowed_action": PASSED_NEXT_ACTION if passed else BLOCKED_NEXT_ACTION,
    }


def _build_report(
    *,
    status: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> str:
    sources = status["source_results"]
    channels = status["channel_results"]
    constituent = sources["constituent_total_return_panel"]["metrics"]
    industry = sources["pit_industry_intervals"]["metrics"]
    cgb = sources["china_10y_government_bond_yield"]["metrics"]
    pe = sources["csi300_official_pe"]["metrics"]
    tsf = sources["tsf_stock_yoy_first_release"]["metrics"]
    channel_lines = [
        f"- `{channel}`：`{record['status']}`"
        for channel, record in channels.items()
    ]
    return "\n".join(
        [
            "# 510300_ASYMMETRIC_STRESS_HAZARD_V1 点时来源准入 V1",
            "",
            "## 裁决",
            "",
            f"- 状态：`{status['status']}`。",
            f"- 研究处置：`{status['research_disposition']}`。",
            "- 必需来源没有全部通过，因此没有构造 M、F、T 特征值，也没有进入 G2。",
            "- 未训练模型，未选择阈值，未读取收益、夏普或回撤，未生成仓位、Paper/Shadow、订单或券商动作。",
            f"- 来源冻结 manifest：`{manifest['manifest_payload_sha256']}`。",
            "",
            "## 通道准入",
            "",
            *channel_lines,
            "",
            "## 整源阻断",
            "",
            f"- DR007：`{sources['dr007_daily']['status']}`。本地没有冻结历史序列，FDR007、R007、FR007 或交易所 R-007 均不得替代。",
            f"- 央行 7 天逆回购政策利率：`{sources['reverse_repo_policy_rate_7d']['status']}`。不得插值或倒推出变更日。",
            f"- 申万一级行业区间：`{sources['pit_industry_intervals']['status']}`。现有代理 payload 缺少独立可验证的版本来源。",
            "",
            "## 已准入来源与逐日 NO_VIEW 边界",
            "",
            f"- 中证 300 官方 PE：{pe['row_count']} 行，{pe['first_observation_date']} 至 {pe['last_observation_date']}；原始 payload 逐行一致。末端 {pe['trailing_no_view_market_sessions']} 个交易日缺值，且因未归档精确发布时间只可从下一交易日开盘起使用。",
            f"- 中国国债 10 年收益率：{cgb['row_count']} 行，{cgb['first_observation_date']} 至 {cgb['last_observation_date']}；冻结观察窗前段 {cgb['leading_no_view_market_sessions']} 个交易日只能 `NO_VIEW`，原始 payload 未归档且采用下一交易日开盘可得时钟。",
            f"- 社融存量同比首发 vintage：{tsf['contiguous_month_count']} 个连续月份，{tsf['first_reference_period']} 至 {tsf['last_reference_period']}；未使用修订值。构造三个月变化所需的前置月份只能产生领先期 `NO_VIEW`。",
            f"- 官方 PIT 成分：{sources['pit_csi300_membership']['metrics']['session_count']} 个交易日、每天严格 300 只；本框架不需要且没有使用未准入的历史权重。",
            f"- 成分股总回报面板：在 {constituent['evaluated_session_count']} 个重叠交易日中，完整 300 只的日期为 {constituent['full_300_member_session_count']} 个；{constituent['no_view_session_count']} 个日期缺少至少一只，合计缺少 {constituent['missing_member_day_count']} 个 member-day，最少仅 {constituent['minimum_valid_member_count']} 只。这些日期均为 `NO_VIEW`，不插值、不用当前成分回填。",
            f"- 行业区间结构覆盖：{industry['evaluated_session_count']} 个交易日中完整覆盖 {industry['full_300_member_session_count']} 个，缺口日期 {industry['no_view_session_count']} 个，合计缺少 {industry['missing_member_day_count']} 个 member-day；即使逐日缺口可按 `NO_VIEW` 处理，独立版本来源仍未通过，故 F3/T2/T3 整体阻断。",
            "",
            "## 允许的后续动作",
            "",
            f"`{status['next_allowed_action']}`",
            "",
            "后续只能用新的版本化来源契约补齐上述来源证据；不得修改 BAD10 标签、窗口、阈值，不得用替代利率、当前成分或当前行业分类救援。",
            "",
            "## 权限边界",
            "",
            f"- `RESEARCH_STATE={status['authority']['research_state']}`",
            f"- `MODEL_POSITION_TARGET={status['authority']['model_position_target']}`",
            f"- `ORDER_AUTHORIZATION={status['authority']['order_authorization']}`",
            f"- `ACTUAL_HOLDINGS_STATE={status['authority']['actual_holdings_state']}`",
            f"- `POSITION_IMPACT={status['authority']['position_impact']}`",
            "- `NO_VIEW` 是研究终态，不是现金仓位建议，也不描述用户实际持仓。",
            "",
        ]
    )


def run_admission(config_path: Path) -> dict[str, Any]:
    config, manifest, verification = verify_frozen_contract(config_path)
    artifacts = _artifact_paths(config)
    _assert_outputs_absent(artifacts)
    evaluation = evaluate_source_admission(
        config,
        project_root=ROOT,
        registered_inputs=verification["registered_inputs"],
    )
    completed_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    status = _status_payload(
        config=config,
        manifest=manifest,
        authority=verification["authority"],
        evaluation=evaluation,
        completed_at=completed_at,
    )
    report = _build_report(status=status, manifest=manifest)
    atomic_write_json_new(artifacts["status"], status)
    atomic_write_text_new(artifacts["human_report"], report)

    receipt: dict[str, Any] = {
        "program_id": config["program"]["program_id"],
        "stage_id": config["program"]["stage_id"],
        "version": config["program"]["version"],
        "status": "IMMUTABLE_SOURCE_ADMISSION_RECEIPT_COMPLETE",
        "completed_at": completed_at,
        "git_head": verification["git_head"],
        "source_contract_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "parent_g1_evidence": {
            "status": manifest["parent_evidence"]["g1_status"],
            "status_sha256": manifest["parent_evidence"][
                "g1_authoritative_status"
            ]["sha256"],
            "receipt_payload_sha256": manifest["parent_evidence"][
                "g1_receipt_payload_sha256"
            ],
            "label_artifacts_read": False,
        },
        "registered_inputs": {
            input_id: _registered_without_resolved_path(record)
            for input_id, record in verification["registered_inputs"].items()
        },
        "outputs": {
            "status": file_evidence(artifacts["status"], project_root=ROOT),
            "human_report": file_evidence(
                artifacts["human_report"], project_root=ROOT
            ),
        },
        "source_admission_status": status["status"],
        "all_required_sources_admitted": status[
            "all_required_sources_admitted"
        ],
        "blocked_channels": status["blocked_channels"],
        "research_disposition": status["research_disposition"],
        "return_evaluation": "NOT_ALLOWED",
        "feature_construction_allowed": status["feature_construction_allowed"],
        "feature_values_constructed": False,
        "g2_allowed": status["g2_allowed"],
        "model_trained": False,
        "portfolio_metrics_read": False,
        "position_impact": 0,
        "model_position_target": "UNSET",
        "order_authorization": "NOT_AUTHORIZED",
        "next_allowed_action": status["next_allowed_action"],
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(artifacts["receipt"], receipt)
    print(strict_json_text(status, pretty=True))
    return status


def verify_results(config_path: Path) -> dict[str, Any]:
    config, manifest, verification = verify_frozen_contract(config_path)
    artifacts = _artifact_paths(config)
    receipt = _mapping_json(artifacts["receipt"], name="来源准入收据")
    receipt_hash = _verify_receipt_self_hash(receipt, name="来源准入收据")
    if receipt.get("status") != "IMMUTABLE_SOURCE_ADMISSION_RECEIPT_COMPLETE":
        raise EvidenceContractError("来源准入收据状态不一致")
    if receipt.get("source_contract_manifest_payload_sha256") != manifest.get(
        "manifest_payload_sha256"
    ):
        raise EvidenceContractError("来源准入收据与冻结 manifest 不一致")
    for name, evidence in receipt["outputs"].items():
        if name not in artifacts:
            raise EvidenceContractError(f"来源准入收据出现未知输出：{name}")
        _verify_file_evidence(
            evidence, expected_path=artifacts[name], name=f"来源准入输出 {name}"
        )
    status = _mapping_json(artifacts["status"], name="来源准入状态")
    if status.get("status") != receipt.get("source_admission_status"):
        raise EvidenceContractError("来源准入状态与收据裁决不一致")
    if status.get("all_required_sources_admitted") is not False:
        raise EvidenceContractError("当前冻结收据应明确记录来源未全部准入")
    if status.get("research_disposition") != "NO_VIEW":
        raise EvidenceContractError("当前来源失败没有落为 NO_VIEW")
    for key in (
        "label_artifacts_read",
        "feature_construction_allowed",
        "feature_values_constructed",
        "g2_allowed",
        "model_trained",
        "probability_threshold_selected",
        "portfolio_metrics_read",
        "position_generated",
        "paper_shadow_generated",
        "broker_action_performed",
        "order_generated",
        "live_trading_authorized",
        "rescue_allowed",
    ):
        if status.get(key) is not False:
            raise EvidenceContractError(f"来源准入越界或终态不一致：{key}")
    if status.get("authority") != {
        "research_state": verification["authority"]["research_state"],
        "model_position_target": verification["authority"][
            "model_position_target"
        ],
        "order_authorization": verification["authority"]["order_authorization"],
        "actual_holdings_state": verification["authority"][
            "actual_holdings_state"
        ],
        "position_impact": verification["authority"]["position_impact"],
    }:
        raise EvidenceContractError("来源准入状态的权限边界不一致")
    committed_outputs = ensure_paths_committed_at_head(
        ROOT,
        [
            artifacts["status"].relative_to(ROOT).as_posix(),
            artifacts["human_report"].relative_to(ROOT).as_posix(),
            artifacts["receipt"].relative_to(ROOT).as_posix(),
        ],
    )
    return {
        "status": "PASS_SOURCE_ADMISSION_RESULTS_AND_RECEIPT_VERIFIED",
        "git_head": git_head(ROOT),
        "source_contract_manifest_payload_sha256": manifest[
            "manifest_payload_sha256"
        ],
        "receipt_payload_sha256": receipt_hash,
        "source_admission_status": status["status"],
        "research_disposition": status["research_disposition"],
        "blocked_channels": status["blocked_channels"],
        "committed_output_count": len(committed_outputs),
        "feature_values_constructed": False,
        "g2_allowed": False,
        "position_impact": 0,
        "order_authorization": "NOT_AUTHORIZED",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行或核验 510300 非对称压力风险 V1 点时来源准入"
    )
    parser.add_argument("--phase", choices=["admit", "verify-results"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        if args.phase == "admit":
            run_admission(args.config)
        else:
            print(strict_json_text(verify_results(args.config), pretty=True))
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"来源准入程序或契约失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

