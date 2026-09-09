"""在首次读取实际 BAD10 行之前冻结 V2 G1 事件预测准入 V1。"""

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
from research.stress_transmission_hazard_v2_g1_event_prediction_admission_v1 import (
    B1_FEATURE_COLUMNS,
    EVENT_READ_COLUMNS,
    EXECUTION_ID,
    PROGRAM_ID,
    SAMPLE_READ_COLUMNS,
)
from scripts.freeze_510300_stress_transmission_hazard_v2 import (
    verify_frozen_manifest as verify_v2_protocol_manifest,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_mft_feature_execution_v1_0_1 import (
    verify_frozen_manifest as verify_mft_correction_manifest,
)


DEFAULT_CONFIG = (
    ROOT
    / "config/510300_stress_transmission_hazard_v2_g1_event_prediction_admission_v1.yaml"
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("G1 事件预测准入配置必须是对象")
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
    if path.stat().st_size != int(contract["bytes"]):
        raise EvidenceContractError(f"{label}字节数漂移：{path}")
    if sha256_file(path) != str(contract["sha256"]):
        raise EvidenceContractError(f"{label}哈希漂移：{path}")
    return file_evidence(path, project_root=ROOT)


def validate_config(config: Mapping[str, Any]) -> None:
    program = config["program"]
    _require_equal(program.get("program_id"), PROGRAM_ID, "项目 identity")
    _require_equal(program.get("execution_id"), EXECUTION_ID, "执行 identity")
    _require_equal(program.get("version"), "1.0.0", "执行版本")
    _require_equal(program.get("stage_id"), "G1_EVENT_PREDICTION_ADMISSION", "阶段")
    _require_equal(program.get("research_state"), "DISCOVERY_ONLY", "研究状态")
    _require_equal(program.get("return_evaluation"), "NOT_ALLOWED", "收益评估")
    _require_equal(
        program.get("actual_label_read_allowed_only_after_manifest_freeze"),
        True,
        "冻结后标签读取权限",
    )
    _require_equal(
        program.get("actual_future_path_value_read_allowed"), False, "未来路径读取权限"
    )
    _require_equal(program.get("model_training_allowed"), False, "模型训练权限")
    _require_equal(program.get("probability_generation_allowed"), False, "概率生成权限")
    _require_equal(
        program.get("alert_threshold_selection_allowed"), False, "警报阈值权限"
    )
    _require_equal(program.get("portfolio_evaluation_allowed"), False, "组合评估权限")
    _require_equal(program.get("paper_or_shadow_allowed"), False, "Paper/Shadow 权限")
    _require_equal(program.get("broker_connection_allowed"), False, "经纪商权限")
    _require_equal(program.get("position_impact"), 0, "仓位影响")
    _require_equal(program.get("executable_assets"), ["510300.SH", "CASH_CNY"], "资产范围")
    _require_equal(program.get("observation_start"), "2015-01-05", "观察起点")
    _require_equal(program.get("observation_cutoff"), "2026-08-14", "观察截止")

    label = config["label_read_contract"]
    _require_equal(label.get("label_id"), "BAD10", "标签 identity")
    _require_equal(label.get("allowed_only_after_frozen_manifest_verifies"), True, "标签时序")
    _require_equal(label.get("threshold"), -0.04, "BAD10 阈值")
    _require_equal(label.get("horizon_market_days"), 10, "BAD10 期限")
    _require_equal(label.get("threshold_or_horizon_search_allowed"), False, "标签搜索权限")
    _require_equal(label.get("actual_future_path_value_columns_may_be_read"), False, "路径值读取")
    _require_equal(label.get("label_may_enter_features"), False, "标签进入特征")
    _require_equal(label.get("label_may_select_feature_availability"), False, "标签选择可见性")
    _require_equal(label.get("label_may_select_probability_threshold"), False, "标签选择阈值")

    inputs = config["inputs"]
    _require_equal(inputs["bad10_events"].get("read_columns"), EVENT_READ_COLUMNS, "事件读取列")
    _require_equal(inputs["bad10_samples"].get("read_columns"), SAMPLE_READ_COLUMNS, "样本读取列")
    forbidden_label_columns = {
        "minimum_path_return",
        "cash_dividend_per_share_in_horizon",
        "first_breach_date",
    }
    sample_forbidden = set(inputs["bad10_samples"].get("forbidden_read_columns", []))
    if not forbidden_label_columns.issubset(sample_forbidden):
        raise EvidenceContractError("BAD10 样本禁止读取列不完整")
    _require_equal(inputs["etf_unadjusted_daily"].get("symbol"), "510300.SH", "ETF identity")
    _require_equal(inputs["etf_unadjusted_daily"].get("price_basis"), "UNADJUSTED", "ETF 价格口径")
    _require_equal(inputs["etf_cash_dividends"].get("symbol"), "510300.SH", "分红 identity")

    b1 = config["b1_feature_contract"]
    _require_equal(b1.get("decision_clock"), "T_CLOSE_15_00_ASIA_SHANGHAI", "B1 决策时钟")
    _require_equal(b1.get("absent_dividend_event_amount"), 0.0, "无分红事件现金项")
    _require_equal(b1.get("missing_price_return_zero_fill_allowed"), False, "缺失收益填零")
    _require_equal(b1.get("realized_volatility_market_days"), 20, "B1 波动窗口")
    _require_equal(b1.get("drawdown_market_days"), 20, "B1 回撤窗口")
    _require_equal(b1.get("downside_return_market_days"), 5, "B1 下行窗口")
    _require_equal(b1.get("feature_columns"), B1_FEATURE_COLUMNS, "B1 固定特征")
    _require_equal(b1.get("future_close_use_allowed"), False, "未来收盘读取")
    _require_equal(b1.get("h00300_execution_or_b1_proxy_allowed"), False, "H00300 代理")

    common = config["common_sample_contract"]
    _require_equal(common.get("join_key"), "origin_date_equals_feature_date", "共同样本键")
    _require_equal(common.get("feature_information_cutoff"), "T_CLOSE", "特征截止")
    _require_equal(common["B2_VS_B1"].get("common_sample_required"), True, "B2 共同样本")
    _require_equal(common["B3_VS_B2"].get("common_sample_required"), True, "B3 共同样本")
    _require_equal(
        common.get("positive_event_inclusion_rule"),
        "AT_LEAST_ONE_ELIGIBLE_POSITIVE_ORIGIN",
        "事件可识别规则",
    )
    _require_equal(
        common.get("positive_weight_rule"),
        "RENORMALIZE_ELIGIBLE_POSITIVE_ORIGINS_TO_SUM_ONE_WITHIN_EACH_EVENT_AND_MODEL_SAMPLE",
        "正样本权重",
    )
    _require_equal(common.get("negative_weight_rule"), "EACH_ELIGIBLE_NON_EVENT_RISK_DAY_EQUALS_ONE", "负样本权重")
    _require_equal(common.get("split_assignment"), "UNASSIGNED_PRE_MODEL_FREEZE", "样本切分")
    _require_equal(common.get("prediction_or_fit_allowed"), False, "共同样本拟合权限")

    gate = config["g1_gate_contract"]
    _require_equal(gate["mechanism_discovery"].get("minimum_independent_events"), 30, "机制事件门")
    _require_equal(gate["mechanism_discovery"].get("minimum_non_event_risk_days"), 750, "机制非事件门")
    _require_equal(gate["full_three_coefficient_model"].get("minimum_independent_events"), 40, "完整模型事件门")
    _require_equal(gate["full_three_coefficient_model"].get("minimum_non_event_risk_days"), 750, "完整模型非事件门")
    _require_equal(gate.get("insufficient_state"), "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY", "G1 失败态")

    sequential = config["sequential_stop_contract"]
    for key in (
        "fit_B0_B1_B2_B3_in_this_execution",
        "generate_probability_in_this_execution",
        "generate_auc_pr_auc_brier_log_loss_in_this_execution",
        "generate_alert_or_threshold_in_this_execution",
        "generate_portfolio_nav_return_sharpe_drawdown_in_this_execution",
    ):
        _require_equal(sequential.get(key), False, key)

    forbidden = set(config["forbidden_actions"])
    required_forbidden = {
        "READ_ACTUAL_LABEL_BEFORE_THIS_MANIFEST_IS_FROZEN_AND_VERIFIED",
        "READ_MINIMUM_PATH_RETURN_OR_OTHER_ACTUAL_FUTURE_PATH_VALUE",
        "CHANGE_BAD10_THRESHOLD_OR_HORIZON",
        "USE_LABEL_TO_SELECT_FEATURE_AVAILABILITY_OR_COMMON_SAMPLE_RULE",
        "INCLUDE_INDUSTRY_SECTOR_OR_SHENWAN_FIELDS",
        "TRAIN_OR_PREDICT_B0_B1_B2_OR_B3_IN_G1_ADMISSION",
        "SELECT_PROBABILITY_THRESHOLD_OR_GENERATE_ALERTS",
        "READ_OR_GENERATE_PORTFOLIO_NAV_RETURN_SHARPE_DRAWDOWN",
        "CREATE_POSITION_ORDER_BROKER_PAPER_SHADOW_OR_LIVE_ACTION",
    }
    if not required_forbidden.issubset(forbidden):
        raise EvidenceContractError(
            f"G1 禁止动作缺项：{sorted(required_forbidden.difference(forbidden))}"
        )
    output_values = [value for key, value in config["outputs"].items() if key != "curated_root"]
    if len(output_values) != len(set(output_values)):
        raise EvidenceContractError("G1 输出路径重复")


def verify_parent_chain(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    verify_v2_protocol_manifest()
    verify_mft_correction_manifest()
    parent = config["parent_chain"]
    evidence = {
        name: _validate_file_identity(contract, f"父链 {name}")
        for name, contract in parent.items()
    }

    g0 = read_json_strict(_project_path(parent["g0_status"]["path"]))
    _require_equal(g0.get("G0_ENGINEERING_AND_CONTRACT"), parent["g0_status"]["required_g0_status"], "G0 状态")
    _require_equal(g0.get("next_allowed_step"), parent["g0_status"]["required_next_step"], "G0 下一步")
    _require_equal(g0.get("return_evaluation"), "NOT_ALLOWED", "G0 收益权限")
    _require_equal(g0.get("position_impact"), 0, "G0 仓位影响")

    replay = read_json_strict(_project_path(parent["mft_clean_replay_receipt"]["path"]))
    _require_equal(replay.get("status"), parent["mft_clean_replay_receipt"]["required_status"], "M/F/T 重放状态")
    _require_equal(replay.get("actual_label_artifact_read"), False, "M/F/T 标签读取")
    _require_equal(replay.get("actual_performance_artifact_read"), False, "M/F/T 绩效读取")

    mft = read_json_strict(_project_path(parent["mft_build_receipt"]["path"]))
    _require_equal(mft.get("status"), parent["mft_build_receipt"]["required_status"], "M/F/T 构建状态")
    _require_equal(mft.get("actual_label_artifact_read"), False, "M/F/T 构建标签读取")
    _require_equal(mft.get("actual_performance_artifact_read"), False, "M/F/T 构建绩效读取")

    bad10 = read_json_strict(_project_path(parent["bad10_ledger_receipt"]["path"]))
    _require_equal(bad10.get("status"), parent["bad10_ledger_receipt"]["required_status"], "BAD10 账本状态")
    _require_equal(bad10.get("event_count"), parent["bad10_ledger_receipt"]["expected_independent_events"], "BAD10 事件数")
    _require_equal(bad10.get("positive_origin_count"), parent["bad10_ledger_receipt"]["expected_positive_origins"], "BAD10 正样本数")
    _require_equal(bad10.get("non_event_risk_day_count"), parent["bad10_ledger_receipt"]["expected_non_event_risk_days"], "BAD10 非事件日数")
    _require_equal(bad10.get("model_trained"), False, "BAD10 模型训练")
    _require_equal(bad10.get("performance_values_read"), False, "BAD10 绩效读取")
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
        evidence = file_evidence(path, project_root=ROOT)
        evidence.pop("path")
        result[relative] = evidence
    return result


def _freeze_outputs(config: Mapping[str, Any]) -> tuple[Path, Path]:
    freeze = config["freeze_contract"]
    return (
        _project_path(str(freeze["manifest_output"])),
        _project_path(str(freeze["freeze_receipt_output"])),
    )


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, receipt_path = _freeze_outputs(config)
    existing = [str(path) for path in (manifest_path, receipt_path) if path.exists()]
    if existing:
        raise EvidenceContractError(f"G1 冻结输出已存在，禁止覆盖：{existing}")
    parent_evidence = verify_parent_chain(config)
    input_evidence = verify_input_identities(config)
    freeze_contract = config["freeze_contract"]
    implementation = _hash_group([str(x) for x in freeze_contract["implementation_files"]])
    governance = _hash_group([str(x) for x in freeze_contract["governance_files"]])
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest: dict[str, Any] = {
        "execution_id": EXECUTION_ID,
        "version": "1.0.0",
        "status": "FROZEN_BEFORE_FIRST_G1_ACTUAL_LABEL_READ",
        "frozen_at": now,
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_bytes": config_path.stat().st_size,
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "parent_chain": parent_evidence,
        "input_files": input_evidence,
        "implementation_files": implementation,
        "governance_files": governance,
        "label_read_contract": config["label_read_contract"],
        "b1_feature_contract": config["b1_feature_contract"],
        "common_sample_contract": config["common_sample_contract"],
        "g1_gate_contract": config["g1_gate_contract"],
        "sequential_stop_contract": config["sequential_stop_contract"],
        "outputs": config["outputs"],
        "forbidden_actions": config["forbidden_actions"],
        "actual_label_artifact_parsed": False,
        "actual_future_path_value_read": False,
        "model_trained": False,
        "probability_generated": False,
        "performance_artifact_read": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_G1_EVENT_PREDICTION_ADMISSION_V1_FREEZE",
        "created_at": now,
        "status": "PASS_G1_COMMON_SAMPLE_B1_WEIGHT_AND_COUNT_RULES_FROZEN_BEFORE_LABEL_READ",
        "manifest": file_evidence(manifest_path, project_root=ROOT),
        "parent_chain_verified": True,
        "input_identities_verified_without_parsing_label_rows": True,
        "actual_label_artifact_parsed": False,
        "actual_future_path_value_read": False,
        "model_trained": False,
        "performance_artifact_read": False,
        "position_impact": 0,
    }
    atomic_write_json_new(receipt_path, receipt)
    print("G1 共同样本、B1、事件权重和计数门已在首次实际标签读取前冻结。")
    return manifest


def verify_frozen_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, _ = _freeze_outputs(config)
    manifest = read_json_strict(manifest_path)
    if not isinstance(manifest, dict):
        raise EvidenceContractError("G1 manifest 必须是对象")
    verify_manifest_payload(manifest)
    _require_equal(manifest.get("execution_id"), EXECUTION_ID, "执行 identity")
    _require_equal(manifest.get("config_sha256"), sha256_file(config_path), "配置哈希")
    _require_equal(manifest.get("actual_label_artifact_parsed"), False, "冻结前标签解析")
    verify_parent_chain(config)
    verify_input_identities(config)
    for group_name in ("implementation_files", "governance_files"):
        group = manifest.get(group_name)
        if not isinstance(group, Mapping):
            raise EvidenceContractError(f"G1 manifest 缺少 {group_name}")
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
            print("G1 冻结 manifest、父链和输入身份验证通过。")
        else:
            freeze(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError) as exc:
        print(f"G1 事件预测准入冻结失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
