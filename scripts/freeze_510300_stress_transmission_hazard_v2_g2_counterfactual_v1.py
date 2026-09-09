"""冻结 V2 反事实 G2 V1；冻结阶段不拟合模型或计算预测指标。"""

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
from scripts.freeze_510300_stress_transmission_hazard_v2_g1_event_prediction_admission_v1_0_1 import (
    verify_frozen_manifest as verify_g1_manifest,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_g1_historical_coverage_audit_v1 import (
    verify_frozen_manifest as verify_historical_audit_manifest,
)


EXECUTION_ID = "510300_STRESS_TRANSMISSION_HAZARD_V2_G2_COUNTERFACTUAL_V1"
DEFAULT_CONFIG = (
    ROOT / "config/510300_stress_transmission_hazard_v2_g2_counterfactual_v1.yaml"
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise EvidenceContractError("反事实 G2 配置必须是对象")
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
    actual = file_evidence(path, project_root=ROOT)
    expected = {
        "path": normalize_project_relative_path(str(contract["path"])).replace("\\", "/"),
        "bytes": int(contract["bytes"]),
        "sha256": str(contract["sha256"]),
    }
    if actual != expected:
        raise EvidenceContractError(f"{label}文件身份漂移：{path}")
    return actual


def _verify_self_digest(payload: Mapping[str, Any], field: str, label: str) -> None:
    expected = str(payload.get(field, ""))
    body = dict(payload)
    body.pop(field, None)
    if canonical_sha256(body) != expected:
        raise EvidenceContractError(f"{label}自身摘要失败")


def validate_config(config: Mapping[str, Any]) -> None:
    program = config["program"]
    _require_equal(program.get("program_id"), "510300_STRESS_TRANSMISSION_HAZARD_V2", "项目 identity")
    _require_equal(program.get("execution_id"), EXECUTION_ID, "执行 identity")
    _require_equal(program.get("version"), "1.0.0", "版本")
    _require_equal(program.get("stage_id"), "G2_STRUCTURAL_INCREMENT_COUNTERFACTUAL", "阶段")
    _require_equal(program.get("research_state"), "DISCOVERY_ONLY", "研究状态")
    _require_equal(program.get("evidence_class"), "RETROSPECTIVE_HISTORICAL_COUNTERFACTUAL", "证据类别")
    _require_equal(program.get("return_evaluation"), "NOT_ALLOWED", "收益权限")
    _require_equal(program.get("portfolio_evaluation_allowed"), False, "组合权限")
    _require_equal(program.get("probability_threshold_selection_allowed"), False, "阈值权限")
    _require_equal(program.get("paper_or_shadow_allowed"), False, "Paper/Shadow 权限")
    _require_equal(program.get("broker_connection_allowed"), False, "经纪商权限")
    _require_equal(program.get("position_impact"), 0, "仓位影响")
    _require_equal(program.get("executable_assets"), ["510300.SH", "CASH_CNY"], "资产范围")
    _require_equal(program.get("observation_start"), "2015-01-05", "观察起点")
    _require_equal(program.get("historical_cutoff"), "2026-08-14", "历史截止日")
    _require_equal(program.get("observation_after_historical_cutoff_allowed"), False, "截止日后观察权限")

    counterfactual = config["counterfactual_authority"]
    _require_equal(counterfactual.get("user_directive"), "假设g1通过，进行g2", "用户指令")
    _require_equal(counterfactual.get("assumption_id"), "ASSUME_G1_MECHANISM_COUNT_GATE_PASS_FOR_G2_ONLY", "反事实假设")
    _require_equal(counterfactual.get("assumed_G1_DATA_AND_EVENTS"), "PASS_BY_USER_COUNTERFACTUAL_ASSUMPTION", "假设 G1")
    _require_equal(counterfactual.get("actual_G1_DATA_AND_EVENTS"), "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY", "真实 G1")
    for key in (
        "actual_g1_state_may_be_overwritten",
        "fabricate_events_or_features_allowed",
        "relax_common_sample_allowed",
        "impute_no_view_allowed",
        "promote_counterfactual_result_to_formal_gate_allowed",
        "formal_g3_authorization_created",
    ):
        _require_equal(counterfactual.get(key), False, key)

    common = config["common_sample_contract"]
    _require_equal(common.get("sample"), "B2_VS_B1_COMMON_SAMPLE_ONLY", "共同样本")
    _require_equal(common.get("eligibility_column"), "b2_vs_b1_eligible", "准入列")
    _require_equal(
        common.get("B1_features"),
        [
            "b1_realized_vol20_risk_percentile",
            "b1_drawdown20_risk_percentile",
            "b1_downside_return5_risk_percentile",
        ],
        "B1 特征",
    )
    _require_equal(common.get("B2_features"), ["T", "T_x_F"], "B2 特征")
    _require_equal(common.get("positive_event_weight_sum"), 1.0, "正事件权重")
    _require_equal(common.get("negative_risk_day_weight"), 1.0, "负样本权重")
    _require_equal(common.get("compare_on_exact_same_rows"), True, "共同样本比较")
    _require_equal(common.get("event_may_be_split_within_a_prediction_fold"), False, "事件拆分权限")
    _require_equal(common.get("missing_or_no_view_imputation_allowed"), False, "缺失插补权限")

    model = config["model_contract"]
    _require_equal(model.get("family"), "BINOMIAL_LOGISTIC_REGRESSION", "模型族")
    _require_equal(model.get("l2_penalty"), 1.0, "L2")
    _require_equal(model.get("intercept_penalized"), False, "截距惩罚")
    _require_equal(model.get("B1_slope_constraints"), "NONNEGATIVE", "B1 符号")
    _require_equal(model.get("B2_beta1_T_constraint"), "NONNEGATIVE", "B2 beta1 符号")
    _require_equal(model.get("B2_beta2_T_x_F_constraint"), "NONNEGATIVE", "B2 beta2 符号")
    _require_equal(model.get("optimizer"), "SCIPY_L_BFGS_B_ANALYTIC_GRADIENT", "优化器")
    _require_equal(model.get("optimizer_max_iterations"), 2000, "最大迭代")
    _require_equal(float(model.get("optimizer_ftol")), 1.0e-12, "ftol")
    _require_equal(float(model.get("optimizer_gtol")), 1.0e-8, "gtol")
    _require_equal(float(model.get("probability_clip")), 1.0e-12, "概率裁剪")
    for key in (
        "hyperparameter_grid_allowed",
        "automatic_variable_selection_allowed",
        "sign_reversal_allowed",
        "tree_boosting_or_neural_model_allowed",
    ):
        _require_equal(model.get(key), False, key)

    split = config["prequential_split_contract"]
    _require_equal(split.get("method"), "ANNUAL_EXPANDING_LABEL_MATURED_PREQUENTIAL", "预测切分")
    _require_equal(split.get("model_vintage_month_day"), "01-01", "模型版本日")
    _require_equal(split.get("training_row_rule"), "HORIZON_END_DATE_STRICTLY_BEFORE_MODEL_VINTAGE", "训练标签时钟")
    _require_equal(split.get("training_positive_event_rule"), "WHOLE_EVENT_MAX_HORIZON_END_STRICTLY_BEFORE_MODEL_VINTAGE", "训练事件规则")
    _require_equal(split.get("evaluation_positive_event_vintage"), "CALENDAR_YEAR_OF_EVENT_MINIMUM_ORIGIN_DATE", "正事件预测版本")
    _require_equal(split.get("evaluation_negative_vintage"), "CALENDAR_YEAR_OF_ORIGIN_DATE", "非事件预测版本")
    _require_equal(split.get("positive_event_rows_use_one_vintage_even_if_EVENT_CROSSES_YEAR"), True, "跨年事件规则")
    _require_equal(split.get("future_calendar_year_training_allowed"), False, "未来年份训练权限")
    _require_equal(split.get("evaluation_row_may_train_its_own_prediction"), False, "自训练权限")
    _require_equal(split.get("minimum_training_positive_events"), 1, "最少训练事件")
    _require_equal(split.get("minimum_training_negative_risk_days"), 1, "最少训练负样本")
    _require_equal(split.get("calibration_split_used"), False, "校准切分")
    _require_equal(split.get("first_possible_model_vintage_year"), 2016, "首模型年")
    _require_equal(split.get("last_model_vintage_year"), 2026, "末模型年")

    eras = config["fixed_eras"]
    _require_equal(
        eras,
        [
            {"id": "ERA_1_2015_2017", "start": "2015-01-05", "end": "2017-12-29"},
            {"id": "ERA_2_2018_2020", "start": "2018-01-02", "end": "2020-12-31"},
            {"id": "ERA_3_2021_2023", "start": "2021-01-04", "end": "2023-12-29"},
            {"id": "ERA_4_2024_CUTOFF", "start": "2024-01-02", "end": "2026-08-14"},
        ],
        "冻结时代",
    )
    gate = config["g2_gate_contract"]
    _require_equal(gate.get("comparison"), "B2_MINUS_B1", "G2 比较")
    _require_equal(gate.get("primary_metric"), "EVENT_WEIGHTED_LOG_LOSS_IMPROVEMENT_B1_MINUS_B2", "G2 主指标")
    _require_equal(gate.get("overall_improvement_minimum_exclusive"), 0.0, "总体门")
    _require_equal(gate.get("same_direction_eras_required"), 3, "时代门")
    _require_equal(gate.get("total_eras"), 4, "时代总数")
    _require_equal(gate.get("no_prediction_era_counts_as_positive"), False, "无预测时代规则")
    _require_equal(gate.get("bootstrap_units"), ["EVENT", "CALENDAR_YEAR"], "Bootstrap 块")
    _require_equal(gate.get("bootstrap_repetitions"), 5000, "Bootstrap 次数")
    _require_equal(gate.get("bootstrap_seed"), 20260903, "Bootstrap 种子")
    _require_equal(gate.get("one_sided_confidence"), 0.90, "单侧置信度")
    _require_equal(gate.get("lower_quantile"), 0.10, "下分位数")
    _require_equal(gate.get("lower_bound_minimum_exclusive"), 0.0, "Bootstrap 门")
    _require_equal(gate.get("formal_status_always"), "NOT_ADMISSIBLE_ACTUAL_G1_REMAINS_NO_VIEW", "正式状态")

    sample_columns = set(config["inputs"]["g1_sample_eligibility"]["read_columns"])
    forbidden_columns = {
        "minimum_path_return",
        "first_breach_date",
        "cash_dividend_per_share_in_horizon",
        "forward_return",
        "portfolio_return",
        "nav",
        "sharpe",
        "drawdown",
    }
    if sample_columns.intersection(forbidden_columns):
        raise EvidenceContractError("G2 输入列越过标签或组合边界")

    required_forbidden = {
        "PRESENT_ASSUMED_G1_PASS_AS_OBSERVED_FACT",
        "FABRICATE_EVENTS_FEATURES_OR_NO_VIEW_ROWS",
        "USE_OBSERVATION_AFTER_2026_08_14",
        "USE_FUTURE_CALENDAR_YEAR_OR_UNMATURED_LABEL_IN_A_PREDICTION_MODEL",
        "RELAX_B2_VS_B1_COMMON_SAMPLE",
        "CHANGE_L2_OR_SEARCH_HYPERPARAMETERS",
        "SELECT_PROBABILITY_THRESHOLD_OR_RUN_G4",
        "READ_OR_GENERATE_PORTFOLIO_RETURN_NAV_SHARPE_OR_DRAWDOWN",
        "CREATE_POSITION_ORDER_BROKER_PAPER_SHADOW_OR_LIVE_ACTION",
    }
    forbidden = set(config["forbidden_actions"])
    if not required_forbidden.issubset(forbidden):
        raise EvidenceContractError(
            f"G2 禁止动作缺项：{sorted(required_forbidden.difference(forbidden))}"
        )
    outputs = list(config["outputs"].values())
    if len(outputs) != len(set(outputs)):
        raise EvidenceContractError("反事实 G2 输出路径重复")


def verify_immutable_parent(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    verify_v2_protocol_manifest()
    verify_g1_manifest()
    verify_historical_audit_manifest()
    parent = config["immutable_parent"]
    evidence = {
        name: _validate_file_identity(contract, f"不可变父链 {name}")
        for name, contract in parent.items()
    }
    g1 = read_json_strict(_project_path(parent["g1_terminal_status"]["path"]))
    if not isinstance(g1, dict):
        raise EvidenceContractError("父 G1 状态必须是对象")
    _verify_self_digest(g1, "status_payload_sha256", "父 G1 状态")
    _require_equal(g1.get("G1_DATA_AND_EVENTS"), parent["g1_terminal_status"]["required_g1_state"], "真实 G1")
    _require_equal(g1.get("branch_state"), parent["g1_terminal_status"]["required_branch_state"], "真实 G1 分支")
    _require_equal(g1.get("model_trained"), False, "父 G1 模型训练")
    _require_equal(g1.get("position_impact"), 0, "父 G1 仓位影响")

    replay = read_json_strict(_project_path(parent["g1_clean_replay_receipt"]["path"]))
    if not isinstance(replay, dict):
        raise EvidenceContractError("父 G1 重放收据必须是对象")
    _verify_self_digest(replay, "receipt_payload_sha256", "父 G1 重放收据")
    _require_equal(replay.get("status"), parent["g1_clean_replay_receipt"]["required_status"], "父 G1 重放")

    audit = read_json_strict(
        _project_path(parent["historical_coverage_audit_status"]["path"])
    )
    if not isinstance(audit, dict):
        raise EvidenceContractError("历史覆盖审计状态必须是对象")
    _verify_self_digest(audit, "status_payload_sha256", "历史覆盖审计状态")
    _require_equal(audit.get("audit_status"), parent["historical_coverage_audit_status"]["required_audit_status"], "历史审计状态")
    _require_equal(audit.get("observation_after_historical_cutoff_read"), parent["historical_coverage_audit_status"]["required_after_cutoff_read"], "历史审计截止日")
    _require_equal(audit.get("G1_DATA_AND_EVENTS"), "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY", "历史审计真实 G1")
    return evidence


def verify_registered_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: _validate_file_identity(contract, f"冻结输入 {name}")
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
    freeze_contract = config["freeze_contract"]
    return (
        _project_path(str(freeze_contract["manifest_output"])),
        _project_path(str(freeze_contract["freeze_receipt_output"])),
    )


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, receipt_path = _freeze_outputs(config)
    existing = [str(path) for path in (manifest_path, receipt_path) if path.exists()]
    if existing:
        raise EvidenceContractError(f"反事实 G2 冻结输出已存在：{existing}")
    business_outputs = [_project_path(str(path)) for path in config["outputs"].values()]
    existing_business = [str(path) for path in business_outputs if path.exists()]
    if existing_business:
        raise EvidenceContractError(f"反事实 G2 业务输出已存在：{existing_business}")
    parent = verify_immutable_parent(config)
    inputs = verify_registered_inputs(config)
    freeze_contract = config["freeze_contract"]
    implementation = _hash_group([str(path) for path in freeze_contract["implementation_files"]])
    governance = _hash_group([str(path) for path in freeze_contract["governance_files"]])
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest: dict[str, Any] = {
        "execution_id": EXECUTION_ID,
        "version": "1.0.0",
        "status": "FROZEN_G2_COUNTERFACTUAL_ASSUMED_G1_PASS_ONLY",
        "frozen_at": now,
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_bytes": config_path.stat().st_size,
        "config_sha256": sha256_file(config_path),
        "config_canonical_sha256": canonical_sha256(config),
        "counterfactual_authority": config["counterfactual_authority"],
        "immutable_parent": parent,
        "inputs": inputs,
        "common_sample_contract": config["common_sample_contract"],
        "model_contract": config["model_contract"],
        "prequential_split_contract": config["prequential_split_contract"],
        "fixed_eras": config["fixed_eras"],
        "g2_gate_contract": config["g2_gate_contract"],
        "implementation_files": implementation,
        "governance_files": governance,
        "outputs": config["outputs"],
        "forbidden_actions": config["forbidden_actions"],
        "actual_g1_state_overwritten": False,
        "model_trained_during_freeze": False,
        "prediction_metric_generated_during_freeze": False,
        "observation_after_historical_cutoff_read": False,
        "portfolio_or_performance_artifact_read": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_G2_COUNTERFACTUAL_V1_FREEZE",
        "created_at": now,
        "status": "PASS_COUNTERFACTUAL_G2_PROTOCOL_FROZEN_BEFORE_MODEL_FIT",
        "manifest": file_evidence(manifest_path, project_root=ROOT),
        "actual_g1_state_retained": True,
        "assumed_g1_scope": "G2_ONLY",
        "model_trained_during_freeze": False,
        "prediction_metric_generated_during_freeze": False,
        "portfolio_or_performance_artifact_read": False,
        "position_impact": 0,
    }
    atomic_write_json_new(receipt_path, receipt)
    print("反事实 G2 V1 已在拟合前冻结；真实 G1 仍为 NO_VIEW。")
    return manifest


def verify_frozen_manifest(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    validate_config(config)
    manifest_path, _ = _freeze_outputs(config)
    manifest = read_json_strict(manifest_path)
    if not isinstance(manifest, dict):
        raise EvidenceContractError("反事实 G2 manifest 必须是对象")
    verify_manifest_payload(manifest)
    _require_equal(manifest.get("execution_id"), EXECUTION_ID, "执行 identity")
    _require_equal(manifest.get("config_sha256"), sha256_file(config_path), "配置哈希")
    _require_equal(manifest.get("actual_g1_state_overwritten"), False, "真实 G1 写入")
    verify_immutable_parent(config)
    verify_registered_inputs(config)
    for group_name in ("implementation_files", "governance_files"):
        group = manifest.get(group_name)
        if not isinstance(group, Mapping):
            raise EvidenceContractError(f"manifest 缺少 {group_name}")
        for relative, metadata in group.items():
            _validate_file_identity(
                {"path": relative, "bytes": metadata["bytes"], "sha256": metadata["sha256"]},
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
            print("反事实 G2 manifest、父链和输入身份验证通过。")
        else:
            freeze(args.config)
        return 0
    except (EvidenceContractError, KeyError, TypeError, ValueError, OSError) as exc:
        print(f"反事实 G2 冻结失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
