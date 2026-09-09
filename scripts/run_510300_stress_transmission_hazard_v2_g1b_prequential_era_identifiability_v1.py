"""冻结并执行 510300 V2 G1B 前序年代可识别性门。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq
import yaml

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.stress_transmission_hazard_v2_g0_1_version_lock_v1 import (
    VersionLockError,
    atomic_write_json_new,
    canonical_sha256,
    file_evidence,
    git_branch,
    git_head,
    project_path,
    sha256_file,
    verify_commit_contains_files,
    verify_file_identity,
    verify_payload_sha256,
)
from research.stress_transmission_hazard_v2_g1b_prequential_era_identifiability_v1 import (
    DERIVED_ALLOWED_FIELDS,
    EXECUTION_ID,
    FORBIDDEN_SOURCE_COLUMNS,
    INPUT_READ_COLUMNS,
    PROGRAM_ID,
    G1BArtifacts,
    G1BIdentifiabilityError,
    audit_prequential_identifiability,
)


DEFAULT_CONFIG = (
    ROOT
    / "config"
    / "510300_stress_transmission_hazard_v2_g1b_prequential_era_identifiability_v1.yaml"
)


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise G1BIdentifiabilityError(f"YAML 顶层必须是对象：{path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise G1BIdentifiabilityError(f"JSON 顶层必须是对象：{path}")
    return payload


def _verify_required_fields(
    payload: Mapping[str, Any],
    required: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key, expected in required.items():
        actual = payload.get(key)
        if actual != expected:
            raise G1BIdentifiabilityError(
                f"{label}.{key} 漂移：actual={actual!r}, expected={expected!r}"
            )


def _verify_parents(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name, contract in config["immutable_parent"].items():
        path = verify_file_identity(root, contract, label=f"immutable_parent.{name}")
        if path.suffix.casefold() != ".json":
            continue
        payload = _load_json(path)
        hash_field = contract.get("payload_hash_field")
        if hash_field:
            verify_payload_sha256(
                payload,
                field=str(hash_field),
                label=f"immutable_parent.{name}",
            )
        _verify_required_fields(
            payload,
            contract.get("required_fields", {}),
            label=f"immutable_parent.{name}",
        )
        loaded[name] = payload
    return loaded


def _verify_config_invariants(config: Mapping[str, Any]) -> None:
    if str(config["program"]["program_id"]) != PROGRAM_ID:
        raise G1BIdentifiabilityError("program_id 漂移")
    if str(config["program"]["execution_id"]) != EXECUTION_ID:
        raise G1BIdentifiabilityError("execution_id 漂移")
    if str(config["program"]["historical_cutoff"]) != "2026-08-14":
        raise G1BIdentifiabilityError("历史截止日必须保持 2026-08-14")
    authorization = config["authorization"]
    forbidden_flags = (
        "b1_b2_or_b3_model_fit_allowed",
        "formal_g2_allowed_in_this_execution",
        "counterfactual_or_full_sample_g2_allowed",
        "return_portfolio_sharpe_drawdown_or_position_read_allowed",
        "broker_paper_shadow_position_order_or_live_action_allowed",
    )
    if any(bool(authorization[key]) for key in forbidden_flags):
        raise G1BIdentifiabilityError("G1B 配置错误地开放了模型、G2、收益或交易权限")
    if list(config["read_contract"]["physical_read_columns"]) != list(
        INPUT_READ_COLUMNS
    ):
        raise G1BIdentifiabilityError("G1B 物理读取列漂移")
    if list(config["read_contract"]["derived_allowed_fields"]) != list(
        DERIVED_ALLOWED_FIELDS
    ):
        raise G1BIdentifiabilityError("G1B 派生字段契约漂移")
    if list(config["read_contract"]["forbidden_source_columns"]) != list(
        FORBIDDEN_SOURCE_COLUMNS
    ):
        raise G1BIdentifiabilityError("G1B 禁止源字段契约漂移")
    if bool(config["read_contract"]["actual_row_values_may_be_read_before_freeze"]):
        raise G1BIdentifiabilityError("冻结前不得读取实际行值")
    if bool(config["prequential_clock"]["event_may_split_across_model_vintages"]):
        raise G1BIdentifiabilityError("正事件不得跨模型版本拆分")
    if bool(config["prequential_clock"]["future_training_allowed"]):
        raise G1BIdentifiabilityError("不得使用未来标签训练")
    if bool(config["prequential_clock"]["full_sample_fit_allowed"]):
        raise G1BIdentifiabilityError("不得使用完整样本拟合")
    if int(config["training_identifiability"]["minimum_positive_independent_events"]) != 1:
        raise G1BIdentifiabilityError("最低训练正事件数必须保持 1")
    if int(config["training_identifiability"]["minimum_negative_risk_days"]) != 1:
        raise G1BIdentifiabilityError("最低训练负风险日数必须保持 1")
    if int(
        config["g1b_gate"]["minimum_prequential_predicted_independent_events_per_era"]
    ) != 5:
        raise G1BIdentifiabilityError("每时代事件门必须保持 5")
    if int(config["g1b_gate"]["minimum_qualified_eras_for_formal_g2"]) != 3:
        raise G1BIdentifiabilityError("合格时代门必须保持 3")
    if int(config["g1b_gate"]["logical_maximum_qualified_eras_before_value_read"]) != 2:
        raise G1BIdentifiabilityError("冻结前逻辑上界必须保持两个时代")
    if bool(config["freeze_scope"]["remote_push_required"]):
        raise G1BIdentifiabilityError("本任务不要求远程推送")
    if not bool(config["freeze_scope"]["local_commit_required_before_value_read"]):
        raise G1BIdentifiabilityError("实际值读取前必须完成本地提交")


def _input_schema_metadata(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        "row_count": int(parquet.metadata.num_rows),
        "column_count": int(parquet.metadata.num_columns),
        "schema_columns": list(parquet.schema_arrow.names),
    }


def _verify_input_contract_without_values(
    root: Path,
    config: Mapping[str, Any],
    parents: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    contract = config["input"]["g1_sample_eligibility"]
    path = verify_file_identity(root, contract, label="G1B 样本输入")
    actual_schema = _input_schema_metadata(path)
    expected_schema = {
        "row_count": int(contract["row_count"]),
        "column_count": int(contract["column_count"]),
        "schema_columns": [str(value) for value in contract["schema_columns"]],
    }
    if actual_schema != expected_schema:
        raise G1BIdentifiabilityError(
            f"G1B 输入 schema 漂移：actual={actual_schema}, expected={expected_schema}"
        )
    g0_manifest = parents["g0_1_manifest"]
    matches = [
        item
        for item in g0_manifest["data_files"]
        if str(item["path"]) == str(contract["path"])
    ]
    if len(matches) != 1:
        raise G1BIdentifiabilityError("G0.1 manifest 未唯一绑定 G1B 输入")
    parent_contract = matches[0]
    keys = (
        "path",
        "bytes",
        "sha256",
        "format",
        "row_count",
        "column_count",
        "columns",
        "persisted_semantic_sha256",
    )
    expected_parent = {
        "path": str(contract["path"]),
        "bytes": int(contract["bytes"]),
        "sha256": str(contract["sha256"]),
        "format": str(contract["format"]),
        "row_count": int(contract["row_count"]),
        "column_count": int(contract["column_count"]),
        "columns": [str(value) for value in contract["schema_columns"]],
        "persisted_semantic_sha256": str(contract["persisted_semantic_sha256"]),
    }
    actual_parent = {key: parent_contract[key] for key in keys}
    if actual_parent != expected_parent:
        raise G1BIdentifiabilityError("G1B 输入与 G0.1 数据 manifest 身份冲突")
    return {
        **file_evidence(path, root=root),
        **actual_schema,
        "format": "PARQUET",
        "persisted_semantic_sha256": str(contract["persisted_semantic_sha256"]),
        "semantic_identity_source": str(
            config["immutable_parent"]["g0_1_manifest"]["path"]
        ),
        "actual_row_values_read": False,
    }


def _implementation_evidence(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = []
    for relative in config["freeze_scope"]["implementation_files"]:
        evidence.append(file_evidence(project_path(root, str(relative)), root=root))
    return evidence


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """在实际样本行值读取前冻结 G1B 的全部规则和输入身份。"""

    config_path = config_path.absolute()
    config = _load_yaml(config_path)
    _verify_config_invariants(config)
    required_branch = str(config["freeze_scope"]["branch_required"])
    current_branch = git_branch(ROOT)
    if current_branch != required_branch:
        raise G1BIdentifiabilityError(
            f"G1B 必须在专用分支冻结：actual={current_branch}, expected={required_branch}"
        )
    manifest_path = project_path(ROOT, str(config["outputs"]["manifest"]))
    receipt_path = project_path(ROOT, str(config["outputs"]["freeze_receipt"]))
    if manifest_path.exists() or receipt_path.exists():
        raise G1BIdentifiabilityError("G1B manifest 或冻结收据已存在，禁止覆盖")

    parents = _verify_parents(ROOT, config)
    input_evidence = _verify_input_contract_without_values(ROOT, config, parents)
    implementation = _implementation_evidence(ROOT, config)
    manifest: dict[str, Any] = {
        "program_id": PROGRAM_ID,
        "execution_id": EXECUTION_ID,
        "version": str(config["program"]["version"]),
        "frozen_at": _now(),
        "status": "FROZEN_G1B_BEFORE_ACTUAL_ALLOWED_FIELD_VALUE_READ",
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "pre_freeze_head_sha": git_head(ROOT),
        "pre_freeze_branch": current_branch,
        "implementation_files": implementation,
        "input": input_evidence,
        "physical_read_columns": list(INPUT_READ_COLUMNS),
        "derived_allowed_fields": list(DERIVED_ALLOWED_FIELDS),
        "forbidden_source_columns": list(FORBIDDEN_SOURCE_COLUMNS),
        "fixed_eras": config["fixed_eras"],
        "prequential_clock": config["prequential_clock"],
        "training_identifiability": config["training_identifiability"],
        "g1a_frozen_counts": config["g1a_frozen_counts"],
        "g1b_gate": config["g1b_gate"],
        "logical_upper_bound_proved_before_actual_row_value_read": True,
        "logical_maximum_qualified_eras": 2,
        "actual_input_row_values_read": False,
        "model_trained": False,
        "formal_g2_run": False,
        "return_or_portfolio_read": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt: dict[str, Any] = {
        "receipt_id": f"{EXECUTION_ID}_FREEZE_RECEIPT",
        "created_at": _now(),
        "status": "PASS_G1B_RULES_AND_INPUT_IDENTITY_FROZEN_BEFORE_VALUE_READ",
        "manifest": file_evidence(manifest_path, root=ROOT),
        "pre_freeze_head_sha": manifest["pre_freeze_head_sha"],
        "pre_freeze_branch": current_branch,
        "schema_metadata_only_inspected": True,
        "actual_input_row_values_read": False,
        "logical_maximum_qualified_eras": 2,
        "model_trained": False,
        "formal_g2_run": False,
        "return_or_portfolio_read": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(receipt_path, receipt)
    print(
        "G1B 已在实际行值读取前冻结：物理读取列 7 个，逻辑最多两个合格时代；"
        "尚未执行 G1B，G2 未运行。",
        flush=True,
    )
    return receipt


def _load_frozen_manifest(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    manifest_path = project_path(root, str(config["outputs"]["manifest"]))
    manifest = _load_json(manifest_path)
    verify_payload_sha256(
        manifest,
        field="manifest_payload_sha256",
        label="G1B manifest",
    )
    config_path = project_path(root, str(manifest["config_file"]))
    if sha256_file(config_path) != str(manifest["config_sha256"]):
        raise G1BIdentifiabilityError("G1B 冻结后配置漂移")
    for contract in manifest["implementation_files"]:
        verify_file_identity(root, contract, label=f"G1B 实现 {contract['path']}")
    parents = _verify_parents(root, config)
    input_evidence = _verify_input_contract_without_values(root, config, parents)
    keys = (
        "path",
        "bytes",
        "sha256",
        "format",
        "row_count",
        "column_count",
        "schema_columns",
        "persisted_semantic_sha256",
    )
    if {key: input_evidence[key] for key in keys} != {
        key: manifest["input"][key] for key in keys
    }:
        raise G1BIdentifiabilityError("G1B 执行前输入身份或 schema 漂移")
    if bool(manifest["actual_input_row_values_read"]):
        raise G1BIdentifiabilityError("G1B manifest 错误声称冻结前已读取实际值")
    return manifest, parents


def _verify_freeze_commit(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected_commit: str,
) -> None:
    if git_head(root) != expected_commit:
        raise G1BIdentifiabilityError(
            f"当前 HEAD 不是指定 G1B 冻结提交：actual={git_head(root)}, expected={expected_commit}"
        )
    verify_commit_contains_files(
        root,
        commit_sha=expected_commit,
        contracts=manifest["implementation_files"],
    )
    generated = [
        str(config["outputs"]["manifest"]),
        str(config["outputs"]["freeze_receipt"]),
    ]
    verify_commit_contains_files(
        root,
        commit_sha=expected_commit,
        contracts=[
            file_evidence(project_path(root, relative), root=root)
            for relative in generated
        ],
    )


def _atomic_write_text_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except FileExistsError as exc:
        raise G1BIdentifiabilityError(f"输出已存在，禁止覆盖：{path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except FileExistsError as exc:
        raise G1BIdentifiabilityError(f"输出已存在，禁止覆盖：{path}") from exc
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        frame.to_csv(handle, index=False, lineterminator="\n", date_format="%Y-%m-%d")
        handle.flush()
        os.fsync(handle.fileno())


def _report_text(artifacts: G1BArtifacts) -> str:
    result = artifacts.result
    lines = [
        "# 510300 压力传导危险率 V2：G1B 前序年代可识别性 V1",
        "",
        "## 裁决",
        "",
        f"- `G1B_PREQUENTIAL_ERA_IDENTIFIABILITY`：`{result['G1B_PREQUENTIAL_ERA_IDENTIFIABILITY']}`。",
        f"- 合格时代：{result['qualified_era_count']}/4；正式 G2 要求至少 {result['minimum_qualified_eras_required']} 个。",
        f"- 真正前序可预测独立事件：{result['prequential_predicted_independent_event_count']}。",
        f"- 冻结逻辑上的最大合格时代数：{result['logical_max_qualified_eras']}。",
        "",
        "## 四时代审计",
        "",
        "| 时代 | G1A事件 | 逻辑上界 | 实际前序事件 | 训练可识别 | 无事件拆分 | 无未来训练 | 合格 |",
        "|---|---:|---:|---:|---|---|---|---|",
    ]
    for row in artifacts.era_audit.itertuples(index=False):
        lines.append(
            f"| `{row.era_id}` | {row.g1a_identifiable_event_count} | "
            f"{row.logical_prequential_upper_bound} | "
            f"{row.prequential_predicted_independent_event_count} | "
            f"{'是' if row.b1_and_b2_training_identifiable else '否'} | "
            f"{'是' if row.no_event_split else '否'} | "
            f"{'是' if row.no_future_training else '否'} | "
            f"{'是' if row.qualified_era else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 时间与读取边界",
            "",
            "- 模型 vintage 使用每个日历季度结束后的首个样本交易日。",
            "- 正事件只有在整事件最大期限严格早于 vintage 时才可进入训练；评价事件整体绑定到其最早原点之前最近的 vintage。",
            "- 本执行只读取 origin_date、horizon_end_date、bad10、event_id、b2_vs_b1_eligible、split_assignment、calendar_year_block_id。",
            "- era_id 由事件最早 origin_date 和原四时代边界派生；没有读取 B1、F、M、T、T×F、T×M、最小路径收益或组合收益。",
            "- 没有拟合 B0/B1/B2/B3，没有生成概率、预测损失、AUC、警报、净值、夏普、仓位或订单。",
            "",
            "## 后续状态",
            "",
            "正式 G2 未运行。若合格时代少于三个，则 V2 历史分支归档且不得通过改时代、降门槛、完整样本拟合或收益结果进行营救。历史四态收益底座与 M/F/T/BAD10 前向观察框架保留为数据基础设施。",
        ]
    )
    return "\n".join(lines) + "\n"


def execute(
    *,
    config_path: Path,
    expected_freeze_commit: str,
) -> dict[str, Any]:
    """在冻结提交验证后仅读取七个许可列并执行 G1B。"""

    config_path = config_path.absolute()
    config = _load_yaml(config_path)
    _verify_config_invariants(config)
    manifest, _ = _load_frozen_manifest(ROOT, config)
    _verify_freeze_commit(
        ROOT,
        config,
        manifest,
        expected_commit=expected_freeze_commit,
    )
    output_keys = (
        "event_audit",
        "vintage_audit",
        "era_audit",
        "report",
        "execution_receipt",
        "final_status",
    )
    output_paths = {
        key: project_path(ROOT, str(config["outputs"][key])) for key in output_keys
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise G1BIdentifiabilityError(f"G1B 输出已存在，禁止覆盖：{existing}")

    input_path = project_path(
        ROOT, str(config["input"]["g1_sample_eligibility"]["path"])
    )
    frame = pd.read_parquet(input_path, columns=list(INPUT_READ_COLUMNS))
    if list(frame.columns) != list(INPUT_READ_COLUMNS):
        raise G1BIdentifiabilityError("Parquet 引擎返回了冻结范围外的列")
    training = config["training_identifiability"]
    gate = config["g1b_gate"]
    artifacts = audit_prequential_identifiability(
        frame,
        fixed_eras=config["fixed_eras"],
        expected_g1a=config["g1a_frozen_counts"],
        minimum_training_positive_events=int(
            training["minimum_positive_independent_events"]
        ),
        minimum_training_negative_risk_days=int(training["minimum_negative_risk_days"]),
        minimum_prequential_events_per_era=int(
            gate["minimum_prequential_predicted_independent_events_per_era"]
        ),
        minimum_qualified_eras=int(gate["minimum_qualified_eras_for_formal_g2"]),
    )
    if int(artifacts.result["logical_max_qualified_eras"]) != int(
        gate["logical_maximum_qualified_eras_before_value_read"]
    ):
        raise G1BIdentifiabilityError("执行结果与冻结前逻辑上界冲突")

    _atomic_write_csv_new(output_paths["event_audit"], artifacts.event_audit)
    _atomic_write_csv_new(output_paths["vintage_audit"], artifacts.vintage_audit)
    _atomic_write_csv_new(output_paths["era_audit"], artifacts.era_audit)
    _atomic_write_text_new(output_paths["report"], _report_text(artifacts))
    artifact_evidence = {
        key: file_evidence(output_paths[key], root=ROOT)
        for key in ("event_audit", "vintage_audit", "era_audit", "report")
    }
    execution_receipt: dict[str, Any] = {
        "receipt_id": f"{EXECUTION_ID}_EXECUTION_RECEIPT",
        "created_at": _now(),
        "status": "PASS_G1B_COUNTING_EXECUTION_NO_MODEL_FIT",
        "freeze_commit_sha": expected_freeze_commit,
        "manifest": file_evidence(
            project_path(ROOT, str(config["outputs"]["manifest"])), root=ROOT
        ),
        "input": file_evidence(input_path, root=ROOT),
        "input_value_read_started_only_after_freeze_commit_verified": True,
        "physical_read_columns": list(INPUT_READ_COLUMNS),
        "read_column_count": len(frame.columns),
        "read_row_count": int(len(frame)),
        "derived_allowed_fields": list(DERIVED_ALLOWED_FIELDS),
        "forbidden_source_columns_read": [],
        "artifacts": artifact_evidence,
        "result": artifacts.result,
        "model_trained": False,
        "formal_g2_run": False,
        "counterfactual_or_full_sample_g2_run": False,
        "return_or_portfolio_read": False,
        "position_impact": 0,
    }
    execution_receipt["receipt_payload_sha256"] = canonical_sha256(execution_receipt)
    atomic_write_json_new(output_paths["execution_receipt"], execution_receipt)

    passed = bool(artifacts.result["formal_g2_allowed"])
    g1b_state = str(artifacts.result["G1B_PREQUENTIAL_ERA_IDENTIFIABILITY"])
    final_status: dict[str, Any] = {
        "MODEL_ID": PROGRAM_ID,
        "EXECUTION_ID": EXECUTION_ID,
        "updated_at": _now(),
        "PRIOR_G0": "PASS",
        "G0_1_POST_REMEDIATION_VERSION_LOCK": "PASS",
        "HISTORICAL_DATA_REMEDIATION": (
            "PASS_PROVABLE_OFFICIAL_SUSPENSION_REMEDIATION"
        ),
        "G1A_COMMON_SAMPLE_EVENT_IDENTIFIABILITY": "PASS_B2_ONLY",
        "G1A_B2_IDENTIFIABLE_EVENTS": 31,
        "G1A_B2_NON_EVENT_RISK_DAYS": 1102,
        "G1A_B2_COMMON_SAMPLE_DAYS": 1279,
        "G1B_PREQUENTIAL_ERA_IDENTIFIABILITY": g1b_state,
        "G1B_PREQUENTIAL_EVENTS_BY_ERA": artifacts.result[
            "prequential_events_by_era"
        ],
        "G1B_QUALIFIED_ERA_COUNT": int(artifacts.result["qualified_era_count"]),
        "G1B_REQUIRED_QUALIFIED_ERA_COUNT": 3,
        "G2_STRUCTURAL_INCREMENT": (
            "NOT_RUN_REQUIRES_SEPARATE_FORMAL_G2_FREEZE"
            if passed
            else "NOT_RUN_BLOCKED_BY_G1B"
        ),
        "FORMAL_G2_MODEL_FIT": "NOT_ALLOWED_IN_THIS_EXECUTION",
        "COUNTERFACTUAL_OR_FULL_SAMPLE_G2": "NOT_ALLOWED",
        "B3_FULL_MODEL": "NO_VIEW_31_LT_40",
        "G3_MACRO_INCREMENT": "NOT_RUN_B3_NOT_IDENTIFIABLE",
        "G4_THROUGH_G7": "NOT_RUN",
        "ECONOMIC_HYPOTHESIS": "UNTESTED",
        "MODEL_TRAINED": False,
        "RETURN_EVALUATION": "NOT_ALLOWED",
        "PORTFOLIO_EVALUATION": "NOT_ALLOWED",
        "HISTORICAL_BRANCH": (
            "OPEN_ONLY_FOR_SEPARATE_FORMAL_G2_FREEZE"
            if passed
            else "ARCHIVED_NO_RESCUE"
        ),
        "FORWARD_DATA_OBSERVATORY": "MAY_CONTINUE_WITHOUT_MODEL_SIGNAL_OR_POSITION",
        "POSITION_IMPACT": 0,
        "freeze_commit_sha": expected_freeze_commit,
        "execution_receipt": file_evidence(
            output_paths["execution_receipt"], root=ROOT
        ),
        "era_audit": artifact_evidence["era_audit"],
        "report": artifact_evidence["report"],
    }
    final_status["status_payload_sha256"] = canonical_sha256(final_status)
    atomic_write_json_new(output_paths["final_status"], final_status)
    print(
        f"G1B 完成：状态={g1b_state}，"
        f"合格时代={artifacts.result['qualified_era_count']}/4，"
        f"时代前序事件={artifacts.result['prequential_events_by_era']}；"
        "没有拟合模型，G2 未运行。",
        flush=True,
    )
    return final_status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="G1B 冻结配置路径",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze", help="在实际允许字段值读取前冻结 G1B")
    execute_parser = subparsers.add_parser(
        "execute", help="验证冻结提交后仅读取七个许可列并执行 G1B"
    )
    execute_parser.add_argument("--expected-freeze-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            freeze(args.config)
        elif args.command == "execute":
            execute(
                config_path=args.config,
                expected_freeze_commit=str(args.expected_freeze_commit),
            )
        else:
            raise G1BIdentifiabilityError(f"未知命令：{args.command}")
    except (
        G1BIdentifiabilityError,
        VersionLockError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        print(f"G1B 执行失败：{exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

