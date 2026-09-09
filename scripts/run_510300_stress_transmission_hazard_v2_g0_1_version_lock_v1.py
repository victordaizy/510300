"""执行 510300 V2 修复后 G0.1 版本锁定与独立 worktree 重放。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.stress_transmission_hazard_v2_g0_1_version_lock_v1 import (
    EXECUTION_ID,
    PROGRAM_ID,
    VersionLockError,
    atomic_write_json_new,
    canonical_sha256,
    file_evidence,
    git_branch,
    git_head,
    materialize_data_files,
    merge_expected_contract,
    normalize_relative_path,
    parquet_evidence,
    project_path,
    resolve_git_scope,
    run_git,
    sha256_file,
    validate_expected_replay_metrics,
    verify_commit_contains_files,
    verify_data_manifest,
    verify_file_identity,
    verify_payload_sha256,
    verify_tracked_worktree_clean,
)


DEFAULT_CONFIG = (
    ROOT
    / "config"
    / "510300_stress_transmission_hazard_v2_g0_1_post_remediation_version_lock_v1.yaml"
)


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise VersionLockError(f"YAML 顶层必须是对象：{path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise VersionLockError(f"JSON 顶层必须是对象：{path}")
    return payload


def _atomic_copy_new(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(target, flags)
    except FileExistsError as exc:
        raise VersionLockError(f"输出已存在，禁止覆盖：{target}") from exc
    try:
        with source.open("rb") as input_handle, os.fdopen(descriptor, "wb") as output_handle:
            shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            output_handle.flush()
            os.fsync(output_handle.fileno())
    except Exception:
        if target.exists():
            target.unlink()
        raise


def _verify_required_fields(
    payload: Mapping[str, Any],
    required: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key, expected in required.items():
        actual = payload.get(key)
        if actual != expected:
            raise VersionLockError(
                f"{label}.{key} 不符合冻结要求：actual={actual!r}, expected={expected!r}"
            )


def _verify_immutable_parents(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for name, contract in config["immutable_parent"].items():
        path = verify_file_identity(root, contract, label=f"immutable_parent.{name}")
        if path.suffix.casefold() == ".json":
            payload = _load_json(path)
            if "receipt_payload_sha256" in payload:
                verify_payload_sha256(
                    payload,
                    field="receipt_payload_sha256",
                    label=f"immutable_parent.{name}",
                )
            if "manifest_payload_sha256" in payload:
                verify_payload_sha256(
                    payload,
                    field="manifest_payload_sha256",
                    label=f"immutable_parent.{name}",
                )
            _verify_required_fields(
                payload,
                contract.get("required_fields", {}),
                label=f"immutable_parent.{name}",
            )
            loaded[name] = payload
    return loaded


def _git_scope(root: Path, config: Mapping[str, Any]) -> list[Path]:
    contract = config["git_scope_contract"]
    return resolve_git_scope(
        root,
        roots=[str(value) for value in contract["roots"]],
        filename_regex=str(contract["filename_regex"]),
        allowed_suffixes=[str(value) for value in contract["allowed_suffixes"]],
        explicit_paths=[str(value) for value in contract["explicit_paths"]],
        excluded_paths=[str(value) for value in contract.get("excluded_paths", [])],
    )


def _add_mapping_contracts(
    target: dict[str, dict[str, Any]],
    mapping: Mapping[str, Any],
    *,
    source_prefix: str,
) -> None:
    for name, contract in mapping.items():
        if not isinstance(contract, Mapping):
            continue
        if {"path", "bytes", "sha256"}.issubset(contract):
            merge_expected_contract(
                target,
                contract,
                source=f"{source_prefix}.{name}",
            )


def _collect_data_contracts(
    root: Path,
    config: Mapping[str, Any],
    *,
    git_scope_relatives: set[str],
) -> list[dict[str, Any]]:
    parents = config["immutable_parent"]
    remediation_config_path = verify_file_identity(
        root,
        parents["remediation_config"],
        label="immutable_parent.remediation_config",
    )
    remediation = _load_yaml(remediation_config_path)
    mft_config_path = verify_file_identity(
        root,
        remediation["inputs"]["mft_config"],
        label="remediation.inputs.mft_config",
    )
    g1_config_path = verify_file_identity(
        root,
        remediation["inputs"]["g1_config"],
        label="remediation.inputs.g1_config",
    )
    mft = _load_yaml(mft_config_path)
    g1 = _load_yaml(g1_config_path)

    expected: dict[str, dict[str, Any]] = {}
    _add_mapping_contracts(
        expected,
        remediation["inputs"],
        source_prefix="remediation.inputs",
    )
    _add_mapping_contracts(expected, mft["inputs"], source_prefix="mft.inputs")
    _add_mapping_contracts(expected, g1["inputs"], source_prefix="g1.inputs")

    collection_path = verify_file_identity(
        root,
        parents["collection_receipt"],
        label="immutable_parent.collection_receipt",
    )
    collection = _load_json(collection_path)
    verify_payload_sha256(
        collection,
        field="receipt_payload_sha256",
        label="官方来源采集收据",
    )
    _add_mapping_contracts(
        expected,
        {
            "official_intervals": collection["official_intervals"],
            "raw_file_index": collection["raw_file_index"],
        },
        source_prefix="collection_receipt",
    )
    raw_index = pd.read_parquet(
        project_path(root, str(collection["raw_file_index"]["path"]))
    )
    required_raw_columns = {"path", "bytes", "sha256"}
    if not required_raw_columns.issubset(raw_index.columns):
        raise VersionLockError("官方原始响应索引缺少路径、字节数或 SHA-256")
    if raw_index["path"].duplicated().any():
        raise VersionLockError("官方原始响应索引含重复路径")
    for row in raw_index.itertuples(index=False):
        merge_expected_contract(
            expected,
            {
                "path": str(row.path),
                "bytes": int(row.bytes),
                "sha256": str(row.sha256),
            },
            source="collection_receipt.raw_file_index",
        )

    build_path = verify_file_identity(
        root,
        parents["remediation_build_receipt"],
        label="immutable_parent.remediation_build_receipt",
    )
    build = _load_json(build_path)
    verify_payload_sha256(
        build,
        field="receipt_payload_sha256",
        label="修复构建收据",
    )
    _add_mapping_contracts(
        expected,
        {
            name: contract
            for name, contract in build["outputs"].items()
            if str(contract.get("path", "")).casefold().endswith(".parquet")
        },
        source_prefix="remediation_build_receipt.outputs",
    )

    for index, contract in enumerate(config.get("additional_data_files", [])):
        merge_expected_contract(
            expected,
            contract,
            source=f"additional_data_files[{index}]",
        )

    data_files: list[dict[str, Any]] = []
    for relative in sorted(expected):
        if relative in git_scope_relatives:
            continue
        contract = expected[relative]
        path = verify_file_identity(root, contract, label=f"数据清单 {relative}")
        if path.suffix.casefold() == ".parquet":
            evidence = parquet_evidence(path, root=root)
            upstream_semantic = contract.get("semantic_sha256")
            if upstream_semantic is not None and evidence["persisted_semantic_sha256"] != str(
                upstream_semantic
            ):
                raise VersionLockError(f"上游构建收据的语义摘要不一致：{relative}")
        else:
            evidence = file_evidence(path, root=root)
            evidence["format"] = path.suffix.lstrip(".").upper() or "BINARY"
        evidence["contract_sources"] = sorted(contract["contract_sources"])
        data_files.append(evidence)
    return data_files


def _load_frozen_manifest(
    root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = project_path(root, str(config["outputs"]["manifest"]))
    manifest = _load_json(manifest_path)
    verify_payload_sha256(
        manifest,
        field="manifest_payload_sha256",
        label="G0.1 冻结 manifest",
    )
    config_path = project_path(root, str(manifest["config_file"]))
    if sha256_file(config_path) != str(manifest["config_sha256"]):
        raise VersionLockError("G0.1 冻结后配置文件发生漂移")
    for contract in manifest["required_git_files"]:
        verify_file_identity(root, contract, label=f"冻结 Git 文件 {contract['path']}")
    _verify_immutable_parents(root, config)
    return manifest


def freeze(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """冻结修复后 Git 范围与全部可重放数据身份。"""

    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    if str(config["program"]["execution_id"]) != EXECUTION_ID:
        raise VersionLockError("G0.1 execution_id 漂移")
    if bool(config["authorization"]["formal_g2_allowed"]):
        raise VersionLockError("G0.1 不得授权正式 G2")
    if bool(config["git_contract"]["remote_push_required"]):
        raise VersionLockError("任务卡明确不要求远程推送")
    if not bool(config["git_contract"]["local_commit_required"]):
        raise VersionLockError("任务卡要求本地 Git 提交")
    current_branch = git_branch(ROOT)
    required_branch = str(config["git_contract"]["branch_required"])
    if current_branch != required_branch:
        raise VersionLockError(
            f"G0.1 必须在冻结分支执行：期望 {required_branch}，实际 {current_branch}"
        )
    manifest_path = project_path(ROOT, str(config["outputs"]["manifest"]))
    receipt_path = project_path(ROOT, str(config["outputs"]["freeze_receipt"]))
    if manifest_path.exists() or receipt_path.exists():
        raise VersionLockError("G0.1 manifest 或冻结收据已存在，禁止覆盖")
    _verify_immutable_parents(ROOT, config)

    git_paths = _git_scope(ROOT, config)
    git_evidence = [file_evidence(path, root=ROOT) for path in git_paths]
    git_relatives = {str(item["path"]) for item in git_evidence}
    data_files = _collect_data_contracts(
        ROOT,
        config,
        git_scope_relatives=git_relatives,
    )
    parquet_files = [item for item in data_files if item.get("format") == "PARQUET"]
    manifest: dict[str, Any] = {
        "program_id": PROGRAM_ID,
        "execution_id": EXECUTION_ID,
        "version": str(config["program"]["version"]),
        "frozen_at": _now(),
        "status": "FROZEN_POST_REMEDIATION_GIT_AND_DATA_VERSION_LOCK",
        "config_file": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "pre_lock_head_sha": git_head(ROOT),
        "pre_lock_branch": current_branch,
        "required_git_files": git_evidence,
        "data_files": data_files,
        "data_summary": {
            "file_count": len(data_files),
            "total_bytes": int(sum(int(item["bytes"]) for item in data_files)),
            "parquet_file_count": len(parquet_files),
            "parquet_total_rows": int(
                sum(int(item["row_count"]) for item in parquet_files)
            ),
        },
        "expected_replay": config["expected_replay"],
        "git_contract": config["git_contract"],
        "authorization": config["authorization"],
        "feature_or_label_value_used_to_select_git_or_data_scope": False,
        "model_trained": False,
        "formal_g2_run": False,
        "portfolio_or_return_read": False,
        "position_impact": 0,
    }
    manifest["manifest_payload_sha256"] = canonical_sha256(manifest)
    atomic_write_json_new(manifest_path, manifest)
    receipt: dict[str, Any] = {
        "receipt_id": f"{EXECUTION_ID}_FREEZE_RECEIPT",
        "created_at": _now(),
        "status": "PASS_G0_1_SCOPE_AND_DATA_IDENTITIES_FROZEN_BEFORE_LOCAL_COMMIT",
        "manifest": file_evidence(manifest_path, root=ROOT),
        "pre_lock_head_sha": manifest["pre_lock_head_sha"],
        "pre_lock_branch": manifest["pre_lock_branch"],
        "required_git_file_count": len(git_evidence),
        "data_file_count": len(data_files),
        "model_trained": False,
        "formal_g2_run": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(receipt_path, receipt)
    print(
        f"G0.1 已冻结：Git 文件 {len(git_evidence)} 个，数据文件 {len(data_files)} 个；"
        "尚未声称本地提交或 clean worktree 重放通过。",
        flush=True,
    )
    return receipt


def materialize(
    *,
    config_path: Path,
    source_root: Path,
    target_root: Path,
) -> dict[str, int]:
    """按冻结 manifest 向独立 worktree 物化重放数据。"""

    config = _load_yaml(config_path.resolve())
    manifest = _load_frozen_manifest(source_root.resolve(), config)
    result = materialize_data_files(
        source_root=source_root,
        target_root=target_root,
        manifest=manifest,
    )
    print(
        "数据物化完成："
        f"复制 {result['copied_file_count']} 个，复用 {result['reused_file_count']} 个，"
        f"复制字节 {result['copied_bytes']}。",
        flush=True,
    )
    return result


def _verify_clean_commit_scope(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected_commit: str,
) -> None:
    verify_tracked_worktree_clean(root)
    verify_commit_contains_files(
        root,
        commit_sha=expected_commit,
        contracts=manifest["required_git_files"],
    )
    generated_paths = [
        str(config["outputs"]["manifest"]),
        str(config["outputs"]["freeze_receipt"]),
    ]
    generated_contracts = [
        file_evidence(project_path(root, relative), root=root)
        for relative in generated_paths
    ]
    verify_commit_contains_files(
        root,
        commit_sha=expected_commit,
        contracts=generated_contracts,
    )


def _compare_replay(
    root: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from scripts.run_510300_stress_transmission_hazard_v2_g1_historical_remediation_v1 import (
        _output_frames,
        _report_text,
        _research_frames,
    )

    build_path = project_path(
        root,
        str(config["immutable_parent"]["remediation_build_receipt"]["path"]),
    )
    build = _load_json(build_path)
    verify_payload_sha256(
        build,
        field="receipt_payload_sha256",
        label="修复构建收据",
    )
    remediation_config = _load_yaml(
        project_path(
            root,
            str(config["immutable_parent"]["remediation_config"]["path"]),
        )
    )
    remediation, coverage, mft, g1, metrics = _research_frames(remediation_config)
    frames = _output_frames(remediation, coverage, mft, g1)
    comparisons: dict[str, Any] = {}
    for name, rebuilt in frames.items():
        expected = build["outputs"][name]
        path = verify_file_identity(root, expected, label=f"构建输出 {name}")
        persisted = pd.read_parquet(path)
        actual_semantic = parquet_evidence(path, root=root)[
            "persisted_semantic_sha256"
        ]
        if actual_semantic != str(expected["semantic_sha256"]):
            raise VersionLockError(f"构建输出持久化语义漂移：{name}")
        try:
            pd.testing.assert_frame_equal(
                persisted.reset_index(drop=True),
                rebuilt.reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
                check_categorical=False,
            )
        except AssertionError as exc:
            raise VersionLockError(f"commit worktree 重放逐值不一致：{name}：{exc}") from exc
        comparisons[name] = {
            "persisted_file_identity_passed": True,
            "persisted_semantic_sha256": actual_semantic,
            "expected_semantic_sha256": str(expected["semantic_sha256"]),
            "exact_values_equal_ignoring_storage_dtype_representation": True,
        }
    if canonical_sha256(metrics) != canonical_sha256(build["metrics"]):
        raise VersionLockError("commit worktree 重放指标与构建收据不一致")
    if canonical_sha256(g1.gate_result) != canonical_sha256(build["g1_gate_result"]):
        raise VersionLockError("commit worktree 重放 G1 门结果与构建收据不一致")
    report = build["outputs"]["report"]
    report_path = verify_file_identity(root, report, label="G1 修复报告")
    if report_path.read_text(encoding="utf-8") != _report_text(metrics, g1.gate_result):
        raise VersionLockError("G1 修复报告与 commit worktree 重放内容不一致")
    validate_expected_replay_metrics(metrics, config["expected_replay"])
    return comparisons, metrics, g1.gate_result


def clean_replay(
    *,
    config_path: Path,
    expected_commit: str,
) -> dict[str, Any]:
    """在指定 commit 的独立 worktree 中全量重算并追加收据。"""

    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    manifest = _load_frozen_manifest(ROOT, config)
    if git_head(ROOT) != expected_commit:
        raise VersionLockError("clean-replay 的 HEAD 与 --expected-commit 不一致")
    _verify_clean_commit_scope(
        ROOT,
        config,
        manifest,
        expected_commit=expected_commit,
    )
    verify_data_manifest(ROOT, manifest)
    replay_path = project_path(ROOT, str(config["outputs"]["clean_replay_receipt"]))
    if replay_path.exists():
        raise VersionLockError(f"G0.1 clean replay 收据已存在：{replay_path}")
    comparisons, metrics, gate = _compare_replay(ROOT, config)
    corrected_status_path = project_path(
        ROOT,
        str(config["immutable_parent"]["corrected_g1_status"]["path"]),
    )
    corrected_status = _load_json(corrected_status_path)
    expected_state = str(config["expected_replay"]["g1_state"])
    if str(corrected_status.get("G1_DATA_AND_EVENTS")) != expected_state:
        raise VersionLockError("V1.0.1 最终 G1 状态与 G0.1 冻结预期不一致")
    receipt: dict[str, Any] = {
        "receipt_id": f"{EXECUTION_ID}_CLEAN_WORKTREE_REPLAY_RECEIPT",
        "created_at": _now(),
        "status": "PASS_COMMIT_PINNED_CLEAN_WORKTREE_FULL_G1_REPLAY",
        "locked_commit_sha": expected_commit,
        "worktree_root": str(ROOT),
        "tracked_worktree_clean_before_replay": True,
        "manifest": file_evidence(
            project_path(ROOT, str(config["outputs"]["manifest"])),
            root=ROOT,
        ),
        "data_file_count": int(manifest["data_summary"]["file_count"]),
        "data_total_bytes": int(manifest["data_summary"]["total_bytes"]),
        "all_data_identities_and_parquet_semantics_passed": True,
        "comparisons": comparisons,
        "metrics": metrics,
        "g1_gate_result": gate,
        "expected_replay_matched": True,
        "model_trained": False,
        "formal_g2_run": False,
        "portfolio_or_return_read": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(replay_path, receipt)
    print(
        "指定 commit 的独立 worktree 全量重放通过："
        f"B2={metrics['g1']['b2_identifiable_event_count']}，"
        f"非事件风险日={metrics['g1']['b2_eligible_non_event_risk_day_count']}，"
        f"共同日={metrics['g1']['b2_common_sample_day_count']}；G2 未运行。",
        flush=True,
    )
    return receipt


def finalize(
    *,
    config_path: Path,
    receipt_source: Path,
    expected_commit: str,
) -> dict[str, Any]:
    """将独立 worktree 收据追加回主工作树并生成 G0.1 状态。"""

    config_path = config_path.resolve()
    config = _load_yaml(config_path)
    manifest = _load_frozen_manifest(ROOT, config)
    result = run_git(
        ROOT,
        ["merge-base", "--is-ancestor", expected_commit, "HEAD"],
        check=False,
    )
    if result.returncode != 0:
        raise VersionLockError("当前分支不包含指定的版本锁定 commit")
    receipt_source = receipt_source.resolve()
    if not receipt_source.is_file():
        raise VersionLockError(f"独立 worktree 收据不存在：{receipt_source}")
    source_receipt = _load_json(receipt_source)
    verify_payload_sha256(
        source_receipt,
        field="receipt_payload_sha256",
        label="独立 worktree G0.1 重放收据",
    )
    _verify_required_fields(
        source_receipt,
        {
            "status": "PASS_COMMIT_PINNED_CLEAN_WORKTREE_FULL_G1_REPLAY",
            "locked_commit_sha": expected_commit,
            "expected_replay_matched": True,
            "model_trained": False,
            "formal_g2_run": False,
            "position_impact": 0,
        },
        label="独立 worktree G0.1 重放收据",
    )
    validate_expected_replay_metrics(
        source_receipt["metrics"],
        config["expected_replay"],
    )
    destination = project_path(
        ROOT,
        str(config["outputs"]["clean_replay_receipt"]),
    )
    _atomic_copy_new(receipt_source, destination)
    if sha256_file(destination) != sha256_file(receipt_source):
        raise VersionLockError("独立 worktree 收据复制后身份不一致")

    status_path = project_path(ROOT, str(config["outputs"]["status"]))
    prior_g0 = _load_json(
        project_path(ROOT, str(config["immutable_parent"]["prior_g0_status"]["path"]))
    )
    corrected = _load_json(
        project_path(ROOT, str(config["immutable_parent"]["corrected_g1_status"]["path"]))
    )
    status: dict[str, Any] = {
        "program_id": PROGRAM_ID,
        "execution_id": EXECUTION_ID,
        "updated_at": _now(),
        "PRIOR_G0": str(prior_g0["G0_ENGINEERING_AND_CONTRACT"]),
        "G0_1_POST_REMEDIATION_VERSION_LOCK": "PASS",
        "REMOTE_PUSH_REQUIRED": False,
        "LOCAL_COMMIT_REQUIRED": True,
        "locked_commit_sha": expected_commit,
        "locked_branch": str(manifest["pre_lock_branch"]),
        "clean_worktree_replay": file_evidence(destination, root=ROOT),
        "G1A_COMMON_SAMPLE_EVENT_IDENTIFIABILITY": "PASS_B2_ONLY",
        "g1a_b2_identifiable_event_count": int(
            corrected["b2_identifiable_event_count"]
        ),
        "g1a_b2_non_event_risk_day_count": int(
            corrected["b2_eligible_non_event_risk_day_count"]
        ),
        "g1a_b2_common_sample_day_count": int(
            source_receipt["metrics"]["g1"]["b2_common_sample_day_count"]
        ),
        "G1B_PREQUENTIAL_ERA_IDENTIFIABILITY": "NOT_RUN_PENDING_SEPARATE_FREEZE",
        "G2_STRUCTURAL_INCREMENT": "NOT_RUN_PENDING_G1B",
        "B3_FULL_MODEL": "NO_VIEW_31_LT_40",
        "model_trained": False,
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_evaluation": "NOT_ALLOWED",
        "position_impact": 0,
        "next_allowed_step": "FREEZE_G1B_PREQUENTIAL_ERA_IDENTIFIABILITY_BEFORE_ALLOWED_FIELD_READ",
    }
    status["status_payload_sha256"] = canonical_sha256(status)
    atomic_write_json_new(status_path, status)
    print(
        f"G0.1 完成：锁定 commit={expected_commit}；"
        "下一步仅允许冻结 G1B，不允许正式 G2。",
        flush=True,
    )
    return status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="G0.1 冻结配置路径",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze", help="冻结 Git 范围和数据身份")
    materialize_parser = subparsers.add_parser(
        "materialize",
        help="只增不改地向独立 worktree 物化数据",
    )
    materialize_parser.add_argument("--source-root", type=Path, required=True)
    materialize_parser.add_argument("--target-root", type=Path, required=True)
    replay_parser = subparsers.add_parser(
        "clean-replay",
        help="在 commit 固定的独立 worktree 中全量重放 G1",
    )
    replay_parser.add_argument("--expected-commit", required=True)
    finalize_parser = subparsers.add_parser(
        "finalize",
        help="追加导入 clean worktree 收据并生成 G0.1 状态",
    )
    finalize_parser.add_argument("--receipt-source", type=Path, required=True)
    finalize_parser.add_argument("--expected-commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            freeze(args.config)
        elif args.command == "materialize":
            materialize(
                config_path=args.config,
                source_root=args.source_root,
                target_root=args.target_root,
            )
        elif args.command == "clean-replay":
            clean_replay(
                config_path=args.config,
                expected_commit=str(args.expected_commit),
            )
        elif args.command == "finalize":
            finalize(
                config_path=args.config,
                receipt_source=args.receipt_source,
                expected_commit=str(args.expected_commit),
            )
        else:
            raise VersionLockError(f"未知命令：{args.command}")
    except (VersionLockError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"G0.1 执行失败：{exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
