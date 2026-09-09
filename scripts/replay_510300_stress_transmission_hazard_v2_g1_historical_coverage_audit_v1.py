"""全新进程重放 V2 G1 历史覆盖审计 V1。"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

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
)
from research.stress_transmission_hazard_v2_g1_historical_coverage_audit_v1 import (
    HistoricalCoverageAuditError,
)
from scripts.build_510300_stress_transmission_hazard_v2_g1_historical_coverage_audit_v1 import (
    TABLE_OUTPUTS,
    _build_report,
    frame_semantic_sha256,
    load_and_build,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_g1_historical_coverage_audit_v1 import (
    DEFAULT_CONFIG,
    load_config,
    verify_frozen_manifest,
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _verify_self_digest(payload: Mapping[str, Any], field: str, label: str) -> None:
    expected = str(payload.get(field, ""))
    body = dict(payload)
    body.pop(field, None)
    if canonical_sha256(body) != expected:
        raise EvidenceContractError(f"{label}自身摘要失败")


def _verify_file_identity(metadata: Mapping[str, Any], label: str) -> Path:
    expected = {key: metadata[key] for key in ("path", "bytes", "sha256")}
    path = _project_path(str(expected["path"]))
    if not path.is_file():
        raise EvidenceContractError(f"{label}不存在：{path}")
    actual = file_evidence(path, project_root=ROOT)
    normalized = {
        "path": normalize_project_relative_path(str(expected["path"])).replace("\\", "/"),
        "bytes": int(expected["bytes"]),
        "sha256": str(expected["sha256"]),
    }
    if actual != normalized:
        raise EvidenceContractError(f"{label}文件身份漂移：{path}")
    return path


def _run_targeted_tests(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = [_project_path(str(path)) for path in config["targeted_test_files"]]
    parent = ROOT / "artifacts" / "tmp"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%f")
    base_temp = parent / f"g1_historical_coverage_audit_v1_pytest_{stamp}"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *[str(path) for path in paths],
        "--basetemp",
        str(base_temp),
    ]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        raise EvidenceContractError(
            f"历史覆盖审计定向测试失败，exit={completed.returncode}：\n{output[-5000:]}"
        )
    match = re.search(r"(?P<count>\d+) passed", output)
    if match is None:
        raise EvidenceContractError("历史覆盖审计测试输出没有 passed 计数")
    return {
        "status": "PASS",
        "exit_code": completed.returncode,
        "passed_test_count": int(match.group("count")),
        "test_file_count": len(paths),
        "test_files": [path.relative_to(ROOT).as_posix() for path in paths],
        "stdout_stderr_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "summary_tail": output.strip()[-1200:],
        "base_temp": base_temp.relative_to(ROOT).as_posix(),
    }


def replay(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = load_config(config_path)
    verify_frozen_manifest(config_path)
    outputs = {
        name: _project_path(str(path)) for name, path in config["outputs"].items()
    }
    terminal_names = ["clean_replay_receipt", "clean_replay_failure_receipt", "status"]
    existing_terminal = [str(outputs[name]) for name in terminal_names if outputs[name].exists()]
    if existing_terminal:
        raise EvidenceContractError(
            f"历史覆盖审计终态输出已存在，禁止覆盖：{existing_terminal}"
        )
    build_receipt = read_json_strict(outputs["build_receipt"])
    if not isinstance(build_receipt, dict):
        raise EvidenceContractError("历史覆盖审计构建收据必须是对象")
    _verify_self_digest(build_receipt, "receipt_payload_sha256", "历史覆盖审计构建收据")
    if build_receipt.get("status") != "PASS_HISTORICAL_COVERAGE_AUDIT_PENDING_CLEAN_REPLAY_G1_UNCHANGED":
        raise EvidenceContractError("历史覆盖审计构建状态不允许重放")

    print("全新进程正在重建历史覆盖诊断，父 G1 保持停止。", flush=True)
    rebuilt = load_and_build(config)
    comparisons: dict[str, Any] = {}
    for output_name, attribute_name in TABLE_OUTPUTS.items():
        metadata = build_receipt["outputs"][output_name]
        stored_path = _verify_file_identity(metadata, f"历史覆盖审计输出 {output_name}")
        stored = pd.read_parquet(stored_path)
        current = getattr(rebuilt, attribute_name)
        if len(stored) != int(metadata["row_count"]):
            raise EvidenceContractError(f"历史覆盖审计输出行数漂移：{output_name}")
        semantic = frame_semantic_sha256(stored)
        if semantic != str(metadata["persisted_semantic_sha256"]):
            raise EvidenceContractError(f"历史覆盖审计输出语义摘要漂移：{output_name}")
        try:
            pd.testing.assert_frame_equal(
                stored.reset_index(drop=True),
                current.reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
                check_categorical=False,
            )
        except AssertionError as exc:
            raise EvidenceContractError(
                f"历史覆盖审计逐值重放不一致 {output_name}：{exc}"
            ) from exc
        comparisons[output_name] = {
            "file_identity_fields": {
                key: metadata[key] for key in ("path", "bytes", "sha256")
            },
            "row_count": int(len(stored)),
            "persisted_semantic_sha256": semantic,
            "frame_equal_ignoring_storage_dtype_representation": True,
        }

    report_path = _verify_file_identity(
        build_receipt["outputs"]["report"], "历史覆盖审计报告"
    )
    if report_path.read_text(encoding="utf-8") != _build_report(rebuilt):
        raise EvidenceContractError("历史覆盖审计报告重放不一致")
    if canonical_sha256(rebuilt.metrics) != canonical_sha256(build_receipt["metrics"]):
        raise EvidenceContractError("历史覆盖审计指标重放不一致")
    if canonical_sha256(rebuilt.audit_state) != canonical_sha256(
        build_receipt["audit_state"]
    ):
        raise EvidenceContractError("历史覆盖审计状态重放不一致")
    if rebuilt.audit_state["G1_DATA_AND_EVENTS"] != "NO_VIEW_INSUFFICIENT_EVENT_IDENTIFIABILITY":
        raise EvidenceContractError("重放意外提升父 G1")
    if rebuilt.metrics["g1_gate_promoted"] is not False:
        raise EvidenceContractError("诊断指标意外提升父 G1")

    print("三张诊断表逐值一致，开始运行冻结的定向测试。", flush=True)
    tests = _run_targeted_tests(config)
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_G1_HISTORICAL_COVERAGE_AUDIT_CLEAN_REPLAY_V1",
        "created_at": now,
        "status": "PASS_FRESH_PROCESS_HISTORICAL_COVERAGE_AUDIT_G1_UNCHANGED",
        "table_comparisons": comparisons,
        "metrics": rebuilt.metrics,
        "audit_state": rebuilt.audit_state,
        "targeted_tests": tests,
        "actual_label_artifact_read": True,
        "actual_future_path_value_read": False,
        "observation_after_historical_cutoff_read": False,
        "source_state_reclassified": False,
        "missing_return_zero_filled": False,
        "g1_gate_promoted": False,
        "model_trained": False,
        "probability_generated": False,
        "prediction_metric_generated": False,
        "portfolio_or_performance_artifact_read": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(outputs["clean_replay_receipt"], receipt)

    status: dict[str, Any] = {
        "program_id": config["program"]["program_id"],
        "execution_id": config["program"]["execution_id"],
        "updated_at": now,
        **rebuilt.audit_state,
        "metrics": rebuilt.metrics,
        "clean_replay_receipt": file_evidence(
            outputs["clean_replay_receipt"], project_root=ROOT
        ),
        "parent_g1_status": "RETAINED_IMMUTABLE",
        "diagnostic_authority": "NON_ADMISSIBLE_DIAGNOSTIC_ONLY",
        "actual_label_artifact_read": True,
        "actual_future_path_value_read": False,
        "observation_after_historical_cutoff_read": False,
        "source_state_reclassified": False,
        "missing_return_zero_filled": False,
        "g1_gate_promoted": False,
        "model_trained": False,
        "probability_generated": False,
        "position_impact": 0,
    }
    status["status_payload_sha256"] = canonical_sha256(status)
    atomic_write_json_new(outputs["status"], status)
    print(
        "历史覆盖审计全新进程重放通过；父 G1 仍为 NO_VIEW，"
        "只允许进入另行冻结的官方历史源探针与索引修正评审。"
    )
    return status


def _write_failure_receipt(config_path: Path, exc: BaseException) -> None:
    try:
        config = load_config(config_path)
        path = _project_path(config["outputs"]["clean_replay_failure_receipt"])
        if path.exists():
            return
        payload = {
            "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_G1_HISTORICAL_COVERAGE_AUDIT_CLEAN_REPLAY_V1_FAILURE",
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "status": "FAIL_HISTORICAL_COVERAGE_AUDIT_NO_GATE_PROMOTION",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "actual_future_path_value_read": False,
            "observation_after_historical_cutoff_read": False,
            "g1_gate_promoted": False,
            "model_trained": False,
            "portfolio_or_performance_artifact_read": False,
            "position_impact": 0,
        }
        payload["receipt_payload_sha256"] = canonical_sha256(payload)
        atomic_write_json_new(path, payload)
    except Exception as receipt_error:
        print(f"历史覆盖审计失败收据写入也失败：{receipt_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        replay(args.config)
        return 0
    except (
        EvidenceContractError,
        HistoricalCoverageAuditError,
        AssertionError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as exc:
        _write_failure_receipt(args.config, exc)
        print(f"历史覆盖审计重放失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
