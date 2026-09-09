"""在全新进程重放 V2 G1 事件预测准入，并写入正式 G1 状态。"""

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
from research.stress_transmission_hazard_v2_g1_event_prediction_admission_v1 import (
    G1EventPredictionAdmissionError,
)
from scripts.build_510300_stress_transmission_hazard_v2_g1_event_prediction_admission_v1 import (
    TABLE_OUTPUTS,
    _build_report,
    frame_semantic_sha256,
    load_and_build,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_g1_event_prediction_admission_v1 import (
    DEFAULT_CONFIG,
    load_config,
    verify_frozen_manifest,
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def _verify_receipt_payload(receipt: Mapping[str, Any]) -> None:
    expected = str(receipt.get("receipt_payload_sha256", ""))
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256", None)
    if canonical_sha256(payload) != expected:
        raise EvidenceContractError("G1 构建收据自身摘要失败")


def _verify_evidence(evidence: Mapping[str, Any], label: str) -> Path:
    path = _project_path(str(evidence["path"]))
    actual = file_evidence(path, project_root=ROOT)
    if actual != dict(evidence):
        raise EvidenceContractError(f"{label}文件身份漂移：{path}")
    return path


def _run_targeted_tests(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = [
        str(_project_path(str(path)))
        for path in config["clean_replay_contract"]["targeted_test_files"]
    ]
    parent = ROOT / "artifacts" / "tmp"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%f")
    base_temp = parent / f"g1_admission_v1_pytest_{stamp}"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *paths,
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
            f"G1 定向测试失败，exit={completed.returncode}：\n{output[-5000:]}"
        )
    match = re.search(r"(?P<count>\d+) passed", output)
    if match is None:
        raise EvidenceContractError("G1 测试输出没有 passed 计数")
    return {
        "status": "PASS",
        "exit_code": completed.returncode,
        "passed_test_count": int(match.group("count")),
        "test_file_count": len(paths),
        "test_files": [Path(path).relative_to(ROOT).as_posix() for path in paths],
        "stdout_stderr_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "summary_tail": output.strip()[-1200:],
        "base_temp": base_temp.relative_to(ROOT).as_posix(),
    }


def replay(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    verify_frozen_manifest(config_path)
    config = load_config(config_path)
    outputs = {
        name: _project_path(str(path))
        for name, path in config["outputs"].items()
        if name != "curated_root"
    }
    for required in [*TABLE_OUTPUTS.keys(), "report", "build_receipt"]:
        if not outputs[required].is_file():
            raise EvidenceContractError(f"G1 重放前缺少构建输出：{outputs[required]}")
    forbidden_existing = [
        str(outputs[name])
        for name in ("clean_replay_receipt", "clean_replay_failure_receipt", "status")
        if outputs[name].exists()
    ]
    if forbidden_existing:
        raise EvidenceContractError(f"G1 重放终态已存在，禁止覆盖：{forbidden_existing}")

    build_receipt = read_json_strict(outputs["build_receipt"])
    if not isinstance(build_receipt, dict):
        raise EvidenceContractError("G1 构建收据必须是对象")
    _verify_receipt_payload(build_receipt)
    if build_receipt.get("status") != (
        "PASS_G1_COMMON_SAMPLE_AND_IDENTIFIABILITY_COUNTS_PENDING_CLEAN_REPLAY"
    ):
        raise EvidenceContractError("G1 构建收据状态不允许重放")
    if build_receipt.get("actual_label_artifact_read") is not True:
        raise EvidenceContractError("G1 构建收据未声明实际标签读取")
    for forbidden_flag in (
        "actual_future_path_value_read",
        "minimum_path_return_read",
        "model_trained",
        "probability_generated",
        "prediction_metric_generated",
        "portfolio_or_performance_artifact_read",
        "portfolio_generated",
    ):
        if build_receipt.get(forbidden_flag) is not False:
            raise EvidenceContractError(f"G1 构建收据禁止标志异常：{forbidden_flag}")

    print("开始在全新进程重建 G1 共同样本和事件可识别性。", flush=True)
    rebuilt = load_and_build(config)
    comparisons: dict[str, Any] = {}
    for output_name, attribute_name in TABLE_OUTPUTS.items():
        evidence = build_receipt["outputs"][output_name]
        stored_path = _verify_evidence(evidence, f"G1 输出 {output_name}")
        stored = pd.read_parquet(stored_path)
        current = getattr(rebuilt, attribute_name)
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
                f"G1 重放逐值不一致 {output_name}：{exc}"
            ) from exc
        semantic = frame_semantic_sha256(stored)
        if semantic != str(evidence["persisted_semantic_sha256"]):
            raise EvidenceContractError(f"G1 持久化语义摘要漂移：{output_name}")
        comparisons[output_name] = {
            "row_count": int(len(stored)),
            "persisted_semantic_sha256": semantic,
            "frame_equal_ignoring_storage_dtype_representation": True,
        }
    expected_report = _build_report(rebuilt)
    actual_report = outputs["report"].read_text(encoding="utf-8")
    if actual_report != expected_report:
        raise EvidenceContractError("G1 人类可读报告重放不一致")
    if canonical_sha256(rebuilt.metrics) != canonical_sha256(build_receipt["metrics"]):
        raise EvidenceContractError("G1 重放计数指标不一致")
    if canonical_sha256(rebuilt.gate_result) != canonical_sha256(
        build_receipt["gate_result"]
    ):
        raise EvidenceContractError("G1 重放门禁裁决不一致")

    print("G1 三张表逐值一致，开始运行四个冻结范围测试文件。", flush=True)
    tests = _run_targeted_tests(config)
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    replay_receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_G1_EVENT_PREDICTION_ADMISSION_CLEAN_REPLAY_V1",
        "created_at": now,
        "status": "PASS_FRESH_PROCESS_G1_COMMON_SAMPLE_AND_IDENTIFIABILITY_REPLAY",
        "frozen_manifest": file_evidence(
            _project_path(config["freeze_contract"]["manifest_output"]),
            project_root=ROOT,
        ),
        "build_receipt": file_evidence(outputs["build_receipt"], project_root=ROOT),
        "table_comparisons": comparisons,
        "metrics": rebuilt.metrics,
        "gate_result": rebuilt.gate_result,
        "targeted_tests": tests,
        "actual_label_artifact_read": True,
        "actual_future_path_value_read": False,
        "model_trained": False,
        "probability_generated": False,
        "prediction_metric_generated": False,
        "portfolio_or_performance_artifact_read": False,
        "position_impact": 0,
    }
    replay_receipt["receipt_payload_sha256"] = canonical_sha256(replay_receipt)
    atomic_write_json_new(outputs["clean_replay_receipt"], replay_receipt)

    gate = rebuilt.gate_result
    status: dict[str, Any] = {
        "program_id": config["program"]["program_id"],
        "execution_id": config["program"]["execution_id"],
        "updated_at": now,
        "G0_ENGINEERING_AND_CONTRACT": "PASS",
        "G1_DATA_AND_EVENTS": gate["G1_DATA_AND_EVENTS"],
        "G1_MECHANISM_DISCOVERY_PREREQUISITE": (
            "PASS" if gate["mechanism_discovery_prerequisite_passed"] else "FAIL"
        ),
        "G1_FULL_THREE_COEFFICIENT_MODEL_PREREQUISITE": (
            "PASS"
            if gate["full_three_coefficient_model_prerequisite_passed"]
            else "FAIL"
        ),
        "G2_STRUCTURAL_INCREMENT": "NOT_RUN",
        "G3_MACRO_INCREMENT": "NOT_RUN",
        "G4_THROUGH_G7": "NOT_RUN",
        "metrics": rebuilt.metrics,
        "next_allowed_step": gate["next_allowed_step"],
        "additional_step_if_full_model_passed": gate[
            "additional_step_if_full_model_passed"
        ],
        "clean_replay_receipt": file_evidence(
            outputs["clean_replay_receipt"], project_root=ROOT
        ),
        "actual_label_artifact_read": True,
        "actual_future_path_value_read": False,
        "model_trained": False,
        "probability_generated": False,
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_evaluation": "NOT_ALLOWED_BEFORE_G0_THROUGH_G4_ALL_PASS",
        "paper_or_shadow": "NOT_AUTHORIZED",
        "broker_connection": "NOT_AUTHORIZED",
        "position_impact": 0,
    }
    status["status_payload_sha256"] = canonical_sha256(status)
    atomic_write_json_new(outputs["status"], status)
    print(
        "G1 准入重放通过；"
        f"G1={gate['G1_DATA_AND_EVENTS']}；"
        "G2-G7 未运行，收益与交易权限保持关闭。"
    )
    return status


def _write_failure_receipt(config_path: Path, exc: BaseException) -> None:
    try:
        config = load_config(config_path)
        path = _project_path(config["outputs"]["clean_replay_failure_receipt"])
        if path.exists():
            return
        payload = {
            "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_G1_EVENT_PREDICTION_ADMISSION_CLEAN_REPLAY_V1_FAILURE",
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "status": "FAIL_G1_CLEAN_REPLAY_NO_GATE_PROMOTION",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "actual_future_path_value_read": False,
            "model_trained": False,
            "probability_generated": False,
            "portfolio_or_performance_artifact_read": False,
            "position_impact": 0,
        }
        payload["receipt_payload_sha256"] = canonical_sha256(payload)
        atomic_write_json_new(path, payload)
    except Exception as failure_receipt_error:
        print(f"G1 失败收据写入也失败：{failure_receipt_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        replay(args.config)
        return 0
    except (
        EvidenceContractError,
        G1EventPredictionAdmissionError,
        AssertionError,
        KeyError,
        TypeError,
        ValueError,
        OSError,
    ) as exc:
        _write_failure_receipt(args.config, exc)
        print(f"G1 事件预测准入重放失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
