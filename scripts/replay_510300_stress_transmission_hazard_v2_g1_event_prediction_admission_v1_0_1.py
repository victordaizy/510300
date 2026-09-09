"""执行 G1 准入 V1.0.1：修正文件身份子集比较后重放。"""

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
    DEFAULT_CONFIG as ORIGINAL_CONFIG,
    load_config as load_original_config,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_g1_event_prediction_admission_v1_0_1 import (
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


def verify_file_identity_subset(
    metadata: Mapping[str, Any], *, label: str
) -> Path:
    """只把 path/bytes/sha256 当作文件身份；其余元数据由调用方单独核验。"""

    required = ("path", "bytes", "sha256")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise EvidenceContractError(f"{label}缺少文件身份字段：{missing}")
    expected = {key: metadata[key] for key in required}
    path = _project_path(str(expected["path"]))
    if not path.is_file():
        raise EvidenceContractError(f"{label}不存在：{path}")
    actual = file_evidence(path, project_root=ROOT)
    expected_normalized = {
        "path": normalize_project_relative_path(str(expected["path"])).replace(
            "\\", "/"
        ),
        "bytes": int(expected["bytes"]),
        "sha256": str(expected["sha256"]),
    }
    if actual != expected_normalized:
        raise EvidenceContractError(f"{label}文件身份漂移：{path}")
    return path


def _run_targeted_tests(correction: Mapping[str, Any]) -> dict[str, Any]:
    paths = [str(_project_path(str(path))) for path in correction["targeted_test_files"]]
    parent = ROOT / "artifacts" / "tmp"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%f")
    base_temp = parent / f"g1_admission_v1_0_1_pytest_{stamp}"
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
            f"G1 V1.0.1 定向测试失败，exit={completed.returncode}：\n{output[-5000:]}"
        )
    match = re.search(r"(?P<count>\d+) passed", output)
    if match is None:
        raise EvidenceContractError("G1 V1.0.1 测试输出没有 passed 计数")
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
    correction = load_config(config_path)
    verify_frozen_manifest(config_path)
    original = load_original_config(ORIGINAL_CONFIG)
    original_outputs = {
        name: _project_path(str(path))
        for name, path in original["outputs"].items()
        if name != "curated_root"
    }
    corrected_outputs = {
        name: _project_path(str(path)) for name, path in correction["outputs"].items()
    }
    existing = [str(path) for path in corrected_outputs.values() if path.exists()]
    if existing:
        raise EvidenceContractError(f"G1 V1.0.1 终态输出已存在，禁止覆盖：{existing}")

    build_receipt = read_json_strict(original_outputs["build_receipt"])
    if not isinstance(build_receipt, dict):
        raise EvidenceContractError("G1 V1 构建收据必须是对象")
    _verify_self_digest(build_receipt, "receipt_payload_sha256", "G1 V1 构建收据")
    print("V1.0.1 在全新进程重建 G1，只修正文件身份字段比较。", flush=True)
    rebuilt = load_and_build(original)
    comparisons: dict[str, Any] = {}
    for output_name, attribute_name in TABLE_OUTPUTS.items():
        metadata = build_receipt["outputs"][output_name]
        stored_path = verify_file_identity_subset(
            metadata, label=f"G1 V1 输出 {output_name}"
        )
        stored = pd.read_parquet(stored_path)
        current = getattr(rebuilt, attribute_name)
        if len(stored) != int(metadata["row_count"]):
            raise EvidenceContractError(f"G1 V1 输出行数漂移：{output_name}")
        semantic = frame_semantic_sha256(stored)
        if semantic != str(metadata["persisted_semantic_sha256"]):
            raise EvidenceContractError(f"G1 V1 输出语义摘要漂移：{output_name}")
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
                f"G1 V1.0.1 重放逐值不一致 {output_name}：{exc}"
            ) from exc
        comparisons[output_name] = {
            "file_identity_fields": {
                key: metadata[key] for key in ("path", "bytes", "sha256")
            },
            "row_count": int(len(stored)),
            "persisted_semantic_sha256": semantic,
            "frame_equal_ignoring_storage_dtype_representation": True,
        }

    if original_outputs["report"].read_text(encoding="utf-8") != _build_report(rebuilt):
        raise EvidenceContractError("G1 V1.0.1 报告重放不一致")
    if canonical_sha256(rebuilt.metrics) != canonical_sha256(build_receipt["metrics"]):
        raise EvidenceContractError("G1 V1.0.1 计数指标不一致")
    if canonical_sha256(rebuilt.gate_result) != canonical_sha256(
        build_receipt["gate_result"]
    ):
        raise EvidenceContractError("G1 V1.0.1 门禁结果不一致")

    expected = correction["expected_gate_result"]
    gate = rebuilt.gate_result
    if gate["G1_DATA_AND_EVENTS"] != expected["G1_DATA_AND_EVENTS"]:
        raise EvidenceContractError("G1 状态与追加冻结不一致")
    if gate["mechanism_discovery_prerequisite_passed"] is not expected[
        "mechanism_discovery_prerequisite_passed"
    ]:
        raise EvidenceContractError("G1 机制门与追加冻结不一致")
    if gate["full_three_coefficient_model_prerequisite_passed"] is not expected[
        "full_three_coefficient_model_prerequisite_passed"
    ]:
        raise EvidenceContractError("G1 完整模型门与追加冻结不一致")
    if gate["next_allowed_step"] != expected["next_allowed_step"]:
        raise EvidenceContractError("G1 下一步与追加冻结不一致")
    if rebuilt.metrics["b2_identifiable_event_count"] != int(
        expected["b2_identifiable_event_count"]
    ):
        raise EvidenceContractError("B2 事件数与追加冻结不一致")
    if rebuilt.metrics["b3_identifiable_event_count"] != int(
        expected["b3_identifiable_event_count"]
    ):
        raise EvidenceContractError("B3 事件数与追加冻结不一致")

    print("G1 三张表、行数和语义摘要一致，开始运行五个冻结测试文件。", flush=True)
    tests = _run_targeted_tests(correction)
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_G1_EVENT_PREDICTION_ADMISSION_CLEAN_REPLAY_V1_0_1",
        "created_at": now,
        "status": "PASS_FRESH_PROCESS_REPLAY_AFTER_FILE_EVIDENCE_SUBSET_CORRECTION",
        "original_v1_failure_retained": True,
        "original_table_files_rewritten": False,
        "correction_scope": correction["correction_contract"]["scope"],
        "table_comparisons": comparisons,
        "metrics": rebuilt.metrics,
        "gate_result": gate,
        "targeted_tests": tests,
        "actual_label_artifact_read": True,
        "actual_future_path_value_read": False,
        "model_trained": False,
        "probability_generated": False,
        "prediction_metric_generated": False,
        "portfolio_or_performance_artifact_read": False,
        "position_impact": 0,
    }
    receipt["receipt_payload_sha256"] = canonical_sha256(receipt)
    atomic_write_json_new(corrected_outputs["clean_replay_receipt"], receipt)

    status: dict[str, Any] = {
        "program_id": original["program"]["program_id"],
        "execution_id": correction["program"]["execution_id"],
        "updated_at": now,
        "G0_ENGINEERING_AND_CONTRACT": "PASS",
        "G1_DATA_AND_EVENTS": gate["G1_DATA_AND_EVENTS"],
        "G1_MECHANISM_DISCOVERY_PREREQUISITE": "FAIL",
        "G1_FULL_THREE_COEFFICIENT_MODEL_PREREQUISITE": "FAIL",
        "G2_STRUCTURAL_INCREMENT": "NOT_RUN_BLOCKED_BY_G1",
        "G3_MACRO_INCREMENT": "NOT_RUN_BLOCKED_BY_G1",
        "G4_THROUGH_G7": "NOT_RUN",
        "metrics": rebuilt.metrics,
        "next_allowed_step": gate["next_allowed_step"],
        "branch_state": "STOPPED_AT_G1_NO_FEATURE_OR_LABEL_RESCUE",
        "clean_replay_receipt": file_evidence(
            corrected_outputs["clean_replay_receipt"], project_root=ROOT
        ),
        "original_v1_replay_failure": "RETAINED",
        "correction_scope": correction["correction_contract"]["scope"],
        "actual_label_artifact_read": True,
        "actual_future_path_value_read": False,
        "model_trained": False,
        "probability_generated": False,
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_evaluation": "NOT_ALLOWED",
        "paper_or_shadow": "NOT_AUTHORIZED",
        "broker_connection": "NOT_AUTHORIZED",
        "position_impact": 0,
    }
    status["status_payload_sha256"] = canonical_sha256(status)
    atomic_write_json_new(corrected_outputs["status"], status)
    print(
        "G1 V1.0.1 重放通过；23 个事件低于 30/40 门，"
        "V2 在 G1 停止，G2-G7 未运行。"
    )
    return status


def _write_failure_receipt(config_path: Path, exc: BaseException) -> None:
    try:
        correction = load_config(config_path)
        path = _project_path(correction["outputs"]["clean_replay_failure_receipt"])
        if path.exists():
            return
        payload = {
            "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_G1_EVENT_PREDICTION_ADMISSION_CLEAN_REPLAY_V1_0_1_FAILURE",
            "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "status": "FAIL_CORRECTED_G1_CLEAN_REPLAY_NO_GATE_PROMOTION",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "original_v1_failure_retained": True,
            "model_trained": False,
            "probability_generated": False,
            "portfolio_or_performance_artifact_read": False,
            "position_impact": 0,
        }
        payload["receipt_payload_sha256"] = canonical_sha256(payload)
        atomic_write_json_new(path, payload)
    except Exception as receipt_error:
        print(f"G1 V1.0.1 失败收据写入也失败：{receipt_error}")


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
        print(f"G1 V1.0.1 重放失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
