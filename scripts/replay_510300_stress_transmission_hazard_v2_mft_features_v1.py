"""在全新进程重建冻结 M/F/T，并运行定向测试以裁决 G0。"""

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
    sha256_file,
)
from scripts.build_510300_stress_transmission_hazard_v2_mft_features_v1 import (
    TABLE_OUTPUTS,
    frame_semantic_sha256,
    load_and_build,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_mft_feature_execution_v1 import (
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
    actual = canonical_sha256(payload)
    if expected != actual:
        raise EvidenceContractError(
            f"M/F/T 构建收据自身摘要失败：expected={expected}, actual={actual}"
        )


def _verify_output_evidence(evidence: Mapping[str, Any], label: str) -> Path:
    path = _project_path(str(evidence["path"]))
    if not path.is_file():
        raise EvidenceContractError(f"{label}不存在：{path}")
    if path.stat().st_size != int(evidence["bytes"]) or sha256_file(path) != str(
        evidence["sha256"]
    ):
        raise EvidenceContractError(f"{label}字节或哈希漂移：{path}")
    return path


def _run_targeted_tests(config: Mapping[str, Any]) -> dict[str, Any]:
    test_files = [str(_project_path(path)) for path in config["clean_replay_contract"]["targeted_test_files"]]
    temporary_parent = ROOT / "artifacts" / "tmp"
    temporary_parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%f")
    base_temp = temporary_parent / f"mft_g0_pytest_{stamp}"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        *test_files,
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
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    if completed.returncode != 0:
        tail = output[-4000:]
        raise EvidenceContractError(
            f"G0 定向测试失败，exit={completed.returncode}：\n{tail}"
        )
    match = re.search(r"(?P<count>\d+) passed", output)
    if match is None:
        raise EvidenceContractError("G0 定向测试虽返回0，但没有可解析的 passed 计数")
    return {
        "status": "PASS",
        "exit_code": completed.returncode,
        "passed_test_count": int(match.group("count")),
        "test_file_count": len(test_files),
        "test_files": [Path(path).relative_to(ROOT).as_posix() for path in test_files],
        "stdout_stderr_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "summary_tail": output.strip()[-1000:],
        "base_temp": base_temp.relative_to(ROOT).as_posix(),
    }


def replay(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    verify_frozen_manifest(config_path)
    config = load_config(config_path)
    outputs = {
        key: _project_path(str(value))
        for key, value in config["outputs"].items()
        if key != "curated_root"
    }
    for name in ("replay_receipt", "g0_status"):
        if outputs[name].exists():
            raise EvidenceContractError(f"G0 输出已存在，禁止覆盖：{outputs[name]}")
    receipt = read_json_strict(outputs["receipt"])
    if not isinstance(receipt, dict):
        raise EvidenceContractError("M/F/T 构建收据必须是对象")
    _verify_receipt_payload(receipt)
    if receipt.get("status") != (
        "PASS_POINT_IN_TIME_M_F_T_WITH_EXPLICIT_NO_VIEW_PENDING_CLEAN_REPLAY"
    ):
        raise EvidenceContractError("M/F/T 构建收据不是待重放通过状态")
    if receipt.get("actual_label_artifact_read") is not False:
        raise EvidenceContractError("M/F/T 构建阶段意外读取实际标签")
    if receipt.get("actual_performance_artifact_read") is not False:
        raise EvidenceContractError("M/F/T 构建阶段意外读取绩效")

    print("在全新进程中按冻结输入重建 M/F/T；不写第二份特征值。", flush=True)
    rebuilt = load_and_build(config)
    comparison: dict[str, Any] = {}
    for output_name, attribute_name in TABLE_OUTPUTS.items():
        evidence = receipt["outputs"][output_name]
        path = _verify_output_evidence(evidence, f"已固化输出 {output_name}")
        stored = pd.read_parquet(path)
        current = getattr(rebuilt, attribute_name)
        try:
            pd.testing.assert_frame_equal(
                stored,
                current,
                check_dtype=True,
                check_exact=True,
                check_categorical=True,
            )
        except AssertionError as exc:
            raise EvidenceContractError(
                f"干净重放表语义不一致 {output_name}：{exc}"
            ) from exc
        semantic = frame_semantic_sha256(current)
        if semantic != str(evidence["semantic_sha256"]):
            raise EvidenceContractError(f"干净重放语义摘要不一致：{output_name}")
        comparison[output_name] = {
            "row_count": int(len(current)),
            "semantic_sha256": semantic,
            "frame_equal": True,
        }

    print("逐表语义重放一致，开始运行冻结范围内的定向测试。", flush=True)
    tests = _run_targeted_tests(config)
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    replay_receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_MFT_CLEAN_REPLAY_V1",
        "created_at": now,
        "status": "PASS_FRESH_PROCESS_SEMANTIC_REPLAY_AND_TARGETED_TESTS",
        "frozen_manifest_verified": True,
        "semantic_frame_comparison": comparison,
        "targeted_tests": tests,
        "g0_requirements": {
            "clean_environment_replay": "PASS",
            "code_config_result_manifest_hash_chain": "PASS",
            "industry_fields_cannot_enter_core": "PASS",
            "four_state_return_tests": "PASS",
            "corporate_action_and_suspension_tests": "PASS",
            "macro_available_at_and_no_future_tests": "PASS",
            "new_member_history_tests": "PASS",
        },
        "actual_label_artifact_read": False,
        "actual_future_return_artifact_read": False,
        "actual_performance_artifact_read": False,
        "position_impact": 0,
    }
    replay_receipt["receipt_payload_sha256"] = canonical_sha256(replay_receipt)
    atomic_write_json_new(outputs["replay_receipt"], replay_receipt)
    g0_status = {
        "program_id": config["program"]["program_id"],
        "execution_id": config["program"]["execution_id"],
        "updated_at": now,
        "G0_ENGINEERING_AND_CONTRACT": "PASS",
        "G1_DATA_AND_EVENTS": "NOT_ADJUDICATED",
        "G2_THROUGH_G7": "NOT_RUN",
        "next_allowed_step": "FREEZE_EVENT_LEVEL_PREDICTION_EXECUTION_BEFORE_READING_ACTUAL_LABELS",
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_evaluation": "NOT_ALLOWED_BEFORE_G0_THROUGH_G4_ALL_PASS",
        "paper_or_shadow": "NOT_AUTHORIZED",
        "broker_connection": "NOT_AUTHORIZED",
        "position_impact": 0,
        "mft_build_receipt": file_evidence(outputs["receipt"], project_root=ROOT),
        "clean_replay_receipt": file_evidence(outputs["replay_receipt"], project_root=ROOT),
    }
    atomic_write_json_new(outputs["g0_status"], g0_status)
    print("G0 工程与契约门通过；G1 尚未裁决，G2-G7 未运行。")
    return replay_receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    try:
        replay(args.config)
        return 0
    except (
        EvidenceContractError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"M/F/T 干净重放失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
