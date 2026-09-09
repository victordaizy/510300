"""执行 M/F/T V1.0.1：仅规范化已声明的持久化 dtype 后做完整重放。"""

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
    DEFAULT_CONFIG as V1_CONFIG,
    load_config as load_v1_config,
)
from scripts.freeze_510300_stress_transmission_hazard_v2_mft_feature_execution_v1_0_1 import (
    DEFAULT_CONFIG,
    load_config,
    verify_frozen_manifest,
)


def _project_path(relative: str) -> Path:
    return ROOT / Path(normalize_project_relative_path(relative))


def canonicalize_declared_persisted_dtypes(
    *,
    output_name: str,
    rebuilt: pd.DataFrame,
    stored: pd.DataFrame,
    correction: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """只执行配置中两个明确声明的 object→float64 转换。"""

    if rebuilt.columns.tolist() != stored.columns.tolist():
        raise EvidenceContractError(f"{output_name} 列顺序不一致")
    if len(rebuilt) != len(stored):
        raise EvidenceContractError(f"{output_name} 行数不一致")
    result = rebuilt.copy()
    allowed = correction["canonicalization_rule"]["allowed_conversions"].get(
        output_name, {}
    )
    applied: list[dict[str, str]] = []
    for column in result.columns:
        rebuilt_dtype = str(result[column].dtype)
        stored_dtype = str(stored[column].dtype)
        if rebuilt_dtype == stored_dtype:
            continue
        specification = allowed.get(column)
        if specification is None:
            raise EvidenceContractError(
                f"{output_name}.{column} 出现未声明 dtype 差异："
                f"rebuilt={rebuilt_dtype}, stored={stored_dtype}"
            )
        if rebuilt_dtype != str(specification["from"]) or stored_dtype != str(
            specification["to"]
        ):
            raise EvidenceContractError(
                f"{output_name}.{column} dtype 差异不符合唯一修正规则："
                f"rebuilt={rebuilt_dtype}, stored={stored_dtype}"
            )
        if specification["from"] != "object" or specification["to"] != "float64":
            raise EvidenceContractError("V1.0.1 只实现 object→float64 规范化")
        converted = pd.to_numeric(result[column], errors="coerce").astype("float64")
        original_missing = result[column].isna()
        if converted.isna().ne(original_missing).any():
            raise EvidenceContractError(f"{output_name}.{column} 规范化改变缺失位置")
        result[column] = converted
        applied.append(
            {"column": column, "from": rebuilt_dtype, "to": str(result[column].dtype)}
        )
    expected_applied = len(allowed)
    if len(applied) != expected_applied:
        raise EvidenceContractError(
            f"{output_name} 预期规范化 {expected_applied} 列，实际 {len(applied)} 列"
        )
    residual = [
        column
        for column in result.columns
        if str(result[column].dtype) != str(stored[column].dtype)
    ]
    if residual:
        raise EvidenceContractError(f"{output_name} 仍有 dtype 差异：{residual}")
    return result, applied


def _verify_file_evidence(evidence: Mapping[str, Any], label: str) -> Path:
    path = _project_path(str(evidence["path"]))
    if not path.is_file():
        raise EvidenceContractError(f"{label}不存在：{path}")
    if path.stat().st_size != int(evidence["bytes"]) or sha256_file(path) != str(
        evidence["sha256"]
    ):
        raise EvidenceContractError(f"{label}字节或哈希漂移：{path}")
    return path


def _verify_v1_receipt(receipt: Mapping[str, Any]) -> None:
    expected = str(receipt.get("receipt_payload_sha256", ""))
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256", None)
    if canonical_sha256(payload) != expected:
        raise EvidenceContractError("V1 M/F/T 构建收据自身摘要失败")


def _run_targeted_tests(correction: Mapping[str, Any]) -> dict[str, Any]:
    paths = [str(_project_path(path)) for path in correction["targeted_test_files"]]
    parent = ROOT / "artifacts" / "tmp"
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%f")
    base_temp = parent / f"mft_g0_v1_0_1_pytest_{stamp}"
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
            f"V1.0.1 定向测试失败，exit={completed.returncode}：\n{output[-4000:]}"
        )
    match = re.search(r"(?P<count>\d+) passed", output)
    if match is None:
        raise EvidenceContractError("V1.0.1 测试输出没有 passed 计数")
    return {
        "status": "PASS",
        "exit_code": completed.returncode,
        "passed_test_count": int(match.group("count")),
        "test_file_count": len(paths),
        "test_files": [Path(path).relative_to(ROOT).as_posix() for path in paths],
        "stdout_stderr_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "summary_tail": output.strip()[-1000:],
        "base_temp": base_temp.relative_to(ROOT).as_posix(),
    }


def replay(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    correction = load_config(config_path)
    verify_frozen_manifest(config_path)
    v1 = load_v1_config(V1_CONFIG)
    corrected_outputs = {
        name: _project_path(path)
        for name, path in correction["corrected_outputs"].items()
    }
    existing = [str(path) for path in corrected_outputs.values() if path.exists()]
    if existing:
        raise EvidenceContractError(f"V1.0.1 重放输出已存在，禁止覆盖：{existing}")
    v1_receipt_path = _project_path(correction["immutable_parent"]["build_receipt"]["path"])
    receipt = read_json_strict(v1_receipt_path)
    if not isinstance(receipt, dict):
        raise EvidenceContractError("V1 M/F/T 构建收据必须是对象")
    _verify_v1_receipt(receipt)

    print("V1.0.1 在全新进程重建全部 M/F/T，仅允许两个声明的 dtype 规范化。", flush=True)
    rebuilt = load_and_build(v1)
    comparison: dict[str, Any] = {}
    for output_name, attribute_name in TABLE_OUTPUTS.items():
        evidence = receipt["outputs"][output_name]
        stored_path = _verify_file_evidence(evidence, f"V1 输出 {output_name}")
        stored = pd.read_parquet(stored_path)
        current = getattr(rebuilt, attribute_name)
        normalized, applied = canonicalize_declared_persisted_dtypes(
            output_name=output_name,
            rebuilt=current,
            stored=stored,
            correction=correction,
        )
        try:
            pd.testing.assert_frame_equal(
                stored,
                normalized,
                check_dtype=True,
                check_exact=True,
                check_categorical=True,
            )
        except AssertionError as exc:
            raise EvidenceContractError(
                f"V1.0.1 重放逐值不一致 {output_name}：{exc}"
            ) from exc
        semantic = frame_semantic_sha256(normalized)
        if output_name in correction["affected_outputs"]:
            expected_semantic = correction["affected_outputs"][output_name][
                "persisted_semantic_sha256"
            ]
        else:
            expected_semantic = evidence["semantic_sha256"]
        if semantic != expected_semantic:
            raise EvidenceContractError(f"V1.0.1 写后语义摘要不一致：{output_name}")
        comparison[output_name] = {
            "row_count": int(len(normalized)),
            "semantic_sha256": semantic,
            "frame_equal": True,
            "declared_dtype_canonicalizations": applied,
        }

    print("V1.0.1 十张表逐值一致，开始运行七个冻结范围测试文件。", flush=True)
    tests = _run_targeted_tests(correction)
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    replay_receipt: dict[str, Any] = {
        "receipt_id": "510300_STRESS_TRANSMISSION_HAZARD_V2_MFT_CLEAN_REPLAY_V1_0_1",
        "created_at": now,
        "status": "PASS_FRESH_PROCESS_REPLAY_AFTER_EXACT_PERSISTED_SCHEMA_CANONICALIZATION",
        "original_v1_failure_retained": True,
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
            "persisted_schema_comparison": "PASS_AFTER_TWO_DECLARED_OBJECT_TO_FLOAT64_CANONICALIZATIONS",
        },
        "feature_formula_changed": False,
        "information_clock_changed": False,
        "feature_values_changed": False,
        "actual_label_artifact_read": False,
        "actual_future_return_artifact_read": False,
        "actual_performance_artifact_read": False,
        "position_impact": 0,
    }
    replay_receipt["receipt_payload_sha256"] = canonical_sha256(replay_receipt)
    atomic_write_json_new(corrected_outputs["replay_receipt"], replay_receipt)
    g0_status = {
        "program_id": correction["program"]["program_id"],
        "execution_id": correction["program"]["correction_id"],
        "updated_at": now,
        "G0_ENGINEERING_AND_CONTRACT": "PASS",
        "G1_DATA_AND_EVENTS": "NOT_ADJUDICATED",
        "G2_THROUGH_G7": "NOT_RUN",
        "original_v1_replay_failure": "RETAINED",
        "v1_0_1_correction_scope": correction["program"]["correction_scope"],
        "next_allowed_step": "FREEZE_EVENT_LEVEL_PREDICTION_EXECUTION_BEFORE_READING_ACTUAL_LABELS",
        "return_evaluation": "NOT_ALLOWED",
        "portfolio_evaluation": "NOT_ALLOWED_BEFORE_G0_THROUGH_G4_ALL_PASS",
        "paper_or_shadow": "NOT_AUTHORIZED",
        "broker_connection": "NOT_AUTHORIZED",
        "position_impact": 0,
        "mft_build_receipt": file_evidence(v1_receipt_path, project_root=ROOT),
        "clean_replay_receipt": file_evidence(
            corrected_outputs["replay_receipt"], project_root=ROOT
        ),
    }
    atomic_write_json_new(corrected_outputs["g0_status"], g0_status)
    print("V1.0.1 重放和定向测试通过；G0 通过，G1 尚未裁决。")
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
        print(f"M/F/T V1.0.1 干净重放失败：{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
