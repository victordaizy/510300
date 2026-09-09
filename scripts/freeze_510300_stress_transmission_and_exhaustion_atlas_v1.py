"""在首次新图谱未来结果读取前冻结压力传导与耗竭机制研究。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from stress_transmission_and_exhaustion_atlas_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    audit_inputs,
    load_config,
    load_inputs,
    sha256_file,
)


TRACKED_FILES = [
    "config/510300_stress_transmission_and_exhaustion_atlas_v1.yaml",
    "docs/510300_STRESS_TRANSMISSION_AND_EXHAUSTION_ATLAS_V1_SPEC.md",
    "research/stress_transmission_and_exhaustion_atlas_v1.py",
    "scripts/freeze_510300_stress_transmission_and_exhaustion_atlas_v1.py",
    "scripts/run_510300_stress_transmission_and_exhaustion_atlas_v1.py",
    "scripts/audit_510300_stress_transmission_and_exhaustion_atlas_v1_outputs.py",
    "tests/test_510300_stress_transmission_and_exhaustion_atlas_v1.py",
]


def _project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _aggregate_hash(mapping: dict[str, str]) -> str:
    canonical = json.dumps(mapping, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _run_synthetic_tests() -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_510300_stress_transmission_and_exhaustion_atlas_v1.py",
        "-q",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    result = {
        "command": " ".join(command),
        "return_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "new_atlas_future_outcomes_read": False,
    }
    if completed.returncode != 0:
        raise RuntimeError(f"预冻结纯合成测试失败：{result}")
    return result


def freeze() -> dict[str, Any]:
    config = load_config()
    if MANIFEST_PATH.exists():
        raise FileExistsError(f"冻结清单已经存在，禁止覆盖：{MANIFEST_PATH}")

    generated_path_keys = [
        "input_audit",
        "mechanism_panel",
        "outcome_panel",
        "router_event_atlas",
        "price_damage_events",
        "phase_landmarks",
        "event_trajectories",
        "event_period_summary",
        "event_source_summary",
        "leading_results",
        "question_a_rows",
        "question_b_rows",
        "model_results",
        "era_mechanism_results",
        "result_json",
        "result_markdown",
        "output_audit_json",
        "output_audit_markdown",
        "program_failure_json",
    ]
    preexisting = [
        config["paths"][key]
        for key in generated_path_keys
        if _project_path(config["paths"][key]).exists()
    ]
    if preexisting:
        raise RuntimeError(f"冻结前已经存在正式图谱结果或结果审计：{preexisting}")

    tests = _run_synthetic_tests()
    inputs = load_inputs(config)
    data_audit = audit_inputs(config, inputs)
    if not data_audit["passed"]:
        raise RuntimeError(f"输入数据合同未通过，禁止冻结：{data_audit}")
    if data_audit["boundaries"]["new_atlas_future_outcomes_read"]:
        raise RuntimeError("输入审计错误地读取了新图谱未来结果")

    tracked_hashes: dict[str, str] = {}
    for relative in TRACKED_FILES:
        path = _project_path(relative)
        if not path.exists():
            raise FileNotFoundError(f"缺少待冻结实现文件：{relative}")
        tracked_hashes[relative] = sha256_file(path)

    input_hashes: dict[str, str] = {}
    for specification in config["inputs"].values():
        relative = specification["path"]
        path = _project_path(relative)
        if not path.exists():
            raise FileNotFoundError(f"缺少待绑定输入：{relative}")
        input_hashes[relative] = sha256_file(path)

    prior_manifests = sorted(
        path
        for path in (ROOT / "config").glob("510300_*_manifest.json")
        if path.resolve() != MANIFEST_PATH.resolve()
    )
    prior_hashes = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in prior_manifests
    }
    frozen_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest = {
        "schema_version": "1.0.0",
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "state": "FROZEN_BEFORE_FIRST_NEW_ATLAS_OUTCOME_READ",
        "implementation_frozen": True,
        "frozen_before_new_atlas_outcome_read": True,
        "new_atlas_outcome_read_before_freeze": False,
        "strategy_return_read_before_freeze": False,
        "result_preexisted_at_freeze": False,
        "frozen_at_asia_shanghai": frozen_at,
        "evidence_class": config["protocol"]["evidence_class"],
        "research_stage": config["protocol"]["research_stage"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "tracked_files": tracked_hashes,
        "tracked_files_aggregate_sha256": _aggregate_hash(tracked_hashes),
        "input_files": input_hashes,
        "input_files_aggregate_sha256": _aggregate_hash(input_hashes),
        "prefreeze_checks": {
            "synthetic_tests": tests,
            "input_data_contract": data_audit,
            "protected_outcome_artifacts_absent": True,
        },
        "selection_bias_control": {
            "prior_510300_manifest_count": len(prior_manifests),
            "total_510300_manifest_count_including_current": len(prior_manifests) + 1,
            "prior_manifest_hashes": prior_hashes,
            "prior_manifest_hashes_aggregate_sha256": _aggregate_hash(prior_hashes),
            "count_is_conservative_upper_bound_not_independence_claim": True,
        },
        "no_rescue_contract": {
            "parameter_rescue_after_result": "FORBIDDEN",
            "threshold_rescue_after_result": "FORBIDDEN",
            "window_rescue_after_result": "FORBIDDEN",
            "label_rescue_after_result": "FORBIDDEN",
            "source_relabel_after_result": "FORBIDDEN",
            "event_redefinition_after_result": "FORBIDDEN",
            "alternate_outcome_after_result": "FORBIDDEN",
            "year_deletion_after_result": "FORBIDDEN",
            "hmm_or_machine_learning_rescue": "FORBIDDEN",
        },
        "execution_boundaries": {
            "research_mode": "DISCOVERY_ONLY",
            "old_router_modified": False,
            "systemic_holdout_opened": False,
            "strategy_backtest": "NOT_RUN",
            "position_mapping": "DISABLED",
            "paper_signal": "DISABLED",
            "shadow_signal": "DISABLED",
            "current_signal": "NOT_CREATED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _atomic_json(MANIFEST_PATH, manifest)

    receipt_path = _project_path(config["paths"]["freeze_receipt"])
    receipt = {
        "project_id": manifest["project_id"],
        "status": manifest["state"],
        "frozen_at_asia_shanghai": frozen_at,
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "config_sha256": manifest["config_sha256"],
        "tracked_file_count": len(tracked_hashes),
        "input_file_count": len(input_hashes),
        "prefreeze_synthetic_tests_passed": True,
        "prefreeze_input_data_contract_passed": True,
        "result_preexisted_at_freeze": False,
        "new_atlas_outcome_read_before_freeze": False,
        "strategy_return_read_before_freeze": False,
        "live_trading_authorized": False,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> int:
    receipt = freeze()
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
