"""冻结 V1.0.1 首次运行的 NaN 复刻比较器程序修正。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST = ROOT / "config" / "510300_oracle_information_budget_v1_0_1_acceptance_manifest.json"
CORRECTION_MANIFEST = ROOT / "config" / "510300_oracle_information_budget_v1_0_1_acceptance_program_correction_manifest.json"
FAILURE_RECEIPT = ROOT / "reports" / "research" / "510300_oracle_information_budget_v1_0_1_acceptance_program_failure_receipt.json"
CORRECTED_CODE = ROOT / "research" / "oracle_information_budget_v1_0_1_acceptance.py"
PROJECT_ID = "510300_ORACLE_INFORMATION_BUDGET_V1_0_1_ACCEPTANCE"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    if CORRECTION_MANIFEST.exists():
        print(f"程序修正清单已存在，拒绝覆盖：{CORRECTION_MANIFEST}", file=sys.stderr)
        return 2
    if not BASE_MANIFEST.exists():
        print("V1.0.1基础冻结清单不存在", file=sys.stderr)
        return 2
    base = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    relative_code = "research/oracle_information_budget_v1_0_1_acceptance.py"
    old_code_hash = base["tracked_files"][relative_code]
    corrected_code_hash = sha256_file(CORRECTED_CODE)
    if corrected_code_hash == old_code_hash:
        print("程序代码哈希没有变化，无需修正", file=sys.stderr)
        return 2
    output_relatives = [
        "data/research/510300_oracle_information_budget_v1_0_1_acceptance/random_axis_conservative_summary.parquet",
        "data/research/510300_oracle_information_budget_v1_0_1_acceptance/random_joint_conservative_summary.parquet",
        "data/research/510300_oracle_information_budget_v1_0_1_acceptance/adversarial_conservative_summary.parquet",
        "data/research/510300_oracle_information_budget_v1_0_1_acceptance/timing_conservative_summary.parquet",
        "data/research/510300_oracle_information_budget_v1_0_1_acceptance/sharpe_1_2_feasible_region.csv",
        "reports/research/510300_oracle_information_budget_v1_0_1_acceptance_result.json",
        "reports/research/510300_ORACLE_INFORMATION_BUDGET_V1_0_1_ACCEPTANCE_RESULT.md",
        "reports/research/510300_post_atlas_project_state_v1.json",
        "reports/research/510300_POST_ATLAS_PROJECT_STATE_V1.md",
    ]
    existing_outputs = [
        relative for relative in output_relatives if (ROOT / relative).exists()
    ]
    if existing_outputs:
        print(f"程序修正前已存在正式输出：{existing_outputs}", file=sys.stderr)
        return 2
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    receipt = {
        "schema_version": "1.0.0",
        "project_id": PROJECT_ID,
        "status": "PROGRAM_FAILED_BEFORE_FORMAL_OUTPUT_WRITE",
        "failed_at_asia_shanghai": now,
        "failed_cell": "UP_00",
        "error_type": "ContractError",
        "error": "复刻比较器把新旧双方均为NaN的全现金夏普误判为不一致",
        "completed_novel_decision_cells": 0,
        "formal_outputs_written": False,
        "research_logic_changed": False,
        "live_trading_authorized": False,
    }
    atomic_json(FAILURE_RECEIPT, receipt)
    correction = {
        "schema_version": "1.0.0",
        "project_id": PROJECT_ID,
        "state": "FROZEN_NAN_COMPARATOR_PROGRAM_CORRECTION_BEFORE_RERUN",
        "frozen_at_asia_shanghai": datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "base_manifest_path": BASE_MANIFEST.relative_to(ROOT).as_posix(),
        "base_manifest_sha256": sha256_file(BASE_MANIFEST),
        "old_code_sha256": old_code_hash,
        "corrected_code_path": CORRECTED_CODE.relative_to(ROOT).as_posix(),
        "corrected_code_sha256": corrected_code_hash,
        "correction_script_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "correction_script_sha256": sha256_file(Path(__file__).resolve()),
        "failure_receipt_path": FAILURE_RECEIPT.relative_to(ROOT).as_posix(),
        "failure_receipt_sha256": sha256_file(FAILURE_RECEIPT),
        "correction_scope": "AUDIT_COMPARATOR_TREATS_NAN_AS_EQUAL_ONLY_WHEN_BOTH_SIDES_ARE_NAN",
        "formal_outputs_existed_before_correction": False,
        "first_cell_up00_nan_outcome_known": True,
        "remaining_conservative_outputs_known_before_correction": False,
        "random_seed_changed": False,
        "error_grid_changed": False,
        "account_engine_changed": False,
        "serial_metric_changed": False,
        "pass_gate_changed": False,
        "governance_changed": False,
    }
    atomic_json(CORRECTION_MANIFEST, correction)
    print(json.dumps(correction, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
