"""冻结 V1.0.1 确定性表 NaN 复刻比较器的第二次程序修正。"""

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
PRIOR_CORRECTION = ROOT / "config" / "510300_oracle_information_budget_v1_0_1_acceptance_program_correction_manifest.json"
CORRECTION_MANIFEST = ROOT / "config" / "510300_oracle_information_budget_v1_0_1_acceptance_program_correction_2_manifest.json"
FAILURE_RECEIPT = ROOT / "reports" / "research" / "510300_oracle_information_budget_v1_0_1_acceptance_program_failure_receipt_2.json"
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
        print(f"第二程序修正清单已存在，拒绝覆盖：{CORRECTION_MANIFEST}", file=sys.stderr)
        return 2
    if not BASE_MANIFEST.exists() or not PRIOR_CORRECTION.exists():
        print("基础冻结清单或第一程序修正清单不存在", file=sys.stderr)
        return 2
    base = json.loads(BASE_MANIFEST.read_text(encoding="utf-8"))
    prior = json.loads(PRIOR_CORRECTION.read_text(encoding="utf-8"))
    old_code_hash = prior["corrected_code_sha256"]
    corrected_code_hash = sha256_file(CORRECTED_CODE)
    if corrected_code_hash == old_code_hash:
        print("程序代码哈希没有变化，无需第二修正", file=sys.stderr)
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
        print(f"第二程序修正前已存在正式输出：{existing_outputs}", file=sys.stderr)
        return 2
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    receipt = {
        "schema_version": "1.0.0",
        "project_id": PROJECT_ID,
        "status": "PROGRAM_FAILED_BEFORE_FORMAL_OUTPUT_WRITE",
        "failed_at_asia_shanghai": now,
        "failed_stage": "ADVERSARIAL_V1_ACCOUNT_REPLAY_COMPARATOR",
        "error_type": "ContractError",
        "error": "确定性表复刻比较器把新旧双方均为NaN的全现金普通夏普误判为不一致",
        "random_axis_cells_computed_in_memory_then_discarded": 144,
        "random_joint_cells_computed_in_memory_then_discarded": 216,
        "adversarial_cells_computed_in_memory_then_discarded": 360,
        "timing_cells_computed": 0,
        "formal_outputs_written": False,
        "research_logic_changed": False,
        "results_exposed_in_failure_output": False,
        "live_trading_authorized": False,
    }
    atomic_json(FAILURE_RECEIPT, receipt)
    correction = {
        "schema_version": "1.0.0",
        "project_id": PROJECT_ID,
        "state": "FROZEN_DETERMINISTIC_NAN_COMPARATOR_CORRECTION_BEFORE_RERUN",
        "frozen_at_asia_shanghai": datetime.now(
            ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "base_manifest_path": BASE_MANIFEST.relative_to(ROOT).as_posix(),
        "base_manifest_sha256": sha256_file(BASE_MANIFEST),
        "prior_correction_manifest_path": PRIOR_CORRECTION.relative_to(ROOT).as_posix(),
        "prior_correction_manifest_sha256": sha256_file(PRIOR_CORRECTION),
        "old_code_sha256": old_code_hash,
        "corrected_code_path": CORRECTED_CODE.relative_to(ROOT).as_posix(),
        "corrected_code_sha256": corrected_code_hash,
        "correction_script_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "correction_script_sha256": sha256_file(Path(__file__).resolve()),
        "failure_receipt_path": FAILURE_RECEIPT.relative_to(ROOT).as_posix(),
        "failure_receipt_sha256": sha256_file(FAILURE_RECEIPT),
        "correction_scope": "DETERMINISTIC_AUDIT_COMPARATOR_TREATS_NAN_AS_EQUAL_ONLY_WHEN_BOTH_SIDES_ARE_NAN",
        "formal_outputs_existed_before_correction": False,
        "random_and_adversarial_metrics_computed_but_not_persisted_or_exposed": True,
        "timing_and_final_adjudication_known_before_correction": False,
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
