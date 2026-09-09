"""修正冻结审计器的日期分辨率假阴性，不改动任何研究逻辑或输入。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_510300_regime_transition_router_v1_outputs as frozen_audit  # noqa: E402


ORIGINAL_AUDIT = (
    ROOT / "scripts" / "audit_510300_regime_transition_router_v1_outputs.py"
)
CORRECTED_AUDIT = Path(__file__).resolve()
MANIFEST_PATH = ROOT / "config" / "510300_regime_transition_router_v1_manifest.json"
AUDIT_PATH = (
    ROOT / "reports" / "audit" / "510300_regime_transition_router_v1_output_audit.json"
)
INITIAL_FAILURE_PATH = (
    ROOT
    / "reports"
    / "audit"
    / "510300_regime_transition_router_v1_output_audit_initial_failure.json"
)
CORRECTION_RECEIPT_PATH = (
    ROOT
    / "reports"
    / "audit"
    / "510300_regime_transition_router_v1_output_audit_v1_0_1_correction.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _verify_episode_table_valuewise(
    stored: pd.DataFrame,
    expected: pd.DataFrame,
) -> bool:
    """按值比较日期，避免 pandas 对 datetime64[us]/[ms] 分辨率敏感。"""

    if len(stored) != len(expected):
        return False
    stored = stored.reset_index(drop=True).copy()
    expected = expected.reset_index(drop=True).copy()
    for column in ["start_date", "end_date"]:
        stored_dates = pd.to_datetime(stored[column], errors="raise").to_numpy(
            dtype="datetime64[ns]"
        )
        expected_dates = pd.to_datetime(expected[column], errors="raise").to_numpy(
            dtype="datetime64[ns]"
        )
        if not np.array_equal(stored_dates, expected_dates):
            return False
    for column in ["episode_id", "signal_rows"]:
        if stored[column].astype(int).tolist() != expected[column].astype(int).tolist():
            return False
    if stored["state"].astype(str).tolist() != expected["state"].astype(str).tolist():
        return False
    for column in ["left_censored", "right_censored"]:
        if [frozen_audit._as_bool(value) for value in stored[column]] != [
            bool(value) for value in expected[column]
        ]:
            return False
    return True


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    frozen_hash = manifest["tracked_files"][
        "scripts/audit_510300_regime_transition_router_v1_outputs.py"
    ]
    if _sha256(ORIGINAL_AUDIT) != frozen_hash:
        raise RuntimeError("原冻结审计器发生哈希漂移，禁止执行修正版")

    initial_bytes = AUDIT_PATH.read_bytes()
    initial = json.loads(initial_bytes.decode("utf-8"))
    if initial.get("status") != "FAIL_INDEPENDENT_OUTPUT_RECOMPUTATION":
        raise RuntimeError("当前审计结果不是待修正的初始失败状态")
    if initial.get("failed_checks") != ["episodes_recomputed"]:
        raise RuntimeError("初始审计存在日期分辨率以外的失败，禁止窄修正")
    _atomic_text(INITIAL_FAILURE_PATH, initial_bytes.decode("utf-8"))

    frozen_audit._verify_episode_table = _verify_episode_table_valuewise
    corrected = frozen_audit.audit()
    if corrected.get("status") != "PASS_INDEPENDENT_OUTPUT_RECOMPUTATION":
        raise RuntimeError(f"修正后独立复核仍未通过：{corrected.get('failed_checks')}")

    receipt = {
        "project_id": corrected["project_id"],
        "status": "PASS_POSTFREEZE_AUDIT_IMPLEMENTATION_CORRECTION",
        "correction_scope": "EPISODE_TABLE_DATE_DTYPE_RESOLUTION_COMPARISON_ONLY",
        "failure_mechanism": {
            "stored_date_dtype": "datetime64[us]",
            "recomputed_date_dtype": "datetime64[ms]",
            "all_episode_values_equal": True,
            "pandas_dataframe_equals_returned_false_due_to_dtype_resolution": True,
        },
        "research_logic_changed": False,
        "research_inputs_changed": False,
        "frozen_audit_file_changed": False,
        "formal_research_module_imported": False,
        "original_frozen_audit_sha256": frozen_hash,
        "postfreeze_correction_script": CORRECTED_AUDIT.relative_to(ROOT).as_posix(),
        "postfreeze_correction_script_sha256": _sha256(CORRECTED_AUDIT),
        "initial_failure_preserved_at": INITIAL_FAILURE_PATH.relative_to(ROOT).as_posix(),
        "initial_failure_sha256": hashlib.sha256(initial_bytes).hexdigest(),
        "corrected_audit_path": AUDIT_PATH.relative_to(ROOT).as_posix(),
        "corrected_audit_sha256": _sha256(AUDIT_PATH),
        "corrected_audit_status": corrected["status"],
        "corrected_failed_checks": corrected["failed_checks"],
        "formal_result_status": corrected["formal_result_status"],
        "phase_1_state_gate_passed": corrected["phase_1_state_gate_passed"],
        "phase_2_status": corrected["phase_2_status"],
        "live_trading_authorized": False,
    }
    _atomic_json(CORRECTION_RECEIPT_PATH, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
