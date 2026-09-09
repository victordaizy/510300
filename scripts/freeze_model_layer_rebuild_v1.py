"""冻结统一模型协议、实现、测试和输入数据；冻结后方可运行收益。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.model_layer_rebuild_v1 import audit_and_load_inputs, load_config, sha256_file


MANIFEST = ROOT / "config" / "model_layer_rebuild_v1_manifest.json"
FROZEN_FILES = [
    "docs/510300_MODEL_LAYER_REBUILD_V1_SPEC.md",
    "config/model_layer_rebuild_v1.yaml",
    "research/model_layer_rebuild_v1.py",
    "scripts/audit_model_layer_rebuild_v1.py",
    "scripts/run_model_layer_rebuild_v1.py",
    "scripts/freeze_model_layer_rebuild_v1.py",
    "tests/test_model_layer_rebuild_v1.py",
]


def _git_output(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_frozen_tests() -> dict[str, object]:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "pytest",
        "tests/test_model_layer_rebuild_v1.py",
        "-q",
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"冻结前测试失败：\n{result.stdout}\n{result.stderr}")
    return {"command": command, "return_code": result.returncode, "stdout": result.stdout.strip()}


def main() -> int:
    config = load_config()
    _, audit = audit_and_load_inputs(ROOT, config)
    if audit["technical_gate"]["status"] != "PASS":
        raise RuntimeError("技术数据闸门未通过，禁止冻结")
    if audit["valuation_gate"]["status"] != "NO_VIEW_BLOCKED_NON_VINTAGE_HISTORY":
        raise RuntimeError("估值闸门状态偏离冻结边界")
    if audit["r5_replay_gate"]["decomposition_input_status"] != "PASS_PRESERVED_DERIVED_SIGNAL_SNAPSHOT":
        raise RuntimeError("R5保存快照不完整，禁止冻结")
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    test_result = _run_frozen_tests()
    data_files = {
        contract["file"]: contract["sha256"]
        for contract in config["data_contracts"].values()
    }
    for relative, expected in data_files.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"输入数据哈希偏离冻结配置：{relative}")
    payload = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_output("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_output("status", "--porcelain")),
        "implementation_frozen": True,
        "reporting_revision": config["protocol"]["reporting_revision"],
        "reporting_fix_only": config["protocol"]["reporting_fix_only"],
        "reporting_fix_reason": config["protocol"]["reporting_fix_reason"],
        "historical_evidence_label": "HISTORICALLY_CONTAMINATED",
        "true_forward_start": None,
        "frozen_files": {
            relative: sha256_file(ROOT / relative) for relative in FROZEN_FILES
        },
        "data_files": data_files,
        "data_gate_status_at_freeze": {
            "technical": audit["technical_gate"]["status"],
            "valuation": audit["valuation_gate"]["status"],
            "r5_raw_replay": audit["r5_replay_gate"]["raw_replay_status"],
            "r5_decomposition_input": audit["r5_replay_gate"]["decomposition_input_status"],
        },
        "trial_registry": {
            "registered_technical_trials": config["evaluation"]["multiple_testing"]["registered_technical_trial_count"],
            "return_tested_technical_candidates": config["evaluation"]["multiple_testing"]["return_tested_technical_candidate_count"],
            "registered_val01_trials": len(config["valuation_models"]["val01_registered_trials"]),
            "registered_val02_trials": len(config["valuation_models"]["val02_registered_trials"]),
        },
        "test_result": test_result,
        "governance": {
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "combination_run_allowed": False,
        },
    }
    payload["manifest_content_sha256"] = _canonical_hash(payload)
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
