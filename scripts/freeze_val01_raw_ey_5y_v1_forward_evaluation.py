"""冻结VAL01未来收益标签、预测诊断与条件停止结论。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.val01_raw_ey_5y_v1_forward_evaluation import (
    SIGNAL_VALUE_COLUMNS_FORBIDDEN_IN_LABELS,
    load_config,
    sha256_file,
    verify_input_hashes,
)


FROZEN_FILES = (
    "config/val01_raw_ey_5y_v1_forward_evaluation.yaml",
    "docs/VAL01_RAW_EY_5Y_V1_FORWARD_LABEL_PROTOCOL.md",
    "research/val01_raw_ey_5y_v1_forward_evaluation.py",
    "scripts/evaluate_val01_raw_ey_5y_v1_forward.py",
    "scripts/freeze_val01_raw_ey_5y_v1_forward_evaluation.py",
    "tests/test_val01_raw_ey_5y_v1_forward_evaluation.py",
)


def _git_output(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_tests() -> dict[str, object]:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m", "pytest",
        "tests/test_val01_raw_ey_5y_v1_forward_evaluation.py",
        "tests/test_val01_raw_ey_5y_v1.py",
        "-q",
    ]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"冻结前测试失败：\n{result.stdout}\n{result.stderr}")
    return {"command": command, "return_code": 0, "stdout": result.stdout.strip()}


def main() -> int:
    config = load_config()
    if verify_input_hashes(config)["status"] != "PASS":
        raise RuntimeError("预测评价输入哈希不一致")
    outputs = (
        config["artifacts"]["label_table"],
        config["artifacts"]["report_json"],
        config["artifacts"]["report_markdown"],
    )
    missing = [path for path in (*FROZEN_FILES, *outputs) if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    report = json.loads((ROOT / config["artifacts"]["report_json"]).read_text(encoding="utf-8"))
    allowed_statuses = {
        config["primary_predictive_gate"]["pass_status"],
        config["primary_predictive_gate"]["fail_status"],
    }
    if report["status"] not in allowed_statuses:
        raise RuntimeError(f"预测屏幕没有形成正式结论：{report['status']}")
    if report["governance"]["alpha_pass"]:
        raise RuntimeError("受污染历史不得取得Alpha Pass")
    forbidden_true = (
        "historical_strategy_return_calculated",
        "historical_position_mapping_performed",
        "current_position_mapping_performed",
        "target_shares_generated",
        "orders_generated",
        "broker_connection_performed",
        "upstream_model_files_mutated",
    )
    if any(report["governance"][name] for name in forbidden_true):
        raise RuntimeError("预测屏幕越过治理边界")
    labels = pd.read_parquet(ROOT / config["artifacts"]["label_table"])
    leaked = SIGNAL_VALUE_COLUMNS_FORBIDDEN_IN_LABELS.intersection(labels.columns)
    if leaked:
        raise RuntimeError(f"标签表泄漏信号字段：{sorted(leaked)}")
    tests = _run_tests()
    upstream = config["data_contracts"]["upstream_model_manifest"]
    payload: dict[str, object] = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_output("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_output("status", "--porcelain")),
        "upstream_model_manifest": {
            "file": upstream["file"],
            "sha256": sha256_file(ROOT / upstream["file"]),
            "mutated": False,
        },
        "frozen_files": {path: sha256_file(ROOT / path) for path in FROZEN_FILES},
        "output_files": {path: sha256_file(ROOT / path) for path in outputs},
        "status_at_freeze": {
            "predictive_screen": report["status"],
            "primary_gate_passed": report["primary_predictive_gate"]["passed"],
            "failed_checks": report["primary_predictive_gate"]["failed_checks"],
            "maturity_by_horizon": report["label_audit"]["maturity_by_horizon"],
            "historical_label": report["historical_evidence_label"],
        },
        "governance": report["governance"],
        "test_result": tests,
    }
    payload["manifest_content_sha256"] = _canonical_hash(payload)
    manifest = ROOT / config["artifacts"]["manifest"]
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
