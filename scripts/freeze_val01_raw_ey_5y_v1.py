"""冻结VAL01_RAW_EY_5Y_V1模型卡、实现和无收益标签证据。"""

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

from research.val01_raw_ey_5y_v1 import (
    FORBIDDEN_OUTPUT_TOKENS,
    load_config,
    sha256_file,
    verify_input_hashes,
)


FROZEN_FILES = (
    "config/val01_raw_ey_5y_v1.yaml",
    "docs/VAL01_RAW_EY_5Y_V1_MODEL_CARD.md",
    "research/val01_raw_ey_5y_v1.py",
    "scripts/build_val01_raw_ey_5y_v1.py",
    "scripts/freeze_val01_raw_ey_5y_v1.py",
    "tests/test_val01_raw_ey_5y_v1.py",
)


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


def _canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_tests() -> dict[str, object]:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "pytest",
        "tests/test_val01_raw_ey_5y_v1.py",
        "tests/test_point_in_time_valuation_price_acquisition.py",
        "tests/test_point_in_time_valuation_v2_audit.py",
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
    return {"command": command, "return_code": 0, "stdout": result.stdout.strip()}


def main() -> int:
    config = load_config()
    if verify_input_hashes(config)["status"] != "PASS":
        raise RuntimeError("冻结输入哈希不一致")
    outputs = (
        config["artifacts"]["signal_inputs"],
        config["artifacts"]["report_json"],
        config["artifacts"]["report_markdown"],
    )
    missing = [path for path in (*FROZEN_FILES, *outputs) if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    report = json.loads((ROOT / config["artifacts"]["report_json"]).read_text(encoding="utf-8"))
    if report["status"] != "PASS_SIGNAL_INPUTS_NO_RETURN_LABELS":
        raise RuntimeError(f"信号输入闸门未通过：{report['status']}")
    if any(report["governance"].values()):
        raise RuntimeError("治理边界异常")
    signals = pd.read_parquet(ROOT / config["artifacts"]["signal_inputs"])
    lowered = [str(column).lower() for column in signals.columns]
    leaked = [
        column for column in lowered if any(token in column for token in FORBIDDEN_OUTPUT_TOKENS)
    ]
    if leaked:
        raise RuntimeError(f"冻结输出含禁止字段：{leaked}")
    tests = _run_tests()
    prior_manifest = config["data_contracts"]["upstream_v1_1_manifest"]
    payload: dict[str, object] = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_output("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_output("status", "--porcelain")),
        "upstream_v1_1_manifest": {
            "file": prior_manifest["file"],
            "sha256": sha256_file(ROOT / prior_manifest["file"]),
            "mutated": False,
        },
        "frozen_files": {path: sha256_file(ROOT / path) for path in FROZEN_FILES},
        "output_files": {path: sha256_file(ROOT / path) for path in outputs},
        "status_at_freeze": {
            "signal_input_gate": report["status"],
            "row_count": report["signal_input_summary"]["row_count"],
            "first_percentile_ready_date": report["signal_input_summary"][
                "first_percentile_ready_date"
            ],
            "first_earliest_execution_date": report["signal_input_summary"][
                "first_earliest_execution_date"
            ],
            "minimum_joint_valid_weight_coverage": report["signal_input_summary"][
                "minimum_joint_valid_weight_coverage"
            ],
            "historical_label": "HISTORICALLY_CONTAMINATED_NOT_STRICT_OOS",
        },
        "governance": report["governance"],
        "test_result": tests,
    }
    payload["manifest_content_sha256"] = _canonical_hash(payload)
    manifest_path = ROOT / config["artifacts"]["manifest"]
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
