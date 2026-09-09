"""冻结点时估值V2审计的规范、实现、测试、输入和审计产物指纹。"""

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

from research.point_in_time_valuation_v2_audit import (
    audit_hashes,
    canonical_directory_hash,
    load_config,
    sha256_file,
)


FROZEN_FILES = (
    "docs/000300_POINT_IN_TIME_VALUATION_V2_RECONSTRUCTIBILITY_SPEC.md",
    "config/point_in_time_valuation_v2_audit.yaml",
    "research/point_in_time_valuation_v2_audit.py",
    "scripts/audit_point_in_time_valuation_v2.py",
    "scripts/freeze_point_in_time_valuation_v2_audit.py",
    "tests/test_point_in_time_valuation_v2_audit.py",
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


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_tests() -> dict[str, object]:
    command = [
        str(ROOT / ".venv" / "Scripts" / "python.exe"),
        "-m",
        "pytest",
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
        raise RuntimeError(f"审计冻结前测试失败：\n{result.stdout}\n{result.stderr}")
    return {
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout.strip(),
    }


def main() -> int:
    config = load_config()
    manifest_path = ROOT / config["artifacts"]["manifest"]
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    input_hashes = audit_hashes(config)
    if input_hashes["status"] != "PASS":
        raise RuntimeError("输入数据哈希偏离审计配置，禁止冻结")
    report_path = ROOT / config["artifacts"]["report_json"]
    markdown_path = ROOT / config["artifacts"]["report_markdown"]
    coverage_path = ROOT / config["artifacts"]["snapshot_coverage"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    coverage = pd.read_parquet(coverage_path)
    if report["formal_valuation_run_status"] != "NO_VIEW":
        raise RuntimeError("正式估值闸门意外放行，禁止冻结当前审计")
    if report["r5_replay"]["engineering_replay_status"] != "PASS_ENGINEERING_REPLAY_ONLY":
        raise RuntimeError("R5工程重放没有通过")
    if len(coverage) != 120:
        raise RuntimeError(f"月度覆盖表应为120行，实际{len(coverage)}行")
    if any(report["governance"].values()):
        raise RuntimeError("治理边界异常：审计过程中出现收益、IC、仓位、订单、券商连接或冻结文件变更")
    test_result = _run_tests()
    checkpoint_contract = config["data_contracts"]["financial_checkpoints"]
    checkpoint_hash, checkpoint_files = canonical_directory_hash(
        ROOT / checkpoint_contract["directory"]
    )
    if len(checkpoint_files) != int(checkpoint_contract["expected_file_count"]):
        raise RuntimeError("财务检查点数量偏离配置")
    payload = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo(config["protocol"]["timezone"])).isoformat(),
        "source_commit": _git_output("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_output("status", "--porcelain")),
        "frozen_files": {
            relative: sha256_file(ROOT / relative) for relative in FROZEN_FILES
        },
        "input_files": {
            row["file"]: row["actual_sha256"] for row in input_hashes["rows"]
        },
        "financial_checkpoint_directory": {
            "directory": checkpoint_contract["directory"],
            "file_count": len(checkpoint_files),
            "content_sha256": checkpoint_hash,
        },
        "output_files": {
            config["artifacts"]["snapshot_coverage"]: sha256_file(coverage_path),
            config["artifacts"]["report_json"]: sha256_file(report_path),
            config["artifacts"]["report_markdown"]: sha256_file(markdown_path),
        },
        "status_at_freeze": {
            "overall": report["overall_status"],
            "formal_valuation_run": report["formal_valuation_run_status"],
            "five_year_raw_ey": report["history_gates"]["five_year"]["raw_ey_status"],
            "five_year_normalized_ey": report["history_gates"]["five_year"]["normalized_ey_status"],
            "five_year_normalized_ey_spread": report["history_gates"]["five_year"]["normalized_ey_spread_status"],
            "seven_year_raw_ey": report["history_gates"]["seven_year"]["raw_ey_status"],
            "r5_engineering_replay": report["r5_replay"]["engineering_replay_status"],
            "r5_economic_evidence": report["r5_replay"]["economic_evidence_status"],
        },
        "governance": report["governance"],
        "test_result": test_result,
    }
    payload["manifest_content_sha256"] = _canonical_hash(payload)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
