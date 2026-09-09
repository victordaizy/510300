"""在读取真实DAILY_01特征分布和未来标签前冻结预测实现。"""

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

from research.daily_01_overnight_absorption_v1 import audit_and_load_inputs, load_config, sha256_file
from research.daily_01_overnight_absorption_v1_evaluation import load_evaluation_config, verify_parent_protocol


MANIFEST = ROOT / "config" / "daily_01_overnight_absorption_v1_evaluation_manifest.json"
FROZEN_FILES = [
    "docs/510300_DAILY_01_OVERNIGHT_ABSORPTION_V1_EVALUATION_ADDENDUM.md",
    "config/daily_01_overnight_absorption_v1_evaluation.yaml",
    "research/daily_01_overnight_absorption_v1_evaluation.py",
    "scripts/run_daily_01_overnight_absorption_v1_evaluation.py",
    "scripts/freeze_daily_01_overnight_absorption_v1_evaluation.py",
    "tests/test_daily_01_overnight_absorption_v1_evaluation.py",
]


def _git_output(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    evaluation_config = load_evaluation_config()
    parent_manifest = verify_parent_protocol(ROOT, evaluation_config)
    _, _, audit = audit_and_load_inputs(ROOT, load_config())
    if audit["status"] != "PASS":
        raise RuntimeError("数据闸门未通过，禁止冻结预测实现")
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    payload = {
        "project_id": evaluation_config["evaluation_protocol"]["project_id"],
        "version": evaluation_config["evaluation_protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_output("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_output("status", "--porcelain")),
        "freeze_stage": "PREDICTIVE_IMPLEMENTATION_FROZEN_BEFORE_REAL_FEATURE_OR_OUTCOME_RUN",
        "parent_protocol_manifest": evaluation_config["evaluation_protocol"]["parent_protocol_manifest"],
        "parent_protocol_manifest_sha256": evaluation_config["evaluation_protocol"]["parent_protocol_manifest_sha256"],
        "parent_manifest_content_sha256": parent_manifest["manifest_content_sha256"],
        "frozen_files": {relative: sha256_file(ROOT / relative) for relative in FROZEN_FILES},
        "data_gate_status_at_freeze": audit["status"],
        "real_feature_values_seen_at_freeze": False,
        "predictive_outcomes_seen_at_freeze": False,
        "historical_run_completed": False,
        "one_historical_run_allowed": True,
        "strategy_backtest_authorized": False,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
    }
    payload["manifest_content_sha256"] = _canonical_hash(payload)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
