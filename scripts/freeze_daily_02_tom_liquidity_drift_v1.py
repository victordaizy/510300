"""在读取真实换月收益前冻结DAILY_02协议、实现和输入哈希。"""

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

from research.daily_02_tom_liquidity_drift_v1 import audit_and_load_inputs, load_config, sha256_file


MANIFEST = ROOT / "config" / "daily_02_tom_liquidity_drift_v1_manifest.json"
FROZEN_FILES = [
    "docs/510300_DAILY_02_TOM_LIQUIDITY_DRIFT_V1_SPEC.md",
    "config/daily_02_tom_liquidity_drift_v1.yaml",
    "research/daily_02_tom_liquidity_drift_v1.py",
    "scripts/freeze_daily_02_tom_liquidity_drift_v1.py",
    "scripts/run_daily_02_tom_liquidity_drift_v1.py",
    "tests/test_daily_02_tom_liquidity_drift_v1.py",
]


def canonical_hash(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_output(*arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    if MANIFEST.exists():
        raise FileExistsError("DAILY_02冻结清单已存在，禁止覆盖")
    config = load_config()
    _, _, _, audit = audit_and_load_inputs(ROOT, config)
    if audit["status"] != "PASS":
        raise RuntimeError(f"数据闸门未通过，禁止冻结：{audit['errors']}")
    missing = [path for path in FROZEN_FILES if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    data_files = {
        contract["file"]: contract["sha256"]
        for contract in config["data_contracts"].values()
    }
    payload = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": git_output("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(git_output("status", "--porcelain")),
        "freeze_stage": "PROTOCOL_IMPLEMENTATION_AND_DATA_FROZEN_BEFORE_REAL_EVENT_RETURN_RUN",
        "historical_evidence_label": "HISTORICALLY_CONTAMINATED_DISCOVERY_ONLY",
        "registered_candidate_count": 1,
        "frozen_files": {path: sha256_file(ROOT / path) for path in FROZEN_FILES},
        "data_files": data_files,
        "data_gate_status_at_freeze": audit["status"],
        "date_only_episode_support_seen": audit["complete_event_episodes"],
        "real_event_returns_seen_at_freeze": False,
        "strategy_backtest_seen_at_freeze": False,
        "historical_run_completed": False,
        "one_historical_run_allowed": True,
        "governance": config["governance"],
    }
    payload["manifest_content_sha256"] = canonical_hash(payload)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
