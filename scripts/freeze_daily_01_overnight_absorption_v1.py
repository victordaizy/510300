"""在任何未来收益计算前冻结DAILY_01协议、公式、测试与数据哈希。"""

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

from research.daily_01_overnight_absorption_v1 import (
    audit_and_load_inputs,
    load_config,
    sha256_file,
)


MANIFEST = ROOT / "config" / "daily_01_overnight_absorption_v1_protocol_manifest.json"
FROZEN_FILES = [
    "docs/510300_DAILY_01_OVERNIGHT_ABSORPTION_V1_SPEC.md",
    "config/daily_01_overnight_absorption_v1.yaml",
    "research/daily_01_overnight_absorption_v1.py",
    "scripts/audit_daily_01_overnight_absorption_v1.py",
    "scripts/freeze_daily_01_overnight_absorption_v1.py",
    "tests/test_daily_01_overnight_absorption_v1.py",
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


def main() -> int:
    config = load_config()
    _, _, audit = audit_and_load_inputs(ROOT, config)
    if audit["status"] != "PASS":
        raise RuntimeError("DAILY_01数据闸门未通过，禁止冻结协议")
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")

    data_files = {
        contract["file"]: contract["sha256"]
        for contract in config["data_contracts"].values()
    }
    for relative, expected in data_files.items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"数据哈希偏离冻结配置：{relative}")

    payload = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_output("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_output("status", "--porcelain")),
        "freeze_stage": "PROTOCOL_AND_FEATURE_FORMULA_FROZEN_BEFORE_OUTCOME_TEST",
        "historical_evidence_label": "HISTORICAL_DISCOVERY_ONLY",
        "historical_contamination_cutoff": config["protocol"]["historical_contamination_cutoff"],
        "historical_evaluation_cutoff": config["protocol"]["historical_evaluation_cutoff"],
        "true_forward_start": None,
        "registered_candidate_count": config["trial_registry"]["registered_candidate_count"],
        "frozen_files": {relative: sha256_file(ROOT / relative) for relative in FROZEN_FILES},
        "data_files": data_files,
        "data_gate_status_at_freeze": audit["status"],
        "feature_values_computed_on_real_data": False,
        "predictive_outcomes_computed": False,
        "strategy_backtest_computed": False,
        "governance": {
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }
    payload["manifest_content_sha256"] = _canonical_hash(payload)
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(MANIFEST)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
