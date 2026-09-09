"""冻结TECH_03模型、实现和数据哈希；冻结后才能计算历史收益。"""

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

from research.tech_03_donchian_55_20_v1 import (
    audit_and_load_inputs,
    load_config,
    sha256_file,
)


MANIFEST = ROOT / "config" / "tech_03_donchian_55_20_v1_manifest.json"
FROZEN_FILES = [
    "docs/TECH_03_DONCHIAN_55_20_V1_SPEC.md",
    "config/tech_03_donchian_55_20_v1.yaml",
    "research/tech_03_donchian_55_20_v1.py",
    "scripts/audit_tech_03_donchian_55_20_v1.py",
    "scripts/run_tech_03_donchian_55_20_v1.py",
    "scripts/freeze_tech_03_donchian_55_20_v1.py",
    "tests/test_tech_03_donchian_55_20_v1.py",
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
        raise RuntimeError("数据闸门未通过，不能冻结收益实现")
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    data_files = {
        contract["file"]: contract["sha256"]
        for key, contract in config["data_contracts"].items()
        if key != "distributions"
    }
    distribution = config["data_contracts"]["distributions"]
    data_files[distribution["file"]] = distribution["sha256"]
    data_files[distribution["coverage_file"]] = distribution["coverage_sha256"]
    for relative, expected in data_files.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise ValueError(f"数据哈希偏离冻结配置：{relative}")
    payload = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_output("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_output("status", "--porcelain")),
        "implementation_frozen": True,
        "historical_evidence_label": "HISTORICALLY_CONTAMINATED",
        "historical_contamination_cutoff": config["protocol"]["historical_contamination_cutoff"],
        "true_forward_start": None,
        "registered_trial_count": config["trial_registry"]["registered_trial_count"],
        "return_tested_trial_count": config["trial_registry"]["return_tested_trial_count_after_run"],
        "frozen_files": {
            relative: sha256_file(ROOT / relative) for relative in FROZEN_FILES
        },
        "data_files": data_files,
        "data_gate_status_at_freeze": audit["status"],
        "governance": {
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
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
