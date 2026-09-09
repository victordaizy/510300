"""冻结510300唐奇安发现协议；不会运行数据收益或生成仓位。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.donchian_discovery_data_audit import validate_registry


MANIFEST_FILE = ROOT / "config" / "donchian_discovery_v1_manifest.json"
REGISTRY_FILE = ROOT / "config" / "donchian_discovery_v1.yaml"
FROZEN_FILES = (
    "docs/510300_DONCHIAN_DISCOVERY_V1_SPEC.md",
    "config/donchian_discovery_v1.yaml",
    "research/donchian_discovery_data_audit.py",
    "scripts/audit_donchian_discovery_data.py",
    "scripts/freeze_donchian_discovery_protocol.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _stable_payload(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "frozen_at"}


def build_payload() -> dict:
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    validation = validate_registry(registry)
    contracts = registry["data_contracts"]
    return {
        "project_id": validation["project_id"],
        "version": registry["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_value("status", "--porcelain")),
        "state": "DISCOVERY_ONLY",
        "freeze_stage": "PROTOCOL_FROZEN_DATA_GATE_PENDING",
        "candidate_count": validation["candidate_count"],
        "candidate_ids": validation["candidate_ids"],
        "historical_evidence_label": registry["protocol"][
            "historical_evidence_label"
        ],
        "historical_contamination_cutoff": registry["protocol"][
            "historical_contamination_cutoff"
        ],
        "true_forward_start": None,
        "return_calculation_allowed_at_freeze": False,
        "return_implementation_frozen": False,
        "data_snapshot_hashes": {
            "market": contracts["market"]["frozen_snapshot_sha256"],
            "benchmark": contracts["benchmark"]["frozen_snapshot_sha256"],
            "distribution_complete_snapshot": contracts["distributions"][
                "frozen_snapshot_sha256"
            ],
            "distribution_observed_incomplete_snapshot": contracts[
                "distributions"
            ]["observed_incomplete_snapshot_sha256"],
        },
        "frozen_files": {relative: sha256(ROOT / relative) for relative in FROZEN_FILES},
        "governance": {
            "modifies_r5": False,
            "reopens_r6": False,
            "creates_r7": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结510300唐奇安发现协议")
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验现有冻结清单，不创建文件",
    )
    args = parser.parse_args()
    payload = build_payload()
    if MANIFEST_FILE.exists():
        existing = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if _stable_payload(existing) != _stable_payload(payload):
            raise RuntimeError("冻结指纹已变化；禁止覆盖，请创建下一协议版本")
        print("唐奇安发现协议冻结清单已存在，指纹一致。")
        return 0
    if args.check:
        raise FileNotFoundError(f"冻结清单不存在：{MANIFEST_FILE}")
    MANIFEST_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
