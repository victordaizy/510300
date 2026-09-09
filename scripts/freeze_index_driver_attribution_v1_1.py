"""冻结 510300 点时指数驱动归因 V1.1。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FILE = ROOT / "config" / "index_driver_attribution_v1_1.yaml"
MANIFEST_FILE = ROOT / "config" / "index_driver_attribution_v1_1_manifest.json"
PARENT_MANIFEST_FILE = ROOT / "config" / "index_driver_attribution_v1_manifest.json"
FROZEN_FILES = (
    "docs/INDEX_DRIVER_ATTRIBUTION_V1_1_ADDENDUM.md",
    "config/index_driver_attribution_v1_1.yaml",
    "research/index_driver_attribution_v1_1.py",
    "scripts/build_index_driver_attribution_v1_1.py",
    "scripts/freeze_index_driver_attribution_v1_1.py",
    "tests/test_index_driver_attribution_v1_1.py",
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
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _stable_payload(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "frozen_at"}


def main() -> int:
    if not PARENT_MANIFEST_FILE.exists():
        raise FileNotFoundError("父版本 V1 冻结清单缺失")
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"V1.1 冻结文件缺失：{missing}")
    contract = yaml.safe_load(CONTRACT_FILE.read_text(encoding="utf-8"))
    governance = contract["governance"]
    forbidden_true = (
        "future_return_targets_defined",
        "future_return_reading_allowed",
        "predictive_threshold_search_allowed",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_authorized",
    )
    if any(governance[key] for key in forbidden_true):
        raise ValueError("V1.1 治理状态允许了禁止能力")
    industry_settings = contract["inputs"]["industry_intervals"]
    if industry_settings["fallback_taxonomy"] is not None:
        raise ValueError("V1.1 禁止行业分类回退补洞")
    if industry_settings["conflicting_l1_overlap_rule"] != "硬失败":
        raise ValueError("V1.1 的不同一级行业重叠必须硬失败")

    input_paths = {
        name: ROOT / settings["path"]
        for name, settings in contract["inputs"].items()
    }
    missing_inputs = [str(path) for path in input_paths.values() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"V1.1 冻结输入缺失：{missing_inputs}")
    payload = {
        "project_id": contract["protocol"]["project_id"],
        "version": contract["protocol"]["version"],
        "parent_version": contract["protocol"]["parent_version"],
        "parent_manifest_sha256": sha256(PARENT_MANIFEST_FILE),
        "state": contract["protocol"]["state"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_value("status", "--porcelain")),
        "data_start": contract["protocol"]["data_start"],
        "data_cutoff": contract["protocol"]["data_cutoff"],
        "frozen_files": {
            relative: sha256(ROOT / relative) for relative in FROZEN_FILES
        },
        "parent_frozen_files": json.loads(
            PARENT_MANIFEST_FILE.read_text(encoding="utf-8")
        )["frozen_files"],
        "input_files": {
            settings["path"]: sha256(input_paths[name])
            for name, settings in contract["inputs"].items()
        },
        "governance": {
            "future_return_reading_allowed": False,
            "predictive_threshold_search_allowed": False,
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }
    if MANIFEST_FILE.exists():
        existing = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if _stable_payload(existing) != _stable_payload(payload):
            raise RuntimeError("归因 V1.1 冻结指纹已变化；禁止覆盖")
        print("归因 V1.1 冻结清单已存在，指纹一致。")
        return 0
    MANIFEST_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

