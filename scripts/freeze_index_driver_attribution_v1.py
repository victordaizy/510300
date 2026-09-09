"""冻结 510300 点时指数驱动归因 V1；不会读取未来收益。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FILE = ROOT / "config" / "index_driver_attribution_v1.yaml"
MANIFEST_FILE = ROOT / "config" / "index_driver_attribution_v1_manifest.json"
FROZEN_FILES = (
    "CONTEXT.md",
    "docs/INDEX_DRIVER_ATTRIBUTION_V1_SPEC.md",
    "config/index_driver_attribution_v1.yaml",
    "research/index_driver_attribution.py",
    "scripts/build_index_driver_attribution_v1.py",
    "scripts/freeze_index_driver_attribution_v1.py",
    "tests/test_index_driver_attribution.py",
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


def main() -> int:
    missing = [relative for relative in FROZEN_FILES if not (ROOT / relative).exists()]
    if missing:
        raise FileNotFoundError(f"冻结文件缺失：{missing}")
    contract = yaml.safe_load(CONTRACT_FILE.read_text(encoding="utf-8"))
    governance = contract["governance"]
    if governance["future_return_targets_defined"]:
        raise ValueError("归因 V1 禁止定义未来收益目标")
    if governance["future_return_reading_allowed"]:
        raise ValueError("归因 V1 禁止读取未来收益")
    for key in (
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_authorized",
    ):
        if governance[key]:
            raise ValueError(f"归因 V1 安全状态非法：{key}=true")
    if contract["inputs"]["industry_intervals"]["fallback_taxonomy"] is not None:
        raise ValueError("归因 V1 禁止行业分类回退补洞")

    input_paths = {
        name: ROOT / settings["path"]
        for name, settings in contract["inputs"].items()
    }
    missing_inputs = [str(path) for path in input_paths.values() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError(f"冻结输入缺失：{missing_inputs}")

    payload = {
        "project_id": contract["protocol"]["project_id"],
        "version": contract["protocol"]["version"],
        "state": contract["protocol"]["state"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_value("status", "--porcelain")),
        "data_start": contract["protocol"]["data_start"],
        "data_cutoff": contract["protocol"]["data_cutoff"],
        "frozen_files": {
            relative: sha256(ROOT / relative) for relative in FROZEN_FILES
        },
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
            raise RuntimeError("归因 V1 冻结指纹已变化；禁止覆盖，请创建下一版本")
        print("归因 V1 冻结清单已存在，指纹一致。")
        return 0
    MANIFEST_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

