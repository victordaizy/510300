"""在读取新历史结果前冻结 510300 驱动周期发现协议 V1。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_FILE = ROOT / "config" / "index_driver_episode_discovery_v1.yaml"
MANIFEST_FILE = ROOT / "config" / "index_driver_episode_discovery_v1_manifest.json"
PARENT_MANIFEST_FILE = ROOT / "config" / "index_driver_attribution_v1_3_manifest.json"
FROZEN_FILES = (
    "CONTEXT.md",
    "docs/INDEX_DRIVER_EPISODE_DISCOVERY_V1_SPEC.md",
    "config/index_driver_episode_discovery_v1.yaml",
    "research/index_driver_episode.py",
    "scripts/freeze_index_driver_episode_v1.py",
    "scripts/run_index_driver_episode_v1.py",
    "tests/test_index_driver_episode.py",
)


def sha256(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _validate_contract(contract: dict) -> None:
    protocol = contract["protocol"]
    governance = contract["governance"]
    if protocol["candidate_budget"] != 2:
        raise ValueError("驱动周期 V1 候选预算必须严格等于 2")
    if protocol["true_forward_start"] is not None:
        raise ValueError("历史发现冻结时 true_forward_start 必须为空")
    if not protocol["candidate_addition_after_freeze_forbidden"]:
        raise ValueError("冻结后必须禁止追加候选")
    if not protocol["threshold_change_after_freeze_forbidden"]:
        raise ValueError("冻结后必须禁止修改阈值")
    forbidden_true = {
        "historical_run_may_emit_forecast_eligible",
        "valuation_may_trigger_entry",
        "r6_reselection_allowed",
        "machine_learning_allowed",
        "predictive_threshold_search_allowed",
        "position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_authorized",
    }
    enabled = sorted(key for key in forbidden_true if governance.get(key) is True)
    if enabled:
        raise ValueError(f"研究安全开关不得启用：{enabled}")
    if governance["missing_input_output"] != "NO_VIEW":
        raise ValueError("缺失输入的唯一输出必须是 NO_VIEW")


def main() -> int:
    if not CONTRACT_FILE.exists():
        raise FileNotFoundError(CONTRACT_FILE)
    if not PARENT_MANIFEST_FILE.exists():
        raise FileNotFoundError("缺少已经冻结的归因 V1.3 清单")
    contract = yaml.safe_load(CONTRACT_FILE.read_text(encoding="utf-8"))
    _validate_contract(contract)

    missing_frozen = [path for path in FROZEN_FILES if not (ROOT / path).exists()]
    if missing_frozen:
        raise FileNotFoundError(f"冻结文件缺失：{missing_frozen}")
    input_paths = [str(value) for value in contract["inputs"].values()]
    missing_inputs = [path for path in input_paths if not (ROOT / path).exists()]
    if missing_inputs:
        raise FileNotFoundError(f"冻结输入缺失：{missing_inputs}")

    test = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_index_driver_episode.py", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if test.returncode != 0:
        raise RuntimeError(
            "驱动周期冻结测试失败：\n" + test.stdout + "\n" + test.stderr
        )

    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    manifest = {
        "project_id": contract["protocol"]["project_id"],
        "version": contract["protocol"]["version"],
        "state": "FROZEN_BEFORE_HISTORICAL_EVALUATION",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "source_worktree_dirty": bool(status.stdout.strip()),
        "test_command": f"{sys.executable} -m pytest tests/test_index_driver_episode.py -q",
        "test_result": test.stdout.strip(),
        "frozen_files": {path: sha256(ROOT / path) for path in FROZEN_FILES},
        "input_files": {path: sha256(ROOT / path) for path in input_paths},
        "parent_attribution_manifest": {
            "path": str(PARENT_MANIFEST_FILE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(PARENT_MANIFEST_FILE),
        },
        "governance": contract["governance"],
    }
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(MANIFEST_FILE)
    print(
        json.dumps(
            {
                "状态": manifest["state"],
                "冻结文件数": len(manifest["frozen_files"]),
                "冻结输入数": len(manifest["input_files"]),
                "测试": manifest["test_result"],
                "清单": str(MANIFEST_FILE),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

