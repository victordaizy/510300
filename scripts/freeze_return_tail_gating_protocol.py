"""一次性冻结尾部风险门控发现协议；不会启动前向、仓位或订单。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = ROOT / "config" / "return_tail_gating_manifest.json"
REGISTRY_FILE = ROOT / "config" / "return_tail_hypothesis_registry.yaml"
FROZEN_FILES = (
    "docs/RETURN_TAIL_GATING_RESEARCH_SPEC.md",
    "config/return_tail_hypothesis_registry.yaml",
    "research/return_tail_data_feasibility.py",
    "scripts/audit_return_tail_gating_data.py",
    "scripts/freeze_return_tail_gating_protocol.py",
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
    registry = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    candidate_count = (
        len(registry["hypotheses"])
        + len(registry["combination_models"])
        + 1
    )
    if candidate_count != int(registry["protocol"]["candidate_budget"]):
        raise ValueError("候选数量与冻结预算不一致")
    if registry["protocol"]["true_forward_start"] is not None:
        raise ValueError("数据可行性尚未通过时，true_forward_start必须为null")
    dirty = bool(_git_value("status", "--porcelain"))
    payload = {
        "project_id": registry["protocol"]["project_id"],
        "version": registry["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": dirty,
        "candidate_count": candidate_count,
        "state": "DISCOVERY_ONLY",
        "true_forward_start": None,
        "forward_start_reason": (
            "数据可行性、实现代码提交和可追加前向采集器尚未全部通过；不得自动使用2026-08-17。"
        ),
        "frozen_files": {relative: sha256(ROOT / relative) for relative in FROZEN_FILES},
        "governance": {
            "position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
            "live_trading_authorized": False,
        },
    }
    if MANIFEST_FILE.exists():
        existing = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if _stable_payload(existing) != _stable_payload(payload):
            raise RuntimeError("冻结指纹已变化；禁止覆盖，请创建下一协议版本")
        print("尾部风险门控冻结清单已存在，指纹一致。")
        return 0
    MANIFEST_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

