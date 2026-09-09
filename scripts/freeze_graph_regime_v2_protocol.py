"""冻结图形状态马丁-海龟V2协议，不运行收益。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "graph_regime_martin_turtle_v2.yaml"
MANIFEST = ROOT / "config" / "graph_regime_martin_turtle_v2_protocol_manifest.json"
FROZEN_FILES = (
    "docs/510300_GRAPH_REGIME_MARTIN_TURTLE_V2_SPEC.md",
    "config/graph_regime_martin_turtle_v2.yaml",
    "scripts/build_510300_complete_dividends.py",
    "scripts/freeze_graph_regime_v2_protocol.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate(config: dict) -> None:
    protocol = config["protocol"]
    candidates = config["candidates"]
    ids = [candidate["id"] for candidate in candidates]
    expected = [
        "T1_TURTLE_GRAPH_EXIT",
        "M0_CAPPED_MARTINGALE_ONLY",
        "P0_BIAS28_SWING_ONLY",
        "S1_MARTINGALE_TURTLE_SWITCH",
        "S2_BIAS28_TURTLE_SWITCH",
    ]
    if protocol["project_id"] != "510300_GRAPH_REGIME_MARTIN_TURTLE_DISCOVERY_V2":
        raise ValueError("V2项目ID错误")
    if ids != expected or len(ids) != int(protocol["registered_candidate_count"]):
        raise ValueError("V2候选集合或顺序发生变化")
    if protocol["true_forward_start"] is not None:
        raise ValueError("真正前向起点必须保持null")
    if config["martingale_module"]["maximum_layers"] != 3:
        raise ValueError("马丁格尔必须封顶为三层")
    if config["martingale_module"]["unlimited_doubling_allowed"] is not False:
        raise ValueError("禁止无限翻倍")
    governance = config["governance"]
    for key in (
        "options_enabled",
        "futures_enabled",
        "leverage_enabled",
        "shorting_enabled",
        "human_retrospective_segmentation_enabled",
        "machine_learning_image_classifier_enabled",
        "live_position_mapping_enabled",
        "order_generation_enabled",
        "broker_connection_enabled",
        "live_trading_authorized",
    ):
        if governance[key] is not False:
            raise ValueError(f"治理开关必须关闭：{key}")


def _stable(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "frozen_at"}


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate(config)
    missing = [path for path in FROZEN_FILES if not (ROOT / path).exists()]
    if missing:
        raise FileNotFoundError(f"V2冻结文件缺失：{missing}")
    payload = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_value("status", "--porcelain")),
        "state": "DISCOVERY_ONLY",
        "freeze_stage": config["protocol"]["freeze_stage"],
        "candidate_ids": [item["id"] for item in config["candidates"]],
        "external_frozen_baseline": config["protocol"]["external_frozen_baseline"],
        "true_forward_start": None,
        "return_calculation_allowed_at_protocol_freeze": False,
        "frozen_files": {path: sha256(ROOT / path) for path in FROZEN_FILES},
        "governance": {
            "modifies_v1": False,
            "modifies_r5": False,
            "reopens_r6": False,
            "creates_r7": False,
            "live_position_mapping_enabled": False,
            "order_generation_enabled": False,
            "broker_connection_enabled": False,
        },
    }
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if _stable(existing) != _stable(payload):
            raise RuntimeError("V2协议指纹已变化，禁止覆盖")
        print("V2协议冻结清单已存在，指纹一致。")
        return 0
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

