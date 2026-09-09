"""冻结极端两级类马丁协议，不计算收益。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "extreme_two_level_martin_v1.yaml"
MANIFEST = ROOT / "config" / "extreme_two_level_martin_v1_protocol_manifest.json"
FILES = (
    "config/extreme_two_level_martin_v1.yaml",
    "docs/510300_EXTREME_TWO_LEVEL_MARTIN_V1_SPEC.md",
    "scripts/freeze_extreme_two_level_martin_protocol.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    module = config["module"]
    if module["target_exposures"] != [0.50, 1.00]:
        raise ValueError("仓位必须冻结为50%和100%")
    if "-2*" not in module["second_layer_trigger"]:
        raise ValueError("第二层必须冻结为下跌2ATR")
    if "-3*" not in module["hard_invalidation"]:
        raise ValueError("硬失效必须冻结为下跌3ATR")
    if config["protocol"]["true_forward_start"] is not None:
        raise ValueError("真正前向起点必须为null")
    payload = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "state": "DISCOVERY_ONLY",
        "freeze_stage": "PROTOCOL_FROZEN_IMPLEMENTATION_PENDING",
        "candidate_ids": [item["id"] for item in config["candidates"]],
        "true_forward_start": None,
        "return_calculation_allowed": False,
        "frozen_files": {path: sha256(ROOT / path) for path in FILES},
        "governance": config["governance"],
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        stable = lambda item: {k: v for k, v in item.items() if k != "frozen_at"}
        if stable(old) != stable(payload):
            raise RuntimeError("类马丁协议指纹变化，禁止覆盖")
        print("类马丁协议已冻结，指纹一致。")
        return 0
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
