"""冻结510300波动压缩变盘事件研究协议，不计算结果。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "volatility_compression_change_point_v1.yaml"
MANIFEST = ROOT / "config" / "volatility_compression_change_point_v1_protocol_manifest.json"
FILES = (
    "config/volatility_compression_change_point_v1.yaml",
    "docs/510300_VOLATILITY_COMPRESSION_CHANGE_POINT_V1_SPEC.md",
    "scripts/freeze_volatility_compression_protocol.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["protocol"]["true_forward_start"] is not None:
        raise ValueError("真正前向起点必须为null")
    if config["definitions"]["event_selection"]["cooldown_trading_days"] != 20:
        raise ValueError("事件冷却期必须固定为20日")
    if config["matched_control"]["repetitions"] != 5000:
        raise ValueError("匹配对照必须固定5000次")
    payload = {
        "project_id": config["protocol"]["project_id"],
        "version": config["protocol"]["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "state": "DISCOVERY_ONLY",
        "freeze_stage": "PROTOCOL_FROZEN_IMPLEMENTATION_PENDING",
        "true_forward_start": None,
        "return_or_outcome_calculation_allowed": False,
        "frozen_files": {path: sha256(ROOT / path) for path in FILES},
        "governance": config["governance"],
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        stable = lambda item: {k: v for k, v in item.items() if k != "frozen_at"}
        if stable(old) != stable(payload):
            raise RuntimeError("波动压缩协议指纹变化，禁止覆盖")
        print("波动压缩协议已冻结，指纹一致。")
        return 0
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
