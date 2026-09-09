"""冻结极端两级类马丁实现与输入，之后才允许计算收益。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "config" / "extreme_two_level_martin_v1_implementation_manifest.json"
PROTOCOL_MANIFEST = ROOT / "config" / "extreme_two_level_martin_v1_protocol_manifest.json"
IMPLEMENTATION_FILES = (
    "research/extreme_two_level_martin_v1.py",
    "scripts/run_extreme_two_level_martin_v1.py",
    "scripts/freeze_extreme_two_level_martin_implementation.py",
    "tests/test_extreme_two_level_martin_v1.py",
)
INPUT_FILES = (
    "config/graph_regime_martin_turtle_v2.yaml",
    "research/graph_regime_martin_turtle_v2.py",
    "reports/backtest/graph_regime_martin_turtle_v2.json",
    "data/raw/r6/510300_daily.parquet",
    "data/raw/r6/H00300_total_return_daily.parquet",
    "data/reference/510300_dividends.csv",
    "data/reference/510300_dividends_coverage.json",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    protocol = json.loads(PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
    for relative_path, expected in protocol["frozen_files"].items():
        if sha256(ROOT / relative_path) != expected:
            raise RuntimeError(f"协议冻结后发生变化：{relative_path}")
    missing = [
        path for path in (*IMPLEMENTATION_FILES, *INPUT_FILES) if not (ROOT / path).exists()
    ]
    if missing:
        raise FileNotFoundError(f"类马丁实现冻结文件缺失：{missing}")
    payload = {
        "project_id": protocol["project_id"],
        "version": protocol["version"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "state": "DISCOVERY_ONLY",
        "freeze_stage": "IMPLEMENTATION_FROZEN_HISTORICAL_TEST_ALLOWED",
        "protocol_manifest_sha256": sha256(PROTOCOL_MANIFEST),
        "candidate_ids": protocol["candidate_ids"],
        "return_calculation_allowed": True,
        "true_forward_start": None,
        "implementation_files": {
            path: sha256(ROOT / path) for path in IMPLEMENTATION_FILES
        },
        "input_files": {path: sha256(ROOT / path) for path in INPUT_FILES},
        "governance": protocol["governance"],
    }
    if MANIFEST.exists():
        old = json.loads(MANIFEST.read_text(encoding="utf-8"))
        stable = lambda item: {k: v for k, v in item.items() if k != "frozen_at"}
        if stable(old) != stable(payload):
            raise RuntimeError("类马丁实现指纹变化，禁止覆盖")
        print("类马丁实现已冻结，指纹一致。")
        return 0
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
