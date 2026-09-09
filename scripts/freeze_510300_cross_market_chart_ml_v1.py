"""在下载训练数据和查看510300结果前冻结跨市场模型协议与实现。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "510300_cross_market_chart_ml_v1.yaml"
MANIFEST = ROOT / "config" / "510300_cross_market_chart_ml_v1_manifest.json"
FILES = [
    "config/510300_cross_market_chart_ml_v1.yaml",
    "docs/510300_CROSS_MARKET_CHART_ML_V1_SPEC.md",
    "scripts/collect_510300_cross_market_chart_ml_v1.py",
    "research/cross_market_chart_ml_v1.py",
    "scripts/run_510300_cross_market_chart_ml_v1.py",
    "scripts/freeze_510300_cross_market_chart_ml_v1.py",
    "tests/test_510300_cross_market_chart_ml_v1.py",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if MANIFEST.exists():
        raise RuntimeError("冻结清单已存在，禁止覆盖")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    target_report = ROOT / config["outputs"]["target_report_json"]
    if target_report.exists():
        raise RuntimeError("目标结果已经存在，不能再声称揭盲前冻结")
    hashes = {relative: _sha256(ROOT / relative) for relative in FILES}
    manifest = {
        "project_id": config["protocol"]["project_id"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "stage": "V1_0_1_SAMPLE_COUNT_FEASIBILITY_CORRECTED_BEFORE_FIRST_MODEL_FIT_AND_TARGET_REVEAL",
        "factor_count": len(config["factors"]["definitions"]),
        "target_reveal_count_at_freeze": 0,
        "files": hashes,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已冻结：{MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
