"""冻结V1.0.1只读数组机械修正。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "510300_frozen_microstructure_sharpe_audit_v1_0_1_correction.json"
MANIFEST = ROOT / "config" / "510300_frozen_microstructure_sharpe_audit_v1_0_1_manifest.json"


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if MANIFEST.exists():
        raise FileExistsError("V1.0.1冻结清单已存在，禁止覆盖")
    correction = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_manifest = ROOT / correction["parent_manifest"]
    parent_implementation = ROOT / correction["parent_implementation"]
    if sha256_file(parent_manifest) != correction["parent_manifest_sha256"]:
        raise ValueError("父清单哈希漂移")
    if sha256_file(parent_implementation) != correction["parent_implementation_sha256"]:
        raise ValueError("父实现哈希漂移")
    result_path = ROOT / "reports" / "audit" / "510300_frozen_microstructure_sharpe_audit_v1.json"
    if result_path.exists():
        raise FileExistsError("结果已经存在，禁止声称修正在结果前冻结")

    tracked = [
        "config/510300_frozen_microstructure_sharpe_audit_v1_0_1_correction.json",
        "config/510300_frozen_microstructure_sharpe_audit_v1_manifest.json",
        "research/frozen_microstructure_sharpe_audit_v1.py",
        "research/frozen_microstructure_sharpe_audit_v1_0_1.py",
        "scripts/freeze_510300_frozen_microstructure_sharpe_audit_v1_0_1.py",
        "scripts/run_510300_frozen_microstructure_sharpe_audit_v1_0_1.py",
        "reports/audit/510300_frozen_microstructure_sharpe_audit_v1_run_failure.json"
    ]
    for relative in tracked:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(f"V1.0.1冻结文件不存在：{relative}")
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    payload = {
        "project_id": correction["project_id"],
        "state": "FROZEN_MECHANICAL_CORRECTION_BEFORE_METRICS",
        "implementation_frozen": True,
        "frozen_at_asia_shanghai": now,
        "metrics_computed_before_freeze": False,
        "result_preexisted_at_freeze": False,
        "permitted_change": correction["permitted_change"],
        "tracked_files": {relative: sha256_file(ROOT / relative) for relative in tracked},
        "unchanged": correction["unchanged"],
        "live_trading_authorized": False,
    }
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("V1.0.1机械修正已在指标计算前冻结")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
