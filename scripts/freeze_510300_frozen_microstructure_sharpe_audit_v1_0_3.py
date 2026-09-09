"""冻结V1.0.3完整布尔字面量机械修正。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "510300_frozen_microstructure_sharpe_audit_v1_0_3_correction.json"
MANIFEST = ROOT / "config" / "510300_frozen_microstructure_sharpe_audit_v1_0_3_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if MANIFEST.exists():
        raise FileExistsError("V1.0.3冻结清单已存在，禁止覆盖")
    correction = json.loads(CONFIG.read_text(encoding="utf-8"))
    parent_manifest = ROOT / correction["parent_manifest"]
    parent_implementation = ROOT / correction["parent_implementation"]
    if sha256_file(parent_manifest) != correction["parent_manifest_sha256"]:
        raise ValueError("V1.0.2父清单哈希漂移")
    if sha256_file(parent_implementation) != correction["parent_implementation_sha256"]:
        raise ValueError("V1.0.2父实现哈希漂移")
    result_path = ROOT / "reports" / "audit" / "510300_frozen_microstructure_sharpe_audit_v1.json"
    if result_path.exists():
        raise FileExistsError("结果已经存在，禁止覆盖")
    tracked = [
        "config/510300_frozen_microstructure_sharpe_audit_v1_0_3_correction.json",
        "config/510300_frozen_microstructure_sharpe_audit_v1_0_2_manifest.json",
        "research/frozen_microstructure_sharpe_audit_v1_0_2.py",
        "research/frozen_microstructure_sharpe_audit_v1_0_3.py",
        "scripts/freeze_510300_frozen_microstructure_sharpe_audit_v1_0_3.py",
        "scripts/run_510300_frozen_microstructure_sharpe_audit_v1_0_3.py",
        "reports/audit/510300_frozen_microstructure_sharpe_audit_v1_0_2_run_failure.json"
    ]
    for relative in tracked:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(f"V1.0.3冻结文件不存在：{relative}")
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    payload = {
        "project_id": correction["project_id"],
        "state": "FROZEN_MECHANICAL_COMPLETE_BOOLEAN_BINDING_AFTER_UNREAD_IN_MEMORY_METRICS",
        "implementation_frozen": True,
        "frozen_at_asia_shanghai": now,
        "metrics_computed_in_memory_before_freeze": True,
        "metric_values_read_by_operator_before_freeze": False,
        "result_preexisted_at_freeze": False,
        "permitted_change": correction["permitted_change"],
        "static_scan": correction["static_scan"],
        "tracked_files": {relative: sha256_file(ROOT / relative) for relative in tracked},
        "unchanged": correction["unchanged"],
        "live_trading_authorized": False,
    }
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("V1.0.3完整布尔字面量机械修正已冻结")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
