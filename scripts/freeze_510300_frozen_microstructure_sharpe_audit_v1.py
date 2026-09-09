"""冻结510300既有微观结构仓位的夏普审计合同。"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "510300_frozen_microstructure_sharpe_audit_v1.yaml"
MANIFEST = ROOT / "config" / "510300_frozen_microstructure_sharpe_audit_v1_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    if MANIFEST.exists():
        raise FileExistsError("冻结清单已存在，禁止覆盖")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    result_path = ROOT / config["paths"]["result"]
    if result_path.exists():
        raise FileExistsError("结果已存在，不能声称结果前冻结")

    tracked = [
        "config/510300_frozen_microstructure_sharpe_audit_v1.yaml",
        "research/frozen_microstructure_sharpe_audit_v1.py",
        "scripts/freeze_510300_frozen_microstructure_sharpe_audit_v1.py",
        "scripts/run_510300_frozen_microstructure_sharpe_audit_v1.py",
        "tests/test_frozen_microstructure_sharpe_audit_v1.py",
    ]
    input_files = [
        item["path"]
        for item in config["inputs"].values()
    ]
    for relative in tracked + input_files:
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(f"冻结文件不存在：{relative}")

    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    payload = {
        "project_id": config["protocol"]["project_id"],
        "state": "FROZEN_BEFORE_SHARPE_COMPUTATION",
        "implementation_frozen": True,
        "frozen_at_asia_shanghai": now,
        "result_preexisted_at_freeze": False,
        "evidence_class": config["protocol"]["evidence_class"],
        "config_sha256": sha256_file(CONFIG),
        "tracked_files": {relative: sha256_file(ROOT / relative) for relative in tracked},
        "input_files": {relative: sha256_file(ROOT / relative) for relative in input_files},
        "boundaries": {
            "candidate_already_historically_selected": True,
            "sharpe_metric_computed_before_this_freeze": False,
            "parameter_change_after_freeze": "forbidden",
            "date_change_after_freeze": "forbidden",
            "cost_change_after_freeze": "forbidden",
            "live_trading_authorized": False,
        },
    }
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    receipt_path = ROOT / config["paths"]["freeze_receipt"]
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "project_id": payload["project_id"],
        "status": "PASS_FROZEN_BEFORE_SHARPE_COMPUTATION",
        "frozen_at_asia_shanghai": now,
        "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_file_count": len(tracked),
        "input_file_count": len(input_files),
        "result_preexisted_at_freeze": False,
        "live_trading_authorized": False,
    }
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("审计协议已在夏普计算前冻结")
    print(f"清单：{MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
