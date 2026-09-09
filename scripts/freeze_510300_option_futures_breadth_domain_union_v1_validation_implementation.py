"""在首次验证执行前冻结独立验证实现。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "research" / "validate_option_futures_breadth_domain_union_v1_frozen.py"
CONFIG = ROOT / "config" / "510300_option_futures_breadth_domain_union_v1.yaml"
MANIFEST = ROOT / "config" / "510300_option_futures_breadth_domain_union_v1_manifest.json"
FREEZE_RECEIPT = (
    ROOT / "reports" / "frozen" / "510300_option_futures_breadth_domain_union_v1_freeze_receipt.json"
)
OUTPUT = (
    ROOT
    / "reports"
    / "frozen"
    / "510300_option_futures_breadth_domain_union_v1_validation_implementation_receipt.json"
)
TIMEZONE = ZoneInfo("Asia/Shanghai")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    required = [VALIDATOR, CONFIG, MANIFEST, FREEZE_RECEIPT]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"验证冻结输入缺失：{missing}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    receipt = json.loads(FREEZE_RECEIPT.read_text(encoding="utf-8"))
    if sha256(CONFIG) != manifest["config_sha256"]:
        raise RuntimeError("配置哈希漂移")
    if sha256(MANIFEST) != receipt["manifest_sha256"]:
        raise RuntimeError("清单哈希漂移")
    if (ROOT / "reports" / "validation" / "510300_option_futures_breadth_domain_union_v1_validation.json").exists():
        raise RuntimeError("验证报告已存在，不能再声称首次执行前冻结实现")
    payload = {
        "status": "VALIDATION_IMPLEMENTATION_FROZEN_BEFORE_FIRST_RUN",
        "project_id": manifest["project_id"],
        "frozen_at": datetime.now(TIMEZONE).isoformat(),
        "validator_path": str(VALIDATOR.relative_to(ROOT)).replace("\\", "/"),
        "validator_sha256": sha256(VALIDATOR),
        "config_sha256": sha256(CONFIG),
        "manifest_sha256": sha256(MANIFEST),
        "freeze_receipt_sha256": sha256(FREEZE_RECEIPT),
        "validation_report_existed_before_freeze": False,
        "allowed_next_action": "仅运行此哈希对应实现一次，并接受通过或永久淘汰结果",
    }
    atomic_json(payload, OUTPUT)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
