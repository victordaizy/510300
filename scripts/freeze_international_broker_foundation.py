"""冻结国际化券商V1数据底座；不会批准收益检验、仓位或订单。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.international_broker_data_contract import run_audit  # noqa: E402


MANIFEST_FILE = ROOT / "config" / "international_broker_foundation_manifest.json"
FROZEN_FILES = (
    "docs/INTERNATIONAL_BROKER_RESEARCH_SPEC.md",
    "config/international_broker_research.yaml",
    "data/reference/international_broker/broker_universe.csv",
    "data/reference/international_broker/source_registry.csv",
    "data/reference/international_broker/international_broker_events_seed.csv",
    "research/international_broker_data_contract.py",
    "scripts/audit_international_broker_data.py",
    "scripts/build_international_broker_event_table.py",
    "scripts/freeze_international_broker_foundation.py",
    "tests/test_international_broker_data_contract.py",
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
    report = run_audit(write_reports=True)
    if report["return_test_allowed"] is not False:
        raise RuntimeError("底座冻结时收益检验必须保持禁止")
    payload = {
        "project_id": report["project_id"],
        "version": "1.0.0",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "source_commit": _git_value("rev-parse", "HEAD"),
        "worktree_dirty_at_freeze": bool(_git_value("status", "--porcelain")),
        "state": report["project_state"],
        "safety_state": report["safety_state"],
        "isolated_from_510300": report["isolated_from_510300"],
        "return_test_allowed": False,
        "universe_status": report["universe"]["status"],
        "evidence_registry_status": report["evidence_registry"]["status"],
        "event_table_status": report["tables"]["international_broker_events"][
            "status"
        ],
        "current_gate": report["gates"]["G1_UNIVERSE"],
        "frozen_files": {
            relative: sha256(ROOT / relative) for relative in FROZEN_FILES
        },
        "explicit_non_authorizations": [
            "NO_510300_ALPHA_INPUT",
            "NO_POSITION_MAPPING",
            "NO_ORDER_GENERATION",
            "NO_BROKER_CONNECTION",
            "NO_RETURN_TEST_UNTIL_NEXT_APPROVED_PROTOCOL",
        ],
    }
    if MANIFEST_FILE.exists():
        existing = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        if _stable_payload(existing) != _stable_payload(payload):
            raise RuntimeError("冻结指纹已变化；禁止覆盖，请创建下一协议版本")
        print("国际化券商数据底座冻结清单已存在，指纹一致。")
        return 0
    MANIFEST_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

