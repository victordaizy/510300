"""V1.11.1：继承目录修复，缺少成熟度资料时明确保留无法判断。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_priority_forward_codex_automation_v1_11 as parent

PARENT_HASH = "5137c85917ed292d71fee2a8c7b0543e7f4f810e2a50ad80959eb6918bd749f0"
MANIFEST = ROOT / "config/priority_forward_research_operations_v1_11_1_manifest.json"
ADDITIONS = {
    "config/priority_forward_research_operations_v1_11_manifest.json",
    "scripts/run_priority_forward_codex_automation_v1_11_1.py",
    "scripts/freeze_priority_forward_research_operations_v1_11_1.py",
    "scripts/install_priority_forward_morning_task_v1_11_1.ps1",
}
_original_collection_status = parent.collection_status


def verify_manifest(path: Path, expected_hash: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash):
        raise ValueError("必须提供固定的64位清单哈希")
    if parent.PATHS.checked(path) != MANIFEST or parent.digest(MANIFEST) != expected_hash:
        raise ValueError("V1.11.1清单路径或哈希不一致")
    parent.verify_manifest(parent.MANIFEST, PARENT_HASH)
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    previous = json.loads(parent.MANIFEST.read_text(encoding="utf-8"))
    required = {row["path"] for row in previous["files"]} | ADDITIONS
    expected = parent.records(sorted(required))
    if payload.get("manifest_id") != "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_11_1" or payload.get("predecessor_sha256") != PARENT_HASH:
        raise ValueError("V1.11.1清单身份或父版本不一致")
    if payload.get("files") != expected or payload.get("content_sha256") != parent.content_digest(expected):
        raise ValueError("V1.11.1受控文件不一致")
    return {
        "status": "PASS_V1_11_1_FROZEN_ENTRYPOINT_VERIFIED",
        "manifest_sha256": expected_hash,
        "predecessor_sha256": PARENT_HASH,
        "v1_10_predecessor_files_unchanged": 49,
        "tracked_file_count": len(expected),
        "collection_triggered": False,
        "data_purchase_budget_cny": 0,
        "position_impact": 0,
    }


def collection_status(now: datetime) -> dict:
    path = parent.PATHS.checked("reports/data_quality/510300_primary_market_readiness_v1_2.json")
    value = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if not isinstance(value, dict) or not isinstance(value.get("daily_quality"), list):
        return {
            "generated_at": now.isoformat(),
            "target_date": now.date().isoformat(),
            "status": "NO_VIEW_MISSING_OR_INCOMPLETE_READINESS",
            "strict_route_quality_dates_in_readiness": None,
            "strict_route_quality_days_in_readiness": None,
            "legacy_complete_quality_days_separate": None,
            "readiness_generated_at": value.get("generated_at") if isinstance(value, dict) else None,
            "data_purchase_budget_cny": 0,
            "position_impact": 0,
        }
    return _original_collection_status(now)


def main() -> int:
    parser = argparse.ArgumentParser(description="免费PCF采集V1.11.1入口")
    parser.add_argument("--phase", choices=("morning",), default="morning")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--config", type=Path, default=parent.CONFIG)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--now")
    args = parser.parse_args()
    if args.now and not args.dry_run:
        raise ValueError("指定时间仅限不采集的模拟检查")
    if parent.PATHS.checked(args.config) != parent.CONFIG:
        raise ValueError("仅接受继承的固定PCF编排配置")
    verified = verify_manifest(args.manifest, args.expected_manifest_sha256)
    if args.verify_only:
        print(json.dumps(verified, ensure_ascii=False, indent=2))
        return 0
    if args.status_only:
        print(json.dumps(collection_status(datetime.now(ZoneInfo("Asia/Shanghai"))), ensure_ascii=False, indent=2))
        return 0
    original_argv = sys.argv
    original_status = parent.collection_status
    sys.argv = [sys.argv[0], "--phase", "morning", "--config", str(parent.CONFIG), "--manifest", str(parent.MANIFEST), "--expected-manifest-sha256", PARENT_HASH]
    if args.dry_run:
        sys.argv.append("--dry-run")
    if args.now:
        sys.argv.extend(["--now", args.now])
    parent.collection_status = collection_status
    try:
        return parent.main()
    finally:
        sys.argv = original_argv
        parent.collection_status = original_status


if __name__ == "__main__":
    raise SystemExit(main())
