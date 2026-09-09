"""一次性登记目录兼容版本，拒绝覆盖原清单或接纳旧受控文件变动。"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from scripts import run_priority_forward_codex_automation_v1_11 as entry


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结免费 PCF/IOPV 目录兼容修复")
    parser.add_argument("--mode", choices=("freeze", "verify"), required=True)
    args = parser.parse_args()
    if args.mode == "verify":
        print(json.dumps(entry.verify_manifest(entry.MANIFEST, entry.digest(entry.MANIFEST)), ensure_ascii=False, indent=2))
        return 0
    base = entry.verified_predecessor()
    rows = entry.records(sorted({row["path"] for row in base["files"]} | entry.REQUIRED_ADDITIONS))
    payload = {
        "schema_version": "1.11.0",
        "manifest_id": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_11",
        "status": "FROZEN_OPERATIONAL_REPAIR_READY",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "supersedes_runtime_manifest": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_10",
        "predecessor_sha256": entry.BASE_HASH,
        "repair_scope": "APPROVED_DATA_JUNCTION_MANIFEST_AND_RAW_STORAGE_PATHS",
        "approved_resolved_data_root": str(entry.PATHS.approved_data_root),
        "active_execution_phase": "morning_pcf_only",
        "data_purchase_budget_cny": 0,
        "research_protocol_changed": False,
        "maturity_thresholds_changed": False,
        "position_impact": 0,
        "pcf_runtime": base["pcf_runtime"],
        "storage_compatible_collector": "scripts/collect_510300_primary_market_v1_4.py",
        "supervisor_config": "config/priority_forward_supervisor_v1_7.yaml",
        "same_day_claim_directory_shared_with_predecessor": "reports/audit/priority_forward_task_claims_v1_10",
        "deduplication_evidence_contract": base["deduplication_evidence_contract"],
        "deployment_evidence_is_separate": True,
        "files": rows,
        "content_sha256": entry.content_digest(rows),
    }
    descriptor = os.open(entry.MANIFEST, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(entry.verify_manifest(entry.MANIFEST, entry.digest(entry.MANIFEST)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
