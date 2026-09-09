"""一次性登记成熟度资料缺失的状态修正，保留V1.11原清单。"""

from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from scripts import run_priority_forward_codex_automation_v1_11_1 as entry


def main() -> int:
    entry.parent.verify_manifest(entry.parent.MANIFEST, entry.PARENT_HASH)
    previous = json.loads(entry.parent.MANIFEST.read_text(encoding="utf-8"))
    rows = entry.parent.records(sorted({row["path"] for row in previous["files"]} | entry.ADDITIONS))
    payload = {
        "manifest_id": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_11_1",
        "status": "FROZEN_OPERATIONAL_REPAIR_READY",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "supersedes_runtime_manifest": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_11",
        "predecessor_sha256": entry.PARENT_HASH,
        "change": "MISSING_READINESS_IS_NO_VIEW_NOT_ZERO",
        "collection_protocol_changed": False,
        "data_purchase_budget_cny": 0,
        "position_impact": 0,
        "files": rows,
        "content_sha256": entry.parent.content_digest(rows),
    }
    descriptor = os.open(entry.MANIFEST, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(entry.verify_manifest(entry.MANIFEST, entry.parent.digest(entry.MANIFEST)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
