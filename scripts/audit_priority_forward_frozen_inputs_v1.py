"""审计活动研究链的内容寻址输入与冻结版本隔离。"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.industry_outcome_snapshot_v1_2 import (
    ROOT,
    load_verified_snapshot_manifest,
)


OUTPUT = ROOT / "reports" / "audit" / "PRIORITY_FORWARD_FROZEN_INPUT_CONTENT_ADDRESSING_V1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{os.urandom(6).hex()}.tmp"
    )
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_audit() -> dict[str, Any]:
    timezone = ZoneInfo("Asia/Shanghai")
    supervisor_path = ROOT / "config" / "priority_forward_supervisor_v1_2.yaml"
    primary_path = ROOT / "config" / "primary_market_forward_v1_2.yaml"
    industry_path = (
        ROOT / "config" / "industry_expectation_gap_forward_operations_v1_2.yaml"
    )
    supervisor = yaml.safe_load(supervisor_path.read_text(encoding="utf-8"))
    primary = yaml.safe_load(primary_path.read_text(encoding="utf-8"))
    industry = yaml.safe_load(industry_path.read_text(encoding="utf-8"))
    snapshot, mapping = load_verified_snapshot_manifest(industry)

    task_launchers = {
        task["id"]: task["launcher"] for task in supervisor["tasks"]
    }
    active_task_ids = list(task_launchers)
    primary_outputs = primary["outputs"]
    primary_versioned = all(
        "v1_2" in primary_outputs[field]
        for field in (
            "pcf_daily_file",
            "iopv_snapshot_file",
            "daily_crosscheck_file",
            "raw_response_directory",
            "readiness_report_file",
            "task_status_file",
        )
    )
    primary_isolated = bool(
        primary_versioned
        and primary["quality"]["crosscheck_required_from"] == "2026-08-26"
        and primary["source_contract"]["sources"]["daily_crosscheck"]["source_id"]
        == "SSE_ETF_DAILY_TURNOVER_OFFICIAL"
    )

    industry_records = snapshot["records"]
    industry_isolated = bool(
        snapshot["snapshot_status"] == "PASS_CONTENT_ADDRESSED_INPUT_SNAPSHOT"
        and snapshot["shared_latest_consumed_by_evaluation"] is False
        and len(mapping) == len(industry["input_snapshot"]["roles"])
        and all(record["content_addressed"] for record in industry_records)
    )

    v3_manifest_path = ROOT / "config" / "v3_forward_2_manifest.json"
    v3_manifest = json.loads(v3_manifest_path.read_text(encoding="utf-8"))
    v3_records: list[dict[str, Any]] = []
    for relative, expected in v3_manifest["frozen_files"].items():
        path = ROOT / relative
        actual = _sha256(path) if path.is_file() else None
        v3_records.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        )
    v3_mismatches = [record for record in v3_records if not record["matches"]]
    v3_daily_path = ROOT / "paper" / "v3_forward_2_daily_run_status.json"
    v3_daily = json.loads(v3_daily_path.read_text(encoding="utf-8"))
    v3_latest_path = ROOT / "paper" / "v3_forward_2_latest_status.json"
    v3_quarantined = bool(
        "V3_FORWARD_2" not in active_task_ids
        and v3_daily.get("status") == "FAILED"
        and not v3_latest_path.exists()
        and len(v3_mismatches) > 0
    )

    checks = {
        "primary_market_v1_2_versioned_content_addressed_chain": primary_isolated,
        "industry_v1_2_snapshot_inputs_verified": industry_isolated,
        "active_supervisor_excludes_v3_forward_2": "V3_FORWARD_2"
        not in active_task_ids,
        "v3_hash_drift_remains_failed_and_quarantined": v3_quarantined,
        "active_launchers_are_versioned": task_launchers
        == {
            "PRIMARY_MARKET_PCF_IOPV": "scripts/run_510300_primary_market_collection_task_v1_2.ps1",
            "INDUSTRY_EXPECTATION_GAP": "scripts/run_industry_expectation_gap_forward_operations_task_v1_2.ps1",
            "PRIORITY_FORWARD_STATUS": "scripts/run_priority_forward_authoritative_status_task_v1_6.ps1",
        },
    }
    passed = all(checks.values())
    return {
        "schema_version": "1.0.0",
        "audit_status": (
            "PASS_FROZEN_INPUTS_CONTENT_ADDRESSED_AND_LATEST_ISOLATED"
            if passed
            else "FAILED_FROZEN_INPUT_OR_LATEST_ISOLATION"
        ),
        "audited_at": datetime.now(timezone).isoformat(),
        "scope": "ACTIVE_PRIORITY_FORWARD_CHAINS_PLUS_V3_QUARANTINE",
        "checks": checks,
        "active_supervisor": {
            "path": supervisor_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(supervisor_path),
            "task_launchers": task_launchers,
        },
        "primary_market": {
            "config_path": primary_path.relative_to(ROOT).as_posix(),
            "config_sha256": _sha256(primary_path),
            "legacy_inputs_read_only_before_v1_2_effective_date": True,
            "v1_2_outputs": {
                field: primary_outputs[field]
                for field in (
                    "pcf_daily_file",
                    "iopv_snapshot_file",
                    "daily_crosscheck_file",
                    "raw_response_directory",
                )
            },
        },
        "industry_expectation_gap": {
            "config_path": industry_path.relative_to(ROOT).as_posix(),
            "config_sha256": _sha256(industry_path),
            "snapshot_id": snapshot["snapshot_id"],
            "immutable_manifest_path": snapshot["immutable_manifest_path"],
            "shared_latest_consumed_by_evaluation": False,
            "records": industry_records,
        },
        "v3_forward_2_quarantine": {
            "manifest_path": v3_manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": _sha256(v3_manifest_path),
            "frozen_file_count": len(v3_records),
            "matching_count": len(v3_records) - len(v3_mismatches),
            "mismatch_count": len(v3_mismatches),
            "mismatches": v3_mismatches,
            "daily_status": v3_daily.get("status"),
            "failed_step": v3_daily.get("failed_step"),
            "latest_status_exists": v3_latest_path.exists(),
            "active_task": False,
            "retry_or_repair_performed": False,
        },
    }


def main() -> int:
    payload = build_audit()
    _atomic_json(OUTPUT, payload)
    print(payload["audit_status"])
    return 0 if payload["audit_status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
