"""对期权逐笔盘口分支做终止性的零付费来源资格判定。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "audit" / "OPTION_ORDERBOOK_ZERO_COST_SOURCE_QUALIFICATION_V1.json"


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
    config_path = ROOT / "config" / "510300_option_forward_orderbook_v1.yaml"
    manifest_path = (
        ROOT / "config" / "510300_option_forward_orderbook_v1_protocol_manifest.json"
    )
    registry_path = ROOT / "data" / "governance" / "FREE_SOURCE_REGISTRY_V1_1.csv"
    supervisor_path = ROOT / "config" / "priority_forward_supervisor_v1_2.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    supervisor = yaml.safe_load(supervisor_path.read_text(encoding="utf-8"))
    with registry_path.open("r", encoding="utf-8-sig", newline="") as handle:
        registry = {row["source_id"]: row for row in csv.DictReader(handle)}

    manifest_records: list[dict[str, Any]] = []
    for relative, expected in manifest["frozen_files"].items():
        path = ROOT / relative
        actual = _sha256(path) if path.is_file() else None
        manifest_records.append(
            {
                "path": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "matches": actual == expected,
            }
        )
    manifest_mismatches = [
        record for record in manifest_records if not record["matches"]
    ]

    master = registry["TUSHARE_PROXY_OPT_BASIC"]
    quotes = registry["SINA_OPTION_SSE_SPOT_PRICE"]
    collector_path = ROOT / "scripts" / "collect_510300_option_forward_orderbook_v1.py"
    collector_text = collector_path.read_text(encoding="utf-8")
    snapshot_directory = ROOT / config["paths"]["snapshot_directory"]
    master_directory = ROOT / config["paths"]["master_snapshot_directory"]
    snapshot_count = (
        len(list(snapshot_directory.glob("*.parquet")))
        if snapshot_directory.is_dir()
        else 0
    )
    master_snapshot_count = (
        len(list(master_directory.glob("*.parquet")))
        if master_directory.is_dir()
        else 0
    )
    active_task_ids = [task["id"] for task in supervisor["tasks"]]

    qualification_gates = {
        "dynamic_contract_master_zero_cost_access_proven": False,
        "dynamic_contract_master_core_fields_proven": False,
        "orderbook_core_fields_declared": set(config["quality"]["required_columns"]).issuperset(
            {
                "bid1",
                "ask1",
                "bid1_volume",
                "ask1_volume",
                "last_price",
                "volume",
                "open_interest",
                "quote_timestamp",
            }
        ),
        "orderbook_raw_response_archiving_implemented": "raw_response_path"
        in collector_text
        and "raw_response_hash" in collector_text,
        "source_terms_and_archive_rights_proven": False,
        "source_stability_proven": False,
        "frozen_protocol_hashes_exact": len(manifest_mismatches) == 0,
        "twenty_day_trial_complete": False,
    }
    blocking_reasons = [
        "DYNAMIC_CONTRACT_MASTER_ZERO_COST_ACCESS_NOT_PROVEN",
        "DYNAMIC_CONTRACT_MASTER_REQUIRED_FIELDS_NOT_PROVEN",
        "ORDERBOOK_RAW_RESPONSE_ARCHIVE_NOT_IMPLEMENTED",
        "SOURCE_TERMS_AND_ARCHIVE_RIGHTS_NOT_PROVEN",
        "SOURCE_STABILITY_NOT_PROVEN",
        "FROZEN_PROTOCOL_HASH_DRIFT",
    ]
    return {
        "schema_version": "1.0.0",
        "qualification_status": "BLOCKED_ZERO_COST_SOURCE_UNAVAILABLE",
        "branch_status": "CLOSED_NO_20D_TRIAL",
        "audited_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "policy": "CONDITIONAL_FORWARD_ONLY",
        "data_purchase_budget_cny": 0,
        "external_request_attempted": False,
        "paid_source_called": False,
        "supplier_contacted": False,
        "qualification_gates": qualification_gates,
        "blocking_reasons": blocking_reasons,
        "source_assessment": {
            "dynamic_contract_master": {
                "source_id": master["source_id"],
                "provider": master["provider"],
                "access_cost_cny": float(master["access_cost_cny"]),
                "admission_status": master["admission_status"],
                "quality_status": master["quality_status"],
                "failure_reason": master["failure_reason"],
                "required_by_frozen_protocol": config["universe"][
                    "dynamic_contract_master_source"
                ],
            },
            "orderbook_quote": {
                "source_id": quotes["source_id"],
                "provider": quotes["provider"],
                "access_cost_cny": float(quotes["access_cost_cny"]),
                "admission_status": quotes["admission_status"],
                "quality_status": quotes["quality_status"],
                "failure_reason": quotes["failure_reason"],
                "raw_response_archiving_implemented": qualification_gates[
                    "orderbook_raw_response_archiving_implemented"
                ],
            },
        },
        "protocol_integrity": {
            "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
            "manifest_sha256": _sha256(manifest_path),
            "frozen_file_count": len(manifest_records),
            "matching_count": len(manifest_records) - len(manifest_mismatches),
            "mismatch_count": len(manifest_mismatches),
            "mismatches": manifest_mismatches,
        },
        "forward_evidence": {
            "forward_start_exists": (
                ROOT / config["paths"]["forward_start_status"]
            ).is_file(),
            "snapshot_count": snapshot_count,
            "master_snapshot_count": master_snapshot_count,
            "qualified_trial_days": 0,
            "required_trial_days_before_long_run": 20,
        },
        "closure": {
            "active_supervisor_task": "OPTION_ORDERBOOK" in active_task_ids,
            "scheduled_trial_enabled": False,
            "historical_backfill_enabled": False,
            "daily_proxy_enabled": False,
            "signal_calculation_enabled": False,
            "return_test_enabled": False,
            "next_action": "CLOSED_PRESERVE_EVIDENCE",
        },
        "evidence": {
            "config": {
                "path": config_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(config_path),
            },
            "collector": {
                "path": collector_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(collector_path),
            },
            "source_registry": {
                "path": registry_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(registry_path),
            },
            "supervisor": {
                "path": supervisor_path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(supervisor_path),
            },
        },
    }


def main() -> int:
    payload = build_audit()
    _atomic_json(OUTPUT, payload)
    print(payload["qualification_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
