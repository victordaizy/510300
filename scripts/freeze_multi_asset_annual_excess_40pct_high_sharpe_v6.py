"""冻结或验证40个百分点高夏普研究V6汇率修正扩展。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v5 import (
    verify as verify_parent_manifest,
)
from scripts.freeze_us_leveraged_multi_asset_absolute_momentum_v1 import (
    verify as verify_v1_manifest,
)


CONFIG = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v6.yaml"
MANIFEST = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v6_manifest.json"
TRACKED_FILES = [
    "config/multi_asset_annual_excess_40pct_high_sharpe_v6.yaml",
    "docs/MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V6_SPEC.md",
    "scripts/freeze_multi_asset_annual_excess_40pct_high_sharpe_v6.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_contract() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V6扩展协议必须是YAML对象")
    protocol = payload.get("protocol", {})
    objective = payload.get("objective", {})
    correction = payload.get("correction_lane", {})
    safety = payload.get("safety", {})
    failures: list[str] = []
    if protocol.get("protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V6":
        failures.append("protocol_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V5":
        failures.append("parent_protocol_id")
    expected = {
        "initial_capital_cny": 500000.0,
        "minimum_annualized_net_excess": 0.40,
        "minimum_strategy_net_sharpe": 1.50,
        "user_transaction_fee_rate_per_leg": 0.0001,
    }
    for key, value in expected.items():
        if float(objective.get(key, float("nan"))) != value:
            failures.append(key)
    if objective.get("benchmark") != "H00300_TOTAL_RETURN":
        failures.append("benchmark")
    if int(correction.get("maximum_corrected_versions", 0)) != 1:
        failures.append("maximum_corrected_versions")
    permitted = correction.get("permitted_changes", [])
    if permitted != [
        "FX_SOURCE_TO_CHINA_FOREIGN_EXCHANGE_TRADE_SYSTEM_OFFICIAL_USD_CNY_CENTRAL_PARITY",
        "MAXIMUM_FX_STALENESS_CALENDAR_DAYS_FROM_7_TO_14_FOR_CHINA_STATUTORY_HOLIDAYS",
        "VERSION_IDENTIFIERS_PATHS_STATUS_AND_PROVENANCE",
    ]:
        failures.append("permitted_changes")
    if correction.get("acquisition_may_compute_strategy_return_or_rank") is not False:
        failures.append("acquisition_boundary")
    if any(bool(safety.get(name, True)) for name in safety):
        failures.append("safety")
    if failures:
        raise ValueError(f"V6扩展协议被弱化或损坏：{sorted(set(failures))}")
    return payload


def tracked_hashes() -> dict[str, str]:
    missing = [item for item in TRACKED_FILES if not (ROOT / item).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少V6冻结文件：{missing}")
    return {item: sha256_file(ROOT / item) for item in TRACKED_FILES}


def content_hash(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_v1_failure(contract: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / contract["inputs"]["v1_failure_report"]
    report = json.loads(path.read_text(encoding="utf-8"))
    required = contract["registered_failure"]
    checks = {
        "status": report.get("status") == required["required_status"],
        "error": report.get("error") == required["required_error"],
        "performance_unavailable": report.get("performance_metrics_available") is False,
        "replication_closed": report.get("decision", {}).get("sealed_replication_open") is False,
        "target_not_verified": report.get("decision", {}).get(
            "historical_result_verifies_40pct_target"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V1失败报告不符合V6登记条件：{checks}")
    return {
        "checks": checks,
        "report_path": path.relative_to(ROOT).as_posix(),
        "report_sha256": sha256_file(path),
        "error": report["error"],
    }


def freeze() -> dict[str, Any]:
    contract = load_contract()
    parent = verify_parent_manifest()
    v1 = verify_v1_manifest()
    if parent["failure_count"] or v1["failure_count"]:
        raise RuntimeError("父协议或V1候选冻结清单验证失败")
    failure = verify_v1_failure(contract)
    files = tracked_hashes()
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V6_MANIFEST",
        "status": contract["protocol"]["status"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_id": contract["protocol"]["protocol_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": content_hash(files),
        "parent_manifest_verification": parent,
        "v1_manifest_verification": v1,
        "v1_failure": failure,
        "objective": contract["objective"],
        "correction_lane": contract["correction_lane"],
        "historical_result_can_verify_target": False,
        "paper_or_live_authorized": False,
    }
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for field in (
            "tracked_files",
            "tracked_content_sha256",
            "v1_failure",
            "objective",
            "correction_lane",
        ):
            if existing.get(field) != manifest.get(field):
                raise RuntimeError("现有V6清单与当前内容不同；不得覆盖")
        return existing
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_MULTI_ASSET_40PCT_HIGH_SHARPE_V6_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    contract = load_contract()
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live_files = tracked_hashes()
    parent = verify_parent_manifest()
    v1 = verify_v1_manifest()
    failures: list[str] = []
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != content_hash(live_files):
        failures.append("tracked_content_sha256")
    try:
        if existing.get("v1_failure") != verify_v1_failure(contract):
            failures.append("v1_failure")
    except Exception:
        failures.append("v1_failure")
    if parent["failure_count"]:
        failures.append("parent_manifest")
    if v1["failure_count"]:
        failures.append("v1_manifest")
    return {
        "status": (
            "PASS_MULTI_ASSET_40PCT_HIGH_SHARPE_V6_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_MULTI_ASSET_40PCT_HIGH_SHARPE_V6_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": content_hash(live_files),
        "parent_manifest_status": parent["status"],
        "v1_manifest_status": v1["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证40个百分点高夏普V6扩展协议")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
