"""冻结或验证QDII跨市场折价回归零方差评估修正V2。"""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.qdii_cross_market_discount_reversion_zero_variance_v2 import (
    CONFIG,
    load_correction_contract,
)
from scripts.freeze_qdii_cross_market_discount_reversion_v1 import (
    verify as verify_v1_manifest,
)


MANIFEST = (
    ROOT
    / "config"
    / "qdii_cross_market_discount_reversion_zero_variance_v2_manifest.json"
)
TRACKED_FILES = [
    "config/qdii_cross_market_discount_reversion_zero_variance_v2.yaml",
    "docs/QDII_CROSS_MARKET_DISCOUNT_REVERSION_ZERO_VARIANCE_V2_SPEC.md",
    "research/qdii_cross_market_discount_reversion_zero_variance_v2.py",
    "scripts/run_qdii_cross_market_discount_reversion_zero_variance_v2.py",
    "scripts/freeze_qdii_cross_market_discount_reversion_zero_variance_v2.py",
]
SOURCE_DEPENDENCIES = [
    "config/qdii_cross_market_discount_reversion_v1.yaml",
    "config/qdii_cross_market_discount_reversion_v1_manifest.json",
    "reports/research/qdii_cross_market_discount_reversion_v1_visible.json",
    "reports/research/QDII_CROSS_MARKET_DISCOUNT_REVERSION_V1_VISIBLE.md",
    "reports/data_quality/qdii_cross_market_discount_reversion_v1_inputs.json",
    "research/qdii_cross_market_discount_reversion_v1.py",
    "research/broad_liquid_etf_liquidity_shock_reversal_v1.py",
    "research/multi_asset_annual_excess_40pct_high_sharpe_v1.py",
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


def file_record(relative: str) -> dict[str, Any]:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(f"缺少V2冻结依赖：{relative}")
    return {
        "path": relative,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def current_tracked_hashes() -> dict[str, str]:
    return {relative: file_record(relative)["sha256"] for relative in TRACKED_FILES}


def tracked_content_hash(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def current_source_snapshot() -> dict[str, dict[str, Any]]:
    return {relative: file_record(relative) for relative in SOURCE_DEPENDENCIES}


def verify_v1_failure_preservation(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["source_freeze"]
    manifest_path = ROOT / source["v1_manifest"]
    report_path = ROOT / source["v1_visible_failure_report_json"]
    markdown_path = ROOT / source["v1_visible_failure_report_markdown"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    checks = {
        "v1_manifest_hash_unchanged": sha256_file(manifest_path)
        == source["expected_v1_manifest_sha256"],
        "v1_failure_report_hash_unchanged": sha256_file(report_path)
        == source["expected_v1_visible_failure_report_sha256"],
        "v1_failure_markdown_hash_unchanged": sha256_file(markdown_path)
        == source["expected_v1_visible_failure_markdown_sha256"],
        "v1_failure_status_unchanged": report.get("status")
        == source["expected_v1_failure_status"],
        "v1_failure_error_type_unchanged": report.get("error_type")
        == source["expected_v1_error_type"],
        "v1_failure_error_unchanged": report.get("error")
        == source["expected_v1_error"],
        "v1_performance_metrics_remain_unavailable": report.get(
            "performance_metrics_available"
        )
        is False,
        "v1_sealed_replication_remains_closed": report.get("decision", {}).get(
            "sealed_replication_open"
        )
        is False,
    }
    return {
        "status": (
            "PASS_V1_FAILURE_PRESERVED"
            if all(checks.values())
            else "FAIL_V1_FAILURE_NOT_PRESERVED"
        ),
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
    }


def freeze() -> dict[str, Any]:
    contract = load_correction_contract(CONFIG)
    v1_verification = verify_v1_manifest()
    if v1_verification["failure_count"]:
        raise RuntimeError("V1候选冻结清单验证失败；不得冻结V2")
    preservation = verify_v1_failure_preservation(contract)
    if not preservation["all_checks_pass"]:
        raise RuntimeError(f"V1失败未原样保留：{preservation['checks']}")
    files = current_tracked_hashes()
    source_snapshot = current_source_snapshot()
    manifest = {
        "schema_version": "2.0.0",
        "manifest_id": (
            "QDII_CROSS_MARKET_DISCOUNT_REVERSION_ZERO_VARIANCE_V2_MANIFEST"
        ),
        "status": "FROZEN_EVALUATION_CORRECTION_BEFORE_V2_VISIBLE_REPORT",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_candidate_id": contract["protocol"]["parent_candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "correction_class": contract["protocol"]["correction_class"],
        "economic_signal_execution_or_cost_parameter_changed": False,
        "same_visible_path_reused": True,
        "new_holdout": False,
        "initial_capital_cny": float(contract["objective"]["initial_capital_cny"]),
        "user_transaction_fee_rate_per_leg": float(
            contract["objective"]["user_transaction_fee_rate_per_leg"]
        ),
        "minimum_annualized_net_excess": float(
            contract["objective"]["minimum_annualized_net_excess"]
        ),
        "minimum_strategy_net_sharpe": float(
            contract["objective"]["minimum_strategy_net_sharpe"]
        ),
        "tracked_files": files,
        "tracked_content_sha256": tracked_content_hash(files),
        "source_snapshot": source_snapshot,
        "v1_manifest_verification": v1_verification,
        "v1_failure_preservation": preservation,
        "performance_or_future_return_computed_during_v2_freeze": False,
        "historical_result_can_verify_target": False,
        "paper_or_live_authorized": False,
    }
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for field in (
            "tracked_files",
            "tracked_content_sha256",
            "source_snapshot",
            "v1_failure_preservation",
        ):
            if existing.get(field) != manifest.get(field):
                raise RuntimeError("现有V2冻结清单与当前内容不同；不得覆盖")
        return existing
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_QDII_ZERO_VARIANCE_V2_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    contract = load_correction_contract(CONFIG)
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live_files = current_tracked_hashes()
    live_sources = current_source_snapshot()
    v1_verification = verify_v1_manifest()
    preservation = verify_v1_failure_preservation(contract)
    failures: list[str] = []
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != tracked_content_hash(live_files):
        failures.append("tracked_content_sha256")
    if existing.get("source_snapshot") != live_sources:
        failures.append("source_snapshot")
    if v1_verification["failure_count"]:
        failures.append("v1_manifest")
    if not preservation["all_checks_pass"]:
        failures.append("v1_failure_preservation")
    return {
        "status": (
            "PASS_QDII_CROSS_MARKET_DISCOUNT_REVERSION_ZERO_VARIANCE_V2_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_QDII_CROSS_MARKET_DISCOUNT_REVERSION_ZERO_VARIANCE_V2_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": tracked_content_hash(live_files),
        "tracked_file_count": int(len(live_files)),
        "v1_manifest_status": v1_verification["status"],
        "v1_failure_preservation_status": preservation["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="冻结或验证QDII跨市场折价回归零方差评估修正V2"
    )
    parser.add_argument("--verify", action="store_true", help="只验证现有冻结清单")
    args = parser.parse_args()
    payload = verify() if args.verify else freeze()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    if args.verify and payload["failure_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
