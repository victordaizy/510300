"""冻结或验证V12R评价器模式修正。"""

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

from research.digital_asset_basis_ranked_spot_long_v12r_schema_correction import (
    CONFIG,
    _load_correction,
    load_contract,
)
from scripts.freeze_digital_asset_basis_ranked_spot_long_v12 import (
    verify as verify_base_manifest,
)
from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v22 import (
    verify as verify_parent_manifest,
)
from scripts.freeze_qdii_cross_market_discount_reversion_zero_variance_v2 import (
    verify as verify_evaluator_manifest,
)


MANIFEST = (
    ROOT
    / "config"
    / "digital_asset_basis_ranked_spot_long_v12r_schema_correction_manifest.json"
)
TRACKED_FILES = [
    "config/digital_asset_basis_ranked_spot_long_v12r_schema_correction.yaml",
    "docs/DIGITAL_ASSET_BASIS_RANKED_SPOT_LONG_V12R_SCHEMA_CORRECTION_SPEC.md",
    "research/digital_asset_basis_ranked_spot_long_v12r_schema_correction.py",
    "scripts/freeze_digital_asset_basis_ranked_spot_long_v12r_schema_correction.py",
    "scripts/run_digital_asset_basis_ranked_spot_long_v12r_schema_correction.py",
    "tests/test_digital_asset_basis_ranked_spot_long_v12r_schema_correction.py",
]
SOURCE_DEPENDENCIES = [
    "config/digital_asset_basis_ranked_spot_long_v12.yaml",
    "research/digital_asset_basis_ranked_spot_long_v12.py",
    "config/digital_asset_basis_ranked_spot_long_v12_manifest.json",
    "reports/research/digital_asset_basis_ranked_spot_long_v12_visible.json",
    "data/raw/digital_asset_current_quarter_basis_v11/collection_status.json",
    "data/raw/digital_asset_current_quarter_basis_v11/spot_1h_2021_2026.parquet",
    "data/raw/digital_asset_current_quarter_basis_v11/current_quarter_1h_2021_2026.parquet",
    "data/raw/digital_asset_current_quarter_basis_v11/fx_2021_2026.parquet",
    "data/raw/digital_asset_current_quarter_basis_v11/H00300_2021_2026.parquet",
    "config/multi_asset_annual_excess_40pct_high_sharpe_v22_manifest.json",
    "config/qdii_cross_market_discount_reversion_zero_variance_v2.yaml",
    "config/qdii_cross_market_discount_reversion_zero_variance_v2_manifest.json",
    "research/qdii_cross_market_discount_reversion_zero_variance_v2.py",
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
        raise FileNotFoundError(f"缺少V12R模式修正冻结文件：{relative}")
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


def source_snapshot() -> dict[str, dict[str, Any]]:
    return {relative: file_record(relative) for relative in SOURCE_DEPENDENCIES}


def verify_base_failure() -> dict[str, Any]:
    correction = _load_correction()
    report_path = ROOT / correction["base"]["failed_report"]
    manifest_path = ROOT / correction["base"]["manifest"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    verification = verify_base_manifest()
    checks = {
        "candidate": report.get("candidate_id")
        == correction["protocol"]["correction_of_candidate_id"],
        "status": report.get("status") == correction["base"]["required_failure_status"],
        "error_type": report.get("error_type") == correction["base"]["required_error_type"],
        "error": report.get("error") == correction["base"]["required_error"],
        "performance_unavailable": report.get("performance_metrics_available")
        is correction["base"]["required_performance_metrics_available"],
        "manifest_verified": verification.get("failure_count") == 0,
        "manifest_snapshot": report.get("manifest_verification", {}).get("manifest_sha256")
        == sha256_file(manifest_path),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V12失败不符合V12R修正前提：{checks}")
    base_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "checks": checks,
        "manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(report_path),
        "tracked_content_sha256": base_manifest["tracked_content_sha256"],
        "prefreeze_data_coverage": base_manifest["prefreeze_data_coverage"],
        "performance_metrics_available": False,
    }


def _assert_outputs_absent(contract: dict[str, Any]) -> None:
    names = (
        "input_audit_json",
        "visible_daily_returns",
        "visible_targets",
        "visible_trades",
        "visible_report_json",
        "visible_report_markdown",
    )
    existing = [
        contract["outputs"][name]
        for name in names
        if (ROOT / contract["outputs"][name]).exists()
    ]
    if existing:
        raise RuntimeError(f"冻结前已存在V12R模式修正结果：{existing}")


def _dependencies() -> dict[str, dict[str, Any]]:
    return {
        "parent": verify_parent_manifest(),
        "base_v12": verify_base_manifest(),
        "evaluator": verify_evaluator_manifest(),
    }


def freeze() -> dict[str, Any]:
    contract = load_contract(CONFIG)
    if MANIFEST.exists():
        verification = verify()
        if verification["failure_count"]:
            raise RuntimeError(f"现有V12R模式修正清单验证失败：{verification['failures']}")
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    _assert_outputs_absent(contract)
    dependencies = _dependencies()
    if any(item["failure_count"] for item in dependencies.values()):
        raise RuntimeError("V12R模式修正依赖冻结清单验证失败")
    failure = verify_base_failure()
    correction = _load_correction()
    files = current_tracked_hashes()
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "DIGITAL_ASSET_BASIS_RANKED_SPOT_LONG_V12R_SCHEMA_CORRECTION_MANIFEST",
        "status": correction["protocol"]["status"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_id": contract["protocol"]["candidate_id"],
        "correction_of_candidate_id": contract["protocol"][
            "correction_of_candidate_id"
        ],
        "tracked_files": files,
        "tracked_content_sha256": tracked_content_hash(files),
        "source_snapshot": source_snapshot(),
        "dependency_verifications": dependencies,
        "base_failure": failure,
        "correction": correction["correction"],
        "performance_or_future_return_computed_during_freeze": False,
        "signal_portfolio_cost_sample_or_engine_changed": False,
        "historical_result_can_verify_target": False,
        "paper_or_live_authorized": False,
    }
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_BASIS_RANKED_SPOT_LONG_V12R_SCHEMA_CORRECTION_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    load_contract(CONFIG)
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live_files = current_tracked_hashes()
    live_sources = source_snapshot()
    dependencies = _dependencies()
    failures: list[str] = []
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != tracked_content_hash(live_files):
        failures.append("tracked_content_sha256")
    if existing.get("source_snapshot") != live_sources:
        failures.append("source_snapshot")
    try:
        if existing.get("base_failure") != verify_base_failure():
            failures.append("base_failure")
    except Exception:
        failures.append("base_failure")
    if existing.get("correction") != _load_correction()["correction"]:
        failures.append("correction")
    if existing.get("performance_or_future_return_computed_during_freeze") is not False:
        failures.append("freeze_computed_outcome")
    if existing.get("signal_portfolio_cost_sample_or_engine_changed") is not False:
        failures.append("expanded_correction")
    for name, item in dependencies.items():
        if item["failure_count"]:
            failures.append(f"{name}_manifest")
    return {
        "status": (
            "PASS_BASIS_RANKED_SPOT_LONG_V12R_SCHEMA_CORRECTION_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_BASIS_RANKED_SPOT_LONG_V12R_SCHEMA_CORRECTION_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": tracked_content_hash(live_files),
        "tracked_file_count": int(len(live_files)),
        "dependency_statuses": {
            name: item["status"] for name, item in dependencies.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证V12R评价器模式修正")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
