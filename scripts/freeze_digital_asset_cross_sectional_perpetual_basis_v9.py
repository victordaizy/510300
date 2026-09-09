"""冻结或验证多币种永续基差因子V9。"""

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

from research.digital_asset_cross_sectional_perpetual_basis_v9 import CONFIG, load_contract
from scripts.freeze_digital_asset_cross_sectional_hourly_reversal_v8 import (
    verify as verify_v8_manifest,
)
from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v19 import (
    verify as verify_parent_manifest,
)
from scripts.freeze_qdii_cross_market_discount_reversion_zero_variance_v2 import (
    verify as verify_evaluator_manifest,
)


MANIFEST = ROOT / "config" / "digital_asset_cross_sectional_perpetual_basis_v9_manifest.json"
TRACKED_FILES = [
    "config/digital_asset_cross_sectional_perpetual_basis_v9.yaml",
    "docs/DIGITAL_ASSET_CROSS_SECTIONAL_PERPETUAL_BASIS_V9_SPEC.md",
    "research/digital_asset_cross_sectional_perpetual_basis_v9.py",
    "scripts/freeze_digital_asset_cross_sectional_perpetual_basis_v9.py",
    "scripts/run_digital_asset_cross_sectional_perpetual_basis_v9.py",
    "tests/test_digital_asset_cross_sectional_perpetual_basis_v9.py",
]
VISIBLE_SOURCE_DEPENDENCIES = [
    "data/raw/digital_asset_cross_sectional_perpetual_basis_v9/collection_status.json",
    "data/raw/digital_asset_cross_sectional_perpetual_basis_v9/spot_1h_visible_through_2023.parquet",
    "data/raw/digital_asset_cross_sectional_hourly_reversal_v8/perpetual_1h_visible_through_2023.parquet",
    "data/raw/digital_asset_cross_sectional_hourly_reversal_v8/funding_rates_visible_through_2023.parquet",
    "data/raw/digital_asset_cross_sectional_hourly_reversal_v8/visible_fx_through_2023.parquet",
    "data/raw/digital_asset_cross_sectional_hourly_reversal_v8/visible_H00300_through_2023.parquet",
    "config/multi_asset_annual_excess_40pct_high_sharpe_v19_manifest.json",
    "config/digital_asset_cross_sectional_hourly_reversal_v8_manifest.json",
    "research/digital_asset_cross_sectional_hourly_reversal_v8.py",
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
        raise FileNotFoundError(f"缺少永续基差V9冻结文件：{relative}")
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


def current_visible_source_snapshot() -> dict[str, dict[str, Any]]:
    return {relative: file_record(relative) for relative in VISIBLE_SOURCE_DEPENDENCIES}


def prefreeze_data_coverage(contract: dict[str, Any]) -> dict[str, Any]:
    status_path = ROOT / contract["inputs"]["source_status"]
    status = json.loads(status_path.read_text(encoding="utf-8"))
    gate = contract["prefreeze_coverage_gate"]
    symbols = list(contract["universe"]["fixed_symbols"])
    spot = status.get("coverage", {}).get("spot_hourly", {})
    gap = status.get("coverage", {}).get("venue_wide_gap_audit", {})
    forbidden = (
        "basis_value_or_rank_computed",
        "long_or_short_direction_computed",
        "signal_or_target_computed",
        "strategy_or_benchmark_return_computed",
        "sealed_period_2024_onward_read",
        "account_or_authenticated_endpoint_used",
    )
    checks = {
        "status": status.get("status") == "PASS_INPUT_ACQUISITION_ONLY",
        "candidate_id": status.get("candidate_id") == contract["protocol"]["candidate_id"],
        "symbols": status.get("symbols") == symbols,
        "required_symbol_count": len(symbols) == int(gate["required_symbols"]),
        "spot_rows": all(
            spot.get(symbol, {}).get("row_count")
            == int(gate["required_spot_hourly_rows_per_symbol"])
            for symbol in symbols
        ),
        "spot_no_duplicates": all(
            spot.get(symbol, {}).get("duplicate_time_count") == 0 for symbol in symbols
        ),
        "spot_endpoints": all(
            spot.get(symbol, {}).get("first_time")
            == pd_timestamp(gate["required_grid_start"])
            and spot.get(symbol, {}).get("last_time")
            == pd_timestamp(gate["required_grid_end"])
            for symbol in symbols
        ),
        "synchronized_missing_hours": gap.get("all_symbols_share_identical_observed_grid") is True,
        "missing_hour_count": gap.get("common_venue_wide_missing_hour_count")
        == int(gate["required_common_venue_wide_spot_missing_hours"]),
        "decision_hours_complete": gap.get("missing_signal_23utc_or_execution_00utc_count")
        == int(gate["required_missing_signal_or_execution_hours"]),
        "missing_not_backfilled": gap.get("missing_values_backfilled") is False,
        "forbidden_computations_absent": all(status.get(key) is False for key in forbidden),
    }
    output_mapping = {
        "spot_1h": contract["inputs"]["spot_1h"],
        "perpetual_1h": contract["inputs"]["perpetual_1h"],
        "funding_rates": contract["inputs"]["funding_rates"],
        "fx": contract["inputs"]["visible_fx"],
        "benchmark": contract["inputs"]["visible_benchmark"],
    }
    checks["input_files_match_collection_receipt"] = all(
        status.get("outputs", {}).get(key, {}).get("path") == relative
        and status.get("outputs", {}).get(key, {}).get("sha256")
        == sha256_file(ROOT / relative)
        for key, relative in output_mapping.items()
    )
    return {
        "all_checks_pass": bool(all(checks.values())),
        "checks": checks,
        "status_path": status_path.relative_to(ROOT).as_posix(),
        "status_sha256": sha256_file(status_path),
        "spot_rows_per_symbol": int(gate["required_spot_hourly_rows_per_symbol"]),
        "perpetual_rows_per_symbol": int(gate["required_perpetual_hourly_rows_per_symbol"]),
        "funding_rows_per_symbol": int(gate["required_funding_rows_per_symbol"]),
        "common_venue_wide_missing_hour_count": int(
            gate["required_common_venue_wide_spot_missing_hours"]
        ),
        "parquet_values_read": False,
        "basis_value_or_rank_computed": False,
        "signal_or_target_computed": False,
        "strategy_or_benchmark_return_computed": False,
        "sealed_replication_read": False,
    }


def pd_timestamp(value: Any) -> str:
    from pandas import Timestamp

    return Timestamp(value).isoformat()


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
        raise RuntimeError(f"冻结前已存在永续基差V9可见结果：{existing}")


def _dependencies() -> dict[str, dict[str, Any]]:
    return {
        "parent": verify_parent_manifest(),
        "v8_engine": verify_v8_manifest(),
        "evaluator": verify_evaluator_manifest(),
    }


def freeze() -> dict[str, Any]:
    contract = load_contract(CONFIG)
    if MANIFEST.exists():
        verification = verify()
        if verification["failure_count"]:
            raise RuntimeError(f"现有永续基差V9清单验证失败：{verification['failures']}")
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    _assert_outputs_absent(contract)
    dependencies = _dependencies()
    if any(item["failure_count"] for item in dependencies.values()):
        raise RuntimeError("永续基差V9依赖冻结清单验证失败")
    coverage = prefreeze_data_coverage(contract)
    if not coverage["all_checks_pass"]:
        raise RuntimeError(f"永续基差V9冻结前覆盖门失败：{coverage['checks']}")
    files = current_tracked_hashes()
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "DIGITAL_ASSET_CROSS_SECTIONAL_PERPETUAL_BASIS_V9_MANIFEST",
        "status": "FROZEN_BEFORE_VISIBLE_OUTCOME",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": tracked_content_hash(files),
        "visible_source_snapshot": current_visible_source_snapshot(),
        "dependency_verifications": dependencies,
        "prefreeze_data_coverage": coverage,
        "initial_capital_cny": float(contract["account"]["initial_capital_cny"]),
        "user_transaction_fee_rate_per_leg": float(
            contract["account"]["user_transaction_fee_rate_per_leg"]
        ),
        "minimum_annualized_net_excess": float(
            contract["visible_gates"]["minimum_annualized_net_excess"]
        ),
        "minimum_strategy_net_sharpe": float(
            contract["visible_gates"]["minimum_strategy_net_sharpe"]
        ),
        "performance_or_future_return_computed_during_freeze": False,
        "basis_value_or_rank_computed_during_freeze": False,
        "signal_direction_or_target_count_computed_during_freeze": False,
        "sealed_replication_read_during_freeze": False,
        "historical_result_can_verify_target": False,
        "paper_or_live_authorized": False,
    }
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_CROSS_SECTIONAL_PERPETUAL_BASIS_V9_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    load_contract(CONFIG)
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live_files = current_tracked_hashes()
    live_sources = current_visible_source_snapshot()
    dependencies = _dependencies()
    failures: list[str] = []
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != tracked_content_hash(live_files):
        failures.append("tracked_content_sha256")
    if existing.get("visible_source_snapshot") != live_sources:
        failures.append("visible_source_snapshot")
    if not existing.get("prefreeze_data_coverage", {}).get("all_checks_pass", False):
        failures.append("prefreeze_data_coverage")
    if existing.get("performance_or_future_return_computed_during_freeze") is not False:
        failures.append("freeze_read_future_return")
    if existing.get("basis_value_or_rank_computed_during_freeze") is not False:
        failures.append("freeze_computed_basis")
    if existing.get("sealed_replication_read_during_freeze") is not False:
        failures.append("freeze_read_replication")
    for name, item in dependencies.items():
        if item["failure_count"]:
            failures.append(f"{name}_manifest")
    return {
        "status": (
            "PASS_CROSS_SECTIONAL_PERPETUAL_BASIS_V9_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_CROSS_SECTIONAL_PERPETUAL_BASIS_V9_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": tracked_content_hash(live_files),
        "tracked_file_count": int(len(live_files)),
        "dependency_statuses": {name: item["status"] for name, item in dependencies.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证多币种永续基差V9")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
