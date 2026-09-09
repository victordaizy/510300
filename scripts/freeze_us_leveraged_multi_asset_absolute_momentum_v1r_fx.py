"""冻结或验证美国杠杆多资产V1R汇率来源修正版。"""

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

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.us_leveraged_multi_asset_absolute_momentum_v1 import (
    build_monthly_selections as build_v1_selections,
    load_contract as load_v1_contract,
    load_visible_inputs as load_v1_visible_inputs,
)
from research.us_leveraged_multi_asset_absolute_momentum_v1r_fx import (
    ALLOWED_CHANGED_PATHS,
    CONFIG,
    build_monthly_selections,
    changed_paths,
    load_contract,
    load_visible_inputs,
)
from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v6 import (
    verify as verify_parent_manifest,
)
from scripts.freeze_qdii_cross_market_discount_reversion_zero_variance_v2 import (
    verify as verify_evaluator_manifest,
)
from scripts.freeze_us_leveraged_multi_asset_absolute_momentum_v1 import (
    verify as verify_v1_manifest,
)


MANIFEST = ROOT / "config" / "us_leveraged_multi_asset_absolute_momentum_v1r_fx_source_correction_manifest.json"
TRACKED_FILES = [
    "config/us_leveraged_multi_asset_absolute_momentum_v1r_fx_source_correction.yaml",
    "docs/US_LEVERAGED_MULTI_ASSET_ABSOLUTE_MOMENTUM_V1R_FX_SOURCE_CORRECTION_SPEC.md",
    "research/us_leveraged_multi_asset_absolute_momentum_v1r_fx.py",
    "scripts/download_us_leveraged_multi_asset_fx_v1r.py",
    "scripts/freeze_us_leveraged_multi_asset_absolute_momentum_v1r_fx.py",
    "scripts/run_us_leveraged_multi_asset_absolute_momentum_v1r_fx.py",
    "tests/test_download_us_leveraged_multi_asset_fx_v1r.py",
    "tests/test_us_leveraged_multi_asset_absolute_momentum_v1r_fx.py",
]
VISIBLE_SOURCE_DEPENDENCIES = [
    "data/raw/us_leveraged_multi_asset_v1r_fx/collection_status.json",
    "data/raw/us_leveraged_multi_asset_v1r_fx/chinamoney_usd_cny_2008_2026_receipt.json",
    "data/raw/us_leveraged_multi_asset_v1r_fx/visible_fx_through_2019.parquet",
    "data/raw/us_leveraged_multi_asset_v1/product_master.parquet",
    "data/raw/us_leveraged_multi_asset_v1/visible_panel_through_2019.parquet",
    "data/raw/us_leveraged_multi_asset_v1/visible_H00300_through_2019.parquet",
    "config/multi_asset_annual_excess_40pct_high_sharpe_v6_manifest.json",
    "config/us_leveraged_multi_asset_absolute_momentum_v1_manifest.json",
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
        raise FileNotFoundError(f"缺少V1R冻结文件：{relative}")
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
    """只快照可见与冻结依赖，不访问任何封存输入。"""

    return {relative: file_record(relative) for relative in VISIBLE_SOURCE_DEPENDENCIES}


def prefreeze_correction_checks(contract: dict[str, Any]) -> dict[str, Any]:
    """证明信号逐行未变，并验证官方汇率严格滞后覆盖。"""

    partition = contract["historical_partition"]
    corrected_panel, master, _fx, _benchmark, corrected_audit = load_visible_inputs(
        contract, signal_only=True
    )
    corrected = build_monthly_selections(
        corrected_panel,
        contract,
        start=partition["visible_start"],
        end=partition["visible_end"],
    )
    base_contract = load_v1_contract()
    base_panel, _master, _base_fx, _base_benchmark, base_audit = load_v1_visible_inputs(
        base_contract, signal_only=True
    )
    base = build_v1_selections(
        base_panel,
        base_contract,
        start=partition["visible_start"],
        end=partition["visible_end"],
    )
    for corrected_frame, base_frame, label in zip(
        corrected[:3], base[:3], ("selections", "schedule", "features"), strict=True
    ):
        try:
            pd.testing.assert_frame_equal(
                corrected_frame.reset_index(drop=True),
                base_frame.reset_index(drop=True),
                check_dtype=True,
                check_exact=True,
            )
        except AssertionError as exc:
            raise RuntimeError(f"V1R与V1的{label}不一致") from exc
    status = json.loads((ROOT / contract["inputs"]["source_status"]).read_text(encoding="utf-8"))
    coverage = status.get("visible_strict_lag_coverage", {})
    actual_changes = changed_paths(contract)
    checks = {
        "actual_changed_paths_exactly_authorized": set(actual_changes)
        == ALLOWED_CHANGED_PATHS,
        "selections_identical_to_v1": True,
        "schedule_identical_to_v1": True,
        "features_identical_to_v1": True,
        "official_fx_acquisition_pass": status.get("status")
        == "PASS_FX_SOURCE_CORRECTION_INPUT_ACQUISITION_ONLY",
        "strategy_return_or_rank_not_computed": status.get(
            "strategy_total_return_or_rank_computed"
        )
        is False,
        "security_inputs_reused": status.get("security_price_inputs_reused_without_change")
        is True,
        "strictly_lagged_fx": coverage.get("strictly_lagged") is True,
        "maximum_fx_age_within_14_days": int(
            coverage.get("maximum_fx_age_calendar_days", 999)
        )
        <= 14,
        "future_open_or_strategy_return_not_read": corrected_audit[
            "future_open_or_strategy_return_read"
        ]
        is False
        and base_audit["future_open_or_strategy_return_read"] is False,
        "sealed_replication_not_read": corrected_audit["sealed_panel_or_benchmark_read"]
        is False
        and base_audit["sealed_panel_or_benchmark_read"] is False,
    }
    signal_audit = corrected[3]
    return {
        "status": (
            "PASS_PREFREEZE_V1R_FX_CORRECTION"
            if all(checks.values())
            else "FAIL_PREFREEZE_V1R_FX_CORRECTION"
        ),
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "actual_changed_paths": actual_changes,
        "signal_date_count": int(signal_audit["signal_date_count"]),
        "selection_rows": int(signal_audit["selection_row_count"]),
        "dates_with_two_eligible_assets": int(
            signal_audit["dates_with_two_eligible_assets"]
        ),
        "official_fx_coverage": coverage,
        "future_open_or_strategy_return_read": False,
        "performance_or_future_return_computed": False,
        "sealed_replication_read": False,
    }


def _assert_visible_outputs_absent(contract: dict[str, Any]) -> None:
    outputs = contract["outputs"]
    names = (
        "input_audit_json",
        "visible_daily_returns",
        "visible_selections",
        "visible_trades",
        "visible_report_json",
        "visible_report_markdown",
    )
    existing = [outputs[name] for name in names if (ROOT / outputs[name]).exists()]
    if existing:
        raise RuntimeError(f"V1R冻结前已存在可见结果：{existing}")


def freeze() -> dict[str, Any]:
    contract = load_contract(CONFIG)
    if MANIFEST.exists():
        verification = verify()
        if verification["failure_count"]:
            raise RuntimeError(f"现有V1R清单验证失败：{verification['failures']}")
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    _assert_visible_outputs_absent(contract)
    parent = verify_parent_manifest()
    v1 = verify_v1_manifest()
    evaluator = verify_evaluator_manifest()
    if parent["failure_count"] or v1["failure_count"] or evaluator["failure_count"]:
        raise RuntimeError("V1R父协议、V1或保守评价器清单验证失败")
    files = current_tracked_hashes()
    visible_sources = current_visible_source_snapshot()
    correction = prefreeze_correction_checks(contract)
    if not correction["all_checks_pass"]:
        raise RuntimeError(f"V1R冻结前修正检查失败：{correction['checks']}")
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "US_LEVERAGED_MULTI_ASSET_ABSOLUTE_MOMENTUM_V1R_FX_MANIFEST",
        "status": "FROZEN_BEFORE_CORRECTED_VISIBLE_OUTCOME",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": tracked_content_hash(files),
        "visible_source_snapshot": visible_sources,
        "parent_manifest_verification": parent,
        "v1_manifest_verification": v1,
        "conservative_evaluator_manifest_verification": evaluator,
        "prefreeze_correction_checks": correction,
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
        "sealed_replication_read_during_freeze": False,
        "historical_result_can_verify_target": False,
        "paper_or_live_authorized": False,
    }
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_US_LEVERAGED_MULTI_ASSET_V1R_FX_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    load_contract(CONFIG)
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live_files = current_tracked_hashes()
    live_sources = current_visible_source_snapshot()
    parent = verify_parent_manifest()
    v1 = verify_v1_manifest()
    evaluator = verify_evaluator_manifest()
    failures: list[str] = []
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != tracked_content_hash(live_files):
        failures.append("tracked_content_sha256")
    if existing.get("visible_source_snapshot") != live_sources:
        failures.append("visible_source_snapshot")
    if not existing.get("prefreeze_correction_checks", {}).get("all_checks_pass", False):
        failures.append("prefreeze_correction_checks")
    if existing.get("performance_or_future_return_computed_during_freeze") is not False:
        failures.append("freeze_read_future_return")
    if existing.get("sealed_replication_read_during_freeze") is not False:
        failures.append("freeze_read_replication")
    if parent["failure_count"]:
        failures.append("parent_manifest")
    if v1["failure_count"]:
        failures.append("v1_manifest")
    if evaluator["failure_count"]:
        failures.append("evaluator_manifest")
    return {
        "status": (
            "PASS_US_LEVERAGED_MULTI_ASSET_V1R_FX_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_US_LEVERAGED_MULTI_ASSET_V1R_FX_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": tracked_content_hash(live_files),
        "tracked_file_count": int(len(live_files)),
        "parent_manifest_status": parent["status"],
        "v1_manifest_status": v1["status"],
        "conservative_evaluator_manifest_status": evaluator["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证美国杠杆多资产V1R汇率修正版")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
