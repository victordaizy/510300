"""冻结或验证BTC现货容量感知长周期趋势V3。"""

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

from research.btc_spot_capacity_aware_long_horizon_trend_v3 import CONFIG, load_contract
from scripts.freeze_btc_spot_long_horizon_trend_v2 import (
    prefreeze_data_coverage,
    verify as verify_v2_manifest,
)
from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v12 import (
    verify as verify_parent_manifest,
)
from scripts.freeze_qdii_cross_market_discount_reversion_zero_variance_v2 import (
    verify as verify_evaluator_manifest,
)


MANIFEST = ROOT / "config" / "btc_spot_capacity_aware_long_horizon_trend_v3_manifest.json"
TRACKED_FILES = [
    "config/btc_spot_capacity_aware_long_horizon_trend_v3.yaml",
    "docs/BTC_SPOT_CAPACITY_AWARE_LONG_HORIZON_TREND_V3_SPEC.md",
    "research/btc_spot_capacity_aware_long_horizon_trend_v3.py",
    "scripts/freeze_btc_spot_capacity_aware_long_horizon_trend_v3.py",
    "scripts/run_btc_spot_capacity_aware_long_horizon_trend_v3.py",
    "tests/test_btc_spot_capacity_aware_long_horizon_trend_v3.py",
]
VISIBLE_SOURCE_DEPENDENCIES = [
    "data/raw/digital_asset_spot_vol_scaled_trend_v1/collection_status.json",
    "data/raw/digital_asset_spot_vol_scaled_trend_v1/product_master.parquet",
    "data/raw/digital_asset_spot_vol_scaled_trend_v1/visible_panel_through_2022.parquet",
    "data/raw/digital_asset_spot_vol_scaled_trend_v1/visible_fx_through_2022.parquet",
    "data/raw/digital_asset_spot_vol_scaled_trend_v1/visible_H00300_through_2022.parquet",
    "config/multi_asset_annual_excess_40pct_high_sharpe_v12_manifest.json",
    "config/btc_spot_long_horizon_trend_v2_manifest.json",
    "config/qdii_cross_market_discount_reversion_zero_variance_v2.yaml",
    "config/qdii_cross_market_discount_reversion_zero_variance_v2_manifest.json",
    "research/qdii_cross_market_discount_reversion_zero_variance_v2.py",
    "research/btc_spot_long_horizon_trend_v2.py",
    "research/digital_asset_spot_volatility_scaled_trend_v1.py",
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
        raise FileNotFoundError(f"缺少BTC V3冻结文件：{relative}")
    return {"path": relative, "bytes": int(path.stat().st_size), "sha256": sha256_file(path)}


def current_tracked_hashes() -> dict[str, str]:
    return {relative: file_record(relative)["sha256"] for relative in TRACKED_FILES}


def tracked_content_hash(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def current_visible_source_snapshot() -> dict[str, dict[str, Any]]:
    return {relative: file_record(relative) for relative in VISIBLE_SOURCE_DEPENDENCIES}


def _assert_visible_outputs_absent(contract: dict[str, Any]) -> None:
    outputs = contract["outputs"]
    names = (
        "input_audit_json", "visible_daily_returns", "visible_targets",
        "visible_trades", "visible_report_json", "visible_report_markdown",
    )
    existing = [outputs[name] for name in names if (ROOT / outputs[name]).exists()]
    if existing:
        raise RuntimeError(f"冻结前已存在BTC V3可见结果：{existing}")


def _dependencies() -> dict[str, dict[str, Any]]:
    return {
        "parent": verify_parent_manifest(),
        "v2": verify_v2_manifest(),
        "evaluator": verify_evaluator_manifest(),
    }


def freeze() -> dict[str, Any]:
    contract = load_contract(CONFIG)
    if MANIFEST.exists():
        verification = verify()
        if verification["failure_count"]:
            raise RuntimeError(f"现有BTC V3清单验证失败：{verification['failures']}")
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    _assert_visible_outputs_absent(contract)
    dependencies = _dependencies()
    if any(item["failure_count"] for item in dependencies.values()):
        raise RuntimeError("BTC V3父协议、V2或评价器冻结清单验证失败")
    files = current_tracked_hashes()
    sources = current_visible_source_snapshot()
    coverage = prefreeze_data_coverage(contract)
    if not coverage["all_checks_pass"]:
        raise RuntimeError(f"BTC V3冻结前原始数据覆盖门失败：{coverage['checks']}")
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "BTC_SPOT_CAPACITY_AWARE_LONG_HORIZON_TREND_V3_MANIFEST",
        "status": "FROZEN_BEFORE_VISIBLE_OUTCOME",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": tracked_content_hash(files),
        "visible_source_snapshot": sources,
        "dependency_verifications": dependencies,
        "prefreeze_data_coverage": coverage,
        "initial_capital_cny": float(contract["account"]["initial_capital_cny"]),
        "user_transaction_fee_rate_per_leg": float(contract["account"]["user_transaction_fee_rate_per_leg"]),
        "minimum_annualized_net_excess": float(contract["visible_gates"]["minimum_annualized_net_excess"]),
        "minimum_strategy_net_sharpe": float(contract["visible_gates"]["minimum_strategy_net_sharpe"]),
        "maximum_order_fraction": float(
            contract["universe"]["maximum_order_fraction_of_prior_median_dollar_turnover"]
        ),
        "terminal_winddown_calendar_days": int(contract["portfolio"]["terminal_winddown_calendar_days"]),
        "performance_or_future_return_computed_during_freeze": False,
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
            "status": "FAIL_MISSING_BTC_CAPACITY_AWARE_V3_MANIFEST",
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
    if existing.get("terminal_winddown_calendar_days") != 30:
        failures.append("terminal_winddown_calendar_days")
    if existing.get("performance_or_future_return_computed_during_freeze") is not False:
        failures.append("freeze_read_future_return")
    if existing.get("sealed_replication_read_during_freeze") is not False:
        failures.append("freeze_read_replication")
    for name, item in dependencies.items():
        if item["failure_count"]:
            failures.append(f"{name}_manifest")
    return {
        "status": (
            "PASS_BTC_CAPACITY_AWARE_V3_MANIFEST_VERIFIED"
            if not failures else "FAIL_BTC_CAPACITY_AWARE_V3_MANIFEST"
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
    parser = argparse.ArgumentParser(description="冻结或验证BTC现货容量感知长周期趋势V3")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
