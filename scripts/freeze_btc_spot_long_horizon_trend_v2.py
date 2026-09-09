"""冻结或验证BTC现货长周期趋势V2。"""

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

from research.btc_spot_long_horizon_trend_v2 import (
    CONFIG,
    load_contract,
    load_visible_inputs,
)
from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v11 import (
    verify as verify_parent_manifest,
)
from scripts.freeze_qdii_cross_market_discount_reversion_zero_variance_v2 import (
    verify as verify_evaluator_manifest,
)


MANIFEST = ROOT / "config" / "btc_spot_long_horizon_trend_v2_manifest.json"
TRACKED_FILES = [
    "config/btc_spot_long_horizon_trend_v2.yaml",
    "docs/BTC_SPOT_LONG_HORIZON_TREND_V2_SPEC.md",
    "research/btc_spot_long_horizon_trend_v2.py",
    "scripts/freeze_btc_spot_long_horizon_trend_v2.py",
    "scripts/run_btc_spot_long_horizon_trend_v2.py",
    "tests/test_btc_spot_long_horizon_trend_v2.py",
]
VISIBLE_SOURCE_DEPENDENCIES = [
    "data/raw/digital_asset_spot_vol_scaled_trend_v1/collection_status.json",
    "data/raw/digital_asset_spot_vol_scaled_trend_v1/product_master.parquet",
    "data/raw/digital_asset_spot_vol_scaled_trend_v1/visible_panel_through_2022.parquet",
    "data/raw/digital_asset_spot_vol_scaled_trend_v1/visible_fx_through_2022.parquet",
    "data/raw/digital_asset_spot_vol_scaled_trend_v1/visible_H00300_through_2022.parquet",
    "config/multi_asset_annual_excess_40pct_high_sharpe_v11_manifest.json",
    "config/qdii_cross_market_discount_reversion_zero_variance_v2.yaml",
    "config/qdii_cross_market_discount_reversion_zero_variance_v2_manifest.json",
    "research/qdii_cross_market_discount_reversion_zero_variance_v2.py",
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
        raise FileNotFoundError(f"缺少BTC长周期候选冻结文件：{relative}")
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
    panel, master, _fx, _benchmark, input_audit = load_visible_inputs(
        contract,
        signal_only=True,
    )
    dates = pd.DatetimeIndex(pd.to_datetime(panel["date"])).sort_values().unique()
    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    start = pd.Timestamp(contract["historical_partition"]["visible_start"])
    end = pd.Timestamp(contract["historical_partition"]["visible_end"])
    sundays = dates[dates.weekday == 6]
    executions = sundays + pd.Timedelta(days=1)
    signal_date_count = int(((executions >= start) & (executions <= end)).sum())
    gate = contract["prefreeze_coverage_gate"]
    checks = {
        "required_products": int(panel["product_id"].nunique())
        == int(gate["required_products"])
        and int(master["product_id"].nunique()) == int(gate["required_products"]),
        "minimum_visible_panel_rows": int(len(panel))
        >= int(gate["minimum_visible_panel_rows"]),
        "minimum_visible_complete_calendar_days": int(len(dates))
        >= int(gate["minimum_visible_complete_calendar_days"])
        and dates.equals(expected),
        "minimum_visible_signal_dates": signal_date_count
        >= int(gate["minimum_visible_signal_dates"]),
        "future_open_or_strategy_return_not_read": input_audit[
            "future_open_or_strategy_return_read"
        ]
        is False,
        "sealed_replication_not_read": input_audit["sealed_inputs_read"] is False,
        "signal_direction_or_target_count_not_computed": gate[
            "signal_direction_or_target_count_may_be_used"
        ]
        is False,
    }
    return {
        "status": (
            "PASS_PREFREEZE_BTC_LONG_HORIZON_RAW_DATA_COVERAGE"
            if all(checks.values())
            else "FAIL_PREFREEZE_BTC_LONG_HORIZON_RAW_DATA_COVERAGE"
        ),
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "visible_panel_rows": int(len(panel)),
        "product_count": int(panel["product_id"].nunique()),
        "complete_calendar_day_count": int(len(dates)),
        "signal_date_count": signal_date_count,
        "first_input_date": dates.min().date().isoformat(),
        "last_input_date": dates.max().date().isoformat(),
        "signal_direction_or_target_count_computed": False,
        "performance_or_future_return_computed": False,
        "future_open_read": False,
        "sealed_replication_read": False,
    }


def _assert_visible_outputs_absent(contract: dict[str, Any]) -> None:
    outputs = contract["outputs"]
    names = (
        "input_audit_json",
        "visible_daily_returns",
        "visible_targets",
        "visible_trades",
        "visible_report_json",
        "visible_report_markdown",
    )
    existing = [outputs[name] for name in names if (ROOT / outputs[name]).exists()]
    if existing:
        raise RuntimeError(f"冻结前已存在BTC长周期候选可见结果：{existing}")


def _dependency_verifications() -> dict[str, dict[str, Any]]:
    return {
        "parent": verify_parent_manifest(),
        "evaluator": verify_evaluator_manifest(),
    }


def freeze() -> dict[str, Any]:
    contract = load_contract(CONFIG)
    if MANIFEST.exists():
        verification = verify()
        if verification["failure_count"]:
            raise RuntimeError(f"现有BTC候选清单验证失败：{verification['failures']}")
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    _assert_visible_outputs_absent(contract)
    dependencies = _dependency_verifications()
    if any(item["failure_count"] for item in dependencies.values()):
        raise RuntimeError("BTC候选父协议或保守评价器冻结清单验证失败")
    files = current_tracked_hashes()
    sources = current_visible_source_snapshot()
    coverage = prefreeze_data_coverage(contract)
    if not coverage["all_checks_pass"]:
        raise RuntimeError(f"BTC候选冻结前原始数据覆盖门失败：{coverage['checks']}")
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "BTC_SPOT_LONG_HORIZON_TREND_V2_MANIFEST",
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
            "status": "FAIL_MISSING_BTC_LONG_HORIZON_TREND_V2_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    load_contract(CONFIG)
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live_files = current_tracked_hashes()
    live_sources = current_visible_source_snapshot()
    dependencies = _dependency_verifications()
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
    if existing.get("signal_direction_or_target_count_computed_during_freeze") is not False:
        failures.append("freeze_computed_signal_direction")
    if existing.get("sealed_replication_read_during_freeze") is not False:
        failures.append("freeze_read_replication")
    for name, item in dependencies.items():
        if item["failure_count"]:
            failures.append(f"{name}_manifest")
    return {
        "status": (
            "PASS_BTC_LONG_HORIZON_TREND_V2_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_BTC_LONG_HORIZON_TREND_V2_MANIFEST"
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
    parser = argparse.ArgumentParser(description="冻结或验证BTC现货长周期趋势V2")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
