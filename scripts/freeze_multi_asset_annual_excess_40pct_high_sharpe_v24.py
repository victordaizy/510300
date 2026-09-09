"""冻结或验证年化净超额40个百分点高夏普研究V24。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from scripts.freeze_digital_asset_current_quarter_signal_perpetual_factor_v13 import (
    verify as verify_v13_manifest,
)
from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v23 import (
    verify as verify_parent_manifest,
)


CONFIG = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v24.yaml"
MANIFEST = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v24_manifest.json"
TRACKED_FILES = [
    "config/multi_asset_annual_excess_40pct_high_sharpe_v24.yaml",
    "docs/MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V24_SPEC.md",
    "scripts/freeze_multi_asset_annual_excess_40pct_high_sharpe_v24.py",
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
        raise ValueError("V24配置必须是YAML对象")
    failures: list[str] = []
    protocol = payload.get("protocol", {})
    if protocol.get("protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V24":
        failures.append("protocol_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V23":
        failures.append("parent_protocol_id")
    objective = payload.get("objective", {})
    for key, expected in {
        "initial_capital_cny": 500000.0,
        "minimum_annualized_net_excess": 0.40,
        "minimum_strategy_net_sharpe": 1.50,
        "user_transaction_fee_rate_per_leg": 0.0001,
    }.items():
        if float(objective.get(key, math.nan)) != expected:
            failures.append(key)
    if objective.get("benchmark") != "H00300_TOTAL_RETURN":
        failures.append("benchmark")
    if objective.get("base_and_stress_must_both_pass") is not True:
        failures.append("base_and_stress")
    lane = payload.get("new_lane", {})
    fixed_lane: dict[str, Any] = {
        "candidate_id": "DIGITAL_ASSET_CURRENT_QUARTER_SIGNAL_PERPETUAL_FACTOR_V14",
        "fixed_symbols": ["BTCUSDT", "ETHUSDT"],
        "signal": "FIVE_DAY_MEAN_LOG_SPOT_DIV_CURRENT_QUARTER_FUTURES_BASIS",
        "long_rule": "LONG_HIGHER_MEAN_BASIS_PERPETUAL",
        "short_rule": "SHORT_LOWER_MEAN_BASIS_PERPETUAL",
        "target_long_gross": 0.45,
        "target_short_gross": 0.45,
        "target_total_gross": 0.90,
        "funding_notional_price_policy": "OFFICIAL_MARK_PRICE_ELSE_SAME_SETTLEMENT_HOUR_PERPETUAL_OPEN",
        "base_total_perpetual_cost_bps_per_leg": 4.0,
        "stress_total_perpetual_cost_bps_per_leg": 16.0,
        "parameter_grid_search_allowed": False,
    }
    for key, expected in fixed_lane.items():
        value = lane.get(key)
        if isinstance(expected, bool):
            if value is not expected:
                failures.append(key)
        elif isinstance(expected, float):
            if float(value if value is not None else math.nan) != expected:
                failures.append(key)
        elif value != expected:
            failures.append(key)
    if any(bool(value) for value in payload.get("safety", {}).values()):
        failures.append("safety")
    if failures:
        raise ValueError(f"V24配置被弱化或损坏：{sorted(set(failures))}")
    return payload


def current_tracked_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in TRACKED_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"缺少V24冻结文件：{relative}")
        result[relative] = sha256_file(path)
    return result


def tracked_content_hash(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def v13_failure_snapshot(contract: dict[str, Any]) -> dict[str, Any]:
    report_path = ROOT / contract["inputs"]["v13_visible_report"]
    manifest_path = ROOT / contract["inputs"]["v13_manifest"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = contract["registered_v13_failure"]
    checks = {
        "candidate": report.get("candidate_id") == required["candidate_id"],
        "status": report.get("status") == required["required_status"],
        "error_type": report.get("error_type") == required["required_error_type"],
        "error": report.get("error") == required["required_error"],
        "no_performance": report.get("performance_metrics_available")
        is required["required_performance_metrics_available"],
        "manifest_status": report.get("manifest_verification", {}).get("status")
        == required["required_manifest_status"],
        "tracked_hash": report.get("manifest_verification", {}).get(
            "tracked_content_sha256"
        )
        == required["required_tracked_content_sha256"],
        "v13_manifest_verified": verify_v13_manifest()["failure_count"] == 0,
    }
    return {
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "report_sha256": sha256_file(report_path),
        "manifest_sha256": sha256_file(manifest_path),
        "performance_metrics_available": False,
    }


def freeze() -> dict[str, Any]:
    contract = load_contract()
    if MANIFEST.exists():
        verification = verify()
        if verification["failure_count"]:
            raise RuntimeError(f"现有V24清单验证失败：{verification['failures']}")
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    parent = verify_parent_manifest()
    v13 = verify_v13_manifest()
    failure = v13_failure_snapshot(contract)
    if parent["failure_count"] or v13["failure_count"] or not failure["all_checks_pass"]:
        raise RuntimeError("V24依赖或V13失败登记不完整")
    files = current_tracked_hashes()
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V24_MANIFEST",
        "status": "FROZEN_BEFORE_FUNDING_MARK_FALLBACK_FACTOR_OUTCOME",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_id": contract["protocol"]["protocol_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": tracked_content_hash(files),
        "parent_verification": parent,
        "v13_verification": v13,
        "v13_failure": failure,
        "objective": contract["objective"],
        "new_lane": contract["new_lane"],
        "basis_rank_or_return_computed_during_freeze": False,
        "historical_result_can_verify_target": False,
        "paper_or_live_authorized": False,
    }
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_MULTI_ASSET_40PCT_HIGH_SHARPE_V24_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    contract = load_contract()
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live_files = current_tracked_hashes()
    parent = verify_parent_manifest()
    v13 = verify_v13_manifest()
    failure = v13_failure_snapshot(contract)
    failures: list[str] = []
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != tracked_content_hash(live_files):
        failures.append("tracked_content_sha256")
    if existing.get("v13_failure") != failure:
        failures.append("v13_failure")
    if parent["failure_count"]:
        failures.append("parent_manifest")
    if v13["failure_count"]:
        failures.append("v13_manifest")
    if existing.get("basis_rank_or_return_computed_during_freeze") is not False:
        failures.append("prefreeze_outcome")
    return {
        "status": (
            "PASS_MULTI_ASSET_40PCT_HIGH_SHARPE_V24_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_MULTI_ASSET_40PCT_HIGH_SHARPE_V24_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": tracked_content_hash(live_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证V24")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
