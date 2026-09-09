"""冻结或验证40个百分点高夏普研究V19。"""

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

from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v18 import (
    verify as verify_parent_manifest,
)


CONFIG = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v19.yaml"
MANIFEST = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v19_manifest.json"
TRACKED_FILES = [
    "config/multi_asset_annual_excess_40pct_high_sharpe_v19.yaml",
    "docs/MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V19_SPEC.md",
    "scripts/freeze_multi_asset_annual_excess_40pct_high_sharpe_v19.py",
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
        raise ValueError("V19扩展协议必须是YAML对象")
    objective = payload.get("objective", {})
    lane = payload.get("new_lane", {})
    failures: list[str] = []
    if payload.get("protocol", {}).get("protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V19":
        failures.append("protocol_id")
    if payload.get("protocol", {}).get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V18":
        failures.append("parent_protocol_id")
    for key, value in {
        "initial_capital_cny": 500000.0,
        "minimum_annualized_net_excess": 0.40,
        "minimum_strategy_net_sharpe": 1.50,
        "user_transaction_fee_rate_per_leg": 0.0001,
    }.items():
        if float(objective.get(key, float("nan"))) != value:
            failures.append(key)
    if len(lane.get("fixed_symbols", [])) != 15:
        failures.append("fixed_symbols")
    if lane.get("basis_formula") != "SPOT_CLOSE_DIV_PERPETUAL_CLOSE_MINUS_1":
        failures.append("basis_formula")
    if int(lane.get("long_count", 0)) != 3 or int(lane.get("short_count", 0)) != 3:
        failures.append("side_counts")
    if lane.get("traded_instrument") != "PERPETUAL_ONLY" or lane.get("spot_position_allowed") is not False:
        failures.append("traded_instrument")
    if float(lane.get("target_long_gross", float("nan"))) != 0.5:
        failures.append("target_long_gross")
    if float(lane.get("target_short_gross", float("nan"))) != 0.5:
        failures.append("target_short_gross")
    if float(lane.get("maximum_total_gross_exposure", float("nan"))) != 1.0:
        failures.append("maximum_total_gross_exposure")
    if float(lane.get("maximum_order_fraction", float("nan"))) != 0.001:
        failures.append("maximum_order_fraction")
    if float(lane.get("base_total_perpetual_cost_bps_per_leg", float("nan"))) != 4.0:
        failures.append("base_cost")
    if float(lane.get("stress_total_perpetual_cost_bps_per_leg", float("nan"))) != 16.0:
        failures.append("stress_cost")
    if any(bool(value) for value in payload.get("safety", {}).values()):
        failures.append("safety")
    if failures:
        raise ValueError(f"V19扩展协议被弱化或损坏：{sorted(set(failures))}")
    return payload


def tracked_hashes() -> dict[str, str]:
    missing = [item for item in TRACKED_FILES if not (ROOT / item).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少V19冻结文件：{missing}")
    return {item: sha256_file(ROOT / item) for item in TRACKED_FILES}


def content_hash(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_v8_failure(contract: dict[str, Any]) -> dict[str, Any]:
    manifest_path = ROOT / contract["inputs"]["v8_manifest"]
    report_path = ROOT / contract["inputs"]["v8_visible_report"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = contract["registered_v8_failure"]
    checks = {
        "candidate": report.get("candidate_id") == required["candidate_id"],
        "status": report.get("status") == required["required_status"],
        "error_type": report.get("error_type") == required["required_error_type"],
        "error_fragment": required["required_error_fragment"] in str(report.get("error", "")),
        "performance_unavailable": report.get("performance_metrics_available") is False,
        "manifest_verified": report.get("manifest_verification", {}).get("failure_count") == 0,
        "manifest_snapshot": report.get("manifest_verification", {}).get("manifest_sha256")
        == sha256_file(manifest_path),
        "replication_closed": report.get("decision", {}).get("sealed_replication_open") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V8失败不符合V19登记条件：{checks}")
    return {
        "checks": checks,
        "manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(report_path),
        "tracked_content_sha256": manifest["tracked_content_sha256"],
        "error_type": report["error_type"],
        "error": report["error"],
        "performance_metrics_available": False,
    }


def freeze() -> dict[str, Any]:
    contract = load_contract()
    parent = verify_parent_manifest()
    if parent["failure_count"]:
        raise RuntimeError("V19父协议冻结清单验证失败")
    failure = verify_v8_failure(contract)
    files = tracked_hashes()
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V19_MANIFEST",
        "status": contract["protocol"]["status"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_id": contract["protocol"]["protocol_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": content_hash(files),
        "parent_manifest_sha256": sha256_file(ROOT / contract["inputs"]["parent_manifest"]),
        "v8_failure": failure,
        "objective": contract["objective"],
        "new_lane": contract["new_lane"],
        "historical_result_can_verify_target": False,
        "paper_or_live_authorized": False,
    }
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for field in (
            "tracked_files",
            "tracked_content_sha256",
            "parent_manifest_sha256",
            "v8_failure",
            "objective",
            "new_lane",
        ):
            if existing.get(field) != manifest.get(field):
                raise RuntimeError("现有V19清单与当前内容不同；不得覆盖")
        return existing
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_MULTI_ASSET_40PCT_HIGH_SHARPE_V19_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    contract = load_contract()
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live_files = tracked_hashes()
    failures: list[str] = []
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != content_hash(live_files):
        failures.append("tracked_content_sha256")
    if existing.get("parent_manifest_sha256") != sha256_file(
        ROOT / contract["inputs"]["parent_manifest"]
    ):
        failures.append("parent_manifest_snapshot")
    try:
        if existing.get("v8_failure") != verify_v8_failure(contract):
            failures.append("v8_failure")
    except Exception:
        failures.append("v8_failure")
    parent = verify_parent_manifest()
    if parent["failure_count"]:
        failures.append("parent_manifest")
    return {
        "status": (
            "PASS_MULTI_ASSET_40PCT_HIGH_SHARPE_V19_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_MULTI_ASSET_40PCT_HIGH_SHARPE_V19_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": content_hash(live_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证40个百分点高夏普V19扩展协议")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
