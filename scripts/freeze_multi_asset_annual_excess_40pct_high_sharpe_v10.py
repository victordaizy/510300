"""冻结或验证40个百分点高夏普研究V10期末容量减仓。"""

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

from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v9 import (
    verify as verify_parent_manifest,
)
from scripts.freeze_us_multi_asset_capacity_aware_absolute_momentum_v3 import (
    verify as verify_v3_manifest,
)


CONFIG = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v10.yaml"
MANIFEST = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v10_manifest.json"
TRACKED_FILES = [
    "config/multi_asset_annual_excess_40pct_high_sharpe_v10.yaml",
    "docs/MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V10_SPEC.md",
    "scripts/freeze_multi_asset_annual_excess_40pct_high_sharpe_v10.py",
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
        raise ValueError("V10扩展协议必须是YAML对象")
    protocol = payload.get("protocol", {})
    objective = payload.get("objective", {})
    lane = payload.get("new_lane", {})
    safety = payload.get("safety", {})
    failures: list[str] = []
    if protocol.get("protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V10":
        failures.append("protocol_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V9":
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
    if lane.get("fixed_tickers") != ["UPRO", "TQQQ", "TLT", "UGL"]:
        failures.append("fixed_tickers")
    if lane.get("selection_formula_unchanged_from_v3") is not True:
        failures.append("selection_formula")
    if float(lane.get("maximum_order_fraction_unchanged", float("nan"))) != 0.0005:
        failures.append("capacity_fraction")
    if int(lane.get("terminal_winddown_common_trading_days", 0)) != 20:
        failures.append("winddown_days")
    if any(bool(safety.get(name, True)) for name in safety):
        failures.append("safety")
    if failures:
        raise ValueError(f"V10扩展协议被弱化或损坏：{sorted(set(failures))}")
    return payload


def tracked_hashes() -> dict[str, str]:
    missing = [item for item in TRACKED_FILES if not (ROOT / item).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少V10冻结文件：{missing}")
    return {item: sha256_file(ROOT / item) for item in TRACKED_FILES}


def content_hash(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_v3_failure(contract: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / contract["inputs"]["v3_failure_report"]
    report = json.loads(path.read_text(encoding="utf-8"))
    required = contract["registered_failure"]
    checks = {
        "candidate": report.get("candidate_id") == required["candidate_id"],
        "status": report.get("status") == required["required_status"],
        "error": report.get("error") == required["required_error"],
        "performance_unavailable": report.get("performance_metrics_available") is False,
        "replication_closed": report.get("decision", {}).get("sealed_replication_open") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V3失败报告不符合V10登记条件：{checks}")
    return {
        "checks": checks,
        "report_path": path.relative_to(ROOT).as_posix(),
        "report_sha256": sha256_file(path),
        "error": report["error"],
    }


def freeze() -> dict[str, Any]:
    contract = load_contract()
    parent = verify_parent_manifest()
    v3 = verify_v3_manifest()
    if parent["failure_count"] or v3["failure_count"]:
        raise RuntimeError("V10父协议或V3冻结清单验证失败")
    failure = verify_v3_failure(contract)
    files = tracked_hashes()
    parent_path = ROOT / contract["inputs"]["parent_manifest"]
    v3_path = ROOT / contract["inputs"]["v3_manifest"]
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V10_MANIFEST",
        "status": contract["protocol"]["status"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_id": contract["protocol"]["protocol_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": content_hash(files),
        "parent_manifest_sha256": sha256_file(parent_path),
        "v3_manifest_sha256": sha256_file(v3_path),
        "parent_manifest_verification_at_freeze": parent,
        "v3_manifest_verification_at_freeze": v3,
        "v3_failure": failure,
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
            "v3_manifest_sha256",
            "v3_failure",
            "objective",
            "new_lane",
        ):
            if existing.get(field) != manifest.get(field):
                raise RuntimeError("现有V10清单与当前内容不同；不得覆盖")
        return existing
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_MULTI_ASSET_40PCT_HIGH_SHARPE_V10_MANIFEST",
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
    if existing.get("v3_manifest_sha256") != sha256_file(
        ROOT / contract["inputs"]["v3_manifest"]
    ):
        failures.append("v3_manifest_snapshot")
    try:
        if existing.get("v3_failure") != verify_v3_failure(contract):
            failures.append("v3_failure")
    except Exception:
        failures.append("v3_failure")
    return {
        "status": (
            "PASS_MULTI_ASSET_40PCT_HIGH_SHARPE_V10_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_MULTI_ASSET_40PCT_HIGH_SHARPE_V10_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": content_hash(live_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证40个百分点高夏普V10扩展协议")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
