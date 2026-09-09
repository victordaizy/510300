"""冻结或验证40个百分点高夏普研究V23。"""

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

from scripts.freeze_digital_asset_basis_ranked_spot_long_v12r_schema_correction import (
    verify as verify_v12r_manifest,
)
from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v22 import (
    verify as verify_parent_manifest,
)


CONFIG = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v23.yaml"
MANIFEST = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v23_manifest.json"
TRACKED_FILES = [
    "config/multi_asset_annual_excess_40pct_high_sharpe_v23.yaml",
    "docs/MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V23_SPEC.md",
    "scripts/freeze_multi_asset_annual_excess_40pct_high_sharpe_v23.py",
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
        raise ValueError("V23扩展协议必须是YAML对象")
    objective = payload.get("objective", {})
    lane = payload.get("new_lane", {})
    failures: list[str] = []
    if payload.get("protocol", {}).get("protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V23":
        failures.append("protocol_id")
    if payload.get("protocol", {}).get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V22":
        failures.append("parent_protocol_id")
    for key, value in {
        "initial_capital_cny": 500000.0,
        "minimum_annualized_net_excess": 0.40,
        "minimum_strategy_net_sharpe": 1.50,
        "user_transaction_fee_rate_per_leg": 0.0001,
    }.items():
        if float(objective.get(key, float("nan"))) != value:
            failures.append(key)
    if lane.get("fixed_symbols") != ["BTCUSDT", "ETHUSDT"]:
        failures.append("fixed_symbols")
    for key, expected in {
        "target_long_gross": 0.45,
        "target_short_gross": 0.45,
        "target_total_gross": 0.90,
        "maximum_order_fraction": 0.001,
        "base_total_perpetual_cost_bps_per_leg": 4.0,
        "stress_total_perpetual_cost_bps_per_leg": 16.0,
    }.items():
        if float(lane.get(key, float("nan"))) != expected:
            failures.append(key)
    if lane.get("actual_funding_history_required") is not True:
        failures.append("funding_history")
    if any(bool(value) for value in payload.get("safety", {}).values()):
        failures.append("safety")
    if failures:
        raise ValueError(f"V23扩展协议被弱化或损坏：{sorted(set(failures))}")
    return payload


def tracked_hashes() -> dict[str, str]:
    missing = [item for item in TRACKED_FILES if not (ROOT / item).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少V23冻结文件：{missing}")
    return {item: sha256_file(ROOT / item) for item in TRACKED_FILES}


def content_hash(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_v12r_rejection(contract: dict[str, Any]) -> dict[str, Any]:
    manifest_path = ROOT / contract["inputs"]["v12r_manifest"]
    report_path = ROOT / contract["inputs"]["v12r_visible_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    required = contract["registered_v12r_rejection"]
    verification = verify_v12r_manifest()
    metrics = report.get("evaluation", {}).get("metrics", {})
    checks = {
        "candidate": report.get("candidate_id") == required["candidate_id"],
        "status": report.get("status") == required["required_status"],
        "all_gates": report.get("evaluation", {}).get("all_visible_gates_pass")
        is required["required_all_visible_gates_pass"],
        "base_excess": metrics.get("base_annualized_excess")
        == required["required_base_annualized_excess"],
        "stress_excess": metrics.get("stress_annualized_excess")
        == required["required_stress_annualized_excess"],
        "base_sharpe": metrics.get("base_strategy_net_sharpe")
        == required["required_base_sharpe"],
        "stress_sharpe": metrics.get("stress_strategy_net_sharpe")
        == required["required_stress_sharpe"],
        "manifest_verified": verification.get("failure_count") == 0,
        "manifest_snapshot": report.get("manifest_verification", {}).get("manifest_sha256")
        == sha256_file(manifest_path),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V12R拒绝不符合V23登记条件：{checks}")
    return {
        "checks": checks,
        "manifest_sha256": sha256_file(manifest_path),
        "report_sha256": sha256_file(report_path),
        "base_annualized_excess": metrics["base_annualized_excess"],
        "stress_annualized_excess": metrics["stress_annualized_excess"],
        "base_sharpe": metrics["base_strategy_net_sharpe"],
        "stress_sharpe": metrics["stress_strategy_net_sharpe"],
    }


def freeze() -> dict[str, Any]:
    contract = load_contract()
    parent = verify_parent_manifest()
    if parent["failure_count"]:
        raise RuntimeError("V23父协议冻结清单验证失败")
    rejection = verify_v12r_rejection(contract)
    files = tracked_hashes()
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V23_MANIFEST",
        "status": contract["protocol"]["status"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_id": contract["protocol"]["protocol_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": content_hash(files),
        "parent_manifest_sha256": sha256_file(ROOT / contract["inputs"]["parent_manifest"]),
        "v12r_rejection": rejection,
        "objective": contract["objective"],
        "new_lane": contract["new_lane"],
        "basis_rank_or_return_computed_during_freeze": False,
        "historical_result_can_verify_target": False,
        "paper_or_live_authorized": False,
    }
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for field in (
            "tracked_files",
            "tracked_content_sha256",
            "parent_manifest_sha256",
            "v12r_rejection",
            "objective",
            "new_lane",
        ):
            if existing.get(field) != manifest.get(field):
                raise RuntimeError("现有V23清单与当前内容不同；不得覆盖")
        return existing
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_MULTI_ASSET_40PCT_HIGH_SHARPE_V23_MANIFEST",
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
    try:
        if existing.get("v12r_rejection") != verify_v12r_rejection(contract):
            failures.append("v12r_rejection")
    except Exception:
        failures.append("v12r_rejection")
    parent = verify_parent_manifest()
    if parent["failure_count"]:
        failures.append("parent_manifest")
    return {
        "status": (
            "PASS_MULTI_ASSET_40PCT_HIGH_SHARPE_V23_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_MULTI_ASSET_40PCT_HIGH_SHARPE_V23_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": content_hash(live_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证40个百分点高夏普V23扩展协议")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
