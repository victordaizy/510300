"""冻结或验证40个百分点高夏普研究V5扩展协议。"""

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

from scripts.freeze_cb_double_low_rotation_v1 import verify as verify_cb_manifest
from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v4 import (
    verify as verify_parent_manifest,
)


CONFIG = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v5.yaml"
MANIFEST = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v5_manifest.json"
TRACKED_FILES = [
    "config/multi_asset_annual_excess_40pct_high_sharpe_v5.yaml",
    "docs/MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V5_SPEC.md",
    "scripts/freeze_multi_asset_annual_excess_40pct_high_sharpe_v5.py",
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
        raise ValueError("V5扩展协议必须是YAML对象")
    protocol = payload.get("protocol", {})
    objective = payload.get("objective", {})
    lane = payload.get("new_lane", {})
    governance = payload.get("governance", {})
    safety = payload.get("safety", {})
    failures: list[str] = []
    if protocol.get("protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V5":
        failures.append("protocol_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V4":
        failures.append("parent_protocol_id")
    expected_numbers = {
        "initial_capital_cny": 500_000.0,
        "minimum_annualized_net_excess": 0.40,
        "minimum_strategy_net_sharpe": 1.50,
        "user_transaction_fee_rate_per_leg": 0.0001,
    }
    for key, expected in expected_numbers.items():
        try:
            actual = float(objective.get(key, float("nan")))
        except (TypeError, ValueError):
            actual = float("nan")
        if actual != expected:
            failures.append(key)
    if objective.get("benchmark") != "H00300_TOTAL_RETURN":
        failures.append("benchmark")
    if lane.get("lane") != "US_LISTED_DAILY_LEVERAGED_MULTI_ASSET_ABSOLUTE_MOMENTUM":
        failures.append("lane")
    if lane.get("fixed_product_universe") != ["UPRO", "TQQQ", "TMF", "UGL"]:
        failures.append("fixed_product_universe")
    if lane.get("maximum_candidates_before_family_specific_replication") != 1:
        failures.append("maximum_candidates")
    if any(
        bool(lane.get(name, True))
        for name in (
            "account_level_borrowing_allowed",
            "account_level_margin_allowed",
            "short_sale_allowed",
        )
    ):
        failures.append("account_leverage_or_short")
    if not bool(lane.get("product_embedded_daily_leverage_allowed")):
        failures.append("embedded_leverage_disclosure")
    if float(lane.get("maximum_account_gross_exposure", float("nan"))) != 1.0:
        failures.append("maximum_account_gross_exposure")
    if not bool(governance.get("formula_costs_dates_and_universe_must_freeze_before_visible_outcome")):
        failures.append("formula_freeze")
    if governance.get("price_acquisition_may_not_compute_strategy_return_or_rank") is not True:
        failures.append("acquisition_boundary")
    if governance.get("historical_result_may_not_authorize_paper_or_live") is not True:
        failures.append("historical_authorization")
    if any(
        bool(safety.get(name, True))
        for name in (
            "paper_position_generation",
            "shadow_signal_generation",
            "order_generation",
            "broker_connection",
            "live_trading_authorized",
        )
    ):
        failures.append("safety")
    if failures:
        raise ValueError(f"V5扩展协议核心条件被弱化：{sorted(set(failures))}")
    return payload


def tracked_hashes() -> dict[str, str]:
    missing = [relative for relative in TRACKED_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少V5冻结文件：{missing}")
    return {relative: sha256_file(ROOT / relative) for relative in TRACKED_FILES}


def content_hash(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_cb_failure(contract: dict[str, Any]) -> dict[str, Any]:
    report_path = ROOT / contract["inputs"]["cb_failure_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    error = str(report.get("error", ""))
    checks = {
        "failed_contract": report.get("status") == "FAILED_DATA_OR_EXECUTION_CONTRACT",
        "performance_unavailable": report.get("performance_metrics_available") is False,
        "terminal_exit_failure": "评价期末仍有无法退出的可转债持仓" in error,
        "replication_closed": report.get("decision", {}).get("sealed_replication_open")
        is False,
        "historical_target_not_verified": report.get("decision", {}).get(
            "historical_result_verifies_40pct_target"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError("可转债候选尚未形成可登记V5的冻结失败状态")
    return {
        "checks": checks,
        "report_path": report_path.relative_to(ROOT).as_posix(),
        "report_sha256": sha256_file(report_path),
        "error": error,
    }


def freeze() -> dict[str, Any]:
    contract = load_contract()
    parent = verify_parent_manifest()
    cb = verify_cb_manifest()
    if parent["failure_count"] or cb["failure_count"]:
        raise RuntimeError("父协议或可转债候选冻结清单验证失败")
    failure = verify_cb_failure(contract)
    files = tracked_hashes()
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V5_MANIFEST",
        "status": "FROZEN_BEFORE_US_LEVERAGED_MULTI_ASSET_CANDIDATE_OUTCOME",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_id": contract["protocol"]["protocol_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": content_hash(files),
        "parent_manifest_verification": parent,
        "cb_manifest_verification": cb,
        "cb_failure": failure,
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
            "cb_failure",
            "objective",
            "new_lane",
        ):
            if existing.get(field) != manifest.get(field):
                raise RuntimeError("现有V5清单与当前内容不同；不得覆盖")
        return existing
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_MULTI_ASSET_40PCT_HIGH_SHARPE_V5_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    contract = load_contract()
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parent = verify_parent_manifest()
    cb = verify_cb_manifest()
    live_files = tracked_hashes()
    failures: list[str] = []
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != content_hash(live_files):
        failures.append("tracked_content_sha256")
    try:
        failure = verify_cb_failure(contract)
        if existing.get("cb_failure") != failure:
            failures.append("cb_failure")
    except Exception:
        failures.append("cb_failure")
    if parent["failure_count"]:
        failures.append("parent_manifest")
    if cb["failure_count"]:
        failures.append("cb_manifest")
    return {
        "status": (
            "PASS_MULTI_ASSET_40PCT_HIGH_SHARPE_V5_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_MULTI_ASSET_40PCT_HIGH_SHARPE_V5_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": content_hash(live_files),
        "parent_manifest_status": parent["status"],
        "cb_manifest_status": cb["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证40个百分点高夏普V5扩展协议")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    result = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("failure_count", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
