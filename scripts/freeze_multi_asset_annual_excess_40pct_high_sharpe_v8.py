"""冻结或验证40个百分点高夏普研究V8流动性美债重构。"""

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

from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v7 import (
    verify as verify_parent_manifest,
)


CONFIG = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v8.yaml"
MANIFEST = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v8_manifest.json"
TRACKED_FILES = [
    "config/multi_asset_annual_excess_40pct_high_sharpe_v8.yaml",
    "docs/MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V8_SPEC.md",
    "scripts/freeze_multi_asset_annual_excess_40pct_high_sharpe_v8.py",
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
        raise ValueError("V8扩展协议必须是YAML对象")
    protocol = payload.get("protocol", {})
    objective = payload.get("objective", {})
    failure = payload.get("registered_failure", {})
    lane = payload.get("new_lane", {})
    safety = payload.get("safety", {})
    failures: list[str] = []
    if protocol.get("protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V8":
        failures.append("protocol_id")
    if protocol.get("parent_protocol_id") != "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V7":
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
    if objective.get("benchmark") != "H00300_TOTAL_RETURN":
        failures.append("benchmark")
    if failure.get("candidate_id") != "DIGITAL_ASSET_SPOT_VOLATILITY_SCALED_TREND_V1":
        failures.append("registered_candidate")
    if int(failure.get("required_positive_target_dates", 0)) != 180:
        failures.append("required_positive_target_dates")
    if int(failure.get("observed_positive_target_dates", 0)) != 176:
        failures.append("observed_positive_target_dates")
    if lane.get("fixed_tickers") != ["UPRO", "TQQQ", "TLT", "UGL"]:
        failures.append("fixed_tickers")
    if lane.get("replaced_infeasible_product") != "TMF" or lane.get("replacement_product") != "TLT":
        failures.append("replacement")
    if not bool(lane.get("all_signal_dates_windows_ranking_weights_costs_and_gates_unchanged_from_v1")):
        failures.append("formula_preservation")
    if int(lane.get("maximum_candidates_before_family_specific_replication", 0)) != 1:
        failures.append("maximum_candidates")
    if any(
        bool(lane.get(name, True))
        for name in (
            "account_borrowing_allowed",
            "account_margin_allowed",
            "short_sale_allowed",
            "derivative_position_allowed",
        )
    ):
        failures.append("account_leverage_or_short")
    if float(lane.get("maximum_account_gross_exposure", float("nan"))) != 1.0:
        failures.append("gross_exposure")
    if any(bool(safety.get(name, True)) for name in safety):
        failures.append("safety")
    if failures:
        raise ValueError(f"V8扩展协议被弱化或损坏：{sorted(set(failures))}")
    return payload


def tracked_hashes() -> dict[str, str]:
    missing = [item for item in TRACKED_FILES if not (ROOT / item).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少V8冻结文件：{missing}")
    return {item: sha256_file(ROOT / item) for item in TRACKED_FILES}


def content_hash(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_digital_asset_prefreeze_failure(contract: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / contract["inputs"]["digital_asset_prefreeze_failure"]
    report = json.loads(path.read_text(encoding="utf-8"))
    required = contract["registered_failure"]
    checks = {
        "candidate": report.get("candidate_id") == required["candidate_id"],
        "status": report.get("status") == required["required_status"],
        "required_count": report.get("required_signal_dates_with_positive_target_weight")
        == required["required_positive_target_dates"],
        "observed_count": report.get("observed_signal_dates_with_positive_target_weight")
        == required["observed_positive_target_dates"],
        "performance_unavailable": report.get("performance_metrics_available") is False,
        "sealed_not_read": report.get("sealed_replication_read") is False,
        "not_frozen": report.get("candidate_frozen") is False,
        "closed": report.get("candidate_closed") is True,
        "not_reparameterized": report.get("candidate_may_be_reparameterized_after_failure")
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"数字资产冻结前失败报告不符合V8登记条件：{checks}")
    return {
        "checks": checks,
        "report_path": path.relative_to(ROOT).as_posix(),
        "report_sha256": sha256_file(path),
        "required_positive_target_dates": required["required_positive_target_dates"],
        "observed_positive_target_dates": required["observed_positive_target_dates"],
    }


def freeze() -> dict[str, Any]:
    contract = load_contract()
    parent = verify_parent_manifest()
    if parent["failure_count"]:
        raise RuntimeError("V7父协议冻结清单验证失败")
    failure = verify_digital_asset_prefreeze_failure(contract)
    files = tracked_hashes()
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V8_MANIFEST",
        "status": contract["protocol"]["status"],
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_id": contract["protocol"]["protocol_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": content_hash(files),
        "parent_manifest_verification": parent,
        "digital_asset_prefreeze_failure": failure,
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
            "digital_asset_prefreeze_failure",
            "objective",
            "new_lane",
        ):
            if existing.get(field) != manifest.get(field):
                raise RuntimeError("现有V8清单与当前内容不同；不得覆盖")
        return existing
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_MULTI_ASSET_40PCT_HIGH_SHARPE_V8_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    contract = load_contract()
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live_files = tracked_hashes()
    parent = verify_parent_manifest()
    failures: list[str] = []
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != content_hash(live_files):
        failures.append("tracked_content_sha256")
    try:
        if existing.get("digital_asset_prefreeze_failure") != verify_digital_asset_prefreeze_failure(contract):
            failures.append("digital_asset_prefreeze_failure")
    except Exception:
        failures.append("digital_asset_prefreeze_failure")
    if parent["failure_count"]:
        failures.append("parent_manifest")
    return {
        "status": (
            "PASS_MULTI_ASSET_40PCT_HIGH_SHARPE_V8_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_MULTI_ASSET_40PCT_HIGH_SHARPE_V8_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": content_hash(live_files),
        "parent_manifest_status": parent["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证40个百分点高夏普V8扩展协议")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
