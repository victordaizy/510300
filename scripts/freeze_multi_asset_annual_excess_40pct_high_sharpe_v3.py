"""冻结或验证40个百分点高夏普研究V3扩展协议。"""

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

from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v2 import (
    verify as verify_parent_manifest,
)
from scripts.freeze_qdii_cross_market_discount_reversion_v1 import (
    verify as verify_qdii_v1_manifest,
)
from scripts.freeze_qdii_cross_market_discount_reversion_zero_variance_v2 import (
    verify as verify_qdii_v2_manifest,
)


CONFIG = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v3.yaml"
MANIFEST = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v3_manifest.json"
TRACKED_FILES = [
    "config/multi_asset_annual_excess_40pct_high_sharpe_v3.yaml",
    "docs/MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V3_SPEC.md",
    "scripts/freeze_multi_asset_annual_excess_40pct_high_sharpe_v3.py",
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
        raise ValueError("V3扩展协议必须是YAML对象")
    objective = payload.get("objective", {})
    failures: list[str] = []
    expected = {
        "initial_capital_cny": 500000.0,
        "minimum_annualized_net_excess": 0.40,
        "minimum_strategy_net_sharpe": 1.50,
        "user_transaction_fee_rate_per_leg": 0.0001,
    }
    for key, value in expected.items():
        if float(objective.get(key, -1.0)) != value:
            failures.append(key)
    if objective.get("benchmark") != "H00300_TOTAL_RETURN":
        failures.append("benchmark")
    lane = payload.get("new_lane", {})
    if lane.get("lane") != "A_SHARE_PRICE_LIMIT_CONTINUATION":
        failures.append("lane")
    if lane.get("maximum_candidates_before_family_specific_replication") != 1:
        failures.append("maximum_candidates")
    if bool(lane.get("short_sale_allowed", True)) or bool(
        lane.get("leverage_allowed", True)
    ):
        failures.append("short_or_leverage")
    if any(bool(payload.get("safety", {}).get(name, True)) for name in (
        "paper_position_generation",
        "shadow_signal_generation",
        "order_generation",
        "broker_connection",
        "live_trading_authorized",
    )):
        failures.append("safety")
    if failures:
        raise ValueError(f"V3扩展协议核心条件被弱化：{sorted(set(failures))}")
    return payload


def tracked_hashes() -> dict[str, str]:
    missing = [relative for relative in TRACKED_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少V3冻结文件：{missing}")
    return {relative: sha256_file(ROOT / relative) for relative in TRACKED_FILES}


def content_hash(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_qdii_rejection(contract: dict[str, Any]) -> dict[str, Any]:
    inputs = contract["inputs"]
    v1_report_path = ROOT / inputs["qdii_v1_failure_report"]
    v2_report_path = ROOT / inputs["qdii_v2_rejection_report"]
    v1_report = json.loads(v1_report_path.read_text(encoding="utf-8"))
    v2_report = json.loads(v2_report_path.read_text(encoding="utf-8"))
    checks = {
        "v1_failure_preserved": v1_report.get("status")
        == "FAILED_DATA_OR_EXECUTION_CONTRACT",
        "v1_performance_unavailable": v1_report.get("performance_metrics_available")
        is False,
        "v2_rejected": v2_report.get("status")
        == "REJECTED_VISIBLE_40PCT_OR_HIGH_SHARPE_GATE_FROZEN",
        "v2_all_visible_gates_failed": v2_report.get("evaluation", {}).get(
            "all_visible_gates_pass"
        )
        is False,
        "v2_sealed_closed": v2_report.get("decision", {}).get(
            "sealed_replication_open"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError("QDII旧路线尚未形成可登记V3的冻结拒绝状态")
    return {
        "checks": checks,
        "v1_report_path": v1_report_path.relative_to(ROOT).as_posix(),
        "v1_report_sha256": sha256_file(v1_report_path),
        "v2_report_path": v2_report_path.relative_to(ROOT).as_posix(),
        "v2_report_sha256": sha256_file(v2_report_path),
    }


def freeze() -> dict[str, Any]:
    contract = load_contract()
    parent = verify_parent_manifest()
    qdii_v1 = verify_qdii_v1_manifest()
    qdii_v2 = verify_qdii_v2_manifest()
    if parent["failure_count"] or qdii_v1["failure_count"] or qdii_v2["failure_count"]:
        raise RuntimeError("父协议或QDII冻结清单验证失败")
    qdii_rejection = verify_qdii_rejection(contract)
    files = tracked_hashes()
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V3_MANIFEST",
        "status": "FROZEN_BEFORE_A_SHARE_LIMIT_UP_CANDIDATE_OUTCOME",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_id": contract["protocol"]["protocol_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": content_hash(files),
        "parent_manifest_verification": parent,
        "qdii_v1_manifest_verification": qdii_v1,
        "qdii_v2_manifest_verification": qdii_v2,
        "qdii_rejection": qdii_rejection,
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
            "qdii_rejection",
            "objective",
            "new_lane",
        ):
            if existing.get(field) != manifest.get(field):
                raise RuntimeError("现有V3清单与当前内容不同；不得覆盖")
        return existing
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_MULTI_ASSET_40PCT_HIGH_SHARPE_V3_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    contract = load_contract()
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parent = verify_parent_manifest()
    qdii_v1 = verify_qdii_v1_manifest()
    qdii_v2 = verify_qdii_v2_manifest()
    live_files = tracked_hashes()
    failures: list[str] = []
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != content_hash(live_files):
        failures.append("tracked_content_sha256")
    try:
        rejection = verify_qdii_rejection(contract)
        if existing.get("qdii_rejection") != rejection:
            failures.append("qdii_rejection")
    except Exception:
        failures.append("qdii_rejection")
    if parent["failure_count"]:
        failures.append("parent_manifest")
    if qdii_v1["failure_count"]:
        failures.append("qdii_v1_manifest")
    if qdii_v2["failure_count"]:
        failures.append("qdii_v2_manifest")
    return {
        "status": (
            "PASS_MULTI_ASSET_40PCT_HIGH_SHARPE_V3_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_MULTI_ASSET_40PCT_HIGH_SHARPE_V3_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": content_hash(live_files),
        "parent_manifest_status": parent["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证40个百分点高夏普V3扩展协议")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    result = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result.get("failure_count", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
