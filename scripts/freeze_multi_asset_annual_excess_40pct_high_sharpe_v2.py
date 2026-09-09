"""冻结或验证40个百分点高夏普研究V2扩展协议。"""

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

from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v1 import (
    verify as verify_parent_manifest,
)


CONFIG = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v2.yaml"
MANIFEST = ROOT / "config" / "multi_asset_annual_excess_40pct_high_sharpe_v2_manifest.json"
TRACKED_FILES = [
    "config/multi_asset_annual_excess_40pct_high_sharpe_v2.yaml",
    "docs/MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V2_SPEC.md",
    "scripts/freeze_multi_asset_annual_excess_40pct_high_sharpe_v2.py",
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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_contract() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("V2扩展协议必须是YAML对象")
    objective = payload.get("objective", {})
    expected = {
        "initial_capital_cny": 500000.0,
        "minimum_annualized_net_excess": 0.40,
        "minimum_strategy_net_sharpe": 1.50,
        "user_transaction_fee_rate_per_leg": 0.0001,
    }
    failures = [
        key for key, value in expected.items() if float(objective.get(key, -1.0)) != value
    ]
    if objective.get("benchmark") != "H00300_TOTAL_RETURN":
        failures.append("benchmark")
    if payload.get("new_lane", {}).get("maximum_candidates_before_new_holdout") != 1:
        failures.append("maximum_candidates_before_new_holdout")
    if failures:
        raise ValueError(f"V2扩展协议核心条件被弱化：{sorted(failures)}")
    return payload


def tracked_hashes() -> dict[str, str]:
    missing = [relative for relative in TRACKED_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少V2冻结文件：{missing}")
    return {relative: sha256_file(ROOT / relative) for relative in TRACKED_FILES}


def content_hash(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_closeout(contract: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / contract["inputs"]["v1_closeout_json"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "goal_not_achieved": payload.get("goal_achieved") is False,
        "no_surviving_candidates": not payload.get("aggregate_decision", {}).get(
            "surviving_candidates"
        ),
        "sealed_periods_not_opened": payload.get("aggregate_decision", {}).get(
            "sealed_replication_periods_opened"
        )
        is False,
    }
    if not all(checks.values()):
        raise RuntimeError("V1结案状态不允许登记V2新路线")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(path),
        "checks": checks,
    }


def freeze() -> dict[str, Any]:
    contract = load_contract()
    parent = verify_parent_manifest()
    if parent["failure_count"]:
        raise RuntimeError("V1父协议冻结清单验证失败")
    closeout = verify_closeout(contract)
    files = tracked_hashes()
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "MULTI_ASSET_ANNUAL_EXCESS_40PCT_HIGH_SHARPE_V2_MANIFEST",
        "status": "FROZEN_BEFORE_QDII_RELATIVE_VALUE_CANDIDATE_OUTCOME",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "protocol_id": contract["protocol"]["protocol_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": content_hash(files),
        "parent_manifest_verification": parent,
        "v1_closeout": closeout,
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
            "v1_closeout",
            "objective",
            "new_lane",
        ):
            if existing.get(field) != manifest.get(field):
                raise RuntimeError("现有V2清单与当前内容不同；不得覆盖")
        return existing
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_V2_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    contract = load_contract()
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parent = verify_parent_manifest()
    failures: list[str] = []
    live_files = tracked_hashes()
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != content_hash(live_files):
        failures.append("tracked_content_sha256")
    try:
        closeout = verify_closeout(contract)
        if existing.get("v1_closeout") != closeout:
            failures.append("v1_closeout")
    except Exception:
        failures.append("v1_closeout")
    if parent["failure_count"]:
        failures.append("parent_manifest")
    return {
        "status": (
            "PASS_MULTI_ASSET_40PCT_HIGH_SHARPE_V2_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_MULTI_ASSET_40PCT_HIGH_SHARPE_V2_MANIFEST"
        ),
        "failure_count": len(failures),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": content_hash(live_files),
        "parent_manifest_status": parent["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证40个百分点高夏普V2扩展协议")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    result = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("failure_count", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
