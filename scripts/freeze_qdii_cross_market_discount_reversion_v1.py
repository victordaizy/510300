"""冻结或验证QDII跨市场折价回归V1。"""

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

from research.qdii_cross_market_discount_reversion_v1 import (
    CONFIG,
    build_discount_signals,
    load_contract,
    load_research_inputs,
)
from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v2 import (
    verify as verify_parent_manifest,
)


MANIFEST = ROOT / "config" / "qdii_cross_market_discount_reversion_v1_manifest.json"
TRACKED_FILES = [
    "config/qdii_cross_market_discount_reversion_v1.yaml",
    "docs/QDII_CROSS_MARKET_DISCOUNT_REVERSION_V1_SPEC.md",
    "scripts/acquire_qdii_cross_market_discount_reversion_v1_inputs.py",
    "research/qdii_cross_market_discount_reversion_v1.py",
    "scripts/run_qdii_cross_market_discount_reversion_v1.py",
    "scripts/freeze_qdii_cross_market_discount_reversion_v1.py",
]
DEPENDENCY_FILES = [
    "research/broad_liquid_etf_liquidity_shock_reversal_v1.py",
    "research/multi_asset_annual_excess_40pct_high_sharpe_v1.py",
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


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结输入：{path}")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def current_tracked_hashes() -> dict[str, str]:
    missing = [relative for relative in TRACKED_FILES if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少QDII候选冻结文件：{missing}")
    return {relative: sha256_file(ROOT / relative) for relative in TRACKED_FILES}


def tracked_content_hash(files: dict[str, str]) -> str:
    canonical = "\n".join(f"{path}:{files[path]}" for path in sorted(files))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def current_input_snapshot(contract: dict[str, Any]) -> dict[str, Any]:
    inputs = contract["inputs"]
    snapshot: dict[str, Any] = {}
    for key in (
        "etf_total_return_panel",
        "fund_master",
        "benchmark_total_return",
        "benchmark_price_ohlc",
        "parent_manifest",
        "v1_closeout_json",
    ):
        snapshot[key] = file_record(ROOT / inputs[key])
    index_directory = ROOT / inputs["global_index_directory"]
    snapshot["global_indices"] = {
        str(group["group_id"]): file_record(index_directory / str(group["index_file"]))
        for group in contract["tracked_groups"]
    }
    fx_files = sorted((ROOT / inputs["fx_directory"]).glob(inputs["fx_file_pattern"]))
    if not fx_files:
        raise FileNotFoundError("没有人民币中间价冻结输入")
    snapshot["fx_files"] = [file_record(path) for path in fx_files]
    input_audit = ROOT / contract["outputs"]["input_audit_json"]
    snapshot["input_audit"] = file_record(input_audit)
    ndx_metadata = index_directory / "NDX_daily.metadata.json"
    snapshot["ndx_metadata"] = file_record(ndx_metadata)
    snapshot["evaluation_dependencies"] = {
        relative: file_record(ROOT / relative) for relative in DEPENDENCY_FILES
    }
    return snapshot


def prefreeze_coverage(contract: dict[str, Any]) -> dict[str, Any]:
    """只构造信号覆盖，不读取信号后的收益。"""

    partition = contract["historical_partition"]
    start = pd.Timestamp(partition["visible_start"])
    end = pd.Timestamp(partition["visible_end"])
    panel, master, benchmark, fx, indices = load_research_inputs(contract)
    signals, _, _, audit = build_discount_signals(
        panel,
        master,
        benchmark,
        fx,
        indices,
        contract,
        start=start,
        end=end,
    )
    coverage = contract["prefreeze_coverage_gate"]
    years = int(signals["signal_date"].dt.year.nunique()) if not signals.empty else 0
    groups = int(signals["group_id"].nunique()) if not signals.empty else 0
    checks = {
        "minimum_executable_signal_rows": len(signals)
        >= int(coverage["minimum_executable_signal_rows"]),
        "minimum_signal_years": years >= int(coverage["minimum_signal_years"]),
        "minimum_tracking_groups_with_signals": groups
        >= int(coverage["minimum_tracking_groups_with_signals"]),
        "foreign_date_strictly_before_signal": bool(
            audit["foreign_date_strictly_before_signal"]
        ),
        "fx_date_not_after_signal": bool(audit["fx_date_not_after_signal"]),
        "no_future_return_used": not bool(audit["future_return_used_in_signal_construction"]),
        "same_open_not_used_for_signal": not bool(
            audit["same_open_used_for_signal_and_execution"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"冻结前信号覆盖闸门失败：{checks}")
    return {
        "status": "PASS_SIGNAL_COVERAGE_ONLY_NO_FUTURE_RETURN_VIEW",
        "checks": checks,
        "audit": audit,
        "performance_or_future_return_computed": False,
    }


def freeze() -> dict[str, Any]:
    contract = load_contract(CONFIG)
    parent = verify_parent_manifest()
    if parent["failure_count"]:
        raise RuntimeError("V2父协议冻结清单验证失败")
    files = current_tracked_hashes()
    inputs = current_input_snapshot(contract)
    coverage = prefreeze_coverage(contract)
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "QDII_CROSS_MARKET_DISCOUNT_REVERSION_V1_MANIFEST",
        "status": "FROZEN_BEFORE_VISIBLE_OUTCOME",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
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
        "visible_partition": [
            contract["historical_partition"]["visible_start"],
            contract["historical_partition"]["visible_end"],
        ],
        "sealed_replication_partition": [
            contract["historical_partition"]["sealed_replication_start"],
            contract["historical_partition"]["sealed_replication_end"],
        ],
        "tracked_files": files,
        "tracked_content_sha256": tracked_content_hash(files),
        "input_snapshot": inputs,
        "prefreeze_signal_coverage": coverage,
        "parent_manifest_verification": parent,
        "historical_result_can_verify_target": False,
        "paper_or_live_authorized": False,
    }
    if MANIFEST.exists():
        existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
        for field in (
            "tracked_files",
            "tracked_content_sha256",
            "input_snapshot",
            "prefreeze_signal_coverage",
            "visible_partition",
            "sealed_replication_partition",
        ):
            if existing.get(field) != manifest.get(field):
                raise RuntimeError("现有QDII冻结清单与当前内容不同；不得覆盖")
        return existing
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_QDII_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    contract = load_contract(CONFIG)
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parent = verify_parent_manifest()
    failures: list[str] = []
    live_files = current_tracked_hashes()
    live_inputs = current_input_snapshot(contract)
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != tracked_content_hash(live_files):
        failures.append("tracked_content_sha256")
    if existing.get("input_snapshot") != live_inputs:
        failures.append("input_snapshot")
    if parent["failure_count"]:
        failures.append("parent_manifest")
    return {
        "status": (
            "PASS_QDII_CROSS_MARKET_DISCOUNT_REVERSION_V1_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_QDII_CROSS_MARKET_DISCOUNT_REVERSION_V1_MANIFEST"
        ),
        "failure_count": len(failures),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": tracked_content_hash(live_files),
        "tracked_file_count": len(live_files),
        "parent_manifest_status": parent["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证QDII跨市场折价回归V1")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    result = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("failure_count", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
