"""冻结或验证A股涨停后可成交延续V1候选。"""

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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.a_share_limit_up_continuation_v1 import (
    CONFIG,
    build_limit_up_signals,
    load_contract,
    load_visible_inputs,
)
from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v3 import (
    verify as verify_parent_manifest,
)
from scripts.freeze_qdii_cross_market_discount_reversion_zero_variance_v2 import (
    verify as verify_evaluator_manifest,
)


MANIFEST = ROOT / "config" / "a_share_limit_up_continuation_v1_manifest.json"
TRACKED_FILES = [
    "config/a_share_limit_up_continuation_v1.yaml",
    "docs/A_SHARE_LIMIT_UP_CONTINUATION_V1_SPEC.md",
    "research/a_share_limit_up_continuation_v1.py",
    "scripts/run_a_share_limit_up_continuation_v1.py",
    "scripts/freeze_a_share_limit_up_continuation_v1.py",
]
VISIBLE_SOURCE_DEPENDENCIES = [
    "data/raw/a_share_hash_holdout_v2/training_panel.parquet",
    "data/raw/a_share_hash_holdout_v2/stock_master_split.parquet",
    "data/raw/a_share_hash_holdout_v2/training_H00300.parquet",
    "config/multi_asset_annual_excess_40pct_high_sharpe_v3_manifest.json",
    "config/qdii_cross_market_discount_reversion_zero_variance_v2.yaml",
    "config/qdii_cross_market_discount_reversion_zero_variance_v2_manifest.json",
    "research/qdii_cross_market_discount_reversion_zero_variance_v2.py",
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
        raise FileNotFoundError(f"缺少涨停候选冻结文件：{relative}")
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
    """只快照可见输入与冻结评价依赖，绝不访问复验面板。"""

    return {relative: file_record(relative) for relative in VISIBLE_SOURCE_DEPENDENCIES}


def prefreeze_signal_coverage(contract: dict[str, Any]) -> dict[str, Any]:
    """冻结前只计算信号数量覆盖，不读取下一开盘或后续收益。"""

    partition = contract["historical_partition"]
    panel, master, benchmark, input_audit = load_visible_inputs(
        contract, signal_only=True
    )
    signals, signal_audit = build_limit_up_signals(
        panel,
        master,
        benchmark,
        contract,
        start=partition["visible_start"],
        end=partition["visible_end"],
    )
    gate = contract["prefreeze_coverage_gate"]
    checks = {
        "minimum_signal_rows": int(len(signals)) >= int(gate["minimum_signal_rows"]),
        "minimum_signal_years": int(signal_audit["signal_year_count"])
        >= int(gate["minimum_signal_years"]),
        "minimum_unique_signal_stocks": int(signal_audit["unique_signal_stocks"])
        >= int(gate["minimum_unique_signal_stocks"]),
        "future_open_or_return_not_read": input_audit["future_open_or_return_read"]
        is False
        and signal_audit["future_open_or_return_read"] is False,
        "sealed_replication_not_read": input_audit["sealed_panel_or_benchmark_read"]
        is False,
    }
    return {
        "status": (
            "PASS_PREFREEZE_SIGNAL_COVERAGE"
            if all(checks.values())
            else "FAIL_PREFREEZE_SIGNAL_COVERAGE"
        ),
        "checks": checks,
        "all_checks_pass": bool(all(checks.values())),
        "signal_rows": int(len(signals)),
        "signal_year_count": int(signal_audit["signal_year_count"]),
        "signal_years": signal_audit["signal_years"],
        "unique_signal_stocks": int(signal_audit["unique_signal_stocks"]),
        "first_signal_date": signal_audit["first_signal_date"],
        "last_signal_date": signal_audit["last_signal_date"],
        "panel_columns_read": input_audit["panel_columns_read"],
        "benchmark_columns_read": input_audit["benchmark_columns_read"],
        "future_open_or_return_read": False,
        "performance_or_future_return_computed": False,
        "sealed_replication_read": False,
    }


def _assert_visible_outputs_absent(contract: dict[str, Any]) -> None:
    outputs = contract["outputs"]
    names = (
        "input_audit_json",
        "visible_daily_returns",
        "visible_signals",
        "visible_trades",
        "visible_report_json",
        "visible_report_markdown",
    )
    existing = [outputs[name] for name in names if (ROOT / outputs[name]).exists()]
    if existing:
        raise RuntimeError(f"冻结前已存在可见结果，不得倒序冻结：{existing}")


def freeze() -> dict[str, Any]:
    contract = load_contract(CONFIG)
    if MANIFEST.exists():
        verification = verify()
        if verification["failure_count"]:
            raise RuntimeError(f"现有涨停候选清单验证失败：{verification['failures']}")
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    _assert_visible_outputs_absent(contract)
    parent = verify_parent_manifest()
    evaluator = verify_evaluator_manifest()
    if parent["failure_count"] or evaluator["failure_count"]:
        raise RuntimeError("父协议或保守夏普评价器冻结清单验证失败")
    files = current_tracked_hashes()
    visible_sources = current_visible_source_snapshot()
    coverage = prefreeze_signal_coverage(contract)
    if not coverage["all_checks_pass"]:
        raise RuntimeError(f"冻结前信号覆盖门失败：{coverage['checks']}")
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "A_SHARE_LIMIT_UP_CONTINUATION_V1_MANIFEST",
        "status": "FROZEN_BEFORE_VISIBLE_OUTCOME",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": tracked_content_hash(files),
        "visible_source_snapshot": visible_sources,
        "parent_manifest_verification": parent,
        "conservative_evaluator_manifest_verification": evaluator,
        "prefreeze_signal_coverage": coverage,
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
        "sealed_replication_read_during_freeze": False,
        "historical_result_can_verify_target": False,
        "paper_or_live_authorized": False,
    }
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_A_SHARE_LIMIT_UP_CONTINUATION_V1_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    load_contract(CONFIG)
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live_files = current_tracked_hashes()
    live_sources = current_visible_source_snapshot()
    parent = verify_parent_manifest()
    evaluator = verify_evaluator_manifest()
    failures: list[str] = []
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != tracked_content_hash(live_files):
        failures.append("tracked_content_sha256")
    if existing.get("visible_source_snapshot") != live_sources:
        failures.append("visible_source_snapshot")
    if not existing.get("prefreeze_signal_coverage", {}).get("all_checks_pass", False):
        failures.append("prefreeze_signal_coverage")
    if existing.get("performance_or_future_return_computed_during_freeze") is not False:
        failures.append("freeze_read_future_return")
    if existing.get("sealed_replication_read_during_freeze") is not False:
        failures.append("freeze_read_replication")
    if parent["failure_count"]:
        failures.append("parent_manifest")
    if evaluator["failure_count"]:
        failures.append("conservative_evaluator_manifest")
    return {
        "status": (
            "PASS_A_SHARE_LIMIT_UP_CONTINUATION_V1_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_A_SHARE_LIMIT_UP_CONTINUATION_V1_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": tracked_content_hash(live_files),
        "tracked_file_count": int(len(live_files)),
        "parent_manifest_status": parent["status"],
        "conservative_evaluator_manifest_status": evaluator["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证A股涨停后可成交延续V1")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
