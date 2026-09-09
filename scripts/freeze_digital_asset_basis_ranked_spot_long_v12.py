"""冻结或验证数字资产基差排序现货单多V12。"""

from __future__ import annotations

import argparse
import copy
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

from research.digital_asset_basis_ranked_spot_long_v12 import CONFIG, load_contract
from scripts.freeze_digital_asset_current_quarter_basis_factor_v11 import (
    prefreeze_data_coverage as v11_prefreeze_data_coverage,
    verify as verify_v11_manifest,
)
from scripts.freeze_multi_asset_annual_excess_40pct_high_sharpe_v22 import (
    verify as verify_parent_manifest,
)
from scripts.freeze_qdii_cross_market_discount_reversion_zero_variance_v2 import (
    verify as verify_evaluator_manifest,
)


MANIFEST = ROOT / "config" / "digital_asset_basis_ranked_spot_long_v12_manifest.json"
TRACKED_FILES = [
    "config/digital_asset_basis_ranked_spot_long_v12.yaml",
    "docs/DIGITAL_ASSET_BASIS_RANKED_SPOT_LONG_V12_SPEC.md",
    "research/digital_asset_basis_ranked_spot_long_v12.py",
    "scripts/freeze_digital_asset_basis_ranked_spot_long_v12.py",
    "scripts/run_digital_asset_basis_ranked_spot_long_v12.py",
    "tests/test_digital_asset_basis_ranked_spot_long_v12.py",
]
VISIBLE_SOURCE_DEPENDENCIES = [
    "data/raw/digital_asset_current_quarter_basis_v11/collection_status.json",
    "data/raw/digital_asset_current_quarter_basis_v11/spot_1h_2021_2026.parquet",
    "data/raw/digital_asset_current_quarter_basis_v11/current_quarter_1h_2021_2026.parquet",
    "data/raw/digital_asset_current_quarter_basis_v11/fx_2021_2026.parquet",
    "data/raw/digital_asset_current_quarter_basis_v11/H00300_2021_2026.parquet",
    "config/multi_asset_annual_excess_40pct_high_sharpe_v22_manifest.json",
    "config/digital_asset_current_quarter_basis_factor_v11_manifest.json",
    "research/digital_asset_current_quarter_basis_factor_v11.py",
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
        raise FileNotFoundError(f"缺少基差排序现货单多V12冻结文件：{relative}")
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
    return {relative: file_record(relative) for relative in VISIBLE_SOURCE_DEPENDENCIES}


def prefreeze_data_coverage(contract: dict[str, Any]) -> dict[str, Any]:
    source_contract = copy.deepcopy(contract)
    source_contract["protocol"]["candidate_id"] = (
        "DIGITAL_ASSET_CURRENT_QUARTER_BASIS_FACTOR_V11"
    )
    coverage = v11_prefreeze_data_coverage(source_contract)
    coverage["source_candidate_id"] = source_contract["protocol"]["candidate_id"]
    coverage["target_candidate_id"] = contract["protocol"]["candidate_id"]
    coverage["source_outputs_reused_without_basis_or_return_read"] = True
    return coverage


def _assert_outputs_absent(contract: dict[str, Any]) -> None:
    names = (
        "input_audit_json",
        "visible_daily_returns",
        "visible_targets",
        "visible_trades",
        "visible_report_json",
        "visible_report_markdown",
    )
    existing = [
        contract["outputs"][name]
        for name in names
        if (ROOT / contract["outputs"][name]).exists()
    ]
    if existing:
        raise RuntimeError(f"冻结前已存在基差排序现货单多V12结果：{existing}")


def _dependencies() -> dict[str, dict[str, Any]]:
    return {
        "parent": verify_parent_manifest(),
        "v11_source_engine": verify_v11_manifest(),
        "evaluator": verify_evaluator_manifest(),
    }


def freeze() -> dict[str, Any]:
    contract = load_contract(CONFIG)
    if MANIFEST.exists():
        verification = verify()
        if verification["failure_count"]:
            raise RuntimeError(f"现有基差排序现货单多V12清单验证失败：{verification['failures']}")
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    _assert_outputs_absent(contract)
    dependencies = _dependencies()
    if any(item["failure_count"] for item in dependencies.values()):
        raise RuntimeError("基差排序现货单多V12依赖冻结清单验证失败")
    coverage = prefreeze_data_coverage(contract)
    if not coverage["all_checks_pass"]:
        raise RuntimeError(f"基差排序现货单多V12冻结前覆盖门失败：{coverage['checks']}")
    files = current_tracked_hashes()
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "DIGITAL_ASSET_BASIS_RANKED_SPOT_LONG_V12_MANIFEST",
        "status": "FROZEN_BEFORE_VISIBLE_OUTCOME",
        "frozen_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "candidate_id": contract["protocol"]["candidate_id"],
        "parent_protocol_id": contract["protocol"]["parent_protocol_id"],
        "tracked_files": files,
        "tracked_content_sha256": tracked_content_hash(files),
        "visible_source_snapshot": current_visible_source_snapshot(),
        "dependency_verifications": dependencies,
        "prefreeze_data_coverage": coverage,
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
        "basis_value_or_rank_computed_during_freeze": False,
        "direction_or_target_computed_during_freeze": False,
        "historical_result_can_verify_target": False,
        "paper_or_live_authorized": False,
    }
    atomic_json(MANIFEST, manifest)
    return manifest


def verify() -> dict[str, Any]:
    if not MANIFEST.is_file():
        return {
            "status": "FAIL_MISSING_BASIS_RANKED_SPOT_LONG_V12_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    load_contract(CONFIG)
    existing = json.loads(MANIFEST.read_text(encoding="utf-8"))
    live_files = current_tracked_hashes()
    live_sources = current_visible_source_snapshot()
    dependencies = _dependencies()
    failures: list[str] = []
    if existing.get("tracked_files") != live_files:
        failures.append("tracked_files")
    if existing.get("tracked_content_sha256") != tracked_content_hash(live_files):
        failures.append("tracked_content_sha256")
    if existing.get("visible_source_snapshot") != live_sources:
        failures.append("visible_source_snapshot")
    if not existing.get("prefreeze_data_coverage", {}).get("all_checks_pass", False):
        failures.append("prefreeze_data_coverage")
    for field in (
        "performance_or_future_return_computed_during_freeze",
        "basis_value_or_rank_computed_during_freeze",
        "direction_or_target_computed_during_freeze",
    ):
        if existing.get(field) is not False:
            failures.append(field)
    for name, item in dependencies.items():
        if item["failure_count"]:
            failures.append(f"{name}_manifest")
    return {
        "status": (
            "PASS_BASIS_RANKED_SPOT_LONG_V12_MANIFEST_VERIFIED"
            if not failures
            else "FAIL_BASIS_RANKED_SPOT_LONG_V12_MANIFEST"
        ),
        "failure_count": int(len(failures)),
        "failures": failures,
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST),
        "tracked_content_sha256": tracked_content_hash(live_files),
        "tracked_file_count": int(len(live_files)),
        "dependency_statuses": {
            name: item["status"] for name, item in dependencies.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证数字资产基差排序现货单多V12")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    payload = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if payload.get("failure_count", 0) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
