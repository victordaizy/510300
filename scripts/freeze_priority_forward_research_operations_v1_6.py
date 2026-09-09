"""生成并核验 V1.6 零付费研究运行清单。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config/priority_forward_research_operations_v1_6_manifest.json"
ENTRYPOINT = ROOT / "scripts/run_priority_forward_codex_automation_v1_6.py"
WORKSPACE_SEEDS = (
    "config/priority_forward_supervisor_v1_2.yaml",
    "config/primary_market_forward_v1_2.yaml",
    "config/priority_forward_authoritative_status_v1_6.yaml",
    "config/industry_expectation_gap_forward_operations_v1_2.yaml",
    "config/industry_expectation_gap_outcome_refresh_v1_2.yaml",
    "config/a_share_hs_official_cash_option_floor_alpha_v1_0_2_final_free_source_repair.json",
    "config/a_share_hs_official_cash_option_floor_alpha_v1_0_2_final_free_source_repair_protocol_manifest.json",
    "data/governance/FREE_SOURCE_REGISTRY_V1_1.csv",
    "reports/audit/PRIORITY_FORWARD_WINDOWS_SCHEDULER_V1_6_PERSISTENCE_VALIDATION.json",
    "reports/audit/PRIORITY_FORWARD_FROZEN_INPUT_CONTENT_ADDRESSING_V1.json",
    "reports/audit/OPTION_ORDERBOOK_ZERO_COST_SOURCE_QUALIFICATION_V1.json",
    "reports/audit/A_SHARE_HS_OFFICIAL_CASH_OPTION_FLOOR_ALPHA_V1_FINAL_FREE_SOURCE_REPAIR.json",
    "scripts/run_priority_forward_codex_automation_v1_6.py",
    "scripts/run_priority_forward_codex_automation_v1_5.py",
    "scripts/run_priority_forward_supervisor.py",
    "scripts/render_priority_forward_research_status.py",
    "scripts/run_510300_primary_market_collection_task_v1_2.ps1",
    "scripts/run_510300_primary_market_forward_v1_2.ps1",
    "scripts/collect_510300_primary_market_v1_2.py",
    "scripts/analyze_510300_primary_market_readiness_v1_2.py",
    "scripts/free_source_storage_v1_2.py",
    "market_data/etf_daily_crosscheck_v1_2.py",
    "market_data/etf_primary_market.py",
    "research/primary_market_forward_readiness_v1_2.py",
    "scripts/run_industry_expectation_gap_forward_operations_task_v1_2.ps1",
    "scripts/refresh_industry_expectation_gap_outcome_inputs_v1_2.py",
    "scripts/freeze_industry_expectation_gap_outcome_inputs_v1_2.py",
    "scripts/industry_outcome_snapshot_v1_2.py",
    "scripts/run_industry_expectation_gap_forward_evaluation_v1_2.py",
    "scripts/run_industry_expectation_gap_forward_operations_v1_2.py",
    "scripts/build_industry_expectation_gap_origin_maturity.py",
    "scripts/run_priority_forward_authoritative_status_task_v1_6.ps1",
    "scripts/render_priority_forward_authoritative_status_v1_6.py",
    "scripts/audit_priority_forward_frozen_inputs_v1.py",
    "scripts/audit_option_orderbook_zero_cost_source_v1.py",
    "scripts/run_a_share_hs_cash_option_floor_alpha_v1_0_2_final_free_source_repair.py",
)
LOCAL_PACKAGE_PREFIXES = ("scripts", "research", "market_data")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def module_path(module: str) -> Path | None:
    if not module or module.split(".", 1)[0] not in LOCAL_PACKAGE_PREFIXES:
        return None
    candidate = ROOT.joinpath(*module.split(".")).with_suffix(".py")
    if candidate.is_file():
        return candidate.resolve()
    package = ROOT.joinpath(*module.split("."), "__init__.py")
    if package.is_file():
        return package.resolve()
    return None


def discover_python_dependencies(seed_paths: set[Path]) -> set[Path]:
    discovered = {path.resolve() for path in seed_paths}
    queue = deque(path for path in discovered if path.suffix.lower() == ".py")
    parsed: set[Path] = set()
    while queue:
        path = queue.popleft()
        if path in parsed:
            continue
        parsed.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)
        for module in sorted(modules):
            dependency = module_path(module)
            if dependency is not None and dependency not in discovered:
                discovered.add(dependency)
                queue.append(dependency)
    return discovered


def tracked_paths() -> list[Path]:
    seeds = {ROOT / path for path in WORKSPACE_SEEDS}
    missing = [relative(path) if path.exists() else str(path) for path in seeds if not path.is_file()]
    if missing:
        raise RuntimeError(f"V1.6 清单种子文件缺失：{sorted(missing)}")
    return sorted(discover_python_dependencies(seeds), key=relative)


def file_records(paths: list[Path]) -> list[dict[str, Any]]:
    return [
        {"path": relative(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in paths
    ]


def create_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def build_manifest() -> dict[str, Any]:
    records = file_records(tracked_paths())
    content_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "1.6.0",
        "manifest_id": "PRIORITY_FORWARD_RESEARCH_OPERATIONS_V1_6",
        "status": "FROZEN_ZERO_PAID_FOUNDATION_ACTIVE_AWAITING_NEXT_WINDOW",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "data_purchase_budget_cny": 0,
        "zero_purchase_lock_days": 180,
        "maximum_active_research_streams": 2,
        "active_research_streams": [
            "PRIMARY_MARKET_PCF_IOPV",
            "INDUSTRY_EXPECTATION_GAP",
        ],
        "execution_entrypoint": relative(ENTRYPOINT),
        "supervisor_config": "config/priority_forward_supervisor_v1_2.yaml",
        "files": records,
        "content_sha256": content_sha256,
        "mutable_outputs_not_hashed": [
            "reports/data_quality/510300_primary_market_readiness_v1_2.json",
            "reports/forward/industry_expectation_gap_v1_evaluation/operations_status_v1_2.json",
            "reports/audit/priority_forward_authoritative_status_v1_6.json",
            "reports/audit/priority_forward_*_runs_v1_6/*.json",
            "output/**",
        ],
        "terminal_branches": {
            "cash_option_floor": "NO_VIEW_FREE_DATA_INSUFFICIENT",
            "option_orderbook": "BLOCKED_ZERO_COST_SOURCE_UNAVAILABLE",
        },
        "windows_logout_reboot_persistence": "BLOCKED_PERMISSION_S4U_REGISTRATION_ACCESS_DENIED",
        "research_only": True,
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }


def verify_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError("V1.6 运行清单不存在")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = file_records(tracked_paths())
    failures: list[dict[str, Any]] = []
    if manifest.get("files") != expected:
        failures.append({"failure": "TRACKED_FILE_RECORDS"})
    content_sha256 = hashlib.sha256(
        json.dumps(expected, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if manifest.get("content_sha256") != content_sha256:
        failures.append(
            {
                "failure": "CONTENT_SHA256",
                "expected": manifest.get("content_sha256"),
                "actual": content_sha256,
            }
        )
    if manifest.get("data_purchase_budget_cny") != 0:
        failures.append({"failure": "DATA_PURCHASE_BUDGET"})
    if manifest.get("maximum_active_research_streams") != 2:
        failures.append({"failure": "ACTIVE_STREAM_CAP"})
    result = {
        "status": (
            "PASS_PRIORITY_FORWARD_V1_6_MANIFEST_VERIFIED"
            if not failures
            else "FAILED_PRIORITY_FORWARD_V1_6_MANIFEST_VERIFICATION"
        ),
        "failure_count": len(failures),
        "failures": failures,
        "manifest_path": relative(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "content_sha256": content_sha256,
        "tracked_file_count": len(expected),
    }
    if failures:
        raise RuntimeError(json.dumps(result, ensure_ascii=False))
    return result


def freeze_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        return {**verify_manifest(), "manifest_reused": True}
    manifest = build_manifest()
    create_json_exclusive(MANIFEST_PATH, manifest)
    return {**verify_manifest(), "manifest_reused": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结 V1.6 零付费研究运行清单")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    arguments = parser.parse_args()
    result = freeze_manifest() if arguments.mode == "freeze" else verify_manifest()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
