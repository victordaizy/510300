"""在读取任何候选未来事件收益前冻结 510300 有锚稀疏网格 V1。"""

from __future__ import annotations

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

from research.anchored_sparse_mean_reversion_grid_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    cost_model,
    load_config,
    project_path,
    sha256_file,
)


TRACKED_FILES = [
    "config/510300_anchored_sparse_mean_reversion_grid_v1.yaml",
    "docs/510300_ANCHORED_SPARSE_MEAN_REVERSION_GRID_V1_SPEC.md",
    "research/anchored_sparse_mean_reversion_grid_v1.py",
    "scripts/run_510300_anchored_sparse_mean_reversion_grid_v1.py",
    "scripts/freeze_510300_anchored_sparse_mean_reversion_grid_v1.py",
    "tests/test_510300_anchored_sparse_mean_reversion_grid_v1.py",
]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _tracked_hashes() -> dict[str, str]:
    missing = [relative for relative in TRACKED_FILES if not project_path(relative).is_file()]
    if missing:
        raise FileNotFoundError(f"缺少待冻结实现文件：{missing}")
    return {relative: sha256_file(project_path(relative)) for relative in TRACKED_FILES}


def _input_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key, specification in config["inputs"].items():
        for field in ("path", "metadata_path"):
            relative = specification.get(field)
            if not relative:
                continue
            path = project_path(relative)
            if not path.is_file():
                raise FileNotFoundError(f"缺少待绑定输入：{relative}")
            snapshot[f"{key}.{field}"] = {
                "path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
    return snapshot


def _aggregate_hash(values: dict[str, Any]) -> str:
    import hashlib

    canonical = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _result_paths(config: dict[str, Any]) -> list[Path]:
    return [
        project_path(config["paths"][key])
        for key in (
            "input_audit_json",
            "result_json",
            "result_markdown",
            "feature_table",
            "event_table",
            "event_outcome_table",
            "gate_receipt",
        )
    ]


def freeze() -> dict[str, Any]:
    """创建一次性冻结清单，并拒绝覆盖或结果后冻结。"""

    config = load_config(CONFIG_PATH)
    if MANIFEST_PATH.exists():
        raise FileExistsError(f"冻结清单已经存在，禁止覆盖：{MANIFEST_PATH}")
    preexisting = [path.relative_to(ROOT).as_posix() for path in _result_paths(config) if path.exists()]
    if preexisting:
        raise RuntimeError(f"冻结前已有结果产物，拒绝冻结：{preexisting}")
    tracked = _tracked_hashes()
    inputs = _input_snapshot(config)
    model = cost_model(config)
    frozen_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    manifest = {
        "schema_version": "1.0.0",
        "manifest_id": "510300_ANCHORED_SPARSE_MEAN_REVERSION_GRID_V1_MANIFEST",
        "project_id": config["protocol"]["project_id"],
        "state": "FROZEN_BEFORE_FIRST_RESULT",
        "frozen_at_asia_shanghai": frozen_at,
        "result_preexisted_at_freeze": False,
        "tracked_files": tracked,
        "tracked_files_sha256": _aggregate_hash(tracked),
        "input_snapshot": inputs,
        "input_snapshot_sha256": _aggregate_hash(inputs),
        "fixed_costs": {
            "base_round_trip_rate": model.base_round_trip_rate,
            "stress_round_trip_rate": model.stress_round_trip_rate,
        },
        "fixed_primary_horizon_minutes": int(
            config["event_definition"]["primary_horizon_trading_minutes"]
        ),
        "fixed_minimum_primary_events": int(
            config["phase_a_gates"]["minimum_completed_primary_events"]
        ),
        "phase_b_before_phase_a_pass": "NOT_ALLOWED",
        "historical_iopv_used": False,
        "tests_are_runtime_inputs": False,
        "boundaries": {
            "execution_asset": "510300.SH",
            "allowed_holdings": ["510300.SH", "CASH_CNY"],
            "one_shot": True,
            "parameter_rescue": "FORBIDDEN",
            "window_rescue": "FORBIDDEN",
            "cost_rescue": "FORBIDDEN",
            "proxy_rescue": "FORBIDDEN",
            "subperiod_rescue": "FORBIDDEN",
            "paper_or_shadow_position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _atomic_json(MANIFEST_PATH, manifest)
    receipt_path = project_path(config["paths"]["freeze_receipt"])
    receipt = {
        "project_id": manifest["project_id"],
        "status": "FROZEN_BEFORE_FIRST_RESULT",
        "frozen_at_asia_shanghai": frozen_at,
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "tracked_file_count": len(tracked),
        "input_file_count": len(inputs),
        "tracked_files_sha256": manifest["tracked_files_sha256"],
        "input_snapshot_sha256": manifest["input_snapshot_sha256"],
        "result_preexisted_at_freeze": False,
        "phase_b_grid_backtest": "NOT_ALLOWED",
        "live_trading_authorized": False,
    }
    _atomic_json(receipt_path, receipt)
    return receipt


def verify() -> dict[str, Any]:
    """在任何正式结果读取前验证实现和输入均未漂移。"""

    if not MANIFEST_PATH.is_file():
        return {
            "status": "FAIL_MISSING_MANIFEST",
            "failure_count": 1,
            "failures": ["missing_manifest"],
        }
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    config = load_config(CONFIG_PATH)
    live_tracked = _tracked_hashes()
    live_inputs = _input_snapshot(config)
    failures: list[str] = []
    if manifest.get("tracked_files") != live_tracked:
        failures.append("tracked_files")
    if manifest.get("tracked_files_sha256") != _aggregate_hash(live_tracked):
        failures.append("tracked_files_sha256")
    if manifest.get("input_snapshot") != live_inputs:
        failures.append("input_snapshot")
    if manifest.get("input_snapshot_sha256") != _aggregate_hash(live_inputs):
        failures.append("input_snapshot_sha256")
    return {
        "status": "PASS_FROZEN_IMPLEMENTATION_AND_INPUTS"
        if not failures
        else "FAIL_FROZEN_IMPLEMENTATION_OR_INPUT_DRIFT",
        "failure_count": len(failures),
        "failures": failures,
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "tracked_file_count": len(live_tracked),
        "input_file_count": len(live_inputs),
        "tracked_files_sha256": _aggregate_hash(live_tracked),
        "input_snapshot_sha256": _aggregate_hash(live_inputs),
        "result_preexisted_at_freeze": bool(manifest.get("result_preexisted_at_freeze")),
        "live_trading_authorized": False,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="冻结或验证 510300 有锚稀疏网格 V1")
    parser.add_argument("--mode", choices=("freeze", "verify"), default="verify")
    args = parser.parse_args()
    result = freeze() if args.mode == "freeze" else verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if int(result.get("failure_count", 0)) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

