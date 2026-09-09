"""在任何新机制结果读取前冻结510300机制图谱V1。"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path, PurePosixPath
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "research") not in sys.path:
    sys.path.insert(0, str(ROOT / "research"))

from mechanism_atlas_v1 import (  # noqa: E402
    CONFIG_PATH,
    MANIFEST_PATH,
    ContractError,
    audit_inputs,
    load_config,
    load_inputs,
    sha256_file,
    validate_manifest,
    write_input_audit,
)


TRACKED_FILES = [
    "config/510300_mechanism_atlas_v1.yaml",
    "docs/510300_MECHANISM_ATLAS_V1_SPEC.md",
    "research/mechanism_atlas_v1.py",
    "scripts/audit_510300_mechanism_atlas_v1_inputs.py",
    "scripts/freeze_510300_mechanism_atlas_v1.py",
    "scripts/run_510300_mechanism_atlas_v1.py",
    "tests/test_510300_mechanism_atlas_v1.py",
]


def project_path(value: str) -> Path:
    return ROOT / PurePosixPath(value)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    try:
        config = load_config()
        if MANIFEST_PATH.exists():
            validation = validate_manifest(config)
            payload = {
                "status": "ALREADY_FROZEN",
                "project_id": config["protocol"]["project_id"],
                "manifest": validation,
                "return_evaluation": "NOT_ALLOWED",
                "live_trading_authorized": False,
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0

        result_paths = [
            project_path(config["paths"]["result_json"]),
            project_path(config["paths"]["result_markdown"]),
            project_path(config["paths"]["state_panel"]),
            project_path(config["paths"]["local_projection_results"]),
        ]
        existing_results = [str(path) for path in result_paths if path.exists()]
        if existing_results:
            raise ContractError(
                f"首次冻结前已经存在候选结果产物：{existing_results}"
            )

        inputs = load_inputs(config)
        audit = audit_inputs(config, inputs)
        audit_path = write_input_audit(config, audit)

        missing_tracked = [
            relative for relative in TRACKED_FILES if not project_path(relative).exists()
        ]
        if missing_tracked:
            raise ContractError(f"冻结跟踪文件缺失：{missing_tracked}")
        tracked_hashes = {
            relative: sha256_file(project_path(relative)) for relative in TRACKED_FILES
        }
        input_paths = {
            specification["path"]
            for specification in config["inputs"].values()
        }
        input_paths.add(audit_path.relative_to(ROOT).as_posix())
        input_hashes = {
            relative: sha256_file(project_path(relative))
            for relative in sorted(input_paths)
        }
        manifest = {
            "project_id": config["protocol"]["project_id"],
            "version": config["protocol"]["version"],
            "state": "FROZEN_BEFORE_FIRST_RESULT",
            "frozen_at": datetime.now().astimezone().isoformat(),
            "config_path": CONFIG_PATH.relative_to(ROOT).as_posix(),
            "config_sha256": sha256_file(CONFIG_PATH),
            "tracked_files": tracked_hashes,
            "input_files": input_hashes,
            "input_audit_status": audit["status"],
            "source_branch_status": audit["source_branch"]["status"],
            "anchor_event_dates": audit["anchor_events"]["actual"],
            "one_shot": True,
            "parameter_rescue_after_result": "FORBIDDEN",
            "return_backtest_allowed": False,
            "live_trading_authorized": False,
        }
        atomic_json(MANIFEST_PATH, manifest)
        receipt_path = project_path(config["paths"]["freeze_receipt"])
        receipt = {
            "project_id": config["protocol"]["project_id"],
            "status": "FROZEN_BEFORE_FIRST_RESULT",
            "frozen_at": manifest["frozen_at"],
            "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": sha256_file(MANIFEST_PATH),
            "input_audit_path": audit_path.relative_to(ROOT).as_posix(),
            "input_audit_sha256": sha256_file(audit_path),
            "tracked_file_count": len(tracked_hashes),
            "input_file_count": len(input_hashes),
            "source_branch_status": audit["source_branch"]["status"],
            "return_evaluation": "NOT_ALLOWED",
            "net_sharpe": "NOT_COMPUTED",
            "paper_or_shadow_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        }
        atomic_json(receipt_path, receipt)
    except (ContractError, FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "status": "NO_VIEW_PROTOCOL_FREEZE_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "return_evaluation": "NOT_ALLOWED",
            "net_sharpe": "NOT_COMPUTED",
            "live_trading_authorized": False,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

