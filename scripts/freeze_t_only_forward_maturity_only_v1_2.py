"""生成或验证 T-only 成熟度输出 V1.2 冻结清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = ZoneInfo("Asia/Shanghai")
MANIFEST_PATH = (
    ROOT / "config" / "t_only_forward_v1_1_maturity_only_v1_2_manifest.json"
)

TRACKED_FILES = (
    "config/t_only_forward_v1_1.yaml",
    "config/t_only_forward_v1_1_freeze_manifest.json",
    "config/t_only_forward_v1_automation_manifest.json",
    "docs/510300_T_ONLY_FORWARD_V1_1_MATURITY_OUTPUT_V1_2.md",
    "scripts/freeze_t_only_forward_maturity_only_v1_2.py",
    "scripts/install_t_only_forward_v1_1_maturity_task_v1_2.ps1",
    "scripts/refresh_t_only_forward_v1.py",
    "scripts/run_t_only_forward_v1_1.py",
    "scripts/run_t_only_forward_v1_daily.py",
    "scripts/run_t_only_forward_v1_1_maturity_stage_v1_2.py",
    "scripts/run_t_only_forward_v1_daily_v1_2.py",
    "scripts/t_only_forward_maturity_only_v1_2.py",
    "tests/test_t_only_forward_maturity_only_v1_2.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative_path in sorted(TRACKED_FILES):
        path = ROOT / relative_path
        if not path.is_file():
            raise RuntimeError(f"冻结文件缺失：{relative_path}")
        records.append(
            {
                "path": relative_path,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def content_hash(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_manifest() -> dict[str, Any]:
    records = file_records()
    return {
        "schema_version": "1.2.0",
        "project_id": "510300_T_ONLY_FORWARD_V1_1_MATURITY_OUTPUT_V1_2",
        "frozen_at": datetime.now(TIMEZONE).isoformat(),
        "status": "FROZEN_MATURITY_OUTPUT_ONLY",
        "change_class": "OPERATIONAL_OUTPUT_GOVERNANCE_FIX",
        "research_parameters_changed": False,
        "maturity_thresholds_changed": False,
        "forward_start_changed": False,
        "historical_signals_changed": False,
        "backfill_enabled": False,
        "schedule": {
            "timezone": "Asia/Shanghai",
            "days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "time": "16:30",
            "start_when_available": False,
            "automatic_restart_count": 0,
            "multiple_instances": "IgnoreNew",
        },
        "maturity_contract": {
            "minimum_new_trading_days": 252,
            "minimum_closed_cycles": 3,
            "pre_maturity_view": "NO_VIEW_UNTIL_FORWARD_MATURITY",
            "pre_maturity_public_fields": [
                "new_trading_days",
                "closed_cycles",
                "data_completeness",
                "ledger_completeness",
                "run_completeness",
            ],
        },
        "files": records,
        "content_sha256": content_hash(records),
    }


def write_manifest() -> dict[str, Any]:
    if MANIFEST_PATH.exists():
        raise RuntimeError(f"冻结清单已存在，拒绝覆盖：{MANIFEST_PATH}")
    manifest = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2))
    return {
        "status": "FROZEN",
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "content_sha256": manifest["content_sha256"],
        "tracked_file_count": len(manifest["files"]),
    }


def verify_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(f"冻结清单缺失：{MANIFEST_PATH}")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    actual_records = file_records()
    failures: list[dict[str, Any]] = []
    if manifest.get("files") != actual_records:
        expected_by_path = {
            str(item["path"]): item for item in manifest.get("files", [])
        }
        actual_by_path = {str(item["path"]): item for item in actual_records}
        for path in sorted(set(expected_by_path).union(actual_by_path)):
            if expected_by_path.get(path) != actual_by_path.get(path):
                failures.append(
                    {
                        "path": path,
                        "expected": expected_by_path.get(path),
                        "actual": actual_by_path.get(path),
                    }
                )
    actual_content_sha256 = content_hash(actual_records)
    if manifest.get("content_sha256") != actual_content_sha256:
        failures.append(
            {
                "path": "<manifest-content>",
                "expected": manifest.get("content_sha256"),
                "actual": actual_content_sha256,
            }
        )
    return {
        "status": "PASS" if not failures else "FAILED",
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "content_sha256": actual_content_sha256,
        "tracked_file_count": len(actual_records),
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="冻结或验证 T-only V1.2 运维清单")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    arguments = parser.parse_args()
    result = write_manifest() if arguments.write else verify_manifest()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"FROZEN", "PASS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
