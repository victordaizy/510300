"""T-only V1.1 的 V1.2 成熟度口径日更入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_t_only_forward_v1_daily import (
    acquire_lock,
    release_lock,
    run_stage,
)
from scripts.t_only_forward_maturity_only_v1_2 import (
    DATA_GATE,
    GUIDE_JSON,
    LEDGER,
    REPORT_JSON,
    assert_maturity_only_payload,
    atomic_json,
    build_operational_failure_snapshot,
    publish_snapshot,
    read_json,
    sanitize_existing_outputs,
    sha256_file,
)


TIMEZONE = ZoneInfo("Asia/Shanghai")
DEFAULT_MANIFEST = (
    ROOT / "config" / "t_only_forward_v1_1_maturity_only_v1_2_manifest.json"
)
REFRESH_SCRIPT = ROOT / "scripts" / "refresh_t_only_forward_v1.py"
MATURITY_STAGE = (
    ROOT / "scripts" / "run_t_only_forward_v1_1_maturity_stage_v1_2.py"
)
RUN_STATUS = ROOT / "paper" / "t_only_forward_v1_1" / "daily_run_status_v1_2.json"
RECEIPT_DIRECTORY = (
    ROOT / "paper" / "t_only_forward_v1_1" / "daily_run_receipts_v1_2"
)
LOG_ROOT = ROOT / "output" / "t_only_forward_v1_1_maturity_logs_v1_2"

ACCEPTED_MANIFEST_STATUS = "FROZEN_MATURITY_OUTPUT_ONLY"
NETWORK_FAILURE_MARKERS = (
    "connectionerror",
    "connecttimeout",
    "readtimeout",
    "remotedisconnected",
    "proxyerror",
    "httperror",
    "urlerror",
    "connection aborted",
    "connection reset",
    "timed out",
    "temporary failure",
    "name resolution",
)


def resolve_workspace_path(path: Path) -> Path:
    candidate = path if path.is_absolute() else ROOT / path
    resolved = candidate.resolve()
    resolved.relative_to(ROOT.resolve())
    return resolved


def _content_hash(records: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def verify_manifest(
    manifest_path: Path, expected_manifest_sha256: str
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise RuntimeError("期望清单 SHA-256 必须是 64 位小写十六进制")
    resolved_manifest = resolve_workspace_path(manifest_path)
    if resolved_manifest != DEFAULT_MANIFEST.resolve():
        raise RuntimeError(f"V1.2 入口只接受固定清单：{DEFAULT_MANIFEST}")
    if not resolved_manifest.is_file():
        raise RuntimeError(f"V1.2 冻结清单缺失：{resolved_manifest}")
    actual_manifest_sha256 = sha256_file(resolved_manifest)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise RuntimeError(
            "V1.2 清单文件哈希不匹配："
            f"预期={expected_manifest_sha256}，实际={actual_manifest_sha256}"
        )

    manifest = read_json(resolved_manifest)
    if manifest.get("status") != ACCEPTED_MANIFEST_STATUS:
        raise RuntimeError(f"V1.2 清单状态不可执行：{manifest.get('status')}")
    if manifest.get("research_parameters_changed") is not False:
        raise RuntimeError("V1.2 清单没有确认研究参数保持不变")
    if manifest.get("maturity_thresholds_changed") is not False:
        raise RuntimeError("V1.2 清单没有确认成熟门槛保持不变")

    expected_records = manifest.get("files")
    if not isinstance(expected_records, list) or not expected_records:
        raise RuntimeError("V1.2 清单没有受控文件")
    actual_records: list[dict[str, Any]] = []
    seen: set[str] = set()
    failures: list[dict[str, Any]] = []
    for expected in expected_records:
        if not isinstance(expected, Mapping):
            raise RuntimeError("V1.2 清单文件记录必须是对象")
        relative_path = str(expected.get("path") or "")
        if not relative_path or relative_path in seen:
            raise RuntimeError(f"V1.2 清单路径为空或重复：{relative_path!r}")
        seen.add(relative_path)
        path = resolve_workspace_path(Path(relative_path))
        actual = {
            "path": relative_path,
            "sha256": sha256_file(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None,
        }
        actual_records.append(actual)
        if actual != dict(expected):
            failures.append(
                {
                    "path": relative_path,
                    "expected_sha256": expected.get("sha256"),
                    "actual_sha256": actual["sha256"],
                    "expected_bytes": expected.get("bytes"),
                    "actual_bytes": actual["bytes"],
                }
            )
    actual_records.sort(key=lambda item: str(item["path"]))
    expected_sorted = sorted(
        (dict(item) for item in expected_records), key=lambda item: str(item["path"])
    )
    content_sha256 = _content_hash(actual_records)
    if actual_records != expected_sorted:
        failures.append({"path": "<record-order-or-shape>", "status": "MISMATCH"})
    if content_sha256 != manifest.get("content_sha256"):
        failures.append(
            {
                "path": "<manifest-content>",
                "expected_sha256": manifest.get("content_sha256"),
                "actual_sha256": content_sha256,
            }
        )
    required_tracked = {
        Path(__file__).resolve().relative_to(ROOT.resolve()).as_posix(),
        MATURITY_STAGE.resolve().relative_to(ROOT.resolve()).as_posix(),
        Path("scripts/t_only_forward_maturity_only_v1_2.py").as_posix(),
        REFRESH_SCRIPT.resolve().relative_to(ROOT.resolve()).as_posix(),
        Path("config/t_only_forward_v1_1_freeze_manifest.json").as_posix(),
        Path("config/t_only_forward_v1_automation_manifest.json").as_posix(),
    }
    for required in sorted(required_tracked.difference(seen)):
        failures.append({"path": required, "status": "REQUIRED_FILE_NOT_TRACKED"})
    if failures:
        raise RuntimeError(
            "V1.2 冻结入口校验失败："
            + json.dumps(failures, ensure_ascii=False, separators=(",", ":"))
        )
    return {
        "status": "PASS_V1_2_FROZEN_ENTRYPOINT_VERIFIED",
        "manifest_path": resolved_manifest.relative_to(ROOT).as_posix(),
        "manifest_sha256": actual_manifest_sha256,
        "content_sha256": content_sha256,
        "tracked_file_count": len(actual_records),
    }


def classify_stage_failure(stage: str, stage_record: Mapping[str, Any] | None) -> str:
    if stage != "refresh" or not isinstance(stage_record, Mapping):
        return "PROGRAM_FAILED"
    exit_code = int(stage_record.get("exit_code", 1))
    if exit_code in {2, 3}:
        return "DATA_GATE_FAILED"
    stderr_file = stage_record.get("stderr_file")
    stderr = ""
    if stderr_file:
        path = resolve_workspace_path(Path(str(stderr_file)))
        if path.is_file():
            stderr = path.read_text(encoding="utf-8", errors="replace").lower()
    if any(marker in stderr for marker in NETWORK_FAILURE_MARKERS):
        return "EXTERNAL_FREE_SOURCE_FAILED"
    return "PROGRAM_FAILED"


def audit_outputs(run_started: datetime, run_id: str) -> dict[str, Any]:
    for required in (DATA_GATE, REPORT_JSON, GUIDE_JSON):
        if not required.is_file():
            raise RuntimeError(f"本次运行输出缺失：{required.relative_to(ROOT)}")
    data_gate = read_json(DATA_GATE)
    snapshot = read_json(REPORT_JSON)
    guide = read_json(GUIDE_JSON)
    assert_maturity_only_payload(snapshot)
    assert_maturity_only_payload(guide)
    if data_gate.get("status") != "PASS":
        raise RuntimeError(f"本次数据闸门未通过：{data_gate.get('status')}")
    retrieved_at = datetime.fromisoformat(str(data_gate["retrieved_at"]))
    if retrieved_at < run_started:
        raise RuntimeError("数据闸门不是本次日更生成")
    if snapshot.get("as_of_market_date") != data_gate.get("actual_last_date"):
        raise RuntimeError("成熟度状态与数据闸门的行情截止日不一致")
    if guide.get("as_of_market_date") != snapshot.get("as_of_market_date"):
        raise RuntimeError("每日交付物与成熟度状态的行情截止日不一致")
    if guide.get("current_view") != snapshot.get("current_view"):
        raise RuntimeError("每日交付物与成熟度状态的当前视图不一致")

    new_days = int(snapshot.get("new_trading_days", 0))
    ledger_rows = 0
    duplicate_ledger_dates = 0
    if LEDGER.is_file():
        ledger = pd.read_parquet(LEDGER)
        dates = pd.to_datetime(ledger["date"], errors="raise").dt.normalize()
        ledger_rows = int(len(ledger))
        duplicate_ledger_dates = int(dates.duplicated().sum())
    if ledger_rows != new_days or duplicate_ledger_dates != 0:
        raise RuntimeError("影子账本与成熟度计数不一致")

    run_completeness = {
        "status": "PASS",
        "complete": True,
        "run_id": run_id,
        "data_gate_refreshed_current_run": True,
        "market_date_consistent": True,
        "public_schema_verified": True,
    }
    snapshot = sanitize_existing_outputs(run_completeness=run_completeness)
    if snapshot["completeness"]["overall_complete"] is not True:
        raise RuntimeError("T-only 数据、账本或运行完整性未全部通过")
    return {
        "status": "PASS",
        "current_view": snapshot["current_view"],
        "as_of_market_date": snapshot["as_of_market_date"],
        "new_trading_days": snapshot["new_trading_days"],
        "closed_cycles": snapshot["closed_cycles"],
        "data_completeness": snapshot["completeness"]["data"]["status"],
        "ledger_completeness": snapshot["completeness"]["ledger"]["status"],
        "run_completeness": snapshot["completeness"]["run"]["status"],
        "overall_complete": snapshot["completeness"]["overall_complete"],
        "evaluation_status": snapshot["evaluation_status"],
        "data_gate_sha256": sha256_file(DATA_GATE),
        "public_status_sha256": sha256_file(REPORT_JSON),
        "public_guide_sha256": sha256_file(GUIDE_JSON),
    }


def overall_success_status(snapshot_status: str) -> str:
    if snapshot_status == "MATURE_REVIEW_REQUIRED":
        return "SUCCESS_MATURE_REVIEW_REQUIRED"
    if snapshot_status in {"COLLECTING", "COLLECTING_FORWARD_NOT_STARTED"}:
        return "SUCCESS_MATURITY_ONLY_COLLECTING"
    raise RuntimeError(f"成熟度状态非法：{snapshot_status}")


def write_receipt(run_id: str, payload: Mapping[str, Any]) -> tuple[Path, str]:
    RECEIPT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    receipt_path = RECEIPT_DIRECTORY / f"{run_id}.json"
    receipt_text = json.dumps(payload, ensure_ascii=False, indent=2)
    with receipt_path.open("x", encoding="utf-8") as handle:
        handle.write(receipt_text)
    return receipt_path, sha256_file(receipt_path)


def _read_previous_public_snapshot() -> dict[str, Any] | None:
    if not REPORT_JSON.is_file():
        return None
    payload = read_json(REPORT_JSON)
    if payload.get("artifact_role") == "T_ONLY_MATURITY_AND_COMPLETENESS_ONLY":
        assert_maturity_only_payload(payload)
        return payload
    return {
        "project_id": payload.get("project_id"),
        "as_of_market_date": payload.get("as_of_market_date"),
        "forward_signal_start": payload.get("forward_signal_start"),
        "new_trading_days": int(payload.get("new_trading_days", 0)),
        "closed_cycles": int(payload.get("closed_cycles", 0)),
        "maturity": payload.get("maturity"),
        "source_detailed_status_sha256": sha256_file(REPORT_JSON),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 T-only V1.2 成熟度口径日更")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--stage-timeout-seconds", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    if arguments.stage_timeout_seconds < 30:
        raise ValueError("阶段超时不得少于 30 秒")

    verification = verify_manifest(
        arguments.manifest, arguments.expected_manifest_sha256
    )
    if arguments.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "current_view": "NO_VIEW_NO_DATA_REFRESH_PERFORMED",
                    "manifest_verification": verification,
                    "stages_that_would_run": ["refresh", "maturity_only_forward"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    started = datetime.now(TIMEZONE)
    run_id = started.strftime("%Y%m%dT%H%M%S%f%z")
    log_directory = LOG_ROOT / run_id
    log_directory.mkdir(parents=True, exist_ok=False)
    previous_snapshot = _read_previous_public_snapshot()
    status: dict[str, Any] = {
        "schema_version": "1.2.0",
        "project_id": "510300_T_ONLY_FORWARD_V1_1_DAILY_MATURITY_ONLY",
        "run_id": run_id,
        "started_at": started.isoformat(),
        "finished_at": None,
        "overall_status": "RUNNING",
        "exit_code": None,
        "manifest_verification": verification,
        "log_directory": log_directory.relative_to(ROOT).as_posix(),
        "stages": [],
        "output_audit": None,
        "failure": None,
        "receipt": None,
    }
    atomic_json(RUN_STATUS, status)
    lock_descriptor: int | None = None
    current_stage = "startup"
    failed_stage_record: Mapping[str, Any] | None = None
    exit_code = 1
    try:
        lock_descriptor = acquire_lock(started)
        current_stage = "refresh"
        refresh = run_stage(
            name="refresh",
            script=REFRESH_SCRIPT,
            log_directory=log_directory,
            timeout_seconds=arguments.stage_timeout_seconds,
        )
        status["stages"].append(refresh)
        if int(refresh["exit_code"]) != 0:
            failed_stage_record = refresh
            raise RuntimeError(f"刷新阶段失败，退出码：{refresh['exit_code']}")

        current_stage = "maturity_only_forward"
        forward = run_stage(
            name="maturity_only_forward",
            script=MATURITY_STAGE,
            log_directory=log_directory,
            timeout_seconds=arguments.stage_timeout_seconds,
        )
        status["stages"].append(forward)
        if int(forward["exit_code"]) != 0:
            failed_stage_record = forward
            raise RuntimeError(f"成熟度前瞻阶段失败，退出码：{forward['exit_code']}")

        current_stage = "output_audit"
        output_audit = audit_outputs(started, run_id)
        status["output_audit"] = output_audit
        public_snapshot = read_json(REPORT_JSON)
        status["overall_status"] = overall_success_status(
            str(public_snapshot["status"])
        )
        exit_code = 0
    except Exception as exc:
        failure_class = classify_stage_failure(current_stage, failed_stage_record)
        status["overall_status"] = "NO_VIEW_OPERATIONAL_FAILURE"
        status["failure"] = {
            "class": failure_class,
            "stage": current_stage,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        failure_snapshot = build_operational_failure_snapshot(
            previous_snapshot=previous_snapshot,
            run_id=run_id,
            failure_class=failure_class,
            failure_stage=current_stage,
            failure_message=str(exc),
        )
        publish_snapshot(failure_snapshot)
        exit_code = 1
    finally:
        release_lock(lock_descriptor)
        finished = datetime.now(TIMEZONE)
        status["finished_at"] = finished.isoformat()
        status["duration_seconds"] = (finished - started).total_seconds()
        status["exit_code"] = exit_code
        receipt_payload = dict(status)
        receipt_payload["public_status_sha256"] = (
            sha256_file(REPORT_JSON) if REPORT_JSON.is_file() else None
        )
        receipt_path, receipt_sha256 = write_receipt(run_id, receipt_payload)
        status["receipt"] = {
            "path": receipt_path.relative_to(ROOT).as_posix(),
            "sha256": receipt_sha256,
            "immutable_create_mode": "CREATE_NEW",
        }
        atomic_json(RUN_STATUS, status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
