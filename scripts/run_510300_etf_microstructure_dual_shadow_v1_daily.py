"""双影子V1严格同日前瞻Shadow日终入口。

仅允许在上海本地当天运行；交易日收盘后依次生成基础快照、上交所官方补全
快照和冻结规则重放。只有当日全部信号输入与状态在当日首次形成时，才写入
不可变prospective claim。错过日期不得在以后补记为前瞻证据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ID = "510300_ETF_MICROSTRUCTURE_DUAL_SHADOW_V1"
TIMEZONE = ZoneInfo("Asia/Shanghai")
NOT_BEFORE = "20:30:00"
CALENDAR_FILE = PROJECT_ROOT / "data" / "reference" / "sse_trade_calendar_2026.csv"
COLLECTOR = (
    PROJECT_ROOT
    / "scripts"
    / "collect_510300_etf_microstructure_dual_shadow_v1_snapshot.py"
)
SSE_BUILDER = (
    PROJECT_ROOT
    / "scripts"
    / "build_510300_etf_microstructure_dual_shadow_v1_sse_snapshot.py"
)
EVALUATOR = (
    PROJECT_ROOT
    / "research"
    / "run_510300_etf_microstructure_dual_shadow_v1_forward.py"
)
FORWARD_RECEIPT = (
    PROJECT_ROOT
    / "reports"
    / "frozen"
    / "510300_etf_microstructure_dual_shadow_v1_forward_implementation_receipt.json"
)
SSE_RECEIPT = (
    PROJECT_ROOT
    / "reports"
    / "frozen"
    / "510300_etf_microstructure_dual_shadow_v1_sse_snapshot_parser_receipt.json"
)
FORWARD_ROOT = (
    PROJECT_ROOT / "data" / "forward" / "510300_etf_microstructure_dual_shadow_v1"
)
CLAIM_ROOT = FORWARD_ROOT / "prospective_claims"
ATTEMPT_ROOT = FORWARD_ROOT / "daily_attempts"
STATUS_FILE = (
    PROJECT_ROOT
    / "reports"
    / "forward"
    / "510300_etf_microstructure_dual_shadow_v1_daily_status.json"
)
LEDGER_FILE = (
    PROJECT_ROOT
    / "reports"
    / "forward"
    / "510300_etf_microstructure_dual_shadow_v1_prospective_ledger.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _immutable_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_implementations() -> dict[str, Any]:
    forward = json.loads(FORWARD_RECEIPT.read_text(encoding="utf-8"))
    sse = json.loads(SSE_RECEIPT.read_text(encoding="utf-8"))
    expected = {
        COLLECTOR: forward["implementations"][
            "scripts\\collect_510300_etf_microstructure_dual_shadow_v1_snapshot.py"
        ]["sha256"],
        EVALUATOR: forward["implementations"][
            "research\\run_510300_etf_microstructure_dual_shadow_v1_forward.py"
        ]["sha256"],
        SSE_BUILDER: sse["implementation_sha256"],
    }
    rows: list[dict[str, str]] = []
    for path, expected_hash in expected.items():
        actual = _sha256(path)
        if actual != expected_hash:
            raise ValueError(
                f"日终入口依赖哈希漂移：{path.name},expected={expected_hash},actual={actual}"
            )
        rows.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": actual,
            }
        )
    return {"status": "PASS", "implementations": rows}


def _calendar() -> pd.DatetimeIndex:
    frame = pd.read_csv(CALENDAR_FILE, dtype=str)
    if "trade_date" not in frame.columns:
        raise ValueError("上交所交易日历缺少trade_date")
    values = pd.to_datetime(frame["trade_date"], errors="raise").dt.normalize()
    return pd.DatetimeIndex(values.sort_values().unique())


def _run_json(command: list[str], *, timeout_seconds: int = 360) -> dict[str, Any]:
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
        check=False,
    )
    if process.returncode != 0:
        stderr_tail = process.stderr[-3000:]
        stdout_tail = process.stdout[-3000:]
        raise RuntimeError(
            f"子流程失败 exit={process.returncode};stdout_tail={stdout_tail};stderr_tail={stderr_tail}"
        )
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"子流程标准输出不是单一JSON：{process.stdout[-3000:]}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("子流程JSON顶层不是对象")
    return payload


def _new_attempt_path(as_of: pd.Timestamp, now: datetime) -> Path:
    return (
        ATTEMPT_ROOT
        / as_of.strftime("%Y%m%d")
        / f"attempt_{now.strftime('%Y%m%dT%H%M%S_%f%z')}.json"
    )


def _write_status_and_attempt(
    payload: dict[str, Any], as_of: pd.Timestamp, now: datetime
) -> None:
    _immutable_json(payload, _new_attempt_path(as_of, now))
    _atomic_json(payload, STATUS_FILE)


def _load_claims() -> list[dict[str, Any]]:
    if not CLAIM_ROOT.exists():
        return []
    claims: list[dict[str, Any]] = []
    for path in sorted(CLAIM_ROOT.glob("*.json")):
        claim = json.loads(path.read_text(encoding="utf-8"))
        claim["claim_file"] = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        claim["claim_sha256"] = _sha256(path)
        claims.append(claim)
    return claims


def _update_ledger(now: datetime) -> dict[str, Any]:
    claims = _load_claims()
    payload = {
        "project_id": PROJECT_ID,
        "status": "PROSPECTIVE_SHADOW_LEDGER_ACTIVE",
        "updated_at_asia_shanghai": now.isoformat(),
        "prospective_claim_count": int(len(claims)),
        "first_signal_date": claims[0]["signal_date"] if claims else None,
        "last_signal_date": claims[-1]["signal_date"] if claims else None,
        "latest_research_state": (
            claims[-1]["fixed_rule_state"] if claims else None
        ),
        "claims": claims,
        "boundaries": {
            "same_day_only_no_backfill": True,
            "research_shadow_only": True,
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _atomic_json(payload, LEDGER_FILE)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="运行双影子V1严格同日前瞻日终流程")
    parser.add_argument("--as-of", default=None, help="仅允许上海本地当天，格式YYYY-MM-DD")
    arguments = parser.parse_args()
    now = datetime.now(TIMEZONE)
    today = pd.Timestamp(now.date())
    as_of = pd.Timestamp(arguments.as_of or now.date()).normalize()
    if as_of != today:
        payload = {
            "project_id": PROJECT_ID,
            "status": "MISSED_OR_NOT_DUE_NO_BACKFILL",
            "as_of": as_of.date().isoformat(),
            "local_today": today.date().isoformat(),
            "reason": "严格前瞻入口拒绝为非当天日期补写claim",
            "generated_at_asia_shanghai": now.isoformat(),
        }
        _write_status_and_attempt(payload, as_of, now)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    claim_path = CLAIM_ROOT / f"{as_of.strftime('%Y%m%d')}.json"
    if claim_path.exists():
        claim = json.loads(claim_path.read_text(encoding="utf-8"))
        payload = {
            "project_id": PROJECT_ID,
            "status": "ALREADY_CLAIMED_NO_DUPLICATE_RUN",
            "as_of": as_of.date().isoformat(),
            "claim_file": str(claim_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "claim_sha256": _sha256(claim_path),
            "fixed_rule_state": claim["fixed_rule_state"],
            "generated_at_asia_shanghai": now.isoformat(),
        }
        _atomic_json(payload, STATUS_FILE)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    implementation_audit = _verify_implementations()
    calendar = _calendar()
    if as_of not in calendar:
        payload = {
            "project_id": PROJECT_ID,
            "status": "NOT_TRADING_DAY",
            "as_of": as_of.date().isoformat(),
            "implementation_audit": implementation_audit,
            "generated_at_asia_shanghai": now.isoformat(),
        }
        _write_status_and_attempt(payload, as_of, now)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    if now.strftime("%H:%M:%S") < NOT_BEFORE:
        payload = {
            "project_id": PROJECT_ID,
            "status": "NOT_DUE_BEFORE_20_30",
            "as_of": as_of.date().isoformat(),
            "current_time": now.strftime("%H:%M:%S"),
            "implementation_audit": implementation_audit,
            "generated_at_asia_shanghai": now.isoformat(),
        }
        _write_status_and_attempt(payload, as_of, now)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    run_token = now.strftime("%Y%m%dT%H%M%S%z")
    try:
        base = _run_json(
            [
                sys.executable,
                str(COLLECTOR),
                "--end",
                as_of.date().isoformat(),
                "--run-id",
                f"{run_token}_daily_base",
            ]
        )
        base_manifest_path = Path(base["snapshot_manifest"])
        sse = _run_json(
            [
                sys.executable,
                str(SSE_BUILDER),
                "--base-snapshot-dir",
                str(base_manifest_path.parent),
                "--run-id",
                f"{run_token}_daily_sse",
            ]
        )
        sse_manifest_path = Path(sse["snapshot_manifest"])
        replay = _run_json(
            [
                sys.executable,
                str(EVALUATOR),
                "--snapshot-dir",
                str(sse_manifest_path.parent),
                "--run-id",
                f"{run_token}_daily_replay",
            ],
            timeout_seconds=480,
        )
    except Exception as exc:
        payload = {
            "project_id": PROJECT_ID,
            "status": "PROGRAM_FAILED_NO_CLAIM",
            "as_of": as_of.date().isoformat(),
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "implementation_audit": implementation_audit,
            "generated_at_asia_shanghai": now.isoformat(),
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "live_trading_authorized": False,
        }
        _write_status_and_attempt(payload, as_of, now)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    as_of_text = as_of.date().isoformat()
    complete_dates = set(sse["signal_calendar"]["complete_signal_dates"])
    daily_rows = replay["fixed_rule_evidence"]["daily_post_cutoff"]
    matching = [row for row in daily_rows if row["date"] == as_of_text]
    same_day_collection = pd.Timestamp(
        sse["retrieved_at_asia_shanghai"]
    ).date() == as_of.date()
    eligible = bool(
        as_of_text in complete_dates
        and len(matching) == 1
        and same_day_collection
        and replay["freeze"]["historical_prefix_replication"]["total_mismatches"] == 0
    )
    if not eligible:
        payload = {
            "project_id": PROJECT_ID,
            "status": "PARTIAL_SUCCESS_SAME_DAY_INPUT_NOT_COMPLETE_NO_CLAIM",
            "as_of": as_of_text,
            "same_day_collection": same_day_collection,
            "listed_as_complete_signal": as_of_text in complete_dates,
            "matching_fixed_rule_rows": int(len(matching)),
            "base_snapshot_manifest": str(base_manifest_path),
            "sse_snapshot_manifest": str(sse_manifest_path),
            "replay_status": replay["status"],
            "implementation_audit": implementation_audit,
            "generated_at_asia_shanghai": now.isoformat(),
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "live_trading_authorized": False,
        }
        _write_status_and_attempt(payload, as_of, now)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 2

    row = matching[0]
    future_dates = calendar[calendar > as_of]
    next_execution_date = future_dates[0].date().isoformat() if len(future_dates) else None
    replay_report = PROJECT_ROOT / "data" / "forward" / "510300_etf_microstructure_dual_shadow_v1" / "runs" / f"{run_token}_daily_replay" / "run_report.json"
    claim = {
        "project_id": PROJECT_ID,
        "status": "PROSPECTIVE_SIGNAL_CLAIMED_SHADOW_ONLY",
        "signal_date": as_of_text,
        "expected_t_plus_1_execution_date": next_execution_date,
        "claimed_at_asia_shanghai": datetime.now(TIMEZONE).isoformat(),
        "fixed_rule_state": row,
        "evidence": {
            "base_snapshot_manifest": str(base_manifest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "base_snapshot_manifest_sha256": _sha256(base_manifest_path),
            "sse_snapshot_manifest": str(sse_manifest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sse_snapshot_manifest_sha256": _sha256(sse_manifest_path),
            "replay_report": str(replay_report.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "replay_report_sha256": _sha256(replay_report),
        },
        "quality": {
            "same_day_only_no_backfill": True,
            "complete_signal_inputs": True,
            "historical_prefix_total_mismatches": 0,
            "replay_status": replay["status"],
        },
        "boundaries": {
            "research_shadow_only": True,
            "position_mapping": "DISABLED",
            "order_generation": "DISABLED",
            "broker_connection": "DISABLED",
            "live_trading_authorized": False,
        },
    }
    _immutable_json(claim, claim_path)
    ledger = _update_ledger(datetime.now(TIMEZONE))
    payload = {
        "project_id": PROJECT_ID,
        "status": "PROSPECTIVE_SIGNAL_CLAIMED_SHADOW_ONLY",
        "as_of": as_of_text,
        "claim_file": str(claim_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "claim_sha256": _sha256(claim_path),
        "fixed_rule_state": row,
        "combined_extended_all_five_starts_pass": replay["fixed_rule_evidence"][
            "combined_extended_all_five_starts_pass"
        ],
        "prospective_claim_count": ledger["prospective_claim_count"],
        "implementation_audit": implementation_audit,
        "generated_at_asia_shanghai": datetime.now(TIMEZONE).isoformat(),
        "position_mapping": "DISABLED",
        "order_generation": "DISABLED",
        "live_trading_authorized": False,
    }
    _write_status_and_attempt(payload, as_of, now)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
