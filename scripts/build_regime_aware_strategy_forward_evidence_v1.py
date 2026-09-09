"""从冻结后的逐观察日账本生成 V1 不可变条件证据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.regime_aware_strategy_allocator_v1 import (
    CONFIG,
    ContractError,
    EvidenceError,
    load_contract,
    load_jsonl,
    parse_zoned_datetime,
    sha256_file,
    validate_environment_records,
    validate_strategy_records,
)
from research.regime_aware_strategy_forward_evidence_v1 import (
    build_evidence_core,
    evaluate_strategy_environment_forward_evidence,
    validate_forward_observations,
)
from scripts.freeze_regime_aware_strategy_allocator_v1 import MANIFEST, verify


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def append_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    """保留既有字节顺序，批量追加后原子替换。"""

    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    appended = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )
    atomic_text(path, existing + appended)


def _build_receipt_id(generated_at: datetime, manifest_sha256: str) -> str:
    return (
        f"{generated_at.astimezone(ZoneInfo('UTC')):%Y%m%dT%H%M%S%fZ}_"
        f"{os.getpid()}_{manifest_sha256[:12]}"
    )


def write_build_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
    """为成功、无证据和失败状态统一写入最新状态及不可变回执。"""

    contract = load_contract(CONFIG)
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    manifest_sha = sha256_file(MANIFEST)
    receipt_id = _build_receipt_id(generated_at, manifest_sha)
    status_path = ROOT / PurePosixPath(
        contract["forward_evaluation"]["latest_build_status_json"]
    )
    receipt_directory = ROOT / PurePosixPath(
        contract["forward_evaluation"]["immutable_build_receipt_directory"]
    )
    receipt_path = receipt_directory / f"{receipt_id}.json"
    if receipt_path.exists():
        raise FileExistsError(f"前向证据构建回执已存在：{receipt_path}")
    payload.update(
        {
            "artifact_generated_at": generated_at.isoformat(),
            "build_receipt_id": receipt_id,
            "build_receipt_path": receipt_path.relative_to(ROOT).as_posix(),
            "allocator_manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
            "allocator_manifest_sha256": manifest_sha,
        }
    )
    atomic_json(status_path, payload)
    failed = payload.get("run_status") == "FAILED" or payload.get("status") == "FORWARD_EVIDENCE_BUILD_FAILED"
    receipt = {
        "schema_version": "1.0.0",
        "receipt_id": receipt_id,
        "immutable_receipt": True,
        "generated_at": generated_at.isoformat(),
        "evaluator_id": contract["forward_evaluation"]["evaluator_id"],
        "candidate_id": contract["protocol"]["candidate_id"],
        "run_status": "FAILED" if failed else "SUCCESS",
        "build_status": payload.get("status"),
        "decision_at": payload.get("decision_at"),
        "allocator_manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "allocator_manifest_sha256": manifest_sha,
        "status_json": {
            "path": status_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(status_path),
        },
        "validated_observation_count": payload.get("validated_observation_count", 0),
        "generated_evidence_count": payload.get("generated_evidence_count", 0),
        "strategy_evidence_appended": payload.get("strategy_evidence_appended", False),
        "conditional_forward_pass_generated": payload.get(
            "conditional_forward_pass_generated", False
        ),
        "historical_winner_ranking_used": False,
        "historical_backfill_used": False,
        "paper_position_generation": False,
        "shadow_signal_generation": False,
        "order_generation": False,
        "live_trading_authorized": False,
    }
    atomic_json(receipt_path, receipt)
    return {
        "status_json": status_path.relative_to(ROOT).as_posix(),
        "status_json_sha256": sha256_file(status_path),
        "receipt": receipt_path.relative_to(ROOT).as_posix(),
        "receipt_sha256": sha256_file(receipt_path),
    }


def _evidence_id(
    *,
    strategy_id: str,
    environment_id: str,
    decision_at: datetime,
    ledger_sha256: str,
) -> str:
    canonical = "|".join(
        [strategy_id, environment_id, decision_at.isoformat(), ledger_sha256]
    )
    suffix = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]
    return f"RAE_{decision_at:%Y%m%dT%H%M%S}_{suffix}"


def build(*, decision_at: datetime) -> dict[str, Any]:
    runtime_now = datetime.now(ZoneInfo("Asia/Shanghai"))
    if decision_at > runtime_now:
        raise EvidenceError("前向证据评价时点不得晚于当前可得时间")
    verification = verify()
    if verification["failure_count"]:
        raise ContractError(f"V1 清单验证失败：{verification['failures']}")
    contract = load_contract(CONFIG)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_frozen_at = parse_zoned_datetime(manifest["frozen_at"], "manifest.frozen_at")
    ledger_path = ROOT / PurePosixPath(
        contract["inputs"]["forward_observation_ledger_jsonl"]
    )
    if not ledger_path.is_file():
        return {
            "schema_version": "1.0.0",
            "evaluator_id": contract["forward_evaluation"]["evaluator_id"],
            "status": "FORWARD_OBSERVATION_LEDGER_MISSING_NO_VIEW",
            "decision_at": decision_at.isoformat(),
            "ledger_path": ledger_path.relative_to(ROOT).as_posix(),
            "ledger_sha256": None,
            "validated_observation_count": 0,
            "generated_evidence_count": 0,
            "strategy_evidence_appended": False,
            "cash_only_remains_required": True,
            "live_trading_authorized": False,
        }
    environment_path = ROOT / PurePosixPath(
        contract["inputs"]["environment_evidence_jsonl"]
    )
    raw_environment = load_jsonl(environment_path, allow_missing=True)
    if not raw_environment:
        return {
            "schema_version": "1.0.0",
            "evaluator_id": contract["forward_evaluation"]["evaluator_id"],
            "status": "ENVIRONMENT_EVIDENCE_MISSING_OR_EMPTY_NO_VIEW",
            "decision_at": decision_at.isoformat(),
            "environment_evidence_path": environment_path.relative_to(ROOT).as_posix(),
            "environment_evidence_sha256": (
                sha256_file(environment_path) if environment_path.is_file() else None
            ),
            "validated_environment_record_count": 0,
            "generated_evidence_count": 0,
            "strategy_evidence_appended": False,
            "cash_only_remains_required": True,
            "live_trading_authorized": False,
        }
    environment_records = validate_environment_records(
        raw_environment,
        contract,
        root=ROOT,
    )
    raw_observations = load_jsonl(ledger_path, allow_missing=False)
    observations = validate_forward_observations(
        raw_observations,
        contract,
        manifest_frozen_at=manifest_frozen_at,
        environment_records=environment_records,
        root=ROOT,
    )
    groups = sorted(
        {
            (record["strategy_id"], record["environment_id"])
            for record in observations
            if record["available_at"] <= decision_at
        }
    )
    if not groups:
        return {
            "schema_version": "1.0.0",
            "evaluator_id": contract["forward_evaluation"]["evaluator_id"],
            "status": "NO_POINT_IN_TIME_FORWARD_OBSERVATIONS_AT_DECISION",
            "decision_at": decision_at.isoformat(),
            "ledger_path": ledger_path.relative_to(ROOT).as_posix(),
            "ledger_sha256": sha256_file(ledger_path),
            "validated_observation_count": int(len(observations)),
            "environment_evidence_path": environment_path.relative_to(ROOT).as_posix(),
            "environment_evidence_sha256": sha256_file(environment_path),
            "validated_environment_record_count": int(len(environment_records)),
            "generated_evidence_count": 0,
            "strategy_evidence_appended": False,
            "cash_only_remains_required": True,
            "live_trading_authorized": False,
        }

    strategy_evidence_path = ROOT / PurePosixPath(
        contract["inputs"]["strategy_evidence_jsonl"]
    )
    previous_raw_evidence = load_jsonl(strategy_evidence_path, allow_missing=True)
    previous_evidence = validate_strategy_records(
        previous_raw_evidence,
        contract,
        manifest_frozen_at=manifest_frozen_at,
        manifest_tracked_content_sha256=manifest["tracked_content_sha256"],
        root=ROOT,
    )
    ledger_sha = sha256_file(ledger_path)
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    report_directory = ROOT / PurePosixPath(
        contract["forward_evaluation"]["immutable_report_directory"]
    )
    pending_records: list[dict[str, Any]] = []
    report_summaries: list[dict[str, Any]] = []
    skipped_summaries: list[dict[str, Any]] = []
    existing_ids = {
        record.get("evidence_id")
        for record in previous_evidence
        if isinstance(record.get("evidence_id"), str)
    }
    for strategy_id, environment_id in groups:
        evidence_id = _evidence_id(
            strategy_id=strategy_id,
            environment_id=environment_id,
            decision_at=decision_at,
            ledger_sha256=ledger_sha,
        )
        if evidence_id in existing_ids:
            skipped_summaries.append(
                {
                    "strategy_id": strategy_id,
                    "environment_id": environment_id,
                    "reason": "ALREADY_EVALUATED_SAME_DECISION_AND_LEDGER",
                    "evidence_id": evidence_id,
                }
            )
            continue
        evaluation = evaluate_strategy_environment_forward_evidence(
            observations,
            contract,
            strategy_id=strategy_id,
            environment_id=environment_id,
            decision_at=decision_at,
            previous_evidence=previous_evidence,
        )
        matching_previous = [
            record
            for record in previous_evidence
            if record["strategy_id"] == strategy_id
            and record["environment_id"] == environment_id
            and record["available_at"] <= decision_at
        ]
        latest_previous = matching_previous[-1] if matching_previous else None
        previous_days = (
            int(latest_previous["forward_observation_days"] or 0)
            if latest_previous is not None
            else 0
        )
        current_days = int(evaluation["forward_observation_days"])
        if current_days < previous_days:
            raise EvidenceError(
                f"前向完整观察数倒退：{strategy_id}/{environment_id}，"
                f"previous={previous_days}，current={current_days}"
            )
        current_latest_observation = parse_zoned_datetime(
            evaluation["latest_forward_observation_at"],
            "evaluation.latest_forward_observation_at",
        )
        previous_latest_observation = (
            latest_previous["latest_forward_observation_at"]
            if latest_previous is not None
            else None
        )
        if (
            previous_latest_observation is not None
            and current_latest_observation <= previous_latest_observation
        ):
            skipped_summaries.append(
                {
                    "strategy_id": strategy_id,
                    "environment_id": environment_id,
                    "reason": "NO_NEW_FORWARD_OBSERVATION",
                    "forward_observation_days": current_days,
                }
            )
            continue
        confirmation_progress = evaluation["confirmation_progress"]
        if (
            evaluation["preliminary_statistical_pass"]
            and confirmation_progress["previous_confirmation_count"] > 0
            and not confirmation_progress["increment_allowed"]
        ):
            skipped_summaries.append(
                {
                    "strategy_id": strategy_id,
                    "environment_id": environment_id,
                    "reason": "CONFIRMATION_REQUIRES_168H_AND_7_NEW_COMPLETE_DAYS",
                    "forward_observation_days": current_days,
                    "confirmation_progress": confirmation_progress,
                }
            )
            continue
        core = build_evidence_core(
            evaluation,
            evidence_id=evidence_id,
            recorded_at=generated_at,
            available_at=generated_at,
        )
        report_path = report_directory / f"{evidence_id}.json"
        if report_path.exists():
            raise FileExistsError(f"不可变条件证据报告已存在：{report_path}")
        report = {
            "schema_version": "1.0.0",
            "evaluator_id": contract["forward_evaluation"]["evaluator_id"],
            "candidate_id": contract["protocol"]["candidate_id"],
            "allocator_tracked_content_sha256": manifest["tracked_content_sha256"],
            "allocator_manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
            "allocator_manifest_sha256": verification["manifest_sha256"],
            "generated_at": generated_at.isoformat(),
            "decision_at": decision_at.isoformat(),
            "ledger": {
                "path": ledger_path.relative_to(ROOT).as_posix(),
                "bytes": int(ledger_path.stat().st_size),
                "sha256": ledger_sha,
                "validated_observation_count": int(len(observations)),
            },
            "environment_evidence": {
                "path": environment_path.relative_to(ROOT).as_posix(),
                "bytes": int(environment_path.stat().st_size),
                "sha256": sha256_file(environment_path),
                "validated_record_count": int(len(environment_records)),
            },
            "evidence_core": core,
            "evaluation": evaluation,
            "performance_used_for_recent_winner_ranking": False,
            "historical_backfill_used": False,
            "paper_position_generation": False,
            "shadow_signal_generation": False,
            "order_generation": False,
            "live_trading_authorized": False,
        }
        atomic_json(report_path, report)
        report_sha = sha256_file(report_path)
        record = {
            **core,
            "source_path": report_path.relative_to(ROOT).as_posix(),
            "source_sha256": report_sha,
        }
        pending_records.append(record)
        report_summaries.append(
            {
                "evidence_id": evidence_id,
                "strategy_id": strategy_id,
                "environment_id": environment_id,
                "meta_evidence_status": evaluation["meta_evidence_status"],
                "forward_observation_days": evaluation["forward_observation_days"],
                "independent_environment_episodes": evaluation[
                    "independent_environment_episodes"
                ],
                "preliminary_statistical_pass": evaluation[
                    "preliminary_statistical_pass"
                ],
                "report_path": report_path.relative_to(ROOT).as_posix(),
                "report_sha256": report_sha,
            }
        )
    if pending_records:
        append_jsonl_atomic(strategy_evidence_path, pending_records)
    evidence_path_exists = strategy_evidence_path.is_file()
    return {
        "schema_version": "1.0.0",
        "evaluator_id": contract["forward_evaluation"]["evaluator_id"],
        "status": (
            "FORWARD_CONDITIONAL_EVIDENCE_APPENDED"
            if pending_records
            else "NO_NEW_QUALIFYING_FORWARD_EVIDENCE"
        ),
        "decision_at": decision_at.isoformat(),
        "generated_at": generated_at.isoformat(),
        "ledger_path": ledger_path.relative_to(ROOT).as_posix(),
        "ledger_sha256": ledger_sha,
        "validated_observation_count": int(len(observations)),
        "environment_evidence_path": environment_path.relative_to(ROOT).as_posix(),
        "environment_evidence_sha256": sha256_file(environment_path),
        "validated_environment_record_count": int(len(environment_records)),
        "generated_evidence_count": int(len(pending_records)),
        "strategy_evidence_path": strategy_evidence_path.relative_to(ROOT).as_posix(),
        "strategy_evidence_sha256": (
            sha256_file(strategy_evidence_path) if evidence_path_exists else None
        ),
        "strategy_evidence_appended": bool(pending_records),
        "evidence_reports": report_summaries,
        "skipped_groups": skipped_summaries,
        "conditional_forward_pass_generated": any(
            summary["meta_evidence_status"] == "CONDITIONAL_FORWARD_PASSED"
            for summary in report_summaries
        ),
        "allocator_cash_fallback_must_be_recomputed": True,
        "historical_winner_ranking_used": False,
        "historical_backfill_used": False,
        "live_trading_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="从真正前向观察账本生成 V1 条件证据")
    parser.add_argument(
        "--decision-at",
        help="带时区 ISO 评价时点；省略时使用当前亚洲/上海时间",
    )
    args = parser.parse_args()
    exit_code = 0
    try:
        decision_at = (
            datetime.now(ZoneInfo("Asia/Shanghai"))
            if args.decision_at is None
            else parse_zoned_datetime(args.decision_at, "--decision-at")
        )
        payload = build(decision_at=decision_at)
    except (ContractError, EvidenceError, FileNotFoundError, FileExistsError, OSError, ValueError, json.JSONDecodeError) as exc:
        payload = {
            "schema_version": "1.0.0",
            "evaluator_id": "REGIME_AWARE_STRATEGY_FORWARD_EVIDENCE_V1",
            "run_status": "FAILED",
            "status": "FORWARD_EVIDENCE_BUILD_FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "strategy_evidence_appended": False,
            "strategy_eligibility_claimed": False,
            "live_trading_authorized": False,
        }
        exit_code = 2
    if MANIFEST.is_file():
        try:
            payload["artifact_write"] = write_build_artifacts(payload)
        except (OSError, ValueError, FileExistsError, json.JSONDecodeError) as artifact_exc:
            payload["artifact_write_error"] = str(artifact_exc)
            exit_code = 2
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
