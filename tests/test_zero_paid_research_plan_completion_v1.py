from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import audit_zero_paid_research_plan_completion_v1 as audit


def test_frozen_manifests_are_current_and_exact() -> None:
    main = audit.verify_manifest(audit.MAIN_MANIFEST)
    t_only = audit.verify_manifest(audit.T_ONLY_MANIFEST)
    cb_sufficiency = audit.verify_file_map_manifest(audit.CB_SUFFICIENCY_ARCHIVE)
    cb_lifecycle = audit.verify_file_map_manifest(audit.CB_LIFECYCLE_ARCHIVE)

    assert main["status"] == "PASS"
    assert main["manifest_sha256"] == audit.MAIN_MANIFEST_SHA256
    assert main["failure_count"] == 0
    assert t_only["status"] == "PASS"
    assert t_only["manifest_sha256"] == audit.T_ONLY_MANIFEST_SHA256
    assert t_only["failure_count"] == 0
    assert cb_sufficiency["status"] == "PASS"
    assert cb_sufficiency["tracked_file_count"] == 5
    assert cb_sufficiency["failure_count"] == 0
    assert cb_lifecycle["status"] == "PASS"
    assert cb_lifecycle["tracked_file_count"] == 2569
    assert cb_lifecycle["failure_count"] == 0

    frozen_inputs = audit.read_json(audit.FROZEN_INPUT_AUDIT)
    industry = audit.read_json(audit.INDUSTRY_STATUS)
    industry_boundary = audit.verify_industry_frozen_input_boundary(
        frozen_inputs, industry, audit.SUPERVISOR_CONFIG
    )
    assert industry_boundary["status"] == "PASS"
    assert industry_boundary["snapshot_record_count"] == 4
    assert industry_boundary["failure_count"] == 0
    assert industry_boundary["checks"]["current_supervisor_launchers_exact"] is True
    assert (
        industry_boundary["checks"]["current_supervisor_evidence_globs_exact"]
        is True
    )

    v1_8_industry_boundary = audit.verify_industry_frozen_input_boundary(
        frozen_inputs,
        industry,
        audit.V1_8_SUPERVISOR_CONFIG,
        pcf_task_launcher=(
            "scripts/run_510300_primary_market_collection_task_v1_2_1.ps1"
        ),
        pcf_evidence_glob=(
            "reports/data_quality/primary_market_task_runs_v1_2_1/*.json"
        ),
    )
    assert v1_8_industry_boundary["status"] == "PASS"
    assert (
        v1_8_industry_boundary["checks"]["current_supervisor_launchers_exact"]
        is True
    )
    assert (
        v1_8_industry_boundary["checks"][
            "current_supervisor_evidence_globs_exact"
        ]
        is True
    )


def test_zero_paid_policy_and_source_registry_contract_are_complete() -> None:
    result = audit.verify_zero_paid_data_contract(
        audit.ZERO_PAID_POLICY,
        audit.CANONICAL_SOURCE_REGISTRY,
        audit.SOURCE_REGISTRY,
    )

    assert result["status"] == "PASS"
    assert result["policy_status"] == "PASS"
    assert result["policy_sha256"] == audit.ZERO_PAID_POLICY_SHA256
    assert result["missing_policy_tokens"] == []
    assert result["registry_status"] == "PASS"
    assert result["registry_row_count"] == 16
    assert result["missing_registry_columns"] == []
    assert result["empty_required_registry_cells"] == []
    assert result["duplicate_source_ids"] == []
    assert result["nonzero_cost_source_ids"] == []
    assert result["raw_preservation_not_required"] == []
    assert result["registry_checks"]["canonical_matches_frozen_v1_1"] is True


def test_industry_maturity_contract_recomputes_eligibility() -> None:
    payload = audit.read_json(audit.INDUSTRY_STATUS)

    pending = audit.verify_industry_maturity_contract(payload)
    tampered = json.loads(json.dumps(payload))
    tampered["maturity"]["calibration"]["eligible"] = True
    failed = audit.verify_industry_maturity_contract(tampered)

    assert pending["status"] == "PENDING_MATURITY"
    assert pending["contract_valid"] is True
    assert pending["origin_cluster_count"] == 1
    assert pending["origin_cluster_count_recomputed"] == 1
    assert pending["calibration_eligible_recomputed"] is False
    assert failed["status"] == "FAILED_CONTRACT"
    assert "CALIBRATION_ELIGIBILITY_MISMATCH" in failed["failures"]


def test_pcf_maturity_contract_recomputes_daily_quality_counts() -> None:
    payload = audit.read_json(audit.PCF_READINESS)

    pending = audit.verify_pcf_maturity_contract(payload)
    tampered = json.loads(json.dumps(payload))
    tampered["full_coverage_days"] = 120
    failed = audit.verify_pcf_maturity_contract(tampered)

    assert pending["status"] == "PENDING_MATURITY"
    assert pending["contract_valid"] is True
    assert pending["observed_trading_days_recomputed"] == 4
    assert pending["full_coverage_days_recomputed"] == 2
    assert pending["crosscheck_passed_day_count_recomputed"] == 0
    assert failed["status"] == "FAILED_CONTRACT"
    assert "COUNT_MISMATCH:full_coverage_days" in failed["failures"]


def test_t_only_maturity_contract_recomputes_thresholds_and_integrity() -> None:
    payload = audit.read_json(audit.T_ONLY_STATUS)
    pending = audit.verify_t_only_maturity_contract(payload)
    mature = json.loads(json.dumps(payload))
    mature.update(
        {
            "new_trading_days": 252,
            "closed_cycles": 3,
            "current_view": "MATURE_REVIEW_REQUIRED",
            "evaluation_status": "MATURE_THRESHOLD_REACHED_REVIEW_SEPARATELY",
        }
    )
    mature["maturity"]["mature"] = True
    mature["completeness"]["ledger"].update(
        {
            "expected_rows": 252,
            "actual_rows": 252,
            "duplicate_dates": 0,
            "complete": True,
        }
    )
    mature["completeness"]["overall_complete"] = True

    passed = audit.verify_t_only_maturity_contract(mature)
    mature["maturity"]["trading_days_required"] = 251
    failed = audit.verify_t_only_maturity_contract(mature)

    assert pending["status"] == "PENDING_MATURITY"
    assert pending["contract_valid"] is True
    assert pending["maturity_recomputed"] is False
    assert passed["status"] == "PASS"
    assert passed["complete"] is True
    assert failed["status"] == "FAILED_CONTRACT"
    assert "MATURITY_THRESHOLDS_CHANGED" in failed["failures"]


def test_v1_7_test_boundary_is_bound_to_current_frozen_manifest() -> None:
    boundary = audit.read_json(audit.V1_7_TEST_BOUNDARY_AUDIT)
    live_check = audit.verify_runtime_test_boundary(
        audit.MAIN_MANIFEST, audit.SUPERVISOR_CONFIG
    )

    assert boundary["status"] == "PASS_TESTS_EXCLUDED_FROM_RUNTIME_MODEL_BOUNDARY"
    assert boundary["frozen_manifest"]["manifest_sha256"] == audit.MAIN_MANIFEST_SHA256
    assert boundary["findings"]["tracked_test_file_count"] == 0
    assert boundary["findings"]["runtime_test_import_count"] == 0
    assert boundary["findings"]["config_launcher_count"] == 3
    assert boundary["findings"]["config_launchers_all_tracked"] is True
    assert boundary["boundary"]["tests_are_runtime_or_model_inputs"] is False
    assert boundary["boundary"]["tests_are_hashed_in_frozen_manifest"] is False
    assert boundary["boundary"]["runtime_files_remain_frozen"] is True
    assert live_check["status"] == "PASS"
    assert live_check["tracked_file_count"] == 40
    assert live_check["tracked_python_file_count"] == 23
    assert live_check["tracked_test_file_count"] == 0
    assert live_check["runtime_test_import_count"] == 0
    assert live_check["python_parse_failure_count"] == 0
    assert live_check["config_launcher_count"] == 3
    assert live_check["config_launchers_all_tracked"] is True


def test_close_windows_are_read_from_both_frozen_supervisors() -> None:
    expected = {
        "INDUSTRY_EXPECTATION_GAP": (
            audit.time(17, 5),
            audit.time(23, 30),
        ),
        "PRIORITY_FORWARD_STATUS": (
            audit.time(17, 15),
            audit.time(23, 50),
        ),
    }

    assert audit.supervisor_close_task_windows(audit.SUPERVISOR_CONFIG) == expected
    assert audit.supervisor_close_task_windows(audit.V1_8_SUPERVISOR_CONFIG) == expected


def test_active_runtime_switch_requires_exact_v1_8_deployment_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment_audit = tmp_path / "deployment.json"
    monkeypatch.setattr(audit, "V1_8_DEPLOYMENT_AUDIT", deployment_audit)

    legacy = audit.active_runtime_contract()
    assert legacy["runtime_version"] == "V1_7"
    assert legacy["deployment_audit_present"] is False
    assert legacy["deployment_audit_valid"] is True

    deployment_audit.write_text(
        json.dumps(
            {
                "status": "PASS_V1_8_DEPLOYED_FOR_NEXT_ELIGIBLE_WINDOW",
                "effective_from": "2026-08-27T09:20:00+08:00",
                "same_day_pcf_retry_performed": True,
                "frozen_manifest": {
                    "path": "config/priority_forward_research_operations_v1_8_manifest.json",
                    "sha256": audit.V1_8_MANIFEST_SHA256,
                },
            }
        ),
        encoding="utf-8",
    )
    rejected = audit.active_runtime_contract()
    assert rejected["runtime_version"] == "V1_7"
    assert rejected["deployment_audit_present"] is True
    assert rejected["deployment_audit_valid"] is False

    payload = json.loads(deployment_audit.read_text(encoding="utf-8"))
    payload["same_day_pcf_retry_performed"] = False
    deployment_audit.write_text(json.dumps(payload), encoding="utf-8")
    activated = audit.active_runtime_contract()
    assert activated["runtime_version"] == "V1_8"
    assert activated["deployment_audit_valid"] is True
    assert activated["manifest_sha256"] == audit.V1_8_MANIFEST_SHA256
    assert activated["runner"] == "run_priority_forward_codex_automation_v1_8.py"

    assert audit.pcf_runtime_contract_for_date(
        "2026-08-27", "V1_7"
    )["runtime_version"] == "V1_7"
    assert audit.pcf_runtime_contract_for_date(
        "2026-08-26", "V1_8"
    )["runtime_version"] == "V1_7"
    assert audit.pcf_runtime_contract_for_date(
        "2026-08-27", "V1_8"
    )["runtime_version"] == "V1_8"


def test_runtime_test_boundary_rejects_test_import_path_and_untracked_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scripts = tmp_path / "scripts"
    tests = tmp_path / "tests"
    config = tmp_path / "config"
    scripts.mkdir()
    tests.mkdir()
    config.mkdir()
    (scripts / "runner.py").write_text("import pytest\n", encoding="utf-8")
    (tests / "test_helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    manifest = config / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {"path": "scripts/runner.py"},
                    {"path": "tests/test_helper.py"},
                ]
            }
        ),
        encoding="utf-8",
    )
    supervisor = config / "supervisor.yaml"
    supervisor.write_text(
        "tasks:\n  - launcher: scripts/untracked.ps1\n", encoding="utf-8"
    )
    monkeypatch.setattr(audit, "ROOT", tmp_path)

    result = audit.verify_runtime_test_boundary(manifest, supervisor)

    assert result["status"] == "FAILED"
    assert result["tracked_test_file_count"] == 1
    assert result["runtime_test_import_count"] == 1
    assert result["config_launchers_all_tracked"] is False
    assert result["missing_config_launchers"] == ["scripts/untracked.ps1"]


def test_task_exact_requires_pinned_runner_hash_and_no_backfill_settings() -> None:
    snapshot = {
        "query_status": "PASS",
        "state": "Ready",
        "arguments": (
            "runner_v1_7.py --expected-manifest-sha256 "
            + audit.MAIN_MANIFEST_SHA256
        ),
        "restart_count": 0,
        "start_when_available": False,
        "multiple_instances": "IgnoreNew",
    }

    assert audit.task_exact(
        snapshot,
        runner="runner_v1_7.py",
        manifest_sha256=audit.MAIN_MANIFEST_SHA256,
    )
    changed = dict(snapshot, restart_count=1)
    assert not audit.task_exact(
        changed,
        runner="runner_v1_7.py",
        manifest_sha256=audit.MAIN_MANIFEST_SHA256,
    )


def test_receipts_are_selected_by_payload_date_not_filename(tmp_path: Path) -> None:
    (tmp_path / "unrelated-name.json").write_text(
        json.dumps(
            {"started_at": "2026-08-26T09:25:01+08:00", "phase": "morning"}
        ),
        encoding="utf-8",
    )
    (tmp_path / "looks-current-20260826.json").write_text(
        json.dumps({"started_at": "2026-08-25T09:25:01+08:00"}), encoding="utf-8"
    )

    receipts = audit.receipts_for_local_date(tmp_path, "2026-08-26")

    assert len(receipts) == 1
    assert receipts[0]["path"].endswith("unrelated-name.json")
    assert len(audit.receipts_for_local_date(tmp_path, "2026-08-26", phase="morning")) == 1
    assert audit.receipts_for_local_date(tmp_path, "2026-08-26", phase="close") == []


def test_pcf_atomic_run_evidence_requires_one_consistent_success_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    disabled = {
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }
    task_receipt = {
        "schema_version": "1.2.0",
        "receipt_id": "task-1",
        "immutable_receipt": True,
        "task_name": audit.PCF_TASK_NAME,
        "run_status": "SUCCESS",
        "collection_status": "COMPLETE_QUALITY_DAY",
        "started_at": "2026-08-26T09:25:01+08:00",
        "ended_at": "2026-08-26T15:01:00+08:00",
        "task_exit_code": 0,
        "latest_observed_trade_date": "2026-08-26",
        "full_coverage_days": 3,
        **disabled,
    }
    task_receipt_path = tmp_path / "receipts" / "pcf-task.json"
    claim_path = tmp_path / "claims" / "20260826_PRIMARY_MARKET_PCF_IOPV.json"
    task_receipt_path.parent.mkdir(parents=True)
    claim_path.parent.mkdir(parents=True)
    task_receipt["receipt_file"] = task_receipt_path.relative_to(tmp_path).as_posix()
    codex_receipt = {
        "schema_version": "1.1.0",
        "receipt_id": "codex-1",
        "immutable_receipt": True,
        "dry_run": False,
        "phase": "morning",
        "started_at": "2026-08-26T09:20:00+08:00",
        "atomic_same_day_claim_enabled": True,
        "receipt_written": True,
        "log_written": True,
        "research_only": True,
        "task_results": [
            {
                "task_id": "PRIMARY_MARKET_PCF_IOPV",
                "decision": "RUN_NOW",
                "launcher": (
                    "scripts/run_510300_primary_market_collection_task_v1_2.ps1"
                ),
                "claim_file": claim_path.relative_to(tmp_path).as_posix(),
                "claim_acquired": True,
                "exit_code": 0,
                "post_evidence": [task_receipt["receipt_file"]],
            }
        ],
        **disabled,
    }
    claim = {
        "schema_version": "1.0.0",
        "claim_status": "ATOMIC_ATTEMPT_CLAIMED",
        "immutable_receipt": True,
        "task_id": "PRIMARY_MARKET_PCF_IOPV",
        "target_date": "2026-08-26",
        "phase": "morning",
        "claimed_at": "2026-08-26T09:25:00+08:00",
        "research_only": True,
        **disabled,
    }
    task_receipt_path.write_text(json.dumps(task_receipt), encoding="utf-8")
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    readiness = {
        "generated_at": "2026-08-26T15:00:30+08:00",
        "last_observed_trade_date": "2026-08-26",
        "full_coverage_days": 3,
        "daily_quality": [
            {"trade_date": "2026-08-26", "complete_quality_day": True}
        ],
    }

    passed = audit.verify_pcf_atomic_run_evidence(
        [task_receipt],
        [codex_receipt],
        [claim],
        "2026-08-26",
        readiness=readiness,
    )
    patched_task_receipt = {
        **task_receipt,
        "schema_version": "1.2.1",
        "started_at": "2026-08-27T09:25:01+08:00",
        "ended_at": "2026-08-27T15:01:00+08:00",
        "latest_observed_trade_date": "2026-08-27",
        "runtime_patch_id": "PCF_IOPV_PYTHON_MODULE_ENTRYPOINT_V1_2_1",
        "python_module_entrypoint": True,
        "runner_file": "scripts/run_510300_primary_market_forward_v1_2_1.ps1",
    }
    patched_codex_receipt = {
        **codex_receipt,
        "started_at": "2026-08-27T09:20:00+08:00",
        "task_results": [
            {
                **codex_receipt["task_results"][0],
                "launcher": (
                    "scripts/run_510300_primary_market_collection_task_v1_2_1.ps1"
                ),
            }
        ],
    }
    patched_claim = {
        **claim,
        "target_date": "2026-08-27",
        "claimed_at": "2026-08-27T09:25:00+08:00",
    }
    task_receipt_path.write_text(json.dumps(patched_task_receipt), encoding="utf-8")
    claim_path.write_text(json.dumps(patched_claim), encoding="utf-8")
    patched_passed = audit.verify_pcf_atomic_run_evidence(
        [patched_task_receipt],
        [patched_codex_receipt],
        [patched_claim],
        "2026-08-27",
        readiness={
            **readiness,
            "generated_at": "2026-08-27T15:00:30+08:00",
            "last_observed_trade_date": "2026-08-27",
            "daily_quality": [
                {"trade_date": "2026-08-27", "complete_quality_day": True}
            ],
        },
        task_schema_version="1.2.1",
        task_launcher=(
            "scripts/run_510300_primary_market_collection_task_v1_2_1.ps1"
        ),
        forward_runner="scripts/run_510300_primary_market_forward_v1_2_1.ps1",
        runtime_patch_id="PCF_IOPV_PYTHON_MODULE_ENTRYPOINT_V1_2_1",
    )
    task_receipt_path.write_text(json.dumps(task_receipt), encoding="utf-8")
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    failed_task_receipt = {
        **task_receipt,
        "run_status": "FAILED",
        "collection_status": "EXTERNAL_FREE_SOURCE_FAILED",
        "task_exit_code": 3,
        "latest_observed_trade_date": None,
    }
    task_receipt_path.write_text(json.dumps(failed_task_receipt), encoding="utf-8")
    failed_run = audit.verify_pcf_atomic_run_evidence(
        [failed_task_receipt],
        [
            {
                **codex_receipt,
                "task_results": [
                    {**codex_receipt["task_results"][0], "exit_code": 3}
                ],
            }
        ],
        [claim],
        "2026-08-26",
        readiness=readiness,
    )
    mismatched_external_task_receipt = {
        **failed_task_receipt,
        "task_exit_code": 1,
    }
    task_receipt_path.write_text(
        json.dumps(mismatched_external_task_receipt), encoding="utf-8"
    )
    mismatched_external_failure = audit.verify_pcf_atomic_run_evidence(
        [mismatched_external_task_receipt],
        [
            {
                **codex_receipt,
                "task_results": [
                    {**codex_receipt["task_results"][0], "exit_code": 1}
                ],
            }
        ],
        [claim],
        "2026-08-26",
        readiness=readiness,
    )
    task_receipt_path.write_text(json.dumps(task_receipt), encoding="utf-8")
    duplicate = audit.verify_pcf_atomic_run_evidence(
        [task_receipt, task_receipt],
        [codex_receipt],
        [claim],
        "2026-08-26",
        readiness=readiness,
    )
    duplicate_launcher = audit.verify_pcf_atomic_run_evidence(
        [task_receipt],
        [codex_receipt, codex_receipt],
        [claim],
        "2026-08-26",
        readiness=readiness,
    )
    late_noop_receipt = {
        **codex_receipt,
        "receipt_id": "codex-late-noop",
        "started_at": "2026-08-26T09:36:00+08:00",
        "task_results": [
            {
                "task_id": "PRIMARY_MARKET_PCF_IOPV",
                "decision": "MISSED_START_WINDOW",
                "launcher": (
                    "scripts/run_510300_primary_market_collection_task_v1_2.ps1"
                ),
                "claim_file": None,
                "claim_acquired": None,
                "exit_code": None,
                "post_evidence": [],
            }
        ],
    }
    late_noop = audit.verify_pcf_atomic_run_evidence(
        [task_receipt],
        [codex_receipt, late_noop_receipt],
        [claim],
        "2026-08-26",
        readiness=readiness,
    )
    malformed_late_noop = audit.verify_pcf_atomic_run_evidence(
        [task_receipt],
        [
            codex_receipt,
            {
                **late_noop_receipt,
                "task_results": [
                    {**late_noop_receipt["task_results"][0], "exit_code": 0}
                ],
            },
        ],
        [claim],
        "2026-08-26",
        readiness=readiness,
    )

    assert passed["status"] == "PASS"
    assert passed["contract_valid"] is True
    assert passed["successful"] is True
    assert patched_passed["status"] == "PASS"
    assert patched_passed["runtime_patch_id"] == (
        "PCF_IOPV_PYTHON_MODULE_ENTRYPOINT_V1_2_1"
    )
    assert failed_run["status"] == "FAILED_RUN"
    assert failed_run["contract_valid"] is True
    assert failed_run["successful"] is False
    assert mismatched_external_failure["status"] == "FAILED_CONTRACT"
    assert "EXTERNAL_FAILURE_EXIT_CODE_MISMATCH" in (
        mismatched_external_failure["task_failures"][0]["reasons"]
    )
    assert duplicate["status"] == "FAILED_CONTRACT"
    assert duplicate["contract_valid"] is False
    assert duplicate_launcher["status"] == "FAILED_CONTRACT"
    assert duplicate_launcher["actual_launcher_result_count"] == 2
    assert late_noop["status"] == "PASS"
    assert late_noop["contract_valid"] is True
    assert late_noop["codex_receipt_count"] == 2
    assert late_noop["actual_launcher_result_count"] == 1
    assert malformed_late_noop["status"] == "FAILED_CONTRACT"
    assert "PCF_TASK_NOOP_HAS_EXIT_CODE" in (
        malformed_late_noop["codex_failures"][0]["reasons"]
    )


def test_pcf_source_receipts_verify_all_sources_and_raw_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_root = (
        tmp_path
        / "reports"
        / "data_quality"
        / "free_source_acquisitions"
        / "primary_market_v1_2"
    )
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "PCF_SOURCE_RECEIPT_DIRECTORY", receipt_root)
    raw_paths: list[Path] = []
    for source_id in sorted(audit.PCF_REQUIRED_SOURCE_IDS):
        raw_bytes = source_id.encode("utf-8")
        raw_hash = hashlib.sha256(raw_bytes).hexdigest()
        raw_path = tmp_path / "data" / "raw" / f"{raw_hash}.json"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(raw_bytes)
        raw_paths.append(raw_path)
        receipt_directory = receipt_root / source_id / "20260826"
        receipt_directory.mkdir(parents=True, exist_ok=True)
        receipt = {
            "source_id": source_id,
            "source_url_or_endpoint": "https://example.invalid/source",
            "acquired_at": "2026-08-26T10:00:00+08:00",
            "market_date": "2026-08-26",
            "raw_response_hash": raw_hash,
            "raw_response_path": raw_path.relative_to(tmp_path).as_posix(),
            "parser_version": "Parser@1",
            "schema_version": "FREE_SOURCE_ACQUISITION_RECEIPT_V1",
            "quality_status": "PASS_RAW_CAPTURED_AND_PARSED",
            "fallback_source": "",
            "failure_reason": None,
            "raw_response_bytes": len(raw_bytes),
            "access_cost_cny": 0,
            "paid_data_used": False,
            "immutable_receipt": True,
        }
        receipt_path = receipt_directory / f"{source_id}.json"
        receipt_path.write_text(
            json.dumps(receipt), encoding="utf-8"
        )

    passed = audit.verify_pcf_source_receipts("2026-08-26")
    unadmitted_directory = receipt_root / "UNADMITTED_SOURCE" / "20260826"
    unadmitted_directory.mkdir(parents=True)
    unadmitted_path = unadmitted_directory / "unexpected.json"
    unadmitted_path.write_text(
        json.dumps(
            {
                **json.loads(receipt_path.read_text(encoding="utf-8")),
                "source_id": "UNADMITTED_SOURCE",
            }
        ),
        encoding="utf-8",
    )
    unadmitted = audit.verify_pcf_source_receipts("2026-08-26")
    unadmitted_path.unlink()
    raw_paths[0].write_bytes(b"changed")
    failed = audit.verify_pcf_source_receipts("2026-08-26")

    assert passed["status"] == "PASS"
    assert passed["contract_valid"] is True
    assert passed["all_required_sources_succeeded"] is True
    assert passed["receipt_count"] == 3
    assert passed["verified_raw_response_count"] == 3
    assert unadmitted["status"] == "FAILED_CONTRACT"
    assert unadmitted["invalid_receipt_count"] == 1
    assert failed["status"] == "FAILED_CONTRACT"
    assert failed["all_required_sources_succeeded"] is False
    assert failed["invalid_receipt_count"] == 1


def test_pcf_operational_acceptance_counts_missing_and_failed_opportunities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_directory = tmp_path / "receipts"
    receipt_directory.mkdir()
    calendar = tmp_path / "calendar.csv"
    trade_dates = [f"2026-09-{day:02d}" for day in range(1, 21)]
    calendar.write_text(
        "trade_date\n" + "\n".join(trade_dates) + "\n", encoding="utf-8"
    )
    disabled = {
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }
    for index, trade_date in enumerate(trade_dates):
        payload = {
            "schema_version": "1.2.1",
            "receipt_id": f"receipt-{index}",
            "immutable_receipt": True,
            "task_name": audit.PCF_TASK_NAME,
            "runtime_patch_id": "PCF_IOPV_PYTHON_MODULE_ENTRYPOINT_V1_2_1",
            "python_module_entrypoint": True,
            "runner_file": (
                "scripts/run_510300_primary_market_forward_v1_2_1.ps1"
            ),
            "run_status": "SUCCESS",
            "collection_status": "COMPLETE_QUALITY_DAY",
            "started_at": f"{trade_date}T09:25:00+08:00",
            "task_exit_code": 0,
            **disabled,
        }
        (receipt_directory / f"{index:02d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "TRADE_CALENDAR", calendar)
    monkeypatch.setattr(audit, "PCF_TASK_RECEIPT_DIRECTORY", receipt_directory)
    monkeypatch.setattr(
        audit,
        "PCF_TASK_RECEIPT_DIRECTORY_V1_2_1",
        tmp_path / "receipts-v1-2-1",
    )
    as_of = datetime(2026, 9, 20, 16, 0, tzinfo=ZoneInfo("Asia/Shanghai"))

    passed = audit.verify_pcf_operational_acceptance(as_of)
    for index in (0, 1):
        path = receipt_directory / f"{index:02d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(
            {
                "run_status": "FAILED",
                "collection_status": "PROGRAM_FAILED",
                "task_exit_code": 1,
            }
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
    failed_internal = audit.verify_pcf_operational_acceptance(as_of)
    for index in (0, 1):
        path = receipt_directory / f"{index:02d}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(
            {
                "run_status": "SUCCESS",
                "collection_status": "COMPLETE_QUALITY_DAY",
                "task_exit_code": 0,
            }
        )
        path.write_text(json.dumps(payload), encoding="utf-8")
    (receipt_directory / "18.json").unlink()
    (receipt_directory / "19.json").unlink()
    failed_missing = audit.verify_pcf_operational_acceptance(as_of)

    assert passed["status"] == "PASS"
    assert passed["opportunity_count"] == 20
    assert passed["internal_program_success_rate"] == 1.0
    assert passed["complete_quality_day_rate"] == 1.0
    assert failed_internal["status"] == "FAILED_INTERNAL_PROGRAM_GATE"
    assert failed_internal["internal_program_success_rate"] == 0.9
    assert failed_missing["status"] == "FAILED_INTERNAL_PROGRAM_GATE"
    assert failed_missing["missing_receipt_dates"] == ["2026-09-19", "2026-09-20"]


def test_t_only_daily_evidence_binds_receipt_latest_status_and_public_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    public_status = tmp_path / "reports" / "t_only_status.json"
    public_status.parent.mkdir(parents=True)
    public_status.write_text(
        json.dumps({"current_view": "NO_VIEW_UNTIL_FORWARD_MATURITY"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "T_ONLY_STATUS", public_status)
    run_id = "20260826T163000000000_1"
    receipt_path = tmp_path / "receipts" / f"{run_id}.json"
    receipt_path.parent.mkdir()
    payload = {
        "schema_version": "1.2.0",
        "project_id": "510300_T_ONLY_FORWARD_V1_1_DAILY_MATURITY_ONLY",
        "run_id": run_id,
        "started_at": "2026-08-26T16:30:00+08:00",
        "finished_at": "2026-08-26T16:31:00+08:00",
        "overall_status": "SUCCESS_MATURITY_ONLY_COLLECTING",
        "exit_code": 0,
        "manifest_verification": {
            "status": "PASS_V1_2_FROZEN_ENTRYPOINT_VERIFIED",
            "manifest_sha256": audit.T_ONLY_MANIFEST_SHA256,
        },
        "log_directory": "output/t_only",
        "stages": [],
        "output_audit": {
            "status": "PASS",
            "overall_complete": True,
            "current_view": "NO_VIEW_UNTIL_FORWARD_MATURITY",
        },
        "failure": None,
        "public_status_sha256": audit.sha256_file(public_status),
    }
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    latest_status = {
        **payload,
        "receipt": {
            "path": receipt_path.relative_to(tmp_path).as_posix(),
            "sha256": audit.sha256_file(receipt_path),
            "immutable_create_mode": "CREATE_NEW",
        },
    }

    passed = audit.verify_t_only_daily_evidence(
        [receipt_path], latest_status, "2026-08-26"
    )
    failed_payload = {
        **payload,
        "overall_status": "NO_VIEW_OPERATIONAL_FAILURE",
        "exit_code": 1,
        "output_audit": None,
        "failure": {"class": "DATA_GATE_FAILED", "stage": "refresh"},
    }
    receipt_path.write_text(json.dumps(failed_payload), encoding="utf-8")
    failed_latest = {
        **failed_payload,
        "receipt": {
            "path": receipt_path.relative_to(tmp_path).as_posix(),
            "sha256": audit.sha256_file(receipt_path),
            "immutable_create_mode": "CREATE_NEW",
        },
    }
    failed_run = audit.verify_t_only_daily_evidence(
        [receipt_path], failed_latest, "2026-08-26"
    )

    assert passed["status"] == "PASS"
    assert passed["contract_valid"] is True
    assert passed["successful"] is True
    assert passed["latest_status_matches_receipt"] is True
    assert failed_run["status"] == "FAILED_RUN"
    assert failed_run["contract_valid"] is True
    assert failed_run["successful"] is False


def test_close_acceptance_requires_atomic_same_day_receipt_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    disabled = {
        "position_mapping_enabled": False,
        "order_generation_enabled": False,
        "broker_connection_enabled": False,
        "live_trading_enabled": False,
    }
    industry_path = tmp_path / "receipts" / "industry.json"
    status_path = tmp_path / "receipts" / "status.json"
    codex_path = tmp_path / "receipts" / "codex.json"
    snapshot_path = tmp_path / "reports" / "audit" / "snapshot.json"
    central_path = tmp_path / "reports" / "audit" / "central.json"
    claim_directory = (
        tmp_path / "reports" / "audit" / "priority_forward_task_claims_v1_7"
    )
    log_path = tmp_path / "output" / "priority_forward" / "close.log"
    industry_path.parent.mkdir(parents=True)
    snapshot_path.parent.mkdir(parents=True)
    claim_directory.mkdir(parents=True)
    log_path.parent.mkdir(parents=True)
    log_path.write_text("收盘编排测试日志\n", encoding="utf-8")

    industry_relative = industry_path.relative_to(tmp_path).as_posix()
    central = {
        "generated_at": "2026-08-26T19:06:00+08:00",
        "overall_research_status": "NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET",
        "decision": "PAUSE_AND_FIX_FOUNDATION",
        "active_research_streams": [
            {
                "id": "INDUSTRY_EXPECTATION_GAP",
                "latest_task": {"path": industry_relative},
            }
        ],
    }
    central_bytes = json.dumps(central, sort_keys=True).encode("utf-8")
    snapshot_path.write_bytes(central_bytes)
    central_path.write_bytes(central_bytes)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "CENTRAL_STATUS", central_path)

    industry = {
        "schema_version": "1.2.0",
        "immutable_receipt": True,
        "task_name": "Codex-Industry-Expectation-Gap-Forward-Operations-V1.2",
        "started_at": "2026-08-26T19:01:00+08:00",
        "run_status": "SUCCESS",
        "collection_status": "COMPLETED_ZERO_PAID_INPUT_VERIFIED",
        "task_exit_code": 0,
        "paid_provider_call_enabled": False,
        "shared_latest_consumed_by_evaluation": False,
        **disabled,
    }
    status = {
        "schema_version": "1.6.0",
        "immutable_receipt": True,
        "task_name": "Codex-Priority-Forward-Authoritative-Status-V1.6",
        "started_at": "2026-08-26T19:05:00+08:00",
        "ended_at": "2026-08-26T19:07:00+08:00",
        "run_status": "SUCCESS",
        "task_exit_code": 0,
        "overall_research_status": "NO_VERIFIED_TRADABLE_ALPHA_OR_STRONG_BETA_YET",
        "decision": "PAUSE_AND_FIX_FOUNDATION",
        "report_snapshot_file": snapshot_path.relative_to(tmp_path).as_posix(),
        **disabled,
    }
    claim_paths = {
        task_id: claim_directory / f"20260826_{task_id}.json"
        for task_id in ("INDUSTRY_EXPECTATION_GAP", "PRIORITY_FORWARD_STATUS")
    }
    codex = {
        "schema_version": "1.1.0",
        "receipt_id": "close-1",
        "immutable_receipt": True,
        "dry_run": False,
        "phase": "close",
        "status": "SUCCESS_OR_ALREADY_ATTEMPTED",
        "started_at": "2026-08-26T19:00:00+08:00",
        "ended_at": "2026-08-26T19:07:30+08:00",
        "exit_code": 0,
        "atomic_same_day_claim_enabled": True,
        "receipt_written": True,
        "log_written": True,
        "research_only": True,
        "receipt_file": codex_path.relative_to(tmp_path).as_posix(),
        "log_file": log_path.relative_to(tmp_path).as_posix(),
        "task_results": [
            {
                "task_id": "INDUSTRY_EXPECTATION_GAP",
                "decision": "RUN_NOW",
                "launcher": (
                    "scripts/run_industry_expectation_gap_forward_operations_task_v1_2.ps1"
                ),
                "claim_file": claim_paths[
                    "INDUSTRY_EXPECTATION_GAP"
                ].relative_to(tmp_path).as_posix(),
                "claim_acquired": True,
                "exit_code": 0,
                "post_evidence": [industry_relative],
            },
            {
                "task_id": "PRIORITY_FORWARD_STATUS",
                "decision": "RUN_NOW",
                "launcher": (
                    "scripts/run_priority_forward_authoritative_status_task_v1_6.ps1"
                ),
                "claim_file": claim_paths[
                    "PRIORITY_FORWARD_STATUS"
                ].relative_to(tmp_path).as_posix(),
                "claim_acquired": True,
                "exit_code": 0,
                "post_evidence": [status_path.relative_to(tmp_path).as_posix()],
            },
        ],
        **disabled,
    }
    industry_path.write_text(json.dumps(industry), encoding="utf-8")
    status_path.write_text(json.dumps(status), encoding="utf-8")
    codex_path.write_text(json.dumps(codex), encoding="utf-8")
    claims = [
        {
            "schema_version": "1.0.0",
            "claim_status": "ATOMIC_ATTEMPT_CLAIMED",
            "immutable_receipt": True,
            "task_id": task_id,
            "target_date": "2026-08-26",
            "phase": "close",
            "claimed_at": "2026-08-26T19:00:30+08:00",
            "research_only": True,
            **disabled,
        }
        for task_id in ("INDUSTRY_EXPECTATION_GAP", "PRIORITY_FORWARD_STATUS")
    ]
    for claim in claims:
        claim_paths[claim["task_id"]].write_text(json.dumps(claim), encoding="utf-8")

    passed = audit.verify_close_acceptance_evidence(
        [industry_path], [status_path], [codex_path], claims, central, "2026-08-26"
    )
    external_industry = {
        **industry,
        "run_status": "FAILED",
        "collection_status": "EXTERNAL_FREE_SOURCE_FAILED",
        "task_exit_code": 3,
    }
    external_codex = {
        **codex,
        "status": "FAILED",
        "exit_code": 1,
        "task_results": [
            {**codex["task_results"][0], "exit_code": 3},
            codex["task_results"][1],
        ],
    }
    industry_path.write_text(json.dumps(external_industry), encoding="utf-8")
    codex_path.write_text(json.dumps(external_codex), encoding="utf-8")
    external_failure_accepted = audit.verify_close_acceptance_evidence(
        [industry_path], [status_path], [codex_path], claims, central, "2026-08-26"
    )
    industry_path.write_text(json.dumps(industry), encoding="utf-8")
    status_path.write_text(
        json.dumps({**status, "run_status": "FAILED", "task_exit_code": 1}),
        encoding="utf-8",
    )
    failed_codex = {
        **codex,
        "status": "FAILED",
        "exit_code": 1,
        "task_results": [
            codex["task_results"][0],
            {**codex["task_results"][1], "exit_code": 1},
        ],
    }
    codex_path.write_text(json.dumps(failed_codex), encoding="utf-8")
    failed_run = audit.verify_close_acceptance_evidence(
        [industry_path], [status_path], [codex_path], claims, central, "2026-08-26"
    )
    duplicate_codex = audit.verify_close_acceptance_evidence(
        [industry_path],
        [status_path],
        [codex_path, codex_path],
        claims,
        central,
        "2026-08-26",
    )

    late_central = {
        **central,
        "generated_at": "2026-08-26T22:41:26.700000+08:00",
    }
    late_central_bytes = json.dumps(late_central, sort_keys=True).encode("utf-8")
    snapshot_path.write_bytes(late_central_bytes)
    central_path.write_bytes(late_central_bytes)
    late_industry = {
        **industry,
        "started_at": "2026-08-26T22:41:12+08:00",
        "run_status": "FAILED",
        "collection_status": "PROGRAM_FAILED",
        "task_exit_code": 1,
    }
    late_status = {
        **status,
        "started_at": "2026-08-26T22:41:26+08:00",
        "ended_at": "2026-08-26T22:41:27+08:00",
    }
    late_claims = [
        {**claim, "claimed_at": "2026-08-26T22:41:11+08:00"}
        for claim in claims
    ]
    late_codex = {
        **codex,
        "status": "FAILED",
        "started_at": "2026-08-26T22:41:11+08:00",
        "ended_at": "2026-08-26T22:41:27+08:00",
        "exit_code": 1,
        "task_results": [
            {**codex["task_results"][0], "exit_code": 1},
            codex["task_results"][1],
        ],
    }
    industry_path.write_text(json.dumps(late_industry), encoding="utf-8")
    status_path.write_text(json.dumps(late_status), encoding="utf-8")
    codex_path.write_text(json.dumps(late_codex), encoding="utf-8")
    for claim in late_claims:
        claim_paths[claim["task_id"]].write_text(json.dumps(claim), encoding="utf-8")
    late_failed_run = audit.verify_close_acceptance_evidence(
        [industry_path],
        [status_path],
        [codex_path],
        late_claims,
        late_central,
        "2026-08-26",
    )
    industry_path.write_text(
        json.dumps({**late_industry, "started_at": "2026-08-26T23:31:00+08:00"}),
        encoding="utf-8",
    )
    after_frozen_window = audit.verify_close_acceptance_evidence(
        [industry_path],
        [status_path],
        [codex_path],
        late_claims,
        late_central,
        "2026-08-26",
    )

    assert passed["status"] == "PASS"
    assert passed["successful"] is True
    assert passed["status_snapshot_valid"] is True
    assert passed["central_uses_current_industry_receipt"] is True
    assert external_failure_accepted["status"] == "PASS"
    assert external_failure_accepted["industry_acceptable"] is True
    assert failed_run["status"] == "FAILED_RUN"
    assert failed_run["contract_valid"] is True
    assert duplicate_codex["status"] == "FAILED_CONTRACT"
    assert late_failed_run["status"] == "FAILED_RUN"
    assert late_failed_run["contract_valid"] is True
    assert late_failed_run["task_windows"]["INDUSTRY_EXPECTATION_GAP"] == {
        "schedule": "17:05",
        "latest_start": "23:30",
    }
    assert after_frozen_window["status"] == "FAILED_CONTRACT"
    assert any(
        "OUTSIDE_CLOSE_WINDOW" in failure.get("reasons", [])
        for failure in after_frozen_window["structural_failures"]
    )
    assert duplicate_codex["contract_valid"] is False


def test_t_only_forbidden_key_walk_finds_nested_leak() -> None:
    payload = {"maturity": {"mature": False}, "nested": [{"total_return": 0.1}]}
    matches = audit.walk_keys(payload).intersection(audit.FORBIDDEN_T_ONLY_KEYS)
    assert matches == {"total_return"}


def test_requirement_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="未知完成状态"):
        audit.requirement("R", "要求", "MAYBE", {}, [])


def test_current_audit_has_no_contradiction_before_first_window() -> None:
    snapshot = audit.build_audit(
        datetime(2026, 8, 26, 4, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    json.dumps(snapshot, ensure_ascii=False)

    assert snapshot["overall_status"] == "IN_PROGRESS_PENDING_REQUIRED_EVIDENCE"
    assert snapshot["goal_completion_proven"] is False
    assert snapshot["contradiction_requirement_ids"] == []
    assert "R07" in snapshot["pending_requirement_ids"]
    assert "R12" in snapshot["pending_requirement_ids"]
    assert "R16" in snapshot["pending_requirement_ids"]
    manifest_requirement = next(
        item for item in snapshot["requirements"] if item["id"] == "R06"
    )
    assert manifest_requirement["status"] == "PASS"
    assert (
        manifest_requirement["current_state"]["test_boundary"]
        == "PASS_TESTS_EXCLUDED_FROM_RUNTIME_MODEL_BOUNDARY"
    )
    assert manifest_requirement["current_state"]["tracked_test_file_count"] == 0
    assert manifest_requirement["current_state"]["runtime_test_import_count"] == 0
    assert manifest_requirement["current_state"]["config_launchers_all_tracked"] is True
    schedule_requirement = next(
        item for item in snapshot["requirements"] if item["id"] == "R26"
    )
    assert schedule_requirement["status"] == "PASS"
    assert schedule_requirement["current_state"]["active_automation_ids"] == [
        "510300",
        "510300-pcf-iopv",
    ]
    assert schedule_requirement["current_state"]["enabled_windows_task_names"] == [
        audit.PCF_TASK_NAME,
        audit.T_ONLY_TASK_NAME,
    ]
    cb_requirement = next(
        item for item in snapshot["requirements"] if item["id"] == "R23"
    )
    assert cb_requirement["status"] == "PASS"
    assert cb_requirement["current_state"]["official_right_terms_complete_count"] == 854
    cb_lifecycle_requirement = next(
        item for item in snapshot["requirements"] if item["id"] == "R24"
    )
    assert cb_lifecycle_requirement["status"] == "TERMINAL_EXPECTED"
    assert cb_lifecycle_requirement["current_state"][
        "official_complete_lifecycle_count"
    ] == 845


def test_2026_08_26_program_failure_enters_r25_denominator_only_after_cutoff() -> None:
    before_cutoff = audit.build_audit(
        datetime(2026, 8, 26, 15, 9, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    after_cutoff = audit.build_audit(
        datetime(2026, 8, 26, 15, 10, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    before_by_id = {item["id"]: item for item in before_cutoff["requirements"]}
    after_by_id = {item["id"]: item for item in after_cutoff["requirements"]}

    assert before_by_id["R25"]["current_state"]["opportunity_count"] == 0
    assert after_by_id["R25"]["current_state"]["opportunity_count"] == 1
    assert after_by_id["R25"]["current_state"]["opportunity_dates"] == [
        "2026-08-26"
    ]
    assert after_by_id["R25"]["current_state"][
        "internal_program_success_count"
    ] == 0
    assert after_by_id["R25"]["current_state"]["complete_quality_day_count"] == 0
    assert after_by_id["R25"]["current_state"]["invalid_receipt_dates"] == []
    assert after_by_id["R07"]["status"] == "PENDING_NEXT_WINDOW"
    assert after_cutoff["contradiction_requirement_ids"] == []


def test_deployed_v1_8_schedule_does_not_relabel_2026_08_26_v1_7_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment_path = tmp_path / "deployment.json"
    schedule_path = tmp_path / "schedule.json"
    deployment_path.write_text(
        json.dumps(
            {
                "status": "PASS_V1_8_DEPLOYED_FOR_NEXT_ELIGIBLE_WINDOW",
                "effective_from": "2026-08-27T09:20:00+08:00",
                "same_day_pcf_retry_performed": False,
                "frozen_manifest": {
                    "path": "config/priority_forward_research_operations_v1_8_manifest.json",
                    "sha256": audit.V1_8_MANIFEST_SHA256,
                },
            }
        ),
        encoding="utf-8",
    )
    schedule_path.write_text(
        json.dumps(
            {
                "status": "PASS_UNAUTHORIZED_SCHEDULES_PAUSED_OR_DISABLED",
                "frozen_manifest": {"sha256": audit.V1_8_MANIFEST_SHA256},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(audit, "V1_8_DEPLOYMENT_AUDIT", deployment_path)
    monkeypatch.setattr(audit, "ACTIVE_SCHEDULE_SURFACE_AUDIT_V1_8", schedule_path)

    pcf_task = {
        "query_status": "PASS",
        "state": "Ready",
        "arguments": (
            "run_priority_forward_codex_automation_v1_8.py "
            f"--expected-manifest-sha256 {audit.V1_8_MANIFEST_SHA256}"
        ),
        "restart_count": 0,
        "start_when_available": False,
        "multiple_instances": "IgnoreNew",
        "logon_type": "Interactive",
    }
    t_only_task = {
        "query_status": "PASS",
        "state": "Ready",
        "arguments": (
            "run_t_only_forward_v1_daily_v1_2.py "
            f"--expected-manifest-sha256 {audit.T_ONLY_MANIFEST_SHA256}"
        ),
        "restart_count": 0,
        "start_when_available": False,
        "multiple_instances": "IgnoreNew",
        "logon_type": "Interactive",
    }

    def task_snapshot(task_name: str) -> dict:
        return pcf_task if task_name == audit.PCF_TASK_NAME else t_only_task

    monkeypatch.setattr(audit, "windows_task_snapshot", task_snapshot)
    monkeypatch.setattr(
        audit,
        "automation_snapshot",
        lambda specification: {
            "id": specification["id"],
            "status": "PASS",
            "checks": {
                "active": True,
                "runner_pinned": specification["required_runner"]
                == "run_priority_forward_codex_automation_v1_8.py",
                "manifest_sha256_pinned": specification[
                    "required_manifest_sha256"
                ]
                == audit.V1_8_MANIFEST_SHA256,
                "schedule_matches": True,
            },
        },
    )
    monkeypatch.setattr(
        audit,
        "workspace_windows_task_inventory",
        lambda: {
            "query_status": "PASS",
            "tasks": [
                {"task_name": audit.PCF_TASK_NAME, "state": "Ready"},
                {"task_name": audit.T_ONLY_TASK_NAME, "state": "Ready"},
                *[
                    {"task_name": name, "state": "Disabled"}
                    for name in sorted(audit.LEGACY_WINDOWS_TASK_NAMES)
                ],
            ],
        },
    )
    monkeypatch.setattr(
        audit,
        "workspace_automation_inventory",
        lambda: {
            "query_status": "PASS",
            "automations": [
                {"id": "510300", "status": "ACTIVE", "runners": ["run_priority_forward_codex_automation_v1_8.py"]},
                {"id": "510300-pcf-iopv", "status": "ACTIVE", "runners": ["run_priority_forward_codex_automation_v1_8.py"]},
                {"id": "510300-2", "status": "PAUSED", "runners": []},
                {"id": "v3-forward-1", "status": "PAUSED", "runners": []},
            ],
        },
    )

    snapshot = audit.build_audit(
        datetime(2026, 8, 26, 4, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    by_id = {item["id"]: item for item in snapshot["requirements"]}

    assert snapshot["contradiction_requirement_ids"] == []
    assert by_id["R06"]["status"] == "PASS"
    assert by_id["R06"]["current_state"]["active_runtime"]["runtime_version"] == "V1_8"
    assert by_id["R07"]["current_state"]["runtime_contract"]["runtime_version"] == "V1_7"
    assert by_id["R09"]["status"] == "PASS"
    assert by_id["R09"]["current_state"]["live_boundary"]["current_supervisor_path"] == (
        "config/priority_forward_supervisor_v1_4.yaml"
    )
    assert by_id["R18"]["status"] == "PASS"
    assert "V1_8" in by_id["R18"]["title"]
    assert by_id["R26"]["status"] == "PASS"
    assert "V1_7" in by_id["R22"]["title"]

    next_day_snapshot = audit.build_audit(
        datetime(2026, 8, 27, 4, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    )
    next_day_by_id = {
        item["id"]: item for item in next_day_snapshot["requirements"]
    }
    assert next_day_snapshot["contradiction_requirement_ids"] == []
    assert next_day_by_id["R07"]["status"] == "PENDING_WINDOW"
    assert next_day_by_id["R07"]["current_state"]["runtime_contract"][
        "runtime_version"
    ] == "V1_8"
    assert next_day_by_id["R25"]["current_state"]["opportunity_count"] == 1
    assert next_day_by_id["R25"]["current_state"][
        "internal_program_success_count"
    ] == 0
    assert "V1_8" in next_day_by_id["R22"]["title"]
