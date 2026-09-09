from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import audit_priority_forward_v1_8_deployment as deployment
from scripts import audit_zero_paid_research_plan_completion_v1 as completion


def test_validate_automation_requires_v1_8_runner_manifest_and_phase(
    tmp_path: Path,
) -> None:
    path = tmp_path / "automation.toml"
    path.write_text(
        "\n".join(
            [
                'id = "510300"',
                'kind = "heartbeat"',
                'name = "优先前瞻研究收盘后验收"',
                f'prompt = {json.dumps(deployment.CLOSE_AUTOMATION_PROMPT, ensure_ascii=False)}',
                'status = "ACTIVE"',
                'rrule = "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR;BYHOUR=19;BYMINUTE=0"',
                'target_thread_id = "019ffba4-af4f-7053-97b1-25a4f51957cf"',
            ]
        ),
        encoding="utf-8",
    )
    specification = {**deployment.AUTOMATION_SPECS[1], "path": path}

    passed = deployment.validate_automation(specification)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "run_priority_forward_codex_automation_v1_8.py",
            "run_priority_forward_codex_automation_v1_7.py",
        ),
        encoding="utf-8",
    )
    failed = deployment.validate_automation(specification)

    assert passed["status"] == "PASS"
    assert passed["failures"] == []
    assert failed["status"] == "FAILED"
    assert "AUTOMATION_PROMPT_STILL_REFERENCES_V1_7_RUNNER" in failed["failures"]


def test_write_deployment_evidence_writes_activation_record_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    schedule_path = tmp_path / "schedule.json"
    deployment_path = tmp_path / "deployment.json"
    monkeypatch.setattr(deployment, "SCHEDULE_AUDIT", schedule_path)
    monkeypatch.setattr(deployment, "DEPLOYMENT_AUDIT", deployment_path)
    monkeypatch.setattr(completion, "V1_8_DEPLOYMENT_AUDIT", deployment_path)
    validation = {
        "status": "READY_TO_WRITE",
        "generated_at": "2026-08-26T19:10:00+08:00",
        "manifest": {
            "content_sha256": "a" * 64,
            "tracked_file_count": 41,
            "failure_count": 0,
        },
        "close_acceptance": {"status": "PASS", "r22_status": "PASS"},
        "windows_task": {"status": "PASS"},
        "automations": [
            {"id": "510300-pcf-iopv", "status": "PASS"},
            {"id": "510300", "status": "PASS"},
        ],
        "schedule_surface": {
            "checks": {"pcf_windows_task_exact": True},
            "enabled_windows_task_names": [
                completion.PCF_TASK_NAME,
                completion.T_ONLY_TASK_NAME,
            ],
            "disabled_legacy_task_names": sorted(
                completion.LEGACY_WINDOWS_TASK_NAMES
            ),
            "windows_inventory": {"tasks": []},
            "automation_inventory": {"automations": []},
        },
    }

    result = deployment.write_deployment_evidence(validation)
    payload = json.loads(deployment_path.read_text(encoding="utf-8"))

    assert schedule_path.is_file()
    assert deployment_path.is_file()
    assert result["active_runtime"] == "V1_8"
    assert payload["status"] == "PASS_V1_8_DEPLOYED_FOR_NEXT_ELIGIBLE_WINDOW"
    assert payload["same_day_pcf_retry_performed"] is False
    assert payload["same_day_backfill_performed"] is False
    assert payload["research_run_triggered_by_deployment"] is False
    assert payload["historical_evidence_modified"] is False


def test_close_acceptance_allows_a_valid_failed_run_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    central_audit = tmp_path / "central.json"
    monkeypatch.setattr(deployment, "CENTRAL_AUDIT", central_audit)
    payload = {
        "generated_at": "2026-08-26T19:10:00+08:00",
        "contradiction_requirement_ids": [],
        "requirements": [
            {
                "id": "R22",
                "status": "PENDING_NEXT_WINDOW",
                "current_state": {
                    "acceptance_status": "FAILED_RUN",
                    "contract_valid": True,
                },
            }
        ],
    }
    central_audit.write_text(json.dumps(payload), encoding="utf-8")

    accepted = deployment.close_acceptance_snapshot()
    payload["requirements"][0]["current_state"]["contract_valid"] = False
    central_audit.write_text(json.dumps(payload), encoding="utf-8")
    rejected = deployment.close_acceptance_snapshot()

    assert accepted["status"] == "PASS"
    assert accepted["close_attempt_accepted"] is True
    assert rejected["status"] == "FAILED"
    assert rejected["close_attempt_accepted"] is False


def test_write_deployment_evidence_rejects_failed_validation(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="部署前置核验未通过"):
        deployment.write_deployment_evidence(
            {"status": "FAILED_VALIDATION", "generated_at": "2026-08-26T19:10:00+08:00"}
        )

    assert list(tmp_path.iterdir()) == []
