from __future__ import annotations

from scripts.audit_priority_forward_frozen_inputs_v1 import build_audit


def test_active_inputs_are_isolated_and_v3_drift_is_quarantined() -> None:
    audit = build_audit()
    assert audit["audit_status"] == (
        "PASS_FROZEN_INPUTS_CONTENT_ADDRESSED_AND_LATEST_ISOLATED"
    )
    assert all(audit["checks"].values())
    industry = audit["industry_expectation_gap"]
    assert industry["shared_latest_consumed_by_evaluation"] is False
    assert len(industry["records"]) == 4
    v3 = audit["v3_forward_2_quarantine"]
    assert v3["mismatch_count"] >= 1
    assert v3["daily_status"] == "FAILED"
    assert v3["latest_status_exists"] is False
    assert v3["active_task"] is False
    assert v3["retry_or_repair_performed"] is False
