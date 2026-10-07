"""覆盖实际公告中的倒填、计划完成、背景引用和未知编号风险。"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.factor96_repurchase_missing_originals_analysis_v1 import build, latest_event


def test_old_target_is_first_tianshan_plan_not_background_2025_plan():
    roots, changes, followups, events = build()
    row = next(r for r in changes if r["document_id"] == "1224735615")
    assert row["confirmed_target_roots"] == ["002532.SZ_ORIGINAL_1213972123"]
    assert "002532.SZ_ORIGINAL_1223094341" in row["context_only_roots"]
    assert next(r for r in followups if r["document_id"] == "1225411010")["target_roots"] == []


def test_actual_cancellation_is_not_backfilled_to_effective_date():
    events = build()[3]
    root = "002532.SZ_ORIGINAL_1213972123"
    earlier = latest_event(events, root, "2025-12-29T23:59:59+08:00")
    later = latest_event(events, root, "2025-12-31T23:59:59+08:00")
    assert not earlier["completion_confirmed_by_document"]
    assert later["status"] == "CANCELLATION_COMPLETED_REPORTED"
    assert later["root_affected_shares"] == 23148000


def test_scheduled_cancellation_does_not_auto_complete_after_date():
    events = build()[3]
    for root in ["603195.SH_ORIGINAL_1219840335", "688303.SH_ORIGINAL_1217605023", "603833.SH_ORIGINAL_1214939768"]:
        row = latest_event(events, root, "2026-09-22T23:59:59+08:00")
        assert not row["completion_confirmed_by_document"]
        assert row["remaining_inventory_shares"] is None


def test_used_inventory_cannot_be_seen_until_publication():
    events = build()[3]
    root = "600745.SH_ORIGINAL_1217433900"
    before = latest_event(events, root, "2026-09-10T23:59:59+08:00")
    after = latest_event(events, root, "2026-09-12T23:59:59+08:00")
    assert before["remaining_inventory_shares"] is None
    assert after["remaining_inventory_shares"] == 0
    assert after["status"] == "REPURCHASE_INVENTORY_FULLY_USED_REPORTED"


def test_retrospective_shareholder_approval_is_not_earlier_source():
    events = build()[3]
    value = latest_event(events, "002532.SZ_ORIGINAL_1213972123", "2025-11-11T23:59:59+08:00")
    assert value["document_ids"] == ["1224735615"]
    assert value["scope"] == "LATEST_REVIEWED_EVENT_ONLY_NOT_COMPLETE_CURRENT_PLAN_STATE"


def test_unresolved_printed_announcement_number_and_mixed_purpose_survive():
    roots = build()[0]
    daquan = next(r for r in roots if r["symbol"] == "688303.SH")
    oppein = next(r for r in roots if r["symbol"] == "603833.SH")
    assert daquan["announcement_number"] is None and daquan["announcement_number_raw"] == "2023-0XX"
    assert oppein["purpose"] == "MIXED_EMPLOYEE_INCENTIVE_AND_CONVERTIBLE_BOND"
    assert all(not r["trading_feature_admitted"] for r in roots)


def test_partial_inventory_allocation_keeps_two_plan_roots():
    followups = build()[2]
    gongniu = next(r for r in followups if r["document_id"] == "1224921839")
    assert gongniu["root_shares"] == {"603195.SH_ORIGINAL_1219840335": 73, "603195.SH_ORIGINAL_1223283990": 63817}
    assert sum(gongniu["root_shares"].values()) == 63890
