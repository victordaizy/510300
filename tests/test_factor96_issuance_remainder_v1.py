"""检验跨轮日期窗口拼接，不将旧预算结束改写为已完成。"""
import pytest

from research.factor96_issuance_remainder_v1 import reconcile


def windows(total=3):
    return {
        "p": {"window_id": "p", "start": "2025-01-01", "end": "2025-01-04", "declared_total": total,
              "received": 1, "complete": False, "child_windows": ["a", "b"], "status": "SPLIT"},
        "a": {"window_id": "a", "start": "2025-01-01", "end": "2025-01-02", "declared_total": 1,
              "received": 1, "complete": True, "status": "COMPLETE_SINGLE_RESPONSE"},
        "b": {"window_id": "b", "start": "2025-01-03", "end": "2025-01-04", "declared_total": None,
              "received": 0, "complete": False, "status": "NOT_REQUESTED"}}


def replacement(start="2025-01-03", end="2025-01-04"):
    return {"windows": [{"window_id": "b", "start": start, "end": end, "declared_total": 2,
                         "received": 2, "complete": True, "status": "COMPLETE_SINGLE_RESPONSE"}]}


def test_complete_children_close_parent_without_changing_old_state():
    old = windows()
    result, rows = reconcile("p", old, {"b": replacement()})
    assert result["complete"] and result["received"] == 3
    assert old["p"]["complete"] is False and old["b"]["received"] == 0
    assert len(rows) == 3


def test_missing_replacement_remains_incomplete():
    result, _ = reconcile("p", windows(), {})
    assert result["complete"] is False and result["received"] == 1


def test_old_parent_total_drift_is_not_silently_accepted():
    result, _ = reconcile("p", windows(total=4), {"b": replacement()})
    assert result["complete"] is False and result["status"] == "OLD_PARENT_NEW_CHILD_TOTAL_MISMATCH"


def test_replacement_cannot_expand_date_range():
    with pytest.raises(AssertionError):
        reconcile("p", windows(), {"b": replacement(start="2025-01-02")})


def test_date_gap_between_siblings_is_rejected():
    old = windows()
    old["a"]["end"] = "2025-01-01"
    with pytest.raises(AssertionError):
        reconcile("p", old, {"b": replacement()})


def test_date_overlap_between_siblings_is_rejected():
    old = windows()
    old["a"]["end"] = "2025-01-03"
    with pytest.raises(AssertionError):
        reconcile("p", old, {"b": replacement()})
