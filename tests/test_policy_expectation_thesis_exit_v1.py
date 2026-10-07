"""兑现边界、可知钟及实例隔离的必要测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research import policy_expectation_thesis_exit_v1 as study


def runtime_stub(data, dividends, parents, risks, policy, cost, start):
    return {"policy": policy, "runtime_rules": rules}


def row(lo, hi):
    return {"t1_surprise_min_bp": lo, "t1_surprise_max_bp": hi}


def test_one_sided_bounds_do_not_need_invented_endpoint():
    assert study.tenor_state(row(2.5, np.nan), "t1") == "TIGHTER"
    assert study.tenor_state(row(np.nan, -5), "t1") == "EASIER"
    assert study.tenor_state(row(np.nan, np.nan), "t1") == "UNIDENTIFIED"


def test_joint_direction_preserves_missing_and_mixed():
    assert study.joint_state("TIGHTER", "MATCH") == "DUAL_CREDIT_UNDERDELIVERY"
    assert study.joint_state("MATCH", "TIGHTER") == "DUAL_CREDIT_UNDERDELIVERY"
    assert study.joint_state("TIGHTER", "EASIER") == "MIXED_DIRECTION_NO_CANCELLATION"
    assert study.joint_state("TIGHTER", "UNIDENTIFIED") == "ABSTAIN_UNIDENTIFIED_TENOR"


def test_non_prior_modified_version_is_never_cancel_signal():
    frame = pd.DataFrame([{"month": "2023-06", "status": "ADMITTED_SAVED_HISTORICAL_SURVEY",
        "announcement_at": "2023-06-20T09:15:00+08:00", "survey_published_at": "2023-06-19T15:00:00+08:00",
        "survey_modified_at": "2023-06-21T00:00:00+08:00", "t1_surprise_min_bp": 1, "t1_surprise_max_bp": 2,
        "t5_surprise_min_bp": 1, "t5_surprise_max_bp": 2}])
    classified = study.classify_surveys(frame)
    assert classified.joint_credit_state.iloc[0] == "ABSTAIN_NON_PRIOR_VERSION"


def test_policy_after_close_only_affects_next_complete_close():
    data = pd.DataFrame({"date": pd.to_datetime(["2023-06-19", "2023-06-20", "2023-06-21"])})
    surveys = pd.DataFrame([{"month": "2023-06", "announcement_at": "2023-06-20T16:00:00+08:00", "joint_credit_state": "DUAL_CREDIT_UNDERDELIVERY"}])
    result = study.attach_events(data, surveys)
    assert result.credit_cancellation_event.tolist() == [False, False, True]


def test_event_is_not_carried_forward_or_backfilled():
    data = pd.DataFrame({"date": pd.to_datetime(["2023-06-19", "2023-06-20", "2023-06-21"])})
    surveys = pd.DataFrame([{"month": "2023-06", "announcement_at": "2023-06-20T09:15:00+08:00", "joint_credit_state": "DUAL_CREDIT_UNDERDELIVERY"}])
    result = study.attach_events(data, surveys)
    assert result.credit_cancellation_event.tolist() == [False, True, False]
    pd.testing.assert_frame_equal(study.attach_events(data.iloc[:1].copy(), surveys), result.iloc[:1])


def test_new_cancel_event_does_not_cancel_core_owner(monkeypatch):
    monkeypatch.setattr(study.parent.rules.support, "exit_decision", lambda *args: "原失效规则")
    adapter = study.SupportBridge()
    old = study.parent.rules.support.exit_decision
    # 使用真实规则缺少字段时会继续原路径；只有COMPLEMENT提前返回新失效。
    active = {"owner": "COMPLEMENT"}
    event_row = pd.Series({"credit_cancellation_event": True})
    assert adapter.exit_decision(active, event_row, 1, study.parent.rules.support.POLICIES[0]) == study.EXIT_REASON
    assert adapter.exit_decision({"owner": "CORE"}, event_row, 1, study.parent.rules.support.POLICIES[0]) == "原失效规则"
    assert study.parent.rules.support.exit_decision is old


def test_original_module_globals_are_never_replaced(monkeypatch):
    execution = study.parent.execution
    original_rules = execution.account.__globals__["rules"]
    # 新函数取自测试模块；声明其规则引用后，独立namespace必须替换且原引用保持。
    monkeypatch.setitem(runtime_stub.__globals__, "rules", original_rules)
    monkeypatch.setattr(execution, "account", runtime_stub)
    result = study.account(None, None, None, None, "BASE", "2020-01-01")
    assert result["policy"] == study.BASELINE
    assert isinstance(result["runtime_rules"], study.RulesBridge)
    assert runtime_stub.__globals__["rules"] is original_rules


def test_matched_and_overdelivery_are_distinct_from_missing():
    assert study.joint_state("MATCH", "MATCH") == "MATCHED_BOTH_TENORS"
    assert study.joint_state("EASIER", "MATCH") == "DUAL_CREDIT_OVERDELIVERY"
    assert study.tenor_state(row(-1, 1), "t1") == "UNIDENTIFIED"
