"""验证文档角色与历史版本时钟，避免同日合并和未来修訂回填。"""
import pytest

from research.factor96_supply_version_ledger_v1 import document_role, visible_schedule, SCHEDULE


def test_deferred_grant_is_original_disclosure():
    assert document_role("关于2021年激励计划暂缓授予部分解除限售股份上市流通公告", "证券代码002601本次解除限售", "002601.SZ") == "ISSUER_DEFERRED_GRANT_UNLOCK"


def test_issuer_correction_of_opinion_is_not_opinion():
    assert document_role("关于核查意见的更正公告", "证券代码300418", "300418.SZ") == "ISSUER_CORRECTION"


def test_supporting_opinion_not_new_event_even_deferred():
    assert document_role("关于暂缓授予解除限售的核查意见", "证券公司核查意见", "002601.SZ") == "SUPPORTING_OPINION"


def test_postponement_differs_from_grant_history():
    assert document_role("限售股延期上市流通公告", "证券代码600515", "600515.SH") == "ISSUER_SCHEDULE_EXTENSION"


def test_future_version_does_not_change_old_view():
    record = visible_schedule(SCHEDULE, "2020-01-24T15:05:00+08:00")
    assert record["document_id"] == "1206469829"
    assert record["scheduled_unlock_date"] == "2020-01-26"
    assert visible_schedule(SCHEDULE, "2019-07-25T15:05:00+08:00") is None


def test_open_ended_is_not_old_date_or_zero_supply():
    record = visible_schedule(SCHEDULE, "2021-01-11T15:05:00+08:00")
    assert record["scheduled_unlock_date"] is None
    assert record["state"] == "OPEN_ENDED_CONTINGENT_NOT_ZERO_SUPPLY"


def test_naive_decision_clock_rejected():
    with pytest.raises(ValueError, match="时区"):
        visible_schedule(SCHEDULE, "2021-01-11")
