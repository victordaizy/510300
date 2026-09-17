"""来源修复中会造成漏公告或误合并的边界检查。"""
import pytest

from scripts.repair_510300_unlock_announcement_source_v1_1 import merge_records, split_dates


def row(key, code="600000"):
    return {"announcementId": key, "secCode": code, "orgId": "o", "announcementTitle": "限售股上市流通公告", "announcementTime": 1, "adjunctUrl": key+".PDF", "announcement_date": "2024-02-01"}


def test_leap_month_partition_has_no_gap_or_overlap():
    assert split_dates("2024-02-01", "2024-02-29") == (("2024-02-01", "2024-02-15"), ("2024-02-16", "2024-02-29"))


def test_union_can_complete_same_identity_without_counting_duplicates():
    assert len(merge_records([[row("a"), row("b")], [row("b"), row("c")]], 3)) == 3
    assert len(merge_records([[row("a")], [row("a")]], 2)) == 1


def test_reused_id_cannot_hide_source_identity_change():
    with pytest.raises(ValueError, match="身份字段变化"):
        merge_records([[row("a")], [row("a", "000001")]], 2)


def test_population_growth_cannot_be_called_complete():
    with pytest.raises(ValueError, match="超过固定声明量"):
        merge_records([[row("a"), row("b")]], 1)
