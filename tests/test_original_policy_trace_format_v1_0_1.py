"""三个必要格式检查：排版、公开钟冲突与完整原反例集合。"""
import pandas as pd

from research.original_policy_trace_format_v1_0_1 import combined_keys, match_text, parse_saved


def test_visible_date_and_split_chinese_terms_do_not_use_event_day():
    raw = b'<html><div class="time">2018/08/31</div><div id="body">2018 \xe5\xb9\xb4 4 \xe6\x9c\x88 25 \xe6\x97\xa5</div></html>'
    doc = parse_saved(raw, "#body", ".time")
    assert doc["source_available_upper"] == pd.Timestamp("2018-08-31 23:59:59", tz="Asia/Shanghai")
    assert match_text("2018年4月25日") in match_text(doc["text"])


def test_conflicting_publication_metadata_remains_unknown():
    raw = b'<html><meta name="PubDate" content="2026-08-01 21:55:09"><div class="time">2024-10-18</div><div id="body">policy</div></html>'
    doc = parse_saved(raw, "#body", ".time")
    assert doc["metadata_visible_calendar_date_conflict"]
    assert pd.isna(doc["source_available_upper"])
    assert doc["visible_publication_upper"] == pd.Timestamp("2024-10-18 23:59:59", tz="Asia/Shanghai")


def test_original_seventeen_dates_and_two_negative_signals_all_retained():
    keys = combined_keys()
    assert len(keys) == 19
    assert pd.Timestamp("2017-04-05") in set(keys.date)
    assert pd.Timestamp("2026-05-11") in set(keys.date)
