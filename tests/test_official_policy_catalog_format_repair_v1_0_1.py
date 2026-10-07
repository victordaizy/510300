"""旧标题、字体拆日期和月份真实未知的必要格式测试。"""
import pandas as pd
import pytest

from research import official_policy_catalog_format_repair_v1_0_1 as module


def doc():
    return {"catalog_id": "测试", "catalog_year": 2017, "title": "2017年中国货币政策大事记",
        "listed_published_date": "2018-02-14", "url": module.inputs.BASE,
        "source_path": "测试.html", "source_sha256": "测试身份", "reference_start": "2017-01-01",
        "reference_end": "2017-12-31", "reference_role": "全年"}


def html(body):
    return f"<title>2017年中国货币政策大事记</title><div>2018-02-14 12:00:00</div><div id='zoom'>{body}</div>".encode()


def test_country_word_missing_old_titles_retained():
    raw = "<td><a istitle='true' href='/旧文.html'>2008年第二季度货币政策大事记</a><span>2008-08-16</span></td>".encode()
    rows = module.index_rows(raw, module.inputs.BASE)
    assert len(rows) == 1 and rows[0]["title"] == "2008年第二季度货币政策大事记"
    with pytest.raises(ValueError):
        module.index_rows(raw.replace("货币政策大事记".encode(), "其他统计资料".encode()), module.inputs.BASE)


def test_font_spans_rejoin_original_date():
    rows, unknown = module.parse_document(html("<p><span>7<span>月</span><span>1</span><span>日，例会召开。</span></span></p>"), doc())
    assert len(rows) == 1 and not unknown
    assert rows[0]["catalog_event_date"] == pd.Timestamp("2017-07-01")
    assert rows[0]["text"] == "7月1日,例会召开。"


def test_month_and_late_month_are_unknown_exact_day():
    rows, unknown = module.parse_document(html("<p>1月，临时流动性支持。</p><p>3月下旬，例会召开。</p><p>4月5日，公布细则。</p>"), doc())
    assert len(rows) == 3 and not unknown
    assert pd.isna(rows[0]["catalog_event_date"]) and pd.isna(rows[1]["catalog_event_date"])
    assert rows[0]["catalog_event_date_lower"] == pd.Timestamp("2017-01-01")
    assert rows[0]["catalog_event_date_upper"] == pd.Timestamp("2017-01-31")
    assert rows[1]["catalog_event_date_lower"] == pd.Timestamp("2017-03-21")
    assert rows[1]["catalog_event_date_upper"] == pd.Timestamp("2017-03-31")
    assert rows[2]["source_available_upper"] > pd.Timestamp("2017-04-05", tz=module.inputs.TZ)
    assert all(row["record_role"] == "RETROSPECTIVE_LEAD_ONLY_NOT_ORIGINAL_ANNOUNCEMENT" for row in rows)
