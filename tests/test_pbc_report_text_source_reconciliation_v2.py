"""检查原件格式与历史日历边界；不改变或验证任何收益参数。"""
import pandas as pd

from research.pbc_report_text_source_reconciliation_v2 import EXPECTED, bind_origins, cover_identity, split_sections


def test_chinese_cover_year_and_wrong_quarter_are_distinguished():
    assert cover_identity("中国货币政策执行报告 二○一一年 第四季度 2012年2月15日", "2011Q4")
    assert cover_identity("中国货币政策执行报告 二〇一五年 第一季度", "2015Q1")
    assert not cover_identity("中国货币政策执行报告 二〇一五年 第二季度", "2015Q1")
    assert not cover_identity("摘要2015年第一季度", "2014Q4")


def test_real_economic_heading_variants_retain_body_not_contents():
    for name in ("宏观经济形势", "宏观经济分析"):
        text = (f"第四部分{name}目录第五部分货币政策趋势下一阶段主要政策思路目录"
                f"第四部分{name}实际经济正文第五部分货币政策趋势展望二、下一阶段主要政策思路实际政策正文")
        result = split_sections(text)
        assert "实际经济正文" in result["economic"]
        assert "目录" not in result["economic"]
        assert "实际政策正文" in result["guidance"]


def test_before_calendar_reports_use_actual_latest_and_missing_does_not_fallback():
    rows = []
    for i, quarter in enumerate(EXPECTED):
        rows.append({"quarter": quarter, "index_publication_date": (pd.Timestamp("2012-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                     "status": "ORIGINAL_DOCUMENT_AND_TWO_SECTIONS", "previous_quarter": EXPECTED[i - 1] if i else None,
                     "economic_similarity": 0.8, "guidance_similarity": 0.9,
                     "historical_first_vintage": "NOT_CERTIFIED", "pdf_path": "test.pdf"})
    rows[3].update(status="NO_VIEW_SOURCE", economic_similarity=None, guidance_similarity=None)
    dates = pd.Series(pd.date_range("2012-01-04", periods=58))
    _, bound = bind_origins(pd.DataFrame(rows), dates)
    assert bound.iloc[0].pbc_quarter == EXPECTED[2]
    assert bound.iloc[1].pbc_quarter == EXPECTED[3]
    assert not bound.iloc[1].pbc_two_channel_known
    assert pd.isna(bound.iloc[1].pbc_guidance_change)
