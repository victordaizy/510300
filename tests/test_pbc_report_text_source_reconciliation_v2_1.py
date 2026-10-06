"""真实Parquet精度与公布钟精度不同，仍须保持同一15:05语义。"""
import pandas as pd

from research.pbc_report_text_source_reconciliation_v2_1 import EXPECTED, bind_origins, cover_identity, split_sections


def test_document_identity_and_economic_chapter_format():
    assert cover_identity("中国货币政策执行报告 二○一一年 第四季度 2012年2月15日", "2011Q4")
    assert not cover_identity("中国货币政策执行报告 二○一一年 第四季度", "2012Q1")
    text = "第四部分宏观经济形势目录第五部分货币政策趋势下一阶段主要政策思路目录第四部分宏观经济形势正文第五部分货币政策趋势展望下一阶段主要政策思路政策正文"
    assert split_sections(text)["economic"] == "第四部分宏观经济形势正文"


def test_real_nanosecond_calendar_and_microsecond_publication_keep_missing():
    rows = [{"quarter": q, "index_publication_date": (pd.Timestamp("2012-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
             "status": "ORIGINAL_DOCUMENT_AND_TWO_SECTIONS", "previous_quarter": EXPECTED[i-1] if i else None,
             "economic_similarity": .8, "guidance_similarity": .9, "historical_first_vintage": "NOT_CERTIFIED", "pdf_path": "test.pdf"}
            for i, q in enumerate(EXPECTED)]
    rows[3].update(status="NO_VIEW_SOURCE", economic_similarity=None, guidance_similarity=None)
    dates = pd.Series(pd.date_range("2012-01-04", periods=58).as_unit("ns"))
    _, bound = bind_origins(pd.DataFrame(rows), dates)
    assert bound.iloc[0].pbc_quarter == EXPECTED[2]
    assert bound.iloc[1].pbc_quarter == EXPECTED[3]
    assert not bound.iloc[1].pbc_two_channel_known
    assert pd.isna(bound.iloc[1].pbc_guidance_change)
    assert bound.date.dtype == dates.dtype
