"""检查文本缺口与公布钟；不测试历史收益。"""
import pandas as pd

from research.pbc_report_text_case_alignment_v1 import EXPECTED, bind_origins


def test_missing_released_report_stays_unknown_instead_of_older_text():
    rows = []
    for i, quarter in enumerate(EXPECTED):
        rows.append({"quarter": quarter, "index_publication_date": (pd.Timestamp("2012-01-01") + pd.Timedelta(days=i)).strftime("%Y-%m-%d"),
                     "status": "ORIGINAL_DOCUMENT_AND_TWO_SECTIONS", "previous_quarter": EXPECTED[i - 1] if i else None,
                     "economic_similarity": 0.8, "guidance_similarity": 0.9})
    rows[1].update(status="NO_VIEW_SOURCE", economic_similarity=None, guidance_similarity=None)
    dates = pd.Series(pd.date_range("2012-01-01", periods=61))
    _, bound = bind_origins(pd.DataFrame(rows), dates)
    current = bound.loc[bound.date.eq("2012-01-03")].iloc[0]
    assert current.pbc_quarter == EXPECTED[1]
    assert not current.pbc_two_channel_known
    assert pd.isna(current.pbc_economic_change)
    assert pd.isna(current.pbc_guidance_change)
    before = bound.loc[bound.date.eq("2012-01-02")].iloc[0]
    assert before.pbc_quarter == EXPECTED[0]
    assert bound.iloc[0].pbc_two_channel_known == False
