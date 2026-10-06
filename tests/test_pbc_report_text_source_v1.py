"""报告季度身份、目录排除及相邻文本的必要语义。"""
import pytest
import pandas as pd

from research.pbc_report_text_source_v1 import first_known_origins, parse_index, parse_page, similarity, split_sections


def test_report_index_keeps_unique_quarter_and_publication_day():
    html = '<table><tr><td><a href="a/index.html">2015年第一季度中国货币政策执行报告</a></td><td>2015-05-08</td></tr></table>'
    row = parse_index(html).iloc[0]
    assert row.quarter == "2015Q1" and row.index_publication_date == "2015-05-08"
    with pytest.raises(ValueError, match="冲突"):
        parse_index(html + html.replace('a/index.html', 'b/index.html'))


def test_sections_use_body_not_table_of_contents_or_summary():
    pages = ["第四部分宏观经济分析……第五部分货币政策趋势……二、下一阶段货币政策主要思路……",
             "第四部分 宏观经济分析\n实际经济描述。第五部分 货币政策趋势\n展望。二、下一阶段货币政策主要思路\n实际政策指引。"]
    result = split_sections(pages)
    assert result["economic"] == "第四部分宏观经济分析实际经济描述。"
    assert result["guidance"].endswith("实际政策指引。")
    assert split_sections(["只有内容摘要和宽松政策描述"])["guidance"] is None


def test_similarity_is_direction_free_and_missing_remains_unknown():
    assert similarity("维持合理充裕流动性。", "维持合理充裕流动性。") == pytest.approx(1.)
    assert similarity("abc123", "积极支持实体经济") is None
    assert similarity("改善发展增长", "稳健调整流动性") == 0.
    row = parse_page('2024-11-08 18:00:00 <a href="report.pdf">报告</a><a href="en.pdf">英文</a>', 'https://www.pbc.gov.cn/a/index.html')
    assert row["page_publication_at"] == "2024-11-08 18:00:00+08:00"
    assert row["pdf_urls"] == ['https://www.pbc.gov.cn/a/report.pdf']


def test_late_friday_publication_uses_next_real_session_and_missing_stays_unknown():
    dates = pd.to_datetime(["2024-11-08", "2024-11-11", "2024-11-12"])
    frame = pd.DataFrame({"publication_upper": ["2024-11-08T23:59:59+08:00", None,
                                                "2024-11-08T15:05:00+08:00"]})
    result = first_known_origins(frame, dates)
    assert result.first_known_origin.iloc[0] == dates[1]
    assert pd.isna(result.first_known_origin.iloc[1])
    assert result.first_known_origin.iloc[2] == dates[0]
