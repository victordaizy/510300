"""验证官方空单元格、零值、旧口径时轴及迁移网址的实际含义。"""
import json

from bs4 import BeautifulSoup
import pandas as pd

from research.nbs_monthly_collection import dump, expand_table, history, parse_release


def test_historical_omitted_cells_do_not_erase_zero_or_invent_values(tmp_path):
    node = {"_id": "table", "name": "历史行业表", "category": "工业"}
    indicators = [{"_id": key, "catalogid": "table", "i_showname": name,
                   "catalogue_path": "工业 / 历史行业表", "du_name": "%", "dp_name": "同比增减%"}
                  for key, name in [("a", "有公布值"), ("b", "没有公布值")]]
    dump(tmp_path / "metadata.json", {"root_id": "monthly", "leaves": [node], "indicators": indicators})

    class OfficialResponse:
        run = tmp_path

        def json(self, endpoint, params=None, payload=None):
            return [
                {"code": "202609MM", "values": []},
                {"code": "202608MM", "values": [{"_id": "a", "da": "000000000000", "value": "0"}]},
                {"code": "202607MM", "values": [{"_id": "a", "da": "000000000000", "value": "-1.5"},
                                                   {"_id": "b", "da": "000000000000", "value": ""}]},
            ], {"raw_path": "原始响应.json.gz", "retrieved_at": "2026-10-04T12:00:00+08:00"}

    history(OfficialResponse(), "190001", "202609")
    frame = pd.read_parquet(tmp_path / "panels/table.parquet")
    zero = frame.loc[frame.indicator_id.eq("a") & frame.period.eq("202608")].iloc[0]
    assert zero.value == 0 and zero.status == "有值"
    omitted = frame.loc[frame.indicator_id.eq("a") & frame.period.eq("202609")].iloc[0]
    assert pd.isna(omitted.value) and not omitted.source_cell_present
    explicit_blank = frame.loc[frame.indicator_id.eq("b") & frame.period.eq("202607")].iloc[0]
    assert pd.isna(explicit_blank.value) and explicit_blank.source_cell_present
    status = json.loads((tmp_path / "history_status.json").read_text(encoding="utf-8"))
    assert not status["failed"]
    assert next(r for r in status["coverage"] if r["indicator_id"] == "b")["status"] == "官方查询全空"


def test_merged_table_preserves_multi_level_header_and_negative_values():
    table = BeautifulSoup(
        '<table><tr><th rowspan="2">行业</th><th colspan="2">增速</th></tr>'
        '<tr><th>当月同比</th><th>累计同比</th></tr>'
        '<tr><td>制造业</td><td>0</td><td>-2.5</td></tr></table>', "html.parser").table
    grid, cells = expand_table(table)
    assert grid == [["行业", "增速", "增速"], ["行业", "当月同比", "累计同比"], ["制造业", "0", "-2.5"]]
    assert cells[0]["rowspan"] == 2 and cells[1]["colspan"] == 2


def test_migrated_url_never_substitutes_for_publication_date(tmp_path):
    html = ('<html><head><meta name="PubDate" content="2020-03-27 09:30:00">'
            '<meta name="ArticleTitle" content="2020年1—2月份全国规模以上工业企业利润"></head>'
            '<body><div class="TRS_Editor"><p>原始公布数据。</p><table><tr><td>工业</td><td>0</td></tr></table>'
            '</div></body></html>').encode("utf-8")
    result = parse_release(html, {"url": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900702.html",
                                  "title": "目录标题", "catalogue_date": "2020-03-27"},
                           {"raw_path": "原始响应.json.gz", "retrieved_at": "2026-10-04T12:00:00+08:00", "sha256": "x"}, tmp_path)
    assert result["published_at"] == "2020-03-27 09:30:00"
    assert result["title"].startswith("2020年") and result["table_count"] == 1


def test_disclosure_page_keeps_formed_date_separate_from_publication_time(tmp_path):
    html = ('<div class="content_top_box"><div><span>成文日期</span>2019年01月28日</div></div>'
            '<h2 class="xxgkNbXqTitle">2018年全国规模以上工业企业利润增长10.3%</h2>'
            '<div class="TRS_PreAppend"><p>利润总额66351.4亿元。</p>'
            '<table><tr><td>采矿业</td><td>5246.4</td></tr></table></div>').encode("utf-8")
    result = parse_release(html, {"url": "https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/201901/t20190128_1768279.html",
                                  "title": "待从官方原页读取", "catalogue_date": None},
                           {"raw_path": "旧稿.json.gz", "retrieved_at": "2026-10-04T12:00:00+08:00", "sha256": "x"}, tmp_path)
    assert result["formed_on"] == "2019-01-28" and result["published_at"] is None
    assert result["title"].startswith("2018年") and result["table_count"] == 1


def test_visible_publication_header_survives_missing_meta_and_migration(tmp_path):
    html = ('<div class="detail-title"><h1>2020年1—9月份全国规模以上工业企业利润下降2.4%</h1>'
            '<div class="detail-title-des"><h2><p>2020/10/27 09:30</p></h2></div></div>'
            '<div class="TRS_Editor"><p>利润原始数据。</p></div>').encode("utf-8")
    result = parse_release(html, {"url": "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900893.html",
                                  "title": "待从官方原页读取", "catalogue_date": None},
                           {"raw_path": "迁移稿.json.gz", "retrieved_at": "2026-10-04T12:00:00+08:00", "sha256": "x"}, tmp_path)
    assert result["published_at"] == "2020/10/27 09:30"
    assert result["title"].startswith("2020年") and result["parser_version"] == 2
