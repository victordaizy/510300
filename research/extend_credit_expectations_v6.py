"""从既定病例同一事前调查原页补充贷款与社融预期，保留单位、时钟和舍入边界。"""
from pathlib import Path
import json
import re
import shutil
import sys

from bs4 import BeautifulSoup
import pandas as pd
import pdfplumber

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.information_change_transmission_v6 import OUT, save, csv, sha, now


def run():
    sources = json.loads((OUT / "sources/三个既定病例的事前报告.json").read_text(encoding="utf-8"))
    loans = pd.read_csv(ROOT / "reports/research/510300_macro_transmission_context_v4/results/104个月_信贷分项与同区间比较.csv").set_index("stat_month")
    monthly = pd.read_csv(OUT / "inputs/monthly.csv").set_index("stat_month")
    tsf_specs = {
        "2021-01": {"actual_yi": 51700.0, "rounding_yi": 50.0, "known_upper": "2021-02-22T23:59:59+08:00", "url": "https://www.ndrc.gov.cn/fggz/fgzh/gnjjjc/hbjr/202102/t20210222_1267615.html", "file": "tsf_202101_ndrc.html", "status": "LATER_OFFICIAL_REPRINT_NOT_ORIGIN_TIME_SOURCE"},
        "2024-08": {"actual_yi": 219000.0, "rounding_yi": 500.0, "known_upper": "2024-09-14T23:59:59+08:00", "url": "https://jrj.sh.gov.cn/SCGK194/20240914/3184186167db42939ec38aa8500a9953.html", "file": None, "status": "OFFICIAL_REPRINT_WEB_READ_HTTP403_NOT_ORIGIN_TIME_SOURCE"},
        "2025-08": {"actual_yi": 265600.0, "rounding_yi": 50.0, "known_upper": "2025-09-12T17:00:00+08:00", "url": "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/523c260b344c4f1390664430295064a9/index.html", "file": "tsf_202508.html", "status": "SAVED_OFFICIAL_PAGE_VISIBLE_AT_ORIGIN"},
    }
    raw = ROOT / "reports/research/510300_macro_transmission_context_v4/sources/tsf_202508.html"
    assert "26.56万亿元" in BeautifulSoup(raw.read_bytes(), "html.parser").get_text()
    shutil.copy2(raw, OUT / "sources/tsf_202508.html")
    first = OUT / "sources/tsf_202101_ndrc.html"
    assert first.exists() and "5.17万亿元" in BeautifulSoup(first.read_bytes(), "html.parser").get_text()
    save("sources/社融回顾补充来源.json", {"at": now(), "records": tsf_specs, "transport_note": "上海官方转载本轮浏览检索可读，直接HTTP403；不反复请求，也不把转载时间倒填。", "local_hashes": {f: sha(OUT / "sources" / f) for f in ["tsf_202101_ndrc.html", "tsf_202508.html"]}})
    rows = []
    for item in sources:
        month = item["stat_month"]
        with pdfplumber.open(OUT / "sources" / item["filename"]) as doc:
            text = doc.pages[item["M1"]["page"] - 1].extract_text() or ""
        china = text.rsplit("China", 1)[-1]
        assert re.search(r"Survey\s+Actual\s+Prior", china)
        for metric, pattern in [("人民币贷款新增", r"New (?:Yuan )?Loans CNY( YTD)?\s+(Jan|Aug)\s+([\d,.]+)b\s+(--|[\d,.]+b)"), ("社会融资增量", r"Aggregate Financing CNY( YTD)?\s+(Jan|Aug)\s+([\d,.]+)b\s+(--|[\d,.]+b)")]:
            found = re.findall(pattern, china)
            assert len(found) == 1, (month, metric, found)
            ytd, calendar_month, number, actual_cell = found[0]
            assert actual_cell == "--"
            assert calendar_month == ("Jan" if month == "2021-01" else "Aug")
            assert bool(ytd) == (month != "2021-01")
            forecast = float(number.replace(",", "")) * 10
            digits = len(number.rsplit(".", 1)[1]) if "." in number else 0
            forecast_round = .5 * 10 ** (-digits) * 10
            if metric == "人民币贷款新增":
                actual = float(loans.loc[month, "rmb_total_ytd_yi"])
                assert str(monthly.loc[month, "loan_period"]) == month
                assert actual == float(monthly.loc[month, "loan_rmb_total_ytd_yi"])
                rounding = float(loans.loc[month, "rmb_total_ytd_rounding_half_yi"])
                known = str(monthly.loc[month, "loan_known_at"])
                url = str(monthly.loc[month, "loan_source_url"])
                status = "INHERITED_OFFICIAL_FINANCIAL_RELEASE_VISIBLE_AT_ORIGIN"
            else:
                spec = tsf_specs[month]
                actual, rounding, known, url, status = spec["actual_yi"], spec["rounding_yi"], spec["known_upper"], spec["url"], spec["status"]
            diff = actual - forecast
            bound = rounding + forecast_round
            sign = "正向且超出展示精度范围" if diff > bound else "负向且超出展示精度范围" if diff < -bound else "差异落在展示精度范围内"
            rows.append({"stat_month": month, "metric": metric, "period": "当年1月" if month == "2021-01" else "当年1至8月累计", "forecast_report_date": item["report_date"], "forecast_source_file": item["filename"], "forecast_page_one_based": item["M1"]["page"], "forecast_source_sha256": item["sha256"], "forecast_source_url": item["url"], "forecast_original_billion_cny": float(number.replace(",", "")), "forecast_yi": forecast, "actual_yi": actual, "actual_minus_forecast_yi": diff, "combined_display_rounding_halfwidth_yi": bound, "difference_lower_yi": diff - bound, "difference_upper_yi": diff + bound, "display_sign_description": sign, "actual_source_known_upper": known, "actual_source_url": url, "actual_source_status": status, "visible_using_this_actual_source_at_original_snapshot": pd.Timestamp(known) <= pd.Timestamp(monthly.loc[month, "snapshot_at"]), "interpretation": "同一调查记录的预测差；不是当月减去累计，不与同比多增贡献混用，不能代替分项预期。"})
    frame = pd.DataFrame(rows)
    csv(frame, "三例同一调查_贷款社融预期与实际.csv")
    # 政府债解释的是同比多增结构，不是相对预期差。特意分别建字段。
    source = pd.read_csv(ROOT / "reports/research/510300_macro_transmission_context_v4/results/2025年6至8月_社融多增来源.csv").set_index("stat_month").loc["2025-08"]
    save("results/2025年8月_三种比较基准不可混用.json", {"at": now(), "three_month_spread_change_pp": 2.8, "spread_actual_minus_forecast_pp": -.1, "spread_four_component_display_rounding_halfwidth_pp": .2, "new_loans_ytd_actual_minus_forecast_yi": float(frame[(frame.stat_month == "2025-08") & (frame.metric == "人民币贷款新增")].actual_minus_forecast_yi.iloc[0]), "tsf_ytd_actual_minus_forecast_yi": float(frame[(frame.stat_month == "2025-08") & (frame.metric == "社会融资增量")].actual_minus_forecast_yi.iloc[0]), "tsf_ytd_yoy_increase_yi": float(source.total_tsf_ytd_yoy_increase_yi), "government_ytd_yoy_increase_yi": float(source.government_bond_ytd_yoy_increase_yi), "government_share_of_tsf_yoy_increase": float(source.government_share_of_tsf_yoy_increase), "government_bond_forecast": None, "government_contribution_to_tsf_forecast_error": None, "unidentified_reason": "没有同口径政府债分项事前预期，不能用99.36%的同比多增占比去分配693亿元的预期差。", "new_return_tests": 0})
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    print(frame[["stat_month", "metric", "forecast_yi", "actual_yi", "actual_minus_forecast_yi", "display_sign_description", "visible_using_this_actual_source_at_original_snapshot"]].to_string(index=False))


if __name__ == "__main__":
    run()
