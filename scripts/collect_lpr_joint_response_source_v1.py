"""读取央行公开LPR公告目录及原文，保留正常公布和未变更月份。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_lpr_joint_response_v1"
SOURCE = OUT / "sources"
BASE = "https://www.pbc.gov.cn"
TEMPLATE = "/en/3688229/3688335/3730276/3883798/19e15ae1-%1.html"
DATE_CORRECTIONS = {
    "2019-08-21": ("2019-08-20", "lpr_2019-08-20_zh.html"),
    "2019-10-20": ("2019-10-21", "lpr_2019-10-21_zh.html"),
}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch(name, url):
    path = SOURCE / name
    if path.exists():
        return path
    try:
        response = requests.get(url, timeout=35)
    except requests.RequestException as exc:
        save(path.with_suffix(path.suffix+".failure.json"), {"url": url, "retrieved_at": now(),
            "status": "SOURCE_FETCH_FAILED", "error_type": type(exc).__name__})
        raise
    path.write_bytes(response.content)
    save(path.with_suffix(path.suffix+".receipt.json"), {"url": url, "status": response.status_code,
        "retrieved_at": now(), "sha256": hashlib.sha256(response.content).hexdigest(),
        "bytes": len(response.content), "historical_first_seen_proven": False})
    response.raise_for_status()
    return path


def catalog():
    source = BeautifulSoup((SOURCE / "pboc_lpr_page4.html").read_bytes(), "html.parser")
    assert source.select_one("input[totalpage]").get("totalpage") == "9"
    assert TEMPLATE in str(source)
    def page(k):
        path = fetch(f"pboc_lpr_page{k}.html", urljoin(BASE, TEMPLATE.replace("%1", str(k))))
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        result = []
        for block in soup.select("div.ListR"):
            link = block.select_one("a[title][href]")
            date = block.select_one("span.prhhdata")
            if link is None or date is None:
                continue
            title = link.get("title", "")
            if title.startswith("Announcement on Loan Prime Rate") and "(" in title:
                matches = re.findall(r"\(([^()]*)\)", title)
                title_date = pd.to_datetime(matches[-1], format="mixed").date().isoformat()
                economic_date, correction_file = DATE_CORRECTIONS.get(title_date, (title_date, ""))
                if correction_file:
                    chinese_text = BeautifulSoup((SOURCE / correction_file).read_bytes(), "html.parser").get_text(" ", strip=True)
                    assert economic_date+" 09:30:00" in chinese_text
                result.append({"release_date": economic_date, "directory_date": date.get_text(strip=True),
                    "original_title_date": title_date, "correction_file": correction_file,
                    "title": title, "url": urljoin(BASE, link["href"]), "catalogue_file": path.name})
        return result
    rows = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for result in pool.map(page, range(1, 10)):
            rows.extend(result)
    frame = pd.DataFrame(rows).sort_values("release_date").reset_index(drop=True)
    assert frame.release_date.is_unique
    selected = frame[frame.release_date.between("2019-08-20", "2025-12-31")].copy()
    periods = pd.to_datetime(selected.release_date).dt.to_period("M")
    expected = pd.period_range("2019-08", "2025-12", freq="M")
    missing = [str(v) for v in expected if v not in set(periods)]
    save(OUT / "lpr_catalogue.json", selected.to_dict("records"))
    save(OUT / "source_calendar_coverage.json", {"at": now(), "source": "人民银行公开LPR公告目录",
        "catalogue_pages": 9, "all_quote_records": len(frame), "selected_records": len(selected),
        "expected_months": len(expected), "missing_months": missing,
        "directory_date_mismatches": selected[selected.release_date.ne(selected.directory_date)].to_dict("records"),
        "status": "PASS_MONTHLY_CALENDAR" if not missing and len(selected) == len(expected) else "INCOMPLETE"})
    print(json.dumps({"目录记录": len(frame), "研究区间公告": len(selected), "缺失月份": missing,
        "首条": selected.head(1).to_dict("records"), "末条": selected.tail(1).to_dict("records")}, ensure_ascii=False), flush=True)


def articles():
    rows = json.loads((OUT / "lpr_catalogue.json").read_text(encoding="utf-8"))
    def article(row):
        filename = "lpr_"+row["original_title_date"]+".html"
        path = fetch(filename, row["url"])
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        text = soup.get_text(" ", strip=True)
        # 正文中的定期报价段落与网站顶部说明不同，优先从最后一次出现的原文开始。
        marker = "The National Interbank Funding Center"
        position = text.rfind(marker)
        if position < 0:
            position = text.rfind("National Interbank Funding Center")
        excerpt = text[position:position+2600] if position >= 0 else text[-4500:]
        rates = re.findall(r"(\d+(?:\.\d+)?)\s*(?:percent|per cent|%)", excerpt, flags=re.I)
        return {**row, "raw_file": filename, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "body_excerpt": excerpt, "rate_tokens": rates}
    collected = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for row in pool.map(article, rows):
            collected.append(row)
            if len(collected) % 20 == 0:
                print(f"已取得{len(collected)}份央行原公告。", flush=True)
    save(OUT / "lpr_original_notices.json", collected)
    print(json.dumps({"公告全文数": len(collected), "首条正文": collected[0]["body_excerpt"],
        "末条正文": collected[-1]["body_excerpt"], "利率提取计数": pd.Series([len(r['rate_tokens']) for r in collected]).value_counts().to_dict()}, ensure_ascii=False), flush=True)


def finalize():
    rows = json.loads((OUT / "lpr_original_notices.json").read_text(encoding="utf-8"))
    frame = pd.DataFrame(rows).sort_values("release_date").reset_index(drop=True)
    periods = pd.to_datetime(frame.release_date).dt.to_period("M")
    expected = pd.period_range("2019-08", "2025-12", freq="M")
    assert len(frame) == 77 and periods.is_unique and set(periods) == set(expected)
    assert frame.release_date.eq(frame.directory_date).all()
    assert frame.rate_tokens.map(len).ge(2).all(), "原公告报价段落提取不完整"
    for row in rows:
        path = SOURCE / row["raw_file"]
        receipt = json.loads(path.with_suffix(path.suffix+".receipt.json").read_text(encoding="utf-8"))
        assert receipt["status"] == 200 and receipt["url"] == row["url"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    columns = ["release_date", "directory_date", "original_title_date", "correction_file", "title", "url", "raw_file", "sha256"]
    frame[columns].to_csv(OUT / "lpr_releases.csv", index=False, encoding="utf-8-sig")
    record = {"at": now(), "status": "PASS_77_MONTHLY_RELEASE_DATES", "monthly_records": len(frame),
        "first_release": frame.release_date.iloc[0], "last_release": frame.release_date.iloc[-1],
        "date_corrections": frame.loc[frame.correction_file.ne(""), columns].to_dict("records"),
        "unusual_title_included": frame.loc[frame.release_date.eq("2024-07-22"), columns].to_dict("records"),
        "historical_first_seen_proven": False,
        "source_boundary": "本次回取官方目录和原文，仅供历史重建；不声称当时实际获取。保留未变动报价月份。",
        "public_download_limit": "中国货币网公共历史接口只返回一年查询限制，未绕过；改用央行公开公告档案。"}
    save(OUT / "source_finalization.json", record)
    print("77个月央行公告来源已齐全，两处英文日期已用中文原公告更正。", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="取得LPR定期公告官方来源")
    parser.add_argument("stage", choices=["catalog", "articles", "finalize"])
    args = parser.parse_args()
    {"catalog": catalog, "articles": articles, "finalize": finalize}[args.stage]()
