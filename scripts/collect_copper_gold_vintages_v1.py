"""读取世界银行每月原版铜、金价格，只取得注册区间的公开文件。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pandas as pd
import pdfplumber
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_copper_gold_monthly_vintage_v1"
SOURCE = OUT / "sources"
BASE = "https://thedocs.worldbank.org"
MONTHS = "January February March April May June July August September October November December".split()
SEED_IDS = {
    "2017_archive": "395981516914586689-0050022018",
    "2021_2024_archive": "5d903e848db1d1b83e0ec8f744e55570-0350012021",
    "2025_archive": "18675f1d1639c7a34d463f59263ba0a2-0050012025",
    "2018_01": "406371515004954502-0050022018",
    "2018_02": "752911517610537957-0050022018",
    "2018_04": "664371522786567489-0050022018",
    "2018_05": "536761525295567334-0050022018",
    "2018_09": "453081536593505013-0050022018",
    "2018_10": "283421538517311854-0050022018",
    "2018_11": "363671541184621547-0050022018",
    "2018_12": "451911543968504757-0050022018",
    "2018_03": "346911520263101497-0050022018",
    "2018_06": "799841528151608411-0050022018",
    "2019_01": "921301546633915027-0050022019",
    "2019_02": "550191549309123169-0050022019",
    "2019_03": "927931551717534246-0050022019",
    "2019_04": "895911554230066764-0050022019",
    "2019_05": "575341556832785539-0050022019",
    "2019_06": "169031559692506553-0050022019",
    "2019_07": "298031562084790383-0050022019",
    "2019_08": "257941564775744979-0050022019",
    "2019_09": "728281567625718446-0050022019",
    "2019_10": "928931570034997598-0050022019",
    "2019_11": "771291572896477076-0050022019",
    "2019_12": "678281575404408706-0050022019",
    "2020_01": "386711578078060390-0050022020",
    "2020_02": "596831580311438199-0050022020",
    "2020_03": "541851583268074222-0050022020",
    "2020_04": "992071585858056509-0050022020",
    "2020_05": "959181588615089571-0050022020",
    "2020_06": "774651591120179792-0050022020",
    "2020_07": "722721593705473133-0050022020",
    "2020_08": "935161596562812622-0050022020",
    "2020_09": "451141599073982216-0050022020",
    "2020_10": "520721601663433090-0050022020",
    "2020_11": "843201604424898761-0050022020",
    "2020_12": "724951606935391601-0050022020",
    "2021_01": "854731609876300889-0050022021",
    "2021_02": "804991612306143358-0050022021",
}
DIRECT_URLS = [
    "https://pubdocs.worldbank.org/en/375121504637019355/CMO-Pink-Sheet-September-2017.pdf",
    "http://pubdocs.worldbank.org/en/484911509640161927/CMO-Pink-Sheet-November-2017.pdf",
    "https://pubdocs.worldbank.org/en/451911543968504757/CMO-Pink-Sheet-December-2018.pdf",
    "https://pubdocs.worldbank.org/en/283421538517311854/CMO-Pink-Sheet-October-2018.pdf",
    "http://pubdocs.worldbank.org/en/664371522786567489/CMO-Pink-Sheet-April-2018.pdf",
]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch(filename, url):
    path = SOURCE / filename
    if path.exists():
        receipt_path = path.with_suffix(path.suffix+".receipt.json")
        assert receipt_path.exists(), "缓存缺少来源回执："+filename
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["url"] == url, "缓存URL与本次选定来源不一致："+filename
        assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), "缓存内容与回执不一致："+filename
        return path
    for attempt in range(2):
        try:
            response = requests.get(url, timeout=35)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            save(path.with_suffix(path.suffix+".failure.json"), {"at": now(), "url": url, "attempt": attempt+1, "error_type": type(exc).__name__})
            if attempt == 1 or not isinstance(exc, (requests.ConnectionError, requests.Timeout)):
                raise
            time.sleep(.5)
    if filename.endswith(".pdf"):
        assert response.content.startswith(b"%PDF"), "响应不是PDF"
    path.write_bytes(response.content)
    save(path.with_suffix(path.suffix+".receipt.json"), {"at": now(), "url": url, "final_url": response.url,
        "status": response.status_code, "sha256": hashlib.sha256(response.content).hexdigest(), "bytes": len(response.content),
        "historical_first_seen_proven": False})
    return path


def month_from_url(url):
    if not re.search(r"Pink[\d -]*Sheet", url, re.I):
        return None
    match = re.search("("+"|".join(MONTHS)+r")[\d-]*(20[12]\d)\.pdf$", url, re.I)
    if not match:
        return None
    month = MONTHS.index(match[1].title())+1
    year = int(match[2])
    return f"{year:04d}-{month:02d}" if 2017 <= year <= 2025 else None


def extract_price_values(line):
    # 原PDF把千分位以前的数字拆开为文本片段，例如“1 0,162”。
    # 铜、金在本注册区间均以含千分位的整数美元报价；以逗号及其后三位锚定单元格，禁止串接相邻列。
    tokens = re.findall(r"\d[\d ]*,\s*\d{3}(?:\.\d+)?", line)
    values = [float(re.sub(r"[,\s]", "", token)) for token in tokens]
    assert len(values) == 11 and all(v > 0 for v in values), "商品行列数异常："+line
    return values


def extract_monthly_periods(header):
    lines = header.splitlines()
    # 部分原表相邻英文月份没有文本空格，例如 NovemberDecember。
    month_lines = [re.sub(r"(?<=[a-z])(?=[A-Z])", " ", line).split() for line in lines if line.startswith("Jan-Dec ")]
    year_lines = [re.findall(r"\b20\d{2}\b", line) for line in lines if line.startswith(("Commodity Unit ", "Unit "))]
    assert len(month_lines) == len(year_lines) == 1, "价格表月份或年份表头不唯一"
    assert len(month_lines[0]) == len(year_lines[0]) == 11, "价格表表头列数异常"
    aliases = {name[:3].lower(): i+1 for i, name in enumerate(MONTHS)}
    return [f"{year}-{aliases[month[:3].lower()]:02d}" for month, year in zip(month_lines[0][-3:], year_lines[0][-3:])]


def catalog():
    def get_page(item):
        name, ident = item
        url = f"{BASE}/en/doc/{ident}/"
        path = fetch("archive_"+name+".html", url)
        soup = BeautifulSoup(path.read_bytes(), "html.parser")
        return [(urljoin(BASE, a["href"]), url) for a in soup.select("a[href]") if month_from_url(a["href"])]
    links = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(get_page, SEED_IDS.items()):
            links.extend(result)
    links.extend((url, "OFFICIAL_URL_DISCOVERED_IN_PUBLIC_SEARCH") for url in DIRECT_URLS)
    extras = OUT / "additional_source_urls.json"
    if extras.exists():
        links.extend((url, "ADDITIONAL_OFFICIAL_SEARCH_OR_BLOG_LINK") for url in json.loads(extras.read_text(encoding="utf-8")))
    exclusions_path = OUT / "source_url_exclusions.json"
    exclusions = json.loads(exclusions_path.read_text(encoding="utf-8")) if exclusions_path.exists() else {}
    by_month = {}
    for url, parent in links:
        if url in exclusions:
            continue
        period = month_from_url(url)
        if period and period not in by_month:
            by_month[period] = {"release_month": period, "url": url, "parent_url": parent}
    rows = sorted(by_month.values(), key=lambda r: r["release_month"])
    missing = [str(p) for p in pd.period_range("2017-01", "2025-12", freq="M") if str(p) not in by_month]
    save(OUT / "source_catalogue.json", rows)
    save(OUT / "source_discovery_coverage.json", {"at": now(), "known_months": len(rows), "missing_months": missing})
    print(json.dumps({"已找到月份": len(rows), "尚缺月份": missing}, ensure_ascii=False), flush=True)


def download():
    rows = json.loads((OUT / "source_catalogue.json").read_text(encoding="utf-8"))
    def get(row):
        previous_failure = SOURCE / ("pink_"+row["release_month"]+".pdf.failure.json")
        if previous_failure.exists():
            prior = json.loads(previous_failure.read_text(encoding="utf-8"))
            if prior["url"] == row["url"] and prior["error_type"] == "HTTPError":
                return {"month": row["release_month"], "status": "FAILED", "error": "保留同一URL已确认的HTTP失败，不重复请求。"}
        try:
            fetch("pink_"+row["release_month"]+".pdf", row["url"])
            return {"month": row["release_month"], "status": "DOWNLOADED"}
        except (requests.RequestException, AssertionError) as exc:
            return {"month": row["release_month"], "status": "FAILED", "error": str(exc)}
    results = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, record in enumerate(pool.map(get, rows), 1):
            results.append(record)
            if i % 20 == 0:
                print(f"已处理{i}份月度原报告。", flush=True)
    save(OUT / "download_status.json", {"at": now(), "results": results})
    print(json.dumps({"下载成功": sum(r["status"] == "DOWNLOADED" for r in results), "失败": [r for r in results if r["status"] == "FAILED"]}, ensure_ascii=False), flush=True)


def parse():
    rows = json.loads((OUT / "source_catalogue.json").read_text(encoding="utf-8"))
    output = []
    text_failures = []
    manual_path = OUT / "source_manual_transcriptions.json"
    manual = json.loads(manual_path.read_text(encoding="utf-8")) if manual_path.exists() else {}
    for source in rows:
        path = SOURCE / ("pink_"+source["release_month"]+".pdf")
        if not path.exists():
            continue
        receipt = json.loads(path.with_suffix(".pdf.receipt.json").read_text(encoding="utf-8"))
        assert receipt["url"] == source["url"]
        assert receipt["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        cache = path.with_suffix(".text.json")
        if cache.exists():
            texts = json.loads(cache.read_text(encoding="utf-8"))
        else:
            with pdfplumber.open(path) as doc:
                texts = [p.extract_text() or "" for p in doc.pages]
            save(cache, texts)
        manual_record = manual.get(source["release_month"])
        if manual_record:
            assert manual_record["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            texts = manual_record["transcribed_page_texts"]
        if not any("Copper" in t and "Gold" in t for t in texts):
            text_failures.append(source["release_month"])
            continue
        header = texts[0][:1800]
        stamp = re.search(r"\b\d{1,2}-[A-Za-z]{3}-20\d{2}\b|\b(?:"+"|".join(MONTHS)+r")\s+\d{1,2},\s*20\d{2}\b", header)
        assert stamp, path.name+"没有表头日期"
        release = pd.Timestamp(stamp[0]).date().isoformat()
        assert release[:7] == source["release_month"], path.name+"表头月份与来源月份不同"
        result = {**source, "release_date": release, "raw_file": str(path.relative_to(OUT)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "pages": len(texts),
            "extraction_method": "MANUAL_VISUAL_TRANSCRIPTION" if manual_record else "PDF_TEXT"}
        for key, commodity, unit in [("copper", "Copper", "$/mt"), ("gold", "Gold", "$/toz")]:
            matching = [(i+1,line) for i,t in enumerate(texts) for line in t.splitlines() if line.startswith(commodity+" ") and unit in line]
            assert len(matching) == 1, path.name+"商品行不唯一："+commodity
            page, line = matching[0]
            numbers = extract_price_values(line)
            result[key+"_page"] = page
            result[key+"_source_row"] = line
            result[key+"_previous_month"] = numbers[-2]
            result[key+"_latest_month"] = numbers[-1]
            result[key+"_three_months"] = numbers[-3:]
            # 月度最后三列的表头单独保留，供版式抽查。
            result[key+"_header"] = texts[page-1].split(commodity+" ")[0][:1100]
            periods = extract_monthly_periods(result[key+"_header"])
            expected = [str(pd.Period(source["release_month"], "M")-shift) for shift in (3, 2, 1)]
            assert periods == expected, path.name+"月度数据列与实际统计期不一致"
            result[key+"_monthly_periods"] = periods
        result["latest_statistic_month"] = str(pd.Period(source["release_month"], "M")-1)
        result["previous_statistic_month"] = str(pd.Period(source["release_month"], "M")-2)
        output.append(result)
        if len(output) % 20 == 0:
            print(f"已核对{len(output)}份原表的两行价格与日期。", flush=True)
    save(OUT / "original_price_rows.json", output)
    pd.DataFrame(output).to_csv(OUT / "原版铜金价格与月份.csv", index=False, encoding="utf-8-sig")
    present = {r["release_month"] for r in output}
    missing = [str(m) for m in pd.period_range("2017-01", "2025-12", freq="M") if str(m) not in present]
    save(OUT / "source_extraction_status.json", {"at": now(), "extracted_months": len(output), "expected_months": 108,
        "missing_months": missing, "all_sources_complete": len(output) == 108,
        "text_extraction_unresolved_months": text_failures,
        "source_boundary": "缺失月份保留NO_VIEW，不能把子样本计算当成完整样本账户验证。"})
    print(f"已提取{len(output)}份原报告的铜、金两行及原表头。", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="世界银行月度铜金原版来源")
    parser.add_argument("stage", choices=["catalog", "download", "parse"])
    args = parser.parse_args()
    {"catalog": catalog, "download": download, "parse": parse}[args.stage]()
