"""从官方发布目录取得2020年后同名收款期限原月报，保留可比同比与公布日。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import sys
from threading import Lock
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.industrial_receivable_source_probe_v1 as probe
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_industrial_receivable_monthly_source_v1"
STUDY = "510300_INDUSTRIAL_RECEIVABLE_MONTHLY_SOURCE_V1"
BASE = "https://www.stats.gov.cn/sj/zxfb/"
EXPECTED = [str(month) for month in pd.period_range("2020-02", "2026-07", freq="M") if month.month != 1]
PANEL = OUT / "released_industrial_receivables.parquet"
_lock = Lock()
_request_count = len(list((OUT / "requests").glob("*.json")))


def fetch(url, name):
    global _request_count
    target = OUT / "requests" / f"{name}.json"
    if target.exists():
        receipt = read(target)
        if receipt.get("raw_path"):
            assert digest(ROOT / receipt["raw_path"]) == receipt["sha256"]
        return receipt
    assert urlparse(url).hostname == "www.stats.gov.cn"
    with _lock:
        if (OUT / "REQUESTS_STOPPED.json").exists():
            return {"url": url, "status": "NOT_REQUESTED_AFTER_SOURCE_STOP"}
        assert _request_count < 200, "超过预定200次请求上限。"
        _request_count += 1
    record = {"requested_at": now(), "url": url, "status": "REQUEST_STARTED"}
    try:
        response = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        raw = OUT / "raw" / f"{name}.html"
        raw.write_bytes(response.content)
        record.update(received_at=now(), final_url=response.url, http_status=response.status_code,
            status="HTTP_OK" if response.ok else "HTTP_FAILURE", bytes=len(response.content),
            raw_path=raw.relative_to(ROOT).as_posix(), sha256=digest(raw))
        if response.status_code in [403, 429]:
            with _lock:
                if not (OUT / "REQUESTS_STOPPED.json").exists():
                    save(OUT / "REQUESTS_STOPPED.json", {"at": now(), "url": url, "http_status": response.status_code}, True)
    except requests.RequestException as exc:
        record.update(received_at=now(), status="TRANSPORT_FAILURE", error=f"{type(exc).__name__}: {exc}")
    save(target, record, True)
    return record


def request_job(job):
    url, name = job
    receipt = fetch(url, name)
    if receipt["status"] == "TRANSPORT_FAILURE":
        receipt = fetch(url, name + "_transport_retry")
    return receipt


def soup_for(receipt):
    assert receipt["status"] == "HTTP_OK", receipt["status"]
    assert digest(ROOT / receipt["raw_path"]) == receipt["sha256"]
    return BeautifulSoup((ROOT / receipt["raw_path"]).read_bytes(), "html.parser", from_encoding="utf-8")


def title_month(title):
    match = re.match(r"^(20\d{2})年(?:1[—－–-](\d{1,2})月份)?全国规模以上工业企业利润", re.sub(r"\s+", "", title))
    if match is None:
        return None
    return f"{match.group(1)}-{int(match.group(2) or 12):02d}"


def catalogue():
    if (OUT / "catalogue.json").exists():
        raise RuntimeError("工业企业目录已取得终态。")
    for name in ["raw", "requests", "code"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    front = read(probe.OUT / "requests/release_catalogue.json")
    assert digest(ROOT / front["raw_path"]) == front["sha256"]
    html = (ROOT / front["raw_path"]).read_text(encoding="utf-8")
    count = int(re.search(r'createPageHTML\((\d+),\s*0,\s*"index",\s*"html"\)', html).group(1))
    assert count == 67
    save(OUT / "protocol.json", {"at": now(), "study_id": STUDY, "expected_months": EXPECTED,
        "source_method": "沿已保存官方最新发布页明确的67页静态分页取得原文链接，只保存2020-02到2026-07全国工业企业利润原月报。",
        "transport": "最多200次请求，同时至多2个公开GET，各25秒；传输失败额外普通重试一次，403/429停止新增。已保存原文直接复用。",
        "semantics": "采用应收账款平均回收期及原报告直接公布的同比增减天数，保留360×平均应收账款÷营业收入×累计月数÷12的定义。不是逐笔发票实测回款时间、违约率或股票资金流。",
        "breaks": "2018主营业务收入和2019含票据两种旧口径不拼入；2020后相同字段也可能因调查范围或基数调整不可直接跨年相减，必须用原报告公布同比。",
        "clock": "报告原页PubDate或正文发布时间，与目录公布日核对；保守日末可用。迁移后的URL日期不当成原公布日期。",
        "feature_extraction": "同名正文值与经济效益表总计行交叉核对；正文直接同比只去相同桌面/移动重复，矛盾不能强行平均。",
        "partial_history": "遗漏月份、失败响应和字段歧义明确保留；不使用后来修订数据库填补。",
        "new_accounts": 0, "new_strategy_returns": 0, "historical_first_vintage_verified": False,
        "current_market_view": "NO_VIEW", "orders_authorized": False, "goal_achieved": False}, True)
    save(OUT / "freeze.json", {"at": now(), "protocol_sha256": digest(OUT / "protocol.json"),
        "code_sha256": digest(Path(__file__)), "probe_manifest_sha256": digest(probe.OUT / "source_measurements.json")}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    pages, failed = [front], []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for left in range(1, count, 2):
            jobs = [(BASE + f"index_{i}.html", f"catalogue_{i}") for i in range(left, min(left + 2, count))]
            for receipt in executor.map(request_job, jobs):
                if receipt["status"] == "HTTP_OK":
                    pages.append(receipt)
                else:
                    failed.append(receipt)
            if len(pages) % 10 == 1:
                print(f"工业企业官方目录已保存{len(pages)}/{count}页。", flush=True)
            if (OUT / "REQUESTS_STOPPED.json").exists():
                break
    found = {}
    for receipt in pages:
        for li in soup_for(receipt).find_all("li"):
            a = li.find("a", href=True)
            if a is None:
                continue
            title = a.get("title", a.get_text(" ", strip=True))
            month = title_month(title)
            if month not in EXPECTED:
                continue
            dates = set(re.findall(r"20\d{2}-\d{2}-\d{2}", li.get_text()))
            assert len(dates) == 1, (month, "目录日期不唯一")
            item = {"stat_month": month, "title": title, "url": urljoin(receipt["url"], a["href"]),
                    "catalogue_publication_date": next(iter(dates)), "catalogue_raw_path": receipt["raw_path"]}
            if month in found:
                assert found[month]["url"] == item["url"], (month, "目录存在不同原报告链接")
            found[month] = item
    save(OUT / "catalogue.json", sorted(found.values(), key=lambda r: r["stat_month"]), True)
    save(OUT / "catalogue_result.json", {"at": now(), "catalogue_pages_saved": len(pages), "failed_pages": failed,
        "expected_months": len(EXPECTED), "located_months": len(found), "missing_months": sorted(set(EXPECTED) - found.keys()),
        "new_strategy_returns": 0, "new_accounts": 0}, True)
    print(f"工业收款目录定位{len(found)}/{len(EXPECTED)}个月，缺口明确保留。", flush=True)


def parse(item, receipt):
    soup = soup_for(receipt)
    text = soup.get_text(" ", strip=True)
    compact = re.sub(r"\s+", "", text)
    title = soup.title.get_text(strip=True) if soup.title else ""
    assert title_month(title) == item["stat_month"], "原文标题与目录统计月不一致。"
    clocks = set(re.findall(r"20\d{2}[-/]\d{2}[-/]\d{2}\s+\d{2}:\d{2}(?::\d{2})?", text))
    pub = soup.find("meta", attrs={"name": "PubDate"})
    if pub is not None:
        clocks.add(pub.get("content", ""))
    parsed_clocks = {pd.Timestamp(clock).tz_localize("Asia/Shanghai") for clock in clocks}
    assert len(parsed_clocks) == 1, "原公布时钟缺失或不唯一。"
    clock = next(iter(parsed_clocks))
    assert clock.strftime("%Y-%m-%d") == item["catalogue_publication_date"], "目录与原文公布日不同。"
    matches = re.findall(r"应收账款平均回收期为(\d+(?:\.\d+)?)天[，,](同比|比上年(?:末)?)(增加|减少|延长|缩短)(\d+(?:\.\d+)?)天", compact)
    values = {(float(level), reference, (-1 if direction in ["减少", "缩短"] else 1) * float(change)) for level, reference, direction, change in matches}
    flat = re.findall(r"应收账款平均回收期为(\d+(?:\.\d+)?)天[，,](同比|与上年(?:末)?)持平", compact)
    values.update((float(level), reference, 0.) for level, reference in flat)
    assert len(values) == 1, "原文收款期限或直接同比未唯一识别。"
    days, reference, change = next(iter(values))
    expected_formula = "应收账款平均回收期=360×平均应收账款÷营业收入×累计月数÷12"
    assert expected_formula in compact.replace("＝", "="), "原文公式口径不符。"
    table_values = []
    for table in soup.find_all("table"):
        heading = re.sub(r"\s+", "", table.get_text())
        if "应收账款平均回收期" not in heading or "营业收入利润率" not in heading:
            continue
        for row in table.find_all("tr"):
            cells = [re.sub(r"\s+", "", cell.get_text()) for cell in row.find_all(["td", "th"], recursive=False)]
            if len(cells) == 9 and cells[0] == "总计" and all(re.fullmatch(r"-?\d+(?:\.\d+)?", v) for v in cells[1:]):
                table_values.append(float(cells[-1]))
    assert set(table_values) == {days}, "正文与同名效益表总计行不一致。"
    return {**item, "published_at": clock.isoformat(), "known_at": (clock.normalize() + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).isoformat(),
        "receivable_collection_days": days, "receivable_days_published_yoy_change": change, "reported_reference": reference,
        "formula": expected_formula, "body_and_total_table_agree": True,
        "raw_path": receipt["raw_path"], "raw_sha256": receipt["sha256"],
        "method_group": "RECEIVABLE_EXCLUDING_NOTES_OPERATING_REVENUE_2020_ONWARD",
        "historical_first_vintage_verified": False}


def acquire():
    assert digest(Path(__file__)) == read(OUT / "freeze.json")["code_sha256"]
    if (OUT / "result.json").exists():
        raise RuntimeError("工业收款月报取得已有终态。")
    catalogue_rows = read(OUT / "catalogue.json")
    cached = {r["receipt"]["url"]: r["receipt"] for r in read(probe.OUT / "source_measurements.json") if r["receipt"]["status"] == "HTTP_OK"}
    records, failed, reused = [], [], []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for start in range(0, len(catalogue_rows), 2):
            items = catalogue_rows[start:start + 2]
            todo = [item for item in items if item["url"] not in cached]
            jobs = [(item["url"], "month_" + item["stat_month"]) for item in todo]
            fresh = {item["url"]: receipt for item, receipt in zip(todo, executor.map(request_job, jobs))}
            for item in items:
                receipt = cached.get(item["url"], fresh.get(item["url"]))
                if item["url"] in cached:
                    reused.append(item["stat_month"])
                try:
                    records.append(parse(item, receipt))
                except (AssertionError, ValueError, KeyError) as exc:
                    failed.append({**item, "receipt": receipt, "error": str(exc)})
            if (len(records) + len(failed)) % 10 == 0:
                print(f"工业收款月报准入{len(records)}份，待核{len(failed)}份。", flush=True)
            if (OUT / "REQUESTS_STOPPED.json").exists():
                break
    save(OUT / "released_records.json", records, True)
    save(OUT / "unresolved_fields.json", failed, True)
    frame = pd.DataFrame(records)
    if len(frame):
        frame["known_at"] = pd.to_datetime(frame.known_at).dt.tz_convert("Asia/Shanghai").dt.as_unit("ns")
        frame.to_parquet(PANEL, index=False)
    missing = sorted(set(EXPECTED) - {r["stat_month"] for r in records})
    save(OUT / "result.json", {"at": now(), "study_id": STUDY,
        "status": "COMPLETE_SAME_DEFINITION_MONTHLY_RECEIVABLE_SOURCES_READY" if not missing else "PARTIAL_SAME_DEFINITION_MONTHLY_RECEIVABLE_SOURCES",
        "expected_months": len(EXPECTED), "admitted_months": len(records), "missing_months": missing,
        "field_or_clock_failures": len(failed), "reused_probe_sources": reused,
        "latest_source": records[-1] if records else None, "new_accounts": 0, "new_strategy_returns": 0,
        "historical_first_vintage_verified": False, "current_market_view": "NO_VIEW",
        "goal_status": "active", "goal_achieved": False, "orders_authorized": False}, True)
    print(f"工业收款来源准入{len(records)}/{len(EXPECTED)}个月，仍缺{len(missing)}个月；尚无新策略收益。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="国家统计局工业企业同口径收款期限月报")
    parser.add_argument("command", choices=["catalogue", "acquire"])
    globals()[parser.parse_args().command]()
