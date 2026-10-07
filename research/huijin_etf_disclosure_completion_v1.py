"""保留初次失败，补齐汇金来源并修正中文ETF检索及跨年度目录解析。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now, digest

PARENT = ROOT / "reports/research/510300_huijin_etf_disclosure_source_v1"
OUT = ROOT / "reports/research/510300_huijin_etf_disclosure_completion_v1"
STUDY = "510300_HUIJIN_ETF_DISCLOSURE_COMPLETION_V1"


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("来源补齐协议已有记录")
    for folder in ["code", "raw", "receipts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    protocol = {"at": now(), "study_id": STUDY,
        "parent_failure": "初次目录有2016请求失败、2021目录一条2025反诈骗声明导致整年解析中断，另4篇文章下载失败。中文紧邻ETF时Unicode单词边界误判，漏标2015-07-08及2025-04-08。没有读取新事件收益。",
        "repair": "复用全部成功响应；仅补请求失败URL及新解析出的目录文章。年度分区与文章日期分开保存，跨年度文章保留并按正文日期解释。直接在正文搜索ETF，不用Unicode单词边界。",
        "transport": "普通HTTPS GET，单次30秒超时，并发2；失败保持，不自动循环。",
        "time_scope": "正文日期与目录一致才可进入事件分类；旧URL路径年份不当作发布时间。现在可见版本不是历史首版。",
        "same_universe": "2012—2026原15个官方年度目录及全部文章，未改变事件选择、账户规则或收益门槛。",
        "source_gate": "全部当前可见目录与文章取得且解析完成，方可进入逐条语义分类。",
        "new_event_returns": 0, "new_model_fits": 0, "new_accounts": 0, "orders_authorized": False, "goal_achieved": False}
    save(OUT / "protocol.json", protocol, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    sources = [PARENT / "source_result.json", PARENT / "official_year_navigation.json", PARENT / "all_documents.json"]
    save(OUT / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)),
         "protocol_sha256": digest(OUT / "protocol.json"),
         "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in sources}}, True)
    print("补齐规则已固定：初次失败保留，ETF中文边界与目录年份单独修正。", flush=True)


def get(url):
    key = hashlib.sha256(url.encode()).hexdigest()[:20]
    old = PARENT / "receipts" / (key + ".json")
    if old.exists():
        record = read(old)
        if record["status"] == "RECEIVED_OFFICIAL_HTML":
            path = PARENT / record["raw_path"]
            assert digest(path) == record["sha256"]
            return {**record, "raw_path": path.relative_to(ROOT).as_posix(), "reused": True}
    own = OUT / "receipts" / (key + ".json")
    if own.exists():
        return read(own)
    record = {"url": url, "started_at": now(), "reused": False}
    try:
        response = requests.get(url, timeout=30)
        path = OUT / "raw" / (key + ".html")
        path.write_bytes(response.content)
        record.update(http_status=response.status_code, raw_path=path.relative_to(ROOT).as_posix(),
                      sha256=digest(path), bytes=len(response.content))
        response.raise_for_status()
        text = BeautifulSoup(response.content, "html.parser").get_text(" ", strip=True)
        if "汇金" not in text or len(text) < 150:
            raise ValueError("响应不是可识别的汇金内容")
        record["status"] = "RECEIVED_OFFICIAL_HTML"
    except (requests.RequestException, ValueError) as exc:
        record.update(status="FAILED_SOURCE_REQUEST", error_type=type(exc).__name__, error=str(exc)[:800])
    record["received_at"] = now()
    save(own, record, True)
    return record


def soup_of(record):
    if record["status"] != "RECEIVED_OFFICIAL_HTML":
        raise ValueError("官方请求未成功")
    path = ROOT / record["raw_path"]
    assert digest(path) == record["sha256"]
    return BeautifulSoup(path.read_bytes(), "html.parser")


def run():
    fixed = read(OUT / "freeze.json")
    assert digest(Path(__file__)) == fixed["code_sha256"]
    assert digest(OUT / "protocol.json") == fixed["protocol_sha256"]
    for name, sha in fixed["sources"].items():
        assert digest(ROOT / name) == sha
    save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    navigation = read(PARENT / "official_year_navigation.json")["years"]
    entries, problems, cross_year, catalogs = {}, [], [], []
    for year in range(2012, 2027):
        url = navigation[str(year)]
        record = get(url)
        catalogs.append(record)
        try:
            soup = soup_of(record)
            raw = (ROOT / record["raw_path"]).read_text(encoding="utf-8", errors="replace")
            assert not any(int(x) > 1 for x in re.findall(r"createPageHTML\s*\(\s*(\d+)", raw)), "存在未处理分页"
            pagination = [a["href"] for a in soup.find_all("a", href=True)
                          if (a.get_text(strip=True) in {"下一页", "下页", "尾页", "Next", "Last"}
                              or (a.get_text(strip=True).isdigit() and len(a.get_text(strip=True)) < 4))
                          and not a["href"].startswith(("#", "javascript:"))]
            assert not pagination, "存在未取得分页链接"
            for node in soup.select(".infor-list-item"):
                day, year_month = node.select_one(".day"), node.select_one(".year")
                date = pd.Timestamp(year_month.get_text(strip=True).replace(".", "-") + "-" + day.get_text(strip=True))
                article = urljoin(url, node.find("a", href=True)["href"])
                row = {"catalog_section_year": year, "catalog_date": str(date.date()), "title": node.select_one("h1").get_text(" ", strip=True),
                       "url": article, "catalog_url": url, "catalog_sha256": record["sha256"], "cross_year_catalog_record": date.year != year}
                if date.year != year:
                    cross_year.append(row)
                if article in entries and entries[article]["catalog_date"] != row["catalog_date"]:
                    raise ValueError("同一文章在两个目录日期不同")
                entries[article] = row
        except (ValueError, AttributeError, AssertionError) as exc:
            problems.append({"year": year, "url": url, "error": str(exc)})
    documents = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {pool.submit(get, row["url"]): row for row in entries.values()}
        for number, future in enumerate(as_completed(pending), 1):
            row, record = pending[future], future.result()
            item = {**row, "receipt": record}
            try:
                soup = soup_of(record)
                body = soup.select_one(".infor-txt-con")
                if body is None:
                    raise ValueError("缺少正文容器")
                date_node = body.find("span", recursive=False)
                if date_node is None:
                    raise ValueError("正文发布日期缺失")
                published = str(pd.Timestamp(date_node.get_text(strip=True)).date())
                body_text = body.get_text("\n", strip=True)
                if published != row["catalog_date"]:
                    raise ValueError("正文发布日期与目录不一致")
                item.update(status="DOCUMENT_PARSED", publication_date=published, body_text=body_text,
                            contains_etf=bool(re.search(r"ETF|交易型开放式指数基金", body_text, re.I)))
            except (ValueError, AttributeError) as exc:
                item.update(status="DOCUMENT_UNRESOLVED", error=str(exc), contains_etf=None)
            documents.append(item)
            if number % 30 == 0 or number == len(entries):
                print(f"正文与日期已核对{number}/{len(entries)}篇，成功旧响应直接复用。", flush=True)
    documents.sort(key=lambda row: (row["catalog_date"], row["url"]))
    candidates = [row for row in documents if row["contains_etf"]]
    failures = [row for row in documents if row["status"] != "DOCUMENT_PARSED"]
    save(OUT / "all_documents.json", documents, True)
    save(OUT / "etf_documents_for_classification.json", candidates, True)
    save(OUT / "catalog_receipts.json", catalogs, True)
    new_requests = len(list((OUT / "receipts").glob("*.json")))
    result = {"at": now(), "study_id": STUDY,
              "status": "CURRENT_OFFICIAL_CATALOG_COMPLETE_CLASSIFICATION_PENDING" if not (problems or failures) else "SOURCE_COVERAGE_INCOMPLETE",
              "catalog_years": list(range(2012, 2027)), "catalog_articles": len(documents),
              "parsed_articles": len(documents)-len(failures), "etf_text_articles": len(candidates),
              "cross_year_catalog_records": cross_year, "catalog_problems": problems,
              "unresolved_articles": [{"url": row["url"], "error": row["error"]} for row in failures],
              "supplementary_requests": new_requests, "historical_first_vintage_verified": False,
              "new_event_returns": 0, "new_model_fits": 0, "new_accounts": 0, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "source_result.json", result, True)
    print(f"来源补齐结束：{result['parsed_articles']}/{len(documents)}篇正文可读，ETF相关{len(candidates)}篇。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="汇金官方来源的定向补齐。")
    parser.add_argument("command", choices=["freeze", "run"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run}[args.command]()
