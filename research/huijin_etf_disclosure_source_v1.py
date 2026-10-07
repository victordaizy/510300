"""取得中央汇金官方年度目录及全部文章，先建立ETF买卖披露母集，不读取事件收益。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now, digest

OUT = ROOT / "reports/research/510300_huijin_etf_disclosure_source_v1"
STUDY = "510300_HUIJIN_ETF_DISCLOSURE_SOURCE_V1"
INDEX = "https://www.huijin-inv.cn/huijin-inv/SC20252/Information_Center.shtml"
YEARS = list(range(2012, 2027))


def freeze():
    if (OUT / "freeze.json").exists():
        raise RuntimeError("官方目录采集已固定，不覆盖记录")
    for name in ["code", "raw", "receipts"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    protocol = {"at": now(), "study_id": STUDY, "user_authority": "reports/research/510300_new_evidence_resume_20260925/authority_update.json",
        "question": "补足此前政策研究缺失的单一发布机构ETF购买、出售及相关表态的公开披露目录。",
        "publisher": "中央汇金投资有限责任公司官方网站", "catalog_years": YEARS, "index_url": INDEX,
        "universe": "由官方年度导航取得2012—2026全部年度列表，并保存列表内全部文章，不只搜索ETF增持关键词。2026仅至本次取得日。",
        "admission_scope": "覆盖结论只限本次官网可见年度目录及所列页面，不等于所有历史版本、全国全部政策事件或未公开交易。",
        "classification": "先保存所有原文，含ETF或交易型开放式指数基金的文稿进入人工逐条判读；买入、卖出、继续支持、持仓回顾及无新增事实分别保留，不由后续价格决定。",
        "clock": "保存官网发布日期、目录日期、正文涉及的经济日期和实际取得时刻。仅有日期的披露，其历史可用上界按当日23:59:59；不以经济交易日期倒填公开时间。未验证历史首版。",
        "failure": "任何年度目录缺失、分页未穷尽或已列文章下载失败，保留为覆盖缺口；不把无资料日期当作无事件。",
        "transport": "普通HTTPS GET，15秒连接读取超时，3个并发；每URL单次请求，有失败单独记录，不自动循环重试。",
        "before_event_returns": True, "new_model_fits": 0, "new_accounts": 0,
        "macro_not_flow": "披露增持ETF不等于披露510300份额、具体买入日或金额，也不等于保证股价不下跌。",
        "prior_work": "旧equity_support_policy_event_v1只有7条跨机构政策链事实及6公告日。本轮首先补单一机构完整目录，旧20日账户失败保留。",
        "orders_authorized": False, "goal_achieved": False}
    save(OUT / "protocol.json", protocol, True)
    shutil.copy2(__file__, OUT / "code" / Path(__file__).name)
    save(OUT / "freeze.json", {"at": now(), "code_sha256": digest(Path(__file__)),
                               "protocol_sha256": digest(OUT / "protocol.json")}, True)
    print("汇金年度全目录采集范围已固定；尚未计算事件后收益。", flush=True)


def fetch(url):
    """一条URL一份原始响应和回执，独立线程写入不同文件。"""
    name = hashlib.sha256(url.encode()).hexdigest()[:20]
    receipt_file = OUT / "receipts" / (name + ".json")
    if receipt_file.exists():
        return read(receipt_file)
    receipt = {"url": url, "started_at": now(), "status": "REQUEST_STARTED"}
    try:
        response = requests.get(url, timeout=15)
        path = OUT / "raw" / (name + ".html")
        path.write_bytes(response.content)
        receipt.update(http_status=response.status_code, final_url=response.url,
                       raw_path=path.relative_to(OUT).as_posix(), sha256=digest(path), bytes=len(response.content))
        response.raise_for_status()
        text = BeautifulSoup(response.content, "html.parser").get_text(" ", strip=True)
        if "汇金" not in text or len(text) < 150:
            raise ValueError("响应不是可识别的汇金内容")
        receipt["status"] = "RECEIVED_OFFICIAL_HTML"
    except (requests.RequestException, ValueError) as exc:
        receipt.update(status="FAILED_SOURCE_REQUEST", error_type=type(exc).__name__, error=str(exc)[:800])
    receipt["received_at"] = now()
    save(receipt_file, receipt, True)
    return receipt


def soup_of(receipt):
    if receipt["status"] != "RECEIVED_OFFICIAL_HTML":
        raise ValueError("来源没有成功取得：" + receipt["url"])
    return BeautifulSoup((OUT / receipt["raw_path"]).read_bytes(), "html.parser")


def catalogue(receipt, year):
    soup = soup_of(receipt)
    entries = []
    for node in soup.select(".infor-list-item"):
        day, year_month, title, link = node.select_one(".day"), node.select_one(".year"), node.select_one("h1"), node.find("a", href=True)
        if not all([day, year_month, title, link]):
            raise ValueError("年度目录行缺日期、标题或链接")
        date = pd.Timestamp(year_month.get_text(strip=True).replace(".", "-") + "-" + day.get_text(strip=True))
        if date.year != year:
            raise ValueError("目录年度与文章日期不同")
        entries.append({"year": year, "catalog_date": str(date.date()), "title": title.get_text(" ", strip=True),
                        "summary": node.select_one("p").get_text(" ", strip=True) if node.select_one("p") else "",
                        "url": urljoin(receipt["url"], link["href"]), "catalog_url": receipt["url"],
                        "catalog_sha256": receipt["sha256"]})
    raw = (OUT / receipt["raw_path"]).read_text(encoding="utf-8", errors="replace")
    pagination = []
    for a in soup.find_all("a", href=True):
        label = a.get_text(" ", strip=True)
        if label in {"下一页", "下页", "尾页", "Next", "Last"} or (label.isdigit() and len(label) < 4):
            href = a["href"]
            if href and not href.startswith(("#", "javascript:")):
                pagination.append(urljoin(receipt["url"], href))
    # 静态JS分页也须显式检查，不能因页面存在文章便宣称已穷尽。
    declared_pages = [int(x) for x in re.findall(r"createPageHTML\s*\(\s*(\d+)", raw)]
    if any(count > 1 for count in declared_pages) and not pagination:
        raise ValueError("存在多页JS目录，当前没有取得下一页链接")
    return entries, sorted(set(pagination))


def collect():
    frozen = read(OUT / "freeze.json")
    assert digest(Path(__file__)) == frozen["code_sha256"]
    assert digest(OUT / "protocol.json") == frozen["protocol_sha256"]
    if (OUT / "source_result.json").exists():
        raise RuntimeError("采集已有终态，不重复请求")
    if not (OUT / "RUN_STARTED.json").exists():
        save(OUT / "RUN_STARTED.json", {"at": now()}, True)
    initial = fetch(INDEX)
    soup = soup_of(initial)
    navigation = {int(a.get_text(strip=True)): urljoin(INDEX, a["href"])
                  for a in soup.find_all("a", href=True) if a.get_text(strip=True).isdigit() and len(a.get_text(strip=True)) == 4}
    missing_years = [year for year in YEARS if year not in navigation]
    save(OUT / "official_year_navigation.json", {"at": now(), "years": navigation, "missing_requested_years": missing_years}, True)
    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(fetch, navigation[year]): year for year in YEARS if year in navigation}
        for future in as_completed(pending):
            year = pending[future]
            results[year] = future.result()
            print(f"已读取{year}年度目录：{results[year]['status']}。", flush=True)
    entries, problems, catalog_receipts = [], [], []
    for year in sorted(results):
        queue, visited = [results[year]], set()
        while queue:
            receipt = queue.pop(0)
            if receipt["url"] in visited:
                continue
            visited.add(receipt["url"])
            catalog_receipts.append(receipt)
            try:
                rows, next_pages = catalogue(receipt, year)
                entries.extend(rows)
                queue.extend(fetch(url) for url in next_pages if url not in visited)
            except ValueError as exc:
                problems.append({"year": year, "url": receipt["url"], "error": str(exc)})
    unique = {}
    for row in entries:
        if row["url"] in unique and unique[row["url"]]["catalog_date"] != row["catalog_date"]:
            raise ValueError("相同文章在目录中日期冲突")
        unique[row["url"]] = row
    entries = sorted(unique.values(), key=lambda row: (row["catalog_date"], row["url"]))
    save(OUT / "catalog_entries.json", entries, True)
    documents = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        pending = {pool.submit(fetch, row["url"]): row for row in entries}
        for number, future in enumerate(as_completed(pending), 1):
            row, receipt = pending[future], future.result()
            record = {**row, "receipt": receipt}
            if receipt["status"] == "RECEIVED_OFFICIAL_HTML":
                soup = soup_of(receipt)
                text = soup.get_text("\n", strip=True)
                record["text"] = text
                record["contains_etf"] = bool(re.search(r"\bETF\b|交易型开放式指数基金", text, flags=re.I))
                record["status"] = "DOCUMENT_SAVED_NEEDS_CLASSIFICATION" if record["contains_etf"] else "DOCUMENT_SAVED_NO_ETF_TEXT"
            else:
                record.update(status="ARTICLE_SOURCE_MISSING", contains_etf=None)
            documents.append(record)
            if number % 10 == 0 or number == len(entries):
                print(f"已取得目录文章{number}/{len(entries)}篇；未读取后续收益。", flush=True)
    documents.sort(key=lambda row: (row["catalog_date"], row["url"]))
    save(OUT / "all_documents.json", documents, True)
    candidates = [row for row in documents if row["contains_etf"]]
    save(OUT / "etf_documents_for_classification.json", candidates, True)
    failures = [row["url"] for row in documents if row["status"] == "ARTICLE_SOURCE_MISSING"]
    result = {"at": now(), "study_id": STUDY,
              "status": "CURRENT_OFFICIAL_CATALOG_SAVED_CLASSIFICATION_PENDING" if not (missing_years or problems or failures) else "SOURCE_COVERAGE_INCOMPLETE",
              "requested_years": YEARS, "catalog_pages": len(catalog_receipts), "catalog_articles": len(entries),
              "successful_articles": len(documents)-len(failures), "etf_text_articles": len(candidates),
              "missing_years": missing_years, "catalog_problems": problems, "failed_article_urls": failures,
              "historical_first_vintage_verified": False, "new_event_returns": 0, "new_model_fits": 0,
              "new_accounts": 0, "goal_achieved": False, "orders_authorized": False}
    save(OUT / "source_result.json", result, True)
    print(f"官方目录保存完成：{len(entries)}篇文章，ETF文本候选{len(candidates)}篇，失败{len(failures)}篇。", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="汇金ETF披露官方全目录来源。")
    parser.add_argument("command", choices=["freeze", "collect"])
    options = parser.parse_args()
    {"freeze": freeze, "collect": collect}[options.command]()
