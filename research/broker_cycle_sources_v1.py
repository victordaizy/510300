"""保存公开券商原报告及作者团队转载，单次收集，不生成交易结果。"""
from __future__ import annotations

import hashlib
import io
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pdfplumber
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_broker_cycle_inspiration_v1"
SOURCES = (
    ("CMS_20260601", "招商证券", "主线·资金·博弈·周期：A股投资的底层逻辑与系统奥义", "2026-06-01",
     "https://finance.sina.cn/2026-06-03/detail-iniaecsk0758056.d.html", "AUTHOR_TEAM_REPRINT"),
    ("HTSC_20190927", "华泰证券", "波动率与换手率构造牛熊指标", "2019-09-27",
     "https://crm.htsc.com.cn/doc/2019/10750101/465187f1-0ae9-416e-b817-65d786a2e0d2.pdf", "BROKER_ORIGINAL_PDF"),
    ("HTSC_20200407", "华泰证券", "牛熊指标在择时轮动中的应用探讨", "2020-04-07",
     "https://crm.htsc.com.cn/doc/2020/10750101/9ae949c5-07ef-4057-a8c9-84ab3047835f.pdf", "BROKER_ORIGINAL_PDF"),
    ("HTSC_20201103", "华泰证券", "权益周期持续上行，月线或偏震", "2020-11-03",
     "https://crm.htsc.com.cn/doc/2020/10750404/0f44af09-c0f1-48fd-bac2-313926e0deeb.pdf", "BROKER_ORIGINAL_PDF"),
    ("GJ_MAINLINE", "国金证券", "A股主线周期律：何以成为市场主线", None,
     "https://data.eastmoney.com/report/zw_strategy.jshtml?encodeUrl=jSwLI%2F3Kx+g4q8z680%2FivVAfkb66w7E4gJQVoeP5cTs%3D", "AUTHOR_REPORT_EXCERPT_WITH_PDF_LINK"),
    ("HC_20230824", "华创证券", "筹码博弈系列2：往人少的地方走：券商+顺周期", "2023-08-24",
     "https://www.fxbaogao.com/detail/3861133", "DISTRIBUTOR_REPORT_TEXT_NEEDS_AUTHOR_VERIFICATION"),
    ("CMS_20250626", "招商证券", "A股进入牛市II阶段：成因和方向", "2025-06-26",
     "https://premium-wscn.awtmt.com/3aa85b3c-ce9c-4f54-bd31-8f18c3447300.pdf", "ORIGINAL_BROKER_PDF_ON_DISTRIBUTOR"),
    ("XY_20240602", "兴业证券", "三大指标看本轮调整的位置", "2024-06-02",
     "https://finance.sina.com.cn/stock/bxjj/2024-06-02/doc-inaxityx8553467.shtml", "AUTHOR_REPORT_REPRINT_NEEDS_AUTHOR_VERIFICATION"),
)


def stamp():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def fetch(source):
    sid, broker, title, date, url, provenance = source
    folder = OUT / "sources" / sid
    folder.mkdir(parents=True, exist_ok=False)
    receipt = {"source_id": sid, "broker": broker, "title": title, "report_date": date,
               "requested_url": url, "provenance_pending_body_check": provenance,
               "started_at": stamp(), "retry_count": 0, "first_vintage": "NOT_CERTIFIED",
               "role": "RESEARCH_INSPIRATION_NOT_HISTORICAL_PREDICTOR"}
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    try:
        response = session.get(url, timeout=(10, 25))
        receipt.update(http_status=response.status_code, final_url=response.url,
                       headers={key: response.headers.get(key) for key in ("Content-Type", "Date", "Last-Modified")})
        response.raise_for_status()
        raw = response.content
        if len(raw) > 20_000_000:
            raise ValueError("公开报告超过本次20MB单文件上限。")
        pdf = raw.startswith(b"%PDF")
        raw_path = folder / ("original.pdf" if pdf else "original.html")
        raw_path.write_bytes(raw)
        receipt.update(raw_path=str(raw_path.absolute().relative_to(ROOT)), bytes=len(raw),
                       sha256=hashlib.sha256(raw).hexdigest())
        if pdf:
            pages = []
            with pdfplumber.open(io.BytesIO(raw)) as document:
                for number, page in enumerate(document.pages, start=1):
                    pages.append({"page": number, "text": page.extract_text() or ""})
            write_json(folder / "pages.json", pages)
            text = "\n\n".join(f"【PDF第{page['page']}页】\n{page['text']}" for page in pages)
            receipt["pages"] = len(pages)
        else:
            response.encoding = response.apparent_encoding or "utf-8"
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text("\n", strip=True)
            links = [{"text": anchor.get_text(" ", strip=True), "url": urljoin(response.url, anchor["href"])}
                     for anchor in soup.find_all("a", href=True)
                     if ".pdf" in anchor["href"].lower() or "pdf原文" in anchor.get_text().lower()]
            receipt["public_pdf_links"] = links
        (folder / "text.txt").write_text(text, encoding="utf-8")
        receipt.update(status="SAVED_BODY_PENDING_AUTHOR_AND_METHOD_REVIEW", text_characters=len(text))
    except Exception as error:
        receipt.update(status="SOURCE_REQUEST_OR_EXTRACTION_FAILED", error_type=type(error).__name__, error=str(error))
    finally:
        session.close()
        receipt["ended_at"] = stamp()
        write_json(folder / "receipt.json", receipt)
    return receipt


def main():
    if (OUT / "source_collection_receipt.json").exists() or (OUT / "source_collection_protocol.json").exists():
        raise RuntimeError("本次来源集合已登记或收集，不覆盖保存响应。")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "source_collection_protocol.json", {
        "at": stamp(), "user_direction": "市场变化时允许另立适应性策略，旧结果保留，先读券商原研究找机制",
        "sources": [{"id": row[0], "broker": row[1], "title": row[2], "date": row[3], "url": row[4], "provenance": row[5]} for row in SOURCES],
        "one_request_per_source": True, "maximum_workers": 4, "finance_runs": 0,
        "restriction": "不使用这些现时收集的报告给历史日线填事前评分；不以报告历史收益证明本地策略有效。",
        "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(fetch, SOURCES))
    write_json(OUT / "source_collection_receipt.json", {
        "at": stamp(), "requests": len(receipts), "success": sum(row["status"].startswith("SAVED") for row in receipts),
        "results": receipts, "new_market_bars": 0, "new_labels": 0, "new_fits": 0, "new_accounts": 0})
    for receipt in receipts:
        print(f"{receipt['broker']} {receipt['source_id']}：{receipt['status']}，保存字符{receipt.get('text_characters', 0)}。", flush=True)


if __name__ == "__main__":
    main()
