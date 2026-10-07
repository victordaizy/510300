"""按七个已明确变更对象补原方案，保存原响应，不进入价格或收益研究。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import threading
import time

import pypdfium2 as pdfium
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_repurchase_missing_originals_v1"
PRIOR = ROOT / "reports/research/510300_factor96_repurchase_change_chain_v1"
QUERY = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}
STOP = threading.Event()
SHANGHAI = timezone(timedelta(hours=8))


def now():
    return datetime.now(SHANGHAI).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while data := stream.read(1024 * 1024):
            value.update(data)
    return value.hexdigest()


def prepare():
    assert not (OUT / "protocol.json").exists()
    for folder in ["inputs", "code", "raw", "receipts", "text", "queries", "documents", "local_sources/raw", "local_sources/text", "local_sources/receipts"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    for name in ["identified_missing_originals.json", "change_ledger.json", "review_cards.json", "result.json", "protocol.json"]:
        shutil.copyfile(PRIOR / name, OUT / "inputs" / ("prior_" + name))
    for name in ["metadata.json", "plan_nodes.json"]:
        shutil.copyfile(PRIOR / "inputs" / name, OUT / "inputs" / name)
    for source, name in [(ROOT / "reports/research/510300_factor96_program_v1/status.json", "program_before.json"),
                         (ROOT / "config/510300_existing_data_training_mandate_v1.json", "mandate_before.json"),
                         (ROOT / "reports/research/510300_corporate_repurchase_public_completion_v1/cninfo_stock_directory.json", "stock_directory.json"),
                         (ROOT / "reports/research/510300_corporate_repurchase_original_plan_completion_v1/results/catalogues.json", "previous_original_catalogues.json")]:
        shutil.copyfile(source, OUT / "inputs" / name)
    targets = read(OUT / "inputs/prior_identified_missing_originals.json")
    metas = read(OUT / "inputs/metadata.json")
    directory = {r["code"]: r for r in read(OUT / "inputs/stock_directory.json")["stockList"] if r["category"] == "A股"}
    windows, local_check, local_ids = [], [], set()
    for target in targets:
        symbol, date = target["symbol"], target["target"]["original_approval_date"]
        day = datetime.fromisoformat(date)
        matches = [m for m in metas if m["symbol"] == symbol and any(d[:10] == date for d in m["original_board_dates"])]
        originals = [m for m in matches if m["plan_kind"] == "ORIGINAL_PLAN_DOCUMENT"]
        assert not originals
        follow = [m for m in matches if datetime.fromisoformat(m["known_at"]) >= datetime.fromisoformat(target["known_at"])]
        local_ids.add(target["change_document_id"])
        local_ids.update(m["document_id"] for m in follow)
        local_check.append({"symbol": symbol, "original_approval_date": date,
                            "matching_document_ids": [m["document_id"] for m in matches],
                            "existing_originals": [], "following_document_ids": [m["document_id"] for m in follow]})
        windows.append({"key": symbol + "_" + date.replace("-", ""), "symbol": symbol,
                        "change_document_id": target["change_document_id"], "original_approval_date": date,
                        "org_id": directory[symbol[:6]]["orgId"], "column": "sse" if symbol.endswith(".SH") else "szse",
                        "start": (day - timedelta(days=1)).date().isoformat(), "end": (day + timedelta(days=7)).date().isoformat()})
    old_queries = read(OUT / "inputs/previous_original_catalogues.json")
    for window in windows:
        assert not any(q["window"]["symbol"] == window["symbol"] and q["window"]["start"] <= window["original_approval_date"] <= q["window"]["end"] for q in old_queries)
    docs = {r["document_id"]: r for r in read(ROOT / "reports/research/510300_corporate_repurchase_index_documents_v1/documents.json")}
    sources = []
    for key in sorted(local_ids):
        row = docs[key]
        record = {"document_id": key, "symbol": row["symbol"], "title": row["title"],
                  "source_url": row["source_url"], "catalogue_date": row["catalogue_date"], "reused": True}
        for kind, suffix in [("raw", ".pdf"), ("text", ".json"), ("receipts", ".json")]:
            field = "receipt_path" if kind == "receipts" else kind + "_path"
            source, target = ROOT / row[field], OUT / "local_sources" / kind / (key + suffix)
            assert not target.exists()
            if kind == "raw":
                assert digest(source) == row["raw_sha256"]
            shutil.copyfile(source, target)
            record[kind + "_path"] = target.relative_to(OUT).as_posix()
            record[kind + "_sha256"] = digest(target)
        sources.append(record)
    save(OUT / "local_source_documents.json", sources)
    save(OUT / "local_search_evidence.json", local_check)
    protocol = {"at": now(), "study_id": "510300_FACTOR96_REPURCHASE_MISSING_ORIGINALS_V1", "windows": windows,
                "scope": "仅按上一轮逐条确认的七个目标原批准日补原方案；同时复用同公司、同批准日、变更时钟之后的现有公告，不按价格或收益挑选。",
                "query_rule": "批准日前1日至后7日，巨潮回购关键词，每页30条、最多3页；总量变化或缺页不称完整。传输异常最多补1次，403或429后停止新请求。",
                "pdf_selection": "标题含回购且含方案、报告书或以回购公司股份的公告结尾；排除明确限制性股票、法律意见、独立董事、监事会、提议和持股明细。其余目录全部保留而不自动认作原方案。",
                "clock_rule": "新原文按原公告名义日期保存，2026抓取收据另列；首次历史发布版本未证明，不准入交易因子、不把原方案追认到旧冻结台账中。",
                "follow_up_rule": "后续批准、注销登记或使用完毕只在其公开时钟之后记录；同原批准日只是候选，仍需原文对象和数量对应。",
                "return_test": "NOT_RUN", "free_float_denominator": "MISSING_UNCHANGED", "new_accounts": 0,
                "goal_achieved": False, "orders_authorized": False, "local_sources": len(sources)}
    save(OUT / "protocol.json", protocol)
    shutil.copyfile(Path(__file__), OUT / "code" / Path(__file__).name)
    files = [{"path": p.relative_to(OUT).as_posix(), "bytes": p.stat().st_size, "sha256": digest(p)}
             for p in sorted(OUT.rglob("*")) if p.is_file()]
    save(OUT / "source_request_freeze.json", {"at": now(), "files": files, "code_sha256": digest(Path(__file__))})
    print(f"已固定{len(windows)}个缺失原方案窗口，复用{len(sources)}份变更及后续原文。", flush=True)


def acquire(key, url, payload=None):
    for attempt in [1, 2]:
        receipt_path = OUT / "receipts" / (key + f"_{attempt}.json")
        if receipt_path.exists():
            receipt = read(receipt_path)
        elif STOP.is_set():
            receipt = {"at": now(), "status": "NOT_REQUESTED_AFTER_SOURCE_LIMIT", "source_url": url}
            save(receipt_path, receipt)
        else:
            receipt = {"requested_at": now(), "source_url": url, "payload": payload, "attempt": attempt}
            try:
                response = requests.get(url, headers=HEADERS, timeout=(10, 25)) if payload is None else requests.post(url, data=payload, headers=HEADERS, timeout=(10, 25))
                content = response.content
                raw = OUT / "raw" / (key + f"_{attempt}" + (".pdf" if content.startswith(b"%PDF") else ".bin"))
                assert not raw.exists()
                raw.write_bytes(content)
                receipt.update(status="HTTP_OK" if response.status_code == 200 else "HTTP_ERROR", http_status=response.status_code,
                               raw_path=raw.relative_to(OUT).as_posix(), sha256=digest(raw), bytes=len(content))
                if response.status_code in [403, 429]:
                    STOP.set()
            except requests.RequestException as exc:
                receipt.update(status="REQUEST_FAILED", error_type=type(exc).__name__)
            receipt["completed_at"] = now()
            save(receipt_path, receipt)
            time.sleep(0.35)
        if receipt["status"] != "REQUEST_FAILED" or attempt == 2:
            return receipt, receipt_path.relative_to(OUT).as_posix()


def query(window):
    destination = OUT / "queries" / (window["key"] + ".json")
    if destination.exists():
        return read(destination)
    rows, total, status, receipts = [], None, "PAGE_LIMIT_REACHED", []
    for page in range(1, 4):
        payload = {"pageNum": str(page), "pageSize": "30", "column": window["column"], "tabName": "fulltext",
                   "stock": window["symbol"][:6] + "," + window["org_id"], "searchkey": "回购", "secid": "", "category": "", "trade": "",
                   "seDate": window["start"] + "~" + window["end"], "sortName": "time", "sortType": "desc", "isHLtitle": "false"}
        receipt, path = acquire(window["key"] + f"_page{page}", QUERY, payload)
        receipts.append(path)
        if receipt["status"] != "HTTP_OK":
            status = receipt["status"]
            break
        try:
            body = json.loads((OUT / receipt["raw_path"]).read_bytes())
            count = int(body["totalAnnouncement"])
            if total is not None and total != count:
                status = "QUERY_TOTAL_CHANGED"
                break
            total = count
            batch = body.get("announcements") or []
            for entry in batch:
                assert entry["secCode"] == window["symbol"][:6] and entry["orgId"] == window["org_id"]
                title = re.sub("<[^>]+>", "", entry["announcementTitle"])
                date = datetime.fromtimestamp(entry["announcementTime"] / 1000, SHANGHAI)
                assert window["start"] <= date.date().isoformat() <= window["end"]
                assert "回购" in title and entry["adjunctUrl"].startswith("finalpage/")
                rows.append({"symbol": window["symbol"], "document_id": str(entry["announcementId"]), "title": title,
                             "catalogue_date": date.date().isoformat(), "catalogue_timestamp": date.isoformat(),
                             "source_url": "https://static.cninfo.com.cn/" + entry["adjunctUrl"]})
            if len({r["document_id"] for r in rows}) != len(rows):
                status = "DUPLICATE_PAGINATION_ROWS"
                break
            if len(rows) == total:
                status = "COMPLETE_QUERY"
                break
            if not batch:
                status = "MISSING_PAGE"
                break
        except (ValueError, KeyError, AssertionError, TypeError) as exc:
            status = "RESPONSE_NOT_ADMITTED_" + type(exc).__name__
            break
    result = {"window": window, "status": status, "total": total, "rows": rows, "receipts": receipts}
    save(destination, result)
    print(f"日期窗口{window['key']}：{status}，取得{len(rows)}条目录。", flush=True)
    return result


def selected(title):
    title = re.sub(r"\s+", "", title)
    forbidden = ["限制性", "法律意见", "独立董事", "监事会", "提议", "持股明细"]
    return "回购" in title and not any(w in title for w in forbidden) and ("方案" in title or "报告书" in title or re.search(r"回购(?:公司)?股份的公告$", title) is not None)


def collect():
    freeze = read(OUT / "source_request_freeze.json")
    assert digest(Path(__file__)) == freeze["code_sha256"]
    for item in freeze["files"]:
        assert digest(OUT / item["path"]) == item["sha256"]
    assert not (OUT / "source_result.json").exists()
    protocol = read(OUT / "protocol.json")
    with ThreadPoolExecutor(max_workers=3) as pool:
        catalogues = list(pool.map(query, protocol["windows"]))
    save(OUT / "catalogues.json", catalogues)
    candidates = [r for c in catalogues for r in c["rows"] if selected(r["title"])]
    assert len({r["document_id"] for r in candidates}) == len(candidates)
    save(OUT / "selected_pdf_targets.json", candidates)
    requests_by_id = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(acquire, "pdf_" + row["document_id"], row["source_url"]): row for row in candidates}
        for future in as_completed(futures):
            row = futures[future]
            requests_by_id[row["document_id"]] = future.result()
    results = []
    for row in candidates:
        receipt, path = requests_by_id[row["document_id"]]
        record = {**row, "receipt_path": path, "status": receipt["status"], "raw_path": receipt.get("raw_path"),
                  "raw_sha256": receipt.get("sha256"), "text_path": None, "historical_first_publication_verified": False,
                  "trading_feature_admitted": False}
        if receipt["status"] == "HTTP_OK" and (OUT / receipt["raw_path"]).read_bytes().startswith(b"%PDF"):
            pdf = pdfium.PdfDocument(OUT / receipt["raw_path"])
            pages = []
            try:
                for page in pdf:
                    textpage = page.get_textpage()
                    try:
                        pages.append(textpage.get_text_range())
                    finally:
                        textpage.close()
                        page.close()
            finally:
                pdf.close()
            target = OUT / "text" / (row["document_id"] + ".json")
            save(target, pages)
            record.update(status="PDF_TEXT_SAVED" if len("".join(pages).strip()) >= 50 else "PDF_NO_USABLE_TEXT",
                          text_path=target.relative_to(OUT).as_posix(), text_sha256=digest(target), pages=len(pages),
                          code_in_first_page=row["symbol"][:6] in pages[0])
        elif receipt["status"] == "HTTP_OK":
            record["status"] = "HTTP_OK_NOT_PDF"
        save(OUT / "documents" / (row["document_id"] + ".json"), record)
        results.append(record)
    save(OUT / "documents.json", results)
    receipt_rows = [read(p) for p in sorted((OUT / "receipts").glob("*.json"))]
    result = {"at": now(), "query_windows": len(catalogues), "complete_query_windows": sum(c["status"] == "COMPLETE_QUERY" for c in catalogues),
              "catalogue_rows": sum(len(c["rows"]) for c in catalogues), "selected_documents": len(results),
              "complete_pdf_texts": sum(r["status"] == "PDF_TEXT_SAVED" for r in results),
              "new_http_requests": sum("requested_at" in r for r in receipt_rows),
              "raw_originals_are_confirmed_roots": False, "purpose_lifecycle_complete": False,
              "new_accounts": 0, "T12": "NOT_RUN", "goal_achieved": False, "goal_status": "active"}
    save(OUT / "source_result.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="七个具体原方案缺口的有限来源查询")
    parser.add_argument("action", choices=["prepare", "collect"])
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect}[args.action]()
