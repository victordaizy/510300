"""取得历史指数成分的回购计划与执行原文，保留来源失败及复用记录。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
import threading
import time

import pandas as pd
import pypdfium2 as pdfium
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now, digest
import research.corporate_repurchase_index_catalogue_v1 as catalogue
import research.corporate_repurchase_public_completion_v1 as pilot

OUT = ROOT / "reports/research/510300_corporate_repurchase_index_documents_v1"
STUDY = "510300_CORPORATE_REPURCHASE_INDEX_DOCUMENTS_V1"
CATALOGUE = catalogue.OUT / "transport_completion/results/catalogue.parquet"
CATEGORIES = ["EXECUTION_DISCLOSURE_CANDIDATE", "PLAN_OR_OTHER_REPURCHASE_DOCUMENT"]
STOP = threading.Event()


def freeze():
    for name in ["code", "raw", "receipts", "documents", "text"]:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(CATALOGUE)
    frame = frame[frame.title_category.isin(CATEGORIES)].sort_values(["symbol", "catalogue_date", "document_id"])
    assert frame.document_id.is_unique
    assert frame.company_catalogue_complete.all()
    assert frame.source_url.str.startswith("https://static.cninfo.com.cn/finalpage/").all()
    originals = read(pilot.OUT / "results/resolved_documents.json")
    reusable = {row["document_id"] for row in originals if row["status"] == "PDF_TEXT_SAVED"}
    protocol = {"at": now(), "study_id": STUDY, "period": [catalogue.START, catalogue.END],
        "question": "从已取得完整目录进入发行人原文，建立指数实际回购需求的可解析来源，不把标题直接用作资金变量。",
        "source_categories": CATEGORIES, "source_documents": len(frame),
        "companies": frame.symbol.nunique(), "reusable_documents": int(frame.document_id.isin(reusable).sum()),
        "selection": "纳入已固定历史并集目录中的全部计划或执行候选；不按未来收益、回购金额、方案成败筛选。暂不下载限制性注销、持股清单、债权人和价格调整类目录。",
        "scope_limit": "标题分类仍可能漏掉正文相关信息；取得全部选定原文不等于已证明全指数实际回购覆盖。A/H、限制性股份与多方案必须在事实阶段再分开。",
        "source": "巨潮完整目录给出的发行人公开PDF地址，已保存试点原文直接复用。",
        "requests": "3个并发下载，线程请求后至少间隔0.3秒；传输异常最多追加1次。403/429停止发新请求，失败不填零。",
        "parsing": "PDFium在主线程逐文提取，原始PDF和逐页文本一起保存；无文本不当成无回购。",
        "new_returns_loaded": False, "new_accounts": 0, "new_fits": 0,
        "goal_achieved": False, "orders_authorized": False,
        "sources": {p.relative_to(ROOT).as_posix(): digest(p) for p in [CATALOGUE, pilot.OUT / "results/resolved_documents.json"]},
        "code_sha256": digest(Path(__file__))}
    save(OUT / "protocol.json", protocol, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    frame.to_parquet(OUT / "targets.parquet", index=False)
    print(f"已固定原文范围：{len(frame)}份、{frame.symbol.nunique()}家公司，{protocol['reusable_documents']}份复用已有原文。", flush=True)


def request_pdf(row):
    record = dict(row)
    key = row["symbol"] + "_" + row["document_id"]
    record.update(key=key, reused=False, status="NOT_REQUESTED", raw_path=None, text_path=None)
    for attempt in [1, 2]:
        target = OUT / "receipts" / f"{key}_{attempt}.json"
        receipt = None
        if target.exists():
            receipt = read(target)
        elif STOP.is_set():
            record["status"] = "NOT_REQUESTED_AFTER_SOURCE_LIMIT"
            break
        else:
            receipt = {"requested_at": now(), "source_url": row["source_url"], "attempt": attempt}
            try:
                response = requests.get(row["source_url"], headers=pilot.HEADERS, timeout=(10, 25))
                content = response.content
                is_pdf = content.startswith(b"%PDF")
                path = OUT / "raw" / f"{key}_{attempt}{'.pdf' if is_pdf else '.bin'}"
                path.write_bytes(content)
                receipt.update(http_status=response.status_code, bytes=len(content), raw_path=path.relative_to(ROOT).as_posix(),
                    sha256=digest(path), status="PDF_DOWNLOADED" if response.status_code == 200 and is_pdf else "NO_USABLE_PDF")
                if response.status_code in [403, 429]:
                    STOP.set()
            except requests.RequestException as exc:
                receipt.update(status="REQUEST_FAILED", error_type=type(exc).__name__)
            receipt["completed_at"] = now()
            save(target, receipt, True)
            time.sleep(0.3)
        record.update(status=receipt["status"], raw_path=receipt.get("raw_path"), raw_sha256=receipt.get("sha256"),
            receipt_path=target.relative_to(ROOT).as_posix(), attempts=attempt)
        if receipt["status"] != "REQUEST_FAILED":
            break
    return record


def extract_text(record):
    if record["status"] != "PDF_DOWNLOADED":
        return record
    path = ROOT / record["raw_path"]
    assert digest(path) == record["raw_sha256"]
    try:
        document = pdfium.PdfDocument(path)
        pages = []
        for page in document:
            textpage = page.get_textpage()
            pages.append(textpage.get_text_range())
            textpage.close()
            page.close()
        document.close()
        destination = OUT / "text" / (record["key"] + "_pages.json")
        save(destination, pages, True)
        has_text = len("".join(pages).strip()) >= 50
        record.update(status="PDF_TEXT_SAVED" if has_text else "PDF_NO_USABLE_TEXT",
            text_path=destination.relative_to(ROOT).as_posix(), pages=len(pages),
            text_characters=len("".join(pages)), code_appears_in_first_page=record["symbol"][:6] in (pages[0] if pages else ""))
    except (pdfium.PdfiumError, ValueError) as exc:
        record.update(status="PDF_TEXT_EXTRACTION_FAILED", error_type=type(exc).__name__)
    return record


def run():
    protocol = read(OUT / "protocol.json")
    assert digest(Path(__file__)) == protocol["code_sha256"]
    for path, sha in protocol["sources"].items():
        assert digest(ROOT / path) == sha, path
    targets = pd.read_parquet(OUT / "targets.parquet").to_dict("records")
    pilot_records = {row["document_id"]: row for row in read(pilot.OUT / "results/resolved_documents.json")}
    results, pending = [], []
    for row in targets:
        destination = OUT / "documents" / (row["document_id"] + ".json")
        if destination.exists():
            results.append(read(destination))
        elif row["document_id"] in pilot_records:
            old = pilot_records[row["document_id"]]
            assert old["status"] == "PDF_TEXT_SAVED" and old["source_url"] == row["source_url"]
            raw = pilot.OUT / old["raw_path"]
            assert digest(raw) == old["sha256"]
            text_path = pilot.OUT / old["text_path"]
            pages = read(text_path)
            record = {**row, "key": row["symbol"] + "_" + row["document_id"], "reused": True,
                "status": old["status"], "raw_path": raw.relative_to(ROOT).as_posix(), "raw_sha256": old["sha256"],
                "text_path": text_path.relative_to(ROOT).as_posix(), "pages": old["pages"],
                "text_characters": len("".join(pages)), "code_appears_in_first_page": row["symbol"][:6] in pages[0],
                "receipt_path": (pilot.OUT / old["receipt_path"]).relative_to(ROOT).as_posix()}
            save(destination, record, True)
            results.append(record)
        else:
            pending.append(row)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(request_pdf, row) for row in pending]
        for future in as_completed(futures):
            record = extract_text(future.result())
            save(OUT / "documents" / (record["document_id"] + ".json"), record, True)
            results.append(record)
            if len(results) % 25 == 0:
                print(f"原文处理{len(results)}/{len(targets)}份，文本完整{sum(r['status']=='PDF_TEXT_SAVED' for r in results)}份。", flush=True)
    results.sort(key=lambda row: (row["symbol"], row["catalogue_date"], row["document_id"]))
    save(OUT / "documents.json", results, True)
    result = {"at": now(), "study_id": STUDY,
        "status": "INDEX_ISSUER_ORIGINALS_COMPLETE_FACTS_PENDING" if all(r["status"] == "PDF_TEXT_SAVED" for r in results) else "PARTIAL_INDEX_ISSUER_ORIGINALS_FACTS_PENDING",
        "documents": len(results), "complete_texts": sum(r["status"] == "PDF_TEXT_SAVED" for r in results),
        "reused_documents": sum(r["reused"] for r in results), "companies": len({r["symbol"] for r in results}),
        "statuses": pd.Series([r["status"] for r in results]).value_counts().to_dict(),
        "first_page_code_not_confirmed": sum(r["status"] == "PDF_TEXT_SAVED" and not r["code_appears_in_first_page"] for r in results),
        "new_accounts": 0, "new_fits": 0, "new_actual_amounts_parsed": 0,
        "whole_index_actual_demand_established": False, "independent_forward_observations": 0,
        "current_market_view": "NO_VIEW", "goal_status": "active", "goal_achieved": False,
        "orders_authorized": False, "review_package_created": False}
    save(OUT / "result.json", result, True)
    print(f"原文采集终态：完整文本{result['complete_texts']}/{len(results)}份；尚未直接作为策略信号。", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "freeze":
        freeze()
    else:
        run()
