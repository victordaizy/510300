"""固定配股日历、修订及终止候选的原PDF和全文；本阶段不计算交易收益。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import shutil
import threading
import time

import pandas as pd
import pypdfium2 as pdfium
import requests

from research.factor96_issuance_catalogue_v1 import digest, read, save

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "reports/research/510300_factor96_issuance_catalogue_completion_v1"
OUT = ROOT / "reports/research/510300_factor96_rights_issue_documents_v1"
STOP = threading.Event()
ROLES = {"ISSUER_CALENDAR_DOCUMENT_CANDIDATE", "AMENDMENT_CANDIDATE", "TERMINATION_OR_REJECTION_CANDIDATE"}


def now():
    return datetime.now().astimezone().isoformat()


def freeze():
    assert not (OUT / "freeze.json").exists()
    states = read(PRIOR / "effective_jobs.json")
    rights = [s for s in states if s["query"] == "rights_title"]
    assert len(rights) == 329 and all(s["complete"] for s in rights)
    data = pd.read_parquet(PRIOR / "unique_documents.parquet")
    rights_data = data[data.title.str.contains("配股")].copy()
    targets = rights_data[rights_data.title_role.isin(ROLES)].sort_values(["catalogue_date", "symbol", "document_id"])
    assert len(targets) == 491 and targets.org_id.nunique() == 44
    OUT.mkdir(parents=True, exist_ok=True)
    for name, source in {
        "inputs/prior_catalogue.parquet": PRIOR / "unique_documents.parquet",
        "inputs/prior_query_states.json": PRIOR / "effective_jobs.json",
        "inputs/prior_result.json": PRIOR / "result.json",
        "inputs/current_mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
    }.items():
        destination = OUT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    target_rows = targets.to_dict("records")
    save(OUT / "targets.json", target_rows, True)
    excluded = rights_data[~rights_data.title_role.isin(ROLES)]
    excluded.to_csv(OUT / "本阶段未选中的配股标题.csv", index=False, encoding="utf-8-sig")
    (OUT / "collector_code.py").write_bytes(Path(__file__).read_bytes())
    save(OUT / "protocol.json", {"at": now(), "study_id": "510300_FACTOR96_RIGHTS_ISSUE_DOCUMENTS_V1",
        "phase": "ORIGINAL_PDFS_AND_TEXT_BEFORE_CALENDAR_FIELD_PROTOCOL",
        "source_scope": "已完整的329个配股标题查询；上一轮仍有16个其他查询缺口，但不影响这个标题过滤器的完整状态。",
        "selection": "标题含配股且属于已存日历、修订或终止三个角色的全部491份；44个查询发行人。选择不使用收益。",
        "coverage_boundary": "这是三个标题角色的原文集，不宣称配股全部文件、全部事件或M06完整供给库；其余717份标题单列。超额配股、H股、条件性方案等仍需原文区分。",
        "prefreeze_schema_inspection": "只盘点标题、角色、目录字节字段；未读取策略收益。两份网页工具PDF访问失败，不作为原文证据。",
        "transport": "静态PDF公开入口3并发，每次请求后0.25秒；连接10秒读取30秒；单文件40MiB，传输和5xx最多2次，403/429停止新请求。",
        "extraction": "下载工作线程只保存字节；主线程顺序提取PDF文本，避免PDF库并发。原PDF、失败响应与来源回执都保留。",
        "source_clock": "保存目录时间和本次下载时钟；页面署期在后续字段阶段识别，首次历史HTTP可得仍未证明。",
        "new_accounts": 0, "new_models": 0, "new_returns": 0, "T13": "NOT_RUN",
        "goal_status": "active", "goal_achieved": False, "orders_authorized": False, "external_review": "NOT_PERFORMED"}, True)
    save(OUT / "freeze.json", {"at": now(), "files": [{"path": p.relative_to(OUT).as_posix(), "sha256": digest(p)}
        for p in sorted(OUT.rglob("*")) if p.is_file()]}, True)
    print(f"配股原文范围已冻结：{len(target_rows)}份、44个发行人。", flush=True)


def download(target):
    key = target["document_id"]
    if STOP.is_set():
        return {**target, "status": "NOT_REQUESTED_AFTER_SOURCE_LIMIT", "attempts": 0}
    with requests.Session() as session:
        for attempt in [1, 2]:
            if STOP.is_set():
                return {**target, "status": "NOT_REQUESTED_AFTER_SOURCE_LIMIT", "attempts": attempt - 1}
            receipt = {"requested_at": now(), "url": target["source_url"], "document_id": key, "attempt": attempt}
            path = OUT / "raw" / f"{key}_a{attempt}.bin"
            path.parent.mkdir(exist_ok=True)
            try:
                with session.get(target["source_url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=(10, 30), stream=True) as response:
                    receipt.update(http_status=response.status_code, final_url=response.url,
                        content_type=response.headers.get("Content-Type"), claimed_length=response.headers.get("Content-Length"))
                    count = 0
                    with path.open("xb") as stream:
                        for block in response.iter_content(1024 * 256):
                            count += len(block)
                            if count > 40 * 1024 * 1024:
                                raise ValueError("单个文件超过固定40MiB上限")
                            stream.write(block)
                    receipt["status"] = "HTTP_OK" if response.status_code == 200 else "HTTP_ERROR"
                    if response.status_code in [403, 429]:
                        STOP.set()
            except requests.RequestException as error:
                receipt.update(status="REQUEST_FAILED", error_type=type(error).__name__)
            except ValueError as error:
                receipt.update(status="FIXED_FILE_LIMIT_EXCEEDED", error=str(error))
            if path.exists():
                receipt.update(raw_path=path.relative_to(OUT).as_posix(), bytes=path.stat().st_size, sha256=digest(path))
            receipt["completed_at"] = now()
            receipt_name = f"receipts/{key}_a{attempt}.json"
            save(OUT / receipt_name, receipt, True)
            time.sleep(.25)
            if receipt["status"] == "HTTP_OK":
                with path.open("rb") as stream:
                    is_pdf = stream.read(5) == b"%PDF-"
                return {**target, "status": "PDF_SAVED" if is_pdf else "NON_PDF_RESPONSE_RETAINED",
                    "raw_snapshot": receipt["raw_path"], "raw_sha256": receipt["sha256"], "bytes": receipt["bytes"],
                    "receipt_snapshot": receipt_name, "attempts": attempt}
            retryable = receipt["status"] == "REQUEST_FAILED" or 500 <= receipt.get("http_status", 0) <= 599
            if not retryable or attempt == 2:
                return {**target, "status": receipt["status"], "receipt_snapshot": receipt_name, "attempts": attempt,
                        "raw_snapshot": receipt.get("raw_path"), "raw_sha256": receipt.get("sha256")}


def extract(row):
    if row["status"] != "PDF_SAVED":
        return row
    document = None
    pages = []
    try:
        document = pdfium.PdfDocument(OUT / row["raw_snapshot"])
        for number in range(len(document)):
            page = document[number]
            text_page = page.get_textpage()
            try:
                pages.append({"page": number + 1, "text": text_page.get_text_range()})
            finally:
                text_page.close()
                page.close()
        name = f"text/{row['document_id']}.json"
        save(OUT / name, {"pages": pages}, True)
        row.update(status="PDF_TEXT_SAVED", text_snapshot=name, text_sha256=digest(OUT / name), pages=len(pages),
            text_characters=sum(len(p["text"]) for p in pages), historical_first_publication_verified=False,
            trading_feature_admitted=False)
    except Exception as error:
        row.update(status="PDF_SAVED_TEXT_FAILED", extraction_error_type=type(error).__name__, extraction_error=str(error))
    finally:
        if document is not None:
            document.close()
    return row


def run():
    assert not (OUT / "run_started.json").exists()
    for item in read(OUT / "freeze.json")["files"]:
        assert digest(OUT / item["path"]) == item["sha256"]
    assert digest(Path(__file__)) == digest(OUT / "collector_code.py")
    save(OUT / "run_started.json", {"at": now(), "freeze_sha256": digest(OUT / "freeze.json")}, True)
    targets, rows = read(OUT / "targets.json"), []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(download, target) for target in targets]
        for number, future in enumerate(as_completed(futures), 1):
            row = extract(future.result())
            rows.append(row)
            save(OUT / "documents" / (row["document_id"] + ".json"), row, True)
            if number % 20 == 0 or number == len(targets):
                progress = {"at": now(), "completed": number, "targets": len(targets),
                    "pdf_text_saved": sum(r["status"] == "PDF_TEXT_SAVED" for r in rows)}
                save(OUT / "live_progress.json", progress)
                print(f"配股原文处理{number}/{len(targets)}，PDF和文本齐备{progress['pdf_text_saved']}份。", flush=True)
    rows.sort(key=lambda r: (r["catalogue_date"], r["symbol"], r["document_id"]))
    save(OUT / "documents.json", rows, True)
    status_counts = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
    save(OUT / "result.json", {"at": now(), "study_id": "510300_FACTOR96_RIGHTS_ISSUE_DOCUMENTS_V1",
        "status": "FIXED_DOCUMENT_SCOPE_COMPLETE" if status_counts.get("PDF_TEXT_SAVED", 0) == len(targets) else "FIXED_DOCUMENT_SCOPE_WITH_GAPS",
        "target_documents": len(targets), "statuses": status_counts, "pdf_bytes": sum(r.get("bytes", 0) for r in rows),
        "text_pages": sum(r.get("pages", 0) for r in rows), "text_characters": sum(r.get("text_characters", 0) for r in rows),
        "calendar_fields_extracted": False, "full_M06_calendar_established": False, "T13": "NOT_RUN",
        "new_accounts": 0, "new_models": 0, "new_returns": 0, "goal_status": "active", "goal_achieved": False,
        "orders_authorized": False, "external_review": "NOT_PERFORMED"}, True)
    print(json.dumps(read(OUT / "result.json"), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["freeze", "run"])
    arguments = parser.parse_args()
    freeze() if arguments.stage == "freeze" else run()
