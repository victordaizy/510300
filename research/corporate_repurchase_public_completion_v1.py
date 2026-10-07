"""使用巨潮公开披露补全预定两家公司的回购目录与发行人原文。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import pandas as pd
import pypdfium2 as pdfium
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now, digest
import research.corporate_repurchase_source_probe_v1 as prior

OUT = ROOT / "reports/research/510300_corporate_repurchase_public_completion_v1"
STUDY = "510300_CORPORATE_REPURCHASE_PUBLIC_COMPLETION_V1"
QUERY_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}


def acquire(key, url, payload=None):
    receipt_path = OUT / "receipts" / f"{key}.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        return (OUT / receipt["raw_path"]).read_bytes() if receipt.get("raw_path") else None, receipt
    receipt = {"requested_at": now(), "source_url": url, "payload": payload}
    content = None
    try:
        response = requests.get(url, headers=HEADERS, timeout=(10, 30)) if payload is None else requests.post(url, data=payload, headers=HEADERS, timeout=(10, 30))
        content = response.content
        raw = OUT / "raw" / (key + (".pdf" if content.startswith(b"%PDF") else ".bin"))
        raw.write_bytes(content)
        receipt.update(http_status=response.status_code, status="HTTP_OK" if response.status_code == 200 else "HTTP_ERROR",
            raw_path=raw.relative_to(OUT).as_posix(), sha256=digest(raw), bytes=len(content))
    except requests.RequestException as exc:
        receipt.update(status="REQUEST_FAILED", error_type=type(exc).__name__)
    receipt["completed_at"] = now()
    save(receipt_path, receipt, True)
    return content, receipt


def get_catalogue(code, org, column):
    rows, total = [], None
    for page in range(1, 6):
        if code == "300750" and page == 1:
            body = read(OUT / "cninfo_corrected_page1.json")
        else:
            payload = {"pageNum": str(page), "pageSize": "30", "column": column, "tabName": "fulltext",
                "stock": f"{code},{org}", "searchkey": "回购", "secid": "", "category": "", "trade": "",
                "seDate": f"{prior.START}~{prior.END}", "sortName": "time", "sortType": "desc", "isHLtitle": "false"}
            content, receipt = acquire(f"catalogue_{code}_{page}", QUERY_URL, payload)
            if receipt["status"] != "HTTP_OK":
                return rows, {"code": code, "complete": False, "reason": receipt["status"], "total": total}
            body = json.loads(content)
        count = int(body["totalAnnouncement"])
        if total is not None and total != count:
            raise ValueError("查询期间目录总量变化，不能拼接不同快照。")
        total = count
        batch = body.get("announcements") or []
        if not batch and total:
            raise ValueError("未取得总量时返回空页。")
        for record in batch:
            assert record["secCode"] == code and record["orgId"] == org
            title = re.sub("<[^>]+>", "", record["announcementTitle"])
            assert "回购" in title
            stamp = pd.to_datetime(record["announcementTime"], unit="ms", utc=True).tz_convert("Asia/Shanghai")
            assert pd.Timestamp(prior.START).date() <= stamp.date() <= pd.Timestamp(prior.END).date()
            rows.append({"symbol": code + (".SH" if code.startswith("6") else ".SZ"), "title": title,
                "document_id": str(record["announcementId"]), "catalogue_timestamp": stamp,
                "catalogue_date": stamp.date().isoformat(),
                "source_url": "https://static.cninfo.com.cn/" + record["adjunctUrl"],
                "official_pdf_path": record["adjunctUrl"], "query_page": page})
        if len(rows) >= total:
            break
    assert len({row["document_id"] for row in rows}) == len(rows)
    return rows, {"code": code, "complete": len(rows) == total, "total": total, "rows": len(rows)}


def run():
    for folder in ["raw", "receipts", "results", "code"]:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), prior.OUT / "protocol.json", prior.OUT / "results/catalogues.json",
        OUT / "source_plan.json", OUT / "cninfo_stock_directory.json", OUT / "cninfo_corrected_page1.json"]
    save(OUT / "completion_plan.json", {"at": now(), "study_id": STUDY,
        "action": "补原范围公司来源，不重试挑战脚本；只使用公开巨潮目录与其链接的发行人PDF；复用已经取得的宁德时代目录。",
        "inputs": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
        "new_accounts": 0, "new_fits": 0, "before_new_candidate_returns": True}, True)
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    all_rows, statuses = [], []
    for code, org, column in [("600519", "gssh0600519", "sse"), ("300750", "GD165627", "szse")]:
        rows, status = get_catalogue(code, org, column)
        all_rows.extend(rows)
        statuses.append(status)
        print(f"{code}公开披露目录：{len(rows)}份，完整={status['complete']}。", flush=True)
    pd.DataFrame(all_rows).to_parquet(OUT / "results/catalogue.parquet", index=False)
    save(OUT / "results/catalogue_status.json", statuses, True)
    original = read(prior.OUT / "results/catalogues.json")[0]["rows"]
    matched = []
    for row in original:
        candidates = [r for r in all_rows if r["symbol"] == "600519.SH" and r["title"] == row["title"]
            and abs((pd.Timestamp(r["catalogue_date"]) - pd.Timestamp(row["catalogue_date"])).days) <= 1]
        matched.append({"sse_date": row["catalogue_date"], "sse_title": row["title"], "sse_url": row["source_url"],
            "candidate_document_ids": [r["document_id"] for r in candidates], "unique_match": len(candidates) == 1})
    save(OUT / "results/sse_catalogue_cross_reference.json", matched, True)

    def fetch(row):
        key = row["symbol"] + "_" + row["document_id"]
        content, receipt = acquire(key, row["source_url"])
        valid = receipt["status"] == "HTTP_OK" and content.startswith(b"%PDF")
        return {**row, "key": key, "status": "PDF_DOWNLOADED" if valid else "NO_USABLE_PDF",
            "raw_path": receipt.get("raw_path"), "sha256": receipt.get("sha256"),
            "receipt_path": f"receipts/{key}.json"}
    with ThreadPoolExecutor(max_workers=3) as pool:
        documents = list(pool.map(fetch, all_rows))
    for i, row in enumerate(documents, 1):
        if row["status"] == "PDF_DOWNLOADED":
            doc = pdfium.PdfDocument(OUT / row["raw_path"])
            pages = []
            for page in doc:
                textpage = page.get_textpage()
                pages.append(textpage.get_text_range())
                textpage.close()
                page.close()
            doc.close()
            target = OUT / "results" / f"{row['key']}_pages.json"
            save(target, pages, True)
            row.update(status="PDF_TEXT_SAVED", text_path=target.relative_to(OUT).as_posix(), pages=len(pages))
        if i % 10 == 0:
            print(f"发行人原文已保存并提取{i}/{len(documents)}份。", flush=True)
    save(OUT / "results/documents.json", documents, True)
    complete = all(row["complete"] for row in statuses) and all(row["status"] == "PDF_TEXT_SAVED" for row in documents)
    result = {"at": now(), "study_id": STUDY,
        "status": "PUBLIC_ISSUER_ORIGINALS_COMPLETE_FACTS_PENDING" if complete else "PARTIAL_PUBLIC_ISSUER_ORIGINALS_FACTS_PENDING",
        "catalogue_status": statuses, "documents": len(documents),
        "pdfs_saved": sum(row["status"] == "PDF_TEXT_SAVED" for row in documents),
        "sse_original_catalogue_rows": len(original), "sse_uniquely_matched_rows": sum(row["unique_match"] for row in matched),
        "new_accounts": 0, "new_fits": 0, "new_independent_forward_observations": 0,
        "goal_achieved": False, "orders_authorized": False, "review_package_created": False}
    save(OUT / "result.json", result, True)
    print("两家公司的公开原文补充完成，尚未将其晋升为指数预测信号。", flush=True)


if __name__ == "__main__":
    run()
