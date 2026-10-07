"""保留首次提取错误，复用已下载PDF并完成尚未发出的四个目录探查。"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time

import pypdfium2 as pdfium
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_factor96_issuance_source_probe_v1"


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    assert not (OUT / "result.json").exists()
    protocol = read(OUT / "protocol.json")
    save(OUT / "extraction_failure_01.json", {"at": now(), "exception": "TypeError: 'PdfTextPage' object does not support the context manager protocol",
         "scope": "首页请求完成，自由流通规则PDF已落盘，提取文本前退出；四个目录请求尚未发出。",
         "repair": "仅改用显式关闭PDF文本对象，不重复下载或改动探查范围。", "new_accounts": 0})
    (OUT / "code" / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    pdf_path = OUT / "raw/free_float_rule.pdf"
    data = pdf_path.read_bytes()
    assert data.startswith(b"%PDF")
    pdf = pdfium.PdfDocument(data)
    pages = []
    try:
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            try:
                pages.append(textpage.get_text_range())
            finally:
                textpage.close()
                page.close()
    finally:
        pdf.close()
    save(OUT / "text/free_float_rule.json", pages)
    record = {"key": "free_float_rule", "recovered_at": now(), "url": protocol["free_float_rule"],
              "status": "ORPHAN_PDF_RESPONSE_RECOVERED_AFTER_EXTRACTION_ERROR", "raw_path": "raw/free_float_rule.pdf",
              "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "pages": len(pages),
              "requested_at": None, "http_status": None,
              "clock_note": "原请求在提取异常前未写回执；不伪造请求起止时间和响应码，URL由事前请求协议及代码确定。"}
    save(OUT / "receipts/free_float_rule.json", record)
    summaries = [read(OUT / "receipts/homepage.json"), record]
    stopped = False
    for key, keyword, category in protocol["query_probes"]:
        assert not (OUT / "receipts" / (key + ".json")).exists()
        payload = {"pageNum": "1", "pageSize": "30", "column": "szse", "tabName": "fulltext", "plate": "",
                   "stock": "", "searchkey": keyword, "secid": "", "category": category, "trade": "",
                   "seDate": "2021-01-01~2021-12-31", "sortName": "time", "sortType": "desc", "isHLtitle": "false"}
        row = {"key": key, "requested_at": now(), "url": "https://www.cninfo.com.cn/new/hisAnnouncement/query", "payload": payload}
        if stopped:
            row["status"] = "NOT_REQUESTED_AFTER_SOURCE_LIMIT"
        else:
            try:
                response = requests.post(row["url"], data=payload, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.cninfo.com.cn/"}, timeout=(10, 25))
                data = response.content
                path = OUT / "raw" / (key + ".bin")
                path.write_bytes(data)
                row.update(http_status=response.status_code, raw_path=path.relative_to(OUT).as_posix(),
                           bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                           status="HTTP_OK" if response.status_code == 200 else "HTTP_ERROR")
                stopped = response.status_code in (403, 429)
                if response.status_code == 200:
                    body = response.json()
                    records = body.get("announcements") or []
                    row.update(total=body.get("totalAnnouncement"), first_page_rows=len(records),
                               row_keys=sorted(records[0]) if records else [],
                               sample_titles=[re.sub("<[^>]+>", "", r["announcementTitle"]) for r in records[:10]],
                               sample_security_codes=[r["secCode"] for r in records[:10]])
            except requests.RequestException as exc:
                row.update(status="REQUEST_FAILED", error_type=type(exc).__name__)
            except (ValueError, KeyError) as exc:
                row.update(status="RESPONSE_PARSE_FAILED", error_type=type(exc).__name__)
        row["completed_at"] = now()
        save(OUT / "receipts" / (key + ".json"), row)
        summaries.append(row)
        print(json.dumps({k: v for k, v in row.items() if k not in ["payload", "url", "row_keys"]}, ensure_ascii=False), flush=True)
        time.sleep(.4)
    save(OUT / "result.json", {"at": now(), "requests": summaries, "new_accounts": 0,
         "free_float_denominator_admitted": False, "full_issuance_calendar_admitted": False,
         "goal_achieved": False, "goal_status": "active", "external_review": "NOT_PERFORMED"})


if __name__ == "__main__":
    main()
