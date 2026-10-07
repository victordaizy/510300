"""补取正式历史及当前行情接口文档；不登录行情服务、不请求交易。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pdfplumber
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_pressure_recovery_v1/source_followup_20261001"
JOBS = {
    "history_1_1_3": "https://www.sseinfo.com/services/assortment/document/interface/c/10759166/files/4f38cae1022a4cfa8410bbc2702e2b95.pdf",
    "ldds_level2_2_0_10": "https://www.sseinfo.com/services/assortment/document/interface/c/10759998/files/f3ca62e905764efaa3983a7c20d9e1d9.pdf",
    "step_0_63_20260918": "https://www.sse.com.cn/services/tradingtech/data/c/10832589/files/5ea26a2949c943a7ae2c9838c4c252bf.pdf",
    "is105_1_60_20260918": "https://www.sse.com.cn/services/tradingtech/data/c/10832586/files/5214b19db49641eb9805a3994751f886.pdf",
    "iopv_external_source_1_1": "https://www.sse.com.cn/services/tradingtech/data/c/10826655/files/265bfc4ae00148dd8f90e282509475f7.pdf",
    "official_history_product": "https://www.sseinfo.com/services/assortment/historical/",
}
CN = timezone(timedelta(hours=8))


def capture(item):
    identity, url = item
    result = {"id": identity, "url": url, "started_at": datetime.now(CN).isoformat()}
    try:
        response = requests.get(url, timeout=(8, 25), headers={"User-Agent": "Mozilla/5.0"})
        result.update(received_at=datetime.now(CN).isoformat(), http_status=response.status_code)
        is_pdf = response.content.startswith(b"%PDF")
        filename = identity + (".pdf" if is_pdf else ".html")
        path = OUT / filename
        path.write_bytes(response.content)
        result.update(path=filename, bytes=len(response.content), sha256=hashlib.sha256(response.content).hexdigest(),
                      status="RESPONSE_SAVED" if response.ok else "HTTP_FAILURE_SAVED", historical_market_rows=0)
        if is_pdf:
            with pdfplumber.open(path) as document:
                pages = [{"physical_page": i+1, "text": p.extract_text() or ""} for i, p in enumerate(document.pages)]
            (OUT / (identity + ".pages.json")).write_text(json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8")
            result["pages"] = len(pages)
    except Exception as error:
        result.update(status="FAILED", received_at=datetime.now(CN).isoformat(), error=f"{type(error).__name__}: {error}")
    return result


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "receipts.json").exists():
        raise SystemExit("本批已完成，保留原回执，不重复采集。")
    (OUT / "request_manifest.json").write_text(json.dumps({"registered_at": datetime.now(CN).isoformat(), "jobs": JOBS}, ensure_ascii=False, indent=2), encoding="utf-8")
    with ThreadPoolExecutor(max_workers=6) as executor:
        receipts = list(executor.map(capture, JOBS.items()))
    (OUT / "receipts.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps([{k: r.get(k) for k in ("id", "status", "http_status", "pages")} for r in receipts], ensure_ascii=False))
