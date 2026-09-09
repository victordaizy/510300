"""保存本轮论证核查的公开制度原文；不采集行情或运行策略。"""

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import ssl
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_price_concession_review_20260907/sources"
CN = timezone(timedelta(hours=8))
SOURCES = {
    "sse_2026_rule.docx": "https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/10816492/files/704204728fe74fff89de4f16efda4791.docx",
    "sse_2026_delayed_provisions.docx": "https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/10816492/files/2ebdf02d60684b07a6ca0fd1f8cd1456.docx",
    "sse_2026_rule_status.html": "https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/universal/c/c_20260424_10816492.shtml",
    "sse_2026_rule_announcement.html": "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260424_10816474.shtml",
}


def fetch(item: tuple[str, str]) -> dict:
    name, url = item
    receipt = {"filename": name, "requested_url": url, "requested_at": datetime.now(CN).isoformat()}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
        with urllib.request.urlopen(request, timeout=20, context=ssl.create_default_context()) as response:
            raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise ValueError("制度文档超过本轮单文件8MB上限")
            receipt.update({"status_code": response.status, "resolved_url": response.url,
                            "content_type": response.headers.get("Content-Type"),
                            "received_at": datetime.now(CN).isoformat()})
        target = OUT / name
        if target.exists() and target.read_bytes() != raw:
            raise FileExistsError("已保存原文发生变化，保留旧字节并停止覆盖")
        target.write_bytes(raw)
        receipt.update({"status": "CAPTURED_RAW_BYTES", "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
        if name.endswith(".docx"):
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                tree = ET.fromstring(archive.read("word/document.xml"))
            ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            lines = ["".join(p.itertext()) for p in tree.findall(".//w:p", ns)]
            # 仅抽取文档文本，不改变原始DOCX。
            (OUT / name.replace(".docx", ".txt")).write_text("\n".join(lines), encoding="utf-8")
    except (OSError, ValueError, urllib.error.URLError, zipfile.BadZipFile, ET.ParseError) as exc:
        receipt.update({"status": "CAPTURE_FAILED", "error_type": type(exc).__name__,
                        "error": str(exc), "finished_at": datetime.now(CN).isoformat()})
    return receipt


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    receipt_path = OUT / "raw_capture_receipt.json"
    if receipt_path.exists():
        raise SystemExit("本轮来源回执已存在；不重复抓取或覆盖")
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(fetch, SOURCES.items()))
    receipt_path.write_text(json.dumps({"generated_at": datetime.now(CN).isoformat(),
        "purpose": "PUBLIC_RULE_DOCUMENTS_ONLY", "market_data_downloads": 0,
        "broker_connections": 0, "tls_verification": True, "sources": rows},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"说明": "公开制度来源保存完成", "来源": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
