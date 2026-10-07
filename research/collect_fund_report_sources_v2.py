"""获取指定公开资料并保留回执；不构造信号或读取后续收益。"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pypdfium2 as pdfium

OUT = Path(__file__).resolve().parents[1] / "reports/research/510300_fund_information_source_extension_v2"


def collect(key: str, url: str) -> dict:
    receipt = OUT / "receipts" / (key + ".json")
    if receipt.exists():
        return json.loads(receipt.read_text(encoding="utf-8"))
    path = OUT / "raw" / (key + ".bin")
    proc = subprocess.run(["curl.exe", "--silent", "--show-error", "--location", "--connect-timeout", "10", "--max-time", "40", "--output", str(path), "--write-out", "%{json}", url], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=45)
    meta = json.loads(proc.stdout) if proc.stdout.strip() else {}
    body = path.read_bytes() if path.exists() else b""
    rec = {"key": key, "url": url, "retrieved_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "http": meta.get("http_code"), "resolved_url": meta.get("url_effective"), "transport_exit": proc.returncode, "sha256": hashlib.sha256(body).hexdigest(), "bytes": len(body), "error": proc.stderr[:300]}
    if proc.returncode == 0 and rec["http"] == 200:
        if body.startswith(b"%PDF"):
            rec["type"] = "PDF"
            matches = []
            with pdfium.PdfDocument(path) as doc:
                rec["pages"] = len(doc)
                for index, page in enumerate(doc):
                    tp = page.get_textpage()
                    lines = tp.get_text_range().splitlines()
                    for i, line in enumerate(lines):
                        if any(word in line for word in ("510300", "最新份额", "份额变动", "数据截止", "T-1", "前一日", "前一交易日")):
                            matches.append({"page": index+1, "nearby_lines": lines[max(0,i-2):i+4]})
                    tp.close()
                    page.close()
            (OUT / "results" / (key + "_internal_extract.json")).write_text(json.dumps(matches, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            soup = BeautifulSoup(body, "html.parser")
            rec["type"] = "HTML"
            rec["title"] = soup.title.get_text(" ", strip=True) if soup.title else ""
            rec["access_challenge"] = any(x in soup.get_text() for x in ("请输入验证码", "访问过于频繁", "人机验证", "Access Denied"))
    else:
        rec["type"] = "FETCH_FAILED"
    receipt.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="保存指定公开资金资料")
    parser.add_argument("key")
    parser.add_argument("url")
    args = parser.parse_args()
    print(json.dumps(collect(args.key, args.url), ensure_ascii=False, indent=2))
