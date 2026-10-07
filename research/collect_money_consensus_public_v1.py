"""整理公开周报中的事前货币预期；不读取市场收益，不修改既有研究。"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup
from pdfplumber.utils import extract_text

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_expectations_policy_evidence_v1"
CACHE = OUT / "source_probe/wgc_archive"
CACHE.mkdir(parents=True, exist_ok=True)
BASE = "https://www.gold.org"


def fetch(url: str, suffix: str) -> tuple[dict, bytes]:
    path = CACHE / (hashlib.sha256(url.encode()).hexdigest()[:24] + suffix)
    receipt_path = path.with_suffix(path.suffix + ".json")
    if path.exists() and receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8")), path.read_bytes()
    try:
        r = requests.get(url, timeout=40)
        content = r.content
        rec = {"url": url, "status": r.status_code, "resolved_url": r.url}
    except requests.RequestException as exc:
        content = b""
        rec = {"url": url, "status": "FETCH_FAILED", "error": str(exc)}
    rec.update({"retrieved_at": datetime.now(timezone.utc).isoformat(),
                "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest(),
                "local_path": str(path.relative_to(ROOT)).replace("\\", "/")})
    path.write_bytes(content)
    receipt_path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec, content


def upright_text(page, left_only: bool = False) -> str:
    # PowerPoint 表格里约 1e-8 的旋转误差被库识别为竖排，按实际页面坐标恢复横排。
    chars = []
    for c in page.chars:
        if left_only and c["x0"] > page.width * 0.51:
            continue
        a, b, cc, d, _, _ = c["matrix"]
        almost_horizontal = a > 0 and d > 0 and abs(b) < 1e-5 and abs(cc) < 1e-5
        chars.append(dict(c, upright=True) if almost_horizontal else c)
    return extract_text(chars, x_tolerance=2, y_tolerance=2)


def inspect_article(url: str) -> dict:
    rec, content = fetch(url, ".html")
    result = {"article_url": url, "article_receipt": rec, "pdfs": []}
    if rec["status"] != 200:
        return result
    soup = BeautifulSoup(content, "html.parser")
    result["article_dates"] = [m.get("content") for m in soup.find_all("meta")
                               if any(t in (m.get("property", "") + m.get("name", ""))
                                      for t in ("published", "modified", "date"))]
    pdfs = sorted({urljoin(BASE, a["href"]) for a in soup.find_all("a", href=True)
                   if ".pdf" in a["href"].lower() and
                   ("/download/" in a["href"] or "weekly" in a["href"].lower() or "wmm" in a["href"].lower())})
    for pdf_url in pdfs:
        pr, body = fetch(pdf_url, ".pdf")
        item = {"source_url": pdf_url, "receipt": pr, "forecast_rows": [], "report_date": None}
        result["pdfs"].append(item)
        if pr["status"] != 200 or not body.startswith(b"%PDF"):
            continue
        try:
            with pdfplumber.open(ROOT / pr["local_path"]) as doc:
                cover = upright_text(doc.pages[0])
                dm = re.search(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b", cover)
                if dm:
                    item["report_date"] = pd.to_datetime(dm.group(0), dayfirst=True).date().isoformat()
                for i, page in enumerate(doc.pages[:7]):
                    text = upright_text(page)
                    if "The week ahead" not in text or "consensus expectations" not in text:
                        continue
                    table = upright_text(page, left_only=True)
                    for line in table.splitlines():
                        mm = re.search(r"CN\s+Money\s+Supply\s+M([12])\s+YoY\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?|--?|–)\s*$", line.strip())
                        if mm:
                            item["forecast_rows"].append({"series": "M" + mm.group(1),
                                "previous_pp": float(mm.group(2)),
                                "forecast_pp": None if mm.group(3) in ("-", "--", "–") else float(mm.group(3)),
                                "page": i + 1, "row_text": line.strip()})
                    item["ahead_page"] = i + 1
                    item["ahead_has_M1"] = "Money Supply M1" in table
                    item["ahead_has_M2"] = "Money Supply M2" in table
                    break
        except Exception as exc:
            item["parse_error"] = str(exc)
    return result


def main() -> None:
    # 官网目录在此次读取时明确显示最后一页为 page=5；完整检查该公开目录。
    index_urls = [f"{BASE}/goldhub/gold-focus/author/weekly-markets-monitor?page={i}" for i in range(6)]
    indexes = []
    articles = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for rec, body in pool.map(lambda u: fetch(u, ".html"), index_urls):
            indexes.append(rec)
            if rec["status"] == 200:
                soup = BeautifulSoup(body, "html.parser")
                for a in soup.select("a.m-blog-card__copy[href]"):
                    if "weekly-markets-monitor" in a["href"]:
                        articles.add(urljoin(BASE, a["href"]))
    (OUT / "wgc_archive_inventory.json").write_text(json.dumps({"index_receipts": indexes,
        "article_urls": sorted(articles), "selection": "当前公开作者目录全部六页，不按收益挑选"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"公开目录发现 {len(articles)} 篇周报；开始读取原始页面和 PDF。", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for i, result in enumerate(pool.map(inspect_article, sorted(articles)), 1):
            results.append(result)
            if i % 10 == 0:
                print(f"已检查 {i}/{len(articles)} 篇。", flush=True)
    (OUT / "wgc_forecast_sources.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for result in results:
        for pdf in result["pdfs"]:
            for row in pdf["forecast_rows"]:
                rows.append(dict(row, report_date=pdf["report_date"], source_url=pdf["source_url"],
                                 article_url=result["article_url"], source_sha256=pdf["receipt"]["sha256"],
                                 source_path=pdf["receipt"]["local_path"]))
    pd.DataFrame(rows).to_csv(OUT / "wgc_all_extracted_forecasts.csv", index=False, encoding="utf-8-sig")
    print(f"完成：{len(results)} 篇，提取 M1/M2 预期行 {len(rows)} 条。", flush=True)


if __name__ == "__main__":
    main()
