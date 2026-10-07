"""有限下载新发现的LPR事前原文；不读取证券收益。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_lpr_expectation_source_extension_v2"
TARGETS = [
    ("euronews_2021_05", "https://www.euronews.com/next/2021/05/19/uk-china-economy-lpr"),
    ("yahoo_2021_09", "https://ca.finance.yahoo.com/news/china-seen-holding-benchmark-rate-060355865.html"),
    ("yahoo_2022_07", "https://finance.yahoo.com/news/china-seen-keeping-1-yr-085611090.html"),
    ("yahoo_2024_05", "https://finance.yahoo.com/news/china-expected-stand-pat-lending-061520653.html"),
    ("investing_2026_06", "https://www.investing.com/news/economy-news/china-seen-holding-rates-unchanged-as-uneven-economic-recovery-persists-4749012"),
    ("ms_2024_01", "https://www.marketscreener.com/news/latest/China-set-to-leave-lending-benchmark-LPRs-unchanged-poll-45771448/"),
    ("dunya_2023_05", "https://dunyanews.tv/en/Business/725002-China-set-to-hold-lending-benchmarks-steady-in-May-survey-shows"),
    ("yahoo_2021_02", "https://finance.yahoo.com/news/china-seen-keeping-lending-benchmark-063409567.html"),
    ("yahoo_2026_06", "https://finance.yahoo.com/economy/policy/articles/china-seen-holding-rates-unchanged-062337391.html"),
]


def fetch(item: tuple[str, str]) -> dict:
    key, url = item
    receipt_path = OUT / "receipts" / (key + ".json")
    if receipt_path.exists():
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt = {"key": key, "url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        response = requests.get(url, timeout=(8, 20), headers={"User-Agent": "Mozilla/5.0"})
        receipt.update(http_status=response.status_code, final_url=response.url)
        if response.status_code != 200:
            receipt["status"] = "ACCESS_FAILED"
        else:
            raw = OUT / "raw_local_only" / (key + ".html")
            raw.write_bytes(response.content)
            soup = BeautifulSoup(response.content, "html.parser", from_encoding=response.apparent_encoding)
            structured = []
            for tag in soup.find_all("script", type="application/ld+json"):
                try:
                    structured.append(json.loads(tag.string or tag.get_text()))
                except (json.JSONDecodeError, TypeError):
                    continue
            metadata = {tag.get("property", tag.get("name", "")): tag.get("content") for tag in soup.find_all("meta") if tag.get("content")}
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()
            body = soup.get_text("\n", strip=True)
            text_path = OUT / "raw_local_only" / (key + ".txt")
            text_path.write_text(body, encoding="utf-8")
            receipt.update(status="SAVED_HTTP", sha256=hashlib.sha256(response.content).hexdigest(), bytes=len(response.content), raw_path=raw.relative_to(OUT).as_posix(), text_path=text_path.relative_to(OUT).as_posix(), title=soup.title.get_text(" ", strip=True) if soup.title else None, metadata=metadata)
            (OUT / "raw_local_only" / (key + "_structured.json")).write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")
    except requests.RequestException as error:
        receipt.update(status="ACCESS_FAILED", error=str(error))
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        for receipt in pool.map(fetch, TARGETS):
            print(json.dumps({k:receipt.get(k) for k in ("key", "status", "http_status", "title")}, ensure_ascii=False), flush=True)
