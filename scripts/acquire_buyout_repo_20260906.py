"""保存央行免费买断式逆回购原文；不读取策略收益。"""
from pathlib import Path
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import re
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(r"E:\ResearchData\New project 8\data\raw\510300_total_reverse_repo_v2")
OUT = ROOT / "reports/research/510300_total_reverse_repo_v2"
BASE = "https://www.pbc.gov.cn"
INDEX = BASE + "/zhengcehuobisi/125207/125213/125431/5492845/"

def fetch(url, path):
    if path.exists():
        return path.read_bytes()
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.content
    if len(payload) < 2000:
        raise ValueError(f"央行响应不完整：{url}")
    path.write_bytes(payload)
    return payload

def one(item):
    url, title = item
    name = url.rstrip("/").split("/")[-2] + ".html"
    path = RAW / "buyout_articles" / name
    payload = fetch(url, path)
    s = BeautifulSoup(payload.decode("utf-8"), "lxml")
    body = s.select_one("#zoom")
    if body is None:
        raise ValueError(f"公告缺少正文：{url}")
    full = s.get_text(" ", strip=True)
    m = re.search(r"文章来源[：:]?\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", full)
    if not m:
        raise ValueError(f"公告缺少发布时间：{url}")
    stamp = m.group(1)
    return {"title": title, "source_url": url, "published_at": stamp,
            "included_before_cutoff": stamp[:10] <= "2026-08-14",
            "raw_path": "data/raw/510300_total_reverse_repo_v2/buyout_articles/" + name,
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
            "body": body.get_text(" ", strip=True),
            "retrieved_at": datetime.now().astimezone().isoformat()}

def main():
    (RAW / "buyout_articles").mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    links = {}
    pages = ["index.html", "b0da893b-2.html"]
    for page in pages:
        url = INDEX + page
        payload = fetch(url, RAW / ("buyout_list_" + page))
        s = BeautifulSoup(payload.decode("utf-8"), "lxml")
        for a in s.find_all("a", href=True):
            title = a.get("title", a.get_text(strip=True))
            if "买断式逆回购" in title and "号" in title:
                links[urljoin(BASE, a["href"])] = title
    print(f"栏目共发现 {len(links)} 篇公告", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = []
        for n, record in enumerate(pool.map(one, sorted(links.items())), 1):
            records.append(record)
            print(f"已归档买断式公告 {n}/{len(links)}：{record['published_at']}", flush=True)
    records.sort(key=lambda x: x["published_at"])
    (OUT / "buyout_source_records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"归档篇数": len(records), "截止日内": sum(x["included_before_cutoff"] for x in records),
                      "首篇": records[0]["body"], "末篇截止内": [r for r in records if r["included_before_cutoff"]][-1]["body"]}, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
