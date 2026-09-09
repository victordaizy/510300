"""归档基金业协会月报并保留真实发布日，不将基金净值变化当净申购。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
import hashlib
import json
import re
import requests
import pdfplumber
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_fundamental_and_fund_flow_rebuild_v1"
RAW = Path(r"E:\ResearchData\New project 8\data\raw\510300_fundamental_and_fund_flow_rebuild_v1\amac")
BASE = "https://www.amac.org.cn/sjtj/tjbg/gmjj/"

def get(url, path):
    if path.exists():
        return path.read_bytes()
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    if len(r.content) < 1000:
        raise ValueError(f"官方月报响应不完整：{url}")
    path.write_bytes(r.content)
    return r.content

def one(item):
    a = dict(item)
    url = a["source_url"]
    path = RAW / (url.rstrip("/").split("/")[-1])
    try:
        payload = get(url, path)
        a.update({"raw_path": "data/raw/510300_fundamental_and_fund_flow_rebuild_v1/amac/" + path.name,
                  "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)})
        if payload.startswith(b"%PDF"):
            with pdfplumber.open(path) as pdf:
                a["page_count"] = len(pdf.pages)
                a["extracted_text"] = "\n".join(p.extract_text() or "" for p in pdf.pages)
                a["tables"] = [p.extract_tables() for p in pdf.pages]
            a["state"] = "PDF_ARCHIVED_TEXT_TABLES_EXTRACTED"
        else:
            s = BeautifulSoup(payload.decode("utf-8"), "lxml")
            a["extracted_text"] = s.get_text(" ", strip=True)
            a["attachments"] = sorted(set(urljoin(url,z["href"]) for z in s.find_all("a",href=True) if z["href"].lower().endswith(".pdf")))
            a["images"] = sorted(set(urljoin(url,z["src"]) for z in s.find_all("img",src=True) if "P0" in z["src"]))
            a["state"] = "HTML_ARCHIVED_ADDITIONAL_TABLE_EXTRACTION_REQUIRED"
        a["retrieved_at"] = datetime.now().astimezone().isoformat()
    except Exception as error:
        a.update({"state": "SOURCE_RETRIEVAL_ERROR", "error": str(error)})
    return a

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    RAW.mkdir(parents=True,exist_ok=True)
    records = {}
    for n in range(5):
        url = BASE + ("index.html" if n == 0 else f"index_{n}.html")
        p = RAW / ("list_" + str(n) + ".html")
        s = BeautifulSoup(get(url,p).decode("utf-8"),"lxml")
        for a in s.find_all("a",href=True):
            title = a.get_text(" ",strip=True)
            m = re.search(r"公募基金市场数据[（(](\d{4})年(\d+)月[）)]",title)
            if not m:
                continue
            li = a.find_parent("li")
            date = re.search(r"\d{4}-\d{2}-\d{2}",li.get_text(" ",strip=True) if li else "")
            if not date:
                continue
            source = urljoin(url,a["href"])
            stamp = date.group(0)
            records[source] = {"source_url":source,"title":title,"publication_date":stamp,
                               "report_month":f"{m.group(1)}-{int(m.group(2)):02d}",
                               "publication_clock_precision":"DATE_ONLY_USE_NEXT_A_SHARE_SESSION_OPEN",
                               "included_before_cutoff":stamp <= "2026-08-14","list_raw_path":str(p)}
        print(f"已读取协会目录 {n+1}/5，累计 {len(records)} 份月报",flush=True)
    selected = [r for r in records.values() if r["included_before_cutoff"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        output=[]
        for n, r in enumerate(pool.map(one,sorted(selected,key=lambda x:x["report_month"])),1):
            output.append(r)
            print(f"公募官方月报 {n}/{len(selected)}：{r['report_month']}，{r['state']}",flush=True)
    (OUT / "amac_source_records.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"月报数":len(output),"首月":output[0]["report_month"],"末月":output[-1]["report_month"],
                      "有表格":sum(bool(r.get('tables')) for r in output)},ensure_ascii=False),flush=True)

if __name__ == "__main__":
    main()
