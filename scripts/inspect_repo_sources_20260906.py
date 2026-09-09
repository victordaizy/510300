"""只检查央行公告结构与栏目来源，不读取候选收益。"""
from pathlib import Path
import collections
import json
import re
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_total_reverse_repo_v2"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = ROOT / "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1/pboc_open_market_notice_ledger_20150105_20260814.parquet"

def compact(x):
    return re.sub(r"\s+", "", x).replace(",", "").replace("，", "")

def main():
    print("开始读取已存公告目录", flush=True)
    ledger = pd.read_parquet(LEDGER.resolve())
    terms, shapes, no_terms, records = collections.Counter(), collections.Counter(), [], []
    for n, row in enumerate(ledger.itertuples(), 1):
        path = Path(r"E:\ResearchData\New project 8\data") / Path(row.raw_path).relative_to("data")
        if n % 100 == 1:
            print(f"正在处理 {n}/{len(ledger)}：{path.name}", flush=True)
        s = BeautifulSoup(path.read_bytes().decode("utf-8"), "lxml")
        found = []
        for tr in s.find_all("tr"):
            cells = [compact(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"], recursive=False)]
            if cells and re.fullmatch(r"\d+(?:天|个月|月|年)(?:期)?(?:[（(]\d+天[）)])?", cells[0]):
                terms[cells[0]] += 1
                table = tr.find_parent("table")
                allrows = [[compact(td.get_text(" ", strip=True)) for td in z.find_all(["td", "th"], recursive=False)] for z in table.find_all("tr")]
                header = tuple(allrows[0])
                shapes[header] += 1
                found.append({"cells": cells, "header": header})
        body = s.get_text(" ", strip=True)
        body = body[body.find("文章来源"):]
        body = body[:body.find("法律声明")]
        if not found:
            no_terms.append({"date": str(row.notice_date), "text": body, "path": row.raw_path})
        records.append({"date": str(row.notice_date), "path": row.raw_path, "source_url": row.source_url, "rows": found, "text": body})
        if n % 500 == 0:
            print(f"公告结构已整理 {n}/{len(ledger)}", flush=True)
    (OUT / "ordinary_structure_inventory.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "ordinary_no_tenor_rows.json").write_text(json.dumps(no_terms, ensure_ascii=False, indent=2), encoding="utf-8")
    print("期限分布：", dict(terms), flush=True)
    print("表头分布：", shapes.most_common(25), flush=True)
    print("无期限行公告数：", len(no_terms), flush=True)
    counts = collections.Counter()
    examples = {}
    for r in no_terms:
        t = r["text"]
        k = "央票" if "央行票据" in t else "暂停" if ("不开展" in t or "暂停" in t) else "其他"
        counts[k] += 1
        examples.setdefault(k, r)
    print("无期限分类：", dict(counts), flush=True)
    print(json.dumps(examples, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
