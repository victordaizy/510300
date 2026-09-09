"""补齐已确定十二家金融企业的一季报、三季报原件；零预算，不读取策略收益。"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pypdfium2 as pdfium
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import compact, read, save, now
from research.financial_original_facts_v1_1 import NAMES

OUT=ROOT/"reports/research/510300_financial_quarter_history_v1"
RAW=ROOT/"data/raw/510300_financial_quarter_history_v1"
QUEUE=ROOT/"reports/research/510300_original_earnings_source_completion_v1/missing_original_reports_queue.parquet"
MANIFEST=ROOT/"config/510300_financial_quarter_history_v1_manifest.json"
CACHES=["510300_financial_original_layout_inventory_v1","510300_financial_report_subject_repair_v1","510300_financial_ttm_dependencies_v1"]
REPLACEMENTS={"1218137429":"1218186515","1221433608":"1221451424"}


def identify(pages,source):
    first=next((i for i,p in enumerate(pages[:2]) if compact(p)),None)
    if first is None:return {"status":"NO_VIEW_EMPTY_FIRST_TWO_TEXT_PAGES"}
    title=compact(pages[first]); front=compact("\n".join(pages[:5]))
    year=str(source["report_period"])[:4]; month=str(source["report_period"])[5:7]
    terms=["第一季度","一季度"] if month=="03" else ["第三季度","三季度"]
    checks={"first_text_page_full_name":compact(NAMES[source["ts_code"]]) in title,"first_text_page_year":year in title,
            "first_text_page_quarter":any(x in title for x in terms),"front_code":source["ts_code"].split(".")[0] in front}
    return {"status":"PASS_TITLE_FULL_NAME_CODE_PERIOD" if all(checks.values()) else "NO_VIEW_SUBJECT_OR_PERIOD_NEEDS_LAYOUT_REVIEW", "checks":checks,"first_text_page":first+1,
            "is_financial_fact_admission":False}


def freeze():
    if MANIFEST.exists():raise FileExistsError("金融季度连续原件范围已登记")
    d=pd.read_parquet(QUEUE)
    selected=d.loc[d.ts_code.isin(NAMES)&d.period_type.isin(["Q1","Q3"])].copy()
    selected["original_queue_announcement_id"]=selected.announcement_id
    supplements=[]
    for wrong,right in REPLACEMENTS.items():
        pos=selected.index[selected.announcement_id.eq(wrong)]
        if len(pos)!=1:raise ValueError("已有主体修复不能唯一对应连续队列")
        path=ROOT/"reports/research/510300_financial_report_subject_repair_v1/page_texts"/f"{right}.json"
        s=read(path)["source"]
        for k in ["announcement_id","report_period","event_publication_date","official_pdf_url"]:
            selected.loc[pos,k]=pd.Timestamp(s[k]) if k in ["report_period","event_publication_date"] else s[k]
        selected.loc[pos,"announcement_title"]="已确认集团主体的原始三季报"
        supplements.append(path)
    selected=selected.sort_values(["report_period","ts_code","announcement_id"]).reset_index(drop=True)
    if len(selected)!=252 or selected.duplicated(["ts_code","report_period"]).any():raise ValueError("登记数量或证券期间不符")
    OUT.mkdir(parents=True,exist_ok=True)
    selected.to_parquet(OUT/"selected_before_download.parquet",index=False)
    paths=[Path(__file__),ROOT/"docs/510300_FINANCIAL_QUARTER_HISTORY_V1.md",ROOT/"research/financial_annual_components_v1.py",ROOT/"research/financial_original_facts_v1_1.py",QUEUE,OUT/"selected_before_download.parquet"]+supplements
    for folder in CACHES:paths+=sorted((ROOT/"reports/research"/folder/"page_texts").glob("*.json"))
    save(MANIFEST,{"registered_at":now(),"budget_cny":0,"documents":len(selected),"companies":int(selected.ts_code.nunique()),
        "scope":"2016年一季报至2026年一季报，仅固定一季报和三季报子集；不声称全部披露或滚动盈利已齐备", "maximum_processes":3,
        "known_subject_replacements":REPLACEMENTS,"strategy_returns_read":False,
        "files":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]},exclusive=True)
    print("已登记连续季度来源：十二家公司、二百五十二份报告。",flush=True)


def acquire(row):
    aid=str(row["announcement_id"])
    record=OUT/"records"/f"{aid}.json"; cache=OUT/"page_texts"/f"{aid}.json"
    if record.exists() and cache.exists():
        s=read(record)
        if hashlib.sha256((ROOT/s["raw_path"]).read_bytes()).hexdigest()!=s["sha256"]:raise ValueError("已保存的原件发生变化")
        return s
    s={k:str(row[k]) for k in ["announcement_id","ts_code","sec_name","report_period","event_publication_date","official_pdf_url","announcement_title","original_queue_announcement_id"]}
    s.update({"retrieved_at":now(),"new_financial_facts_admitted":0,"strategy_returns_read":False})
    try:
        reusable=next((ROOT/"reports/research"/folder/"page_texts"/f"{aid}.json" for folder in CACHES if (ROOT/"reports/research"/folder/"page_texts"/f"{aid}.json").exists()),None)
        if reusable:
            d=read(reusable);prev=d["source"]
            for k in ["ts_code","report_period","event_publication_date","official_pdf_url"]:
                l=str(s[k])[:10] if k.endswith("date") or k=="report_period" else s[k]
                r=str(prev[k])[:10] if k.endswith("date") or k=="report_period" else prev[k]
                if l!=r:raise ValueError("复用来源与新队列不一致: "+k)
            content=(ROOT/prev["raw_path"]).read_bytes()
            if hashlib.sha256(content).hexdigest()!=prev["sha256"]:raise ValueError("复用原件哈希变化")
            pages=d["pages"]
            s.update({"raw_path":prev["raw_path"],"reused_from":reusable.relative_to(ROOT).as_posix(),"original_retrieved_at":prev.get("retrieved_at")})
        else:
            path=RAW/f"{aid}.pdf"
            if path.exists():content=path.read_bytes()
            else:
                response=requests.get(s["official_pdf_url"],headers={"User-Agent":"Mozilla/5.0","Referer":"https://www.cninfo.com.cn/"},timeout=(12,35))
                if response.status_code in (403,429):
                    s["source_refused_or_rate_limited"]=True
                response.raise_for_status()
                content=response.content
                if not content.startswith(b"%PDF-"):raise ValueError("原地址没有返回 PDF")
                temp=path.with_suffix(".downloading.pdf");temp.write_bytes(content);temp.replace(path)
            doc=pdfium.PdfDocument(content);pages=[]
            try:
                for i in range(len(doc)):
                    p=doc[i];tp=p.get_textpage()
                    try:pages.append(tp.get_text_range())
                    finally:tp.close();p.close()
            finally:doc.close()
            s.update({"raw_path":path.relative_to(ROOT).as_posix(),"reused_from":None})
        s.update({"status":"ORIGINAL_FULL_TEXT_ARCHIVED_FACT_EXTRACTION_PENDING","sha256":hashlib.sha256(content).hexdigest(),"bytes":len(content),"page_count":len(pages),"subject":identify(pages,s)})
        save(cache,{"source":s,"pages":pages})
    except Exception as exc:
        s.update({"status":"SOURCE_COLLECTION_FAILED","error":str(exc)})
    save(record,s)
    return s


def run():
    for r in read(MANIFEST)["files"]:
        if hashlib.sha256((ROOT/r["path"]).read_bytes()).hexdigest()!=r["sha256"]:raise ValueError("登记来源变化: "+r["path"])
    if (OUT/"result.json").exists():raise FileExistsError("已完成本批，不覆盖结果")
    RAW.mkdir(parents=True,exist_ok=True)
    selected=pd.read_parquet(OUT/"selected_before_download.parquet").to_dict("records")
    iterator=iter(selected);records=[];halt=False
    with ProcessPoolExecutor(max_workers=3) as executor:
        pending={executor.submit(acquire,next(iterator)) for _ in range(3)}
        while pending:
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for f in done:
                r=f.result();records.append(r)
                halt=halt or r.get("source_refused_or_rate_limited",False)
                if not halt:
                    row=next(iterator,None)
                    if row is not None:pending.add(executor.submit(acquire,row))
                if len(records)%12==0 or r["status"]=="SOURCE_COLLECTION_FAILED":
                    save(OUT/"progress.json",{"updated_at":now(),"processed":len(records),"requested":len(selected),"archived":sum(x["status"].startswith("ORIGINAL_") for x in records),"last":r})
                    print(f"连续金融原件 {len(records)}/{len(selected)}，最近 {r['sec_name']} {r['report_period'][:10]}，{r['status']}",flush=True)
    records.sort(key=lambda r:(r["report_period"],r["ts_code"]))
    result={"completed_at":now(),"status":"FIXED_Q1_Q3_SOURCE_COLLECTION_COMPLETE_EXTRACTION_NEXT" if len(records)==len(selected) and all(r["status"].startswith("ORIGINAL_") for r in records) else "FIXED_Q1_Q3_SOURCE_COLLECTION_PARTIAL",
        "requested":len(selected),"processed":len(records),"archived":sum(r["status"].startswith("ORIGINAL_") for r in records),"reused":sum(bool(r.get("reused_from")) for r in records),
        "first_title_subject_passed":sum(r.get("subject",{}).get("status","").startswith("PASS_") for r in records),"pages":sum(r.get("page_count",0) for r in records),
        "pdf_bytes":sum(r.get("bytes",0) for r in records),"halted_after_source_refusal":halt,"new_financial_facts":0,"new_account_evaluations":0,"rows":records}
    save(OUT/"result.json",result,exclusive=True)
    print({k:v for k,v in result.items() if k!="rows"},flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument("--freeze",action="store_true");g.add_argument("--run",action="store_true");a=ap.parse_args()
    freeze() if a.freeze else run()
