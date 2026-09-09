"""从上交所官方目录收集510300原始季度报告，保留完整检查点。"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.intraday_overnight_increment_v1 import now, require, write_json

BASE=Path(r"E:\ResearchData\New project 8")
RAW=BASE/"data/raw/510300_original_fund_subscription_reports_v1"
OUT=ROOT/"reports/research/510300_original_fund_subscription_reports_v1"
MANIFEST=ROOT/"config/510300_original_fund_subscription_source_v1_manifest.json"
QUERY="https://query.sse.com.cn/commonQuery.do"
REFERER="https://www.sse.com.cn/disclosure/fund/announcement/"
STOP_REQUESTS=threading.Event()


def content_id(path):
    relative=path.relative_to(BASE).as_posix() if path.is_relative_to(BASE) else path.relative_to(ROOT).as_posix()
    return {"path":relative,"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}


def freeze():
    require(not MANIFEST.exists(),"原始季报来源规则已登记")
    paths=[Path(__file__),ROOT/"docs/510300_ORIGINAL_FUND_SUBSCRIPTION_SOURCE_V1.md",RAW/"discovery/sse_search_fund.js",RAW/"discovery/sse_2021_periodic_probe.json"]
    write_json(MANIFEST,{"registered_at":now(),"phase":"SOURCE_COLLECTION_ONLY", "files":[content_id(p) for p in paths],
                         "expected_periods":[str(p) for p in pd.period_range("2012Q3","2026Q2",freq="Q")],"new_strategy_returns_read":False},exclusive=True)


def request(url, params=None):
    error=None
    for attempt in range(3):
        if STOP_REQUESTS.is_set():
            raise PermissionError("该来源已拒绝或限流，不再发起请求")
        try:
            response=requests.get(url,params=params,headers={"User-Agent":"Mozilla/5.0","Referer":REFERER},timeout=30)
            if response.status_code in (403,429):
                STOP_REQUESTS.set()
                raise PermissionError(f"官方来源拒绝或限流，停止请求：{response.status_code}")
            response.raise_for_status()
            return response
        except PermissionError:
            raise
        except requests.RequestException as exc:
            error=f"{type(exc).__name__}: {str(exc)[:180]}"
            if attempt < 2:
                time.sleep(1+attempt)
    raise RuntimeError(error)


def catalog_year(year):
    target=RAW/"catalog_checkpoints"/f"{year}.json"
    if target.exists():
        saved=json.loads(target.read_text(encoding="utf-8"))
        require(saved["status"]=="PASS_COMPLETE_YEAR_CATALOG","年份检查点未完整")
        return saved
    rows,receipts=[],[]
    page,total=1,None
    while True:
        params={"isPagination":"true","pageHelp.pageSize":25,"pageHelp.pageNo":page,"pageHelp.beginPage":page,
                "pageHelp.endPage":page,"pageHelp.cacheSize":1,"type":"inParams","sqlId":"COMMON_PL_JJXX_JJGG_NEW_L",
                "SECURITY_CODE":"510300","TITLE":"","BULLETIN_TYPE":"reits03,fund03","START_DATE":f"{year}-01-01",
                "END_DATE":f"{year}-12-31" if year<2026 else "2026-08-14","DATE_ASC":"1","DATE_DESC":"","CODE_ASC":"","CODE_DESC":""}
        response=request(QUERY,params=params)
        payload=response.json()
        require(not payload.get("actionErrors") and not payload.get("fieldErrors"),"官方目录返回查询错误")
        page_info=payload["pageHelp"]
        require(int(page_info["pageNo"])==page,"返回页码不对应")
        if total is None:
            total=int(page_info["total"])
        require(total==int(page_info["total"]),"分页中声明总数变化")
        records=payload.get("result") or []
        require(records==page_info.get("data"),"结果与分页数据不一致")
        receipts.append({"retrieved_at":now(),"request_url":QUERY,"request_params":params,
                         "response_sha256":hashlib.sha256(response.content).hexdigest(),"response":payload})
        rows.extend(records)
        require(all(x.get("SECURITY_CODE")=="510300" for x in rows),"基金代码范围不匹配")
        require(len({x.get("URL") for x in rows})==len(rows),"目录存在重复文件")
        if page>=max(1,int(page_info["pageCount"])):
            break
        require(records and page<20,"目录分页不收敛")
        page+=1
    require(len(rows)==total,"目录实际行数与官方声明不符")
    result={"year":year,"status":"PASS_COMPLETE_YEAR_CATALOG","unique_announcements":len(rows),"pages":receipts,"rows":rows}
    write_json(target,result,exclusive=True)
    return result


def period_from_title(title):
    title=re.sub(r"\s+","",title)
    match=re.search(r"(20\d{2})年第([一二三四1234])季度报告",title)
    if not match:
        return None
    quarter={"一":1,"二":2,"三":3,"四":4}.get(match[2],int(match[2]) if match[2].isdigit() else None)
    return f"{match[1]}Q{quarter}"


def collect_catalog():
    require(MANIFEST.exists(),"先登记来源规则")
    require(not (OUT/"quarterly_catalog_result.json").exists(),"季报目录已完成")
    all_rows=[]
    for year in range(2012,2027):
        result=catalog_year(year)
        all_rows.extend(result["rows"])
        print(f"上交所原始目录：{year}年，{len(result['rows'])}条定期公告",flush=True)
    all_frame=pd.DataFrame(all_rows)
    require(not all_frame.URL.duplicated().any(),"跨年度目录文件重复")
    all_frame.to_parquet(OUT/"all_official_periodic_announcements.parquet",index=False)
    rows=[]
    for item in all_rows:
        period=period_from_title(item["TITLE"])
        if period:
            bad=[x for x in ("更正","修订","更新","摘要","英文","联接") if x in item["TITLE"]]
            rows.append({"period":period,"publication_date":item["SSEDATE"],"title":item["TITLE"],
                         "url":urljoin("https://www.sse.com.cn",item["URL"]),"relative_official_path":item["URL"],
                         "ts_code":"510300.SH","unresolved_title_flags":"|".join(bad)})
    frame=pd.DataFrame(rows).sort_values(["period","publication_date","url"])
    frame.to_parquet(OUT/"official_quarterly_announcements.parquet",index=False)
    frame.to_csv(OUT/"上交所510300原始季度报告目录.csv",index=False,encoding="utf-8-sig")
    expected=set(json.loads(MANIFEST.read_text(encoding="utf-8"))["expected_periods"])
    found=set(frame.period)
    ambiguous=frame.loc[frame.period.duplicated(keep=False)|frame.unresolved_title_flags.ne(""),"period"].unique().tolist()
    result={"completed_at":now(),"status":"PASS_COMPLETE_EXPECTED_QUARTER_CATALOG" if not expected-found and not ambiguous else "PARTIAL_QUARTER_CATALOG_NEEDS_SOURCE_RECONCILIATION",
            "annual_partitions":15,"official_periodic_announcements":len(all_rows),"quarterly_documents":len(frame),
            "expected_quarters":len(expected),"missing_quarters":sorted(expected-found),"additional_quarters":sorted(found-expected),
            "ambiguous_quarters":ambiguous,"source_only":True,"new_strategy_returns_read":False}
    write_json(OUT/"quarterly_catalog_result.json",result,exclusive=True)
    print(json.dumps(result,ensure_ascii=False),flush=True)


def download_one(row):
    key=row["period"]+"_"+hashlib.sha256(row["url"].encode()).hexdigest()[:12]
    receipt=RAW/"pdf_receipts"/(key+".json")
    if receipt.exists():
        item=json.loads(receipt.read_text(encoding="utf-8"))
        path=BASE/item["raw_path"]
        require(path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"],"已有PDF检查点不一致")
        return item
    response=request(row["url"])
    content=response.content
    require(content.startswith(b"%PDF"),"下载内容不是PDF")
    path=RAW/"quarterly_pdfs"/(key+".pdf")
    path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists(),"存在无收据的PDF，需先核对")
    path.write_bytes(content)
    item={**row,"downloaded_at":now(),"status":"PASS_OFFICIAL_PDF_ARCHIVED_NOT_YET_PARSED","raw_path":path.relative_to(BASE).as_posix(),
          "sha256":hashlib.sha256(content).hexdigest(),"bytes":len(content),"http_status":response.status_code}
    write_json(receipt,item,exclusive=True)
    return item


def download():
    require(not (OUT/"quarterly_download_result.json").exists(),"原始季度PDF下载已完成")
    records=pd.read_parquet(OUT/"official_quarterly_announcements.parquet").to_dict("records")
    rows=[]
    # 只有只读的独立PDF请求并发，目录查询与检查点状态依次处理。
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map={executor.submit(download_one,row):row for row in records}
        try:
            for future in as_completed(future_map):
                row=future.result()
                rows.append(row)
                print(f"原始基金季报归档 {len(rows)}/{len(records)}：{row['period']}",flush=True)
        except Exception:
            STOP_REQUESTS.set()
            for pending in future_map:
                pending.cancel()
            raise
    rows.sort(key=lambda x:(x["period"],x["publication_date"],x["url"]))
    result={"completed_at":now(),"status":"PASS_ALL_CATALOGUED_QUARTERLY_PDFS_ARCHIVED","catalogued":len(records),"archived":len(rows),"rows":rows,
            "facts_validated":False,"source_admitted_for_portfolio":False}
    write_json(OUT/"quarterly_download_result.json",result,exclusive=True)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="510300原始季报官方目录与PDF收集")
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze",action="store_true")
    group.add_argument("--catalog",action="store_true")
    group.add_argument("--download",action="store_true")
    args=parser.parse_args()
    if not args.freeze:
        require(MANIFEST.exists(),"先登记来源规则")
        for item in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]:
            path=(BASE if item["path"].startswith("data/") else ROOT)/item["path"]
            require(hashlib.sha256(path.read_bytes()).hexdigest()==item["sha256"],"来源收集文件已变化")
    if args.freeze:
        freeze()
    elif args.catalog:
        collect_catalog()
    else:
        download()
