"""从原始目录补回集团自身季报，拦住公告发布主体与报表主体混用。"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.original_fund_quarterly_facts_v1 import page_texts,compact
from research.intraday_overnight_increment_v1 import now,require,write_json

BASE=Path(r"E:\ResearchData\New project 8")
INVENTORY=ROOT/"reports/research/510300_financial_original_layout_inventory_v1"
OUT=ROOT/"reports/research/510300_financial_report_subject_repair_v1"
RAW=BASE/"data/raw/510300_financial_report_subject_repair_v1"
MANIFEST=ROOT/"config/510300_financial_report_subject_repair_v1_manifest.json"
CASES=[("1218137429","1218186515","2023-10-28","2023-09-30"),("1221433608","1221451424","2024-10-22","2024-09-30")]
NAMES={"600999.SH":"招商证券股份有限公司","601601.SH":"中国太平洋保险(集团)股份有限公司",
       "601328.SH":"交通银行股份有限公司","601288.SH":"中国农业银行股份有限公司",
       "601939.SH":"中国建设银行股份有限公司","600036.SH":"招商银行股份有限公司",
       "600030.SH":"中信证券股份有限公司","601628.SH":"中国人寿保险股份有限公司",
       "000776.SZ":"广发证券股份有限公司","601211.SH":"国泰君安证券股份有限公司",
       "000001.SZ":"平安银行股份有限公司","601318.SH":"中国平安保险(集团)股份有限公司"}


def identity(pages,code,period):
    expected=NAMES[code]
    year=str(pd.Timestamp(period).year)
    first=compact(pages[0])
    front=compact("\n".join(pages[:4]))
    name_valid=expected in first
    code_valid=code.split(".")[0] in front
    period_valid=year in first and ("第三季度" in first if pd.Timestamp(period).month==9 else "第一季度" in first or "一季度" in first)
    return {"status":"PASS_DOCUMENT_SUBJECT_AND_REPORT_PERIOD" if name_valid and code_valid and period_valid else "NO_VIEW_REPORT_SUBJECT_CODE_OR_PERIOD_MISMATCH",
            "expected_issuer_full_name":expected,"expected_name_on_cover":name_valid,"security_code_in_front":code_valid,
            "report_period_on_cover":period_valid,"raw_cover":pages[0]}


def checkpoint(date):
    return BASE/"data/raw/cninfo/periodic_report_metadata_v1_0_1/daily_checkpoints"/(date+"__combined.json.gz")


def freeze():
    paths=[Path(__file__),ROOT/"docs/510300_FINANCIAL_REPORT_SUBJECT_REPAIR_V1.md",INVENTORY/"result.json",
           ROOT/"reports/research/510300_original_earnings_source_completion_v1/batch_01_result.json",
           ROOT/"reports/research/510300_original_earnings_breadth_v1/selected_verified_facts.parquet"]
    paths+=sorted((INVENTORY/"page_texts").glob("*.json"))
    paths+=[checkpoint(date) for _,_,date,_ in CASES]
    write_json(MANIFEST,{"registered_at":now(),"purpose":"ORIGINAL_REPORT_SUBJECT_AND_DATE_REPAIR_ONLY","new_strategy_returns_read":False,
                        "files":[{"path":p.relative_to(BASE).as_posix() if p.is_relative_to(BASE) else p.relative_to(ROOT).as_posix(),
                                  "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]},exclusive=True)


def run():
    require(not (OUT/"result.json").exists(),"集团原始报告修正已完成")
    for row in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]:
        path=(BASE if row["path"].startswith("data/") else ROOT)/row["path"]
        require(hashlib.sha256(path.read_bytes()).hexdigest()==row["sha256"],"原始主体修正输入变化")
    subject_records=[]
    for p in sorted((INVENTORY/"page_texts").glob("*.json")):
        data=json.loads(p.read_text(encoding="utf-8"))
        s=data["source"]
        subject_records.append({"announcement_id":s["announcement_id"],"ts_code":s["ts_code"],"sec_name":s["sec_name"],
                                **identity(data["pages"],s["ts_code"],s["report_period"])})
    write_json(OUT/"first_batch_document_subject_checks.json",{"rows":subject_records},exclusive=True)
    replaced=[]
    RAW.mkdir(parents=True,exist_ok=True)
    for rejected,correct,date,period in CASES:
        cp=checkpoint(date)
        payload=json.loads(gzip.decompress(cp.read_bytes()))
        require(payload["status"]=="COMPLETE" and len(payload["records"])==payload["declared_total"],"原始日期目录不完整")
        selected=[r for r in payload["records"] if str(r["announcementId"])==correct and r["secCode"]=="601318"]
        require(len(selected)==1,"原始集团报告目录不唯一")
        row=selected[0]
        url="https://static.cninfo.com.cn/"+row["adjunctUrl"]
        require(row["announcementTitle"]==f"中国平安{period[:4]}年第三季度报告" and f"/{date}/" in url,"原始目录标题或日期不匹配")
        pdf=RAW/(correct+".pdf")
        if not pdf.exists():
            response=requests.get(url,headers={"User-Agent":"Mozilla/5.0","Referer":"https://www.cninfo.com.cn/"},timeout=(12,35))
            require(response.status_code not in [403,429],"原始来源明确拒绝或限流，停止请求")
            response.raise_for_status()
            require(response.content.startswith(b"%PDF-"),"集团报告网址未返回PDF")
            pdf.write_bytes(response.content)
        content=pdf.read_bytes()
        pages=page_texts(content)
        checked=identity(pages,"601318.SH",period)
        require(checked["status"].startswith("PASS_"),"补回PDF仍非集团自身报告")
        source={"announcement_id":correct,"ts_code":"601318.SH","sec_name":"中国平安","report_period":period,
                "event_publication_date":date,"official_pdf_url":url,"raw_path":pdf.relative_to(BASE).as_posix(),
                "sha256":hashlib.sha256(content).hexdigest(),"bytes":len(content),"retrieved_at":now(),
                "original_catalog_checkpoint":cp.relative_to(BASE).as_posix(),"original_catalog_checkpoint_sha256":hashlib.sha256(cp.read_bytes()).hexdigest(),
                "original_catalog_record":row,"supersedes_wrong_subject_announcement_id_for_new_research":rejected,
                "status":"PASS_GROUP_REPORT_SUBJECT_AND_ORIGINAL_CATALOG_IDENTITY"}
        write_json(OUT/"replacement_records"/(correct+".json"),source,exclusive=True)
        write_json(OUT/"page_texts"/(correct+".json"),{"source":source,"pages":pages,"identity":checked},exclusive=True)
        replaced.append(source)
        print(f"集团原始季报已补回：{period}，公告{date}，{len(pages)}页，主体与证券代码吻合",flush=True)
    facts=pd.read_parquet(ROOT/"reports/research/510300_original_earnings_breadth_v1/selected_verified_facts.parquet",columns=["announcement_id"])
    impact={r:int(facts.announcement_id.astype(str).eq(r).sum()) for r,_,_,_ in CASES}
    result={"completed_at":now(),"status":"TWO_SUBSIDIARY_REPORTS_EXCLUDED_CORRECT_GROUP_REPORTS_ARCHIVED",
            "first_batch_documents":len(subject_records),"first_batch_subject_valid":sum(r["status"].startswith("PASS_") for r in subject_records),
            "first_batch_subject_failures":[{k:r[k] for k in ["announcement_id","ts_code","status"]} for r in subject_records if not r["status"].startswith("PASS_")],
            "replacement_documents":len(replaced),"round_eight_saved_valid_fact_rows_for_rejected_ids":impact,
            "old_event_selection_reason":"同证券同季度保留最早全文公告，未先排除子公司报表主体",
            "legacy_frozen_events_and_results_preserved":True,"new_portfolio_evaluation":False,
            "financial_fact_and_full_point_in_time_admission":"PENDING"}
    write_json(OUT/"result.json",result,exclusive=True)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="恢复集团自身原始财报，保留子公司错配排除证据")
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze",action="store_true")
    group.add_argument("--run",action="store_true")
    args=parser.parse_args()
    freeze() if args.freeze else run()
