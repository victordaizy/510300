"""核对原始季度报告的基金身份、期间、份额衔接与可用日期。"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import re
import unicodedata
from decimal import Decimal,InvalidOperation
from pathlib import Path

import pandas as pd
import pdfplumber
import pypdfium2 as pdfium

from research.adaptive_allocation_v1 import ROOT
from research.intraday_overnight_increment_v1 import now,require,write_json

BASE=Path(r"E:\ResearchData\New project 8")
OUT=ROOT/"reports/research/510300_original_fund_subscription_reports_v1"
MANIFEST=ROOT/"config/510300_original_fund_quarterly_facts_v1_manifest.json"
CALENDAR="data/reference/a_share_hs_trading_calendar_2010_2026_v1.parquet"
LABELS={"beginning_units":"期初基金份额总额","gross_subscription_units":"基金总申购份额",
        "gross_redemption_units":"基金总赎回份额","split_delta_units":"基金拆分变动份额","ending_units":"期末基金份额总额"}


def compact(text):
    return re.sub(r"\s+","",unicodedata.normalize("NFKC",str(text))).replace("−","-").replace("—","-").replace("－","-")


def decimal_cell(cell, *, allow_dash=False):
    if cell is None or not str(cell).strip():
        raise ValueError("数值格缺失")
    text=compact(cell).replace(",","")
    if text=="-" and allow_dash:
        return Decimal(0),"REPORTED_DASH_NO_ACTIVITY_FOR_TABLE_IDENTITY"
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?",text):
        raise ValueError("数值格不能直接解析")
    try:
        return Decimal(text),"DIRECT_ORIGINAL_DECIMAL_VALUE"
    except InvalidOperation as exc:
        raise ValueError("数值格不属于有限十进制数") from exc


def parse_flow_table(table):
    selected={}
    for row in table:
        if len(row)!=2:
            continue
        label=compact(row[0])
        keys=[key for key,text in LABELS.items() if text in label]
        for key in keys:
            if key in selected:
                return {"status":"NO_VIEW_DUPLICATE_SHARE_FLOW_ROW"}
            selected[key]={"label":row[0],"raw_cell":row[1]}
    if set(selected)!=set(LABELS):
        return {"status":"NO_VIEW_INCOMPLETE_SHARE_FLOW_TABLE"}
    values={}
    try:
        for key,item in selected.items():
            value,status=decimal_cell(item["raw_cell"],allow_dash=key=="split_delta_units")
            values[key]=value
            item["cell_status"]=status
            item["exact_decimal_value"]=str(value)
    except ValueError as exc:
        return {"status":"NO_VIEW_UNPROVEN_SHARE_FLOW_VALUE","reason":str(exc),"raw_rows":selected}
    if any(values[key]<0 for key in LABELS if key!="split_delta_units") or values["beginning_units"]<=0:
        return {"status":"NO_VIEW_INVALID_SHARE_FLOW_SIGN_OR_BEGINNING_UNITS","raw_rows":selected}
    residual=values["beginning_units"]+values["gross_subscription_units"]-values["gross_redemption_units"]+values["split_delta_units"]-values["ending_units"]
    if abs(residual)>Decimal("0.01"):
        return {"status":"NO_VIEW_SHARE_FLOW_ACCOUNTING_MISMATCH","identity_residual":str(residual),"raw_rows":selected}
    return {"status":"PASS_ORIGINAL_SHARE_FLOW_TABLE_IDENTITY","values":{k:float(v) for k,v in values.items()},
            "exact_values":{k:str(v) for k,v in values.items()},"identity_residual":str(residual),"raw_rows":selected}


def date_values(text):
    return [pd.Timestamp(int(y),int(m),int(d)) for y,m,d in re.findall(r"(20\d{2})年(\d{1,2})月(\d{1,2})日",compact(text))]


def validate_identity_and_period(cover, overview, period):
    title=compact(cover)
    body=compact(overview)
    if "华泰柏瑞沪深300交易型开放式指数证券投资基金" not in title or "联接" in title:
        return {"status":"NO_VIEW_FUND_COVER_IDENTITY_MISMATCH"}
    if not re.search(r"(?:基金主代码|交易代码|基金代码)510300(?!\d)",body):
        return {"status":"NO_VIEW_FUND_SECURITY_CODE_MISMATCH"}
    q=pd.Period(period,freq="Q")
    match=re.search(r"(20\d{2})年第([1234])季度报告",title)
    if not match or int(match[1])!=q.year or int(match[2])!=q.quarter:
        return {"status":"NO_VIEW_REPORT_QUARTER_COVER_MISMATCH"}
    start=body.find("本报告期自")
    dates=date_values(body[start:start+120]) if start>=0 else []
    if len(dates)<2 or dates[0]!=q.start_time.normalize() or dates[1]!=q.end_time.normalize():
        return {"status":"NO_VIEW_REPORT_PERIOD_BODY_MISMATCH","observed_dates":[str(d.date()) for d in dates]}
    at=title.find("报告送出日期")
    sent=date_values(title[at:at+55]) if at>=0 else []
    if not sent or sent[0]<=q.end_time.normalize():
        return {"status":"NO_VIEW_REPORT_SEND_DATE_UNPROVEN"}
    return {"status":"PASS_ORIGINAL_FUND_IDENTITY_PERIOD_AND_SEND_DATE","reported_send_date":str(sent[0].date()),
            "report_period_start":str(q.start_time.date()),"report_period_end":str(q.end_time.date())}


def available_clock(source, send_date, metadata, calendar):
    official=source.get("publication_date")
    if not source.get("official_publication_catalog_verified") or not official or pd.isna(official):
        return {"status":"NO_VIEW_ORIGINAL_PUBLICATION_CATALOG_MISSING","source_usable":False,"feature_available_session":None}
    public=max(pd.Timestamp(official),pd.Timestamp(send_date))
    later=[]
    for key in ("CreationDate","ModDate"):
        match=re.search(r"(?:D:)?(20\d{2})(\d{2})(\d{2})",str(metadata.get(key,"")))
        if match and pd.Timestamp(int(match[1]),int(match[2]),int(match[3]))>public:
            later.append(key)
    if later:
        return {"status":"NO_VIEW_PDF_METADATA_LATER_THAN_ORIGINAL_PUBLICATION","source_usable":False,"metadata_fields":later,"feature_available_session":None}
    next_days=pd.DatetimeIndex(calendar)[pd.DatetimeIndex(calendar)>public]
    if not len(next_days):
        return {"status":"NO_VIEW_NO_NEXT_MARKET_SESSION","source_usable":False,"feature_available_session":None}
    return {"status":"PASS_OFFICIAL_CATALOG_CLOCK_WITH_CONSERVATIVE_NEXT_SESSION","source_usable":True,
            "effective_publication_date":str(public.date()),"cover_send_date_equals_catalog_date":str(send_date)==str(official),
            "feature_available_session":str(next_days[0].date()),"feature_clock_note":"自该交易日开盘已可用，收盘型策略当日收盘首次使用，再下一开盘执行。"}


def page_texts(content):
    texts=[]
    document=pdfium.PdfDocument(content)
    try:
        for i in range(len(document)):
            page=document[i]
            text_page=page.get_textpage()
            try:
                texts.append(text_page.get_text_range())
            finally:
                text_page.close()
                page.close()
    finally:
        document.close()
    return texts


def parse_report(source,calendar):
    path=BASE/source["raw_path"]
    content=path.read_bytes()
    require(hashlib.sha256(content).hexdigest()==source["sha256"],"原始报告内容与收据不一致")
    texts=page_texts(content)
    identity=validate_identity_and_period(texts[0],texts[1],source["period"])
    result={"period":source["period"],"source":source,"page_count":len(texts),"identity":identity,"source_usable":False}
    if not identity["status"].startswith("PASS_"):
        result["status"]=identity["status"]
        result["raw_cover"]=texts[0]
        result["raw_overview"]=texts[1]
        return result
    candidates=[]
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        result["pdf_metadata"]=pdf.metadata
        for i,text in enumerate(texts):
            clean=compact(text)
            marker=clean.find("开放式基金份额变动")
            if marker<0:
                continue
            section=clean[marker:]
            if "单位:份" not in section[:160]:
                continue
            for table in pdf.pages[i].extract_tables():
                parsed=parse_flow_table(table)
                if parsed["status"]!="NO_VIEW_INCOMPLETE_SHARE_FLOW_TABLE":
                    candidates.append({"source_page":i+1,"unit":"FUND_UNITS","raw_page_text":text,"raw_table":table,**parsed})
    if len(candidates)!=1 or not candidates[0]["status"].startswith("PASS_"):
        result.update({"status":"NO_VIEW_ORIGINAL_FLOW_TABLE_NOT_UNIQUELY_VERIFIED","candidate_tables":candidates})
        return result
    table=candidates[0]
    clock=available_clock(source,identity["reported_send_date"],result["pdf_metadata"],calendar)
    result.update({"status":"PASS_ORIGINAL_FACTS_AND_CLOCK" if clock["source_usable"] else clock["status"],
                   "flow_table":table,"clock":clock,"source_usable":clock["source_usable"],"raw_cover":texts[0]})
    return result


def freeze():
    require(not MANIFEST.exists(),"原始季报事实识别规则已登记")
    paths=[Path(__file__),ROOT/"tests/test_original_fund_quarterly_facts_v1.py",ROOT/"docs/510300_ORIGINAL_FUND_QUARTERLY_FACTS_V1.md",
           OUT/"quarterly_download_v1_1_result.json",OUT/"catalog_title_and_missing_source_amendment_v1_1.json",BASE/CALENDAR]
    write_json(MANIFEST,{"registered_at":now(),"phase":"SOURCE_FACTS_AND_CLOCKS_ONLY","files":[
        {"path":p.relative_to(BASE).as_posix() if p.is_relative_to(BASE) else p.relative_to(ROOT).as_posix(),
         "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],"new_strategy_returns_read":False},exclusive=True)


def run():
    require(not (OUT/"quarterly_facts_result.json").exists(),"原始季报事实已经保存")
    for row in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]:
        path=(BASE if row["path"].startswith("data/") else ROOT)/row["path"]
        require(hashlib.sha256(path.read_bytes()).hexdigest()==row["sha256"],"来源识别输入变化")
    calendar=pd.read_parquet(BASE/CALENDAR)
    days=calendar.loc[calendar.is_open,"trade_date"]
    sources=json.loads((OUT/"quarterly_download_v1_1_result.json").read_text(encoding="utf-8"))["rows"]
    rows=[]
    for n,source in enumerate(sources,1):
        result=parse_report(source,days)
        write_json(OUT/"quarterly_fact_records"/(source["period"]+".json"),result,exclusive=True)
        values=result.get("flow_table",{}).get("values",{})
        row={"period":source["period"],"ts_code":"510300.SH","status":result["status"],"source_usable":result["source_usable"],
             "publication_date":source.get("publication_date"),"reported_send_date":result["identity"].get("reported_send_date"),
             "feature_available_session":result.get("clock",{}).get("feature_available_session"),
             "source_page":result.get("flow_table",{}).get("source_page"),"source_url":source["url"],"source_sha256":source["sha256"],
             "identity_residual":result.get("flow_table",{}).get("identity_residual"),**values}
        if values:
            row["net_subscription_units"]=values["gross_subscription_units"]-values["gross_redemption_units"]
            row["net_units_divided_by_beginning_units"]=row["net_subscription_units"]/values["beginning_units"]
        rows.append(row)
        print(f"原始份额与公布时钟核对 {n}/{len(sources)}：{source['period']}，{result['status']}",flush=True)
    frame=pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
    frame["quarter_transition_residual"]=frame.beginning_units-frame.ending_units.shift(1)
    frame.to_parquet(OUT/"quarterly_share_flow_facts.parquet",index=False)
    frame.to_csv(OUT/"原始季度申购赎回与可用日期.csv",index=False,encoding="utf-8-sig")
    transitions=frame.loc[frame.quarter_transition_residual.notna()]
    bad=transitions.loc[transitions.quarter_transition_residual.abs()>.01]
    valid=frame.loc[frame.source_usable]
    result={"completed_at":now(),"status":"SOURCE_FACTS_CONSTRUCTED_WITH_EXPLICIT_EXCLUSIONS",
            "quarterly_reports":len(frame),"reports_with_verified_share_flow_values":int(frame.beginning_units.notna().sum()),
            "reports_with_facts_and_official_clock":len(valid),"unresolved_reports":frame.loc[~frame.source_usable,["period","status"]].to_dict("records"),
            "verified_adjacent_quarter_transitions":len(transitions)-len(bad),"unexplained_quarter_transitions":bad[["period","quarter_transition_residual"]].to_dict("records"),
            "first_strict_available_session":valid.feature_available_session.min() if len(valid) else None,
            "last_strict_available_session":valid.feature_available_session.max() if len(valid) else None,
            "cash_flow_amount_reconstructed":False,"new_portfolio_evaluation":False,
            "original_publication_cryptographic_receipt_at_historical_time":False}
    write_json(OUT/"quarterly_facts_result.json",result,exclusive=True)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="原始基金季报份额与公布时钟核对")
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze",action="store_true")
    group.add_argument("--run",action="store_true")
    args=parser.parse_args()
    freeze() if args.freeze else run()
