"""保留第一版结果，补充同一原始基金申赎表的相邻跨页识别。"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pdfplumber

from research.original_fund_quarterly_facts_v1 import (
    BASE, OUT, ROOT, CALENDAR, LABELS, compact, parse_flow_table,
    page_texts, available_clock, now, require, write_json,
)

MANIFEST=ROOT/"config/510300_original_fund_quarterly_facts_v1_1_manifest.json"


def join_adjacent_tables(tables):
    rows=[]
    for table in tables:
        for row in table:
            if len(row)==2 and any(label in compact(row[0]) for label in LABELS.values()):
                rows.append(row)
    parsed=parse_flow_table(rows)
    if parsed["status"].startswith("PASS_"):
        observed=[next(k for k,v in LABELS.items() if v in compact(row[0])) for row in rows]
        if observed!=list(LABELS):
            return {"status":"NO_VIEW_CROSS_PAGE_SHARE_FLOW_ROW_ORDER_MISMATCH"}
    return {**parsed,"raw_table":rows}


def repair(old,calendar):
    if old["status"]!="NO_VIEW_ORIGINAL_FLOW_TABLE_NOT_UNIQUELY_VERIFIED":
        return {**old,"source_parser_version":"V1_PRESERVED"}
    source=old["source"]
    content=(BASE/source["raw_path"]).read_bytes()
    require(hashlib.sha256(content).hexdigest()==source["sha256"],"跨页修正的原始报告哈希变化")
    texts=page_texts(content)
    headings=[i for i,text in enumerate(texts) if "开放式基金份额变动" in compact(text)]
    result={**old,"source_parser_version":"V1_1_ADJACENT_PAGES","prior_status":old["status"]}
    if len(headings)!=1 or headings[0]+1>=len(texts):
        return result
    i=headings[0]
    clean=compact(texts[i])
    if "单位:份" not in clean[clean.index("开放式基金份额变动"):][:160]:
        return result
    with pdfplumber.open(BASE/source["raw_path"]) as pdf:
        parsed=join_adjacent_tables(pdf.pages[i].extract_tables()+pdf.pages[i+1].extract_tables())
    result["cross_page_candidate"]=parsed
    if not parsed["status"].startswith("PASS_"):
        return result
    flow={**parsed,"source_page":i+1,"source_pages":[i+1,i+2],"unit":"FUND_UNITS","raw_page_texts":texts[i:i+2]}
    clock=available_clock(source,old["identity"]["reported_send_date"],old["pdf_metadata"],calendar)
    result.update({"flow_table":flow,"clock":clock,"source_usable":clock["source_usable"],
                   "status":"PASS_ORIGINAL_FACTS_AND_CLOCK" if clock["source_usable"] else clock["status"]})
    return result


def freeze():
    paths=[Path(__file__),ROOT/"research/original_fund_quarterly_facts_v1.py",
           ROOT/"tests/test_original_fund_quarterly_facts_v1_1.py",
           ROOT/"docs/510300_ORIGINAL_FUND_QUARTERLY_FACTS_V1_1.md",
           ROOT/"config/510300_original_fund_quarterly_facts_v1_manifest.json",
           OUT/"quarterly_facts_result.json",BASE/CALENDAR]
    paths+=sorted((OUT/"quarterly_fact_records").glob("*.json"))
    write_json(MANIFEST,{"registered_at":now(),"reason":"原有单页读取在九份报告遇到同一张表跨页，不读取策略收益修正格式",
                        "new_strategy_returns_read":False,"files":[
        {"path":p.relative_to(BASE).as_posix() if p.is_relative_to(BASE) else p.relative_to(ROOT).as_posix(),
         "sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]},exclusive=True)


def run():
    require(not (OUT/"quarterly_facts_v1_1_result.json").exists(),"跨页结果已经生成")
    for row in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]:
        path=(BASE if row["path"].startswith("data/") else ROOT)/row["path"]
        require(hashlib.sha256(path.read_bytes()).hexdigest()==row["sha256"],"跨页修正规则或输入变化")
    cal=pd.read_parquet(BASE/CALENDAR)
    rows=[]
    for path in sorted((OUT/"quarterly_fact_records").glob("*.json")):
        result=repair(json.loads(path.read_text(encoding="utf-8")),cal.loc[cal.is_open,"trade_date"])
        write_json(OUT/"quarterly_fact_records_v1_1"/path.name,result,exclusive=True)
        src,flow=result["source"],result.get("flow_table",{})
        values=flow.get("values",{})
        rows.append({"period":result["period"],"ts_code":"510300.SH","status":result["status"],
                     "source_usable":result["source_usable"],"publication_date":src.get("publication_date"),
                     "reported_send_date":result["identity"].get("reported_send_date"),
                     "feature_available_session":result.get("clock",{}).get("feature_available_session"),
                     "source_pages":json.dumps(flow.get("source_pages",[flow.get("source_page")])),
                     "source_url":src["url"],"source_sha256":src["sha256"],"identity_residual":flow.get("identity_residual"),**values})
        print(f"原始申赎跨页核对：{result['period']}，{result['status']}",flush=True)
    frame=pd.DataFrame(rows).sort_values("period").reset_index(drop=True)
    frame["net_subscription_units"]=frame.gross_subscription_units-frame.gross_redemption_units
    frame["net_units_divided_by_beginning_units"]=frame.net_subscription_units/frame.beginning_units
    frame["quarter_transition_residual"]=frame.beginning_units-frame.ending_units.shift(1)
    frame.to_parquet(OUT/"quarterly_share_flow_facts_v1_1.parquet",index=False)
    frame.to_csv(OUT/"原始季度申购赎回与可用日期_跨页修正版.csv",index=False,encoding="utf-8-sig")
    transitions=frame.loc[frame.quarter_transition_residual.notna()]
    bad=transitions.loc[transitions.quarter_transition_residual.abs()>.01]
    result={"completed_at":now(),"status":"SOURCE_FACTS_CORRECTED_WITH_ORIGINAL_CLOCK_EXCLUSION",
            "quarterly_reports":len(frame),"reports_with_verified_share_flow_values":int(frame.beginning_units.notna().sum()),
            "reports_with_facts_and_official_clock":int(frame.source_usable.sum()),
            "unresolved_reports":frame.loc[~frame.source_usable,["period","status"]].to_dict("records"),
            "verified_adjacent_quarter_transitions":len(transitions)-len(bad),
            "unexplained_quarter_transitions":bad[["period","quarter_transition_residual"]].to_dict("records"),
            "cash_flow_amount_reconstructed":False,"new_portfolio_evaluation":False,"prior_parser_outputs_preserved":True}
    write_json(OUT/"quarterly_facts_v1_1_result.json",result,exclusive=True)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description="原始季报同一张申赎表相邻跨页核对")
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze",action="store_true")
    group.add_argument("--run",action="store_true")
    args=parser.parse_args()
    freeze() if args.freeze else run()
