"""修正首份研报的日期空格和同一行年份表头，保留首次来源试验记录。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import collect_forward_eps_pilot_20260906 as base
from research.financial_annual_components_v1 import read,save,now,norm,compact,select_row,rows_in_pages

OUT=ROOT/"reports/research/510300_forward_eps_source_pilot_v1_1"
RAW=ROOT/"data/raw/510300_forward_eps_source_pilot_v1_1"
MANIFEST=ROOT/"config/510300_forward_eps_source_pilot_v1_1_manifest.json"


def parse(pages,metadata,identifier):
    front=compact(pages[0])
    if "平安银行" not in front or "000001" not in front or "guosen.com.cn" not in front:
        raise ValueError("原件公司或券商身份不符")
    if metadata["info_code"]!=identifier or "平安银行" not in json.dumps(metadata["security"],ensure_ascii=False):
        raise ValueError("网页证券目录不符")
    match=re.search(r"证券研究报告[|｜]?(20\d{2})年(\d{2})月(\d{2})日",front)
    if not match:raise ValueError("原件报告日期未识别")
    report_date="-".join(match.groups())
    ls=rows_in_pages(pages,[1],"盈利预测和财务指标")
    header=[]
    # 表头可与栏目名称在同一行；保留年份之间的空格，避免把年份连接成一个数字。
    for r in ls:
        if "营业收入" in compact(r["text"]):break
        header.extend(re.findall(r"(?<!\d)(20\d{2})(E?)(?!\d)",norm(r["text"])))
    if len(header)!=5 or sum(e=="E" for _,e in header)!=3:raise ValueError("不是明确的两年已实现、三年预测表头")
    eps=select_row(ls,"摊薄每股收益",[5],allow_note=False)
    profit=select_row(ls,["归母净利润","净利润"],[5],allow_note=False)
    note=next((r for r in ls if "摊薄每股收益按最新总股本计算" in compact(r["text"])),None)
    if note is None:raise ValueError("股本口径附注缺失")
    dates=[report_date,str(metadata["notice_date"])[:10],str(metadata["eitime"])[:10]]
    forecasts=[]
    for i,(year,flag) in enumerate(header):
        if flag!="E":continue
        for metric,row,mult,unit in [("ANALYST_ANNUAL_EPS_FORECAST",eps,1,"CNY_PER_SHARE"),("ANALYST_ANNUAL_PARENT_PROFIT_FORECAST",profit,1000000,"CNY")]:
            forecasts.append({"report_id":identifier,"ts_code":"000001.SZ","sec_name":"平安银行","institution":"国信证券","target_fiscal_year":int(year),
                "metric_id":metric,"metric_value_exact":str(row["values"][i]*mult),"unit":unit,"source_page":1,"source_raw_row":row["raw_row"],"source_raw_value":row["cells"][i],"all_cells":row["cells"],"selected_column_one_based":i+1,"header":header,
                "report_internal_date":report_date,"provider_notice_date":metadata["notice_date"],"provider_eitime":metadata["eitime"],"conservative_information_date":max(dates),
                "provider_date_fields_are_currently_retrieved_historical_metadata":True,"historical_immutable_snapshot_proven":False,
                "eps_definition":"分析师预测的摊薄每股收益，原文说明按最新总股本计算；未冒充财报基本每股收益",
                "basis_note_raw":note["raw"],"status":"HISTORICAL_DATED_ANALYST_FORECAST_SOURCE_PILOT_NOT_PORTFOLIO_ADMITTED","actual_future_eps_used_as_predictor":False})
    return {"forecasts":forecasts,"report_internal_date":report_date,"date_fields_differ":len(set(dates))>1,
            "original_table_rows":{"eps":eps,"parent_profit":profit},"eps_basis_note":note,"header":header}


def freeze():
    paths=[Path(__file__),ROOT/"scripts/collect_forward_eps_pilot_20260906.py",ROOT/"research/financial_annual_components_v1.py",ROOT/"docs/510300_FORWARD_EPS_RESEARCH_DIRECTION_20260906.md",ROOT/"config/510300_forward_eps_source_pilot_v1_manifest.json"]
    save(MANIFEST,{"registered_at":now(),"amendment":"只修正首份PDF已观察到的日期空格、年份与栏目同一行，以及保留年份之间的分隔；报告范围、目标年度、数值选择不改变。",
        "first_attempt":"首次采集保留原件，日期表达式未识别，未产生预测事实或回测。", "reports":base.IDS,"budget_cny":0,
        "files":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]},exclusive=True)
    print("前瞻研报版式修正已登记。",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument("--freeze",action="store_true");g.add_argument("--run",action="store_true");a=ap.parse_args()
    if a.freeze:freeze()
    else:
        base.OUT,base.RAW,base.MANIFEST,base.parse=OUT,RAW,MANIFEST,parse
        base.run()
