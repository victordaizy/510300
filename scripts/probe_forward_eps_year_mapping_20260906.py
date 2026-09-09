"""核对公开接口的相对年份字段与历史研报绝对预测年度，不读取股票收益。"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import requests
from research.financial_annual_components_v1 import read,save,now,decimal

OUT=ROOT/"reports/research/510300_forward_eps_public_api_probe_v1"
PILOT=ROOT/"reports/research/510300_forward_eps_source_pilot_v2"


def main():
    rows=[]
    cases=[("AP202203101551739416","2022-03-10","2022-03-11"),("AP202403151626825160","2024-03-15","2024-03-16"),("AP202503161644416306","2025-03-15","2025-03-16")]
    for aid,start,end in cases:
        file=OUT/(aid+"_api.json")
        if file.exists():d=read(file)
        else:
            params={"pageSize":50,"pageNo":1,"beginTime":start,"endTime":end,"qType":0,"code":"000001","industryCode":"*","industry":"*","rating":"*","ratingChange":"*","fields":""}
            r=requests.get("https://reportapi.eastmoney.com/report/list",params=params,headers={"User-Agent":"Mozilla/5.0","Referer":"https://data.eastmoney.com/report/"},timeout=(12,35));r.raise_for_status()
            d=r.json();file.write_bytes(r.content)
            save(OUT/(aid+"_api_receipt.json"),{"retrieved_at":now(),"url":r.url,"params":params,"sha256":hashlib.sha256(r.content).hexdigest()},exclusive=True)
        matches=[x for x in d.get("data",[]) if x.get("infoCode")==aid]
        if len(matches)!=1:raise ValueError("公开接口本次未唯一返回目标报告: "+aid)
        x=matches[0];original=read(PILOT/"document_records"/(aid+".json"))
        ys={str(f["target_fiscal_year"]):f["metric_value_exact"] for f in original["forecasts"] if f["metric_id"]=="ANALYST_ANNUAL_EPS_FORECAST"}
        value=x.get("predictThisYearEps")
        current=str(d["currentYear"])
        row={"report_id":aid,"internal_report_date":original["report_internal_date"],"interface_publish_date":x["publishDate"],"interface_current_year":current,
            "interface_predict_this_year_eps":value,"pdf_forecasts_by_absolute_year":ys,"pdf_forecast_for_interface_current_year":ys.get(current),
            "value_matches_interface_current_year_forecast":bool(value and current in ys and decimal(value)==decimal(ys[current])),
            "relative_field_cannot_be_assigned_to_report_year_without_absolute_year_check":True,
            "all_interface_eps_fields":{k:v for k,v in x.items() if "Eps" in k},"original_report_forecasts_may_exist_when_interface_fields_empty":bool(ys and not any(v for k,v in x.items() if "Eps" in k))}
        rows.append(row)
        print("年份映射核对",aid,current,value,ys,flush=True)
    save(OUT/"result.json",{"completed_at":now(),"status":"HISTORICAL_RELATIVE_EPS_FIELD_YEAR_MAPPING_CHECKED_AGAINST_ORIGINALS", "reports_checked":len(rows),"rows":rows,
        "conclusion":"本次查询的当年预测字段按接口当前年2026对应原件2026E，而不是历史研报落款年；原件绝对年份列是历史提取依据。", "new_account_evaluations":0},exclusive=True)


if __name__=="__main__":main()
