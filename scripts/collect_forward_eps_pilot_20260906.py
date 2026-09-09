"""保存历史券商研报的年度前瞻每股收益与原始日期，验证免费来源可行性。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import pandas as pd
import pypdfium2 as pdfium
import requests

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.financial_annual_components_v1 import read,save,now,norm,compact,select_row,rows_in_pages,decimal

OUT=ROOT/"reports/research/510300_forward_eps_source_pilot_v1"
RAW=ROOT/"data/raw/510300_forward_eps_source_pilot_v1"
MANIFEST=ROOT/"config/510300_forward_eps_source_pilot_v1_manifest.json"
IDS=["AP202203101551739416","AP202403151626825160","AP202408161639304268","AP202503161644416306","AP202603221820687530"]
HEADERS={"User-Agent":"Mozilla/5.0","Referer":"https://data.eastmoney.com/report/"}


def freeze():
    paths=[Path(__file__),ROOT/"docs/510300_FORWARD_EPS_RESEARCH_DIRECTION_20260906.md",ROOT/"research/financial_annual_components_v1.py"]
    save(MANIFEST,{"registered_at":now(),"purpose":"前瞻年度每股收益的免费原件与日期可行性核对；不是回测候选筛选", "institution":"国信证券", "observation_company":"000001.SZ", "budget_cny":0,
        "selection":"网页搜索已展示这些报告的摘要和部分财务数值；按同机构同公司形成来源试验链，不宣称数值盲选或预测历史完整。", "report_ids":IDS,
        "files":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]},exclusive=True)
    print("已登记前瞻每股收益来源试验：五份历史研报。",flush=True)


def parse(pages,metadata,identifier):
    front=compact(pages[0])
    if "平安银行" not in front or "000001" not in front or "guosen.com.cn" not in front:
        raise ValueError("券商原件的公司或机构身份不符")
    if metadata["info_code"]!=identifier or "平安银行" not in json.dumps(metadata["security"],ensure_ascii=False):
        raise ValueError("网页证券目录与原件身份不符")
    report_date=re.search(r"证券研究报告[|｜]?\s*(20\d{2})年(\d{2})月(\d{2})日",norm(pages[0]))
    if not report_date:raise ValueError("原件落款日期未识别")
    report_date="-".join(report_date.groups())
    ls=rows_in_pages(pages,[1],"盈利预测和财务指标")
    # 先截取年份表头，不能按当前年份推测预测列。
    header=[]
    for r in ls[1:]:
        if "营业收入" in compact(r["text"]):break
        header.extend(re.findall(r"(?<!\d)(20\d{2})(E?)(?!\d)",compact(r["text"])))
    if len(header)!=5 or sum(e=="E" for y,e in header)!=3:
        raise ValueError("实际值与预测值年份表头不是登记的两实三预期格式")
    eps=select_row(ls,"摊薄每股收益",[5],allow_note=False)
    profit=select_row(ls,["归母净利润","净利润"],[5],allow_note=False)
    note=next((r for r in ls if "摊薄每股收益按最新总股本计算" in compact(r["text"])),None)
    if not note:raise ValueError("每股收益股本口径附注缺失")
    date_values=[report_date,str(metadata["notice_date"])[:10],str(metadata["eitime"])[:10]]
    later=max(date_values)
    forecasts=[]
    for i,(yr,flag) in enumerate(header):
        if flag!="E":continue
        for metric,row,mult,unit in [("ANALYST_ANNUAL_EPS_FORECAST",eps,1,"CNY_PER_SHARE"),("ANALYST_ANNUAL_PARENT_PROFIT_FORECAST",profit,1000000,"CNY")]:
            forecasts.append({"report_id":identifier,"ts_code":"000001.SZ","sec_name":"平安银行","institution":"国信证券","target_fiscal_year":int(yr),
                "metric_id":metric,"metric_value_exact":str(row["values"][i]*mult),"unit":unit,"source_page":1,"source_raw_row":row["raw_row"],"source_raw_value":row["cells"][i],"all_cells":row["cells"],"selected_column_one_based":i+1,"header":header,
                "report_internal_date":report_date,"provider_notice_date":metadata["notice_date"],"provider_eitime":metadata["eitime"],"conservative_information_date":later,
                "provider_date_fields_are_currently_retrieved_historical_metadata":True,"historical_immutable_snapshot_proven":False,
                "eps_definition":"分析师预测的摊薄每股收益，原文说明按最新总股本计算；未冒充财报基本每股收益",
                "basis_note_raw":note["raw"],"status":"HISTORICAL_DATED_ANALYST_FORECAST_SOURCE_PILOT_NOT_PORTFOLIO_ADMITTED","actual_future_eps_used_as_predictor":False})
    return {"forecasts":forecasts,"report_internal_date":report_date,"date_fields_differ":len(set(date_values))>1,
        "original_table_rows":{"eps":eps,"parent_profit":profit},"eps_basis_note":note,"header":header}


def run():
    for r in read(MANIFEST)["files"]:
        if hashlib.sha256((ROOT/r["path"]).read_bytes()).hexdigest()!=r["sha256"]:raise ValueError("登记代码或说明变化")
    if (OUT/"result.json").exists():raise FileExistsError("已完成的来源试验不覆盖")
    RAW.mkdir(parents=True,exist_ok=True)
    records=[]
    for identifier in IDS:
        receipt=OUT/"document_records"/f"{identifier}.json"
        if receipt.exists():records.append(read(receipt));continue
        url="https://data.eastmoney.com/report/zw_stock.jshtml?infocode="+identifier
        response=requests.get(url,headers=HEADERS,timeout=(12,35));response.raise_for_status()
        html=response.content
        text=response.text
        m=re.search(r"var\s+zwinfo\s*=\s*",text)
        if not m:raise ValueError("公开页面未含原始研报元数据")
        meta=json.JSONDecoder().raw_decode(text[m.end():])[0]
        pdf_url=meta["attach_url"]
        if not pdf_url.startswith("https://pdf.dfcfw.com/pdf/"):raise ValueError("页面返回未知附件域")
        pdf=requests.get(pdf_url,headers=HEADERS,timeout=(12,35));pdf.raise_for_status()
        if not pdf.content.startswith(b"%PDF-"):raise ValueError("附件不是 PDF")
        html_path=RAW/(identifier+".html");html_path.write_bytes(html)
        pdf_path=RAW/(identifier+".pdf");pdf_path.write_bytes(pdf.content)
        doc=pdfium.PdfDocument(pdf.content);pages=[]
        try:
            for i in range(len(doc)):
                page=doc[i];tp=page.get_textpage()
                try:pages.append(tp.get_text_range())
                finally:tp.close();page.close()
        finally:doc.close()
        source={"report_id":identifier,"provider_detail_url":url,"raw_html_path":html_path.relative_to(ROOT).as_posix(),"raw_pdf_path":pdf_path.relative_to(ROOT).as_posix(),"pdf_url":pdf_url,
                "pdf_sha256":hashlib.sha256(pdf.content).hexdigest(),"html_sha256":hashlib.sha256(html).hexdigest(),"retrieved_at":now(),"pdf_pages":len(pages)}
        save(OUT/"page_texts"/(identifier+".json"),{"source":source,"pages":pages})
        d={"source":source,"provider_metadata":meta,**parse(pages,meta,identifier)}
        save(receipt,d);records.append(d)
        print("前瞻每股收益原件",identifier,d["report_internal_date"],len(d["forecasts"]),flush=True)
    facts=[r for d in records for r in d["forecasts"]]
    df=pd.DataFrame(facts)
    df.to_parquet(OUT/"annual_forecast_vintages.parquet",index=False)
    df.to_csv(OUT/"原始前瞻年度每股收益与利润预测.csv",index=False,encoding="utf-8-sig")
    links=[]
    eps=df.loc[df.metric_id.eq("ANALYST_ANNUAL_EPS_FORECAST")].copy()
    for year,g in eps.groupby("target_fiscal_year"):
        saved=g.sort_values(["conservative_information_date","report_id"]).to_dict("records")
        for before,after in zip(saved,saved[1:]):
            a,b=decimal(before["metric_value_exact"]),decimal(after["metric_value_exact"])
            links.append({"institution":"国信证券","ts_code":"000001.SZ","target_fiscal_year":int(year),"before_report_id":before["report_id"],"after_report_id":after["report_id"],"before_eps":str(a),"after_eps":str(b),
                "absolute_revision":str(b-a),"relative_revision":str(b/a-1) if a!=0 else None,
                "days_between_saved_reports":(pd.Timestamp(after["conservative_information_date"])-pd.Timestamp(before["conservative_information_date"])).days,
                "unseen_intermediate_reports_may_exist":True,"is_complete_30_or_90_day_revision_factor":False})
    save(OUT/"same_target_year_sampled_revisions.json",{"rows":links},exclusive=True)
    result={"completed_at":now(),"status":"FIVE_ORIGINAL_FORWARD_EPS_REPORTS_ARCHIVED_CONTINUOUS_ANALYST_HISTORY_NEXT","reports":len(records),"institutions":1,"observation_companies":1,"forecast_eps_facts":len(eps),"forecast_parent_profit_facts":len(df)-len(eps),
            "sampled_same_year_revision_links":len(links),"report_and_provider_dates_differ":sum(d["date_fields_differ"] for d in records),"free_source_cost_cny":0,
            "is_market_consensus":False,"exact_next_twelve_month_eps_constructed":False,"historical_point_in_time_portfolio_admission":False,"new_strategy_accounts":0,"goal_achieved":False}
    save(OUT/"result.json",result,exclusive=True);print(result,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument("--freeze",action="store_true");g.add_argument("--run",action="store_true");a=ap.parse_args()
    freeze() if a.freeze else run()
