"""区分原始版本、年报重述分季及次年比较数，构建滚动盈利的明确数值证据。"""
from __future__ import annotations

import argparse
import hashlib
from decimal import Decimal
from pathlib import Path

import pandas as pd

from research.financial_annual_components_v1 import ROOT, SOURCE, OUT as ANNUAL, read, save, now, decimal, compact, rows_in_pages, select_row

OUT = ROOT / "reports/research/510300_financial_ttm_vintage_bridge_v1"
MANIFEST = ROOT / "config/510300_financial_ttm_vintage_bridge_v1_manifest.json"
OLD = ROOT / "reports/research/510300_financial_original_facts_v1_1"
METRICS = ["OPERATING_REVENUE_YTD", "OPERATING_PROFIT_YTD", "PARENT_NET_PROFIT_YTD"]
QUARTERS = {
    "1219306493": {"page":17,"start":"分季度主要财务指标"},
    "1212751519": {"page":22,"start":"主要财务指标(合并报表)","stop":"主要财务指标(母公司)"},
    "1212709852": {"page":11,"start":"九、2021年分季度主要财务数据"},
    "1212626208": {"page":17,"start":"按季度披露的经营业绩指标"},
    "1216222871": {"page":18,"start":"十、2022年分季度主要财务数据"},
    "1212745096": {"page":18,"start":"十、2021年分季度主要财务数据"},
    "1209490205": {"page":11,"start":"2.2.4季度数据"},
    "1219376072": {"page":11,"start":"分季度主要财务数据","stop":"非经常性损益项目"},
    "1212698832": {"page":39,"start":"2021年分季度主","parent_label":"净利润注"},
    "1212669927": {"page":9,"start":"2021年分季度主要财务数据","stop":"非经常性损益项目和金额"},
    "1212730963": {"page":16,"start":"下表列出所示期间本集团分季度","width":8},
}
POLICY_PAGES = {
    "1205547286": {"pages":[3,4], "kind":"NEW_FINANCIAL_INSTRUMENT_RULES_NOT_RESTATED_WITH_PRESENTATION_RECLASSIFICATION", "description":"交通银行2018年改用新金融工具准则，前期不重述；另有列报重分类。归母比较数相同也不能证明准则完全可比。"},
    "1214937073": {"pages":[2], "kind":"COMMON_CONTROL_COMBINATION_RESTATEMENT", "description":"中国人寿次年报告说明上年发生同一控制下合并，前三季度数据重述。"},
    "1218167349": {"pages":[3], "kind":"DEFERRED_TAX_INTERPRETATION_16_TRANSITION", "description":"招商证券2023年执行准则解释第16号的递延所得税规定，调整比较期。"},
}


def facts(doc):
    return doc.get("facts",doc.get("core_facts",[]))


def get(doc, metric):
    return next((r for r in facts(doc) if r["metric_id"] == metric),None)


def exact(row):
    # 旧表浮点列只供显示，计算重新使用原始十进制单元格。
    return decimal(row["source_raw_value"])*Decimal(row["source_unit_multiplier"])


def proof(row, source):
    return {"announcement_id":source["announcement_id"], "report_period":str(source["report_period"])[:10],
        "event_publication_date":str(source["event_publication_date"])[:10], "official_pdf_url":source["official_pdf_url"],
        "source_page":row["source_page"],"raw_row":row["source_raw_row"], "value_exact":str(exact(row)),
        "source_unit_multiplier":row["source_unit_multiplier"]}


def extract_quarters(aid, doc, root=ROOT):
    p=QUARTERS.get(aid)
    if p is None:
        return None
    pages=read(root / "reports/research/510300_financial_ttm_dependencies_v1/page_texts" / f"{aid}.json")["pages"]
    lines=rows_in_pages(pages,[p["page"]],p["start"],p.get("stop"))
    result={"announcement_id":aid,"page":p["page"],"rows":{},"quarter_sums":{}}
    for metric,labels in {
        "OPERATING_REVENUE_YTD":["营业总收入","营业收入"],
        "PARENT_NET_PROFIT_YTD":[p["parent_label"]] if "parent_label" in p else ["归属于母公司股东的净利润","归属于上市公司股东的净利润","归属于本行股东的净利润"],
    }.items():
        row=select_row(lines,labels,[p.get("width",4)],allow_note=False)
        vals=row["values"][:4]
        original=get(doc,metric)
        multiplier=Decimal(original["source_unit_multiplier"])
        total=sum(vals)*multiplier
        if total!=exact(original):
            raise ValueError("全年分季加总不等于全年主表: "+aid+" "+metric)
        result["rows"][metric]=row
        result["quarter_sums"][metric]={"first_quarter":str(vals[0]*multiplier),"first_three_quarters":str(sum(vals[:3])*multiplier),
            "fourth_quarter":str(vals[3]*multiplier),"full_year":str(total),"source_unit_multiplier":int(multiplier),
            "later_annual_vintage_not_original_quarter":True}
    return result


def build(root=ROOT):
    old={}
    for f in (root / "reports/research/510300_financial_original_facts_v1_1/document_records").glob("*.json"):
        d=read(f)
        if facts(d): old[d["source"]["announcement_id"]]=d
    new={f.stem:read(f) for f in (root / "reports/research/510300_financial_annual_components_v1/document_records").glob("*.json")}
    lookup={(d["source"]["ts_code"],str(d["source"]["report_period"])[:10]):d for d in list(old.values())+list(new.values())}
    qtr={aid:extract_quarters(aid,d,root) for aid,d in new.items() if str(d["source"]["report_period"])[5:7]=="12"}
    policy={}
    for aid,p in POLICY_PAGES.items():
        d=read(root / "reports/research/510300_financial_original_layout_inventory_v1/page_texts" / f"{aid}.json")
        policy[aid]={**p,"source":d["source"],"raw_pages":[{"page":n,"text":d["pages"][n-1]} for n in p["pages"]]}
    rows=[]
    for a in read(root / "reports/research/510300_financial_ttm_dependencies_v1/dependency_map.json")["anchors"]:
        current=old[a["announcement_id"]]
        s=current["source"]
        yr=int(str(s["report_period"])[:4]); suffix=str(s["report_period"])[4:10]
        prev=lookup[(s["ts_code"],str(yr-1)+suffix)]
        annual=lookup[(s["ts_code"],str(yr-1)+"-12-31")]
        q=qtr[annual["source"]["announcement_id"]]
        for metric in METRICS:
            c,pr,fy=get(current,metric),get(prev,metric),get(annual,metric)
            r={"ts_code":s["ts_code"],"sec_name":s["sec_name"],"anchor_period":str(s["report_period"])[:10],"metric_id":metric,
               "anchor_announcement_id":a["announcement_id"],"annual_announcement_id":annual["source"]["announcement_id"],"prior_quarter_announcement_id":prev["source"]["announcement_id"],
               "latest_component_publication_date":max(str(d["source"]["event_publication_date"])[:10] for d in [current,prev,annual]),
               "original_three_report_arithmetic":None,"current_report_comparative_prior":None,"current_comparative_minus_original":None,
               "annual_vintage_prior_ytd":None,"annual_vintage_minus_original":None,"annual_vintage_residual_plus_current":None,
               "originals_currency_explicit":all(f is not None and f.get("currency_explicit_in_statement",True) for f in [c,pr,fy]),
               "original_arithmetic_status":"NO_VIEW_MISSING_ORIGINAL_METRIC","comparison_status":"NO_VIEW_NOT_EVALUATED",
               "quarter_bridge_status":"NO_VIEW_NO_ANNUAL_QUARTER_BREAKDOWN_FOR_METRIC","research_value_status":"NOT_ADMITTED_TO_STRATEGY",
               "full_accounting_basis_reconciliation_completed":False,"basic_eps_arithmetic_performed":False,"provenance":{}}
            if a["announcement_id"] in policy:
                r["known_policy_evidence"]=policy[a["announcement_id"]]
            for k,f,d in [("current",c,current),("original_prior",pr,prev),("annual",fy,annual)]:
                if f is not None:r["provenance"][k]=proof(f,d["source"])
            if c is not None and pr is not None and fy is not None:
                r["original_three_report_arithmetic"]=str(exact(c)+exact(fy)-exact(pr))
                r["original_arithmetic_status"]="ARITHMETIC_COMPLETE_EXPLICIT_CNY" if r["originals_currency_explicit"] else "NO_VIEW_CURRENCY_NOT_EXPLICIT"
            if c is not None:
                # 原表首列/第三列的紧邻比较列；平安银行摘要的相邻列是增长率，不能读取。
                if s["ts_code"]=="000001.SZ" and metric=="PARENT_NET_PROFIT_YTD":
                    r["comparison_status"]="NO_VIEW_SUMMARY_ADJACENT_COLUMN_IS_PERCENT_NOT_PARENT_PROFIT"
                else:
                    col=c["selected_numeric_column_one_based"]
                    cells=c["all_numeric_cells_for_verification"]
                    comparative=decimal(cells[col])*Decimal(c["source_unit_multiplier"])
                    r["current_report_comparative_prior"]=str(comparative)
                    r["provenance"]["current_comparative"]={"announcement_id":s["announcement_id"],"source_page":c["source_page"],"raw_row":c["source_raw_row"],"selected_column_one_based":col+1,"raw_value":cells[col],"source_unit_multiplier":c["source_unit_multiplier"],"is_original_prior":False}
                    if pr is not None:
                        diff=comparative-exact(pr)
                        r["current_comparative_minus_original"]=str(diff)
                        r["comparison_status"]="SAME_REPORTED_VALUE_AS_ORIGINAL_PRIOR" if diff==0 else "LATER_COMPARATIVE_DIFFERS_FROM_ORIGINAL_PRIOR"
                    else:r["comparison_status"]="NO_VIEW_ORIGINAL_PRIOR_METRIC_MISSING"
            if q and metric in q["quarter_sums"]:
                sums=q["quarter_sums"][metric]
                pq=Decimal(sums["first_quarter"] if suffix=="-03-31" else sums["first_three_quarters"])
                r["annual_vintage_prior_ytd"]=str(pq)
                r["provenance"]["annual_quarters"]={"announcement_id":annual["source"]["announcement_id"],"row":q["rows"][metric],"sums":sums}
                if pr is not None:r["annual_vintage_minus_original"]=str(pq-exact(pr))
                if c is not None:
                    r["annual_vintage_residual_plus_current"]=str(exact(c)+exact(fy)-pq)
                    cp=r["current_report_comparative_prior"]
                    if cp is not None:
                        r["quarter_bridge_status"]="ANNUAL_QUARTERS_AGREE_WITH_CURRENT_COMPARATIVE" if pq==Decimal(cp) else "NO_VIEW_ANNUAL_QUARTERS_DIFFER_FROM_CURRENT_COMPARATIVE"
                    else:r["quarter_bridge_status"]="ANNUAL_RESIDUAL_AVAILABLE_CURRENT_PARENT_COMPARISON_NOT_EXPLICIT"
            if r["originals_currency_explicit"] and r["quarter_bridge_status"]=="ANNUAL_QUARTERS_AGREE_WITH_CURRENT_COMPARATIVE":
                r["research_value_status"]="VALUE_WITH_MATCHED_ANNUAL_QUARTER_BRIDGE_ONLY_CONTINUOUS_HISTORY_REQUIRED"
            if r["original_arithmetic_status"]=="NO_VIEW_CURRENCY_NOT_EXPLICIT":
                r["research_value_status"]="NO_VIEW_CURRENCY_NOT_EXPLICIT"
            if a["announcement_id"]=="1205547286":
                r["research_value_status"]="NO_VIEW_KNOWN_NEW_INSTRUMENT_RULES_WITHOUT_PRIOR_RESTATEMENT"
            rows.append(r)
    result={"completed_at":now(),"status":"TWELVE_ANCHOR_VINTAGE_BRIDGES_COMPLETED_CONTINUOUS_HISTORY_NEXT","anchors":12,"requested_metric_bridges":len(rows),
        "complete_original_arithmetic":sum(r["original_three_report_arithmetic"] is not None for r in rows),
        "explicit_cny_original_arithmetic":sum(r["original_arithmetic_status"]=="ARITHMETIC_COMPLETE_EXPLICIT_CNY" for r in rows),
        "later_comparative_differs":sum(r["comparison_status"]=="LATER_COMPARATIVE_DIFFERS_FROM_ORIGINAL_PRIOR" for r in rows),
        "annual_quarter_rows":sum(len(q["rows"]) for q in qtr.values() if q),
        "matched_annual_quarter_bridges_with_cny":sum(r["research_value_status"].startswith("VALUE_") for r in rows),
        "annual_quarter_current_comparison_mismatches":sum(r["quarter_bridge_status"]=="NO_VIEW_ANNUAL_QUARTERS_DIFFER_FROM_CURRENT_COMPARATIVE" for r in rows),
        "new_account_evaluations":0,"new_basic_eps_ttm_values":0,"goal_achieved":False,"strategy_return_data_read":False}
    return rows,qtr,result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--preflight",action="store_true");ap.add_argument("--freeze",action="store_true");args=ap.parse_args()
    if args.freeze:
        paths=[Path(__file__),ROOT/"research/financial_annual_components_v1.py",ROOT/"docs/510300_FINANCIAL_TTM_VINTAGE_BRIDGE_V1.md",ROOT/"tests/test_financial_ttm_vintage_bridge_v1.py", SOURCE/"dependency_map.json"]
        paths+=sorted((OLD/"document_records").glob("*.json"))+sorted((ANNUAL/"document_records").glob("*.json"))
        paths+=sorted((SOURCE/"page_texts").glob("*.json"))
        paths+=[ROOT/"reports/research/510300_financial_original_layout_inventory_v1/page_texts"/f"{aid}.json" for aid in POLICY_PAGES]
        save(MANIFEST,{"frozen_at":now(),"strategy_returns_read":False,"annual_quarter_profiles":QUARTERS,"files":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]},exclusive=True)
        print("滚动盈利版本桥接规则与来源已冻结。")
        return
    if not args.preflight:
        for r in read(MANIFEST)["files"]:
            if hashlib.sha256((ROOT/r["path"]).read_bytes()).hexdigest()!=r["sha256"]:raise ValueError("冻结输入变化: "+r["path"])
        if (OUT/"result.json").exists():raise FileExistsError("不能覆盖已完成版本")
    rows,qtr,result=build()
    if not args.preflight:
        save(OUT/"vintage_bridges.json",{"rows":rows},exclusive=True)
        save(OUT/"annual_quarter_breakdown.json",qtr,exclusive=True)
        pd.DataFrame([{k:v for k,v in r.items() if k!="provenance"} for r in rows]).to_csv(OUT/"滚动盈利与版本差异.csv",index=False,encoding="utf-8-sig")
        save(OUT/"result.json",result,exclusive=True)
    print(result)
    print(pd.DataFrame(rows)[["sec_name","metric_id","original_three_report_arithmetic","current_comparative_minus_original","annual_vintage_minus_original","quarter_bridge_status"]].to_string(index=False))


if __name__=="__main__":main()
