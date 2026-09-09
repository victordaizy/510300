"""独立只读核对已保存的盈利、前瞻预测、日期和版本差额，不训练或重算账户。"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads(path.read_text("utf-8"))


def clean(s):
    return re.sub(r"\s+","",unicodedata.normalize("NFKC",str(s)).replace("−","-").replace("—","-").replace("–","-"))


def number(v):
    t=clean(v).replace(",","").replace("人民币","").replace("元/股","").replace("元","")
    if t=="-":return None
    return -Decimal(t[1:-1]) if t.startswith("(") else Decimal(t)


def verify(root=ROOT):
    b=root/"reports/research"
    ar=b/"510300_financial_annual_components_v1"
    annual_facts=0;components=0;identities=0;pdfs={}
    for f in (ar/"document_records").glob("*.json"):
        d=read(f);s=d["source"]
        pages=read(b/"510300_financial_ttm_dependencies_v1/page_texts"/f.name)["pages"]
        pdfs[s["raw_path"]]=s["sha256"]
        for r in d["core_facts"]+d["additional_facts"]:
            assert clean(r["source_raw_row"]) in clean(pages[r["source_page"]-1]),(f.name,r["metric_id"])
            assert r["source_raw_value"]==r["all_numeric_cells"][r["selected_column_one_based"]-1]
            val=number(r["source_raw_value"])
            if r["source_unit_multiplier"] is None or val is None:assert r["metric_value_exact"] is None
            else:assert val*Decimal(r["source_unit_multiplier"])==Decimal(r["metric_value_exact"])
            assert not r["comparative_is_earlier_original"]
        annual_facts+=len(d["core_facts"]);components+=len(d["additional_facts"])
        for relation in d["identities"]:
            residual=[Decimal(a)-Decimal(c) for a,c in zip(relation["left"],relation["right"])]
            assert list(map(str,residual))==relation["residuals_reported_units"]
            assert all(abs(x)<=Decimal(relation["tolerance"]) for x in residual)
            identities+=1
    assert (annual_facts,components,identities)==(59,46,75)
    for name,n in [("current_original_core_facts",59),("ordinary_eps_components",46)]:
        df=pd.read_parquet(ar/(name+".parquet"));assert len(df)==n and not df.duplicated(["announcement_id","metric_id"]).any()
    vintage=b/"510300_financial_ttm_vintage_bridge_v1"
    rows=read(vintage/"vintage_bridges.json")["rows"]
    assert len(rows)==36
    complete=0
    for r in rows:
        p=r["provenance"]
        v={k:Decimal(x["value_exact"]) for k,x in p.items() if k in ["current","annual","original_prior"]}
        if r["original_three_report_arithmetic"] is not None:
            assert v["current"]+v["annual"]-v["original_prior"]==Decimal(r["original_three_report_arithmetic"])
            complete+=1
        if r["current_comparative_minus_original"] is not None:
            assert Decimal(r["current_report_comparative_prior"])-v["original_prior"]==Decimal(r["current_comparative_minus_original"])
        if r["annual_vintage_minus_original"] is not None:
            assert Decimal(r["annual_vintage_prior_ytd"])-v["original_prior"]==Decimal(r["annual_vintage_minus_original"])
        assert not r["basic_eps_arithmetic_performed"]
    assert complete==34
    for aid,q in read(vintage/"annual_quarter_breakdown.json").items():
        if q is None:continue
        for metric,r in q["rows"].items():
            values=[number(x) for x in r["cells"][:4]];s=q["quarter_sums"][metric];m=Decimal(s["source_unit_multiplier"])
            assert sum(values)*m==Decimal(s["full_year"])
            assert sum(values[:3])*m==Decimal(s["first_three_quarters"])
    pilot=b/"510300_forward_eps_source_pilot_v2"
    forecasts=[];eps_by_report={}
    for f in (pilot/"document_records").glob("*.json"):
        d=read(f);s=d["source"];pages=read(pilot/"page_texts"/f.name)["pages"]
        pdfs[s["raw_pdf_path"]]=s["pdf_sha256"]
        html=(root/s["raw_html_path"]).read_bytes();assert hashlib.sha256(html).hexdigest()==s["html_sha256"]
        text=html.decode("utf-8");m=re.search(r"var\s+zwinfo\s*=\s*",text)
        metadata=json.JSONDecoder().raw_decode(text[m.end():])[0]
        assert metadata==d["provider_metadata"] and metadata["info_code"]==s["report_id"]
        for r in d["forecasts"]:
            assert clean(r["source_raw_row"]) in clean(pages[0])
            col=r["selected_column_one_based"]-1
            assert r["header"][col]==[str(r["target_fiscal_year"]),"E"]
            mult=1 if r["metric_id"]=="ANALYST_ANNUAL_EPS_FORECAST" else 1000000
            assert number(r["source_raw_value"])*mult==Decimal(r["metric_value_exact"])
            assert r["conservative_information_date"]==max(r["report_internal_date"],r["provider_notice_date"][:10],r["provider_eitime"][:10])
            assert not r["actual_future_eps_used_as_predictor"] and not r["historical_immutable_snapshot_proven"]
            if mult==1:eps_by_report[(r["report_id"],r["target_fiscal_year"])]=Decimal(r["metric_value_exact"])
        forecasts+=d["forecasts"]
    assert len(forecasts)==30 and len(eps_by_report)==15
    links=read(pilot/"same_target_year_sampled_revisions.json")["rows"]
    for r in links:
        a=eps_by_report[(r["before_report_id"],r["target_fiscal_year"])];c=eps_by_report[(r["after_report_id"],r["target_fiscal_year"])];assert str(c-a)==r["absolute_revision"]
        assert str(c/a-1)==r["relative_revision"]
        assert r["unseen_intermediate_reports_may_exist"] and not r["is_complete_30_or_90_day_revision_factor"]
    api=b/"510300_forward_eps_public_api_probe_v1"
    probe=read(api/"result.json")
    for r in probe["rows"]:
        raw=read(api/(r["report_id"]+"_api.json"));assert str(raw["currentYear"])==r["interface_current_year"]
        if r["value_matches_interface_current_year_forecast"]:
            assert Decimal(r["interface_predict_this_year_eps"])==eps_by_report[(r["report_id"],int(r["interface_current_year"]))]
    history=b/"510300_financial_quarter_history_v1"
    h=read(history/"transport_retry_result.json")
    archived=[r for r in h["all_current_rows"] if r["status"].startswith("ORIGINAL_")]
    assert len(archived)==h["current_archived"]==251
    for s in archived:
        pdfs[s["raw_path"]]=s["sha256"]
        assert read(history/"page_texts"/(s["announcement_id"]+".json"))["source"]["sha256"]==s["sha256"]
    for path,digest in pdfs.items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path
    frozen=0
    names=["510300_financial_annual_components_v1_manifest.json","510300_financial_ttm_vintage_bridge_v1_manifest.json","510300_financial_quarter_history_v1_manifest.json",
        "510300_forward_eps_source_pilot_v1_manifest.json","510300_forward_eps_source_pilot_v1_1_manifest.json","510300_forward_eps_source_pilot_v2_manifest.json"]
    for name in names:
        for r in read(root/"config"/name)["files"]:
            assert hashlib.sha256((root/r["path"]).read_bytes()).hexdigest()==r["sha256"],r["path"]
            frozen+=1
    return {"status":"PASS_READ_ONLY_FINANCIAL_AND_FORWARD_EPS_SAVED_EVIDENCE", "checked_at":pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),
        "annual_core_facts":annual_facts,"annual_eps_component_records":components,"saved_accounting_relations":identities,"vintage_bridges":len(rows),
        "original_analyst_forecasts":len(forecasts),"annual_eps_forecasts":len(eps_by_report),"sampled_same_year_revision_links":len(links),"api_year_mapping_reports":len(probe["rows"]),
        "continuous_quarter_pdfs":len(archived),"unique_pdf_files_checked":len(pdfs),"frozen_references_checked":frozen,
        "new_strategy_return_calculation":False,"new_model_fit":False,"security_audit_performed":False}


if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",type=Path,default=ROOT);ap.add_argument("--output",type=Path);a=ap.parse_args();r=verify(a.root)
    if a.output:a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(r,ensure_ascii=False))
