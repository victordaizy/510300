"""处理研报同一文字行连写两项财务指标的版式，保持原件和预测年度不变。"""
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
from scripts import collect_forward_eps_pilot_v1_1_20260906 as first
from research.financial_annual_components_v1 import read,save,now,norm,decimal,compact

OUT=ROOT/"reports/research/510300_forward_eps_source_pilot_v2"
RAW=ROOT/"data/raw/510300_forward_eps_source_pilot_v2"
MANIFEST=ROOT/"config/510300_forward_eps_source_pilot_v2_manifest.json"


def forecast_row(lines,labels,widths,**kwargs):
    if widths!=[5]:raise ValueError("本版只接纳五个明确年度列")
    text="\n".join(r["raw"] for r in lines)
    text=norm(text)
    number=r"\(?[-+]?\d[\d,]*(?:\.\d+)?\)?"
    found=[]
    for label in ([labels] if isinstance(labels,str) else labels):
        pattern=r"(?m)^\s*"+re.escape(label)+r"\s*\((?:元|百万元)\)\s*"+r"\s+".join("("+number+")" for _ in range(5))+r"(?=\s*(?:[^\d\s.,%+\-()]|$))"
        for m in re.finditer(pattern,text):
            cells=list(m.groups())
            found.append({"label":label,"page":1,"raw_row":m.group(0).strip(),"cells":cells,"values":[decimal(v) for v in cells],"footnote_or_formula":None})
    if len(found)!=1:raise ValueError("五年度预测行不唯一或未识别: "+str(labels))
    return found[0]


def parse(pages,metadata,identifier):
    # 只替换前瞻研报局部行读取；冻结的年报/季报读取器不修改。
    first.select_row=forecast_row
    return first.parse(pages,metadata,identifier)


def freeze():
    paths=[Path(__file__),ROOT/"scripts/collect_forward_eps_pilot_20260906.py",ROOT/"scripts/collect_forward_eps_pilot_v1_1_20260906.py",ROOT/"research/financial_annual_components_v1.py",ROOT/"docs/510300_FORWARD_EPS_RESEARCH_DIRECTION_20260906.md",ROOT/"config/510300_forward_eps_source_pilot_v1_manifest.json",ROOT/"config/510300_forward_eps_source_pilot_v1_1_manifest.json"]
    save(MANIFEST,{"registered_at":now(),"amendment":"2024年原件的每股收益行与总资产收益率连写；按五个已确认年度列和下一文字指标边界提取，不要求整行只含一个指标。",
        "prior_attempts_preserved":True,"selected_reports_unchanged":base.IDS,"budget_cny":0,"strategy_return_outputs_read":False,
        "files":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]},exclusive=True)
    print("五年度前瞻预测表格版式已冻结。",flush=True)


def preflight():
    for identifier in base.IDS[:2]:
        d=read(ROOT/"reports/research/510300_forward_eps_source_pilot_v1_1/page_texts"/(identifier+".json"))
        html=(ROOT/d["source"]["raw_html_path"]).read_text("utf-8")
        m=re.search(r"var\s+zwinfo\s*=\s*",html);meta=json.JSONDecoder().raw_decode(html[m.end():])[0]
        result=parse(d["pages"],meta,identifier)
        assert len(result["forecasts"])==6
        for r in result["forecasts"]:
            assert compact(r["source_raw_row"]) in compact(d["pages"][0])
            assert r["header"][r["selected_column_one_based"]-1][1]=="E"
        print("前瞻表头与拼行预检查通过",identifier,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument("--freeze",action="store_true");g.add_argument("--run",action="store_true");g.add_argument("--preflight",action="store_true");a=ap.parse_args()
    if a.preflight:preflight()
    elif a.freeze:freeze()
    else:
        base.OUT,base.RAW,base.MANIFEST,base.parse=OUT,RAW,MANIFEST,parse
        base.run()
