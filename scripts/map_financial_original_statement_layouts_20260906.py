"""缓存24份原始金融季报的逐页文本，并定位财务指标、合并利润与EPS说明。"""
from __future__ import annotations
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.original_fund_quarterly_facts_v1 import page_texts,compact
from research.intraday_overnight_increment_v1 import now,require,write_json

BASE=Path(r"E:\ResearchData\New project 8")
OUT=ROOT/"reports/research/510300_financial_original_layout_inventory_v1"
SOURCE=ROOT/"reports/research/510300_original_earnings_source_completion_v1/batch_01_result.json"


def run():
    require(not (OUT/"result.json").exists(),"原始金融版式定位已完成")
    rows=json.loads(SOURCE.read_text(encoding="utf-8"))["rows"]
    records=[]
    for n,source in enumerate(rows,1):
        path=BASE/source["raw_path"]
        content=path.read_bytes()
        require(hashlib.sha256(content).hexdigest()==source["sha256"],"金融原始PDF与收据不一致")
        cache=OUT/"page_texts"/(source["announcement_id"]+".json")
        if cache.exists():
            saved=json.loads(cache.read_text(encoding="utf-8"))
            require(saved["sha256"]==source["sha256"],"已缓存金融文本来源变化")
            texts=saved["pages"]
        else:
            texts=page_texts(content)
            write_json(cache,{"saved_at":now(),"sha256":source["sha256"],"source":source,"pages":texts},exclusive=True)
        statements=[]
        for i,text in enumerate(texts):
            normalized=compact(text)
            section={"page":i+1,"consolidated_income_in_heading":"合并利润表" in normalized[:550],
                     "basic_eps_present":"基本每股收益" in normalized or "基本及稀释每股收益" in normalized,
                     "financial_summary_present":any(x in normalized[:1800] for x in ["主要会计数据","主要财务数据","主要会计数据和财务指标","主要财务指标"]),
                     "ordinary_equity_adjustment_note":any(x in normalized for x in ["优先股股息","优先股股利","永续债利息","永续债分配","扣除其他权益","普通股股东的净利润"])}
            if any(v for k,v in section.items() if k!="page"):
                statements.append(section)
        first=compact("\n".join(texts[:4]))
        code=source["ts_code"].split(".")[0]
        record={"announcement_id":source["announcement_id"],"ts_code":source["ts_code"],"sec_name":source["sec_name"],
                "report_period":source["report_period"],"source_sha256":source["sha256"],"page_count":len(texts),
                "security_code_text_present_first_four_pages":code in first,"candidate_pages":statements,
                "fact_extraction_completed":False,"point_in_time_source_admitted":False}
        records.append(record)
        print(f"原始金融版式定位 {n}/{len(rows)}：{source['sec_name']}，{len(texts)}页，候选页{[x['page'] for x in statements]}",flush=True)
    write_json(OUT/"result.json",{"completed_at":now(),"status":"ORIGINAL_LAYOUT_AND_EPS_NOTE_LOCATIONS_CACHED_NOT_FACT_ADMISSION",
                "documents":len(records),"records":records,"new_strategy_returns_read":False,"new_portfolio_evaluation":False},exclusive=True)


if __name__=="__main__":
    run()
