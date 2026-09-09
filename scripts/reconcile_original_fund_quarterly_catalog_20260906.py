"""处理早期季报简称，并独立保留缺少上交所目录的巨潮原始报告。"""
from __future__ import annotations
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.intraday_overnight_increment_v1 import now,require,write_json
from scripts.collect_original_fund_quarterly_reports_20260906 import RAW,OUT,BASE,download_one,STOP_REQUESTS


def main():
    target=OUT/"quarterly_download_v1_1_result.json"
    require(not target.exists(),"完整原始季报归档已经保存")
    source=pd.read_parquet(OUT/"all_official_periodic_announcements.parquet")
    records=[]
    for row in source.to_dict("records"):
        title=re.sub(r"\s+","",row["TITLE"])
        quarter_match=re.search(r"第?([一二三四1234])季度",title)
        if not quarter_match or row["ORG_BULLETIN_TYPE_DESC"] not in ("季报","季度报告"):
            continue
        q={"一":1,"二":2,"三":3,"四":4}.get(quarter_match[1],int(quarter_match[1]) if quarter_match[1].isdigit() else None)
        year_match=re.search(r"(20\d{2})年",title)
        year=int(year_match[1]) if year_match else int(row["SSEDATE"][:4])-(q==4)
        require(not any(word in title for word in ("更正","修订","联接","摘要","英文")),"季度公告需要版本关联")
        records.append({"period":f"{year}Q{q}","publication_date":row["SSEDATE"],"title":row["TITLE"],
                        "url":urljoin("https://www.sse.com.cn",row["URL"]),"relative_official_path":row["URL"],"ts_code":"510300.SH",
                        "year_inferred_from_publication_for_candidate":year_match is None,
                        "period_requires_original_cover_confirmation":True,"official_publication_catalog_verified":True,
                        "clock_status":"OFFICIAL_SSE_ANNOUNCEMENT_DATE_AVAILABLE"})
    require(len(records)==55,"已知上交所季度目录数量变化，需重新核对")
    supplement=json.loads((RAW/"discovery/2013Q2_cninfo_official_candidate_receipt.json").read_text(encoding="utf-8"))
    records.append({"period":"2013Q2","publication_date":None,"reported_date_hint":"2013-07-19",
                    "title":"华泰柏瑞沪深300交易型开放式指数证券投资基金2013年第2季度报告",
                    "url":supplement["url"],"relative_official_path":"finalpage/2013-07-19/62852479.PDF","ts_code":"510300.SH",
                    "year_inferred_from_publication_for_candidate":False,"period_requires_original_cover_confirmation":True,
                    "official_publication_catalog_verified":False,"clock_status":"NO_VIEW_ORIGINAL_PUBLICATION_CATALOG_MISSING"})
    frame=pd.DataFrame(records).sort_values("period")
    require(not frame.period.duplicated().any(),"同季原始报告重复")
    require(set(frame.period)==set(str(p) for p in pd.period_range("2012Q3","2026Q2",freq="Q")),"预期季度未完整对应")
    out_catalog=OUT/"official_quarterly_announcements_v1_1.parquet"
    amendment=OUT/"catalog_title_and_missing_source_amendment_v1_1.json"
    if not amendment.exists():
        frame.to_parquet(out_catalog,index=False)
        frame.to_csv(OUT/"完整季度原始报告候选与公布证据.csv",index=False,encoding="utf-8-sig")
        write_json(amendment,{"registered_at":now(),"status":"56_ORIGINAL_REPORT_CANDIDATES_55_WITH_OFFICIAL_CATALOG_CLOCK",
                    "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "original_classification_result_preserved":"quarterly_catalog_result.json",
                    "catalog_sha256":hashlib.sha256(out_catalog.read_bytes()).hexdigest(),
                    "official_catalog_quarters":55,"official_pdf_only_clock_unresolved_quarters":["2013Q2"],
                    "short_title_period_candidates_require_cover_check":[x["period"] for x in records if x["year_inferred_from_publication_for_candidate"]],
                    "notes":["三份简称只用于定位候选，报告年份和季度必须在原始PDF封面复核。",
                             "2013Q2巨潮原始文件可读取，但上交所年度和7月全类别目录均未发现；其原始公布时钟保持未准入。",
                             "不改变任何策略结果，旧52份标题分类和缺口记录保留。"]},exclusive=True)
    else:
        saved=json.loads(amendment.read_text(encoding="utf-8"))
        require(saved["script_sha256"]==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"目录衔接脚本已变化")
        require(saved["catalog_sha256"]==hashlib.sha256(out_catalog.read_bytes()).hexdigest(),"目录衔接输入变化")
    results=[]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures={executor.submit(download_one,row):row for row in records}
        try:
            for future in as_completed(futures):
                result=future.result()
                results.append(result)
                print(f"原始季度报告归档 {len(results)}/56：{result['period']}",flush=True)
        except Exception:
            STOP_REQUESTS.set()
            for pending in futures:
                pending.cancel()
            raise
    results.sort(key=lambda x:x["period"])
    result={"completed_at":now(),"status":"56_ORIGINAL_REPORT_PDFS_ARCHIVED_55_CATALOG_CLOCKS_AVAILABLE",
            "archived":len(results),"official_catalog_clock_count":55,"clock_unresolved_periods":["2013Q2"],
            "rows":results,"financial_facts_parsed":False,"source_admitted_for_portfolio":False}
    write_json(target,result,exclusive=True)
    print(json.dumps({k:v for k,v in result.items() if k!="rows"},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
