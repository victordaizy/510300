"""2018年消费权重披露增量：公司预告、分析师修正与指数历史收益。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pdfplumber
import requests

from historical_price_gap_causes_v1 import round_trip

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/research/510300_historical_disclosure_increment_v1"
PREV=ROOT/"reports/research/510300_historical_consumer_driver_origins_v1"
EARNINGS=ROOT/"reports/research/510300_historical_consumer_earnings_transmission_v1"
STUDY="510300_HISTORICAL_DISCLOSURE_INCREMENT_V1"
COMPANIES=[("600519","贵州茅台","sse"),("600887","伊利股份","sse"),("002304","洋河股份","szse"),("000858","五粮液","szse")]
HEADERS={"User-Agent":"Mozilla/5.0","Referer":"https://www.cninfo.com.cn/"}


def now():return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,(pd.Timestamp,datetime)):return value.isoformat()
    if value is pd.NaT or value is pd.NA or isinstance(value,float) and not np.isfinite(value):return None
    return value


def save(name,value):
    path=OUT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def prepare():
    if (OUT/"protocol.json").exists():raise RuntimeError("已有设定，不覆盖。")
    previous=json.loads((PREV/"protocol.json").read_text(encoding="utf-8"))
    save("protocol.json",{
        "study_id":STUDY,"recorded_at":now(),"research_mode":"HISTORICAL_ONLY",
        "previous_goal_turn_classification":"PROGRESS_HISTORICAL_CONSUMER_DRIVER_ORIGINS_AND_COMPARABLE_DISCLOSURE",
        "question":"固定四家公司2018半年和前三季度披露中，真正新增了什么？公司预告与分析师盈利假设是否给出不同结论？",
        "fixed_companies":[{"code":c,"name":n} for c,n,_ in COMPANIES],"fixed_periods":["2018H1","2018YTDQ3"],
        "selection_limit":"四家公司延续已见2018年消费下跌案例，不是随机样本或独立验证。前两轮部分节点ETF收益已知。",
        "known_findings":"茅台半年正式营业总收入接近此前粗略披露；洋河前三季度归母利润处于半年预告范围；华泰10月29日报告下调茅台预测，已从检索看到。",
        "announcement_source_window":["2018-04-01","2018-10-31"],
        "prior_anchor_rule":"取正式披露前同期间、同财务口径、同预期主体的最新原始预告；若仅有约数则报告连续偏差，不人为设超预期阈值。无可比预告保留缺失。",
        "range_rule":"有明确区间时，实际值仅按低于下限、区间内、高于上限分组；中点不作为公司精确预期。",
        "qualitative_guidance_rule":"年度计划不能替代半年或单季预测，事后说明不得前移，未检出不等于证明从未存在。",
        "analyst_case":"华泰证券茅台2018年10月29日季报点评及该报告列示的上一份2018年8月同公司报告；使用同机构两份原模型，不当作市场一致预期。",
        "analyst_clock":"仅按报告载明日结束后下一交易日作收益回放，未核实的最早公开时间不回填。",
        "analyst_model_comparison":"比较2018至2020同年度收入、归母利润及利润率；以收入变化和利润率变化的对称分解区分预测修正构成。",
        "event_returns":"八份正式财报按归档日末后下一实际开盘，各5/20个交易日后开盘退出；重叠同日节点合并展示，不算独立样本。两份分析师报告亦按同期限展示。",
        "horizons_sessions":[5,20],"costs":previous["costs"],
        "no_rule_promotion":"本轮首先判断新信息是否真实存在。仅相对公司预告兑现或个别分析师修正，不自动生成整个指数买入条件。不得用本轮收益挑选期限或反转信号。",
        "new_parameters_fitted":0,"new_prospective_tasks":0,"goal_achieved":False,"orders_authorized":False})
    print("已固定四家公司、两个报告期、同口径预告分类和同机构模型比较。",flush=True)


def request(key,url,payload=None):
    body_path=OUT/"raw"/(key+".raw")
    receipt_path=OUT/"receipts"/(key+".json")
    if receipt_path.exists():
        receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
        return body_path.read_bytes() if body_path.exists() else None,receipt
    receipt={"id":key,"url":url,"payload":payload,"retrieved_at":now(),"attempts":1}
    try:
        response=requests.get(url,headers=HEADERS,timeout=(10,30)) if payload is None else requests.post(url,data=payload,headers=HEADERS,timeout=(10,30))
        receipt.update(http_status=response.status_code,final_url=response.url)
        response.raise_for_status()
        body=response.content
        body_path.parent.mkdir(parents=True,exist_ok=True)
        body_path.write_bytes(body)
        receipt.update(status="RAW_SAVED",bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
    except requests.RequestException as exc:
        body=None
        receipt.update(status="SOURCE_FAILED_NO_RETRY",error=str(exc))
    save("receipts/"+key+".json",receipt)
    return body,receipt


def directory():
    initial_path=OUT/"announcement_directories.json"
    if initial_path.exists() and not (OUT/"initial_directory_correction.json").exists():
        save("initial_directory_correction.json",{"recorded_at":now(),"reason":"初次目录分页返回重复公告，原完成标记不可用；五粮液名称含空格，标题检索亦会漏项。改按从原响应取得的股票代码和机构ID查询，逐页以公告ID去重。","initial_manifest":json.loads(initial_path.read_text(encoding="utf-8"))})
    orgs={}
    for path in (OUT/"raw").glob("directory_*_1.raw"):
        for row in json.loads(path.read_bytes()).get("announcements") or []:
            if row.get("orgId"):orgs[row.get("secCode")]=row["orgId"]
    def one(spec):
        code,name,column=spec
        rows={}
        complete=False
        total=None
        for page in range(1,5):
            payload={"pageNum":str(page),"pageSize":"30","column":column,"tabName":"fulltext","stock":code+","+orgs[code],"searchkey":"",
                     "secid":"","category":"","trade":"","seDate":"2018-04-01~2018-10-31","sortName":"time","sortType":"asc","isHLtitle":"false"}
            url="https://www.cninfo.com.cn/new/hisAnnouncement/query?study=historical_disclosure_increment_v1&code="+code+"&page="+str(page)
            body,_=request("stock_directory_"+code+"_"+str(page),url,payload)
            if body is None:break
            data=json.loads(body)
            total=data.get("totalAnnouncement")
            batch=data.get("announcements") or []
            before=len(rows)
            rows.update({r["announcementId"]:r for r in batch})
            if not data.get("hasMore"):
                complete=total is not None and len(rows)==int(total)
                break
            if len(rows)==before:break
        relevant=[]
        for row in rows.values():
            if row.get("secCode")!=code:continue
            title=re.sub("<[^>]+>","",row.get("announcementTitle",""))
            if any(w in title for w in ["第一季度","半年度","第三季度","经营数据","经营指标","业绩","生产经营"]):
                relevant.append({**{k:row.get(k) for k in ["secCode","orgId","announcementId","announcementTime","adjunctUrl"]},"title":title})
        return {"code":code,"name":name,"total_api_rows":total,"retrieved_api_rows":len(rows),"bounded_directory_complete":complete,"relevant":relevant}
    with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(one,COMPANIES))
    save("announcement_directories.json",rows)
    print(json.dumps(rows,ensure_ascii=False,indent=2),flush=True)


def fetch():
    sources=json.loads((OUT/"selected_sources.json").read_text(encoding="utf-8"))
    def one(spec):
        body,receipt=request(spec["id"],spec["url"])
        item=dict(spec,receipt=receipt)
        if body is None:return dict(item,status="SOURCE_FAILED_NO_RETRY")
        if not body.startswith(b"%PDF"):return dict(item,status="NOT_PDF")
        path=OUT/"raw"/(spec["id"]+".pdf")
        if not path.exists():path.write_bytes(body)
        txt_path=OUT/"raw"/(spec["id"]+"_pages.json")
        if not txt_path.exists():
            with pdfplumber.open(BytesIO(body)) as pdf:
                pages=[{"page":i+1,"text":p.extract_text() or ""} for i,p in enumerate(pdf.pages)]
            save("raw/"+spec["id"]+"_pages.json",pages)
            (OUT/"raw"/(spec["id"]+".txt")).write_text("\n".join(f"第{p['page']}页\n{p['text']}" for p in pages),encoding="utf-8")
        return dict(item,status="PDF_TEXT_ACQUIRED",local_pdf=str(path))
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(one,sources))
    save("source_index.json",results)
    print(json.dumps([{k:r.get(k) for k in ["id","status"]} for r in results],ensure_ascii=False,indent=2),flush=True)


def resolve_events():
    if (OUT/"resolved_event_dates.json").exists():raise RuntimeError("已有事件日期，不覆盖。")
    directories=json.loads((OUT/"announcement_directories.json").read_text(encoding="utf-8"))
    events=[]
    keys={"600519":"maotai","600887":"yili","002304":"yanghe","000858":"wuliangye"}
    for d in directories:
        assert d["bounded_directory_complete"]
        for period,title in [("2018H1","2018年半年度报告"),("2018YTDQ3","2018年第三季度报告")]:
            matches=[r for r in d["relevant"] if r["title"] in [title,title+"全文"]]
            assert len(matches)==1,(d["code"],period,matches)
            row=matches[0]
            events.append({"id":keys[d["code"]]+"_"+period,"company":d["name"],"code":d["code"],
                           "period":period,"event_kind":"COMPANY_FULL_REPORT","archive_date":row["adjunctUrl"].split("/")[1],
                           "source_url":"https://static.cninfo.com.cn/"+row["adjunctUrl"],
                           "announcement_id":row["announcementId"],"formal_first_public_time_verified":False})
    for key,date in [("htsc_maotai_h1","2018-08-02"),("htsc_maotai_q3","2018-10-29")]:
        events.append({"id":key,"company":"华泰证券：贵州茅台","code":"600519","period":"2018H1" if key.endswith("h1") else "2018YTDQ3",
                       "event_kind":"ANALYST_REPORT","archive_date":date,"formal_first_public_time_verified":False})
    save("resolved_event_dates.json",{"recorded_at":now(),"before_new_return_computation":True,"events":events,
         "clock_limit":"公司以巨潮归档日、分析师以PDF载明日为观察日，均到下一实际交易日开盘；最早公开时刻未核实，不能当首次消息反应。"})
    print("八份公司报告和两份分析师报告的日期已在新收益计算前保存。",flush=True)


def source_location(source):
    if source.startswith("previous:"):return EARNINGS,source.removeprefix("previous:")
    if source.startswith("origins:"):return PREV,source.removeprefix("origins:")
    return OUT,source


def page_text(source,page):
    base,key=source_location(source)
    saved=base/"raw"/(key+"_pages.json")
    pages=json.loads(saved.read_text(encoding="utf-8"))
    hit=next((r for r in pages if r["page"]==page),None)
    if hit:return hit["text"]
    extra=OUT/"reused_pages"/(key+"_"+str(page)+".json")
    if extra.exists():return json.loads(extra.read_text(encoding="utf-8"))["text"]
    with pdfplumber.open(base/"raw"/(key+".pdf")) as pdf:text=pdf.pages[page-1].extract_text() or ""
    save("reused_pages/"+key+"_"+str(page)+".json",{"source_pdf":str(base/"raw"/(key+".pdf")),"page":page,"text":text})
    return text


def analyze_facts():
    rows=[]
    def values(source,page,label,items,unit="元",decimals=2):
        text=re.sub(r"[\s,，]","",page_text(source,page))
        for item in items:
            token=f"{abs(item):.{decimals}f}"
            assert token in text,f"原值未找到：{source} 第{page}页 {label} {token}"
        rows.append({"source_id":source,"physical_page":page,"label":label,"unit":unit,"values":items})
        return np.array(items,dtype=float)
    old=json.loads((EARNINGS/"财报原值及页码.json").read_text(encoding="utf-8"))
    by_key={r["key"]:r for r in old}
    def reuse(key):
        r=by_key[key]
        return values("previous:"+r["source_id"],r["pdf_page"],r["row_label"],r["values"])
    mt_total=values("previous:maotai_h1",27,"半年营业总收入",[35251464783.33,25493896745.03])
    mt_pbt=values("previous:maotai_h1",28,"半年利润总额",[22719966794.38,16083529518.00])
    yh_profit=values("previous:yanghe_h1",6,"半年归母净利润",[5004991771.22,3908260474.95])
    yh_q3_profit=reuse("y_profit_ytd")[0]
    yh_h1_range=values("yanghe_q1",7,"半年归母净利润预告上下限及上年同期",[468991.26,508073.87,390826.05],"万元")*10000
    yh_q3_range=values("previous:yanghe_h1",15,"前三季度归母净利润预告上下限及上年同期",[669799.45,725616.07,558166.21],"万元")*10000
    wl_revenue=values("wuliangye_h1",5,"半年营业收入",[21421181221.36,15621127660.56])
    wl_pbt=values("wuliangye_h1",28,"半年利润总额",[9855470340.25,6996410414.30])
    wl_profit=values("wuliangye_h1",5,"半年归母净利润",[7110255593.99,4971505554.13])
    text=re.sub(r"\s","",page_text("wuliangye_h1_prelim",1))
    assert "营业收入约212亿元" in text and "利润总额约97亿元" in text
    rows.append({"source_id":"wuliangye_h1_prelim","physical_page":1,"label":"半年粗略预披露",
                 "revenue_cny":212e8,"profit_before_tax_cny":97e8,"approximate":True,"parent_profit_guidance":None})
    mt_prelim=re.sub(r"\s","",page_text("origins:maotai_h1_prelim",1))
    assert "营业总收入350亿元左右" in mt_prelim and "利润总额同比增长40%左右" in mt_prelim
    rows.append({"source_id":"origins:maotai_h1_prelim","physical_page":1,"label":"半年粗略预披露",
                 "total_revenue_cny":350e8,"profit_before_tax_yoy":.40,"approximate":True})
    facts={
        "maotai_2018H1":{"anchor_status":"APPROXIMATE_POINT","anchor_date":"2018-07-16","metrics":[
            {"metric":"营业总收入","preliminary":350e8,"actual":mt_total[0],"relative_gap":mt_total[0]/350e8-1},
            {"metric":"利润总额同比","preliminary":.40,"actual":mt_pbt[0]/mt_pbt[1]-1,"gap_pp":(mt_pbt[0]/mt_pbt[1]-1-.40)*100}],
            "classification":"粗略预披露附近；未设显著超预期阈值。"},
        "maotai_2018YTDQ3":{"anchor_status":"NO_COMPARABLE_GUIDANCE_FOUND","anchor_date":None,"metrics":[],
            "classification":"范围内未取得同期间公司量化预告；10月30日说明晚于本次财报。","prior_source":"previous:maotai_h1","page":9},
        "yili_2018H1":{"anchor_status":"NO_COMPARABLE_GUIDANCE_FOUND","anchor_date":None,"metrics":[],
            "classification":"范围内未取得同期间公司量化预告。","prior_source":"yili_q1","page":10},
        "yili_2018YTDQ3":{"anchor_status":"NO_COMPARABLE_GUIDANCE_FOUND","anchor_date":None,"metrics":[],
            "classification":"范围内未取得同期间公司量化预告。","prior_source":"origins:yili_h1_cninfo","page":18},
        "wuliangye_2018H1":{"anchor_status":"APPROXIMATE_POINT","anchor_date":"2018-08-14","metrics":[
            {"metric":"营业收入","preliminary":212e8,"actual":wl_revenue[0],"relative_gap":wl_revenue[0]/212e8-1},
            {"metric":"利润总额","preliminary":97e8,"actual":wl_pbt[0],"relative_gap":wl_pbt[0]/97e8-1}],
            "actual_parent_profit_cny":wl_profit[0],"classification":"略高于粗略预披露；利润总额不能与归母净利润比较。"},
        "wuliangye_2018YTDQ3":{"anchor_status":"NO_COMPARABLE_GUIDANCE_FOUND","anchor_date":None,"metrics":[],
            "classification":"范围内未取得同期间公司量化预告。","prior_source":"wuliangye_h1","page":11}}
    for key,amount,guide,date in [("yanghe_2018H1",yh_profit[0],yh_h1_range,"2018-04-27"),("yanghe_2018YTDQ3",yh_q3_profit,yh_q3_range,"2018-08-30")]:
        low,high,prior=guide
        state="BELOW_RANGE" if amount<low else "ABOVE_RANGE" if amount>high else "WITHIN_RANGE"
        facts[key]={"anchor_status":"EXPLICIT_RANGE","anchor_date":date,"metrics":[{"metric":"归母净利润","lower":low,"upper":high,"actual":amount,
                    "actual_yoy_using_disclosed_rounded_prior":amount/prior-1,"distance_to_upper":amount/high-1}],
                    "classification":state,"limit":"上年预告基数以万元保留两位，增速有微小舍入差；不以区间中点代替市场预期。"}
    for fact in facts.values():
        if "prior_source" in fact:
            excerpt=page_text(fact["prior_source"],fact["page"])
            assert "不适用" in excerpt and "净利润" in excerpt
    save("八个公司报告的事前预告比较.json",facts)

    models={}
    arrays={"h1":{
        "revenue":[79636,94915,111434],"parent_profit":[36639,44601,52112],"eps":[29.17,35.50,41.48],
        "cost":[7731,9695,12287],"business_tax":[10353,11864,13929],"selling":[3584,3797,4234],
        "management":[5575,5695,6463],"financial":[-80.89,-113.38,-151.17]},
        "q3":{"revenue":[73427,75704,90358],"parent_profit":[32830,34535,42081],"eps":[26.13,27.49,33.50],
        "cost":[7409,7669,8996],"business_tax":[10280,10220,12198],"selling":[3818,3937,4518],
        "management":[5140,4542,5241],"financial":[-261.43,-247.40,-930.60]}}
    for edition,fields in arrays.items():
        models[edition]={}
        for label,array in fields.items():
            models[edition][label]=values("htsc_maotai_"+edition,1 if label in ["revenue","parent_profit","eps"] else 3,label,array,
                                        "元/股" if label=="eps" else "百万元",2 if label in ["eps","financial"] else 0)
    revisions=[]
    errors=[]
    for i,year in enumerate([2018,2019,2020]):
        before,after=models["h1"],models["q3"]
        r0,r1=before["revenue"][i],after["revenue"][i]
        p0,p1=before["parent_profit"][i],after["parent_profit"][i]
        m0,m1=p0/r0,p1/r1
        revenue_part=(r1-r0)*(m0+m1)/2
        margin_part=(m1-m0)*(r0+r1)/2
        errors.append(abs(revenue_part+margin_part-(p1-p0)))
        revisions.append({"forecast_year":year,"prior_revenue_million_cny":r0,"new_revenue_million_cny":r1,"revenue_revision":r1/r0-1,
                          "prior_parent_profit_million_cny":p0,"new_parent_profit_million_cny":p1,"parent_profit_revision":p1/p0-1,
                          "prior_eps":before["eps"][i],"new_eps":after["eps"][i],"eps_revision":after["eps"][i]/before["eps"][i]-1,
                          "prior_model_parent_margin":m0,"new_model_parent_margin":m1,"model_parent_margin_change_pp":(m1-m0)*100,
                          "revenue_component_million_cny":revenue_part,"margin_component_million_cny":margin_part,
                          "revenue_component_share_of_profit_revision":revenue_part/(p1-p0),
                          "model_expense_ratio_changes_pp":{k:(after[k][i]/r1-before[k][i]/r0)*100 for k in ["cost","business_tax","selling","management","financial"]}})
    e0,e1=29.17,26.13
    pe0,pe1=25.5,24.5
    valuation={"prior_target_pe_range":[25,26],"new_target_pe_range":[24,25],"prior_2018_eps":e0,"new_2018_eps":e1,
               "prior_target_midpoint":e0*pe0,"new_target_midpoint":e1*pe1,
               "target_midpoint_revision":e1*pe1/(e0*pe0)-1,
               "eps_component_cny":(e1-e0)*(pe0+pe1)/2,"pe_component_cny":(pe1-pe0)*(e0+e1)/2,
               "limit":"只分解该机构目标价，不分解已实现股价；PE还含增长、风险偏好等，不能将其变化全归为无风险利率。"}
    errors.append(abs(valuation["eps_component_cny"]+valuation["pe_component_cny"]-(valuation["new_target_midpoint"]-valuation["prior_target_midpoint"])))
    assert max(errors)<1e-8
    brokerage={"institution":"华泰证券","prior_document_date":"2018-08-02","new_document_date":"2018-10-29",
               "is_market_consensus":False,"revisions":revisions,"target_valuation":valuation,
               "model_revenue_scope":"两版表均将2017年61,063百万元列为营业收入，该历史量级对应公司营业总收入。保持原模型口径相互比较，不将其直接接到不含金融利息的公司营业收入。",
               "reported_mechanism_change":[
                   {"source":"htsc_maotai_h1","page":1,"kind":"分析师当时判断","claim":"认为供需偏紧，出厂提价和产品组合改善推动增长，批价有所上升。"},
                   {"source":"htsc_maotai_q3","page":1,"kind":"分析师当时判断","claim":"认为消费环境与需求增速放缓，并叠加2017同季高基数，下调后续盈利。"},
                   {"source":"htsc_maotai_q3","page":1,"kind":"券商转述的渠道政策，未取得原通知","claim":"称经销商打款由按月放宽为按季，并加快发货；预收款余额回升可能同时受结算政策影响，不能单独替代需求改善。"}],
               "limitations":["两份模型相隔近三个月，差异混合该期间所有信息，不能全归因于三季报单一事件。",
                              "这些是2018年当时对2018至2020年的历史预测，不是本任务对未来的新预测。",
                              "利润分解是对称乘法恒等式，不识别需求、竞争或政策冲击的因果效应。",
                              "收入预测下调主要影响利润规模；模型费用率的变化反映分析师假设，不是已实现费用。",
                              "公司全年计划和券商年度模型口径与目的不同；公司称符合自身预期不排除券商下调。"]}
    save("华泰历史盈利与估值修正分解.json",brokerage)
    save("原值及页码.json",rows)
    return facts,brokerage,{"source_value_rows":len(rows),"symmetric_identity_max_abs_million_or_cny":max(errors)}


def compute_returns(cfg):
    events=json.loads((OUT/"resolved_event_dates.json").read_text(encoding="utf-8"))["events"]
    market=pd.read_parquet(ROOT/"reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet").sort_values("date").reset_index(drop=True)
    market["date"]=pd.to_datetime(market["date"])
    calendar=pd.DatetimeIndex(market.date)
    assert calendar.is_unique
    dividends=pd.read_csv(ROOT/"data/reference/510300_dividends.csv")
    dividends=dividends.loc[dividends.symbol=="510300.SH"].copy()
    dividends["record_date"]=pd.to_datetime(dividends["record_date"])
    rows=[]
    for spec in events:
        start=int(calendar.searchsorted(pd.Timestamp(spec["archive_date"]),side="right"))
        for horizon in cfg["horizons_sessions"]:
            entry,end=market.iloc[start],market.iloc[start+horizon]
            cash=float(dividends.loc[(dividends.record_date>=entry.date)&(dividends.record_date<end.date),"cash_dividend_per_share"].sum())
            calc=round_trip(entry.open,end.open,cash,cfg["costs"])
            rows.append({**spec,"entry_date":entry.date.date().isoformat(),"exit_date":end.date.date().isoformat(),
                         "entry_index":start,"exit_index":start+horizon,"holding_sessions":horizon,
                         "entry_open":float(entry.open),"exit_open":float(end.open),"dividends_per_share":cash,
                         "gross_return":float((end.open+cash)/entry.open-1),**calc})
    save("全部披露的510300参考收益.json",rows)
    unique=[]
    for date in sorted({r["entry_date"] for r in rows}):
        subset=[r for r in rows if r["entry_date"]==date]
        unique.append({"entry_date":date,"event_ids":list(dict.fromkeys(r["id"] for r in subset)),
                       "five_day_net_return":next(r["net_return"] for r in subset if r["holding_sessions"]==5),
                       "twenty_day_net_return":next(r["net_return"] for r in subset if r["holding_sessions"]==20)})
        for horizon in cfg["horizons_sessions"]:
            assert len({r["net_return"] for r in subset if r["holding_sessions"]==horizon})==1
    save("去除同日重复的收益节点.json",unique)
    return rows,unique


def render_figure(broker,unique):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family":font.get_name(),"axes.unicode_minus":False,"font.size":10,
                         "axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(1,3,figsize=(16,5.8),layout="constrained")
    blue,orange="#285b79","#c57b48"
    revisions=broker["revisions"]
    x=np.arange(3)
    for j,(key,label) in enumerate([("revenue_revision","模型收入"),("parent_profit_revision","归母利润")]):
        bars=axs[0].bar(x+(j-.5)*.35,[r[key]*100 for r in revisions],width=.35,color=[blue,orange][j],label=label)
        axs[0].bar_label(bars,fmt="%.1f",padding=3)
    axs[0].set_xticks(x,["2018年","2019年","2020年"])
    axs[0].set_ylabel("10月模型相对8月模型修正（%）")
    axs[0].set_ylim(-29,6)
    axs[0].legend(frameon=False,loc="upper right")
    axs[0].set_title("同机构修正延伸至后续年度",loc="left",fontweight="bold")
    for j,(key,label) in enumerate([("revenue_component_million_cny","收入规模部分"),("margin_component_million_cny","归母利润率部分")]):
        vals=[-r[key]/100 for r in revisions]
        bottom=np.zeros(3) if j==0 else np.array([-r["revenue_component_million_cny"]/100 for r in revisions])
        bars=axs[1].bar(x,vals,bottom=bottom,color=[blue,orange][j],label=label,width=.55)
        axs[1].bar_label(bars,labels=[f"{v:.1f}" if v>=5 else "" for v in vals],label_type="center",color="white",fontsize=9)
        for i,v in enumerate(vals):
            if v<5:axs[1].text(i,bottom[i]+v+2,f"{v:.1f}",ha="center",va="bottom",color="#333333",fontsize=9)
    axs[1].set_xticks(x,["2018年","2019年","2020年"])
    axs[1].set_ylim(0,125)
    axs[1].set_ylabel("归母利润预测减少（亿元）")
    axs[1].set_title("预测修正的金额组成",loc="left",fontweight="bold")
    axs[1].legend(frameon=False,loc="upper left")
    for j,(key,label) in enumerate([("five_day_net_return","5个交易日"),("twenty_day_net_return","20个交易日")]):
        bars=axs[2].bar(np.arange(len(unique))+(j-.5)*.34,[r[key]*100 for r in unique],width=.34,color=[blue,orange][j],label=label)
        axs[2].bar_label(bars,fmt="%.1f",padding=3,fontsize=8)
    axs[2].set_xticks(np.arange(len(unique)),[r["entry_date"][5:] for r in unique],rotation=35)
    axs[2].axhline(0,color="#999999",lw=.7)
    axs[2].set_ylabel("510300参考交易成本后收益（%）")
    vals=[r[k]*100 for r in unique for k in ["five_day_net_return","twenty_day_net_return"]]
    axs[2].set_ylim(min(vals)-2,max(vals)+4)
    axs[2].set_title("全部归档节点，合并同日重复",loc="left",fontweight="bold")
    axs[2].legend(frameon=False,loc="upper left")
    fig.suptitle("历史预期变化：先分清主体与组成，再看剩余收益",fontsize=17,fontweight="bold")
    fig.supxlabel("左、中为2018年华泰两份原报告的历史模型，不代表市场共识；金额分解不是因果效应。\n右图为已见案例的归档后参考交易，单笔预算10万元、收益分母为实际投入，未构成完整账户。",fontsize=10)
    fig.savefig(OUT/"历史盈利修正与披露后收益.png",dpi=150)
    plt.close(fig)


def report(facts,broker,unique,result):
    pct=lambda v:f"{v*100:+.2f}%"
    labels={"maotai":"贵州茅台","yili":"伊利股份","yanghe":"洋河股份","wuliangye":"五粮液"}
    lines=["# 历史发现：公司兑现、分析师修正与510300之间的距离","",
        "本轮固定四家消费权重公司与2018半年、前三季度两个报告期，比较当时可用的公司预披露与正式数据；另核对同一机构两份茅台模型。研究完全属于历史发现与检验，没有新增前瞻任务。","",
        "新证据最直接的作用是把两件事拆开：公司是否兑现此前披露，以及投资者使用的盈利和估值假设是否改变。二者可以同时呈现‘公司说符合预期，券商却下调预测’，并不逻辑冲突。尚未据此形成达到夏普1.2的指数交易条件。","",
        "## 一、固定八个财报案例，预告并非处处可比较","",
        "|公司|报告期|正式报告前的公司参照|同口径比较结果|","|---|---|---|---|"]
    for key,f in facts.items():
        company,period=key.split("_",1)
        if f["anchor_status"]=="EXPLICIT_RANGE":
            a=f["metrics"][0]
            anchor=f"归母利润{a['lower']/1e8:.2f}—{a['upper']/1e8:.2f}亿元"
            conclusion=f"实际{a['actual']/1e8:.2f}亿元，处于原区间内"
        elif f["anchor_status"]=="APPROXIMATE_POINT":
            anchor="公司初步核算约数"
            a=f["metrics"][0]
            conclusion=f"{a['metric']}相对约数{pct(a['relative_gap'])}；未设意外阈值"
        else:anchor="未取得可比公司量化预告";conclusion="保持缺失，不以同比增速替代预期差"
        lines.append(f"|{labels[company]}|{'半年' if period=='2018H1' else '前三季度'}|{anchor}|{conclusion}|")
    wl=facts["wuliangye_2018H1"]
    lines += ["",f"新增的五粮液对照：归档于8月14日的公司公告给出半年营业收入约212亿元、利润总额约97亿元，正式数分别为{wl['metrics'][0]['actual']/1e8:.2f}亿元和{wl['metrics'][1]['actual']/1e8:.2f}亿元，相对约数为{pct(wl['metrics'][0]['relative_gap'])}和{pct(wl['metrics'][1]['relative_gap'])}。正式归母净利润为{wl['actual_parent_profit_cny']/1e8:.2f}亿元，不能拿来与利润总额约97亿元比较。近似预披露没有给出误差区间，不能事后设置百分比阈值来挑出‘超预期’。","",
        "洋河半年及前三季度利润都处于此前20%—30%的同比预告区间。其余四个公司期间组合，在固定目录及相关定期报告内未取得同口径量化预告；这并不证明市场没有预期，也不证明公开场合从未有其他表达。年度计划、券商观点、公司预告分别处理。","",
        "## 二、茅台两份同机构模型：哪些预期真的改变","",
        "取得华泰证券2018年8月2日与10月29日原报告。8月报告认为供需偏紧，出厂提价和产品组合支撑增长；10月报告认为消费环境与需求增速放缓，并叠加高基数。以下是该机构模型变化，不代表市场一致预期，也不证明其判断正确。","",
        "|预测年度|原收入→新收入（亿元）|原归母利润→新归母利润（亿元）|EPS修正|归母利润率变化|",
        "|---|---:|---:|---:|---:|"]
    for r in broker["revisions"]:
        lines.append(f"|{r['forecast_year']}|{r['prior_revenue_million_cny']/100:.2f} → {r['new_revenue_million_cny']/100:.2f}|{r['prior_parent_profit_million_cny']/100:.2f} → {r['new_parent_profit_million_cny']/100:.2f}|{r['prior_eps']:.2f} → {r['new_eps']:.2f}（{pct(r['eps_revision'])}）|{r['model_parent_margin_change_pp']:+.2f}个百分点|")
    lines += ["","两版模型的收入表头虽写营业收入，2017年基数61,063百万元却对应公司的营业总收入口径。本轮只在同机构、同基数的两份表之间比较，没有把它混接到公司不含金融利息的收入列。","",
        "利润预测=模型收入×模型归母利润率。为避免先算哪项影响结果，采用对称拆分：收入变化乘两版平均利润率，利润率变化乘两版平均收入；两项加总严格回到利润修正。","",
        "|预测年度|利润预测减少（亿元）|其中收入规模部分|其中利润率部分|收入部分占比|",
        "|---|---:|---:|---:|---:|"]
    for r in broker["revisions"]:
        lines.append(f"|{r['forecast_year']}|{(r['prior_parent_profit_million_cny']-r['new_parent_profit_million_cny'])/100:.2f}|{-r['revenue_component_million_cny']/100:.2f}|{-r['margin_component_million_cny']/100:.2f}|{r['revenue_component_share_of_profit_revision']*100:.2f}%|")
    r=broker["revisions"][1];ratios=r["model_expense_ratio_changes_pp"]
    val=broker["target_valuation"]
    lines += ["",f"以2019年为例，模型销售费用率变化{ratios['selling']:+.2f}个百分点、营业税金及附加占收入比例变化{ratios['business_tax']:+.2f}个百分点；收入预测减少同时伴随部分成本费用强度假设上升。因此盈利调整包含规模和利润转化两个层面。这里说的是分析师模型的假设变化，不是2019年已实现的财务结果。","",
        f"估值也在变：对2018年的目标PE从25—26倍调为24—25倍。目标价区间中点由{val['prior_target_midpoint']:.2f}元变为{val['new_target_midpoint']:.2f}元，变化{pct(val['target_midpoint_revision'])}。对EPS×目标PE作同样对称拆分，EPS部分为{val['eps_component_cny']:.2f}元，PE部分为{val['pe_component_cny']:.2f}元。它解释该机构目标价的变化，不是实际股价跌幅的归因。PE还包含增长与风险偏好，不能全部命名为利率变化。","",
        "## 三、继续往上游：预收款回升也可能包含结算政策变化","",
        "10月29日报告称，旺季前经销商打款从按月放宽为按季，并加快发货。公司原表能确认预收款余额较半年末增加，券商的渠道描述则提供了另一种解释：余额不仅受订单需求影响，也受打款频率、发货和收入确认时点影响。由于未取得渠道原通知，本轮将其标为券商转述，不当作已独立核实的政策事实。","",
        "这条链比‘预收款涨就是利好’更具体，但当前资料仍不能算出结算变化与需求变化分别贡献了多少。两份模型之间近三个月的信息也不能全归因于三季报一个事件。公司之后称符合自身预期，不会推翻此前券商实际下调的记录。","",
        "## 四、公开后收益：合并同日重复，不叠加样本数","",
        "八份公司报告和两份券商报告的日期在本轮新收益计算前保存；按归档日或报告载明日结束后下一实际交易日开盘参考买入510300，5/20个交易日后开盘退出。最早公开时刻未核实，这个口径不等于首次消息的全部价格反应。","",
        "|参考入场日|关联披露|5日成本后收益|20日成本后收益|","|---|---|---:|---:|"]
    for node in unique:
        names=[]
        for key in node["event_ids"]:
            if key.startswith("htsc_"):names.append("华泰茅台模型")
            else:
                c,p=key.split("_",1)
                names.append(labels[c]+("半年报" if p=="2018H1" else "三季报"))
        lines.append(f"|{node['entry_date']}|{'、'.join(names)}|{pct(node['five_day_net_return'])}|{pct(node['twenty_day_net_return'])}|")
    lines += ["","每笔示例预算10万元、100份整手，佣金单边万四且最低5元，滑点单边0.1%，价格按0.001元向不利方向取整；按登记日计算分红权益。收益分母为实际投入，不是20万元全账户。十份文件只对应七个不同入场日，窗口仍有重叠；部分收益已在前轮见过，没有新增独立验证。","",
        "正负收益保留全部结果。它们是对应时期的ETF回报，不能证明个别财报或券商修正造成指数涨跌，也不能因为某个期限更好就重新选期限、反向交易或扩大仓位。","",
        "## 五、本轮对交易研究的实际约束","",
        "- 八个正式报告中，只有两个有明确公司预告区间，实际均落在区间内；另两个只有粗略约数，四个缺少可比量化预告。当前样本无法建立‘正式报告突破公司预告上限’的正面交易机制。",
        "- 同机构盈利修正提供了真实可比较的信息更新，但单家公司、单一机构、单次负面修正，不能自动代表沪深300的盈利或风险溢价变化。",
        "- 下一步应验证修正是否在预先固定的权重范围内扩散，以及修正来自经营规模还是利润率。若没有指数层面的共同变化，继续细化个股故事对510300交易的帮助有限。","",
        f"已取得7份新增原始PDF，完成{result['source_value_rows']}条原值记录、8个公司预告比较和3个年度模型分解。原表数值、乘法恒等关系和保存交易损益核回；未运行参数搜索、新完整账户或前瞻任务。夏普1.2目标仍未达到。","",
        f"![盈利修正与收益]({(OUT/'历史盈利修正与披露后收益.png').as_posix()})","",
        "## 来源与计算","",
        "- [华泰2018年8月2日茅台报告](https://crm.htsc.com.cn/doc/2018/10710208/5898348c-fd50-45fc-95b8-b949c21ce296.pdf)与[10月29日茅台报告](https://crm.htsc.com.cn/doc/2018/10710209/b6db5475-e389-42d7-bbb3-3159b5b36d91.pdf)：第1页经营解释及预测，第3页模型。",
        "- [五粮液半年主要经营指标](https://static.cninfo.com.cn/finalpage/2018-08-14/1205282715.PDF)与[半年报](https://static.cninfo.com.cn/finalpage/2018-08-28/1205339576.PDF)：营业收入与利润总额分别比较。",
        "- [洋河一季报](https://static.cninfo.com.cn/finalpage/2018-04-27/1204802358.PDF)、[半年报](https://static.cninfo.com.cn/finalpage/2018-08-30/1205355061.PDF)、[三季报](https://static.cninfo.com.cn/finalpage/2018-10-27/1205543056.PDF)：相邻披露的利润区间与兑现。",
        "- 其余公司财报沿用前两轮已保存原件；正式日期与原链接在resolved_event_dates.json，新增PDF及请求状态在source_index.json。",
        "- 目录采用股票代码与从原响应取得的机构ID查询，204条公告ID去重后与响应总数一致。初次名称检索及分页重复问题已修正并保留说明；目录只支持本次范围内的检索覆盖，不证明全市场信息完备。",
        "- 原值及页码.json保存来源与单位；华泰历史盈利与估值修正分解.json保存计算；全部披露的510300参考收益.json保存价格、日期、份额、费用。脚本research/historical_disclosure_increment_v1.py analyze使用本地资料重算。",""]
    name="历史发现_公司兑现与分析师修正为何不同.md"
    (OUT/name).write_text("\n".join(lines),encoding="utf-8")
    return name


def analyze():
    cfg=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    facts,broker,checks=analyze_facts()
    events,unique=compute_returns(cfg)
    assert len(facts)==8 and len(events)==20 and len(unique)==7
    checks.update(status="PASS_NECESSARY_SOURCE_MODEL_IDENTITY_AND_EVENT_CHECKS",company_reports=8,
                  analyst_reports=2,unique_entry_dates=7,return_rows_including_same_day_duplicates=20,
                  first_public_time_verified=False,independent_validation=False,figure_visually_reviewed=False)
    save("calculation_checks.json",checks)
    result={"study_id":STUDY,"completed_at":now(),"research_mode":"HISTORICAL_ONLY",
            "classification":"PROGRESS_HISTORICAL_COMPARABLE_GUIDANCE_AND_PAIRED_ANALYST_REVISIONS",
            "status":"HISTORICAL_GUIDANCE_COMPARISON_AND_MODEL_REVISION_COMPONENTS_COMPLETED",
            "source_value_rows":checks["source_value_rows"],"new_pdf_documents":7,"company_reports":8,
            "explicit_guidance_range_reports":2,"within_range_reports":2,"above_range_reports":0,
            "approximate_guidance_reports":2,"no_comparable_guidance_reports":4,
            "analyst_revision_years":3,"unique_etf_entry_dates":7,"event_return_rows_including_duplicates":20,
            "independent_validation":False,"new_parameters_fitted":0,"new_full_accounts":0,"new_prospective_tasks":0,
            "net_sharpe":None,"goal_achieved":False,"orders_authorized":False,
            "next_historical_question":"在固定2018年四家消费权重及同一华泰机构的中报、三季报日历中，核查盈利与估值修正是否扩散；只在多权重共同经营机制及公开后剩余收益有支持时再提出指数条件，不把单一个股模型直接当510300信号。"}
    render_figure(broker,unique)
    result["report"]=report(facts,broker,unique,result)
    save("result.json",result)
    print(json.dumps(clean({"结果":result,"模型修正":broker["revisions"],"估值分解":broker["target_valuation"],"七个节点":unique,"必要核对":checks}),ensure_ascii=False,indent=2),flush=True)


def record_progress():
    result=json.loads((OUT/"result.json").read_text(encoding="utf-8"))
    checks=json.loads((OUT/"calculation_checks.json").read_text(encoding="utf-8"))
    events=json.loads((OUT/"全部披露的510300参考收益.json").read_text(encoding="utf-8"))
    residuals=[]
    for row in events:
        pnl=(row["sell_price"]-row["buy_price"])*row["shares"]-row["commissions_cny"]+row["dividend_entitlement_cny"]
        residuals.append(abs(pnl-row["net_pnl_cny"]))
        assert row["paid_cny"]<=100000 and row["shares"]%100==0
        assert row["exit_index"]-row["entry_index"]==row["holding_sessions"]
        assert row["archive_date"]<row["entry_date"]<row["exit_date"]
        assert abs(row["net_pnl_cny"]/row["paid_cny"]-row["net_return"])<1e-12
    assert max(residuals)<.01
    checks.update(saved_event_pnl_recomputation_max_abs_cny=max(residuals),figure_visually_reviewed=True)
    save("calculation_checks.json",checks)
    prefix="reports/research/510300_historical_disclosure_increment_v1/"
    stamp=now()
    receipt={"recorded_at":stamp,"study_id":STUDY,"user_authority":"历史检验和发现，追查因子背后原因；减少繁琐检验。",
             "result":prefix+"result.json","goal_achieved":False,"orders_authorized":False,"changes":[],"old_rejections_retained":True}
    for name in ["510300_historical_cause_discovery_v1.json","510300_existing_data_training_mandate_v1.json"]:
        path=ROOT/"config"/name
        cfg=json.loads(path.read_text(encoding="utf-8"))
        before=dict(cfg)
        if name=="510300_historical_cause_discovery_v1.json":
            cfg.update(latest_completed_study=prefix+"result.json",latest_report=prefix+result["report"],updated_at=stamp)
        else:
            cfg.update(current_round=STUDY,latest_progress_receipt=prefix+"result.json",latest_historical_disclosure_increment=prefix+"result.json",
                       latest_continuation_report=prefix+result["report"],latest_historical_report=prefix+result["report"],
                       latest_continuation_classification=result["classification"],current_driver_continuation_classification=result["classification"],
                       current_driver_consecutive_blocked_goal_turns=0,latest_historical_diagnostic_at=stamp,
                       latest_goal_service_status="active",latest_goal_service_status_observed_at=stamp,goal_status="active",goal_achieved=False,
                       local_goal_work_status="ACTIVE_HISTORICAL_ONLY",
                       last_research_result="固定八个2018财报案例：两个利润区间均兑现、两个仅粗略预披露、四个缺可比预告。取得华泰茅台8月与10月原模型，拆出收入、利润率及目标PE的历史修正，并核算七个不同ETF入场节点；未建立指数信号或达标账户。",
                       last_source_result="新增7份原始PDF；巨潮固定四家公司204条公告去重，完成公司预告及同机构模型比较；报告日期不冒充最早公开时刻，渠道政策仍标为券商转述。",
                       next_research_question=result["next_historical_question"])
        receipt["changes"].append({"path":str(path),"fields":{k:{"before":before.get(k),"after":v} for k,v in cfg.items() if before.get(k)!=v}})
        path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    save("authority_update.json",receipt)
    print("已登记八个公司比较与历史模型修正，目标仍在进行中且未达标。",flush=True)


def main():
    parser=argparse.ArgumentParser(description="历史披露信息增量")
    parser.add_argument("mode",choices=["prepare","directory","fetch","resolve-events","analyze","record-progress"])
    args=parser.parse_args()
    {"prepare":prepare,"directory":directory,"fetch":fetch,"resolve-events":resolve_events,"analyze":analyze,"record-progress":record_progress}[args.mode]()


if __name__=="__main__":main()
