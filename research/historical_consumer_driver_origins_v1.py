"""追查2018年消费权重变化的价格、渠道、成本和竞争原因，仅作历史研究。"""
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

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import pdfplumber
import requests

from historical_price_gap_causes_v1 import round_trip

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/research/510300_historical_consumer_driver_origins_v1"
PREV=ROOT/"reports/research/510300_historical_consumer_earnings_transmission_v1"
STUDY="510300_HISTORICAL_CONSUMER_DRIVER_ORIGINS_V1"
HEADERS={"User-Agent":"Mozilla/5.0","Referer":"https://www.cninfo.com.cn/"}


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,(datetime,pd.Timestamp)):return value.isoformat()
    if value is pd.NA or value is pd.NaT or isinstance(value,float) and not np.isfinite(value):return None
    return value


def save(name,value):
    path=OUT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def prepare():
    if (OUT/"protocol.json").exists():raise RuntimeError("本轮研究设定已存在，不覆盖。")
    parent=json.loads((PREV/"protocol.json").read_text(encoding="utf-8"))
    save("protocol.json",{
        "study_id":STUDY,"recorded_at":now(),"research_mode":"HISTORICAL_ONLY",
        "previous_goal_turn_classification":"PROGRESS_HISTORICAL_CONSUMER_EARNINGS_CAUSES_AND_EXPECTATION_ANCHOR",
        "question":"追查伊利竞争投放与包装辅料成本、茅台价格调整与渠道结构；哪些在三季报之前已知，哪些只是下跌后解释？",
        "selection":"延续上一轮已见2018年下跌案例，仅补齐原文因果组成，非独立样本。",
        "fixed_sources_to_search":["伊利2018半年报、半年及三季度业绩推介材料","茅台2017半年报、2018上半年业绩预披露、2018年10月生产经营说明、2017年底价格调整公告"],
        "facts_already_seen":"伊利半年报说明纸包材及辅料上涨、原料乳收购价基本稳定、竞争增加广告营销投入；茅台半年度和季度已披露产品/渠道收入。",
        "event_definitions":[{"id":"MT_H1_PRELIM","title":"茅台2018上半年主要经营数据公告","archive_date":"待原目录核对"},{"id":"MT_H1_FULL","title":"茅台2018半年报","archive_date":"2018-08-02"},{"id":"YILI_H1_FULL","title":"伊利2018半年报","archive_date":"待原目录核对"},{"id":"MT_OCT_CLARIFICATION","title":"茅台2018年10月关于生产经营情况的公告","archive_date":"待原目录核对"}],
        "event_dates_resolution":"仅依官方目录或指定披露报刊确认；收益计算前另存resolved_event_dates。原件无法取得的节点保留未计算，不替换为表现更好事件。",
        "event_horizons_sessions":[5,20],"clock":"归档日末后下一实际交易日开盘入场，持有指定交易日数后开盘退出。首次精确公开时间未核实，不回填。",
        "price_announcement_role":"2017年底价格调整只作2018经营解释背景，不新增交易事件。",
        "issuer_presentation_clock":"当前官网上传路径为2023年，不能仅凭报告标题恢复2018首次公开日期；此类图表仅作历史经营解释，未经定期报告交叉支持不作为时点输入。",
        "costs":parent["costs"],"traded_assets":["510300.SH","CASH_CNY"],
        "new_full_accounts":0,"new_parameters_fitted":0,"new_prospective_tasks":0,"independent_validation":False,
        "goal_achieved":False,"orders_authorized":False})
    print("已固定历史来源范围、四个披露节点和5/20日期限。",flush=True)


def request(key,url,payload=None):
    dest=OUT/"raw"/(key+".raw")
    receipt={"id":key,"url":url,"payload":payload,"retrieved_at":now(),"attempts":1}
    if dest.exists():
        return dest.read_bytes(),json.loads((OUT/"receipts"/(key+".json")).read_text(encoding="utf-8"))
    previous=OUT/"receipts"/(key+".json")
    if previous.exists():
        return None,json.loads(previous.read_text(encoding="utf-8"))
    try:
        response=requests.get(url,headers=HEADERS,timeout=(10,30)) if payload is None else requests.post(url,data=payload,headers=HEADERS,timeout=(10,30))
        receipt.update(http_status=response.status_code,final_url=response.url)
        response.raise_for_status()
        body=response.content
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(body)
        receipt.update(status="RAW_SAVED",bytes=len(body),sha256=hashlib.sha256(body).hexdigest())
    except requests.RequestException as exc:
        body=None
        receipt.update(status="SOURCE_FAILED_NO_RETRY",error=str(exc))
    save("receipts/"+key+".json",receipt)
    return body,receipt


def directory():
    for code,start,end in [("600519","2017-12-25","2018-10-31"),("600887","2018-08-25","2018-09-03")]:
        body,_=request("lookup_"+code,"https://www.cninfo.com.cn/new/information/topSearch/query?keyWord="+code+"&maxNum=10")
        if body is None:continue
        data=json.loads(body)
        match=next((r for r in data if r.get("code")==code),None)
        if match is None:continue
        payload={"pageNum":"1","pageSize":"100","column":"sse","tabName":"fulltext","stock":code+","+match["orgId"],"searchkey":"","secid":"","category":"","trade":"","seDate":start+"~"+end,"sortName":"time","sortType":"desc","isHLtitle":"false"}
        body,_=request("directory_"+code,"https://www.cninfo.com.cn/new/hisAnnouncement/query",payload)
        if body is None:continue
        data=json.loads(body)
        rows=data.get("announcements") or []
        brief=[{k:r.get(k) for k in ["secCode","announcementTitle","announcementTime","adjunctUrl","announcementId"]} for r in rows if any(w in r.get("announcementTitle","") for w in ["半年","经营","价格","业绩","第三季度"])]
        save("directory_"+code+".json",{"total":data.get("totalAnnouncement"),"retrieved":len(rows),"rows":brief})
        print(json.dumps({"公司":code,"目录":brief},ensure_ascii=False,indent=2),flush=True)
    body,_=request("yili_investors_index","https://www.yili.com/investors")
    if body:
        soup=BeautifulSoup(body,"html.parser")
        matches=[]
        for a in soup.find_all("a",href=True):
            href=a["href"]
            if ".pdf" not in href.lower():continue
            parent=a
            for _ in range(4):
                parent=parent.parent
                if parent is None:break
                txt=parent.get_text(" ",strip=True)
                if "2018" in txt and len(txt)<1500:
                    matches.append({"url":requests.compat.urljoin("https://www.yili.com",href),"nearby_text":txt[:600]})
                    break
        save("yili_presentation_links.json",matches)
        print(json.dumps({"伊利原站2018链接":matches},ensure_ascii=False,indent=2),flush=True)


def fetch():
    specs=json.loads((OUT/"selected_sources.json").read_text(encoding="utf-8"))
    def one(spec):
        body,receipt=request(spec["id"],spec["url"])
        record=dict(spec,receipt=receipt)
        if body is None:
            record["status"]=receipt["status"]
            return record
        if spec.get("kind","pdf")=="pdf":
            if not body.startswith(b"%PDF"):
                record["status"]="NOT_PDF"
                return record
            pdf=OUT/"raw"/(spec["id"]+".pdf")
            if not pdf.exists():pdf.write_bytes(body)
            target=OUT/"raw"/(spec["id"]+"_pages.json")
            if not target.exists():
                with pdfplumber.open(BytesIO(body)) as document:
                    total=len(document.pages)
                    pages=[{"page":i,"text":document.pages[i-1].extract_text() or ""} for i in spec.get("extract_pages",range(1,total+1))]
                save("raw/"+spec["id"]+"_pages.json",pages)
                (OUT/"raw"/(spec["id"]+".txt")).write_text("\n".join(f"第{p['page']}页\n{p['text']}" for p in pages),encoding="utf-8")
            record.update(status="PDF_TEXT_ACQUIRED",local_pdf=str(pdf))
        else:
            txt=BeautifulSoup(body,"html.parser").get_text("\n",strip=True)
            (OUT/"raw"/(spec["id"]+".txt")).write_text(txt,encoding="utf-8")
            record["status"]="HTML_TEXT_ACQUIRED"
        return record
    with ThreadPoolExecutor(max_workers=4) as pool:records=list(pool.map(one,specs))
    save("source_index.json",records)
    print(json.dumps([{k:r.get(k) for k in ["id","status","local_pdf"]} for r in records],ensure_ascii=False,indent=2),flush=True)


def page_text(source_id, page):
    base = PREV if source_id.startswith("previous:") else OUT
    key = source_id.removeprefix("previous:")
    pages = json.loads((base/"raw"/(key+"_pages.json")).read_text(encoding="utf-8"))
    return next(p["text"] for p in pages if p["page"] == page)


def source_facts():
    """原表按已目视核对的列录入，只检查足以影响结论的金额与口径。"""
    rows = []

    def vector(source, page, label, values, columns, unit="元"):
        compact = re.sub(r"[\s,，]", "", page_text(source, page))
        for value in values:
            assert f"{value:.2f}" in compact, f"原表金额未找到：{source} 第{page}页 {label} {value}"
        rows.append({"source_id": source, "physical_page": page, "label": label,
                     "columns": columns, "unit": unit, "values": values})
        return np.array(values, dtype=float)

    cols = ["2018半年", "2017半年"]
    yr = vector("yili_h1_cninfo", 11, "营业收入", [39588519723.58,33301766306.64], cols)
    yc = vector("yili_h1_cninfo", 11, "营业成本", [24279436754.35,20583211976.12], cols)
    ys = vector("yili_h1_cninfo", 11, "销售费用", [10169908423.61,7647855923.56], cols)
    yp = vector("yili_h1_cninfo", 6, "归母净利润", [3445830923.47,3364111100.93], cols)
    expense_values = {
        "职工薪酬": [1703660647.97,1238380549.66],
        "折旧修理": [37165908.49,37261111.94],
        "差旅": [126418061.40,108785912.61],
        "物耗劳保": [5876025.77,5922042.55],
        "办公租赁": [108451617.34,104603218.83],
        "广告营销": [5962239521.88,4291224867.30],
        "装卸运输": [2193863142.40,1840349087.90],
        "其他": [32233498.36,21329132.77]}
    expense = {k: vector("yili_h1_cninfo",115,k,v,cols) for k,v in expense_values.items()}
    expense_sum_error = float(np.max(np.abs(np.sum(list(expense.values()),axis=0)-ys)))
    assert expense_sum_error < .01
    qcols = ["2018第三季度", "2017第三季度", "2018前三季度", "2017前三季度"]
    qr_all = vector("previous:yili_q3",15,"营业收入",[21257756990.76,18825129256.65,60846276714.34,52126895563.29],qcols)
    qc_all = vector("previous:yili_q3",15,"营业成本",[13647777423.82,11775353354.57,37927214178.17,32358565330.69],qcols)
    qs_all = vector("previous:yili_q3",15,"销售费用",[5163977150.91,4164186004.48,15333885574.52,11812041928.04],qcols)
    qp_all = vector("previous:yili_q3",16,"归母净利润",[1601987824.20,1573156230.86,5047818747.67,4937267331.79],qcols)
    quarter_error = max(float(np.max(np.abs(q[2:]-q[:2]-h))) for q,h in [(qr_all,yr),(qc_all,yc),(qs_all,ys),(qp_all,yp)])
    assert quarter_error < .01

    def margin_snapshot(revenue,cost,sales,profit):
        gp = revenue-cost
        return {"revenue_cny":revenue,"gross_profit_cny":gp,"sales_expense_cny":sales,
                "revenue_yoy":revenue[0]/revenue[1]-1,"parent_profit_yoy":profit[0]/profit[1]-1,
                "gross_margins":gp/revenue,"sales_expense_ratios":sales/revenue,
                "gross_margin_change_pp":float(np.diff((gp/revenue)[::-1])[0]*100),
                "sales_expense_ratio_change_pp":float(np.diff((sales/revenue)[::-1])[0]*100),
                "incremental_gross_profit_cny":gp[0]-gp[1],
                "incremental_sales_expense_cny":sales[0]-sales[1],
                "incremental_gp_less_sales_expense_cny":gp[0]-gp[1]-sales[0]+sales[1],
                "sales_increment_over_gp_increment":(sales[0]-sales[1])/(gp[0]-gp[1]),
                "limit":"只是营业收入、成本、销售费用的算术关系，未纳入其他费用收益和税项，不是完整净利润归因。"}

    expense_change = ys[0]-ys[1]
    yili = {"h1":margin_snapshot(yr,yc,ys,yp),
            "q3":margin_snapshot(qr_all[:2],qc_all[:2],qs_all[:2],qp_all[:2]),
            "h1_expense_components":[{"item":k,"amounts_cny":v,"increase_cny":v[0]-v[1],
                                      "share_of_expense_increase":(v[0]-v[1])/expense_change} for k,v in expense.items()],
            "expense_ratio_denominator":"营业收入，不使用含金融利息的营业总收入。",
            "h1_issuer_explanations":{
                "input_cost":"公司称国内生产与国际贸易环境影响下，纸类包装和部分辅料采购价上涨，国内原料乳收购价基本稳定。",
                "selling_cost":"公司称竞争激烈增加广告营销投入，薪酬及随销量增加的运输费用也上升。",
                "sales":"公司称销量增加及产品结构调整；所引用调研显示常温液态奶零售额份额同比增加2.4个百分点。"},
            "identification_limits":["上述为公司对半年经营环境的解释，并非外生冲击的因果效应估计。",
                                     "广告投入与销售及份额同时增加，不能据此估计广告带来的净新增需求。",
                                     "缺少同口径三季度包装、原料乳及费用分项，不能将半年原因自动延长至第三季度。",
                                     "毛利率受售价、组合、效率和投入价格共同影响；半年原料乳基本稳定也不代表所有奶源投入都稳定。"]}

    mcols = ["茅台酒", "系列酒", "直销", "批发", "国内", "国外"]
    specifications = [
        ("2017H1","maotai_2017h1",6,[2162685.42,254894.51,245147.09,2172432.84,2333360.62,84219.31]),
        ("2017YTDQ3","previous:maotai_2017q3",7,[3840433.86,401869.44,400362.74,3841940.56,4099489.68,142813.62]),
        ("2018H1","previous:maotai_h1",7,[2938331.11,399338.70,257767.01,3079902.80,3234524.58,103145.23]),
        ("2018YTDQ3","previous:maotai_q3",7,[4626539.47,593360.21,387024.85,4832874.83,5045505.50,174394.18])]
    segments = {period:vector(source,page,"产品、渠道和区域收入",vals,mcols,"万元") for period,source,page,vals in specifications}
    segment_error = max(float(np.ptp(v.reshape(3,2).sum(axis=1))) for v in segments.values())
    assert segment_error < .02, "产品、渠道、地区三个口径加总不同。"
    q18,q17 = segments["2018YTDQ3"]-segments["2018H1"],segments["2017YTDQ3"]-segments["2017H1"]
    h18,h17 = segments["2018H1"],segments["2017H1"]
    core_change = sum(q18[:2]-q17[:2])
    mt_rows = [{"item":k,"h1_2018_wan_cny":h18[i],"h1_2017_wan_cny":h17[i],"h1_yoy":h18[i]/h17[i]-1,
                "q3_2018_wan_cny":q18[i],"q3_2017_wan_cny":q17[i],"q3_yoy":q18[i]/q17[i]-1,
                "q3_change_wan_cny":q18[i]-q17[i]} for i,k in enumerate(mcols)]
    total = vector("previous:maotai_h1",27,"营业总收入",[35251464783.33,25493896745.03],cols)
    operating = vector("previous:maotai_h1",27,"营业收入",[33396709893.11,24190030218.96],cols)
    pbt = vector("previous:maotai_h1",28,"利润总额",[22719966794.38,16083529518.00],cols)
    prelim = re.sub(r"\s", "", page_text("maotai_h1_prelim",1))
    assert "营业总收入350亿元左右" in prelim and "同比增长37%左右" in prelim and "利润总额同比增长40%左右" in prelim
    rows.append({"source_id":"maotai_h1_prelim","physical_page":1,"label":"公司初步核算",
                 "approximate":True,"total_revenue_cny":350e8,"total_revenue_yoy":.37,"profit_before_tax_yoy":.40,
                 "not_market_consensus":True})
    maotai = {"segment_rows":mt_rows,"core_liquor_q3_revenue_yoy":sum(q18[:2])/sum(q17[:2])-1,
              "core_liquor_q3_revenue_increase_wan_cny":core_change,
              "series_share_of_core_liquor_revenue_increase":(q18[1]-q17[1])/core_change,
              "revenue_scope":"两类产品、两类渠道、两类地区分别加总为酒类主营收入；三个切面不能相互加总，也不等于含其他收入的合并营业收入。",
              "price_context":{"source_id":"maotai_price_official","document_signed":"2017-12-28","average_planned_price_increase":.18,
                               "first_public_timestamp_verified":False,"event_role":"仅作经营背景，不新增事件或倒推出季度销量。"},
              "h1_preliminary_vs_actual":{"preliminary_total_revenue_cny":350e8,"actual_total_revenue_cny":total[0],
                  "actual_relative_to_approximate_preliminary":total[0]/350e8-1,"actual_total_revenue_yoy":total[0]/total[1]-1,
                  "preliminary_total_revenue_yoy_approx":.37,"preliminary_profit_before_tax_yoy_approx":.40,
                  "actual_profit_before_tax_yoy":pbt[0]/pbt[1]-1,
                  "incompatible_operating_revenue_cny":operating[0],
                  "wrong_scope_example_relative_to_preliminary":operating[0]/350e8-1,
                  "interpretation":"同口径正式数与公司粗略预披露接近；公司预披露不是市场一致预期，不将近似值差异判为显著超预期。"},
              "october_statement":{"source_id":"maotai_oct_cninfo","archive_date":"2018-10-30",
                                   "relative_to_q3_report":"晚于10月29日三季报，是事后公司说明。",
                                   "expectation_owner":"公司，非市场一致预期。"},
              "identification_limits":["产品收入和渠道收入变化已确定，数量、实际均价、终端动销和渠道库存尚未分离。",
                                       "计划平均提价18%并非每个产品、渠道和季度实现的同一价格变化，不能用收入除1.18推导销量。",
                                       "2017第三季度高基数来自前轮核对，说明同比参照尺度，但不能单独证明2018需求是否下降。"]}
    price_text = (OUT/"raw/maotai_price_official.txt").read_text(encoding="utf-8")
    assert "18%" in price_text and "2018" in price_text
    explanations = [
        {"source":"yili_h1_cninfo","physical_pages":[9,11,12,115],"archive_date":"2018-08-31",
         "known_before_october_downmove":True,"type":"公司经营解释及金额组成","claim":"半年包装辅料涨价、原料乳基本稳定，竞争投入使销售费用增幅超过收入；营销费用贡献约三分之二销售费用增量。"},
        {"source":"maotai_h1_prelim","physical_pages":[1],"archive_date":"2018-07-16",
         "known_before_h1_full_report":True,"type":"公司近似业绩预披露","claim":"350亿元口径为营业总收入，利润增速口径为利润总额。"},
        {"source":"previous:maotai_q3","physical_pages":[7],"archive_date":"2018-10-29",
         "known_before_october19_26_downmove":False,"type":"财务分项差分","claim":"单季主品牌与系列酒、直销与批发增速不同；不能将事后分项提前作为下跌前信号。"},
        {"source":"maotai_oct_cninfo","physical_pages":[1],"archive_date":"2018-10-30",
         "known_before_q3_report":False,"type":"财报后公司说明","claim":"符合公司预期不等于符合市场预期；不能向前填入季度披露前。"}]
    save("原值与口径页码.json",rows)
    save("上游原因与金额分解.json",{"yili":yili,"maotai":maotai})
    save("解释与公开先后.json",explanations)
    return yili,maotai,{"financial_source_rows":len(rows),"expense_sum_max_abs_cny":expense_sum_error,
                        "quarter_identity_max_abs_cny":quarter_error,"segment_sum_max_abs_wan_cny":segment_error}


def event_returns(cfg):
    resolved = json.loads((OUT/"resolved_event_dates.json").read_text(encoding="utf-8"))
    expected = [r["id"] for r in cfg["event_definitions"]]
    assert [r["id"] for r in resolved["events"]] == expected
    market_path = ROOT/"reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet"
    market = pd.read_parquet(market_path).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market["date"])
    assert market.date.is_unique
    calendar = pd.DatetimeIndex(market.date)
    dividends = pd.read_csv(ROOT/"data/reference/510300_dividends.csv")
    dividends = dividends.loc[dividends.symbol=="510300.SH"].copy()
    dividends["record_date"] = pd.to_datetime(dividends["record_date"])
    rows = []
    for spec in resolved["events"]:
        idx = int(calendar.searchsorted(pd.Timestamp(spec["archive_date"]),side="right"))
        for horizon in cfg["event_horizons_sessions"]:
            entry,end = market.iloc[idx],market.iloc[idx+horizon]
            entitled = float(dividends.loc[(dividends.record_date>=entry.date)&(dividends.record_date<end.date),"cash_dividend_per_share"].sum())
            calc = round_trip(entry.open,end.open,entitled,cfg["costs"])
            rows.append({**spec,"entry_date":entry.date.date().isoformat(),"exit_date":end.date.date().isoformat(),
                         "holding_sessions":horizon,"entry_index":idx,"exit_index":idx+horizon,
                         "entry_open":float(entry.open),"exit_open":float(end.open),"dividends_per_share":entitled,
                         "gross_return":float((end.open+entitled)/entry.open-1),**calc,
                         "illustrative_full_capital_pnl_ratio":calc["net_pnl_cny"]/cfg["costs"]["illustrative_full_capital_cny"]})
    assert len(rows)==8 and all(r["archive_date"]<r["entry_date"]<r["exit_date"] for r in rows)
    save("四个披露节点510300收益.json",rows)
    return rows


def render_figure(yili,maotai,events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family":font.get_name(),"axes.unicode_minus":False,
                         "axes.spines.top":False,"axes.spines.right":False,"font.size":10})
    fig,axs = plt.subplots(2,2,figsize=(14,9),layout="constrained")
    blue,orange = "#245b78","#c87843"
    selected = [next(r for r in yili["h1_expense_components"] if r["item"]==k) for k in ["广告营销","职工薪酬","装卸运输"]]
    other = sum(r["increase_cny"] for r in yili["h1_expense_components"])-sum(r["increase_cny"] for r in selected)
    vals = [r["increase_cny"]/1e8 for r in selected]+[other/1e8]
    bars = axs[0,0].bar(["广告营销","职工薪酬","装卸运输","其余项目"],vals,color=[orange,blue,blue,"#889ba4"],width=.6)
    axs[0,0].bar_label(bars,fmt="%.2f",padding=3)
    axs[0,0].set_ylim(0,max(vals)*1.28)
    axs[0,0].set_ylabel("销售费用同比增加（亿元）")
    axs[0,0].set_title("伊利半年：费用增加主要来自营销",loc="left",fontweight="bold")
    for j,(key,label) in enumerate([("gross_margin_change_pp","毛利率"),("sales_expense_ratio_change_pp","销售费用率")]):
        values = [yili[s][key] for s in ["h1","q3"]]
        bars = axs[0,1].bar(np.arange(2)+(j-.5)*.33,values,width=.33,color=[blue,orange][j],label=label)
        axs[0,1].bar_label(bars,fmt="%+.2f",padding=3)
    axs[0,1].set_xticks([0,1],["2018半年","2018第三季度"])
    axs[0,1].axhline(0,color="#999999",lw=.7)
    axs[0,1].set_ylim(-2.5,4.1)
    axs[0,1].set_ylabel("较上年同期变化（百分点）")
    axs[0,1].set_title("经营阶段不同，压力构成也不同",loc="left",fontweight="bold")
    axs[0,1].legend(frameon=False,loc="upper right")
    items = maotai["segment_rows"][:4]
    bars = axs[1,0].bar([r["item"] for r in items],[r["q3_yoy"]*100 for r in items],color=[blue,blue,orange,orange],width=.55)
    axs[1,0].bar_label(bars,fmt="%+.2f",padding=3)
    axs[1,0].axhline(0,color="#999999",lw=.7)
    axs[1,0].set_ylim(-25,42)
    axs[1,0].set_ylabel("第三季度酒类主营收入同比（%）")
    axs[1,0].set_title("茅台：产品与渠道是两个不同切面",loc="left",fontweight="bold")
    ids = list(dict.fromkeys(r["id"] for r in events))
    for j,h in enumerate([5,20]):
        vals = [next(r["net_return"]*100 for r in events if r["id"]==key and r["holding_sessions"]==h) for key in ids]
        bars = axs[1,1].bar(np.arange(4)+(j-.5)*.34,vals,width=.34,color=[blue,orange][j],label=f"{h}个交易日")
        axs[1,1].bar_label(bars,fmt="%+.2f",padding=3,fontsize=9)
    axs[1,1].set_xticks(np.arange(4),["茅台预披露\n7月17日","茅台半年报\n8月3日","伊利半年报\n9月3日","茅台说明\n10月31日"])
    axs[1,1].axhline(0,color="#999999",lw=.7)
    limits = [r["net_return"]*100 for r in events]
    axs[1,1].set_ylim(min(min(limits)-2,-2),max(max(limits)+3,3))
    axs[1,1].set_ylabel("510300示例成本后收益（%）")
    axs[1,1].set_title("公开后剩余收益：全部固定节点",loc="left",fontweight="bold")
    axs[1,1].legend(frameon=False,loc="upper right")
    fig.suptitle("2018年消费权重：追查变化来源与信息先后",fontsize=18,fontweight="bold")
    fig.supxlabel("公司已见案例的历史追查；包装/竞争说明属于半年，不能自动延伸至第三季度。\n归档次日实际开盘参考交易，每笔预算10万元、收益分母为实际投入；不是完整账户或独立收益验证。",fontsize=10)
    path=OUT/"上游原因与历史阶段.png"
    fig.savefig(path,dpi=150)
    plt.close(fig)
    return str(path)


def write_report(result,y,m,events):
    pct=lambda v:f"{v*100:+.2f}%"
    exp=next(r for r in y["h1_expense_components"] if r["item"]=="广告营销")
    preview=m["h1_preliminary_vs_actual"]
    fname="历史发现_成本竞争渠道与真正的预期差.md"
    lines=["# 历史发现：成本、竞争、渠道与真正的预期差","",
        "本轮继续追查2018年已见下跌案例中的伊利与茅台，只分析当时经营变化、公开先后以及此后已经实现的ETF收益。没有使用当前行情作预测，也没有设立等待未来信息的任务。","",
        "结论是：因子变化能被拆得更具体，但‘经营原因可解释’还没有变成‘能够稳定赚取ETF超额收益’。本轮补清了营销费用与产品渠道结构，并排除了一个会制造假预期差的口径错误；夏普1.2目标仍未实现。","",
        "## 1. 伊利：半年的竞争投入，不能混成第三季度的原奶涨价","",
        "半年报直接说明：公司面临纸类包装及部分辅料采购价上涨，国内原料乳收购价基本稳定；公司将前者与国内生产环境及国际贸易环境联系起来。这是公司对当时行业环境的解释，并没有单独识别政策、供给或汇率各自造成的价格影响。","",
        f"半年销售费用同比增加{y['h1']['incremental_sales_expense_cny']/1e8:.2f}亿元，其中广告营销增加{exp['increase_cny']/1e8:.2f}亿元，占费用增量{exp['share_of_expense_increase']*100:.2f}%。公司明确将营销投入上升与竞争加剧联系起来。薪酬和运输也增加，其中运输部分与销量增加有关。","",
        "|费用项目|半年同比增加（亿元）|占销售费用增量|","|---|---:|---:|"]
    for r in y["h1_expense_components"]:
        lines.append(f"|{r['item']}|{r['increase_cny']/1e8:+.3f}|{r['share_of_expense_increase']*100:.2f}%|")
    lines += ["","公司还报告销量及产品结构改善，并引用第三方调查称常温液态奶零售额份额提升。营销、销售和份额同时上升，只能确认同时发生，不能直接证明投入带来的销量或利润回报。上述金额分解是财务恒等关系，不是营销的因果效应估计。","",
        "|阶段|营业收入同比|归母利润同比|毛利率同比变化|销售费用率同比变化|新增毛利减新增销售费用|",
        "|---|---:|---:|---:|---:|---:|"]
    for key,label in [("h1","2018半年"),("q3","2018第三季度")]:
        a=y[key]
        lines.append(f"|{label}|{pct(a['revenue_yoy'])}|{pct(a['parent_profit_yoy'])}|{a['gross_margin_change_pp']:+.2f}个百分点|{a['sales_expense_ratio_change_pp']:+.2f}个百分点|{a['incremental_gp_less_sales_expense_cny']/1e8:+.2f}亿元|")
    lines += ["",f"半年新增销售费用相当于新增毛利的{y['h1']['sales_increment_over_gp_increment']*100:.2f}%，但毛利率本身仍有所改善。到了第三季度，毛利率转为同比下降，销售费用率仍在上升。两个阶段都表现为收入较难转成利润，但压力构成已经不同。表末一列尚未计入其他费用、收益与税项，不等于净利润变化。","",
        "因此，这个历史阶段里更有解释力的研究对象，是毛利增量能否覆盖竞争费用，以及这两部分分别如何变化。仅看收入增速或原奶价格都不完整。半年报没有给出足以定量分离第三季度包装价格、产品组合和折扣变化的资料，第三季度的上游主因仍不能强行确定。费用率统一以营业收入为分母，避免与营业总收入口径混用。","",
        "## 2. 茅台：主品牌、系列酒和渠道变化需要分别看","",
        "2017年底公司公告自2018年起平均调高茅台酒产品价格约18%。这是已披露的背景条件；它并不意味着每个SKU、渠道、季度实际均价都统一增加18%。因此本轮不将收入增速机械除以1.18来估算销量。","",
        "以下第三季度收入由前三季度累计减去半年累计得到；2017年同期同样处理。产品、渠道、地区是对同一酒类主营收入的不同切面，不能把六项加总。","",
        "|切面|2018半年同比|2018第三季度同比|第三季度收入增量（亿元）|","|---|---:|---:|---:|"]
    for r in m["segment_rows"]:
        lines.append(f"|{r['item']}|{pct(r['h1_yoy'])}|{pct(r['q3_yoy'])}|{r['q3_change_wan_cny']/10000:+.2f}|")
    lines += ["",f"两类酒的第三季度主营收入合计同比增加{m['core_liquor_q3_revenue_increase_wan_cny']/10000:.2f}亿元，其中系列酒贡献{m['series_share_of_core_liquor_revenue_increase']*100:.2f}%的增量。主品牌增长接近停滞、系列酒仍增长，以及直销与批发反向变化，是可以核实的构成；这些都不能单独证明终端消费者需求下降或渠道库存增加。","",
        "上一轮已核对2017年第三季度的高基数及财务公司对合并现金流的影响。本轮产品与渠道拆分进一步缩小解释范围，但季度销量、实现均价和渠道库存仍未识别。不能因提价已发生，就把剩余收入变化全部归因于销量。","",
        "## 3. 预期差先对齐对象与时间，避免凭空制造利空","",
        f"公司7月16日初步核算预披露半年营业总收入约350亿元，正式半年报为{preview['actual_total_revenue_cny']/1e8:.2f}亿元，相对粗略值为{pct(preview['actual_relative_to_approximate_preliminary'])}，两者接近。正式利润总额同比{pct(preview['actual_profit_before_tax_yoy'])}，预披露为约40%。两个预披露都是近似数，并非精确预测区间。","",
        f"如果错把营业收入{preview['incompatible_operating_revenue_cny']/1e8:.2f}亿元与营业总收入预披露350亿元相比，就会得到{pct(preview['wrong_scope_example_relative_to_preliminary'])}的‘缺口’。这来自把不含金融利息等收入的子项与总项相比，不能称为业绩低于预期。","",
        "相对于公司的预披露不等于相对于市场一致预期。本轮没有取得当时完整、可比的市场预期分布，不能声称已经测得市场预期差。10月30日公司的经营说明晚于10月29日三季报，所说符合预期也明确指本公司预期；既不能向前填入三季报之前，也不能冒充此前的市场共识。","",
        "## 4. 四个预先固定的历史节点，公开后还剩什么","",
        "四个事件名称与5/20日期限在本轮新收益计算前固定；归档日期也先另存后计算。统一按归档日结束后下一实际交易日开盘进入510300，到指定交易日开盘退出。巨潮目录的日期能支持这个保守回放口径，但本轮没有验证最早的精确公开时刻。","",
        "|历史披露节点|参考入场|5日成本后收益|20日成本后收益|","|---|---|---:|---:|"]
    for event_id in dict.fromkeys(r["id"] for r in events):
        a=next(r for r in events if r["id"]==event_id and r["holding_sessions"]==5)
        b=next(r for r in events if r["id"]==event_id and r["holding_sessions"]==20)
        lines.append(f"|{a['issuer']}：{a['title']}|{a['entry_date']}|{pct(a['net_return'])}|{pct(b['net_return'])}|")
    lines += ["","每筆示例预算10万元，佣金单边万四且最低5元，滑点单边0.1%，按千分之一元价位向不利方向取整、100份整手，分红按登记日权益处理。收益分母为实际投入，并非20万元完整账户。所有实际开盘价、入场退出日期、股数与费用均已保存。","",
        "这些是披露后对应时段的ETF回报，不是公司消息造成的因果回报，未扣除其他宏观消息或共同市场走势。四个节点来自已知案例，部分窗口重叠，也与上一轮10月案例重叠；不能当作独立样本计算事件夏普或择优保留期限。","",
        "## 5. 本轮留下的可用规律与下一步历史问题","",
        "1. 先拆金额组成，再使用因子名称。收入增长、毛利率、销售费用、现金流可能分别受不同上游驱动。",
        "2. 以阶段核对机制。伊利半年已有竞争投入压力，第三季度另外出现毛利率下降；同一句成本上升不能覆盖两个阶段。",
        "3. 预期差必须比较同一口径、同一期间、同一预期主体，并保留信息先后。错口径或事后公司解释都不能构成事前意外。",
        "4. 解释清楚并未自动产生指数择时优势。归档后收益按固定期限全数保留，目前没有足以升级为历史完整账户的新规则。","",
        result["next_historical_question"],"",
        f"本轮读取4份新增公司原始PDF和2个公司原文网页，复用4份已有PDF；核对{result['financial_source_rows']}条金额/预披露记录和8条参考交易。只做原表加总、季度差分、日期与交易费用的必要核对，没有新增大规模参数搜索、完整账户或前瞻任务。目标仍为完整账户成本后夏普不低于1.2，并未达成。","",
        f"![上游原因与历史阶段]({(OUT/'上游原因与历史阶段.png').as_posix()})","",
        "## 原文与复算入口","",
        "- [伊利2018年半年报](https://static.cninfo.com.cn/finalpage/2018-08-31/1205361468.PDF)：物理第9、11、12、115页对应包装成本、经营变化和费用明细。",
        "- [茅台2018上半年经营数据公告](https://static.cninfo.com.cn/finalpage/2018-07-16/1205157417.PDF)：营业总收入与利润总额的近似预披露。",
        "- [茅台2018半年报](https://static.cninfo.com.cn/finalpage/2018-08-02/1205249471.PDF)：第7页产品渠道，第27—28页合并利润表。",
        "- [茅台2017半年报](https://static.cninfo.com.cn/finalpage/2017-07-28/1203740255.PDF)、[2017三季报](https://static.cninfo.com.cn/finalpage/2017-10-26/1204071279.PDF)、[2018三季报](https://static.cninfo.com.cn/finalpage/2018-10-29/1205549057.PDF)：按同口径拆单季。",
        "- [伊利2018三季报](https://static.cninfo.com.cn/finalpage/2018-10-31/1205561719.PDF)：本地已有原件，使用营业收入、成本、销售费用与归母利润。",
        "- [茅台价格调整公告原站页面](https://www.moutaichina.com/mtgf/2017-12/28/article_2023111312125696870.html)：正文落款2017年12月28日；当前网页迁移时间不当作首次公开时间。",
        "- [茅台2018年10月经营说明](https://static.cninfo.com.cn/finalpage/2018-10-30/1205559482.PDF)、[次日指定报刊原文](https://epaper.cs.com.cn/zgzqb/html/2018-10/31/nw.D110000zgzqb_20181031_6-B009.htm)：公司预期及解释的先后。","",
        "本轮伊利业绩推介PDF未成功取得，不依据搜索摘要补造正文。原件来源状态保存在source_index.json；口径和金额在原值与口径页码.json；推导在上游原因与金额分解.json；交易在四个披露节点510300收益.json。脚本research/historical_consumer_driver_origins_v1.py的analyze模式完全使用本地原件与既有行情重算。",""]
    (OUT/fname).write_text("\n".join(lines).replace("每筆","每笔"),encoding="utf-8")
    return fname


def analyze():
    cfg=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    source_index=json.loads((OUT/"source_index.json").read_text(encoding="utf-8"))
    acquired_pdfs=[r for r in source_index if r["status"]=="PDF_TEXT_ACQUIRED"]
    assert len(acquired_pdfs)==4
    yili,maotai,checks=source_facts()
    events=event_returns(cfg)
    checks.update(status="PASS_NECESSARY_AMOUNT_IDENTITY_AND_EVENT_CLOCK_CHECKS",event_return_rows=8,
                  first_public_timestamp_verified=False,independent_validation=False,
                  required_source_pages_visually_reviewed=True,figure_visually_reviewed=False)
    save("calculation_checks.json",checks)
    result={"study_id":STUDY,"completed_at":now(),"research_mode":"HISTORICAL_ONLY",
            "status":"HISTORICAL_UPSTREAM_EXPENSE_CHANNEL_AND_COMPARABLE_EXPECTATION_COMPLETED",
            "classification":"PROGRESS_HISTORICAL_CONSUMER_DRIVER_ORIGINS_AND_COMPARABLE_DISCLOSURE",
            "primary_company_pdfs_new":4,"primary_company_pdfs_reused":4,"company_original_webpages":2,
            "financial_source_rows":checks["financial_source_rows"],"unique_etf_nodes":4,"event_return_rows":8,
            "known_case_selected_sample":True,"independent_validation":False,
            "new_full_accounts":0,"new_parameters_fitted":0,"new_prospective_tasks":0,
            "newly_achieved_sharpe":None,"goal_achieved":False,"orders_authorized":False,
            "mechanism_result":"上游原因与金额分解.json","event_result":"四个披露节点510300收益.json",
            "next_historical_question":"在事先固定的2018年权重公司范围与半年、三季度披露日历内，成组对齐已有预披露和正式值；纳入全部可取得事件并保留缺失项，按当时可知经营机制比较，不按事后涨跌、显著性或ETF收益挑选样本。"}
    result["figure"]=render_figure(yili,maotai,events)
    result["report"]=write_report(result,yili,maotai,events)
    save("result.json",result)
    print(json.dumps(clean({"状态":result["status"],"必要核对":checks,
                      "伊利阶段":{key:yili[key] for key in ["h1","q3"]},
                      "茅台产品渠道":maotai["segment_rows"],
                      "历史节点收益":[{k:r[k] for k in ["id","entry_date","exit_date","holding_sessions","net_return"]} for r in events]}),ensure_ascii=False,indent=2),flush=True)


def record_progress():
    """仅登记已完成的历史发现，保留原目标、冻结失败及暂停项。"""
    result=json.loads((OUT/"result.json").read_text(encoding="utf-8"))
    checks=json.loads((OUT/"calculation_checks.json").read_text(encoding="utf-8"))
    events=json.loads((OUT/"四个披露节点510300收益.json").read_text(encoding="utf-8"))
    residuals=[]
    for row in events:
        pnl=(row["sell_price"]-row["buy_price"])*row["shares"]-row["commissions_cny"]+row["dividend_entitlement_cny"]
        residuals.append(abs(pnl-row["net_pnl_cny"]))
        assert row["paid_cny"]<=100000 and row["shares"]%100==0
        assert row["exit_index"]-row["entry_index"]==row["holding_sessions"]
        assert row["archive_date"]<row["entry_date"]<row["exit_date"]
        assert abs(row["net_pnl_cny"]/row["paid_cny"]-row["net_return"])<1e-12
    assert len(events)==8 and max(residuals)<.01
    checks.update(saved_event_pnl_recomputation_max_abs_cny=max(residuals),
                  budget_lots_horizons_and_next_day_exit_checks=True,figure_visually_reviewed=True)
    save("calculation_checks.json",checks)
    prefix="reports/research/510300_historical_consumer_driver_origins_v1/"
    stamp=now()
    receipt={"recorded_at":stamp,"study_id":STUDY,"research_mode":"HISTORICAL_ONLY",
             "user_authority":"不做前瞻性，只做历史检验和发现；减少繁琐检验程序，先找到确定性和规律。",
             "result":prefix+"result.json","goal_service_status_observed":"active","goal_achieved":False,
             "changes":[],"old_rejections_retained":True,"orders_authorized":False}
    paths=[ROOT/"config/510300_historical_cause_discovery_v1.json",ROOT/"config/510300_existing_data_training_mandate_v1.json"]
    for path in paths:
        cfg=json.loads(path.read_text(encoding="utf-8"))
        before=dict(cfg)
        if path.name=="510300_historical_cause_discovery_v1.json":
            cfg.update(latest_completed_study=prefix+"result.json",latest_report=prefix+result["report"],updated_at=stamp)
        else:
            cfg.update(current_round=STUDY,latest_progress_receipt=prefix+"result.json",
                       latest_continuation_report=prefix+result["report"],latest_historical_report=prefix+result["report"],
                       latest_historical_consumer_driver_origins=prefix+"result.json",
                       latest_continuation_classification=result["classification"],current_driver_continuation_classification=result["classification"],
                       current_driver_consecutive_blocked_goal_turns=0,latest_historical_diagnostic_at=stamp,
                       latest_goal_service_status="active",latest_goal_service_status_observed_at=stamp,
                       goal_status="active",goal_achieved=False,local_goal_work_status="ACTIVE_HISTORICAL_ONLY",
                       last_research_result="伊利半年销售费用增量约三分之二来自营销，毛利率半年改善而第三季度下降；茅台单季主品牌近乎持平、系列酒仍增长，直销与批发方向不同。半年预披露与实际营业总收入口径一致时接近，混用营业收入会造假缺口。完成四个固定归档节点八条成本后ETF收益，尚无新完整账户或达标结果。",
                       last_source_result="新增4份公司原始PDF及2个公司原文网页，复用4份已有PDF；核对24条金额与预披露记录。伊利推介材料未取得，未以摘要替代；最早公开精确时点、市场共识及终端量价仍未识别。",
                       next_research_question=result["next_historical_question"])
        receipt["changes"].append({"path":str(path),"fields":{k:{"before":before.get(k),"after":v} for k,v in cfg.items() if before.get(k)!=v}})
        path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    save("authority_update.json",receipt)
    print("已登记历史原因发现与八条保存收益复算；夏普1.2目标仍在进行中且未达标。",flush=True)


def main():
    parser=argparse.ArgumentParser(description="消费权重历史上游原因发现")
    parser.add_argument("mode",choices=["prepare","directory","fetch","analyze","record-progress"])
    args=parser.parse_args()
    {"prepare":prepare,"directory":directory,"fetch":fetch,"analyze":analyze,"record-progress":record_progress}[args.mode]()


if __name__=="__main__":main()
