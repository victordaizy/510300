"""历史降准的银行资金机制、宏观状态与指数整体反应。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pdfplumber
import requests
from bs4 import BeautifulSoup

from historical_price_gap_causes_v1 import round_trip

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/research/510300_historical_index_liquidity_transmission_v1"
MACRO=ROOT/"reports/research/510300_macro_dynamic_reframe_v1/inputs"
BREADTH=ROOT/"reports/research/510300_factor96_crowding_overlay_v1_0_1/inputs"
STUDY="510300_HISTORICAL_INDEX_LIQUIDITY_TRANSMISSION_V1"
TSF=ROOT/"data/raw/macro/510300_macro_stress_2015_v2/tsf_stock_yoy_release_vintage_2015_2026.parquet"
SOURCES=[
    {"id":"pbc_2018q2","title":"2018年第二季度中国货币政策执行报告","published_date":"2018-08-10","kind":"pdf","url":"https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/3537682/5dad7896c1c945c5be61518cb4a06605/2018081021363898908.pdf"},
    {"id":"pbc_2018q4","title":"2018年第四季度中国货币政策执行报告","published_date":"2019-02-21","kind":"pdf","url":"https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/3537682/f8d2bf5c14f940bcb7c8bf81df786d89/2019022119340790088.pdf"},
    {"id":"pbc_2019q2","title":"2019年第二季度中国货币政策执行报告","published_date":"2019-08-09","kind":"pdf","url":"https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/3830536/51f3cba3238746d6bce56101159d931e/2019080921013757815.pdf"},
    {"id":"pbc_2019q3","title":"2019年第三季度中国货币政策执行报告","published_date":"2019-11-15","kind":"pdf","url":"https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/3830536/ed13cc6378964d7eb67a85649b922cd4/2019111521260575804.pdf"},
    {"id":"pbc_201810_qa","title":"2018年10月降准答问（央行英文版）","published_date":"2018-10-09","kind":"html","url":"https://xining.pbc.gov.cn/en/3688110/3688172/4048314/2025092319422421099/index.html"},
    {"id":"pbc_2019_chronology","title":"2019年中国货币政策大事记","published_date":"2020-02-19","kind":"html","url":"https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212550560783/index.html"},
    {"id":"gov_20190104","title":"全面降准1个百分点 长期增量资金支持实体经济","published_date":"2019-01-05","kind":"html","url":"https://app.www.gov.cn/govdata/gov/201901/05/433817/article.html"},
    {"id":"gov_20180620","title":"2018年6月20日国务院常务会议","published_date":"2018-06-20","kind":"html","url":"https://app.www.gov.cn/govdata/gov/201806/20/425891/article.html"},
    {"id":"gov_20190904","title":"2019年9月4日国务院常务会议","published_date":"2019-09-04","kind":"html","url":"https://app.www.gov.cn/govdata/gov/201909/04/448004/article.html"},
    {"id":"ustr_20190510","title":"USTR关于301关税行动的声明","published_date":"2019-05-10","kind":"html","url":"https://ustr.gov/about-us/policy-offices/press-office/press-releases/2019/may/statement-us-trade-representative"},
    {"id":"pbc_2019q1","title":"2019年第一季度中国货币政策执行报告","published_date":"2019-05-17","kind":"pdf","url":"https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/3830536/e0d70bbde6d34378bde1fcc3713126ec/2019051720074058265.pdf"},
    {"id":"gov_20180417_replacement","title":"中国央行宣布部分降准（中新网原报道官方转载）","published_date":"2018-04-18","kind":"html","url":"https://www.jstz.gov.cn/a/20180418/1524018214509.shtml"},
    {"id":"gov_20180624","title":"央行定向降准0.5个百分点（新华社电稿）","published_date":"2018-06-25","kind":"html","url":"https://czj.ezhou.gov.cn/zxzx/czxx/201806/t20180625_231398.html"},
    {"id":"pbc_201810_gazette","title":"2018年10月央行文告：银发〔2018〕231号","published_date":None,"kind":"pdf","url":"https://www.pbc.gov.cn/chubanwu/114566/114579/4356045/4356215/2021100915291988091.pdf"},
]


def now():return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,(datetime,pd.Timestamp)):return value.isoformat()
    if isinstance(value,np.generic):return clean(value.item())
    if value is pd.NaT or value is pd.NA or isinstance(value,float) and not np.isfinite(value):return None
    return value


def save(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def prepare():
    if (OUT/"protocol.json").exists():raise RuntimeError("已有设定，保留原文。")
    prior=ROOT/"reports/research/510300_historical_consumer_revision_breadth_v1/protocol.json"
    costs=json.loads(prior.read_text(encoding="utf-8"))["costs"]
    dates=["2018-04-17","2018-06-24","2018-10-07","2019-01-04","2019-05-06","2019-09-06"]
    save("protocol.json",{"study_id":STUDY,"recorded_at":now(),"research_mode":"HISTORICAL_ONLY",
        "user_scope_correction":"你太关注个股了，指数和个股还是不太一样",
        "research_unit":"510300所对应的指数整体，不再沿公司案例扩展；成分只聚合为上涨覆盖面。",
        "question":"相同方向的降准为什么不等于指数上涨：资金置换、增量银行资金、需求状态与市场整体响应分别是什么？",
        "event_scope":"2018—2019年内新宣布且同年开始实施的法定准备金率降低，完整保留六个公告节点。",
        "events":[{"id":"RRR_"+date.replace("-",""),"date":date} for date in dates],
        "excluded_by_definition":["2018年1月实施的2017年已宣布定向降准，不重复计新公告","临时准备金动用安排、MLF和逆回购操作，另作机制背景，不混为降准公告"],
        "known_results":"已见2018、2019总体市场走势和多个相邻政策案例；这是历史发现，不声称独立验证。",
        "event_entry":"载明公告日结束后下一实际交易日开盘统一观察；不按收益选择是否采用公告日盘中入场。",
        "pre_entry_move":"公告日前最后收盘到统一入场开盘的变动，仅是时间窗口描述，可能混合公告日前交易及同时发生的其他消息。",
        "macro_state":"只取公告日00:00之前已有的最近官方PMI新订单及M1/M2发布，保留统计月份和公布日；同比不能代替市场预期差。",
        "credit_mechanism":"官方释放金额拆为置换和增量；若8000亿元为多工具合计，不冒称降准单项净投放。没有数据则缺失，不反算未知量。",
        "horizons_sessions":[5,20],"costs":costs,
        "breadth":"原入场前一交易日固定300成分，复用股东总回报计算此后5/20日上涨占比及等权回报；缺失不补零，明确覆盖数量，只作整体响应不生成个股研究。",
        "breadth_clock":"原入场前一收盘至入场后第5/20个交易日收盘，与ETF开盘到开盘交易口径不同，分别列示。",
        "new_parameters_fitted":0,"new_full_accounts":0,"new_prospective_tasks":0,
        "same_day_information_or_source_clock_uncertainty_retained":True,
        "goal_achieved":False,"orders_authorized":False})
    save("scope_adjustment.json",{"recorded_at":now(),"user_instruction":"你太关注个股了，指数和个股还是不太一样",
         "prior_branch":"510300_HISTORICAL_CONSUMER_REVISION_BREADTH_V1","prior_branch_disposition":"CLOSED_AS_LOCAL_WEIGHT_BOUNDARY",
         "new_primary_focus":"指数货币信用传导、总需求、风险溢价与整体资金约束",
         "single_company_role":"仅在解释指数影响范围时使用，不作为主线，也不把公司超预期直接变成指数信号。"})
    path=ROOT/"config/510300_historical_cause_discovery_v1.json"
    cfg=json.loads(path.read_text(encoding="utf-8"))
    cfg.update(research_unit="历史指数整体驱动变化、当时可用信息及其后已实现指数收益",
               research_focus="INDEX_LEVEL_TRANSMISSION",single_company_role="仅用于必要的权重范围说明，不继续扩展个股主线",
               latest_scope_correction="你太关注个股了，指数和个股还是不太一样",latest_scope_correction_at=now(),
               current_study="reports/research/510300_historical_index_liquidity_transmission_v1/protocol.json")
    path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("已按用户纠偏固定六个历史降准节点，研究对象改为指数整体。",flush=True)


def fetch():
    if (OUT/"result.json").exists():
        raise RuntimeError("已有结果，不改变研究输入。")
    if not (OUT/"credit_context_addendum.json").exists():
        save("credit_context_addendum.json",{
            "recorded_at":now(),"before_new_event_return_computation":True,
            "addition":"加入公告前已发布社融存量同比，只描述已知信用状态，不设分组或择时阈值。",
            "source":str(TSF.relative_to(ROOT)),
            "definition_breaks":"2018年7月加入存款类金融机构资产支持证券和贷款核销；9月加入地方政府专项债。保留每次原公布口径，不计算跨口径环比变化，不当作政策预期差。",
            "later_reports":"央行季度报告用于历史机制和事件日期核对。迟于事件的文本不冒充当时交易输入，也不用于按后续结果划分状态。"})
    def one(source):
        item=dict(source)
        target=OUT/"sources"/(source["id"]+"."+source["kind"])
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            response=requests.get(source["url"],timeout=(15,40),headers={"User-Agent":"Mozilla/5.0"})
            response.raise_for_status()
            if source["kind"]=="pdf" and not response.content.startswith(b"%PDF"):
                raise ValueError("未获得PDF："+source["id"])
            target.write_bytes(response.content)
        data=target.read_bytes()
        if source["kind"]=="pdf":
            with pdfplumber.open(target) as document:
                pages=[{"pdf_page":i+1,"text":p.extract_text() or ""} for i,p in enumerate(document.pages)]
            save("sources/"+source["id"]+"_pages.json",pages)
            body="\n\n".join("PDF页 "+str(p["pdf_page"])+"\n"+p["text"] for p in pages)
            item["pages"]=len(pages)
        else:
            soup=BeautifulSoup(data,"html.parser")
            for node in soup(["script","style"]):node.decompose()
            body=soup.get_text("\n",strip=True)
            if len(body)<100 or "您所访问的页面不存在" in body or "Access Denied" in body:
                raise ValueError("网页未提供有效正文："+source["id"])
        target.with_suffix(".txt").write_text(body,encoding="utf-8")
        item.update(path=str(target.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),retrieved_at=now(),status="RETRIEVED")
        print("已保存官方来源："+source["title"],flush=True)
        return item
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[(source,pool.submit(one,source)) for source in SOURCES]
        for source,future in futures:
            try:results.append(future.result())
            except Exception as exc:
                results.append({**source,"status":"MISSING","error":str(exc)})
                print("来源缺失："+source["id"]+"；"+str(exc),flush=True)
    save("source_manifest.json",results)
    print("来源处理完成，共"+str(len(results))+"项。",flush=True)


def macro_states(protocol):
    money=pd.read_csv(MACRO/"money_104.csv")
    pmi=pd.read_parquet(MACRO/"pmi_new_orders.parquet")
    tsf=pd.read_parquet(TSF)
    money["usable_time"]=pd.to_datetime(money.available_at_upper_bound,utc=True)
    pmi["usable_time"]=pd.to_datetime(pmi.available_at,utc=True)
    tsf["usable_time"]=pd.to_datetime(tsf.available_at,utc=True)
    result=[]
    for event in protocol["events"]:
        cutoff=pd.Timestamp(event["date"],tz="Asia/Shanghai")
        row={"event_id":event["id"],"announcement_date":event["date"],"information_cutoff":cutoff.isoformat()}
        for label,frame in [("money",money),("pmi",pmi),("tsf",tsf)]:
            admitted=frame[frame.usable_time<cutoff].sort_values("usable_time")
            if admitted.empty:raise ValueError("缺少公告前信息："+event["id"]+label)
            item=admitted.iloc[-1]
            row[label+"_month"]=item.stat_month if label=="money" else item.reference_period
            row[label+"_available_at"]=item.usable_time.isoformat()
            row[label+"_source_url"]=item.source_url
            if label=="money":
                row.update(m1_yoy_pct=item.m1_yoy_pp,m2_yoy_pct=item.m2_yoy_pp,money_definition=item.definition_version)
            elif label=="pmi":row["pmi_new_orders"]=item.first_release_value
            else:
                row["tsf_yoy_pct"]=item.first_release_value
                row["tsf_raw_path"]=item.raw_path
                body=BeautifulSoup((ROOT/item.raw_path).read_text(encoding="utf-8"),"html.parser").get_text(" ",strip=True)
                start=body.find("初步统计")
                end=body.find("从结构看",start)
                paragraph=body[start:end]
                row["tsf_component_source_text"]=paragraph
                for key,label in [("rmb_loan","人民币贷款"),("entrusted_loan","委托贷款"),("trust_loan","信托贷款"),("undiscounted_bill","未贴现的银行承兑汇票"),("corporate_bond","企业债券"),("local_special_bond","地方政府专项债券")]:
                    match=re.search(re.escape(label)+r"余额(?:为)?([\d.]+)万亿元，同比(增长|下降)([\d.]+)%",paragraph)
                    row[key+"_stock_trillion_cny"]=float(match[1]) if match else None
                    row[key+"_yoy_pct"]=(1 if match[2]=="增长" else -1)*float(match[3]) if match else None
                month=str(item.reference_period)
                row["tsf_definition"]="2018年7月以前口径" if month<"2018-07" else "含资产证券化及贷款核销" if month<"2018-09" else "另含地方政府专项债"
        result.append(row)
    return result


def mechanism_cards():
    return [
        {"announcement_date":"2018-04-17","implementation_dates":["2018-04-25"],"rrr_released_yi_cny":13000,"mlf_replacement_yi_cny":9000,"increment_after_named_replacement_yi_cny":4000,
         "mechanism":"多数释放资金偿还MLF，余下约4000亿元还与税期对冲；改善银行负债期限和成本，不是向股票账户投放1.3万亿元。",
         "source_ids":["pbc_2018q2","gov_20180417_replacement"],"prior_policy_guidance":"未在本轮建立完整的公告前市场共识，预期差保持未知。"},
        {"announcement_date":"2018-06-24","implementation_dates":["2018-07-05"],"rrr_released_yi_cny":7000,"mlf_replacement_yi_cny":None,"increment_after_named_replacement_yi_cny":None,
         "mechanism":"约5000亿元支持市场化法治化债转股，约2000亿元支持小微企业融资；专项银行资金用途不等于二级股票市场净买入。",
         "source_ids":["pbc_2018q2","gov_20180624","gov_20180620"],"prior_policy_guidance":"6月20日国常会已提出定向降准和推进债转股；方向已有公开铺垫，幅度是否超预期未知。"},
        {"announcement_date":"2018-10-07","implementation_dates":["2018-10-15"],"rrr_released_yi_cny":12000,"mlf_replacement_yi_cny":4500,"increment_after_named_replacement_yi_cny":7500,
         "mechanism":"约4500亿元替换当日到期MLF，约7500亿元为扣除此项后的增量，其中部分仍对冲税期；政策旨在缓解长期资金约束。",
         "source_ids":["pbc_201810_qa","pbc_201810_gazette"],"prior_policy_guidance":"本轮未取得完整的公告前市场共识，且国庆休市期间海外消息也进入下一开盘。"},
        {"announcement_date":"2019-01-04","implementation_dates":["2019-01-15","2019-01-25"],"rrr_released_yi_cny":None,"mlf_replacement_yi_cny":None,"increment_after_named_replacement_yi_cny":None,
         "multi_tool_package_yi_cny":8000,"later_rrr_minus_mlf_yi_cny_text":"3000多亿元",
         "mechanism":"8000亿元是降准置换MLF、普惠金融降准动态考核、TMLF等合计口径。后续央行报告将降准减去一季度MLF不续做列为3000多亿元，不能用8000亿元代表降准单项新增资金。",
         "source_ids":["gov_20190104","pbc_2018q4","pbc_2019_chronology"],"prior_policy_guidance":"没有可量化的公告前降准幅度共识，不能从事后上涨倒推超预期。"},
        {"announcement_date":"2019-05-06","implementation_dates":["2019-05-15","2019-06-17","2019-07-15"],"rrr_released_yi_cny":None,"mlf_replacement_yi_cny":None,"increment_after_named_replacement_yi_cny":None,
         "later_report_released_yi_cny":3000,
         "mechanism":"服务县域的农商行降至8%准备金档次，资金用于民营和小微企业贷款；5月17日发布的一季度报告及二季度报告均写约3000亿元。该数用于事后机制说明，不冒充5月6日公告原数，也不把它算作股票净流入。",
         "source_ids":["pbc_2019q1","pbc_2019q2","pbc_2019_chronology","ustr_20190510"],"prior_policy_guidance":"统一在公告日后开盘观察，故从5月7日开始；窗口内5月10日美国关税上调是并行冲击，不能把窗口跌幅全部归因于降准。"},
        {"announcement_date":"2019-09-06","implementation_dates":["2019-09-16","2019-10-15","2019-11-15"],"rrr_released_yi_cny":9000,"mlf_replacement_yi_cny":None,"increment_after_named_replacement_yi_cny":None,
         "mechanism":"全面降准约8000亿元，符合条件城商行额外定向降准约1000亿元；增加银行长期资金并降低成本，但没有建立二级股票买入金额的对应关系。",
         "source_ids":["pbc_2019q3","pbc_2019_chronology","gov_20190904"],"prior_policy_guidance":"9月4日国常会已经公开提出普遍降准和定向降准，9月6日并非方向首次公开。"},
    ]


def calculate_events(protocol):
    market=pd.read_parquet(MACRO/"market.parquet").sort_values("date").reset_index(drop=True)
    market["date"]=pd.to_datetime(market.date).dt.tz_localize(None).dt.normalize()
    dates=pd.DatetimeIndex(market.date)
    dividends=pd.read_csv(ROOT/"data/reference/510300_dividends.csv")
    dividends=dividends[dividends.symbol.eq("510300.SH")].copy()
    dividends["record_date"]=pd.to_datetime(dividends.record_date)
    members=pd.read_parquet(BREADTH/"membership.parquet",columns=["membership_date","symbol"])
    members["membership_date"]=pd.to_datetime(members.membership_date)
    daily=pd.read_parquet(BREADTH/"classified.parquet",columns=["date","symbol","constituent_return_state","daily_total_shareholder_return","return_is_usable"])
    daily["date"]=pd.to_datetime(daily.date)
    allowed=daily.return_is_usable & daily.constituent_return_state.isin(["TRADED_VALID","OFFICIAL_SUSPENSION"])
    allowed &= np.isfinite(daily.daily_total_shareholder_return) & daily.daily_total_shareholder_return.gt(-1)
    daily["usable_return"]=daily.daily_total_shareholder_return.where(allowed)
    daily=daily[(daily.date>=pd.Timestamp("2018-04-01")) & (daily.date<=pd.Timestamp("2019-11-01"))]
    if daily.duplicated(["date","symbol"]).any():raise ValueError("成分收益有重复日期，不静默覆盖。")
    wide=daily.pivot(index="date",columns="symbol",values="usable_return")
    outputs=[];coverage=[]
    for event in protocol["events"]:
        day=pd.Timestamp(event["date"])
        entry_index=dates.searchsorted(day,side="right")
        before_index=dates.searchsorted(day,side="left")-1
        entry=market.iloc[entry_index]
        membership_date=dates[entry_index-1]
        fixed=members.loc[members.membership_date.eq(membership_date),"symbol"].tolist()
        if len(fixed)!=300 or len(set(fixed))!=300:raise ValueError("成分覆盖不是300："+event["id"])
        for horizon in protocol["horizons_sessions"]:
            exit_row=market.iloc[entry_index+horizon]
            selected=dividends[(dividends.record_date>=entry.date)&(dividends.record_date<exit_row.date)]
            dividend=float(selected.cash_dividend_per_share.sum())
            trade=round_trip(float(entry.open),float(exit_row.open),dividend,protocol["costs"])
            sessions=dates[entry_index:entry_index+horizon]
            returns=wide.reindex(index=sessions,columns=fixed)
            complete=returns.notna().all(axis=0)
            total=(1+returns.loc[:,complete]).prod(axis=0)-1
            total=total.mask(np.isclose(total,0,atol=1e-12,rtol=0),0.)
            usable=int(complete.sum());positive=int(total.gt(0).sum())
            row={"event_id":event["id"],"announcement_date":event["date"],"horizon":horizon,
                 "entry_date":entry.date,"exit_date":exit_row.date,"entry_open":float(entry.open),"exit_open":float(exit_row.open),
                 "pre_announcement_close_date":dates[before_index],"pre_announcement_close":float(market.iloc[before_index].close),
                 "pre_entry_window_return":float(entry.open/market.iloc[before_index].close-1),
                 "dividend_per_share":dividend,"gross_open_to_open_return":float((exit_row.open+dividend)/entry.open-1),**trade,
                 "membership_date":membership_date,"breadth_start_close":membership_date,"breadth_end_close":sessions[-1],
                 "fixed_members":300,"usable_members":usable,"missing_members":300-usable,"positive_members":positive,
                 "breadth_positive_fraction_among_usable":positive/usable if usable else None,
                 "breadth_positive_fraction_lower_bound_all300":positive/300,
                 "breadth_positive_fraction_upper_bound_all300":(positive+300-usable)/300,
                 "equal_weight_return_among_usable":float(total.mean()) if usable else None,
                 "median_member_return_among_usable":float(total.median()) if usable else None}
            outputs.append(row)
            coverage.append({"event_id":event["id"],"horizon":horizon,"membership_date":membership_date,
                             "missing_symbols":list(complete[~complete].index),"missing_member_days":int(returns.isna().sum().sum()),
                             "use":"仅用于整体覆盖核对，不生成公司分析；缺失不视为零收益。"})
    return outputs,coverage


def plot(events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family":font.get_name(),"axes.unicode_minus":False,"font.size":11})
    five=[x for x in events if x["horizon"]==5]
    twenty=[x for x in events if x["horizon"]==20]
    y=np.arange(len(twenty));bg="#f4f3ee"
    fig,axes=plt.subplots(1,2,figsize=(14,6.4),gridspec_kw={"width_ratios":[1.2,1]})
    fig.patch.set_facecolor(bg)
    for shift,rows,color,label in [(-.18,five,"#91a2b1","5个交易日"),(.18,twenty,"#267c77","20个交易日")]:
        bars=axes[0].barh(y+shift,[r["net_return"]*100 for r in rows],height=.31,color=color,label=label)
        for bar,row in zip(bars,rows):
            v=row["net_return"]*100
            axes[0].text(v+(.15 if v>=0 else -.15),bar.get_y()+bar.get_height()/2,f"{v:+.2f}%",ha="left" if v>=0 else "right",va="center",fontsize=10)
    axes[0].axvline(0,color="#aaa69e",lw=1)
    axes[0].set_yticks(y,[r["announcement_date"] for r in twenty])
    axes[0].invert_yaxis();axes[0].margins(x=.26)
    axes[0].set(xlabel="510300事件净收益（占实际投入资金，%）",title="相同方向的政策，后续收益不同")
    axes[0].legend(loc="lower left",frameon=False,ncol=2,fontsize=10)
    values=[r["breadth_positive_fraction_among_usable"]*100 for r in twenty]
    axes[1].barh(y,values,height=.5,color=["#267c77" if x>=50 else "#b77650" for x in values])
    for i,row in enumerate(twenty):
        axes[1].text(values[i]+1.5,i,f"{row['positive_members']}/{row['usable_members']} · {values[i]:.1f}%",va="center",fontsize=10)
    axes[1].axvline(50,color="#8c8a82",lw=1,ls="--")
    axes[1].set_yticks(y,[r["announcement_date"] for r in twenty]);axes[1].invert_yaxis()
    axes[1].set(xlim=(0,116),xlabel="20日上涨成分占完整可用成分（%）",title="多数成分是否一起改善")
    for ax in axes:
        ax.set_facecolor(bg);ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color("#c5c0b5");ax.grid(axis="x",alpha=.15);ax.set_axisbelow(True)
    fig.suptitle("510300历史发现｜降准释放银行资金，指数要看整体传导",x=.05,ha="left",fontsize=18)
    fig.text(.05,.025,"固定2018—2019六个公告；下一交易日开盘观察，含压力费用。上涨覆盖面用收盘口径；缺失不补零。非完整账户。",fontsize=10,color="#66635e")
    fig.tight_layout(rect=(0,.07,1,.92));fig.savefig(OUT/"降准机制与指数整体响应.png",dpi=160,facecolor=bg);plt.close(fig)


def analyze():
    if (OUT/"result.json").exists():raise RuntimeError("已有结果，不覆盖。")
    protocol=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    if not (OUT/"credit_context_addendum.json").exists():raise RuntimeError("先固定信用状态补充口径。")
    manifest=json.loads((OUT/"source_manifest.json").read_text(encoding="utf-8"))
    available={x["id"] for x in manifest if x["status"]=="RETRIEVED"}
    missing=[x["id"] for x in SOURCES if x["id"] not in available]
    if missing:raise ValueError("来源缺失，先保留并处理："+", ".join(missing))
    states=macro_states(protocol)
    save("announcement_known_macro_states.json",states)
    save("policy_mechanism_cards.json",mechanism_cards())
    events,coverage=calculate_events(protocol)
    errors=[abs((x["sell_price"]-x["buy_price"])*x["shares"]-x["commissions_cny"]+x["dividend_entitlement_cny"]-x["net_pnl_cny"]) for x in events]
    assert len(events)==12 and max(errors)<.01
    assert all(pd.Timestamp(x["entry_date"])>pd.Timestamp(x["announcement_date"]) for x in events)
    assert all(pd.Timestamp(x["exit_date"])>pd.Timestamp(x["entry_date"]) for x in events)
    for state in states:
        assert all(pd.Timestamp(state[k+"_available_at"])<pd.Timestamp(state["information_cutoff"]) for k in ["money","pmi","tsf"])
    save("event_returns_and_breadth.json",events)
    save("breadth_coverage.json",coverage)
    checks={"event_count":6,"return_rows":12,"all_macro_before_cutoff":True,"entry_after_announcement_date":True,
            "event_pnl_identity_max_abs_cny":max(errors),"membership_count_each":300,
            "missing_members_range":[min(x["missing_members"] for x in events),max(x["missing_members"] for x in events)],
            "new_parameters_fitted":0,"new_full_accounts":0,"not_a_causal_policy_effect_estimate":True}
    save("calculation_checks.json",checks)
    summary=[]
    for horizon in [5,20]:
        rows=[x for x in events if x["horizon"]==horizon];values=np.array([x["net_return"] for x in rows])
        summary.append({"horizon":horizon,"n":6,"positive_events":int((values>0).sum()),"mean_net_pct":float(values.mean()*100),
                        "median_net_pct":float(np.median(values)*100),"worst_net_pct":float(values.min()*100),"best_net_pct":float(values.max()*100)})
    result={"study_id":STUDY,"completed_at":now(),"status":"HISTORICAL_INDEX_LIQUIDITY_TRANSMISSION_COMPLETED",
            "classification":"PROGRESS_HISTORICAL_INDEX_FUNDING_CREDIT_AND_BREADTH",
            "research_unit":"指数整体机制","event_count":6,"source_count":len(manifest),"summary":summary,
            "new_parameters_fitted":0,"new_full_accounts":0,"new_prospective_tasks":0,"independent_validation":False,
            "net_sharpe":None,"goal_achieved":False,"orders_authorized":False,
            "next_historical_question":"沿指数整体继续：在这些历史窗口内，把资金价格、信用结构及共同风险消息的先后顺序对齐，判断原有坏预期何时改变；不用个股案例扩充信号，不凭后续涨跌补造市场共识。"}
    plot(events)
    save("result.json",result)
    print(json.dumps(clean({"宏观状态":[{k:v for k,v in x.items() if not k.endswith('source_text') and not k.endswith('source_url') and not k.endswith('raw_path')} for x in states],"事件结果":events,"汇总":summary,"必要核对":checks}),ensure_ascii=False,indent=2),flush=True)


def report(events,states,result):
    sources={x["id"]:x for x in json.loads((OUT/"source_manifest.json").read_text(encoding="utf-8"))}
    def citation(key,label=None):
        item=sources[key]
        return "["+(label or item["title"])+"]("+item["url"]+")"
    def local(name,label):return "["+label+"](<"+(OUT/name).as_posix()+">)"
    five={x["announcement_date"]:x for x in events if x["horizon"]==5}
    twenty={x["announcement_date"]:x for x in events if x["horizon"]==20}
    state={x["announcement_date"]:x for x in states}
    lines=["# 510300历史发现：从银行流动性到指数整体定价","",
      "已按用户纠偏把研究对象改为指数整体。个股支线收束，成分股只汇总成上涨覆盖面，主要问题是整体现金流预期、折现率、风险溢价与股票资金供求如何变化。",
      "这轮固定2018—2019年六个新公告节点。20日事件净收益两次为正、四次为负，均值−0.28%，中位数−1.48%。这不足以支持把降准作为独立买入条件，也不能证明降准造成了下跌。","",
      "**先看全部六次结果**","",
      "| 降准公告日 | 公告前最新PMI新订单 | 5日净收益 | 20日净收益 | 20日上涨/完整可用成分 |",
      "|---|---:|---:|---:|---:|"]
    for date,row in twenty.items():
        s=state[date]
        lines.append(f"| {date} | {s['pmi_new_orders']:.1f}（{s['pmi_month']}） | {five[date]['net_return']*100:+.2f}% | {row['net_return']*100:+.2f}% | {row['positive_members']}/{row['usable_members']}（{row['breadth_positive_fraction_among_usable']*100:.1f}%） |")
    lines += ["",
      "收益统一从公告日结束后的下一交易日开盘开始，持有5或20个交易日后开盘结束。每次10万元预算，买卖各按0.1%滑点、0.04%佣金且最低5元、0.001元价格档位和100份整数交易；分母为实际投入资金。2019年1月那次事件的20个交易日窗口计入每份0.059元分红。每次都是单独算例，没有拼成20万元完整账户。",
      "上涨覆盖面固定入场前一个交易日的300只成分，按随后5/20个交易日收盘计算股东总回报。它与ETF开盘到开盘的时钟不同，不能用两者差值拆出权重贡献。20日窗口有0—9只缺失，分母明确保留，未补零；四个亏损节点即使把缺失全部当作上涨，也仍不足半数上涨。该覆盖面是事后描述，不是入场时可用信号。","",
      "**原因之一：释放的是哪种资金，往哪里去**","",
      "| 公告日 | 银行资金机制及口径 |",
      "|---|---|"]
    for card in mechanism_cards():
        refs="、".join(citation(key) for key in card["source_ids"] if key not in ["ustr_20190510","gov_20180620","gov_20190904","pbc_2019_chronology"])
        lines.append("| "+card["announcement_date"]+" | "+card["mechanism"]+" "+refs+" |")
    lines += ["",
      "2018年4月还有一个直接说明因子方向可能被误读的例子：降准释放资金后偿还9000亿元MLF，基础货币可以减少9000亿元，同时银行流动性净增加约4000亿元。原因是法定准备金转成可用的超额准备金，偿还MLF又缩减央行与银行间的债权债务。看见央行缩表或基础货币下降，不能直接判断股票面临流动性收紧。这里的金额来自央行事后解释，不冒充交易当日已经完成的投放。"+citation("pbc_2018q2","央行2018年二季度报告，PDF第8—9页"),"",
      "**原因之二：银行信贷增长与其他融资渠道收缩可以同时发生**","",
      "| 公告日 | 事前已公布统计月 | 人民币贷款存量同比 | 委托贷款同比 | 信托贷款同比 | 未贴现票据同比 | M1 / M2同比 |",
      "|---|---|---:|---:|---:|---:|---:|"]
    for s in states:
        lines.append(f"| {s['announcement_date']} | [{s['tsf_month']}]({s['tsf_source_url']}) | {s['rmb_loan_yoy_pct']:+.1f}% | {s['entrusted_loan_yoy_pct']:+.1f}% | {s['trust_loan_yoy_pct']:+.1f}% | {s['undiscounted_bill_yoy_pct']:+.1f}% | {s['m1_yoy_pct']:.1f}% / {s['m2_yoy_pct']:.1f}% |")
    lines += ["",
      "例如2018年10月降准前，8月人民币贷款存量同比增长12.9%，但委托贷款下降6.2%，未贴现票据下降11.0%。这能确认融资结构分化，不能仅用银行贷款增长给整个指数贴上‘宽信用’标签。央行二季度报告也将表外融资收缩、监管规范的短期叠加、贸易环境变化以及资金供求双方意愿列为传导背景。这是上游机制证据；本轮没有识别它们分别造成多少指数涨跌。"+citation("pbc_2018q2","央行2018年二季度报告，PDF第7、59页"),
      "社融总量原公布同比依次为10.5%、10.3%、10.1%、9.9%、10.7%、10.7%，但2018年7月加入资产证券化和贷款核销，9月加入地方政府专项债。不能把这些首次公布的跨口径数字直接连成同口径变化曲线。M1/M2也不是股票净流入，本轮未将其差额拟合成信号。","",
      "**原因之三：宏观水平、政策方向与预期修正不是同一件事**","",
      "2019年1月和9月，事前最新的PMI新订单同为49.7；20日净收益分别为+5.92%和−1.29%，上涨覆盖面分别为196/299和113/299。2019年5月PMI新订单为51.4，20日净收益仍为−1.68%。所以‘订单高于50就买’或‘低于50再加降准就买’都没有在这组反例中得到支持，不能事后反号补规则。",
      "6月24日之前，2018年6月20日国常会已提出定向降准并加快债转股落地；2019年9月6日之前，9月4日国常会已公开提出普遍降准和定向降准。这确认方向已有公开铺垫，但没有给出市场原先预期的精确幅度，更不能证明市场已完全定价。"+citation("gov_20180620")+"、"+citation("gov_20190904"),
      "2019年5月事件窗口又包含5月10日美国对约2000亿美元中国商品的关税从10%升至25%。这是指数共同风险的并行变化；不能将之后的整体跌幅归为降准的因果效应，也不能拿5月10日的材料伪装成5月6日前已知信息。"+citation("ustr_20190510"),
      "公告日前最后收盘至统一入场开盘的价格变化，六次依次为−0.60%、+0.33%、−2.19%、+3.05%、−5.34%、+1.35%。这些区间包含公告日交易、节假日期间消息或其他冲击，只说明进场前已有价格变化，不能直接标注为纯政策反应。尤其2019年5月统一从5月7日开盘观察，不包含5月6日当天的可交易收益。","",
      "**指数研究的主线**","",
      "| 层次 | 解释指数时真正需要回答的问题 |",
      "|---|---|",
      "| 整体盈利 | 名义需求、信用结构和利润率的变化，是否足以改变按行业权重汇总的盈利预期？ |",
      "| 折现率与风险补偿 | 资金成本、外部金融条件和共同尾部风险发生了什么变化，为什么投资者要求的回报会改变？ |",
      "| 股票资金供求 | 融资约束、申赎、风险减仓与新增配置意愿是否改善，银行资金是否真正经过这些渠道传入股票定价？ |",
      "| 预期与价格 | 消息相比此前公开信息多了什么，买入价格已经反映多少，剩余变化能否覆盖成本？ |",
      "| 整体响应 | 加权指数、等权表现和上涨覆盖面是否一致，还是仅少数权重推动？广度用作描述或之后新决策的信息，不能提前使用。 |","",
      "本轮获得的是可用的机制区分与历史反例，尚未找到可宣称确定获利的组合条件。下一步问题集中于这些指数窗口内信用约束与共同风险预期改变的先后顺序，不再沿公司个案无限扩展。目标仍为20万元完整账户成本后夏普至少1.2；当前没有新增完整账户，目标未达成。","",
      "来源与计算保留在"+local("announcement_known_macro_states.json","公告前宏观状态")+"、"+local("policy_mechanism_cards.json","机制证据")+"、"+local("event_returns_and_breadth.json","六节点收益与覆盖面")+"、"+local("source_manifest.json","来源目录")+"。只做了信息时点、持有期、分红费用和覆盖数量的必要核对，没有搜索参数或新增长期检验。",
      "已修正两处整理问题：失效的4月公告转载页不作证据，改用可读报道交叉核对日期；2019年5月的央行季度报告均为约3000亿元，未把新闻常见的2800亿元错记成季度报告数字。复利计算中1e−12以内的零值浮点误差按零处理，避免将没有涨跌误数为上涨。以上不改变ETF事件收益。"]
    name="历史发现_降准机制与指数整体响应.md"
    (OUT/name).write_text("\n".join(lines)+"\n",encoding="utf-8")
    return name


def finalize():
    result=json.loads((OUT/"result.json").read_text(encoding="utf-8"))
    protocol=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    before=json.loads((OUT/"event_returns_and_breadth.json").read_text(encoding="utf-8"))
    events,coverage=calculate_events(protocol)
    assert len(before)==len(events)
    changes=[]
    for old,new in zip(before,events):
        assert old["event_id"]==new["event_id"] and old["horizon"]==new["horizon"]
        assert old["net_return"]==new["net_return"] and old["net_pnl_cny"]==new["net_pnl_cny"]
        if old["positive_members"]!=new["positive_members"]:
            changes.append({"event_id":new["event_id"],"horizon":new["horizon"],"before":old["positive_members"],"after":new["positive_members"]})
    manifest=json.loads((OUT/"source_manifest.json").read_text(encoding="utf-8"))
    available={x["id"] for x in manifest if x["status"]=="RETRIEVED"}
    assert all(x["id"] in available for x in SOURCES)
    states=json.loads((OUT/"announcement_known_macro_states.json").read_text(encoding="utf-8"))
    cards=mechanism_cards()
    save("policy_mechanism_cards.json",cards)
    save("event_returns_and_breadth.json",events)
    save("breadth_coverage.json",coverage)
    checks=json.loads((OUT/"calculation_checks.json").read_text(encoding="utf-8"))
    checks.update(numerical_zero_tolerance=1e-12,numerical_zero_count_corrections=changes,
                  etf_returns_unchanged_after_annotation_corrections=True,source_records=len(manifest),admitted_sources=len(available),
                  source_and_amount_notes_corrected_at=now())
    save("calculation_checks.json",checks)
    result.update(source_count=len(available),source_records=len(manifest),report=report(events,states,result),finalized_at=now())
    save("result.json",result)
    plot(events)
    print(json.dumps(clean({"说明":"已完成指数主线报告，ETF收益未改变。","报告":result["report"],"零值计数修正":changes,
                           "20日结果":[{k:x[k] for k in ["announcement_date","net_return","positive_members","usable_members"]} for x in events if x["horizon"]==20]}),ensure_ascii=False,indent=2),flush=True)


def record_progress():
    result=json.loads((OUT/"result.json").read_text(encoding="utf-8"))
    assert (OUT/result["report"]).is_file()
    prefix="reports/research/510300_historical_index_liquidity_transmission_v1/"
    stamp=now();receipt={"recorded_at":stamp,"study_id":STUDY,"result":prefix+"result.json",
        "user_scope_correction":"你太关注个股了，指数和个股还是不太一样","changes":[],"goal_achieved":False,"orders_authorized":False}
    for name in ["510300_historical_cause_discovery_v1.json","510300_existing_data_training_mandate_v1.json"]:
        path=ROOT/"config"/name
        cfg=json.loads(path.read_text(encoding="utf-8"));before=dict(cfg)
        if name=="510300_historical_cause_discovery_v1.json":
            cfg.update(latest_completed_study=prefix+"result.json",latest_report=prefix+result["report"],updated_at=stamp)
        else:
            cfg.update(current_round=STUDY,latest_progress_receipt=prefix+"result.json",latest_historical_index_liquidity_transmission=prefix+"result.json",
                latest_continuation_report=prefix+result["report"],latest_historical_report=prefix+result["report"],
                latest_continuation_classification=result["classification"],current_driver_continuation_classification=result["classification"],
                current_driver_consecutive_blocked_goal_turns=0,latest_historical_diagnostic_at=stamp,
                latest_goal_service_status="active",latest_goal_service_status_observed_at=stamp,goal_status="active",goal_achieved=False,
                local_goal_work_status="ACTIVE_HISTORICAL_ONLY",
                last_research_result="主线改为指数整体。固定六个2018—2019降准公告，20日两正四负；同为PMI新订单49.7的2019年1月和9月收益相反。完成银行资金用途、信用结构、政策铺垫和整体覆盖面核对，尚无达标完整账户。",
                last_source_result="14项有效官方文件或官方转载加既有宏观发布、指数行情和成分聚合数据；失效正文保留缺失，2019年5月金额按央行季度报告约3000亿元表述为事后机制证据。",
                next_research_question=result["next_historical_question"])
        receipt["changes"].append({"path":str(path),"fields":{k:{"before":before.get(k),"after":v} for k,v in cfg.items() if before.get(k)!=v}})
        path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    save("authority_update.json",receipt)
    print("已登记指数整体机制研究；保留夏普1.2未达成状态。",flush=True)


def main():
    parser=argparse.ArgumentParser(description="历史指数流动性传导")
    parser.add_argument("mode",choices=["prepare","fetch","analyze","finalize","record-progress"])
    args=parser.parse_args()
    {"prepare":prepare,"fetch":fetch,"analyze":analyze,"finalize":finalize,"record-progress":record_progress}[args.mode]()


if __name__=="__main__":main()
