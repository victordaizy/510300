"""指数信用定价的历史分解；解释既有因子，不改写旧策略。"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from historical_price_gap_causes_v1 import round_trip
from historical_index_liquidity_transmission_v1 import clean

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/research/510300_historical_index_credit_price_decomposition_v1"
PREVIOUS=ROOT/"reports/research/510300_historical_index_liquidity_transmission_v1"
MARKET=ROOT/"reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet"
CURVE=ROOT/"data/raw/macro/china_credit_spread_3y_daily.parquet"
FUNDING=ROOT/"data/raw/macro/510300_macro_stress_2015_v2/fdr007_daily_2015_2026.parquet"
STUDY="510300_HISTORICAL_INDEX_CREDIT_PRICE_DECOMPOSITION_V1"
GROUPS={
    "COST_DOWN_SPREAD_DOWN":"票据利率降、利差收窄",
    "COST_DOWN_SPREAD_NOT_DOWN":"票据利率降、利差未收窄",
    "COST_NOT_DOWN_SPREAD_DOWN":"票据利率未降、利差收窄",
    "COST_NOT_DOWN_SPREAD_NOT_DOWN":"票据利率未降、利差未收窄",
}
SOURCES=[
    {"id":"cfets_fixing","title":"回购定盘利率编制方案","url":"https://www.chinamoney.com.cn/chinese/bkfrr/","required_text":"FDR007"},
    {"id":"chinabond_curve","title":"国债及其他债券收益率曲线简介","url":"https://yield.chinabond.com.cn/cbweb-pbc-web/pbc/more?locale=cn_zh","required_text":"中短期票据"},
]


def now():return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(name,value):
    file=OUT/name;file.parent.mkdir(parents=True,exist_ok=True)
    file.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def prepare():
    if (OUT/"protocol.json").exists():raise RuntimeError("已有设定，不覆盖。")
    old=json.loads((PREVIOUS/"protocol.json").read_text(encoding="utf-8"))
    protocol={"study_id":STUDY,"recorded_at":now(),"previous_goal_turn":"PROGRESS_COMPLETED_SIX_INDEX_EVENTS",
      "mode":"HISTORICAL_DISCOVERY_ONLY","research_unit":"指数整体信用定价与货币资金条件",
      "question":"同样的AAA信用利差收窄，票据自身收益率下降与国债收益率上升分别贡献多少；分解后能否稳定区分历史指数收益？",
      "mechanical_identity":"信用利差变化(bp)=100×AAA票据3年收益率变化(百分点)−100×国债3年收益率变化(百分点)",
      "interpretation_limit":"这是利差的代数分解，不是外生因果识别。AAA二级曲线是高等级市场定价代理，不能代替所有企业贷款成本或低等级信用风险。",
      "curve_lookback_observations":63,
      "lookback_origin":"沿用旧MACRO-02的63个曲线观测日，不修改窗口、债券期限或评级寻找更好收益。",
      "old_strategy_status":"REJECT_DISCOVERY_STOP_NO_RESCUE",
      "old_strategy_report":"reports/backtest/macro_02_credit_spread_trend_v1.md",
      "old_strategy_disposition":"保留原账户和失败；本轮不新增旧策略的趋势过滤、暴露或进出场版本。",
      "groups":GROUPS,"group_threshold_bp":0.,"numerical_zero_tolerance_bp":1e-8,
      "rrr_dates":[e["date"] for e in old["events"]],
      "rrr_role":"六个已知收益节点仅用于解释，不作为新验证样本。",
      "monthly_scope":"2018年1月至2019年12月每月最后一个510300交易日，共24个节点，完整保留。",
      "cutoff":"决策日期00:00以前发布的信息；曲线日期与FDR日期必须早于决策日期。统一保守滞后，不按收益挑当日数据。",
      "entry":"月末节点之后下一实际交易日开盘；固定20个交易日后开盘结束。",
      "horizon_sessions":20,"costs":old["costs"],
      "funding_context":"FDR007最近20个有效观测均值减此前20个有效观测均值，只作资金价格背景，不加到收益分组中筛选。",
      "macro_context":"公告前或月末日前已公布的PMI新订单、M1/M2与社融结构；不拟合阈值，不假称市场共识。",
      "prespecified_candidate_to_inspect":"COST_DOWN_SPREAD_DOWN",
      "candidate_rationale":"若绝对票据利率与相对国债的溢价同时下降，才同时具备较低名义定价和较低相对溢价；仍需检验对指数是否有用。",
      "discovery_stop":"候选组若没有至少6个节点、整体均值及中位数均为正、2018和2019两年各自均值为正，则不继续把它推进为本轮交易条件。无论结果如何，不切换到事后最佳组，不反号、不改窗口。满足也只是继续研究资格。",
      "known_history":"既有策略和多轮研究已经见过这些历史市场数据，本轮不是独立样本外验证。20日月末窗口可能少量重叠，2019年末节点允许跨到2020年，不因之后疫情剔除。",
      "new_parameters_fitted":0,"new_full_accounts":0,"new_prospective_tasks":0,"goal_achieved":False,"orders_authorized":False}
    save("protocol.json",protocol)
    path=ROOT/"config/510300_historical_cause_discovery_v1.json"
    cfg=json.loads(path.read_text(encoding="utf-8"))
    cfg.update(current_study="reports/research/510300_historical_index_credit_price_decomposition_v1/protocol.json",updated_at=now())
    path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("已固定信用定价分解：原63观测窗口、24个月末、四种构成及单一候选的停止条件。",flush=True)


def source_notes():
    notes=[]
    for source in SOURCES:
        item=dict(source);file=OUT/"sources"/(source["id"]+".html")
        file.parent.mkdir(parents=True,exist_ok=True)
        try:
            response=requests.get(source["url"],timeout=(10,25),headers={"User-Agent":"Mozilla/5.0"})
            response.raise_for_status();body=BeautifulSoup(response.content,"html.parser").get_text("\n",strip=True)
            if source["required_text"] not in body:raise ValueError("正文缺少定义关键字")
            file.write_bytes(response.content);file.with_suffix(".txt").write_text(body,encoding="utf-8")
            item.update(status="RETRIEVED",sha256=hashlib.sha256(response.content).hexdigest(),retrieved_at=now(),path=str(file.relative_to(ROOT)))
        except Exception as exc:
            item.update(status="LOCAL_FETCH_MISSING_WEB_READ_AVAILABLE",error=str(exc),note="工具浏览已读到官方定义；不重试批量历史下载，复用已有本地历史数据。")
        notes.append(item)
        print("定义来源："+source["title"]+"；"+item["status"],flush=True)
    save("definition_sources.json",notes)


def group_for(credit_delta,spread_delta):
    return ("COST_DOWN" if credit_delta<-1e-8 else "COST_NOT_DOWN")+("_SPREAD_DOWN" if spread_delta<-1e-8 else "_SPREAD_NOT_DOWN")


def describe(rows):
    if not rows:return {"n":0,"mean_net_pct":None,"median_net_pct":None,"positive_count":0,"worst_pct":None,"best_pct":None}
    x=np.array([r["net_return"] for r in rows])
    return {"n":len(rows),"mean_net_pct":float(x.mean()*100),"median_net_pct":float(np.median(x)*100),
            "positive_count":int((x>0).sum()),"worst_pct":float(x.min()*100),"best_pct":float(x.max()*100)}


def macro_context(nodes):
    base=ROOT/"reports/research/510300_macro_dynamic_reframe_v1/inputs"
    inputs=[("money",pd.read_csv(base/"money_104.csv"),"available_at_upper_bound","stat_month",{"m1_yoy_pct":"m1_yoy_pp","m2_yoy_pct":"m2_yoy_pp"}),
            ("pmi",pd.read_parquet(base/"pmi_new_orders.parquet"),"available_at","reference_period",{"pmi_new_orders":"first_release_value"}),
            ("tsf",pd.read_parquet(ROOT/"data/raw/macro/510300_macro_stress_2015_v2/tsf_stock_yoy_release_vintage_2015_2026.parquet"),"available_at","reference_period",{"tsf_yoy_pct":"first_release_value"})]
    result=[]
    for node in nodes:
        row={"event_id":node["id"],"date":node["date"]}
        cutoff=pd.Timestamp(node["date"],tz="Asia/Shanghai")
        for prefix,frame,time_column,month_column,fields in inputs:
            usable=frame[pd.to_datetime(frame[time_column],utc=True)<cutoff].sort_values(time_column)
            if usable.empty:
                row.update({prefix+"_month":None,prefix+"_status":"MISSING_BEFORE_CUTOFF"})
                row.update({target:None for target in fields})
                continue
            source=usable.iloc[-1]
            row.update({prefix+"_month":source[month_column],prefix+"_status":"AVAILABLE",
                        prefix+"_available_at":source[time_column],prefix+"_source_url":source.source_url})
            row.update({target:source[column] for target,column in fields.items()})
        result.append(row)
    return result


def analyze():
    if (OUT/"result.json").exists():raise RuntimeError("已有结果，不覆盖或重分组。")
    protocol=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    market=pd.read_parquet(MARKET).sort_values("date").reset_index(drop=True)
    market["date"]=pd.to_datetime(market.date).dt.tz_localize(None).dt.normalize()
    dates=pd.DatetimeIndex(market.date)
    curve=pd.read_parquet(CURVE).sort_values("date").reset_index(drop=True)
    curve["date"]=pd.to_datetime(curve.date).dt.normalize()
    funding=pd.read_parquet(FUNDING).sort_values("date").reset_index(drop=True)
    funding["date"]=pd.to_datetime(funding.date).dt.normalize()
    funding["available_time"]=pd.to_datetime(funding.available_at,utc=True)
    dividend=pd.read_csv(ROOT/"data/reference/510300_dividends.csv")
    dividend=dividend[dividend.symbol.eq("510300.SH")].copy();dividend["record_date"]=pd.to_datetime(dividend.record_date)
    monthends=market[market.date.between("2018-01-01","2019-12-31")].groupby(market.date.dt.to_period("M")).date.max()
    nodes=[{"id":"MONTH_"+str(month),"date":day.strftime("%Y-%m-%d"),"kind":"MONTH_END"} for month,day in monthends.items()]
    nodes += [{"id":"RRR_"+day.replace("-",""),"date":day,"kind":"RRR_EXPLANATION"} for day in protocol["rrr_dates"]]
    previous_returns=json.loads((PREVIOUS/"event_returns_and_breadth.json").read_text(encoding="utf-8"))
    old_returns={r["announcement_date"]:r for r in previous_returns if r["horizon"]==20}
    rows=[]
    for node in nodes:
        day=pd.Timestamp(node["date"]);cutoff=pd.Timestamp(node["date"],tz="Asia/Shanghai")
        usable=curve[curve.date<day]
        if len(usable)<=63:raise ValueError("曲线历史不足："+node["id"])
        latest=usable.iloc[-1];earlier=usable.iloc[-64]
        credit_delta=float((latest.cpnote_aaa_3y-earlier.cpnote_aaa_3y)*100)
        govt_delta=float((latest.cgb_3y-earlier.cgb_3y)*100)
        spread_delta=float(latest.credit_spread_3y_bp-earlier.credit_spread_3y_bp)
        f=funding[(funding.date<day)&(funding.available_time<cutoff)].tail(40)
        if len(f)!=40:raise ValueError("资金价格历史不足："+node["id"])
        row={**node,"year":day.year,"group":group_for(credit_delta,spread_delta),"cutoff":cutoff,
             "curve_from":earlier.date,"curve_to":latest.date,"curve_observation_intervals":63,
             "govt_yield_from_pct":float(earlier.cgb_3y),"govt_yield_to_pct":float(latest.cgb_3y),
             "credit_yield_from_pct":float(earlier.cpnote_aaa_3y),"credit_yield_to_pct":float(latest.cpnote_aaa_3y),
             "spread_from_bp":float(earlier.credit_spread_3y_bp),"spread_to_bp":float(latest.credit_spread_3y_bp),
             "credit_yield_delta_bp":credit_delta,"govt_yield_delta_bp":govt_delta,"spread_delta_bp":spread_delta,
             "spread_credit_leg_bp":credit_delta,"spread_govt_leg_bp":-govt_delta,
             "identity_error_bp":credit_delta-govt_delta-spread_delta,
             "fdr_prior20_mean_pct":float(f.iloc[:20].first_release_value.mean()),"fdr_recent20_mean_pct":float(f.iloc[20:].first_release_value.mean()),
             "fdr20mean_change_bp":float((f.iloc[20:].first_release_value.mean()-f.iloc[:20].first_release_value.mean())*100),
             "fdr_from":f.iloc[0].date,"fdr_to":f.iloc[-1].date}
        if node["kind"]=="RRR_EXPLANATION":
            old=old_returns[node["date"]]
            row.update({k:old[k] for k in ["entry_date","exit_date","entry_open","exit_open","net_return","net_pnl_cny","buy_price","sell_price","shares","commissions_cny","dividend_entitlement_cny","paid_cny"]})
        else:
            e=dates.searchsorted(day,side="right");x=e+protocol["horizon_sessions"]
            entry=market.iloc[e];end=market.iloc[x]
            div=float(dividend.loc[(dividend.record_date>=entry.date)&(dividend.record_date<end.date),"cash_dividend_per_share"].sum())
            trade=round_trip(float(entry.open),float(end.open),div,protocol["costs"])
            row.update(entry_date=entry.date,exit_date=end.date,entry_open=float(entry.open),exit_open=float(end.open),dividend_per_share=div,
                       gross_return=float((end.open+div)/entry.open-1),**trade)
        rows.append(row)
    states=macro_context(nodes)
    lookup={x["event_id"]:x for x in states}
    for row in rows:
        context=lookup[row["id"]]
        row.update({k:context[k] for k in ["pmi_new_orders","pmi_month","money_month","m1_yoy_pct","m2_yoy_pct","tsf_month","tsf_yoy_pct"]})
    monthly=[x for x in rows if x["kind"]=="MONTH_END"]
    assert len(monthly)==24 and len(rows)==30
    assert max(abs(x["identity_error_bp"]) for x in rows)<1e-8
    assert all(pd.Timestamp(x["entry_date"])>pd.Timestamp(x["date"]) for x in rows)
    assert all(pd.Timestamp(x["curve_to"])<pd.Timestamp(x["date"]) for x in rows)
    pnl_errors=[abs((x["sell_price"]-x["buy_price"])*x["shares"]-x["commissions_cny"]+x["dividend_entitlement_cny"]-x["net_pnl_cny"]) for x in rows]
    assert max(pnl_errors)<.01
    summary=[]
    for year in [2018,2019,"ALL"]:
        for group,label in GROUPS.items():
            subset=[r for r in monthly if (year=="ALL" or r["year"]==year) and r["group"]==group]
            summary.append({"year":year,"group":group,"label":label,**describe(subset)})
    candidate=protocol["prespecified_candidate_to_inspect"]
    c=next(x for x in summary if x["year"]=="ALL" and x["group"]==candidate)
    years=[next(x for x in summary if x["year"]==y and x["group"]==candidate) for y in [2018,2019]]
    pass_conditions={"at_least_six":c["n"]>=6,"mean_positive":c["mean_net_pct"] is not None and c["mean_net_pct"]>0,
                     "median_positive":c["median_net_pct"] is not None and c["median_net_pct"]>0,
                     "both_calendar_year_means_positive":all(x["n"]>0 and x["mean_net_pct"]>0 for x in years)}
    proceed=all(pass_conditions.values())
    checks={"rows":30,"monthly_nodes":24,"rrr_explanation_nodes":6,"max_identity_error_bp":max(abs(x["identity_error_bp"]) for x in rows),
            "max_pnl_identity_error_cny":max(pnl_errors),"all_curve_dates_before_cutoff":True,"all_entries_after_decision_day":True,
            "lookback_unchanged_from_old_definition":63,"source_status":"EXISTING_OFFICIAL_SERIES_DISCOVERY_ONLY_NO_INDEPENDENT_SUPPLIER_VALIDATION"}
    result={"study_id":STUDY,"completed_at":now(),"status":"HISTORICAL_INDEX_CREDIT_PRICE_DECOMPOSITION_COMPLETED",
            "classification":"PROGRESS_CREDIT_SPREAD_COMPONENTS_AND_CALENDAR_MONTH_EVIDENCE",
            "candidate_decision":"CONTINUE_RESEARCH_ONLY" if proceed else "STOP_CANDIDATE_NO_PARAMETER_RESCUE",
            "candidate_conditions":pass_conditions,"candidate_summary":c,"calendar_year_candidate_summaries":years,
            "all_months":describe(monthly),"summary":summary,"new_full_accounts":0,"new_parameters_fitted":0,"new_prospective_tasks":0,
            "net_sharpe":None,"goal_achieved":False,"orders_authorized":False,"independent_validation":False}
    save("node_states_and_returns.json",rows);save("known_macro_context.json",states);save("group_summary.json",summary)
    save("calculation_checks.json",checks);save("result.json",result)
    print(json.dumps(clean({"结果":result,"六次降准的事前定价":[{k:x[k] for k in ["date","group","credit_yield_delta_bp","govt_yield_delta_bp","spread_delta_bp","fdr20mean_change_bp","net_return"]} for x in rows if x["kind"]=="RRR_EXPLANATION"]}),ensure_ascii=False,indent=2),flush=True)


def plot(rows,result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family":font.get_name(),"axes.unicode_minus":False,"font.size":11})
    events=[x for x in rows if x["kind"]=="RRR_EXPLANATION"]
    fig,axes=plt.subplots(1,2,figsize=(15.2,6.6),gridspec_kw={"width_ratios":[1.45,1]})
    bg="#f5f3ee";fig.patch.set_facecolor(bg);y=np.arange(len(events))
    credit=np.array([x["credit_yield_delta_bp"] for x in events]);govt=np.array([-x["govt_yield_delta_bp"] for x in events])
    axes[0].barh(y,credit,height=.48,color="#327d85",label="票据收益率变化")
    left=np.where(np.sign(credit)==np.sign(govt),credit,0)
    axes[0].barh(y,govt,left=left,height=.48,color="#d8ad71",label="国债收益率变化的反向贡献")
    net=credit+govt;axes[0].scatter(net,y,color="#283641",marker="D",s=42,zorder=4,label="合计：信用利差变化")
    for i,value in enumerate(net):
        axes[0].text(value,i-.32,f"{value:+.2f} bp",ha="center",va="center",fontsize=9,color="#283641")
    axes[0].set_yticks(y,[x["date"] for x in events]);axes[0].invert_yaxis();axes[0].axvline(0,color="#a9a69c",lw=1)
    axes[0].set(xlim=(-80,70),xlabel="公告前63个曲线观测日的变化（基点）",title="利差必须拆成票据与国债两部分")
    axes[0].legend(loc="upper left",bbox_to_anchor=(0,-.20),fontsize=9,frameon=False,ncol=2)
    groups=result["calendar_year_candidate_summaries"]+[result["candidate_summary"]]
    x=np.arange(3);v=[r["mean_net_pct"] for r in groups]
    axes[1].bar(x,v,width=.55,color=["#aab8b4","#7ba19c","#327d85"])
    axes[1].axhline(0,color="#a9a69c",lw=1)
    for i,row in enumerate(groups):
        axes[1].text(i,row["mean_net_pct"]-.12,f"{row['mean_net_pct']:+.2f}%\nn={row['n']}",ha="center",va="top",fontsize=12)
    axes[1].set_xticks(x,["2018年","2019年","两年合计"])
    axes[1].set(ylim=(min(v)-.95,1.1),ylabel="随后20个交易日净收益均值（%）",title="预先指定的“票据利率与利差双降”组")
    axes[1].text(.5,.93,"13次中3次盈利；中位数 −0.59%",transform=axes[1].transAxes,ha="center",va="top",fontsize=11)
    for ax in axes:
        ax.set_facecolor(bg);ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#c6c0b3");ax.set_axisbelow(True)
    axes[1].grid(axis="y",alpha=.15)
    fig.suptitle("510300历史发现｜高等级债券定价改善，仍不足以形成指数买点",fontsize=18,x=.05,ha="left")
    fig.text(.05,.015,"右图完整保留2018—2019年的24个月末，再按事前定价构成分组。下一交易日开盘观察，含压力费用；非完整账户。",fontsize=10,color="#69665e")
    fig.tight_layout(rect=(0,.085,1,.92));fig.savefig(OUT/"利差构成与指数收益.png",dpi=165,facecolor=bg);plt.close(fig)


def finish_report():
    result=json.loads((OUT/"result.json").read_text(encoding="utf-8"))
    rows=json.loads((OUT/"node_states_and_returns.json").read_text(encoding="utf-8"))
    candidate=[x for x in rows if x["kind"]=="MONTH_END" and x["group"]=="COST_DOWN_SPREAD_DOWN"]
    result["candidate_gross_mean_pct"]=float(np.mean([x["gross_return"] for x in candidate])*100)
    result["candidate_average_cost_drag_pp"]=result["candidate_gross_mean_pct"]-result["candidate_summary"]["mean_net_pct"]
    result["next_historical_question"]="停止把高等级债券价格改善直接变成指数买点。围绕2018年第四季度至2019年第一季度，追查民企信用风险分担、银行资本约束与有效融资需求的实际变化，按原公告和首次披露时点研究整体指数重定价；不扩展个股，不恢复旧利差策略。"
    q4="https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/3537682/f8d2bf5c14f940bcb7c8bf81df786d89/2019022119340790088.pdf"
    q1="https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/3830536/e0d70bbde6d34378bde1fcc3713126ec/2019051720074058265.pdf"
    evidence=[
      {"source_url":q4,"source_path":str((PREVIOUS/"sources/pbc_2018q4.pdf").relative_to(ROOT)),"published_date":"2019-02-21","pdf_page":27,
       "finding":"央行将部分民企违约、机构风险偏好下降、融资困难和股权质押风险互相强化列为背景；10月22日的支持工具通过风险缓释和增信改善风险承担安排。",
       "role":"事后机制说明，不是2018年月末可用输入；不能量化为指数跌幅的因果贡献。"},
      {"source_url":q4,"source_path":str((PREVIOUS/"sources/pbc_2018q4.pdf").relative_to(ROOT)),"published_date":"2019-02-21","pdf_page":61,
       "finding":"报告同时指出有效融资需求下降和金融机构风险偏好下降；低利率可能与需求偏弱并存。","role":"经济解释依据，未用于分组或筛选收益。"},
      {"source_url":q1,"source_path":str((PREVIOUS/"sources/pbc_2019q1.pdf").relative_to(ROOT)),"published_date":"2019-05-17","pdf_page":3,
       "finding":"报告将缓解流动性、资本和利率约束以及部门政策协调共同列为传导机制。","role":"后续机制总结，不能倒填2019年1月的已知状态。"},
    ]
    save("upstream_constraint_evidence.json",evidence)
    events=[x for x in rows if x["kind"]=="RRR_EXPLANATION"]
    lines=["# 510300历史发现：信用利差变动的原因与指数收益","",
      "本轮继续按指数整体研究。结论是：把信用利差拆成票据收益率和国债收益率，能够避免方向误读；但‘票据收益率下降且利差收窄’仍没有在本轮历史阶段形成可用买入条件。该候选停止，不改窗口，也不改选表现最好的组。","",
      "**先把因子拆开**","",
      "信用利差变化＝AAA中短期票据3年收益率变化－国债3年收益率变化，所有变化统一换算为基点。1个基点等于0.01个百分点。",
      "中债AAA票据曲线包含高等级超短融、短融和中票的市场定价信息。它不是所有企业实际贷款成本，也没有覆盖低评级与民营企业的全部融资约束。[中债曲线简介](https://yield.chinabond.com.cn/cbweb-pbc-web/pbc/more?locale=cn_zh)。FDR007则是银银间上午交易形成的定盘参考利率，不等于全天加权DR007。[中国货币网编制方案](https://www.chinamoney.com.cn/chinese/bkfrr/)。","",
      "| 降准公告日 | 事前票据收益率变化 | 国债收益率变化 | 利差变化 | 随后20日指数净收益 |",
      "|---|---:|---:|---:|---:|"]
    for r in events:
        lines.append(f"| {r['date']} | {r['credit_yield_delta_bp']:+.2f} bp | {r['govt_yield_delta_bp']:+.2f} bp | {r['spread_delta_bp']:+.2f} bp | {r['net_return']*100:+.2f}% |")
    lines += ["",
      "2019年1月4日之前，票据收益率下降43.63个基点，国债下降50.27个基点，因此利差扩大6.64个基点。这个利差扩大不能解释成票据融资定价上涨。随后指数上涨5.92%，也不能反过来证明所有利差扩大都是利好。",
      "2019年6月末则出现另一种构成：票据收益率上升3.91个基点、国债上升22.37个基点，利差因此收窄18.46个基点。利差改善不代表票据本身变便宜；这一组只有一个月末节点，没有资格概括成规律。",
      "六次降准收益已在上一轮见过，只用于解释。新的月末对照覆盖2018—2019年全部24个节点，按收益率和利差的事前方向形成四组。","",
      "**完整月末对照**","",
      "| 事前定价构成 | 节点数 | 盈利节点 | 20日净收益均值 | 中位数 | 最差 |",
      "|---|---:|---:|---:|---:|---:|"]
    for s in result["summary"]:
        if s["year"]!="ALL":continue
        lines.append(f"| {s['label']} | {s['n']} | {s['positive_count']} | {s['mean_net_pct']:+.2f}% | {s['median_net_pct']:+.2f}% | {s['worst_pct']:+.2f}% |")
    c=result["candidate_summary"]
    lines += ["",
      f"事前指定的候选是票据利率与利差同时下降。这组共{c['n']}个节点，只有{c['positive_count']}次盈利；净收益均值{c['mean_net_pct']:+.2f}%，中位数{c['median_net_pct']:+.2f}%。2018年8次均值−2.25%，2019年5次均值−1.33%，两年均未支持继续。",
      f"该组成本前均值也为{result['candidate_gross_mean_pct']:+.2f}%，平均费用影响约{result['candidate_average_cost_drag_pp']:.2f}个百分点。问题主要是观察到的后续价格收益不足，不能靠降低佣金把它解释成已有优势。",
      "票据利率下降但利差未收窄的组，合计均值虽为+2.38%，但只有5次，2018年均值−2.35%，2019年均值+9.48%；它不是事前指定的候选，也未表现出相同的阶段方向。本轮不将它提升为反向规则。",
      "2019年末节点的持有期跨到2020年2月，包含疫情冲击，按原范围保留。相邻月末的20日窗口可能少量重叠，不能把24次当成完全独立试验。","",
      "**上游原因不能停在债券价格这一层**","",
      "利率下降同时可能反映资金供给增加、需求走弱、对安全资产的偏好变化、风险和流动性补偿变化。仅凭这两条曲线，不能把它们逐项识别出来。我们确认的是价格变动构成，不是这些冲击的精确因果比例。",
      "央行2018年四季度报告指出，部分民企违约后，金融机构风险偏好下降，融资困难与股权质押风险可能互相强化；报告也指出有效融资需求下降。这说明高等级资产定价改善与其他主体融资困难可以并存。10月22日决定设立的民营企业债券融资支持工具，着力改变的是风险分担与增信安排。[央行2018年四季度报告，PDF第27、61页]("+q4+")。",
      "2019年一季度报告把流动性、资本和利率三类约束并列讨论。便宜资金只是其中一项；银行有资本承接风险、企业有有效融资需求，也需要证据。[央行2019年一季度报告，PDF第3页]("+q1+")。这些季度报告晚于有关历史交易节点，本轮仅用作复盘解释，未倒填为当时信号。",
      "接下来的问题因此收窄为：在2018年第四季度至2019年第一季度，哪些公开事件改变了风险承担、银行资本与有效需求，最早何时可知，之后整体指数还有多少可实现收益。主线维持在指数，成分公司不再单独扩展。","",
      "**口径与实用性**","",
      "保留旧MACRO-02的63个曲线观测日、同一3年期限和AAA口径。旧策略仍为REJECT_DISCOVERY_STOP_NO_RESCUE，没有改趋势、仓位或退出去抢救旧账户。当前四组只解释因子构成。",
      "所有输入取节点日期00:00以前，曲线与FDR日期早于节点；每月最后一个510300交易日观察，下一交易日开盘进入，20个交易日后开盘结束。每次10万元预算，沿用单边0.1%滑点、0.04%佣金且最低5元，计入分红、价格档位及整数份额。收益分母为实际投入资金，没有新增完整账户。",
      "既有历史数据已被多轮研究见过，不声称独立验证。债券数据复用已有官方序列，其原状态是历史发现用途、尚无独立供应商验证。2018年1月节点的M1/M2在当前输入表中缺失，保留缺失且不影响本轮利率分组。只核对了利差恒等式、信息时点和交易现金流，没有参数搜索。",
      "目标仍是20万元完整账户成本后夏普至少1.2；当前候选未通过继续条件，未计算完整账户夏普，目标未达成。","",
      "| 月末节点 | 定价构成 | 20日净收益 |",
      "|---|---|---:|"]
    for r in rows:
        if r["kind"]=="MONTH_END":lines.append(f"| {r['date']} | {GROUPS[r['group']]} | {r['net_return']*100:+.2f}% |")
    lines += ["","完整输入与计算："+"、".join("["+label+"](<"+(OUT/name).as_posix()+">)" for name,label in [("protocol.json","事前设定"),("node_states_and_returns.json","30节点分解"),("known_macro_context.json","已公布宏观状态"),("upstream_constraint_evidence.json","上游原因证据")])+"。"]
    name="历史发现_信用利差构成与指数收益.md"
    (OUT/name).write_text("\n".join(lines)+"\n",encoding="utf-8")
    result.update(report=name,report_completed_at=now());save("result.json",result);plot(rows,result)
    print(json.dumps({"报告":name,"候选成本前均值":result["candidate_gross_mean_pct"],"候选成本后均值":c["mean_net_pct"],"下一步":result["next_historical_question"]},ensure_ascii=False,indent=2),flush=True)


def record_progress():
    result=json.loads((OUT/"result.json").read_text(encoding="utf-8"))
    assert (OUT/result["report"]).is_file()
    prefix="reports/research/510300_historical_index_credit_price_decomposition_v1/";stamp=now()
    receipt={"recorded_at":stamp,"study_id":STUDY,"previous_goal_turn_classification":"PROGRESS",
             "current_goal_turn_classification":"PROGRESS_NEW_CAUSE_DECOMPOSITION_AND_STOPPED_CANDIDATE",
             "result":prefix+"result.json","changes":[],"goal_achieved":False,"orders_authorized":False}
    for name in ["510300_historical_cause_discovery_v1.json","510300_existing_data_training_mandate_v1.json"]:
        path=ROOT/"config"/name;cfg=json.loads(path.read_text(encoding="utf-8"));before=dict(cfg)
        if name=="510300_historical_cause_discovery_v1.json":
            cfg.update(latest_completed_study=prefix+"result.json",latest_report=prefix+result["report"],updated_at=stamp)
        else:
            cfg.update(current_round=STUDY,latest_progress_receipt=prefix+"result.json",latest_historical_index_credit_price_decomposition=prefix+"result.json",
                latest_continuation_report=prefix+result["report"],latest_historical_report=prefix+result["report"],
                latest_continuation_classification=result["classification"],current_driver_continuation_classification=result["classification"],
                current_driver_consecutive_blocked_goal_turns=0,latest_historical_diagnostic_at=stamp,
                latest_goal_service_status="active",latest_goal_service_status_observed_at=stamp,goal_status="active",goal_achieved=False,
                local_goal_work_status="ACTIVE_HISTORICAL_ONLY",
                last_research_result="分解AAA票据与国债利率。24个月末中事前指定的票据利率和利差双降组13次，仅3次盈利，20日均值−1.90%，两年分别均为负，候选停止；未改旧MACRO-02或新增完整账户。",
                last_source_result="复用官方中债曲线和FDR007历史，读取两项官方定义；央行季度报告支持风险偏好、资本约束与有效需求是进一步需解释的上游条件。",
                next_research_question=result["next_historical_question"])
        receipt["changes"].append({"path":str(path),"fields":{k:{"before":before.get(k),"after":v} for k,v in cfg.items() if before.get(k)!=v}})
        path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    save("authority_update.json",receipt)
    print("已登记指数信用价格分解及候选停止结论；完整账户夏普目标保持未达成。",flush=True)


def main():
    parser=argparse.ArgumentParser(description="指数信用定价历史分解")
    parser.add_argument("mode",choices=["prepare","sources","analyze","report","record-progress"])
    args=parser.parse_args();{"prepare":prepare,"sources":source_notes,"analyze":analyze,"report":finish_report,"record-progress":record_progress}[args.mode]()


if __name__=="__main__":main()
