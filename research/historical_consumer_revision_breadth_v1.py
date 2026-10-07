"""同机构四家消费权重历史盈利修正及其指数影响范围。"""
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
OUT=ROOT/"reports/research/510300_historical_consumer_revision_breadth_v1"
PREV=ROOT/"reports/research/510300_historical_disclosure_increment_v1"
STUDY="510300_HISTORICAL_CONSUMER_REVISION_BREADTH_V1"
EARNINGS=ROOT/"reports/research/510300_historical_consumer_earnings_transmission_v1"
LABELS={"600519.SH":"贵州茅台","000858.SZ":"五粮液","002304.SZ":"洋河股份","600887.SH":"伊利股份"}
MODEL_FIELDS={"revenue":"营业收入","cost":"营业成本","business_tax":"营业税金及附加",
              "selling":"营业费用","management":"管理费用","finance":"财务费用",
              "parent_profit":"归属母公司净利润","eps":"EPS (元，基本)"}


def now():return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,np.generic):return clean(value.item())
    if isinstance(value,(datetime,pd.Timestamp)):return value.isoformat()
    if value is pd.NaT or value is pd.NA or isinstance(value,float) and not np.isfinite(value):return None
    return value


def save(name,value):
    path=OUT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def prepare():
    if (OUT/"protocol.json").exists():raise RuntimeError("已有设定，不覆盖。")
    parent=json.loads((PREV/"protocol.json").read_text(encoding="utf-8"))
    weights=json.loads((ROOT/"reports/research/510300_historical_consumer_earnings_transmission_v1/复用四公司事前观察窗.json").read_text(encoding="utf-8"))
    save("protocol.json",{
        "study_id":STUDY,"recorded_at":now(),"research_mode":"HISTORICAL_ONLY",
        "previous_goal_turn_classification":"PROGRESS_HISTORICAL_COMPARABLE_GUIDANCE_AND_PAIRED_ANALYST_REVISIONS",
        "question":"2018半年至三季报，华泰对四家消费权重的盈利假设是否共同下调，上游解释及指数影响范围如何？",
        "fixed_companies":[{"symbol":r["symbol"],"name":r["name"],"weight":r["weight"]} for r in weights["rows"]],
        "fixed_institution":"华泰证券","fixed_document_types":["2018半年报点评","2018三季报点评"],
        "source_preference":"同机构官方网站原PDF优先；仅缺原站时允许该机构原文的公开镜像，保留缺失，不换一家结论更强的机构。",
        "document_selection":"每个公司、报告期取直接对应正式财报的同机构点评；若存在多份，记录全目录并选最早可核实日期。资料不全保留缺项。",
        "known_results":"茅台两份报告及其盈利下调已见；四家公司此前下跌与多组ETF后续收益已见。不是独立样本。",
        "forecast_years":[2018,2019,2020],
        "comparison":"同机构、同财年EPS、归母利润、模型收入、利润率及费用率；目标PE只在相同盈利年度时比较，不混用滚年估值。",
        "decomposition":"对利润=收入×利润率作对称恒等拆分，保留全部公司与年度，不按方向或幅度挑选。",
        "weight_source":"复用已有2018年9月末权重，仅描述三季度阶段的影响范围；未核实首次发布时间，不用于事前信号。",
        "constant_pe_sensitivity":"分别计算各财年sum(weight_i*(EPS_new/EPS_old-1))；假设PE、股本口径与其他成分价格不变且所覆盖股票同比例重估。这是机械价格敏感度，非指数盈利增速、实际贡献或可执行策略。",
        "missing_weight_policy":"缺失项报告覆盖权重，不填零，不将已取得部分冒称完整四公司或全指数。",
        "event_clock":"各报告载明日结束后下一实际交易日开盘；最早公开时间未核实，不按正文当前价推定能以该价成交。",
        "horizons_sessions":[5,20],"costs":parent["costs"],
        "traded_assets":["510300.SH","CASH_CNY"],"new_parameters_fitted":0,"new_prospective_tasks":0,
        "new_full_accounts":0,"goal_achieved":False,"orders_authorized":False})
    print("已固定四家公司、华泰原报告、三个预测年度及指数敏感度口径。",flush=True)


def fetch():
    sources=json.loads((OUT/"selected_sources.json").read_text(encoding="utf-8"))
    def one(spec):
        raw=OUT/"raw"/(spec["id"]+".pdf")
        receipt_path=OUT/"receipts"/(spec["id"]+".json")
        if spec.get("reuse"):
            return dict(spec,status="REUSED_LOCAL_PRIMARY_PDF",local_pdf=str(PREV/"raw"/(spec["reuse"]+".pdf")))
        if receipt_path.exists():
            receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
            return dict(spec,status=receipt["status"],receipt=receipt,local_pdf=str(raw) if raw.exists() else None)
        receipt={"id":spec["id"],"url":spec["url"],"retrieved_at":now(),"attempts":1}
        try:
            response=requests.get(spec["url"],headers={"User-Agent":"Mozilla/5.0"},timeout=(10,35))
            receipt.update(http_status=response.status_code,final_url=response.url)
            response.raise_for_status()
            body=response.content
            if not body.startswith(b"%PDF"):
                receipt["status"]="NOT_PDF"
            else:
                raw.parent.mkdir(parents=True,exist_ok=True)
                raw.write_bytes(body)
                with pdfplumber.open(BytesIO(body)) as document:
                    pages=[{"page":i+1,"text":p.extract_text() or ""} for i,p in enumerate(document.pages)]
                save("raw/"+spec["id"]+"_pages.json",pages)
                (OUT/"raw"/(spec["id"]+".txt")).write_text("\n".join(f"第{p['page']}页\n{p['text']}" for p in pages),encoding="utf-8")
                receipt.update(status="PDF_TEXT_ACQUIRED",bytes=len(body),sha256=hashlib.sha256(body).hexdigest(),pages=len(pages))
        except requests.RequestException as exc:
            receipt.update(status="SOURCE_FAILED_NO_RETRY",error=str(exc))
        save("receipts/"+spec["id"]+".json",receipt)
        return dict(spec,status=receipt["status"],receipt=receipt,local_pdf=str(raw) if raw.exists() else None)
    with ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(one,sources))
    save("source_index.json",result)
    print(json.dumps([{k:r.get(k) for k in ["id","status","document_date"]} for r in result],ensure_ascii=False,indent=2),flush=True)


def read_pages(spec):
    folder=PREV if spec.get("reuse") else OUT
    return json.loads((folder/"raw"/(spec["id"]+"_pages.json")).read_text(encoding="utf-8"))


def numeric_row(text,label,count):
    line=next(line for line in text.splitlines() if label in line)
    tail=line.split(label,1)[1]
    tokens=re.findall(r"\(?-?\d[\d,]*(?:\.\d+)?\)?",tail)
    if len(tokens)!=count:
        raise ValueError(f"{label}需要{count}列，实际为{tokens}")
    values=[float(token.replace(",","").replace("(","-").replace(")","")) for token in tokens]
    return values,line


def peer_table(spec):
    text=read_pages(spec)[1]["text"]
    rows=[]
    for line in text.splitlines():
        match=re.match(r"(\d{6}\.(?:SH|SZ))\s+(\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)$",line)
        if match:
            rows.append({"symbol":match[1],"name":match[2],"pe_ttm":float(match[3]),
                         "pe_2018":float(match[4]),"pe_2019":float(match[5]),"pe_2020":float(match[6])})
    values,line=numeric_row(text,"平均值",4)
    assert len(rows)==15 and len({r["symbol"] for r in rows})==15
    computed=float(np.mean([r["pe_2018"] for r in rows]))
    assert abs(computed-values[1])<.011
    return {"source_id":spec["id"],"pdf_page":2,"rows":rows,"reported_average_pe_2018":values[1],
            "computed_average_pe_2018":computed,"average_source_line":line,
            "interpretation":"报告转载的Wind一致预期PE，包含当时股价和盈利分母的共同变化；不是华泰逐公司盈利模型。"}


def analyze_models(cfg,sources):
    values=[];models={}
    for spec in sources:
        pages=read_pages(spec)
        match=re.search(r"(2018)年\s*(\d{2})月\s*(\d{2})日",pages[0]["text"])
        assert match and "-".join(match.groups())==spec["document_date"]
        model={"source_id":spec["id"],"symbol":spec["symbol"],"period":spec["period"],
               "document_date":spec["document_date"],"url":spec["url"],"forecast_years":cfg["forecast_years"]}
        for key,label in MODEL_FIELDS.items():
            numbers,line=numeric_row(pages[2]["text"],label,5)
            model[key]=numbers[2:]
            values.append({"source_id":spec["id"],"pdf_page":3,"row_label":label,
                           "columns":[2016,2017,"2018E","2019E","2020E"],"values":numbers,
                           "unit":"CNY_PER_SHARE" if key=="eps" else "MILLION_CNY","source_line":line})
        for key,label in [("revenue","营业收入 (百万元)"),("parent_profit","归属母公司净利润 (百万元)"),("eps","EPS (元，最新摊薄)")]:
            numbers,_=numeric_row(pages[0]["text"],label,5)
            assert numbers[2:]==model[key]
        models[spec["id"]]=model
    save("模型原值及页码.json",values)
    save("六份历史刊载模型.json",models)
    pe_inputs={"600519.SH":([25,26],[24,25]),"000858.SZ":([22,24],[18,20]),"002304.SZ":([22,24],[18,20])}
    revisions=[];valuations=[];identities=[];peers=[]
    for symbol,(old_pe,new_pe) in pe_inputs.items():
        old=next(m for m in models.values() if m["symbol"]==symbol and m["period"]=="H1")
        new=next(m for m in models.values() if m["symbol"]==symbol and m["period"]=="Q3")
        for i,year in enumerate(cfg["forecast_years"]):
            r0,r1=old["revenue"][i],new["revenue"][i]
            p0,p1=old["parent_profit"][i],new["parent_profit"][i]
            m0,m1=p0/r0,p1/r1
            revenue_part=(r1-r0)*(m0+m1)/2
            margin_part=(m1-m0)*(r0+r1)/2
            identities.append(abs((p1-p0)-revenue_part-margin_part))
            revisions.append({"symbol":symbol,"name":LABELS[symbol],"forecast_year":year,
                "prior_source":old["source_id"],"new_source":new["source_id"],
                "prior_revenue_million_cny":r0,"new_revenue_million_cny":r1,"revenue_revision":r1/r0-1,
                "prior_parent_profit_million_cny":p0,"new_parent_profit_million_cny":p1,
                "parent_profit_revision":p1/p0-1,"prior_eps":old["eps"][i],"new_eps":new["eps"][i],
                "eps_revision":new["eps"][i]/old["eps"][i]-1,
                "prior_parent_margin":m0,"new_parent_margin":m1,"parent_margin_change_pp":(m1-m0)*100,
                "revenue_component_million_cny":revenue_part,"margin_component_million_cny":margin_part,
                "expense_ratio_changes_pp":{key:(new[key][i]/r1-old[key][i]/r0)*100
                    for key in ["cost","business_tax","selling","management","finance"]}})
        e0,e1=old["eps"][0],new["eps"][0]
        pe0,pe1=float(np.mean(old_pe)),float(np.mean(new_pe))
        eps_part=(e1-e0)*(pe0+pe1)/2
        pe_part=(pe1-pe0)*(e0+e1)/2
        identities.append(abs(e1*pe1-e0*pe0-eps_part-pe_part))
        valuations.append({"symbol":symbol,"name":LABELS[symbol],"valuation_forecast_year":2018,
            "prior_source":old["source_id"],"new_source":new["source_id"],"pdf_page":1,
            "prior_target_pe_range":old_pe,"new_target_pe_range":new_pe,
            "prior_target_midpoint_cny":e0*pe0,"new_target_midpoint_cny":e1*pe1,
            "target_midpoint_revision":e1*pe1/(e0*pe0)-1,
            "eps_component_cny":eps_part,"pe_component_cny":pe_part,
            "limit":"仅分解刊载的目标价中点，不是实际股价变化的因果分解。"})
        if symbol!="600519.SH":
            pair=[peer_table(next(s for s in sources if s["id"]==model["source_id"])) for model in [old,new]]
            assert {r["symbol"] for r in pair[0]["rows"]}=={r["symbol"] for r in pair[1]["rows"]}
            peers.append({"symbol":symbol,"same_peer_roster":True,"prior":pair[0],"new":pair[1],
                          "reported_average_pe_2018_revision":pair[1]["reported_average_pe_2018"]/pair[0]["reported_average_pe_2018"]-1})
    conflicts=[
        {"sources":["htsc_wuliangye_h1","htsc_wuliangye_q3"],"pages":[1,3],
         "conflict":"两版开头均写EPS为3.37、4.30、5.31；投资结论及两张预测表均写3.37、4.33、5.34。",
         "handling":"使用两页一致且可与归母利润对应的预测表，并保留正文差异。两版按各自正文比较也都是零修正，因此未变的方向不依赖选哪组。"},
        {"sources":["htsc_yanghe_q3"],"pages":[3],
         "conflict":"同页2018—2020净利润在利润表为8408、10277、12367百万元，在现金流表为8347、10140、12187；财务费用也不一致。",
         "handling":"只称EPS与刊载利润表未变，不称整套模型未变或已完整勾稽。不以现金流表推算替代EPS。"},
        {"sources":["htsc_maotai_h1","htsc_maotai_q3"],"pages":[1,2,3],
         "conflict":"三季点评追述旧2019年EPS为35.40元；原半年点评的预测表为35.50元。",
         "handling":"直接比较两期原报告各自模型，使用原表35.50元，不使用后文追述值。"},
        {"sources":["htsc_wuliangye_q3","company_wuliangye_q3"],"pages":[1,16],
         "conflict":"券商首段将单三季度23.58亿元标为归母净利润；公司原表归母净利润为23.8416亿元。",
         "handling":"本轮不使用该正文数字计算已实现利润，历史实际值仍以公司原表为准。"}]
    save("刊载盈利及目标估值修正.json",{"revisions":revisions,"target_valuations":valuations,
        "source_conflicts":conflicts,"scope":"一家机构的历史刊载预测，不等于全市场共识；两期差异不等于三季报单一消息冲击。",
        "maotai_revenue_scope":"原模型2017年基数61063百万元对应营业总收入，两版同口径互比，不接公司不含金融利息的营业收入。"})
    save("目标估值所参照的同行估值.json",peers)
    covered={r["symbol"] for r in revisions}
    weights={r["symbol"]:r["weight"] for r in cfg["fixed_companies"]}
    sensitivity=[]
    for year in cfg["forecast_years"]:
        parts=[{"symbol":r["symbol"],"name":r["name"],"weight":weights[r["symbol"]],
                "eps_revision":r["eps_revision"],"weighted_sensitivity":weights[r["symbol"]]*r["eps_revision"]}
               for r in revisions if r["forecast_year"]==year]
        sensitivity.append({"forecast_year":year,"covered_parts":parts,
            "covered_subtotal":sum(p["weighted_sensitivity"] for p in parts),"uncovered_component":None})
    coverage={"fixed_company_count":4,"paired_company_count":len(covered),"fixed_weight":sum(weights.values()),
        "covered_weight":sum(weights[s] for s in covered),"missing_weight":sum(weights[s] for s in weights if s not in covered),
        "missing_companies":[s for s in weights if s not in covered],"sensitivity":sensitivity,
        "limit":"仅已覆盖部分在PE不变、EPS同比例传入价格时的机械贡献小计；不填补伊利，不代表全指数盈余增长、实际收益或可交易预测。"}
    save("固定权重覆盖与机械敏感度.json",coverage)
    assert max(identities)<1e-8
    return revisions,valuations,peers,coverage,{"source_value_rows":len(values),"identity_max_abs":max(identities)}


def source_coverage(sources):
    rows=[]
    for symbol in LABELS:
        for period in ["H1","Q3"]:
            matches=[s for s in sources if s["symbol"]==symbol and s["period"]==period]
            rows.append({"symbol":symbol,"name":LABELS[symbol],"period":period,
                         "status":"ORIGINAL_PDF_LOCATED" if matches else "NO_COMPARABLE_SAME_INSTITUTION_REPORT_LOCATED",
                         "selected":matches[0] if matches else None,"located_candidate_count":len(matches),
                         "globally_earliest_publication_verified":False})
    save("八个固定资料位置及缺失.json",{"rows":rows,
        "search_scope":"公司代码、名称、2018中报或三季报、华泰机构及已知标题的公开检索；查原报告相关研究链接。未取得完整券商历史目录。",
        "yanghe_locator":"locators/yanghe_2020_h1.pdf第1页相关研究的链接，定位2018原报告；2020正文未进入2018模型或收益判断。",
        "yili_later_reference":"https://crm.htsc.com.cn/doc/2019/10710101/4205880c-4fdd-4819-b110-bdc755cfacb0.pdf",
        "yili_limit":"公开检索定位到华泰2019年6月26日标注首次覆盖的原报告，但未找到2018所需两期。后来的首次覆盖标签不能证明此前从无报告；保持缺失，不换机构、不以2019模型回填。"})


def upstream_records():
    records=[
        {"company":"贵州茅台","observed":"三年EPS和收入预测均下调，2018目标PE小幅下调。",
         "upstream":"原券商由供需偏紧的判断转向消费环境、需求增速与高基数压力；另称打款由月改季并加快发货。",
         "evidence":"两期华泰报告第1页及第3页；机制是当时分析师解释，渠道原通知未取得。",
         "limit":"无法把两个月以上模型变化全部归给一个事件，也不能量化结算政策与真实需求各占多少。"},
        {"company":"五粮液","observed":"刊载利润表与EPS预测未变，目标PE降低。",
         "upstream":"券商称产品顺价后，投入由经销商直接补贴转向终端；承兑汇票免贴息安排减轻经销商资金压力，改变回款工具。",
         "evidence":"华泰2018年10月28日第1页为政策转述；公司三季报第8页明确经营现金流受收到更多承兑汇票、税费及采购现金增加共同影响。",
         "limit":"公司原文能确认现金流影响，尚不能独立确认券商描述的政策门槛、期限和完整条款；票据增多不等于终端销量必然下降。"},
        {"company":"洋河股份","observed":"刊载利润表与EPS预测未变，目标PE降低；毛利率同比显著上升。",
         "upstream":"公司生产安排改变后，部分消费税从生产成本转到税金及附加；同时券商解释还包含产品组合、提价及市场扩张。",
         "evidence":"公司三季报第6页明确说明税的列报变化，第27页给出成本和税金原值；两期华泰第1页说明经营解释。",
         "limit":"毛利率提升不能全归给定价权。收入减成本再减税金是辅助比较，仍非完全可比的纯经营利润或精确的重分类金额。"}]
    old=json.loads((EARNINGS/"财报原值及页码.json").read_text(encoding="utf-8"))
    fields={row["key"]:row for row in old}
    rev=np.array(fields["y_revenue_ytd"]["values"])
    cost=np.array(fields["y_cost_ytd"]["values"])
    tax=np.array(fields["y_tax_ytd"]["values"])
    margin=1-cost/rev;after_tax=1-(cost+tax)/rev
    result={"mechanisms":records,"reused_yanghe_source_rows":[fields[k] for k in ["y_revenue_ytd","y_cost_ytd","y_tax_ytd"]],
            "yanghe_actual_ytd":{"columns":["2018年前三季度","2017年前三季度"],"gross_margin":margin,
                "revenue_minus_cost_and_business_tax_ratio":after_tax,"gross_margin_change_pp":(margin[0]-margin[1])*100,
                "after_cost_tax_ratio_change_pp":(after_tax[0]-after_tax[1])*100,
                "limit":"将税金合并只减轻列报位置差异，不是税后净利率，也不精确识别重分类金额。"}}
    save("上游原因与观察量的关系.json",result)
    return result


def compute_returns(cfg,sources):
    market=pd.read_parquet(ROOT/"reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet").sort_values("date").reset_index(drop=True)
    market["date"]=pd.to_datetime(market["date"])
    calendar=pd.DatetimeIndex(market.date)
    assert calendar.is_unique
    dividends=pd.read_csv(ROOT/"data/reference/510300_dividends.csv")
    dividends=dividends.loc[dividends.symbol=="510300.SH"].copy()
    dividends["record_date"]=pd.to_datetime(dividends["record_date"])
    rows=[]
    for spec in sources:
        start=int(calendar.searchsorted(pd.Timestamp(spec["document_date"]),side="right"))
        for horizon in cfg["horizons_sessions"]:
            entry,end=market.iloc[start],market.iloc[start+horizon]
            cash=float(dividends.loc[(dividends.record_date>=entry.date)&(dividends.record_date<end.date),"cash_dividend_per_share"].sum())
            calc=round_trip(entry.open,end.open,cash,cfg["costs"])
            rows.append({"source_id":spec["id"],"symbol":spec["symbol"],"name":LABELS[spec["symbol"]],
                "period":spec["period"],"document_date":spec["document_date"],"entry_date":entry.date.date().isoformat(),
                "exit_date":end.date.date().isoformat(),"entry_index":start,"exit_index":start+horizon,
                "holding_sessions":horizon,"entry_open":float(entry.open),"exit_open":float(end.open),
                "dividends_per_share":cash,"gross_return":float((end.open+cash)/entry.open-1),**calc})
    unique=[]
    for date in sorted({r["entry_date"] for r in rows}):
        subset=[r for r in rows if r["entry_date"]==date]
        unique.append({"entry_date":date,"event_ids":list(dict.fromkeys(r["source_id"] for r in subset)),
                      "five_day_net_return":next(r["net_return"] for r in subset if r["holding_sessions"]==5),
                      "twenty_day_net_return":next(r["net_return"] for r in subset if r["holding_sessions"]==20)})
        for horizon in cfg["horizons_sessions"]:
            assert len({r["net_return"] for r in subset if r["holding_sessions"]==horizon})==1
    save("六份报告日期后的510300参考收益.json",rows)
    save("合并同日后的五个收益节点.json",unique)
    return rows,unique


def render_figure(revisions,valuations,coverage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family":font.get_name(),"axes.unicode_minus":False,"font.size":10,
                        "axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(1,3,figsize=(15.6,5.6),layout="constrained")
    symbols=["600519.SH","000858.SZ","002304.SZ"]
    colors=["#285b79","#6fa5a3","#c57b48"]
    x=np.arange(3)
    for i,year in enumerate([2018,2019,2020]):
        ys=[next(r["eps_revision"] for r in revisions if r["symbol"]==s and r["forecast_year"]==year)*100 for s in symbols]
        bars=axs[0].bar(x+(i-1)*.24,ys,width=.23,color=colors[i],label=f"{year}年")
        axs[0].bar_label(bars,fmt="%.1f",padding=3,fontsize=9)
    axs[0].set_xticks(x,[LABELS[s] for s in symbols]);axs[0].set_ylim(-28,7)
    axs[0].set_ylabel("刊载EPS预测修正（%）");axs[0].axhline(0,color="#999999",lw=.7)
    axs[0].set_title("盈利下调未共同出现",loc="left",fontweight="bold")
    axs[0].legend(frameon=False,loc="lower right")
    for i,(key,label) in enumerate([("prior_target_pe_range","半年点评"),("new_target_pe_range","三季点评")]):
        vals=[float(np.mean(next(r[key] for r in valuations if r["symbol"]==s))) for s in symbols]
        bars=axs[1].bar(x+(i-.5)*.32,vals,width=.30,color=colors[i*2],label=label)
        axs[1].bar_label(bars,fmt="%.1f",padding=3)
    axs[1].set_xticks(x,[LABELS[s] for s in symbols]);axs[1].set_ylim(0,32)
    axs[1].set_ylabel("基于2018年EPS的目标PE中点（倍）")
    axs[1].set_title("三家目标估值均下调",loc="left",fontweight="bold")
    axs[1].legend(frameon=False,loc="lower left")
    vals=[r["covered_subtotal"]*100 for r in coverage["sensitivity"]]
    bars=axs[2].bar(x,vals,width=.55,color=colors[0]);axs[2].bar_label(bars,fmt="%.2f",padding=3)
    axs[2].set_xticks(x,["2018年EPS","2019年EPS","2020年EPS"]);axs[2].set_ylim(-1.05,.16)
    axs[2].set_ylabel("固定PE下的已覆盖贡献小计（百分点）")
    axs[2].set_title("覆盖仅占指数权重5.503%",loc="left",fontweight="bold")
    axs[2].axhline(0,color="#999999",lw=.7)
    fig.suptitle("2018年消费权重：盈利修正分化，刊载目标估值普遍降低",fontsize=17,fontweight="bold")
    fig.supxlabel("固定华泰原报告、同一预测年度；伊利两期缺失。右图假设EPS同比例传入股价，PE不变，\n仅为三家已覆盖部分的机械敏感度，不是全指数预测、实际跌幅归因或交易收益。",fontsize=10)
    fig.savefig(OUT/"历史盈利分化与估值下调.png",dpi=150)
    plt.close(fig)


def report(revisions,valuations,peers,coverage,upstream,unique,result):
    pct=lambda value:f"{value*100:+.2f}%"
    lines=["# 历史发现：消费权重的盈利修正分化，目标估值普遍下调","",
        "本轮完成2018年半年点评至三季点评的同机构比较。固定茅台、五粮液、洋河、伊利四家公司，取得华泰六份原报告，覆盖前三家；伊利两期资料未取得，未用其他机构替换。研究只涉及已经发生的历史，不新增前瞻跟踪。","",
        "核心发现：茅台的刊载盈利预测与目标PE同时下调；五粮液和洋河的刊载EPS及利润表预测没有改变，目标PE却同步降低。这个范围内并未出现三家盈利预测共同下调。目标PE变化本身也可能跟随同行已有价格变化，不能直接当作领先原因。","",
        "## 一、同一机构、相同预测年度的原表比较","",
        "|公司|预测年度|半年点评EPS|三季点评EPS|EPS修正|收入预测修正|归母利润预测修正|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in revisions:
        lines.append(f"|{r['name']}|{r['forecast_year']}|{r['prior_eps']:.2f}|{r['new_eps']:.2f}|{pct(r['eps_revision'])}|{pct(r['revenue_revision'])}|{pct(r['parent_profit_revision'])}|")
    lines += ["|伊利股份|2018—2020|未取得|未取得|不计算|不计算|不计算|","",
        "三家公司相邻点评分别为：茅台8月2日／10月29日，五粮液8月27日／10月28日，洋河8月29日／10月28日。所有预测都以报告当时的年份为准，2019、2020列是2018年留下的历史预测，不是本任务对未来的预测。","",
        "五粮液两版首段均写3.37、4.30、5.31元，而两页预测表及投资结论均写3.37、4.33、5.34元。使用表格数值并保留不一致；按正文的两期数值比较也同样没有修正。洋河三季点评的现金流表与利润表中净利润、财务费用存在不一致，因此这里只确认刊载EPS及利润表预测未变，不能扩大为整套模型或分析师真实预期均未变。","",
        "茅台三季点评追述旧2019年EPS为35.40元，原半年点评表格则为35.50元。本轮使用两期原表直接比较。模型的2017年收入基数61063百万元对应营业总收入，只在原模型之间同口径比较。","",
        "## 二、目标估值下调的组成和信息边界","",
        "|公司|2018目标PE：旧→新|目标价中点：旧→新（元）|中点变化|其中EPS部分（元）|其中PE部分（元）|",
        "|---|---|---:|---:|---:|---:|"]
    for r in valuations:
        a,b=r['prior_target_pe_range'],r['new_target_pe_range']
        lines.append(f"|{r['name']}|{a[0]}—{a[1]} → {b[0]}—{b[1]}|{r['prior_target_midpoint_cny']:.2f} → {r['new_target_midpoint_cny']:.2f}|{pct(r['target_midpoint_revision'])}|{r['eps_component_cny']:.2f}|{r['pe_component_cny']:.2f}|")
    lines += ["","采用对称恒等式拆分EPS×PE：EPS变化乘平均PE，加上PE变化乘平均EPS。它只解释该券商目标价中点的变化，不能据此把真实股价下跌分配给盈利、利率或风险溢价。","",
        "五粮液与洋河的报告用同行当时的一致预期PE作为相对估值依据：","",
        "|报告对象|半年点评同行2018年平均PE|三季点评同行2018年平均PE|变化|同行名单是否一致|",
        "|---|---:|---:|---:|---|"]
    for p in peers:
        lines.append(f"|{LABELS[p['symbol']]}|{p['prior']['reported_average_pe_2018']:.2f}|{p['new']['reported_average_pe_2018']:.2f}|{pct(p['reported_average_pe_2018_revision'])}|各自15家名单一致|")
    lines += ["","这些同行PE由价格和一致预期盈利共同决定。资料足以确认相对估值参照下降，尚不能区分无风险利率、权益风险补偿、增长预期与仓位调整的具体贡献。尤其不能把‘同行股价先跌，目标PE随之下调’再包装成导致此前下跌的独立因子。","",
        "## 三、继续追查观察量为什么变化","",
        "**五粮液：回款工具与渠道资金安排。**公司三季报明确将现金流变化与收到更多承兑汇票、税费及采购现金增加联系起来。华泰进一步称，年初承兑汇票免贴息安排用于减轻经销商资金压力，并称核心产品顺价后投入从经销商直接补贴转向终端。公司解释支持现金流不能直接等同于销量的结论；免贴息的具体条件仍属券商转述，未取得渠道原通知。","",
        "**洋河：毛利率受到生产方式和消费税列报变化影响。**公司三季报第6页说明，从2017年9月开始相关消费税由生产成本转入税金及附加。经营观察需要把这两个项目放在一起，才能减少单看毛利率造成的误判。","",
        "|洋河已实现指标|2017年前三季度|2018年前三季度|同比变化|","|---|---:|---:|---:|"]
    y=upstream["yanghe_actual_ytd"]
    for key,label,delta in [("gross_margin","毛利率","gross_margin_change_pp"),("revenue_minus_cost_and_business_tax_ratio","收入减成本、减税金及附加后占收入比","after_cost_tax_ratio_change_pp")]:
        a=y[key];lines.append(f"|{label}|{a[1]*100:.2f}%|{a[0]*100:.2f}%|{y[delta]:+.2f}个百分点|")
    lines += ["","第二行是辅助比较，不是税后净利率，也不声称已精确剥离全部重分类影响。产品组合、提价、税负和经营规模仍同时变化，剩余差异不能全部归于某一个原因。","",
        "**茅台：模型收入预期与利润转化同时修正。**上一轮已拆出收入规模与利润率贡献；本轮保留该分解，并增加两个刊载盈利未下调的同机构对照。当前证据支持公司间的分化，不能把茅台的变化直接推广为消费板块或沪深300盈利全面恶化。","",
        "## 四、从公司证据到510300，影响范围有多大","",
        f"固定四家公司2018年9月末权重合计{coverage['fixed_weight']*100:.3f}%，三家已取得模型的权重合计{coverage['covered_weight']*100:.3f}%，伊利缺失部分为{coverage['missing_weight']*100:.3f}%。该权重的首次公开时刻未核实，仅用于历史影响范围描述。","",
        "计算一个受限的机械量：已覆盖公司权重×各自EPS修正，再求小计。它假设PE不变、EPS变化同比例传入相关股价；不同预测年度分别列示，不把三列相加。","",
        "|使用的历史预测年度|已覆盖部分机械贡献小计|伊利部分|","|---|---:|---|"]
    for r in coverage["sensitivity"]:
        lines.append(f"|{r['forecast_year']}|{r['covered_subtotal']*100:+.3f}个百分点|未知，不填零|")
    lines += ["","因为五粮液和洋河的EPS修正为零，小计全部来自茅台。这不是指数盈利增速，不是实际指数跌幅的解释，也不是全指数预测；余下权重和估值变化没有在这个机械小计中得到识别。不能用一家公司二十多个百分点的EPS下调，直接推出510300应有相同量级的变动。","",
        "## 五、报告日期之后的历史参考交易","",
        "按预先保存的载明日期结束后下一交易日开盘参考买入510300，固定5／20个交易日后开盘退出。报告的最早公开时刻未核实；日期口径只能用于条件性的历史观察，不保证当时已能读取报告，也不覆盖消息的全部首次反应。","",
        "|参考入场日|关联报告|5日成本后收益|20日成本后收益|","|---|---|---:|---:|"]
    name_map={"maotai":"茅台","wuliangye":"五粮液","yanghe":"洋河"}
    for r in unique:
        names=[name_map[key.split('_')[1]]+("半年点评" if key.endswith("h1") else "三季点评") for key in r["event_ids"]]
        lines.append(f"|{r['entry_date']}|{'、'.join(names)}|{pct(r['five_day_net_return'])}|{pct(r['twenty_day_net_return'])}|")
    lines += ["","每笔示例预算10万元，100份整手，佣金单边万四且最低5元，滑点单边0.1%，价格按0.001元向不利方向取整；按登记日计算分红权益。收益分母为实际投入，并非20万元全账户。六份报告只有五个不同入场日，窗口相互重叠，部分收益此前已见，不是五个独立实验。","",
        "这些结果没有把公司消息从同期宏观与市场变化中隔离出来。全部日期和两个期限都保留，未按表现挑选、反向或改期限。当前不能据此声称报告的负面修正带来确定的后续下跌，也未建立可验收的指数交易规则。","",
        "## 六、对后续历史发现的具体推进","",
        "用户指出研究过于关注个股，本轮据此收束个股支线。这一轮削弱了‘三家消费权重盈利预期共同下调’的解释，保留的结论仅用于界定局部影响范围。后续研究主线改为指数整体的货币与信用传导、总需求、风险补偿和资金约束；不再沿同行公司逐个扩展。以整体指数的价格反应、上涨覆盖面及可成交后的剩余收益检验机制。","",
        f"本轮新增4份用于比较的原报告，复用2份；另保存2份定位资料。完成{result['source_value_rows']}条模型原值记录、9个公司年度修正、3个目标价分解、5个ETF日期节点及必要的单位和损益核对。未搜索参数，未新增完整账户，未新增前瞻任务。夏普1.2及并列收益、风险目标仍未证明达成。","",
        f"![历史盈利分化与估值下调]({(OUT/'历史盈利分化与估值下调.png').as_posix()})","",
        "## 原始来源及本地计算","",
        "- [茅台半年点评](https://crm.htsc.com.cn/doc/2018/10710208/5898348c-fd50-45fc-95b8-b949c21ce296.pdf)与[三季点评](https://crm.htsc.com.cn/doc/2018/10710209/b6db5475-e389-42d7-bbb3-3159b5b36d91.pdf)：第1、3页。",
        "- [五粮液半年点评](https://crm.htsc.com.cn/doc/2018/10710208/dc029967-abe2-469f-84a9-11ccf132976c.pdf)与[三季点评](https://crm.htsc.com.cn/doc/2018/10710209/46e11c7a-057f-4781-8569-e3ffe7006bc5.pdf)：第1页经营解释与目标PE，第2页同行表，第3页模型。",
        "- [洋河半年点评](https://crm.htsc.com.cn/doc/2018/10710208/96f1ed80-ebe7-4100-b003-7df543150da5.pdf)与[三季点评](https://crm.htsc.com.cn/doc/2018/10710209/66d14a1d-b9e0-4d20-9945-4c7b0c19f347.pdf)：第1—3页。",
        "- [洋河公司三季报](https://static.cninfo.com.cn/finalpage/2018-10-27/1205543056.PDF)：第6页消费税说明、第27页原表；[五粮液公司三季报](https://static.cninfo.com.cn/finalpage/2018-10-29/1205545060.PDF)：第8页现金流原因。",
        "- 八个固定资料位置及缺失.json保留缺项及检索边界；模型原值及页码.json保留单位与原行；刊载盈利及目标估值修正.json保留计算和文内矛盾。",
        "- 固定权重覆盖与机械敏感度.json、六份报告日期后的510300参考收益.json保留计算值；research/historical_consumer_revision_breadth_v1.py analyze可用本地文件重算。",""]
    filename="历史发现_盈利分化与估值下调的区别.md"
    (OUT/filename).write_text("\n".join(lines),encoding="utf-8")
    return filename


def analyze():
    cfg=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
    sources=json.loads((OUT/"selected_sources.json").read_text(encoding="utf-8"))
    source_coverage(sources)
    revisions,valuations,peers,coverage,checks=analyze_models(cfg,sources)
    upstream=upstream_records()
    events,unique=compute_returns(cfg,sources)
    assert len(sources)==6 and len(revisions)==9 and len(events)==12 and len(unique)==5
    residuals=[]
    for row in events:
        pnl=(row["sell_price"]-row["buy_price"])*row["shares"]-row["commissions_cny"]+row["dividend_entitlement_cny"]
        residuals.append(abs(pnl-row["net_pnl_cny"]))
        assert row["paid_cny"]<=cfg["costs"]["illustrative_event_budget_cny"] and row["shares"]%100==0
        assert row["document_date"]<row["entry_date"]<row["exit_date"]
        assert row["exit_index"]-row["entry_index"]==row["holding_sessions"]
        assert abs(row["net_pnl_cny"]/row["paid_cny"]-row["net_return"])<1e-12
    assert max(residuals)<.01
    checks.update(status="PASS_NECESSARY_MODEL_UNIT_IDENTITY_AND_EVENT_CHECKS",primary_model_pages_visually_reviewed=True,
                  event_pnl_identity_max_abs_cny=max(residuals),figure_visually_reviewed=False,
                  source_conflicts_retained=True,first_public_time_verified=False,independent_validation=False)
    save("calculation_checks.json",checks)
    result={"study_id":STUDY,"completed_at":now(),"research_mode":"HISTORICAL_ONLY",
        "classification":"PROGRESS_HISTORICAL_EARNINGS_REVISION_BREADTH_AND_VALUATION_ENDOGENEITY",
        "status":"HISTORICAL_SAME_INSTITUTION_THREE_COMPANY_COMPARISON_COMPLETED_FOURTH_MISSING",
        "fixed_companies":4,"paired_companies":3,"missing_paired_companies":1,"paired_model_documents":6,
        "new_paired_model_pdfs":4,"reused_model_pdfs":2,"new_locator_pdfs":2,
        "source_value_rows":checks["source_value_rows"],"company_year_revisions":9,
        "eps_cut_companies":1,"unchanged_published_eps_companies":2,"target_pe_cut_companies":3,
        "fixed_weight":coverage["fixed_weight"],"covered_weight":coverage["covered_weight"],
        "unique_etf_entry_dates":len(unique),"return_rows_including_duplicates":len(events),
        "new_parameters_fitted":0,"new_full_accounts":0,"new_prospective_tasks":0,
        "independent_validation":False,"net_sharpe":None,"goal_achieved":False,"orders_authorized":False,
        "scope_correction":"用户指出过于关注个股；收束本支线，指数整体机制为主线，个股仅限解释权重范围。",
        "next_historical_question":"固定2018—2019年新宣布的降准事件，拆开银行流动性释放、实体信用传导和指数价格反应，核对当时已公布的需求信息与公告前后收益；不沿个股继续扩展，不把政策动作直接等同于股票流入。"}
    render_figure(revisions,valuations,coverage)
    result["report"]=report(revisions,valuations,peers,coverage,upstream,unique,result)
    save("result.json",result)
    print(json.dumps(clean({"结果":result,"覆盖及敏感度":coverage,"收益节点":unique,"必要核对":checks}),ensure_ascii=False,indent=2),flush=True)


def record_progress():
    result=json.loads((OUT/"result.json").read_text(encoding="utf-8"))
    checks=json.loads((OUT/"calculation_checks.json").read_text(encoding="utf-8"))
    events=json.loads((OUT/"六份报告日期后的510300参考收益.json").read_text(encoding="utf-8"))
    errors=[abs((r["sell_price"]-r["buy_price"])*r["shares"]-r["commissions_cny"]+r["dividend_entitlement_cny"]-r["net_pnl_cny"]) for r in events]
    assert max(errors)<.01
    checks.update(saved_event_pnl_recomputation_max_abs_cny=max(errors),figure_visually_reviewed=True)
    save("calculation_checks.json",checks)
    prefix="reports/research/510300_historical_consumer_revision_breadth_v1/"
    stamp=now()
    receipt={"recorded_at":stamp,"study_id":STUDY,"user_authority":"历史检验和发现；追查因子背后原因；减少繁琐检验。",
             "result":prefix+"result.json","goal_achieved":False,"orders_authorized":False,"changes":[],"old_rejections_retained":True}
    for name in ["510300_historical_cause_discovery_v1.json","510300_existing_data_training_mandate_v1.json"]:
        path=ROOT/"config"/name
        cfg=json.loads(path.read_text(encoding="utf-8"));before=dict(cfg)
        if name=="510300_historical_cause_discovery_v1.json":
            cfg.update(latest_completed_study=prefix+"result.json",latest_report=prefix+result["report"],updated_at=stamp)
        else:
            cfg.update(current_round=STUDY,latest_progress_receipt=prefix+"result.json",latest_historical_consumer_revision_breadth=prefix+"result.json",
                latest_continuation_report=prefix+result["report"],latest_historical_report=prefix+result["report"],
                latest_continuation_classification=result["classification"],current_driver_continuation_classification=result["classification"],
                current_driver_consecutive_blocked_goal_turns=0,latest_historical_diagnostic_at=stamp,
                latest_goal_service_status="active",latest_goal_service_status_observed_at=stamp,goal_status="active",goal_achieved=False,
                local_goal_work_status="ACTIVE_HISTORICAL_ONLY",
                last_research_result="三家同机构两期刊载模型中茅台盈利下调，五粮液及洋河EPS未变，三家目标PE均降；伊利缺失。明确相对估值可能跟随股价的内生性，完成5个ETF日期节点，尚无达标完整账户。",
                last_source_result="新增4份华泰2018原报告并复用2份；四家公司保留缺失，覆盖权重5.503%。原文EPS和现金流表矛盾保留，核算依刊载利润表，不把单一机构视为全市场共识。",
                next_research_question=result["next_historical_question"])
        receipt["changes"].append({"path":str(path),"fields":{k:{"before":before.get(k),"after":v} for k,v in cfg.items() if before.get(k)!=v}})
        path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    save("authority_update.json",receipt)
    print("已登记历史盈利分化与目标估值下调证据；夏普1.2目标仍在进行中且未达成。",flush=True)


def main():
    parser=argparse.ArgumentParser(description="历史消费盈利修正扩散")
    parser.add_argument("mode",choices=["prepare","fetch","analyze","record-progress"])
    args=parser.parse_args()
    {"prepare":prepare,"fetch":fetch,"analyze":analyze,"record-progress":record_progress}[args.mode]()


if __name__=="__main__":main()
