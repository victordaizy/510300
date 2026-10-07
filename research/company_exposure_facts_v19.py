"""将六家原权重公司的经营事实、公告时钟与既有收益并排保存。"""
from decimal import Decimal
import json
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from real_yield_company_sources_v19 import OUT, CASES, COMPANIES, digest, now, save

NAMES = {"600519.SH":"贵州茅台", "601318.SH":"中国平安", "000858.SZ":"五粮液",
         "600036.SH":"招商银行", "000333.SZ":"美的集团", "600276.SH":"恒瑞医药"}
FACTS = []
PAGES = {}


def compact(text):
    return re.sub(r"\s+", "", text).replace(",", "").replace("，", "")


def source(name, page):
    if name.endswith(".pdf"):
        if name not in PAGES:
            PAGES[name] = json.loads((OUT/"sources"/name.replace(".pdf", ".pages.json")).read_text("utf-8"))
        txt = PAGES[name][page-1]["text"]
    else:
        txt = (OUT/"sources"/name.replace(".html", ".txt")).read_text("utf-8")
    receipt = json.loads((OUT/"sources"/(name+".receipt.json")).read_text("utf-8"))
    date = receipt.get("published_date")
    known = receipt.get("known_at") or (date+"T23:59:59+08:00" if date else None)
    return txt, receipt, known


def add(symbol, doc, page, key, label, current, previous=None, reported_change=None,
        unit="元", scope="合并", period="2020年1—9月", comparison="2019年1—9月",
        change_unit="同比百分比", note="", token_check=True):
    txt, receipt, known = source(doc, page)
    assert compact(label) in compact(txt), (doc, page, label)
    # 页面的列归属已经目视核对；词元核对只检验原数字，没有以邻接数字代替列归属。
    if token_check:
        tokens = {Decimal(t) for t in re.findall(r"(?<![\d.])[+-]?\d+(?:\.\d+)?(?![\d.])", txt.replace(",", ""))}
        for value in [current, previous, reported_change]:
            if value is not None:
                # 平安的同比负值以括号表示；方向取经过目视核对的原列。
                assert Decimal(str(value)) in tokens or (value<0 and Decimal(str(-value)) in tokens), (doc,page,key,value)
    growth = (100*(current/previous-1) if previous is not None and previous>0 else None)
    row = {"fact_id":f"{symbol}:{key}", "symbol":symbol, "company":NAMES[symbol], "metric":key,
           "label":label, "current":current, "previous":previous, "reported_change":reported_change,
           "reported_change_unit":change_unit, "computed_growth_percent":growth,
           "unit":unit, "scope":scope, "period":period, "comparison":comparison, "source_file":doc,
           "pdf_page":page, "known_at_assumed":known,
           "clock_status":"PUBLICATION_DATE_ENDDAY_ASSUMED" if known else "RETROSPECTIVE_DOCUMENT_CLOCK_UNVERIFIED",
           "historical_first_vintage_verified":False,
           "source_url":receipt["url"], "source_sha256":digest(OUT/"sources"/doc), "note":note}
    FACTS.append(row)
    return row


def csv(name, data):
    data.to_csv(OUT/"results"/name,index=False,encoding="utf-8-sig",float_format="%.15g")


def main():
    if (OUT/"results/company_receipt.json").exists():
        raise RuntimeError("公司事实已完成，不覆盖。")
    frozen=json.loads((OUT/"freeze.json").read_text("utf-8"))
    assert digest(OUT/"protocol.json")==frozen["protocol_sha256"]
    # 六家公司均保存，金融企业的现金流不套用制造业的回款含义。
    base = [
      ("600519.SH",3,"元",67214944638.48,60934658070.86,10.31,33827103961.22,30454855385.44,11.07,25111002819.99,27315401962.03,-8.07),
      ("601318.SH",3,"百万元",917070,892751,2.7,103041,129567,-20.5,264313,330888,-20.1),
      ("000858.SZ",3,"元",42492767199.91,None,14.53,14545454674.88,None,15.96,3943471634.98,None,-75.64),
      ("600036.SH",2,"百万元",221430,207730,6.60,76603,77239,-.82,219828,-53732,None),
      ("000333.SZ",3,"千元",216760786,None,-1.88,22018301,None,3.29,25014635,None,-16.03),
      ("600276.SH",3,"元",19413125067.81,16945046466.99,14.57,4258592129.13,3734893398.47,14.02,3986528887.68,2608099282.73,52.85)
    ]
    for s,pg,u,r,rp,rg,p,pp,pgrowth,c,cp,cg in base:
        doc=s[:6]+"_2020Q3.pdf"
        for key,label,a,b,g in [("revenue_9m","营业收入",r,rp,rg),("profit_9m","净利润",p,pp,pgrowth),("cfo_9m","经营活动产生的现金流量净额",c,cp,cg)]:
            add(s,doc,pg,key,label,a,b,g,unit=u,
                note="金融企业经营现金流含金融资产负债变动，不与实业企业现金回款直接比较。" if s in ["600036.SH","601318.SH"] and key=="cfo_9m" else "")
    # 五粮液现金变化的完整金额桥：流入、税费、其他流出，三项必须回到现金流净额变化。
    w="000858.SZ"; doc="000858_2020Q3.pdf"
    for key,label,a,b in [("cfo_bridge","经营活动产生的现金流量净额",3943471634.98,16191322654.60),
                          ("cash_out","经营活动现金流出小计",36225520514.98,27414084026.50),
                          ("tax_cash","支付的各项税费",21224900121.56,13591161939.66)]:
        add(w,doc,6,key,label,a,b)
    # 美的既有借款，也有货币性投资；原报告把财务费用变化主要归于利息收入减少。
    s="000333.SZ"; doc="000333_2020Q3.pdf"
    add(s,doc,6,"finance_expense","财务费用",-1462390,-2114090,unit="千元",note="金额为负；差额是利息净收入等财务项目的变化，不推导单一贷款利率。")
    for key,label,a,b in [("short_borrow","短期借款",11626189,5701838),("due1y","一年内到期的非流动负债",7727739,1460117),
                           ("other_current_assets","其他流动资产",86567797,65011027), ("lending","发放贷款和垫款",16792528,11659497)]:
        add(s,doc,6,key,label,a,b,unit="千元",period="2020-09-30",comparison="2019-12-31",change_unit="较年初百分比")
    add(s,doc,3,"revenue_q3","营业收入",77693764,reported_change=15.71,unit="千元",period="2020年7—9月",comparison="2019年7—9月")
    add(s,doc,3,"profit_q3","净利润",8090006,reported_change=32,unit="千元",period="2020年7—9月",comparison="2019年7—9月")
    # 茅台区分财务费用内的利息收入与营业总收入中的金融业务利息收入。
    s="600519.SH"; doc="600519_2020Q3.pdf"
    add(s,doc,14,"finance_expense","财务费用",-155303715.07,-541979.58)
    add(s,doc,14,"finance_interest_income","利息收入",163867810.61,9042607.71,note="财务费用的其中项，位于合并利润表靠下部分。")
    add(s,doc,14,"operating_interest_income","利息收入",2359932222.05,2573995541.88,note="营业总收入中的利息收入，不能与财务费用其中项混为一行。")
    # 恒瑞用合并口径；利息费用空白保留空白，不填零，也不把上半年无借款外推到年末。
    s="600276.SH"; doc="600276_2020Q3.pdf"
    add(s,doc,9,"finance_expense","财务费用",-183233468.65,-107351950.84)
    add(s,doc,9,"finance_interest_income","利息收入",197924710.25,89028829.16,note="本页前半为母公司资产负债表，所取数字在后半合并利润表。")
    add(s,doc,7,"cash_balance","货币资金",11050932380.92,5043646264.33,period="2020-09-30",comparison="2019-12-31")
    # 招行集团经营结果、集团净息差和本行存款结构各自标明范围。
    s="600036.SH"; doc="600036_2020Q3.pdf"
    add(s,doc,7,"loan_balance","贷款和垫款总额",50062.02,reported_change=11.48,unit="亿元",period="2020-09-30",comparison="2019-12-31",change_unit="较年初百分比")
    add(s,doc,7,"nii","净利息收入",1385.35,reported_change=5.57,unit="亿元")
    add(s,doc,7,"nim_9m","净利息收益率",2.51,unit="百分比",note="集团同比下降14基点。")
    add(s,doc,7,"nim_q3","净利息收益率",2.53,unit="百分比",period="2020年7—9月",comparison="2020年4—6月",note="集团环比上升8基点；不能将同比下降和环比上升混用。")
    add(s,doc,7,"parent_demand_deposit_share","活期存款占比",61.33,unit="百分比",scope="本公司",period="2020-09-30",comparison="无配对基期")
    add(s,doc,7,"parent_average_demand_share","日均余额中活期存款占比",59.74,unit="百分比",scope="本公司",note="较2020上半年提高0.92个百分点；集团净息差与本公司存款占比不组成精确因果分解。")
    # 平安寿险新业务和保险资金投资收益，区别净利润、营运利润和年化收益率。
    s="601318.SH"; doc="601318_2020Q3.pdf"
    add(s,doc,10,"nbv_9m","新业务价值",42844,58805,-27.1,unit="百万元",scope="寿险及健康险")
    add(s,doc,11,"net_investment_income","净投资收益",115796,109100,6.1,unit="百万元",scope="保险资金投资组合")
    add(s,doc,11,"total_investment_income","总投资收益",135869,140301,-3.2,unit="百万元",scope="保险资金投资组合")
    add(s,doc,11,"net_investment_yield","净投资收益率",4.5,4.9,unit="百分比",scope="保险资金投资组合",note="年化收益率，金额增长同时发生，不据金额增幅反推收益率增幅。")
    # 原快报，不用之后年报修订金额替换。
    s="600036.SH"; doc="600036_2020快报.pdf"
    add(s,doc,1,"revenue_fy_flash","营业收入",290508,269703,7.71,unit="百万元",period="2020年",comparison="2019年",note="快报未经审计，保持原版本。")
    add(s,doc,1,"profit_fy_flash","净利润",97342,92867,4.82,unit="百万元",period="2020年",comparison="2019年",note="不替换为后来年报数值。")
    # 2月3日通稿已明确发布；全年报告的签署日不冒充发布时钟。
    s="601318.SH"; doc="601318_20210203全年通稿.html"
    add(s,doc,None,"profit_fy_news","净利润",1430.99,reported_change=-4.2,unit="亿元",period="2020年",comparison="2019年")
    add(s,doc,None,"operating_profit_fy_news","营运利润",1394.70,reported_change=4.9,unit="亿元",period="2020年",comparison="2019年")
    add(s,doc,None,"nbv_fy_news","新业务价值",495.75,reported_change=-34.7,unit="亿元",scope="寿险及健康险",period="2020年",comparison="2019年")
    s="601318.SH"; doc="601318_2020全年报告.pdf"
    for key,label,a,b,sc in [("ev_joint50_group","每年增加50个基点",1381825,1328112,"集团内含价值"),
                            ("ev_joint50_life","每年增加50个基点",878287,824574,"寿险及健康险内含价值"),
                            ("ev_joint50_nbv","每年增加50个基点",53376,49575,"一年新业务价值")]:
        add(s,doc,67,key,label,a,b,unit="百万元",scope=sc,period="2020-12-31情景",comparison="基准情景",
            note="投资收益率与风险贴现率同时增加50基点；不是仅提高折现率的情景。完整报告发布时钟未证实，仅回顾。")
    add(s,doc,79,"bond_fv_profit_loss","减少利润",4377,unit="百万元",scope="指定公允价值债券",period="2020-12-31情景",comparison="基准情景",note="政府债券收益率曲线平行上升50基点的利润减少额；不等于美国TIPS上升50基点。")
    add(s,doc,79,"bond_fv_equity_loss","减少权益",14384,unit="百万元",scope="指定公允价值债券",period="2020-12-31情景",comparison="基准情景",note="与上述内含价值的联合情景不同，不能相互抵销。")
    facts=pd.DataFrame(FACTS)
    assert facts.fact_id.is_unique
    csv("公司原表事实_金额期间范围与发布时间.csv",facts)
    f=facts.set_index("fact_id")
    get=lambda sym,k:f.loc[f"{sym}:{k}"]
    cash=get(w,"cfo_bridge");out=get(w,"cash_out");tax=get(w,"tax_cash")
    cf_in=(cash.current+out.current)-(cash.previous+out.previous)
    tax_delta=tax.current-tax.previous
    other_delta=(out.current-out.previous)-tax_delta
    bridge=pd.DataFrame([{"company":"五粮液","item":label,"change_yi":value/1e8} for label,value in [
        ("经营现金流入变化",cf_in),("支付税费增加",-tax_delta),("其余经营现金流出增加",-other_delta)]])
    np.testing.assert_allclose(bridge.change_yi.sum(),(cash.current-cash.previous)/1e8,atol=1e-12)
    csv("五粮液经营现金流变化_三项金额桥.csv",bridge)
    prelim=get("600036.SH","profit_fy_flash");q3=get("600036.SH","profit_9m")
    q4a=prelim.current-q3.current;q4b=prelim.previous-q3.previous
    save("results/公司派生事实.json",{"at":now(),"wuliangye_cfo_decline_yi":(cash.current-cash.previous)/1e8,
         "wuliangye_tax_delta_yi":tax_delta/1e8,"wuliangye_tax_accounting_share_percent":100*tax_delta/(cash.previous-cash.current),
         "cmb_implied_q4_profit_2020_yi":q4a/100,"cmb_implied_q4_profit_2019_yi":q4b/100,
         "cmb_implied_q4_growth_percent":100*(q4a/q4b-1),
         "implied_q4_scope":"原全年快报减原前三季度，不以后来年报回填；不是单独披露的四季度利润。",
         "causality":"金额桥是会计恒等式；不把一家公司的现金变化说成全国M1变化的原因。"})
    # 用市场已有交易日日历确定公告后第一个开盘，不增补或选择公司公告。
    market=pd.read_csv(OUT/"inputs/market.csv")
    opens=pd.DatetimeIndex([pd.Timestamp(d+"T09:30:00+08:00") for d in market.date])
    docs=[]
    for doc,d in facts.groupby("source_file",sort=False):
        known=d.known_at_assumed.iloc[0]
        first=opens[opens>pd.Timestamp(known)][0].isoformat() if pd.notna(known) else None
        docs.append({"source_file":doc,"symbol":d.symbol.iloc[0],"known_at_assumed":known,
                     "first_next_A_open":first,"clock_status":d.clock_status.iloc[0],"source_url":d.source_url.iloc[0]})
    docs=pd.DataFrame(docs)
    csv("公司文件时钟_未确认年报不提前入场.csv",docs)
    monthly=pd.read_csv(OUT/"inputs/monthly.csv").set_index("stat_month")
    clocks=[]
    for case in CASES:
        a=monthly.loc[case]
        for d in docs.to_dict("records"):
            known=d["known_at_assumed"]
            if pd.isna(known):
                state="RETROSPECTIVE_ONLY_UNKNOWN_PUBLICATION_CLOCK"
            elif pd.Timestamp(known)<=pd.Timestamp(a.snapshot_at):
                state="KNOWN_AT_ORIGIN_ASSUMED"
            elif pd.Timestamp(known)<=pd.Timestamp(a.E0_20_exit_date+"T15:00:00+08:00"):
                state="WINDOW_NEW_INFORMATION_NOT_ORIGIN"
            else:
                state="AFTER_WINDOW_NOT_ORIGIN"
            clocks.append({"stat_month":case,"snapshot_at":a.snapshot_at,**d,"state":state})
    csv("三窗口公司公告可见性.csv",pd.DataFrame(clocks))
    stocks=pd.read_csv(OUT/"inputs/stocks.csv")
    selected=stocks[stocks.symbol.isin(COMPANIES)].copy()
    assert len(selected)==18 and selected.status.eq("AVAILABLE_FIXED_SHARES_CASH").all()
    for case,g in stocks.groupby("stat_month"):
        assert set(g[g.top5].symbol)==set(COMPANIES[:-1])
        pharma=g[g.industry_reference.eq("医药生物")].sort_values("initial_weight",ascending=False).iloc[0]
        assert pharma.symbol==COMPANIES[-1]
    selected["company"]=selected.symbol.map(NAMES)
    selected["company_document"]=selected.symbol.str[:6]+"_2020Q3.pdf"
    selected["company_document_known_at"]=selected.company_document.map(docs.set_index("source_file").known_at_assumed)
    selected["snapshot_at"]=selected.stat_month.map(monthly.snapshot_at)
    assert (pd.to_datetime(selected.company_document_known_at)<=pd.to_datetime(selected.snapshot_at)).all()
    for metric in ["revenue_9m","profit_9m","cfo_9m"]:
        table=facts[facts.metric.eq(metric)].set_index("symbol")
        selected[metric+"_reported_yoy_percent"]=selected.symbol.map(table.reported_change)
    selected["original_weight_percent"]=100*selected.initial_weight
    selected["stock_return20_cc_percent"]=100*selected.stock_return20_cc
    selected["outcome_role"]="原窗口收益，不能充当起点的公司判断依据"
    csv("18个原公司窗口_经营背景与事后收益.csv",selected)
    notes=[
        {"company":"五粮液","source_file":"000858_2020Q3.pdf","page":7,"source_check":"上年末及上年同期票据到期托收额度更高",
         "meaning":"回款跨年、票据托收时点与税款共同改变经营现金；利润与现金流方向可不同，现金流降幅不能直接等同需求降幅。"},
        {"company":"美的集团","source_file":"000333_2020Q3.pdf","page":6,"source_check":"主要系利息收入减少所致",
         "meaning":"借款和金融资产并存；财务费用同比变动同时包含利息收入端，不能仅从贷款增加推断利息负担。"},
        {"company":"贵州茅台","source_file":"600519_2020Q3.pdf","page":6,"source_check":"财务费用减少主要是商业银行存款利息收入增加",
         "meaning":"财务费用中的存款利息和金融子公司营业利息为不同栏；现金流还含集团金融业务，不能视作纯酒类回款。"},
        {"company":"中国平安","source_file":"601318_2020Q3.pdf","page":11,"source_check":"缓解公司在低利率环境下的再投资风险",
         "meaning":"长端利率改变再投资收益和资产负债久期；同时存量公允价值债券有估值变化，新业务还有独立经营渠道。"}
    ]
    for row in notes:
        txt,_,_=source(row["source_file"],row["page"])
        assert compact(row["source_check"]) in compact(txt)
    save("results/原报告解释与推断边界.json",notes)
    save("results/company_receipt.json",{"at":now(),"status":"BUILT_SIX_COMPANIES_WITH_SOURCE_CLOCKS","numeric_facts":len(facts),
         "companies":len(COMPANIES),"company_windows":len(selected),"documents_used":len(docs),
         "unknown_publication_documents":int(docs.known_at_assumed.isna().sum()),
         "cash_bridge_items":len(bridge),"goal_achieved":False,"new_models":0,"new_accounts":0})
    shutil.copy2(Path(__file__),OUT/"code"/Path(__file__).name)
    print(json.dumps({"公司数":len(COMPANIES),"数值事实":len(facts),"窗口公司":len(selected),"文件数":len(docs),
                      "五粮液现金流三项":bridge.to_dict("records"),"招商隐含四季度利润增速":100*(q4a/q4b-1)},ensure_ascii=False))


if __name__=="__main__":
    main()
