"""指数共同重估的汇率组成与跨境配置历史研究。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.historical_index_participation_channels_v1 import digest, now, read, rel, save
from research import historical_index_reopening_constraints_v1 as execution

OUT = ROOT / "reports/research/510300_historical_index_currency_allocation_v1"
PREVIOUS = ROOT / "reports/research/510300_historical_index_allocation_scope_v1"
REPORT = OUT / "历史发现_汇率组成与指数配置.md"
LOCAL = {
    "index": PREVIOUS / "official_index_daily.parquet",
    "previous_months": PREVIOUS / "monthly_comparison.parquet",
    "northbound": ROOT / "data/raw/flow/northbound_legacy_net_flow_v1.parquet",
    "midpoint": ROOT / "data/raw/macro/510300_macro_stress_2015_v2/usdcny_midpoint_daily_2015_2026.parquet",
    "dxy": ROOT / "reports/research/510300_rmb_residual_state_v1/inputs/dxy.parquet",
    "us_yields": ROOT / "reports/research/510300_historical_index_external_reopening_v1/treasury_curves.parquet",
    "cnh_context": ROOT / "reports/research/510300_rmb_residual_state_v1/inputs/cnh.parquet",
}
SOURCES = [
    {"id": "cfets_202210", "url": "https://www.chinamoney.com.cn/english/bmkidxrud/20221101/2489781.html", "role": "10月篮子指数初始公告"},
    {"id": "cfets_202211", "url": "https://www.chinamoney.com.cn/english/bmkidxrud/20221201/2508579.html", "role": "11月篮子指数初始公告"},
    {"id": "cfets_202212", "url": "https://www.chinamoney.com.cn/english/bmkidxrud/20230103/2526845.html", "role": "12月篮子指数初始公告"},
    {"id": "cfets_basket_2023", "url": "https://www.chinamoney.com.cn/chinese/zxpl/20221230/2525761.html", "role": "2023篮子权重制度说明"},
    {"id": "cfets_january_review", "url": "https://www.chinamoney.com.cn/dqs/cm-s-notice-query/fileDownLoad.do?contentId=2561280&mode=save&priority=0", "role": "2023年1月境外人民币综述，月末报价和篮子水平"},
    {"id": "safe_january", "url": "https://www.safe.gov.cn/safe/2023/0215/22326.html", "role": "1月跨境股票配置与贸易收付，披露日晚于统计月"},
    {"id": "cfets_archive", "url": "https://www.chinamoney.com.cn/english/bmkidxrud/", "role": "寻找原始月度公告，不作历史事实来源"},
    {"id": "hkex_archive", "url": "https://www.hkex.com.hk/Mutual-Market/Stock-Connect/Statistics/Historical-Monthly?sc_lang=en", "role": "原始股票互联互通月度资料入口"},
]


def prepare():
    save(OUT / "protocol.json", {
        "study_id": "510300_HISTORICAL_INDEX_CURRENCY_ALLOCATION_V1", "created_at": now(),
        "previous_goal_turn_classification": "PROGRESS_INDEX_SIZE_SCOPE_AND_REBALANCE_DENOMINATOR",
        "question": "固定六个月中，对美元升值是否意味着人民币整体走强；北向净买入的量级和月内时序是否支持沪深300整体需求变化；当时已知的上游信息是什么？",
        "calendar": ["2022-09-30", "2023-03-31"], "main_units": "沿用六个自然月，另列各月全部交易日，不由结果选月份",
        "known_history": "已见指数六个月收益、已有美国政策和国内约束资料；检索阶段已见11月篮子跌2.86%、1月篮子约升1.2%。本轮是历史解释性发现，不是盲测。",
        "deduplication": {
            "northbound": "既有六条北向方向/分位规则全失败，不复活、不改阈值",
            "rmb": "RMB_RESIDUAL_STATE旧规则有来源缺口，本轮不运行旧回归、不补2018缺口、不用残差造新信号",
            "crossborder": "ASHR反应不足完整账户失败，原首日残差和退出不变",
            "domestic_external": "复用已有国内政策、PMI和美国CPI/FOMC时钟；只补未覆盖的配置与汇率构成"
        },
        "measurement": {
            "rmb_usd": "官方9:15中间价的月末值，人民币对美元回报=上月末每美元人民币/本月末值-1；不是即期可成交收益",
            "basket": "CFETS官方月末指数及当时公告变化；2023权重调整明确记录，不假设恒定篮子",
            "dxy": "保存的Yahoo美元指数日收盘；月末各自最后日期，非与中国15:00同步",
            "northbound": "保存的Eastmoney历史北向实际净买额，买额减卖额，单位亿元；月内加总及按实际记录数的日均，非额度余额估计，不代表全部外资或单一指数",
            "no_attribution": "美元篮子和人民币篮子币种及权重不同，不能机械相减得到纯中国冲击；买入和价格内生互动，不给因果贡献百分比"
        },
        "hypotheses": ["全球美元环境同时影响人民币双边报价和股票折现条件", "国内约束改善改变中国资产配置，但须有不同于纯美元变化的证据", "贸易结汇和资金交易可改变汇率，不能把全部升值归因为股票流入"],
        "information_clock": "月底描述与真正月度公告时间分开；比较净收益复用先前月末T+2开盘5/20日窗口。若补官方报告事件，按有日期无小时在当日收盘后可用，次日开盘进入，不回填统计期。",
        "horizons": [5,20], "cost": execution.COST,
        "new_candidates": 0, "new_full_accounts": 0, "new_prospective_tasks": 0,
        "orders_authorized": False, "goal_achieved": False,
        "stop": "没有当时可识别的独立配置机制和价格之后优势，不从六个月结果造状态阈值或拼收益。缺失原始来源保留缺失。"
    }, exclusive=True)
    save(OUT / "source_plan.json", SOURCES, exclusive=True)
    save(OUT / "local_source_manifest.json", [{"id": k, "path": rel(v), "sha256": digest(v)} for k,v in LOCAL.items()], exclusive=True)
    print("固定六个月的汇率构成与股票配置问题，保留既有失败，暂不新增候选。")


def collect():
    plan = read(OUT / "source_plan.json")
    folder = OUT / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    old = read(OUT / "source_manifest.json") if (OUT / "source_manifest.json").exists() else []
    done = {x["id"] for x in old}
    def fetch(item):
        record = dict(item, retrieved_at=now())
        try:
            response = requests.get(item["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=(10,25))
            response.raise_for_status()
            is_pdf = response.content.startswith(b"%PDF")
            path = folder / (item["id"] + (".pdf" if is_pdf else ".html"))
            path.write_bytes(response.content)
            if is_pdf:
                import pdfplumber
                with pdfplumber.open(path) as doc:
                    body = "\n".join(f"【PDF第{i+1}页】\n{p.extract_text() or ''}" for i,p in enumerate(doc.pages))
            else:
                response.encoding = response.apparent_encoding
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                body = soup.get_text("\n", strip=True)
            (folder / (item["id"] + ".txt")).write_text(body, encoding="utf-8")
            record.update(status="SAVED_PENDING_REVIEW", path=rel(path), sha256=digest(path), characters=len(body))
        except Exception as exc:
            record.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc)[:200])
        return record
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, [x for x in plan if x["id"] not in done]))
    save(OUT / "source_manifest.json", old + records)
    print(json.dumps([{k:v for k,v in x.items() if k in ["id","status","characters","error"]} for x in records], ensure_ascii=False))


def facts():
    """将已读原文的单位、当时版本与时点明确存下。"""
    months = [
        ("2022-10", "2022-10-31", "2022-11-01T08:30:00+08:00", 99.86, -1.16, 105.25, -1.23),
        ("2022-11", "2022-11-30", "2022-12-01T08:30:00+08:00", 97.00, -2.86, 102.26, -2.84),
        ("2022-12", "2022-12-30", "2023-01-03T08:30:00+08:00", 98.67, 1.72, 103.67, 1.38),
        ("2023-01", "2023-01-31", "2023-02-01T08:30:00+08:00", 99.83, 1.18, 104.97, 1.25),
        ("2023-02", "2023-02-28", "2023-03-01T08:30:00+08:00", 99.83, 0.00, 104.63, -0.32),
    ]
    baskets = []
    for month, date, published, level, change, bis, bis_change in months:
        source_id = "cfets_" + month.replace("-", "")
        body = (OUT / "sources" / f"{source_id}.txt").read_text(encoding="utf-8")
        compact = re.sub(r"\s+", "", body)
        assert f"{level:.2f}" in compact and f"{bis:.2f}" in compact, source_id
        baskets.append({"month": month, "date": date, "published_at": published,
                        "cfets": level, "cfets_change_pct": change, "bis": bis,
                        "bis_change_pct": bis_change, "source_id": source_id,
                        "source_path": rel(OUT / "sources" / f"{source_id}.html")})
    baskets.append({"month": "2023-03", "date": "2023-03-31", "cfets": None,
                    "cfets_change_pct": None, "bis": None, "bis_change_pct": None,
                    "status": "NOT_OBTAINED_IN_ORIGINAL_MONTHLY_ARCHIVE",
                    "note": "英文月度归档未列3月原文，不用第三方数字或4月读数倒推填充。"})
    save(OUT / "basket_facts.json", baskets)
    expectation = ROOT / "reports/research/510300_historical_index_expectation_anchor_v1/sources/econoday_571335.html"
    text = BeautifulSoup(expectation.read_bytes(), "html.parser").get_text("\n", strip=True)
    at = text.index("US Employment Situation for January")
    excerpt = text[at:at+1400]
    assert "185,000" in excerpt and "3.6%" in excerpt and "4.4%" in excerpt
    save(OUT / "january_jobs_expectation.json", {
        "source": rel(expectation), "sha256": digest(expectation), "source_archive_date": "2023-01-27",
        "source_url": "https://fidelity.econoday.com/byshoweventarticle?fid=571335&cust=fidelityFIPlus&year=2023&lid=0",
        "consensus_is": "周前Econoday共识，非公告前最后一分钟共识", "payrolls_expected": 185000,
        "payrolls_initial_actual": 517000, "payrolls_surprise": 332000,
        "unemployment_expected": 3.6, "unemployment_initial_actual": 3.4,
        "earnings_mom_expected": .3, "earnings_mom_initial_actual": .3,
        "earnings_yoy_expected": 4.4, "earnings_yoy_initial_actual": 4.4,
        "release_at": "2023-02-03T08:30:00-05:00", "known_shanghai": "2023-02-03T21:30:00+08:00",
        "original_release_url": "https://www.bls.gov/news.release/archives/empsit_02032023.pdf",
        "original_release_capture": rel(OUT / "sources/primary_web_text.txt"),
        "retrieval_limit": "直接HTTP请求403；已由浏览工具读取官方原PDF第1至5页，保留工具文字快照，未声称本地有原PDF字节。",
        "source_version": "2023-02-03初值；不替换成3月修订的504000或现行历史值",
        "composition": {"leisure_hospitality": 128000, "professional_business": 82000, "government": 74000,
                        "state_government_education_strike_return": 35000, "healthcare": 58000},
        "boundary": "新增就业包含复工及季调、年度基准变更；工资读数并未超过该周前共识；就业新闻不是单一纯通胀冲击。"
    })
    save(OUT / "source_review.json", {
        "reviewed_at": now(), "five_cfets_articles_checked": True, "cfets_pdf_pages_visually_checked": [1,2],
        "january_pdf_cfets_level": 99.83, "january_pdf_spot_cny": 6.7571,
        "january_pdf_status": "月后综述，仅作对照，不用它代替2月1日初次月报时点",
        "basket_weight_change": "2023-01-01 CFETS与SDR权重更新；同公告明确BIS篮子权重不变",
        "safe_release_date": "2023-02-15", "safe_january_stock_net_buy_usd_billion": 27.7,
        "safe_trade_receipts_surplus_usd_billion": 38.7,
        "safe_nonbank_receipts_surplus_usd_billion": 35.1,
        "safe_bank_fx_settlement_surplus_usd_billion": 2.5,
        "safe_not_additive": "贸易、股票和全部涉外收支范围交叠且时点不同，结售汇不是同一资金统计，不能相加或把差额当作股票资金。",
        "hkex_monthly_request_status": "六个月原页面当前指向的JS均404；北向仍为已保存供应商实际净买额，不标为已取得交易所原文。",
        "march_basket": "缺失保留，不阻断其他五个月及全六个月股票/汇率比较。"
    })
    print("保存五个月篮子原文、就业初值/周前预期与跨境收付口径；3月篮子保留缺失。")


def compute():
    monthly = pd.read_parquet(LOCAL["previous_months"])
    north = pd.read_parquet(LOCAL["northbound"])
    fx = pd.read_parquet(LOCAL["midpoint"])
    dxy = pd.read_parquet(LOCAL["dxy"])
    cnh = pd.read_parquet(LOCAL["cnh_context"])
    yields = pd.read_parquet(LOCAL["us_yields"])
    for frame in [north,fx,dxy,yields,cnh]:
        frame["date"] = pd.to_datetime(frame.date)
        assert not frame.date.duplicated().any()
        frame.sort_values("date", inplace=True)
    baskets = {x["month"]: x for x in read(OUT / "basket_facts.json")}
    rows, daily = [], []
    for m in monthly.itertuples():
        start, end = pd.Timestamp(m.start), pd.Timestamp(m.end)
        flow = north.loc[north.date.gt(start) & north.date.le(end)].copy()
        assert len(flow) > 0 and flow.usable_as_net_flow.all()
        error = flow.north_buy_100m_cny-flow.north_sell_100m_cny-flow.north_net_buy_100m_cny
        assert error.abs().max() < .001, "买卖额与净额不一致，不能直接累计"
        f0 = fx.loc[fx.date.le(start)].iloc[-1]
        f1 = fx.loc[fx.date.le(end)].iloc[-1]
        u0 = dxy.loc[dxy.date.le(start)].iloc[-1]
        u1 = dxy.loc[dxy.date.le(end)].iloc[-1]
        y0 = yields.loc[yields.date.le(start)].iloc[-1]
        y1 = yields.loc[yields.date.le(end)].iloc[-1]
        c0 = cnh.loc[cnh.date.le(start)].iloc[-1]
        c1 = cnh.loc[cnh.date.le(end)].iloc[-1]
        assert f0.date == start and f1.date == end
        assert (end-u1.date).days <= 3 and (start-u0.date).days <= 3
        b = baskets[m.label]
        row = {"month": m.label, "start": start, "end": end,
               "rmb_usd_midpoint_return": f0.first_release_value/f1.first_release_value-1,
               "midpoint_start": f0.first_release_value, "midpoint_end": f1.first_release_value,
               "rmb_usd_cnh_bid_context_return": c0.close/c1.close-1,
               "cnh_start_date": c0.date, "cnh_end_date": c1.date,
               "cnh_start_bar_end_utc": c0.bar_end_utc, "cnh_end_bar_end_utc": c1.bar_end_utc,
               "cnh_start": c0.close, "cnh_end": c1.close,
               "dxy_return": u1.dxy/u0.dxy-1, "dxy_start_date":u0.date, "dxy_end_date":u1.date,
               "cfets_change_pct": b["cfets_change_pct"], "bis_change_pct": b["bis_change_pct"],
               "cfets_level": b["cfets"], "cfets_published_at": b.get("published_at"),
               "north_net_buy_100m":flow.north_net_buy_100m_cny.sum(),
               "north_daily_mean_100m":flow.north_net_buy_100m_cny.mean(),
               "north_observed_days":len(flow), "north_positive_days":int(flow.north_net_buy_100m_cny.gt(0).sum()),
               "north_first_date":flow.date.min(), "north_last_date":flow.date.max(),
               "index_return":m.return_000300, "csi500_return":m.return_000905, "csi1000_return":m.return_000852,
               "next_entry_date":m.next_entry_date, "next5_net_return":m.next5_net_return,
               "next20_net_return":m.next20_net_return,
               "ust2_change_bp":(y1.nominal2-y0.nominal2)*100,
               "ust10_change_bp":(y1.nominal10-y0.nominal10)*100,
               "real10_change_bp":(y1.real10-y0.real10)*100,
               "inflation_compensation10_change_bp":(y1.inflation_compensation10-y0.inflation_compensation10)*100}
        rows.append(row)
        flow["month"] = m.label
        flow["within_month_cumulative_100m"] = flow.north_net_buy_100m_cny.cumsum()
        daily.append(flow)
    table = pd.DataFrame(rows)
    table.to_parquet(OUT / "monthly_currency_allocation.parquet",index=False)
    table.to_csv(OUT / "月度汇率与跨境配置.csv",index=False,encoding="utf-8-sig")
    pd.concat(daily,ignore_index=True).to_parquet(OUT / "northbound_daily.parquet",index=False)
    protocol = read(execution.OUT / "protocol.json")
    protocol["account_calendar"][1] = "2023-05-31"
    market,features,dividends,engine,clocks = execution.inputs(protocol)
    events = [
        {"event_id":"US_JOBS_JAN_INITIAL", "title":"美国1月就业初值", "known_at":"2023-02-03T21:30:00+08:00", "entry_date":"2023-02-06", "why":"周五21:30初次发布，下一A股交易日开盘"},
        {"event_id":"SAFE_JAN_ALLOCATION", "title":"外汇局1月股票配置回顾", "known_at":"2023-02-15T23:59:59+08:00", "entry_date":"2023-02-16", "why":"来源有日期无时刻，保守按当天结束后可用，次开盘"},
        {"event_id":"POWELL_MAR07", "title":"美国更高政策利率终点表态", "known_at":"2023-03-08T12:59:59+08:00", "entry_date":"2023-03-09", "why":"3月7日美国日期，未锁原文首次小时；按美国当日结束转换后再等一个完整A股收盘"},
        {"event_id":"FOMC_MAR22", "title":"银行冲击后的信用条件判断", "known_at":"2023-03-23T02:00:00+08:00", "entry_date":"2023-03-23", "why":"美国3月22日14:00EDT，上海23日02:00，晨间可读取"},
    ]
    windows=[]
    for ev in events:
        indices=market.index[market.date.eq(pd.Timestamp(ev["entry_date"]))]
        assert len(indices)==1
        ev["entry_idx"]=int(indices[0])
        assert pd.Timestamp(ev["known_at"]) < pd.Timestamp(ev["entry_date"]+"T09:30:00+08:00")
        for h in [5,20]:
            windows.append(execution.fixed_event_window(market,dividends,engine,ev,h))
    save(OUT / "event_clocks.json",events)
    save(OUT / "event_returns.json",windows)
    # 同步反应是事实，不拿当天反应回填开盘前判断。
    fx_responses=[]
    for start,end,name in [("2023-02-02","2023-02-03","就业初值前后美国收盘"),
                           ("2023-03-08","2023-03-13","银行冲击窗口，含其他消息")]:
        a,b=pd.Timestamp(start),pd.Timestamp(end)
        u0,u1=dxy.loc[dxy.date.eq(a)].iloc[0],dxy.loc[dxy.date.eq(b)].iloc[0]
        y0,y1=yields.loc[yields.date.eq(a)].iloc[0],yields.loc[yields.date.eq(b)].iloc[0]
        fx_responses.append({"label":name,"start":start,"end":end,"dxy_return":u1.dxy/u0.dxy-1,
            "ust2_change_bp":(y1.nominal2-y0.nominal2)*100,
            "ust10_change_bp":(y1.nominal10-y0.nominal10)*100,
            "real10_change_bp":(y1.real10-y0.real10)*100,
            "inflation_compensation10_change_bp":(y1.inflation_compensation10-y0.inflation_compensation10)*100,
            "causal_share":None,"purpose":"事后机制链描述，不是事件工具变量或新交易条件"})
    save(OUT / "external_reactions.json",fx_responses)
    jan=table.loc[table.month.eq("2023-01")].iloc[0]
    feb=table.loc[table.month.eq("2023-02")].iloc[0]
    save(OUT / "necessary_checks.json",{
        "six_months_retained":len(table)==6, "daily_northbound_rows":sum(len(x) for x in daily),
        "northbound_january_total_to_february_total_change":feb.north_net_buy_100m/jan.north_net_buy_100m-1,
        "northbound_january_daily_to_february_daily_change":feb.north_daily_mean_100m/jan.north_daily_mean_100m-1,
        "basket_months_obtained":5,"new_return_windows":len(windows),
        "window_all_filled":all(x["status"]=="两端成交" for x in windows),
        "net_return_is_full_account":False,"independent_validation":False,
        "source_coverage_limit":"记录天数按供应商实际可得，不把缺少记录擅自补零；没有交易所历史月原件逐月核全。"
    })
    print(table[["month","rmb_usd_midpoint_return","dxy_return","cfets_change_pct","north_net_buy_100m","north_daily_mean_100m","north_observed_days","index_return","ust2_change_bp","next20_net_return"]].round(5).to_string(index=False))
    print(json.dumps(fx_responses,ensure_ascii=False))
    print("新增事件窗口",[(x['event_id'],x['horizon'],round(x['net_return']*100,3)) for x in windows])


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    d = pd.read_parquet(OUT / "monthly_currency_allocation.parquet")
    x = np.arange(len(d))
    fig,axes=plt.subplots(2,1,figsize=(12,8.8),sharex=True)
    colors=["#256D85","#A3ADB3","#C09253"]
    for j,(field,label) in enumerate([("rmb_usd_cnh_bid_context_return","对美元：离岸BID日末"),
                                    ("rmb_usd_midpoint_return","对美元：官方中间价"),
                                    ("cfets_change_pct","对篮子：CFETS指数")]):
        values=d[field].values*(1 if field=="cfets_change_pct" else 100)
        bars=axes[0].bar(x+(j-1)*.25,values,width=.23,label=label,color=colors[j])
        for k,(bar,value) in enumerate(zip(bars,values)):
            if np.isfinite(value):
                axes[0].text(bar.get_x()+bar.get_width()/2,value+(.12 if value>=0 else -.12),f"{value:+.1f}",ha="center",va="bottom" if value>=0 else "top",fontsize=10)
    axes[0].text(5.25,.30,"原文缺失",ha="center",fontsize=9,color="#826D51")
    axes[0].set_ylim(-4.2,6.1)
    axes[0].set_ylabel("人民币升贬值 / %")
    axes[0].legend(frameon=False,ncol=3,fontsize=11,loc="upper left")
    axes[0].set_title("同一月份，不同报价与计价篮子需要分开",loc="left",fontsize=13,pad=14)
    y=d.north_daily_mean_100m.values
    bars=axes[1].bar(x,y,width=.6,color=["#A17165" if z<0 else "#256D85" for z in y])
    for k,(bar,value) in enumerate(zip(bars,y)):
        axes[1].text(k,value+(3 if value>=0 else -3),f"{value:+.2f}\n{d.north_observed_days.iloc[k]}个记录日",ha="center",va="bottom" if value>=0 else "top",fontsize=10)
    axes[1].set_ylim(-62,115)
    axes[1].set_ylabel("北向实际净买额日均 / 亿元")
    axes[1].set_title("1月到2月：净买入明显减弱，月度仍为正",loc="left",fontsize=13,pad=14)
    axes[1].set_xticks(x,d.month)
    for ax in axes:
        ax.axhline(0,lw=.8,color="#9DA9AF")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.12)
        ax.set_axisbelow(True)
    fig.suptitle("指数配置的外部环境，不能用一个“人民币涨跌”代替",fontsize=20,fontweight="bold",y=.98)
    fig.subplots_adjust(top=.88,bottom=.17,left=.10,right=.97,hspace=.37)
    fig.text(.10,.065,"离岸报价为Dukascopy UTC日BID，中间价为中国9:15，二者不同步；均非本账户外汇交易。\n篮子为当时官方口径，2023年权重变化另记；3月月报缺失，不填零。北向来自保存的供应商净买额。\n六个月固定比较；这些描述不构成买入条件，也未识别资金对沪深300的独立因果贡献。",fontsize=10,color="#56656D",linespacing=1.7)
    fig.savefig(OUT / "汇率组成_北向配置变化.png",dpi=160,facecolor="white")
    plt.close(fig)
    print("已生成汇率组成与北向配置图，等待目视核对。")


def publish():
    d=pd.read_parquet(OUT / "monthly_currency_allocation.parquet")
    events=read(OUT / "event_returns.json")
    reactions=read(OUT / "external_reactions.json")
    checks=read(OUT / "necessary_checks.json")
    rows=["| 月份 | 美元指数 | 人民币对美元：中间价 | 人民币对篮子 | 北向净买额/亿元 | 日均/亿元 | 沪深300 |", "|---|---:|---:|---:|---:|---:|---:|"]
    later=["| 月末 | 下一固定进入日 | 随后5日净收益 | 随后20日净收益 |", "|---|---|---:|---:|"]
    for r in d.itertuples():
        basket=f"{r.cfets_change_pct:+.2f}%" if pd.notna(r.cfets_change_pct) else "原月报未取得"
        rows.append(f"| {r.month} | {r.dxy_return:+.2%} | {r.rmb_usd_midpoint_return:+.2%} | {basket} | {r.north_net_buy_100m:+.2f} | {r.north_daily_mean_100m:+.2f} | {r.index_return:+.2%} |")
        later.append(f"| {r.month} | {r.next_entry_date:%Y-%m-%d} | {r.next5_net_return:+.2%} | {r.next20_net_return:+.2%} |")
    e={}
    for row in events:
        e.setdefault(row['event_id'],{})[row['horizon']]=row
    erows=["| 信息 | 本轮进入日 | 5日净收益 | 20日净收益 |", "|---|---|---:|---:|"]
    for key,v in e.items():
        erows.append(f"| {v[5]['title']} | {str(v[5]['entry_date'])[:10]} | {v[5]['net_return']:+.2%} | {v[20]['net_return']:+.2%} |")
    nov=d.loc[d.month.eq('2022-11')].iloc[0]
    jan=d.loc[d.month.eq('2023-01')].iloc[0]
    feb=d.loc[d.month.eq('2023-02')].iloc[0]
    jobs=reactions[0]
    discovery="六个月固定对照分清美元环境、人民币篮子与股票配置：2022年11月DXY跌5.00%、CFETS跌2.86%、中间价几乎不变；离岸报价升值不能与中间价混用。2023年2月篮子持平、DXY涨2.71%、北向月净买仍正但日均从88.31亿元降至4.63亿元。2月就业初值高于周前共识，2Y利率当日升21bp；3月银行冲击时利率反向大降，显示同一利率或美元方向有不同原因。未发现新交易优势。"
    next_question="已有美国CPI、FOMC及其国内背景延伸均失败；本轮新增的就业预期差暴露了外部增长信息这一不同来源。下一轮先核历史就业信息是否已被旧外部规则覆盖，再按固定发布日研究初值构成、周前预期和股票/利率的共同反应；保留全部正反例，不把本次2月赢家月份选作条件，不重救美元或北向方向规则。"
    text=f"""# 历史发现：汇率组成与指数配置

研究对象继续是沪深300整体。固定2022年10月至2023年3月六个自然月，复用原三指数与融资比较，补入官方人民币篮子公告、保存的美元与中间价、北向实际净买额，以及当时上游信息。新取得五份篮子月报和相关原文，3月原月报未取得，保留缺失。没有新增策略或完整账户，夏普1.2仍未达成。

本轮最实用的发现：**美元变弱、人民币对美元变强、人民币对一篮子货币变强、外资买股票，是不同的事实；其变化原因和时钟也不同。** 它们不能压缩成一条“汇率利好→买指数”。

{chr(10).join(rows)}

美元指数使用保存的Yahoo原始日收盘；对美元主列为官方9:15中间价，收益定义为上月每美元人民币报价除以本月报价减1，正数代表人民币升值。不是外汇可成交收益。两地月末时钟不同，月度表只作已发生的描述，不能据此声称同刻因果。北向为供应商保存的实际买额减卖额；无交易所历史原件时，不宣称覆盖与首次发布时间已获官方验证。

![汇率组成与配置](<{(OUT / '汇率组成_北向配置变化.png').as_posix()}>)

2022年11月尤其容易读错。美元指数下跌{abs(nov.dxy_return):.2%}，但CFETS人民币篮子指数下跌2.86%。本轮主列的人民币中间价从{nov.midpoint_start:.4f}到{nov.midpoint_end:.4f}，几乎没有变化。另核已有Dukascopy离岸BID日末报价，从{nov.cnh_start:.5f}到{nov.cnh_end:.5f}，人民币对美元升值{nov.rmb_usd_cnh_bid_context_return:.2%}。此前文字中的“对美元升值”只有明确即期报价口径才成立，不能转写为中间价升值。离岸日末与中国定盘时刻不同；本轮保留原中间价列，补充六个月报价对照解释差异，不替换原信号。[11月篮子原始公告](https://www.chinamoney.com.cn/english/bmkidxrud/20221201/2508579.html)

这一组合支持“全球美元下行有影响、人民币相对其他货币并未普遍增强”的解释；不能据此计算纯中国冲击的比例。DXY与CFETS的币种、权重和取价时点并不相同，二者不能机械相减。它也不能排除国内政策改善：中国股票仍可因活动约束改变而重新定价，即使人民币篮子当月下降。

2023年1月则不同：人民币中间价对美元升值{jan.rmb_usd_midpoint_return:.2%}，CFETS升1.18%，BIS篮子升1.25%，北向实际净买额为{jan.north_net_buy_100m:.2f}亿元，日均{jan.north_daily_mean_100m:.2f}亿元。这提供了外汇和股票配置共同改善的历史事实；具体动因仍包含国内活动约束放松、海外政策变化、贸易结汇等相互作用。2023年CFETS和SDR权重调整已在2022年12月30日公告，BIS篮子权重当时保持不变，因此三者都列明原版本。[1月原始公告](https://www.chinamoney.com.cn/english/bmkidxrud/20230201/2542741.html)、[当时权重公告](https://www.chinamoney.com.cn/chinese/zxpl/20221230/2525761.html)

外汇局2月15日公布的1月数据中，境外投资者净买入境内股票277亿美元，货物贸易涉外收支顺差387亿美元，全部非银行涉外收支顺差351亿美元，而银行结售汇顺差25亿美元。数字属于不同范围和交易环节，不能相加，也不能把股票买入直接当成等额即期结汇。北向1412.90亿元与全部外资277亿美元不应硬凑成同一个数字。官方当时将优化疫情防控、稳增长与海外紧缩外溢缓和列为背景，这是一种当时的解释，不是本轮已识别的因果占比。[外汇局当时原文](https://www.safe.gov.cn/safe/2023/0215/22326.html)

2月发生了什么变化？人民币中间价对美元贬值{abs(feb.rmb_usd_midpoint_return):.2%}，美元指数升{feb.dxy_return:.2%}，CFETS月末指数却与1月相同；BIS篮子仅跌0.32%。这更符合广泛美元走强这一部分解释，不支持把全部双边贬值直接叫作“中国风险失控”。北向当月仍净买{feb.north_net_buy_100m:.2f}亿元，不能写成净流出；但日均从1月的{jan.north_daily_mean_100m:.2f}亿元降到{feb.north_daily_mean_100m:.2f}亿元，按记录日调整后下降{abs(checks['northbound_january_daily_to_february_daily_change']):.2%}。沪深300当月跌2.10%、500和1000上涨，与北向净买入并存。这说明聚合资金方向不足以解释指数分化；尚未识别不同指数的资金归属，未证明哪些投资者买入了哪些篮子。[2月篮子原始公告](https://www.chinamoney.com.cn/english/bmkidxrud/20230301/2561719.html)

上游新证据来自2月3日的美国就业初值。1月27日Econoday周报的非农共识为18.5万人，BLS首次公布51.7万人，偏差+33.2万人；失业率3.4%低于该周报的3.6%，但时薪环比0.3%、同比4.4%都符合该份周前共识。就业增长有服务行业恢复和罢工后复工成分，原发布也说明年度基准与季调更新，不能把51.7万人一概解释成工资通胀超预期。使用的是初值，不是后来下修版本。该周报不是发布前最后一分钟共识。[周前预期原文](https://fidelity.econoday.com/byshoweventarticle?fid=571335&cust=fidelityFIPlus&year=2023&lid=0)、[BLS初次公告](https://www.bls.gov/news.release/archives/empsit_02032023.pdf)

2月2日至3日美国收盘，2年利率升{jobs['ust2_change_bp']:.0f}bp，10年名义利率升{jobs['ust10_change_bp']:.0f}bp，10年实际利率也升{jobs['real10_change_bp']:.0f}bp，同期限名义减实际的通胀补偿按公开两位小数读数没有变化；美元指数上涨{jobs['dxy_return']:.2%}。这与更强增长/劳动市场信息促使利率路径重估相符，反对“名义利率上涨必然全由通胀预期升高”的说法。它仍是整日联合反应，并非已分离其他消息的因果效应，通胀补偿也不等于纯预期通胀。

3月7日鲍威尔公开说明近期经济数据强于此前预期，政策利率终点可能高于之前判断。到3月12日，银行压力又促使美联储提供额外融资支持；3月22日声明明确讨论信用条件收紧对活动、就业和通胀的压力。3月8日至13日2年利率下降102bp、美元指数下降1.95%，这一窗口含银行事件、政策反应和其他信息。**同样是利率或美元下降，因通胀缓和与因金融压力上升而下降，不能映射成同一股票条件。** 3月7日及22日文字只能从各自公布后使用，未回填2月初。[3月7日当时表态](https://www.federalreserve.gov/newsevents/testimony/powell20230307a.htm)、[3月12日融资安排](https://www.federalreserve.gov/newsevents/pressreleases/monetary20230312a.htm)、[3月22日声明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20230322a.htm)

价格还剩多少，不能用统计月的上涨回答。沿用上一轮固定的月末T、T+1收盘可用假设、T+2开盘窗口，完整保留六个月：

{chr(10).join(later)}

这些收益完全复用旧窗口，并非六次新检验。本组六个月中，1月北向净买最强，随后20日净收益仍为-2.50%；仅凭月度资金确认无法证明剩余优势。也不能将10月弱资金后的正收益反向晋升为规则。

另将本轮涉及的四个不同信息节点按各自时钟测量，新增八个描述窗口：

{chr(10).join(erows)}

美国2月3日21:30上海时间就业数据对应2月6日开盘。外汇局资料只有2月15日日期，按该日结束后可用、16日开盘；3月7日证词首次小时未锁定，保守按美国当日结束转换后再等一个完整A股收盘，9日开盘。3月22日FOMC为纽约14:00EDT，即上海23日02:00，23日开盘。未按盈亏移动入场时钟。高低收益都包含后续信息，不是原事件的独立因果收益，也不是建议买入这些新闻。

窗口复用每边万4佣金、最低5元、千1滑点、整手、T+1和分红处理，按10万元名义投入测成本后回报。开盘数量按实际开盘测量用于事件比较，并非证明预先订单可按此精确数量执行；不是20万元完整策略账户，窗口可交叠，不能拼接年化夏普。既有六条北向二元规则、ASHR吸收规则与RMB残差来源限制均保持原状态。

本轮完成的是原因分解和价格时钟修正，没有新的可执行优势。下一步先核就业信息这一上游来源的旧研究覆盖，再在固定历史发布序列中看预期差的构成与利率、股票反应是否存在可辨别的阶段差异；不从本次六个月挑方向、阈值或最佳持有期。

原始资料在`sources`，逐月表为`月度汇率与跨境配置.csv`，逐日北向为`northbound_daily.parquet`，就业初值与周前预期为`january_jobs_expectation.json`，公布后收益为`event_returns.json`。五份CFETS月报直接保存官网字节；BLS原文直接下载403，保留浏览工具读取官方PDF的文字快照，不声称本地有完整原PDF。港交所六个月历史JS均404，北向供应商记录的这一来源限制保留。
"""
    REPORT.write_text(text,encoding="utf-8")
    save(OUT / "result.json",{
        "study_id":read(OUT / "protocol.json")["study_id"],"completed_at":now(),
        "status":"COMPLETED_HISTORICAL_CAUSE_DISCOVERY_NO_NEW_SIGNAL",
        "classification":"PROGRESS_INDEX_CURRENCY_COMPONENTS_AND_ALLOCATION_INFORMATION_CLOCK",
        "report":rel(REPORT),"figure":rel(OUT / "汇率组成_北向配置变化.png"),
        "discovery":discovery,"next_historical_question":next_question,
        "calendar_months":6,"original_basket_releases":5,"northbound_daily_rows":checks['daily_northbound_rows'],
        "new_event_windows":8,"reused_monthly_windows":12,
        "new_candidates":0,"new_full_accounts":0,"new_prospective_tasks":0,
        "net_sharpe":None,"net_cagr":None,"goal_achieved":False,"orders_authorized":False,
        "figure_visually_reviewed":False,"report_status":"PENDING_REVIEW",
        "causal_share_identified":False,"independent_validation":False,
        "limitations":["3月篮子月度原文缺失","历史交易所北向月原件不可得，沿用供应商实际净额","中间价/离岸/美元时钟不同","个别事件首次小时未确定","新描述样本少且历史已被观察"]
    })
    print(discovery)


def finalize():
    result=read(OUT / "result.json")
    result.update(figure_visually_reviewed=True,report_status="REVIEWED",updated_at=now())
    save(OUT / "result.json",result)
    sources=read(OUT / "source_manifest.json")
    reviewed={"cfets_202210","cfets_202211","cfets_202212","cfets_202301","cfets_202302",
              "cfets_basket_2023","cfets_january_review","safe_january","powell_20230307",
              "fomc_20230322","fed_btfp_20230312"}
    for source in sources:
        if source['id'] in reviewed and source['status']=="SAVED_PENDING_REVIEW":
            source.update(status="CONTENT_REVIEWED",reviewed_at=now())
    save(OUT / "source_manifest.json",sources)
    save(OUT / "local_source_manifest.json",[{"id":k,"path":rel(v),"sha256":digest(v)} for k,v in LOCAL.items()])
    save(OUT / "research_receipt.json",{"updated_at":now(),"script_sha256":digest(Path(__file__)),
        "outputs":{rel(p):digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='research_receipt.json'},
        "previous_turn":"PROGRESS","this_turn":"PROGRESS","goal_achieved":False})
    p=ROOT / "config/510300_historical_cause_discovery_v1.json"
    config=read(p)
    config.update(latest_completed_study=rel(OUT / "result.json"),latest_report=rel(REPORT),
        current_study=rel(OUT / "protocol.json"),latest_result_summary=result['discovery'],
        next_historical_question=result['next_historical_question'],updated_at=now(),goal_achieved=False)
    save(p,config)
    p=ROOT / "config/510300_existing_data_training_mandate_v1.json"
    config=read(p)
    config.update(latest_historical_index_currency_allocation=rel(OUT / "result.json"),
        latest_historical_report=rel(REPORT),latest_historical_diagnostic_at=now(),
        current_driver_continuation_classification=result['classification'],current_driver_consecutive_blocked_goal_turns=0,
        next_research_question=result['next_historical_question'],local_goal_work_status="ACTIVE_HISTORICAL_ONLY",goal_achieved=False)
    save(p,config)
    print("汇率组成与指数配置研究已完成；完整账户目标仍未达成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指数汇率与配置历史研究")
    parser.add_argument("action", choices=["prepare", "collect", "facts", "compute", "plot", "publish", "finalize"])
    globals()[parser.parse_args().action]()
