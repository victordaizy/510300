"""固定日历下的指数共同行情、行业抵消和之后可成交收益。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.historical_index_earnings_population_v1 import clean
from research import historical_index_reopening_constraints_v1 as execution

OUT = ROOT / "reports/research/510300_historical_index_sector_repricing_v1"
SECTORS = {"000908": "能源", "000909": "原材料", "000910": "工业", "000911": "可选消费", "000912": "主要消费", "000913": "医药卫生", "000914": "金融地产", "000915": "信息技术", "000916": "通信服务", "000917": "公用事业"}
URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
INPUTS = {
    "old_index": "reports/research/510300_macro_earnings_pricing_bridge_v5/inputs/index_price.parquet",
    "old_pe": "reports/research/510300_macro_earnings_pricing_bridge_v5/inputs/pe.parquet",
    "funding": "data/raw/macro/510300_macro_stress_2015_v2/fdr007_daily_2015_2026.parquet",
    "cgb": "data/raw/macro/china_government_bond_yields_daily.parquet",
    "industry": "data/features/000300_industry_driver_attribution_daily_v1_3.parquet",
    "industry_daily": "data/features/000300_index_driver_summary_daily_v1_3.parquet",
    "market": "reports/research/510300_factor96_t11_date_proxy_account_v1/inputs/market.parquet",
    "dividends": "reports/research/510300_factor96_t11_date_proxy_account_v1/inputs/dividends.csv",
}
POLICY_SOURCES = [
    {"id": "nafmii_20221108", "title": "民营企业债券融资支持工具延期扩容", "url": "https://www.nafmii.org.cn/xhdt/202211/t20221108_311365.html", "page_date": "2022-11-08"},
    {"id": "nafmii_20221123", "title": "首批民营房企信用增进函", "url": "https://www.nafmii.org.cn/xhdt/202211/t20221123_311524.html", "page_date": "2022-11-23"},
    {"id": "gov_20221124", "title": "金融支持房地产市场通知的政府网转发", "url": "https://app.www.gov.cn/govdata/gov/202211/24/494578/article.html", "page_date": "2022-11-24"},
    {"id": "csrc_20221128", "title": "房地产股权融资五项措施", "url": "https://www.csrc.gov.cn/csrc/c100028/c6763083/content.shtml", "page_date": "2022-11-28"},
]


def now():
    return datetime.now().astimezone().isoformat()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("保留已有方案，不覆盖。")
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {
        "study_id": "510300_HISTORICAL_INDEX_SECTOR_REPRICING_V1", "created_at": now(),
        "previous_goal_turn": "PROGRESS_FIXED_INDEX_POPULATIONS_AND_NOMINAL_PROFIT_TRANSMISSION",
        "question": "同一指数阶段的整体涨跌、行业抵消与资金价格怎样并存；广泛同涨之后是否仍有可成交收益？",
        "calendar": ["2022-01", "2023-03"], "base_date": "2021-12-31", "sectors": SECTORS,
        "coverage": "固定十五个完整自然月，全部十条沪深300一级行业价格指数；不挑赢家、转折点或当月最强行业。",
        "deduplication": {"V26": "已做104个月货币发布前后价格/PE恒等式；本轮仅将固定月度恒等式作背景，不称为新增收益机制。", "driver_episode": "M1/T1驱动周期候选历史拒绝封存；不调整其核心组、时长、强度。", "new_increment": "新增官方十条行业指数完整历史，与全部月份和既有资金价格相接；下一开盘后的固定事件窗口另列。"},
        "official_data": "每指数一次有界原始请求；缺日期或接口无响应就保留缺失，不替换为别的行业、不回填。当前取得的历史页面不等于当年不可变首版。",
        "descriptive_states": "十行业月末相对上月末价格严格全部上涨、严格全部下跌、其余混合三类；仅描述，不按结果挑类别形成新账户。",
        "monthly_statistics": "分别保存各行业简单价格回报、同向行业数、行业等权均值与离散度、000300同日价格及PE变化。PE隐含分母不称EPS或风险溢价。",
        "industry_cancellation": "复用原CITIC快照权重归因，只对全部交易日有效的月份链式连接行业贡献：当日贡献乘当日前累计合成财富，再跨日相加。缺日月份为NO_VIEW，未知行业单列。这是原快照组合近似，不是官方指数涨幅归因。",
        "timing": "月份结束后下一实际交易日开盘观察，固定持有5/20个开盘间隔；采用原压力成本、整手与分红；不把形成月份的涨幅算进随后收益。",
        "event_end_coverage": "行情只截至2023-05-31，以覆盖最后一个月末的固定退出；不延长信号形成区间。",
        "funding": "原FDR007当月均值和月末值分别保存；10年国债同月变化另列，不能把这两个利率都叫作政策利率或股票风险溢价。",
        "discovery_status": "已见指数大致走势及相邻旧研究；本轮为知情历史发现，无独立验证，不拟合方向、阈值或持有期。",
        "horizons": [5,20], "new_full_accounts": 0, "new_models": 0, "new_prospective_tasks": 0,
        "orders": 0, "goal_achieved": False,
    }
    save(OUT / "protocol.json", protocol)
    save(OUT / "freeze.json", {"created_at": now(), "protocol_sha256": digest(OUT / "protocol.json")})
    save(OUT / "input_receipts.json", [{"name": name, "path": path, "sha256": digest(ROOT / path)} for name, path in INPUTS.items()])
    print("已固定十五个月、十行业、三类描述状态及5/20日窗口。")


def fetch(code):
    params = {"indexCode": code, "startDate": "20211231", "endDate": "20230331"}
    record = {"index_code": code, "url": URL, "params": params, "retrieved_at": now()}
    try:
        response = requests.get(URL, params=params, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.csindex.com.cn/"}, timeout=(10, 30))
        source = OUT / "sources" / (code + ".json")
        source.parent.mkdir(exist_ok=True)
        source.write_bytes(response.content)
        record.update(http_status=response.status_code, path=rel(source), sha256=digest(source), bytes=len(response.content))
        response.raise_for_status()
        data = response.json()["data"]
        assert isinstance(data, list) and len(data) > 0
        record.update(status="RETRIEVED", rows=len(data), fields=list(data[0]))
    except (requests.RequestException, ValueError, AssertionError, KeyError) as error:
        record.update(status="FAILED", error=str(error)[:400])
    return record


def collect():
    if (OUT / "source_manifest.json").exists():
        raise RuntimeError("已有请求结果，先看结果，不重复采集。")
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, ["000300"] + list(SECTORS)))
    save(OUT / "source_manifest.json", results)
    print(json.dumps({"取得": sum(x["status"] == "RETRIEVED" for x in results), "请求": len(results), "记录": [{k:v for k,v in x.items() if k in ["index_code", "rows", "status", "error"]} for x in results]}, ensure_ascii=False))


def compute():
    protocol = read(OUT / "protocol.json")
    assert digest(OUT / "protocol.json") == read(OUT / "freeze.json")["protocol_sha256"]
    frames = []
    for item in read(OUT / "source_manifest.json"):
        if item["status"] != "RETRIEVED":
            raise RuntimeError("固定来源未全部取得，保留缺失，不换样本。")
        d = pd.DataFrame(read(ROOT / item["path"])["data"])
        assert d.indexCode.eq(item["index_code"]).all()
        d["date"] = pd.to_datetime(d.tradeDate.astype(str), format="%Y%m%d")
        d = d.sort_values("date").reset_index(drop=True)
        assert d.date.is_unique
        for field in ["open", "close", "peg"]:
            d[field] = pd.to_numeric(d[field], errors="coerce")
        assert d.close.gt(0).all() and d.open.gt(0).all()
        d["source_sha256"] = item["sha256"]
        frames.append(d)
    daily = pd.concat(frames, ignore_index=True)
    daily.to_parquet(OUT / "official_index_daily.parquet", index=False)
    prices = daily.pivot(index="date", columns="indexCode", values="close")
    assert prices.shape == (302,11) and prices.notna().all().all()
    assert prices.index[0] == pd.Timestamp("2021-12-31") and prices.index[-1] == pd.Timestamp("2023-03-31")
    index = daily[daily.indexCode.eq("000300")].set_index("date")
    old = pd.read_parquet(ROOT / INPUTS["old_index"])
    old["date"] = pd.to_datetime(old.date)
    pairs = index[["close"]].join(old.set_index("date")[["close"]], how="left", rsuffix="_old")
    assert pairs.close_old.notna().all()
    price_diff = float((pairs.close-pairs.close_old).abs().max())
    assert price_diff <= .00501
    old_pe = pd.read_parquet(ROOT / INPUTS["old_pe"]).set_index("observation_date")
    pe_pairs = index[["peg"]].join(old_pe[["original_pe_official"]],how="left")
    pe_diff = float((pe_pairs.peg-pe_pairs.original_pe_official).abs().max())
    assert pe_pairs.original_pe_official.notna().all() and pe_diff < 1e-10
    funding = pd.read_parquet(ROOT / INPUTS["funding"])
    funding["date"] = pd.to_datetime(funding.date)
    funding["available_utc"] = pd.to_datetime(funding.available_at, utc=True)
    funding = funding.sort_values("date")
    cgb = pd.read_parquet(ROOT / INPUTS["cgb"])
    cgb["date"] = pd.to_datetime(cgb.date)
    cgb = cgb.sort_values("date")
    old_account_protocol = read(execution.OUT / "protocol.json")
    old_account_protocol["account_calendar"][1] = "2023-05-31"
    market, features, dividends, engine, _ = execution.inputs(old_account_protocol)
    rows, sectors, events = [], [], []
    previous = pd.Timestamp("2021-12-31")
    last_funding_mean = float(funding[funding.date.dt.to_period("M").eq(pd.Period("2021-12"))].first_release_value.mean())
    previous_cgb = float(cgb.loc[cgb.date.le(previous), "cgb_10y"].iloc[-1])
    for period in pd.period_range("2022-01", "2023-03",freq="M"):
        month_end = period.end_time.normalize()
        date = prices.index[prices.index.to_period("M") == period][-1]
        return_vector = prices.loc[date]/prices.loc[previous]-1
        sr = return_vector.loc[list(SECTORS)]
        positive, negative = int(sr.gt(0).sum()), int(sr.lt(0).sum())
        state = "ALL_UP" if positive==10 else "ALL_DOWN" if negative==10 else "MIXED"
        for code, value in sr.items():
            sectors.append({"month": str(period), "index_code": code, "industry": SECTORS[code], "start_date": previous, "end_date": date, "price_return": value, "start_close": prices.at[previous,code], "end_close": prices.at[date,code]})
        funding_month = funding[funding.date.dt.to_period("M").eq(period)]
        cutoff = (month_end+pd.Timedelta(hours=23,minutes=59,seconds=59)).tz_localize("Asia/Shanghai")
        assert funding_month.available_utc.le(cutoff).all()
        funding_mean = float(funding_month.first_release_value.mean())
        cgb_rows = cgb[cgb.date.dt.to_period("M").eq(period)]
        cgb_last = cgb_rows.iloc[-1]
        pe0, pe1 = float(index.at[previous,"peg"]), float(index.at[date,"peg"])
        log_price = float(np.log(prices.at[date,"000300"]/prices.at[previous,"000300"])*100)
        log_pe = float(np.log(pe1/pe0)*100)
        implied_log = float(np.log((prices.at[date,"000300"]/pe1)/(prices.at[previous,"000300"]/pe0))*100)
        assert abs(log_price-log_pe-implied_log)<1e-10
        entry_idx = int(np.flatnonzero(market.date.gt(month_end))[0])
        clock={"event_id": "MONTH_END_"+str(period), "title": str(period)+"月末固定观察", "entry_idx":entry_idx}
        observed={h: execution.fixed_event_window(market,dividends,engine,clock,h) for h in protocol["horizons"]}
        events += list(observed.values())
        for event in observed.values():
            assert event["status"] == "两端成交"
        row={"month":str(period),"prior_close_date":previous,"close_date":date,"month_end":month_end,"state":state,"positive_sectors":positive,"negative_sectors":negative,"zero_sectors":10-positive-negative,
             "index_price_return":return_vector["000300"],"sector_equal_mean_price_return":sr.mean(),"sector_median_price_return":sr.median(),"sector_price_return_std":sr.std(ddof=0),
             "sector_min_price_return":sr.min(),"sector_max_price_return":sr.max(),"index_log_price_pp":log_price,"index_log_pe_pp":log_pe,"index_log_implied_denominator_pp":implied_log,"pe_start":pe0,"pe_end":pe1,
             "fdr007_mean_pct":funding_mean,"fdr007_mean_change_bp":(funding_mean-last_funding_mean)*100,"fdr007_last_pct":float(funding_month.iloc[-1].first_release_value),"fdr007_last_date":funding_month.iloc[-1].date,"funding_days":len(funding_month),
             "cgb10_last_pct":float(cgb_last.cgb_10y),"cgb10_change_bp":(float(cgb_last.cgb_10y)-previous_cgb)*100,"cgb10_last_date":cgb_last.date,
             "next_entry_date":market.date.iloc[entry_idx],"next_entry_open":market.open.iloc[entry_idx],"prior_etf_close":float(market.loc[market.date.eq(date),"close"].iloc[0])}
        for horizon,event in observed.items():
            row[f"next{horizon}_net_return"]=event["net_return"]
            row[f"next{horizon}_gross_return"]=event["gross_return"]
            row[f"next{horizon}_exit_date"]=event["exit_date"]
        rows.append(row)
        previous,date_last_funding_mean,previous_cgb=date,funding_mean,float(cgb_last.cgb_10y)
        last_funding_mean=date_last_funding_mean
    monthly=pd.DataFrame(rows)
    monthly.to_parquet(OUT/"monthly_comparison.parquet",index=False)
    monthly.to_csv(OUT/"monthly_comparison.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(sectors).to_csv(OUT/"sector_monthly_returns.csv",index=False,encoding="utf-8-sig")
    save(OUT/"event_returns.json",events)
    groups=[]
    for state in ["ALL_UP","ALL_DOWN","MIXED"]:
        q=monthly[monthly.state.eq(state)]
        groups.append({"state":state,"months":q.month.tolist(),"count":len(q),"formation_mean_index_price_return":q.index_price_return.mean(),"next5_mean_net_return":q.next5_net_return.mean(),"next20_mean_net_return":q.next20_net_return.mean(),"next20_median_net_return":q.next20_net_return.median(),"next20_positive_count":int(q.next20_net_return.gt(0).sum()),"next20_min_net_return":q.next20_net_return.min(),"next20_max_net_return":q.next20_net_return.max(),"is_full_account":False})
    save(OUT/"descriptive_groups.json",groups)
    save(OUT/"price_checks.json",{"official_series":11,"days_each":302,"official_rows":len(daily),"month_rows":len(monthly),"sector_month_rows":len(sectors),"fixed_event_windows":len(events),"official_old_close_max_difference":price_diff,"official_old_pe_max_difference":pe_diff,"new_accounts":0})
    print(monthly[["month","positive_sectors","index_price_return","fdr007_mean_change_bp","cgb10_change_bp","next20_net_return"]].to_string(index=False))


def cancellation():
    daily=pd.read_parquet(ROOT/INPUTS["industry_daily"])
    industries=pd.read_parquet(ROOT/INPUTS["industry"])
    for data in [daily,industries]:
        data["date"]=pd.to_datetime(data.date)
    summary,contributions=[],[]
    for period in pd.period_range("2022-01","2023-03",freq="M"):
        q=daily[daily.date.dt.to_period("M").eq(period)].sort_values("date")
        base={"month":str(period),"calendar_days":len(q),"valid_days":int(q.valid_for_attribution.sum()),"no_view_days":int((~q.valid_for_attribution).sum())}
        if not q.valid_for_attribution.all():
            summary.append({**base,"status":"NO_VIEW_INCOMPLETE_ORIGINAL_MONTH","approximate_return":None,"unavailable_dates":q.loc[~q.valid_for_attribution,"date"].dt.strftime("%Y-%m-%d").tolist()})
            continue
        wealth=1.
        totals={}
        for row in q.itertuples():
            sub=industries[industries.date.eq(row.date)]
            assert sub.weighted_return_contribution_1d.notna().all()
            for entry in sub.itertuples():
                totals[entry.industry_l1]=totals.get(entry.industry_l1,0.)+wealth*entry.weighted_return_contribution_1d
            totals["未归属行业"]=totals.get("未归属行业",0.)+wealth*row.unattributed_contribution_1d
            wealth*=1+row.snapshot_weighted_constituent_return_1d
        aggregate=sum(totals.values())
        assert abs(aggregate-(wealth-1))<1e-10
        plus=sum(max(v,0.) for v in totals.values())
        minus=sum(min(v,0.) for v in totals.values())
        cancellation_ratio=1-abs(aggregate)/(plus-minus) if plus-minus>0 else 0.
        summary.append({**base,"status":"ORIGINAL_SNAPSHOT_PORTFOLIO_APPROXIMATION","approximate_return":wealth-1,"positive_contribution":plus,"negative_contribution":minus,"cancellation_ratio":cancellation_ratio,"positive_industries":sum(v>0 for k,v in totals.items() if k!="未归属行业"),"industry_count":len(totals)-1,"identity_error":aggregate-(wealth-1),"official_index_attribution":False})
        contributions += [{"month":str(period),"industry":k,"linked_contribution":v} for k,v in totals.items()]
    save(OUT/"industry_cancellation.json",summary)
    save(OUT/"industry_linked_contributions.json",contributions)
    print(json.dumps({"完整近似月份":sum(x['no_view_days']==0 for x in summary),"缺日月份":[x['month'] for x in summary if x['no_view_days']>0]},ensure_ascii=False))


def policy_sources():
    if (OUT / "policy_source_manifest.json").exists():
        raise RuntimeError("已有政策来源记录，不重复采集。")
    save(OUT / "policy_source_plan.json", {"created_at": now(), "sources": POLICY_SOURCES, "status": "POST_RESULT_EXPLANATORY_CONTEXT", "selection": "在十五个月价格结果之后补充2022年11月融资约束原文；不是盲测，也不据此新增买卖规则。", "clock": "仅证明所取得网页最迟当日可见；没有核实更早媒体披露，不能将网页日期称首次消息，也不使用文件落款倒填。"})
    def fetch_policy(spec):
        record = {**spec, "retrieved_at": now()}
        try:
            response = requests.get(spec["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=(10, 30))
            path = OUT / "sources" / (spec["id"] + ".html")
            path.write_bytes(response.content)
            record.update(http_status=response.status_code, path=rel(path), sha256=digest(path), bytes=len(response.content))
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser", from_encoding="utf-8")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text_path = path.with_suffix(".txt")
            text_path.write_text(soup.get_text("\n", strip=True), encoding="utf-8")
            record.update(status="RETRIEVED", text_path=rel(text_path), meta=[dict(tag.attrs) for tag in soup.select("meta") if any(k.lower() in ("name", "property") for k in tag.attrs)])
        except requests.RequestException as error:
            record.update(status="FAILED", error=str(error)[:300])
        return record
    with ThreadPoolExecutor(max_workers=3) as pool:
        result = list(pool.map(fetch_policy, POLICY_SOURCES))
    save(OUT / "policy_source_manifest.json", result)
    print(json.dumps([{k:v for k,v in x.items() if k in ["id", "status", "bytes", "error", "meta"]} for x in result], ensure_ascii=False))


def policy_facts():
    sources = {x["id"]: x for x in read(OUT / "policy_source_manifest.json")}
    anchors = {
        "nafmii_20221108": ["2500亿元", "再贷款", "担保增信", "信用风险缓释凭证", "直接购买债券"],
        "nafmii_20221123": ["2022年11月23日", "信用增进函", "20亿元", "15亿元", "12亿元", "拟首批"],
        "gov_20221124": ["2022-11-24", "多展期1年", "封闭运行", "后进先出", "可予免责"],
        "csrc_20221128": ["日期：2022-11-28", "即日起施行", "恢复上市房企", "再融资", "不久前", "不能用于拿地拍地"],
    }
    for source_id, terms in anchors.items():
        source = sources[source_id]
        assert source["status"] == "RETRIEVED"
        assert digest(ROOT / source["path"]) == source["sha256"]
        body = "".join((ROOT / source["text_path"]).read_text(encoding="utf-8").split())
        for term in terms:
            assert term in body, (source_id, term)
    specs = [
        {"source_id": "nafmii_20221108", "nominal_public_at": "2022-11-08T17:43:00+08:00", "precision": "网页PubDate元数据", "first_daily_open_after_cutoff": "2022-11-09", "stage": "支持工具扩容", "expected_supported_bond_financing_yi": 2500, "actual_disbursed_yi": None, "fact": "再贷款支持的工具通过增信、风险缓释及直接购债等方式支持民企发债，范围包含房企；2500亿元是预期支持融资规模。", "changed_constraint": "发行人的再融资条件以及承接债券的信用风险分担。", "not_proven": "没有证明2500亿元已投放、流入股票，也没有确认覆盖所有困难房企。"},
        {"source_id": "nafmii_20221123", "nominal_public_at": "2022-11-23T12:44:00+08:00", "precision": "网页PubDate元数据", "first_daily_open_after_cutoff": "2022-11-24", "stage": "首批增信函", "planned_supported_bond_financing_yi": 20+15+12, "actual_disbursed_yi": None, "fact": "三家民营房企取得信用增进函，拟支持中期票据合计47亿元。", "changed_constraint": "从工具框架推进到具体发行支持环节。", "not_proven": "增信函与拟发行规模不等于本页已经证明发行结算、净融资或最终销售回款。"},
        {"source_id": "gov_20221124", "nominal_public_at": "2022-11-24T23:59:59+08:00", "precision": "本次取得的政府网转发日期上界，未认定首次消息", "first_daily_open_after_cutoff": "2022-11-25", "stage": "贷款、展期和项目保障条款", "fact": "通知涉及合条件存量融资展期、项目资金管理、保交楼借款及配套融资，并安排新增资金偿还优先次序和符合条件的尽职免责。", "changed_constraint": "短期到期压力、新资金受偿安排及金融机构执行顾虑。", "not_proven": "没有证明此前未公开，也不能把监管风险分类安排等同于实际信用损失消失；没有将落款日用作本研究消息日。"},
        {"source_id": "csrc_20221128", "nominal_public_at": "2022-11-28T23:59:59+08:00", "precision": "可见正文日期上界", "first_daily_open_after_cutoff": "2022-11-29", "stage": "股权融资条件调整", "fact": "恢复合条件涉房重组及再融资，调整境外融资、REITs和私募股权相关安排；配套融资用途受限制。", "changed_constraint": "补充权益资本和盘活存量资产的渠道。", "not_proven": "政策已在此前论坛表态，不能称全部意外；允许融资不等于所有指数股东即时增厚收益，仍可能有稀释及发行供给。", "metadata_note": "PubDate是2026-08-01 21:54:47，且others明示页面生成时间相同。保留冲突，采用正文2022-11-28，仅到日期，未制造历史分钟。"},
    ]
    for spec in specs:
        source = sources[spec["source_id"]]
        spec.update(url=source["url"], source_path=source["path"], source_sha256=source["sha256"], first_market_disclosure_verified=False, immutable_historical_vintage=False, used_as_trading_trigger=False)
    competing_path = ROOT / "reports/research/510300_historical_index_external_reopening_v1/mechanism_cards.json"
    competing = read(competing_path)["nov10_information_clock"]
    save(OUT / "policy_facts.json", {"status": "POST_RESULT_EXPLANATION_NO_NEW_TRADING_FILTER", "facts": specs, "competing_information_clock": competing, "competing_clock_source": rel(competing_path), "competing_clock_sha256": digest(competing_path), "expectation_gap": "未取得这四项融资政策的事前定量共识；价格和政策同月变化不识别意外成分或因果贡献。", "actual_net_financing_increment": "NOT_IDENTIFIED", "stock_risk_premium_change": "NOT_IDENTIFIED"})
    print("四份原文要点及公开日期差异已保存；未新增交易触发。")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False, "font.size": 10})
    monthly = pd.read_parquet(OUT / "monthly_comparison.parquet")
    sectors = pd.read_csv(OUT / "sector_monthly_returns.csv", dtype={"index_code": str})
    table = sectors.pivot(index="industry", columns="month", values="price_return").reindex(index=list(SECTORS.values()), columns=monthly.month)
    offsets = pd.DataFrame(read(OUT / "industry_cancellation.json")).set_index("month").reindex(monthly.month)
    fig, axes = plt.subplots(3, 1, figsize=(15, 12.4), gridspec_kw={"height_ratios": [2.4, 1.25, 1.15]}, layout="constrained")
    values = table.to_numpy()*100
    limit = np.ceil(np.nanmax(np.abs(values))/5)*5
    heat = axes[0].imshow(values, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            axes[0].text(j, i, f"{values[i,j]:+.1f}", ha="center", va="center", fontsize=8.2, color="white" if abs(values[i,j]) > .65*limit else "#182432")
    axes[0].set_xticks(range(len(monthly)), monthly.month.str[2:], rotation=0)
    axes[0].set_yticks(range(len(table)), table.index)
    axes[0].set_title("① 官方沪深300十行业：每个固定月份的价格回报（%）", loc="left", pad=14, fontsize=13, weight="bold")
    fig.colorbar(heat, ax=axes[0], fraction=.025, pad=.02, label="价格回报 / %")
    x = np.arange(len(monthly))
    axes[1].bar(x-.19, monthly.index_price_return*100, width=.36, color="#315e85", label="形成月份：沪深300价格回报")
    axes[1].bar(x+.19, monthly.next20_net_return*100, width=.36, color="#d18238", label="月末之后：ETF下一开盘起20个开盘间隔净回报")
    axes[1].axhline(0, color="#555555", linewidth=.8)
    axes[1].set_xticks(x, monthly.month.str[2:])
    axes[1].yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axes[1].set_title("② 形成期上涨与事后可成交收益分别计算；两列的标的及区间不同", loc="left", pad=12, fontsize=12, weight="bold")
    axes[1].legend(loc="lower left", ncols=2, fontsize=9, frameon=False)
    axes[1].set_ylim(min(monthly.index_price_return.min(), monthly.next20_net_return.min())*100-5, max(monthly.index_price_return.max(), monthly.next20_net_return.max())*100+2)
    axes[2].bar(x, offsets.positive_contribution.to_numpy()*100, width=.6, color="#ba554f", label="正贡献行业合计")
    axes[2].bar(x, offsets.negative_contribution.to_numpy()*100, width=.6, color="#4d7980", label="负贡献行业合计")
    axes[2].plot(x, offsets.approximate_return.to_numpy()*100, color="#252d3d", marker="o", markersize=4, label="合成组合月回报")
    for i, value in enumerate(offsets.approximate_return):
        if pd.isna(value):
            axes[2].text(i, 0, "缺日", ha="center", va="center", fontsize=9, color="#595959")
    axes[2].axhline(0, color="#555555", linewidth=.8)
    axes[2].set_xticks(x, monthly.month.str[2:])
    axes[2].set_ylabel("回报 / 贡献（百分点）")
    axes[2].set_title("③ 原快照组合的行业抵消近似：中信行业口径，与上图十行业分类不同", loc="left", pad=12, fontsize=12, weight="bold")
    axes[2].legend(loc="lower left", ncols=3, fontsize=9, frameon=False)
    axes[2].set_ylim(-13, 12)
    for ax in axes[1:]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e7eb", linewidth=.5)
        ax.set_axisbelow(True)
    fig.suptitle("指数整体的共同变化与行业抵消｜2022年1月—2023年3月", fontsize=18, weight="bold")
    fig.supxlabel("历史描述，未识别新增交易优势。下图是快照组合近似，非官方指数贡献；5/20日窗口可重叠，不能拼为账户。", fontsize=10, color="#444444")
    path = OUT / "指数行业_共同变化与抵消.png"
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)
    print("已生成全部十五个月的行业价格、后续窗口与抵消图。")


def publish():
    monthly = pd.read_parquet(OUT / "monthly_comparison.parquet")
    sectors = pd.read_csv(OUT / "sector_monthly_returns.csv", dtype={"index_code": str})
    offsets = pd.DataFrame(read(OUT / "industry_cancellation.json")).set_index("month")
    contributions = pd.DataFrame(read(OUT / "industry_linked_contributions.json"))
    groups = read(OUT / "descriptive_groups.json")
    policies = read(OUT / "policy_facts.json")
    nov = monthly.set_index("month").loc["2022-11"]
    may = monthly.set_index("month").loc["2022-05"]
    march = monthly.set_index("month").loc["2023-03"]
    aligned = offsets[["approximate_return"]].join(monthly.set_index("month")[["index_price_return"]])
    differences = (aligned.approximate_return-aligned.index_price_return).abs().dropna()
    financial = float(contributions[(contributions.month=="2022-11") & contributions.industry.isin(["银行", "非银行金融", "房地产"])].linked_contribution.sum())
    nov_sectors = sectors[sectors.month.eq("2022-11")].set_index("industry")
    may_sectors = sectors[sectors.month.eq("2022-05")].set_index("industry")
    table = ["| 形成月份 | 上涨行业/10 | 沪深300当月价格 | FDR007月均变化/bp | 10年国债月末变化/bp | 下一开盘起20间隔净收益 |", "|---|---:|---:|---:|---:|---:|"]
    for row in monthly.itertuples():
        table.append(f"| {row.month} | {row.positive_sectors} | {row.index_price_return:+.2%} | {row.fdr007_mean_change_bp:+.2f} | {row.cgb10_change_bp:+.2f} | {row.next20_net_return:+.2%} |")
    offset_table = ["| 月份 | 合成组合正贡献合计/百分点 | 负贡献合计/百分点 | 合成组合回报 | 官方沪深300价格回报 |", "|---|---:|---:|---:|---:|"]
    for month in ["2022-05", "2022-11", "2023-03"]:
        q = offsets.loc[month]
        offset_table.append(f"| {month} | {q.positive_contribution*100:+.3f} | {q.negative_contribution*100:+.3f} | {q.approximate_return:+.3%} | {monthly.set_index('month').loc[month,'index_price_return']:+.3%} |")
    def link(path, label):
        return f"[{label}](<{Path(path).as_posix()}>)"
    classification = "PROGRESS_FIXED_SECTOR_COMOVEMENT_AND_CREDIT_CONSTRAINT_CONTEXT"
    summary = "十五个固定月份的十行业价格已比较。2022年11月十行业全部上涨，沪深300+9.81%，FDR007月均及10年国债收益率同时上升；融资政策改变再融资、风险分担与权益补充条件，但实际新增资金和政策意外部分未识别。2023年3月指数接近平盘伴随明显行业抵消。11月月末后的固定20间隔净收益-1.81%；没有新增账户或交易优势。"
    next_question = "先去重2022年末至2023年初的历史融资总量与构成研究；核对政策后的融资改善是实际净新增资金、旧债滚续还是授信意向，以及需求兑现如何影响指数整体。只用已公开的总量证据，不扩展个股案例，不把本轮月份分组调成新规则。"
    findings = {"classification": classification, "november": nov.to_dict(), "november_approx_financial_property_contribution": financial, "may": may.to_dict(), "march": march.to_dict(), "descriptive_groups": groups, "policy_facts": policies, "offset_approximation": {"valid_months": int(offsets.approximate_return.notna().sum()), "missing_months": offsets[offsets.approximate_return.isna()].index.tolist(), "largest_absolute_difference_from_official_index": float(differences.max()), "largest_difference_month": differences.idxmax(), "official_index_attribution": False}, "unidentified": ["政策前一致预期和政策意外幅度", "信用支持实际净新增融资规模", "股票风险溢价的独立变化", "对各项同步消息的涨幅因果归属", "资金利率上升各来源的净贡献"], "goal_achieved": False, "new_accounts": 0, "new_models": 0}
    save(OUT / "mechanism_findings.json", findings)
    report_path = OUT / "历史发现_指数共同重估与行业抵消.md"
    figure = OUT / "指数行业_共同变化与抵消.png"
    report = f"""**指数历史发现：共同重估、行业抵消与信用约束**

本轮直接研究沪深300整体及000908—000917十条行业指数。固定覆盖2022年1月至2023年3月的15个自然月，使用中证官方11条指数价格序列，形成期和月末之后的510300可成交收益分别计算。新结果显示：行业普涨可以与资金利率上升同时出现，指数接近平盘也可能伴随强烈的行业抵消。研究没有扩展个股案例，没有新增模型或完整账户，夏普1.2及年化10%的目标仍未实现。

口径上，这十条并非十个互相独立的中证一级行业：000914将金融与房地产两个一级行业合并，其余按保存代码观察。按[官方000914编制方案](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000914hbook.pdf)保留这项说明，没有另加地产指数重复计数；原方案“十个一级行业”的措辞已在口径说明中纠正，固定代码、计数和结果均不变。

**第一项发现：2022年11月的共同上涨，不能简单归为国内观测利率下降。**

当月十个行业全部上涨，沪深300价格回报{nov.index_price_return:+.2%}；金融地产{nov_sectors.loc['金融地产','price_return']:+.2%}，通信服务{nov_sectors.loc['通信服务','price_return']:+.2%}，主要消费{nov_sectors.loc['主要消费','price_return']:+.2%}。与此同时，FDR007定盘利率月均上升{nov.fdr007_mean_change_bp:.2f}个基点，中债10年国债收益率月末值上升{nov.cgb10_change_bp:.2f}个基点。两种利率的期限及统计方法不同，分别保留；FDR007不是股市流入量，长债收益率也不是纯粹的预期短端政策利率。

官方PE从{nov.pe_start:.2f}变为{nov.pe_end:.2f}。同端点对数恒等式为：价格变化{nov.index_log_price_pp:.3f}个对数百分点 = PE变化{nov.index_log_pe_pp:.3f} + P/PE隐含分母变化{nov.index_log_implied_denominator_pp:.3f}。这说明价格和PE在该月共同上移；不能据此将PE变化全部命名为风险溢价下降，也不能将P/PE称作实际EPS增长。此前V26已研究这类恒等式，这里只复用为背景。

信用支持的原文补充了原因链条中更具体的一环：

| 本次来源标注日期 | 政策或执行阶段 | 直接改变的约束 | 尚不能据此确认 |
|---|---|---|---|
| 11月8日17:43 | 民企债券支持工具延期扩容，预期支持约2500亿元发债 | 增信、风险缓释与购债支持下的再融资条件 | 已实际投放2500亿元，或资金已买入股票 |
| 11月23日12:44 | 三家民企首批信用增进函，拟支持合计47亿元中票 | 框架推进到具体发行支持环节 | 47亿元均已发行结算，或均为净新增资金 |
| 11月24日政府网转发 | 贷款、存量融资展期和项目融资保障安排 | 到期压力、新资金受偿次序及执行顾虑 | 这一天才首次获知政策，或信用损失已经消失 |
| 11月28日正文日期 | 股权融资、重组及盘活存量资产安排 | 权益资本补充与资产处置渠道 | 所有企业已筹到钱、股东收益即时增厚或政策完全意外 |

前两行分别来自[交易商协会扩容公告]({POLICY_SOURCES[0]['url']})与[首批信用增进函说明]({POLICY_SOURCES[1]['url']})。2500亿元是预期支持的发债规模；47亿元是三份增信函拟支持的合计规模，不能重复相加作为资金流量，更不是股票净流入。

[金融支持房地产通知的政府网版本]({POLICY_SOURCES[2]['url']})将融资延续、项目资金管理及新增资金受偿安排连接起来。因此，上游原因不只是“利率高低”，还包括金融机构是否愿意续贷、谁承担损失、新资金如何得到偿还。政策设计同时覆盖债务人、债权人和项目链条，有可能通过多个行业进入指数定价。这是有原文支持的传导解释，尚未识别它对实际损失和指数涨幅的净因果效应。

[证监会五项安排]({POLICY_SOURCES[3]['url']})增加了权益融资及存量资产处置渠道，并限定有关募集资金用途。答记者问本身提及此前论坛已经表态，因此11月28日不能自动标为完全未预期的政策冲击。融资条件改善还可能伴随股权稀释、发行供给和项目需求不足，不能把政策方向直接等同于指数利润或股价必然上升。

同期还包括11月10日19:42已有的[国内政策优化方向](https://www.mct.gov.cn/whzx/szyw/202211/t20221110_937380.htm)，以及21:30的[美国10月CPI发布](https://www.bls.gov/news.release/archives/cpi_11102022.htm)，其公开时钟沿用原研究。月度观察不足以分离这些同时发生的信息。这里将“信用风险约束和活动约束有所松动”列为解释，而不宣称信用政策单独导致9.81%的上涨。

本轮未取得四项融资政策公布前的定量一致预期，也没有完成真实净融资到位规模的核算。因而不能计算这些政策的预期差，不能证明股票风险溢价下降了多少，也不能用政策文字填出原本不存在的买入胜率。

**第二项发现：行业上涨数量和指数涨幅之间，还隔着权重及行业抵消。**

2022年5月有8个行业上涨，沪深300只涨{may.index_price_return:.2%}；当月能源涨{may_sectors.loc['能源','price_return']:.2%}，可选消费涨{may_sectors.loc['可选消费','price_return']:.2%}，金融地产却跌{abs(may_sectors.loc['金融地产','price_return']):.2%}。行业一票一票计数，不能替代指数的权重计算。2022年6月同为8个行业上涨，指数涨幅达到9.62%，说明仅“8/10上涨”并未描述出相同的市场结构。

原有中信行业快照组合可以展示抵消的量级。按每天之前的合成财富连接行业贡献，所得三个月对照为：

{chr(10).join(offset_table)}

2023年3月官方指数仅跌{abs(march.index_price_return):.2%}，而快照组合中约+2.035个百分点正贡献与-2.437个百分点负贡献相抵。这种横截面结构应描述为行业力量互相抵消，不能仅从指数净变动很小就断言市场没有发生变化。2022年11月则大部分行业同向；快照组合内银行、非银行金融、房地产合计贡献约{financial*100:.3f}个百分点，但消费及其他行业也贡献了明显涨幅。

上述贡献是**旧快照合成组合近似，不是官方沪深300涨幅分解**。它使用中信行业分类、当时月度权重快照及成分含分红回报，十条官方行业价格指数采用另一套分类；两者没有混合。13个完整月份中，近似组合与官方价格回报最大绝对差为{differences.max()*100:.3f}个百分点，出现在{differences.idxmax()}。2022年6月和12月原账本分别有14日和15日不具备完整归因条件，整月保持NO_VIEW，没有补零。图表展示的是自身加总恒等式及量级，不宣称精确复制指数。

**第三项发现：已经发生的普涨，不等于月末之后还存在正收益。**

全部固定月份如下。前四个数值列描述形成月份；最后一列从月末后的下一实际交易日开盘开始，持有20个开盘间隔，以510300整手、佣金与压力滑点核算，含持有期间适用的现金分红。没有把形成月的涨幅算入随后交易。

{chr(10).join(table)}

预先固定的三类状态中，十行业全部上涨仅有2022年11月一例。12月1日开盘买入至12月29日开盘退出，固定20间隔净收益为{nov.next20_net_return:+.2%}；固定5间隔至12月8日的净收益为{nov.next5_net_return:+.2%}。这两项不同结果都保留，不能根据它们挑更好持有期。其余14个月属于混合状态，20间隔平均净收益{groups[2]['next20_mean_net_return']:+.2%}，其中4个月为正；十行业全部下跌没有样本，相关统计为空。

这些是15个月的描述，窗口之间可能重叠，不构成独立试验或完整账户。只有一个全涨月份，不能估计其长期胜率；也没有将条件放宽为“8个或9个上涨”补样本，更没有反向操作负结果。形成期的官方指数价格回报与随后ETF成本后收益标的和区间不同，不应直接相减当成因果效果。

**实用含义与边界。**

指数研究需要把共同影响与行业权重放在一起：资金价格、风险承接条件、经营需求和资本约束可能相互抵消；上游原因决定了同样的利率变化在不同阶段为何伴随不同指数表现。本轮证据具体支持两点：2022年11月的上涨无需以国内观测利率下降为前提；2023年3月的指数平淡掩盖了行业之间较大的相反贡献。它们帮助解释历史，尚不足以形成新增可交易优势。

本轮完成11条官方日序列共3322行、150个行业月观察、15个月整体对照、30个固定窗口、13个月近似贡献以及4份新增政策原网页。只核对价格端点、加总恒等式、事件先后和原文关键数字；没有增加长周期参数搜索。官方历史价格、PE和旧权重均为目前取得的历史数据，不是本机当年保存的不可变版本。证监会页的PubDate元数据实际为2026年页面生成时间，正文日期为2022年11月28日；已单独保留这一差异，未用元数据伪造历史分钟。政府网11月24日是本次取得的转发日期，不认定首次消息。

成本沿用原压力口径：每边佣金0.04%、最低5元，每边滑点0.10%，最小价位0.001元，100份整手及T+1限制。固定窗口按10万元名义资金独立计算，以实际买入支付额作为收益分母；没有相加成20万元账户。新完整账户为0，净夏普和年化收益为NOT_COMPUTED，既有拒绝封存规则保持原结论。

下一项历史问题：{next_question}

{link(figure, '查看全部月份的行业图')}。数据与脚本：{link(OUT / 'monthly_comparison.csv', '15个月完整比较')}；{link(OUT / 'sector_monthly_returns.csv', '150行行业回报')}；{link(OUT / 'policy_facts.json', '政策原文要点与时钟')}；{link(OUT / 'mechanism_findings.json', '发现及未识别环节')}；{link(Path(__file__), '完整研究脚本')}。

完成时间：{now()}。
"""
    report_path.write_text(report, encoding="utf-8")
    result = {"study_id": read(OUT / "protocol.json")["study_id"], "status": "COMPLETED_HISTORICAL_MECHANISM_STUDY_NO_TRADING_CANDIDATE", "completed_at": now(), "classification": classification, "discovery": summary, "next_historical_question": next_question, "report": rel(report_path), "figure": rel(figure), "figure_visually_reviewed": False, "report_status": "WRITTEN_PENDING_FINAL_REVIEW", "months": 15, "sectors": 10, "official_rows": 3322, "event_windows": 30, "policy_sources": 4, "new_accounts": 0, "new_models": 0, "account_status": "NOT_RUN_NO_ADMITTED_TRADING_CANDIDATE", "net_sharpe": None, "net_cagr": None, "metric_status": "NOT_COMPUTED", "goal_achieved": False}
    save(OUT / "result.json", result)
    files = [report_path, figure, OUT / "monthly_comparison.parquet", OUT / "sector_monthly_returns.csv", OUT / "event_returns.json", OUT / "industry_cancellation.json", OUT / "policy_facts.json", OUT / "mechanism_findings.json"]
    save(OUT / "research_receipt.json", {"created_at": now(), "script": rel(Path(__file__)), "script_sha256": digest(Path(__file__)), "protocol_sha256": digest(OUT / "protocol.json"), "output_hashes": {rel(p): digest(p) for p in files}, "price_checks": rel(OUT / "price_checks.json"), "goal_achieved": False})
    print(json.dumps({"报告": rel(report_path), "近似贡献最大偏离_百分点": differences.max()*100, "新增账户": 0, "目标完成": False}, ensure_ascii=False))


def finalize():
    assert digest(OUT / "protocol.json") == read(OUT / "freeze.json")["protocol_sha256"]
    monthly = pd.read_parquet(OUT / "monthly_comparison.parquet")
    sectors = pd.read_csv(OUT / "sector_monthly_returns.csv", dtype={"index_code": str})
    events = read(OUT / "event_returns.json")
    assert len(monthly) == 15 and len(sectors) == 150 and len(events) == 30
    for month, group in sectors.groupby("month"):
        row = monthly.set_index("month").loc[month]
        assert len(group) == 10 and int(group.price_return.gt(0).sum()) == row.positive_sectors
    largest_event_error = 0.
    for event in events:
        row = monthly.set_index("month").loc[event["event_id"].removeprefix("MONTH_END_")]
        assert event["status"] == "两端成交"
        assert pd.Timestamp(event["entry_date"]) > row.month_end
        assert pd.Timestamp(event["exit_date"]) > pd.Timestamp(event["entry_date"])
        assert pd.Timestamp(event["exit_date"]) == row[f"next{event['horizon']}_exit_date"]
        err = abs(event["net_pnl"]/event["paid_cny"]-event["net_return"])
        largest_event_error = max(largest_event_error, err)
        assert err < 1e-12
        assert abs(row[f"next{event['horizon']}_net_return"]-event["net_return"]) < 1e-12
    source_rows = read(OUT / "source_manifest.json")
    assert all(x["status"] == "RETRIEVED" and x["rows"] == 302 for x in source_rows)
    policies = read(OUT / "policy_facts.json")
    assert len(policies["facts"]) == 4 and all(not x["used_as_trading_trigger"] for x in policies["facts"])
    note = {"created_at": now(), "kind": "TERMINOLOGY_CORRECTION_NO_RULE_CHANGE", "original_wording": "十个一级行业", "corrected_wording": "000908—000917十条行业指数，其中000914合并金融与房地产两个一级行业", "source_url": "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000914hbook.pdf", "source_access": "官方编制方案的检索文本，2021年12月更新版本V1.0", "index_selection_changed": False, "data_changed": False, "grouping_rule_changed": False, "account_added": False}
    save(OUT / "methodology_note.json", note)
    save(OUT / "calculation_checks.json", {"checked_at": now(), "status": "SAVED_VALUES_AND_CALENDAR_CHECKED", "month_rows": 15, "sector_month_rows": 150, "event_windows": 30, "largest_event_return_recomputation_error": largest_event_error, "positive_sector_counts_recomputed": True, "event_returns_match_monthly_table": True, "frozen_protocol_unchanged": True, "policy_facts_used_for_trading": False, "new_full_accounts": 0, "is_independent_validation": False})
    result = read(OUT / "result.json")
    result.update(report_status="REVIEWED", figure_visually_reviewed=True, saved_values_checked=True, updated_at=now(), methodology_note=rel(OUT / "methodology_note.json"))
    save(OUT / "result.json", result)
    receipt = read(OUT / "research_receipt.json")
    receipt.update(updated_at=now(), script_sha256=digest(Path(__file__)), calculation_checks=rel(OUT / "calculation_checks.json"))
    receipt["output_hashes"].update({rel(OUT / name): digest(OUT / name) for name in ["result.json", "methodology_note.json", "calculation_checks.json"]})
    save(OUT / "research_receipt.json", receipt)
    cause_path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    cause = read(cause_path)
    cause.update(latest_completed_study=rel(OUT / "result.json"), latest_report=result["report"], current_study=rel(OUT / "protocol.json"), latest_result_summary=result["discovery"], next_historical_question=result["next_historical_question"], updated_at=now(), goal_achieved=False)
    save(cause_path, cause)
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(latest_historical_index_sector_repricing=rel(OUT / "result.json"), latest_historical_report=result["report"], latest_historical_diagnostic_at=now(), current_driver_continuation_classification=result["classification"], current_driver_consecutive_blocked_goal_turns=0, next_research_question=result["next_historical_question"], local_goal_work_status="ACTIVE_HISTORICAL_ONLY", goal_achieved=False)
    save(mandate_path, mandate)
    print(json.dumps({"报告状态": result["report_status"], "月度与窗口核对": "通过", "口径说明": note["corrected_wording"], "新增账户": 0, "目标完成": False}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指数行业共同变化与剩余收益的历史比较")
    parser.add_argument("action", choices=["prepare", "collect", "compute", "cancellation", "policy_sources", "policy_facts", "plot", "publish", "finalize"])
    action=parser.parse_args().action
    {"prepare": prepare, "collect": collect, "compute": compute, "cancellation": cancellation, "policy_sources": policy_sources, "policy_facts": policy_facts, "plot": plot, "publish": publish, "finalize": finalize}[action]()
