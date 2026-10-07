"""沪深300历史需求渠道：份额转换、融资周转与融资约束，固定半年描述。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import historical_index_reopening_constraints_v1 as execution
from research.factor96_margin_repair_v1 import clean, now, save

OUT = ROOT / "reports/research/510300_historical_index_participation_channels_v1"
REPORT = OUT / "历史发现_指数份额转换与融资需求.md"
MONTHS = pd.period_range("2022-10", "2023-03", freq="M").astype(str).tolist()
POOL = ["510300.SH", "510310.SH", "510330.SH"]
FILES = {
    "weekly_shares": "data/raw/flow/510300_cross_etf_forced_flow_sse_weekly_v1.parquet",
    "daily_shares": "data/raw/flow/510300_fund_share_daily_tushare.parquet",
    "etf_margin": "data/raw/flow/510300_margin_detail_daily_tushare.parquet",
    "market_margin": "reports/research/510300_factor96_mechanism_batch_v1/data_repair/margin_complete.parquet",
    "prices": "data/raw/all_etf_momentum_v1r/etf_total_return_panel_tushare_adj.parquet",
    "official_indices": "reports/research/510300_historical_index_sector_repricing_v1/official_index_daily.parquet",
    "weekly_previous": "data/research/510300_cross_etf_forced_flow_binary_screen_v1/weekly_features.parquet",
}
SOURCES = [
    {"id": "sse_margin_definition", "url": "https://www.sse.com.cn/market/othersdata/margin/sum/", "date": None, "role": "现行字段说明，不作为2022年政策事件或历史时钟证明"},
    {"id": "sse_expansion", "url": "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20221021_5710381.shtml", "date": "2022-10-21", "role": "上交所原公告"},
    {"id": "szse_expansion", "url": "https://www.szse.cn/aboutus/trends/news/t20221021_596641.html", "date": "2022-10-21", "role": "深交所原公告"},
    {"id": "sse_etf_arbitrage", "url": "https://etf.sse.com.cn/fund/learning/strategy/c/5704303.shtml", "date": "2022-06-23", "role": "上交所投资者教育，只证明机制可能性"},
    {"id": "fund_prospectus_2022", "url": "https://pdf.dfcfw.com/pdf/H2_AN202206111571368639_1.pdf", "date": "2022-06-11", "role": "华泰柏瑞基金原始说明书，东方财富承载副本"},
    {"id": "csf_rate_cut_relay", "url": "https://www.cs.com.cn/xwzx/hg/202210/t20221020_6303475.html", "date": "2022-10-20", "role": "中国证券报当日报道，转述中证金融，非官网原件"},
    {"id": "csf_pilot_relay", "url": "https://m.yicai.com/news/101681395.html", "date": "2023-02-21", "role": "第一财经当日报道，转述中证金融，非官网原件"},
]
EVENTS = [
    {"event_id": "CSF_COST_20221020", "date": "2022-10-20", "title": "转融资成本调整与市场化试点启动公告", "source_ids": ["csf_rate_cut_relay"]},
    {"event_id": "MARGIN_SCOPE_20221021", "date": "2022-10-21", "title": "两市融资融券标的扩围公告", "source_ids": ["sse_expansion", "szse_expansion"]},
    {"event_id": "CSF_PILOT_20230221", "date": "2023-02-21", "title": "市场化转融资试点上线", "source_ids": ["csf_pilot_relay"]},
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare():
    protocol = {
        "study_id": "510300_HISTORICAL_INDEX_PARTICIPATION_CHANNELS_V1", "created_at": now(),
        "question": "融资渠道放宽后，实际借入与归还、ETF份额转换和指数价格是否同向；哪些上游行为能够识别？",
        "months": MONTHS, "same_index_etfs": POOL,
        "calendar": "每月最后一个官方周度快照与上月最后快照之间；三只ETF、两融、指数统一实际起止日期。另存逐周与单ETF逐日明细。",
        "why_period": "沿用既定2022Q4至2023Q1信用传导阶段，不因资金或收益符号移动边界；此前该期行情已见，不是盲测。",
        "deduplication": {
            "cross_etf_flow": "原5候选REJECTED_FIXED_CROSS_ETF_FORCED_FLOW_BINARY_FAMILY_NO_RESCUE，不重跑规则",
            "share_premium": "原8候选REJECTED_FIXED_ETF_SHARE_PREMIUM_LEVEL_BINARY_FAMILY_NO_RESCUE，不重跑规则",
            "margin_factors": "F01/F02/F06既有两种表达没有合格账户，不调整门槛、延迟或方向",
            "new_increment": "融资供给制度、借入与隐含偿还的周转分解、单ETF与固定同指数ETF池的范围差异，同期和公布后回报分开",
        },
        "measurements": {
            "share_flow_proxy": "sum((Q_t-Q_prev)*raw_close_t)，份额转换市值代理，非实收现金、未完成买单或全市场净流入",
            "market_margin": "两市全标的；隐含偿还=融资买入-余额变化。扩围改变可融资范围，汇总无法排除新标的贡献",
            "etf_margin": "510300本身的融资，已包含于全市场两融，不相加；偿还原报数与余额恒等式残差单列",
            "same_index_pool": "固定旧池三只上交所沪深300ETF，不是所有沪深300ETF，更不是全A股资金",
        },
        "source_clock": "周份额T统计保守T+1收盘可用，T+2开盘比较；历史首次发布时间仍未证明，不重复已关闭的时钟搜寻",
        "event_windows": [5, 20], "constraint_events": EVENTS,
        "constraint_event_clock": "使用公告日期当日结束的可见上界，下一实际交易日开盘；三个已识别制度节点，不声称是全部政策样本",
        "cost": execution.COST, "market_end": "2023-05-31",
        "new_candidates": 0, "new_full_accounts": 0, "goal_achieved": False,
        "independent_validation": False, "orders_authorized": False,
        "stop": "仅描述所有固定区间，不按正负收益筛月份，不把事后逆向申购、扩围或余额变化改成策略；未知动机与对冲保留未知。",
    }
    save(OUT / "protocol.json", protocol, exclusive=True)
    save(OUT / "source_plan.json", SOURCES, exclusive=True)
    print("已固定半年区间、三个制度节点与原有成本；不新增参数搜索。")


def collect():
    def one(item):
        destination = OUT / "sources"
        destination.mkdir(parents=True, exist_ok=True)
        record = dict(item, retrieved_at=now())
        try:
            response = requests.get(item["url"], timeout=(10, 30), headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            is_pdf = response.content.startswith(b"%PDF")
            path = destination / (item["id"] + (".pdf" if is_pdf else ".html"))
            path.write_bytes(response.content)
            if is_pdf:
                import pdfplumber
                with pdfplumber.open(path) as document:
                    body = "\n".join(f"【PDF页码{i+1}】\n{page.extract_text() or ''}" for i, page in enumerate(document.pages))
            else:
                response.encoding = response.apparent_encoding
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.decompose()
                body = soup.get_text("\n", strip=True)
            (destination / (item["id"] + ".txt")).write_text(body, encoding="utf-8")
            record.update(status="SAVED_PENDING_CONTENT_REVIEW", raw_path=rel(path), bytes=len(response.content), sha256=digest(path))
        except Exception as exc:
            record.update(status="FETCH_FAILED", error=f"{type(exc).__name__}: {str(exc)[:240]}")
        return record
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(one, SOURCES))
    save(OUT / "source_manifest.json", records)
    print(json.dumps([{k: r.get(k) for k in ["id", "status", "bytes", "error"]} for r in records], ensure_ascii=False))


def build():
    protocol = read(OUT / "protocol.json")
    base_protocol = read(execution.OUT / "protocol.json")
    base_protocol["account_calendar"][1] = protocol["market_end"]
    market, _, dividends, engine, _ = execution.inputs(base_protocol)
    source = {k: pd.read_parquet(ROOT / v) for k, v in FILES.items() if k != "prices"}
    for frame in source.values():
        frame["date"] = pd.to_datetime(frame.date).dt.normalize()
    prices = pd.read_parquet(ROOT / FILES["prices"], columns=["date", "con_code", "raw_close"])
    prices["date"] = pd.to_datetime(prices.date).dt.normalize()
    shares = source["weekly_shares"].loc[lambda d: d.ts_code.isin(POOL), ["date", "ts_code", "fund_shares"]].copy()
    prices = prices.rename(columns={"con_code": "ts_code"})
    shares = shares.merge(prices.loc[prices.ts_code.isin(POOL)], on=["date", "ts_code"], how="left", validate="one_to_one")
    shares = shares.sort_values(["ts_code", "date"])
    shares["previous_date"] = shares.groupby("ts_code").date.shift()
    shares["share_change"] = shares.groupby("ts_code").fund_shares.diff()
    shares["flow_proxy_cny"] = shares.share_change * shares.raw_close
    assert shares.raw_close.notna().all()
    weekly = shares.groupby("date", as_index=False).agg(flow_proxy_cny=("flow_proxy_cny", "sum"), positive_funds=("share_change", lambda s: int(s.gt(0).sum())), funds=("ts_code", "nunique"))
    assert weekly.funds.eq(3).all()
    weekly = weekly.merge(source["weekly_previous"][["date", "hs300_weekly_flow_cny"]], on="date", validate="one_to_one")
    keep = weekly.date.between("2022-10-01", "2023-03-31")
    max_pool_error = float((weekly.loc[keep, "flow_proxy_cny"] - weekly.loc[keep, "hs300_weekly_flow_cny"]).abs().max())
    assert max_pool_error < 1e-4
    daily = market[["date", "open", "close", "amount", "total_simple"]].copy()
    em = source["etf_margin"].sort_values("date").copy()
    em["etf_balance_change"] = em.rzye.diff()
    em["etf_repay_implied"] = em.rzmre - em.etf_balance_change
    em["etf_identity_residual"] = em.etf_balance_change - (em.rzmre - em.rzche)
    em = em.rename(columns={"rzye": "etf_balance", "rzmre": "etf_buy", "rzche": "etf_repay_reported"})
    mm = source["market_margin"].sort_values("date").copy()
    mm["market_balance_change"] = mm.market_rzye.diff()
    mm["market_repay_implied"] = mm.market_rzmre - mm.market_balance_change
    daily = daily.merge(em[["date", "etf_balance", "etf_buy", "etf_repay_reported", "etf_repay_implied", "etf_balance_change", "etf_identity_residual"]], on="date", how="left", validate="one_to_one")
    daily = daily.merge(mm[["date", "market_rzye", "market_rzmre", "market_balance_change", "market_repay_implied"]], on="date", how="left", validate="one_to_one")
    ds = source["daily_shares"][["date", "fund_shares"]].copy()
    daily = daily.merge(ds, on="date", how="left", validate="one_to_one")
    daily["etf_share_change"] = daily.fund_shares.diff()
    daily["etf_daily_share_proxy_cny"] = daily.etf_share_change * daily.close
    daily["etf_financing_buy_turnover_ratio"] = daily.etf_buy / daily.amount
    official = source["official_indices"]
    print("官方指数字段：", list(official.columns))
    # 指数输入必须明确代码；不根据回报结果选序列。
    codecol = "indexCode"
    official = official.loc[official[codecol].astype(str).eq("000300")].copy()
    if official.empty:
        raise ValueError("官方指数文件没有明确的000300序列")
    closecol = "close" if "close" in official.columns else "close_price"
    csi = official.set_index("date")[closecol]
    ends = weekly.assign(month=weekly.date.dt.to_period("M").astype(str)).groupby("month").date.max()
    comparison, events, intervals = [], [], []
    required = ["etf_balance", "etf_buy", "etf_repay_reported", "market_rzye", "market_rzmre", "fund_shares"]

    def window(a, b):
        block = daily.loc[daily.date.gt(a) & daily.date.le(b)].copy()
        assert len(block) and block[required].notna().all().all(), (a, b)
        before = daily.loc[daily.date.eq(a)].iloc[0]
        after = block.iloc[-1]
        flow = shares.loc[shares.date.gt(a) & shares.date.le(b)]
        r = {"start": a, "end": b, "trading_days": len(block),
             "csi300_price_return": float(csi.loc[b] / csi.loc[a] - 1),
             "etf_gross_total_return": float((1 + block.total_simple).prod() - 1),
             "same_index_pool_proxy_cny": float(flow.flow_proxy_cny.sum()),
             "etf_share_change": float(after.fund_shares - before.fund_shares),
             "etf_daily_share_proxy_cny": float(block.etf_daily_share_proxy_cny.sum()),
             "etf_buy_cny": float(block.etf_buy.sum()),
             "etf_repay_reported_cny": float(block.etf_repay_reported.sum()),
             "etf_balance_change_cny": float(after.etf_balance - before.etf_balance),
             "etf_identity_residual_cny": float(block.etf_identity_residual.sum()),
             "etf_buy_mean_daily_cny": float(block.etf_buy.mean()),
             "etf_repay_mean_daily_cny": float(block.etf_repay_reported.mean()),
             "etf_financing_buy_turnover_ratio": float(block.etf_buy.sum() / block.amount.sum()),
             "market_buy_cny": float(block.market_rzmre.sum()),
             "market_repay_implied_cny": float(block.market_repay_implied.sum()),
             "market_buy_mean_daily_cny": float(block.market_rzmre.mean()),
             "market_repay_mean_daily_cny": float(block.market_repay_implied.mean()),
             "market_balance_change_cny": float(after.market_rzye - before.market_rzye),
             "market_balance_start_cny": float(before.market_rzye), "market_balance_end_cny": float(after.market_rzye)}
        for symbol in POOL:
            r[f"share_proxy_{symbol[:6]}_cny"] = float(flow.loc[flow.ts_code.eq(symbol), "flow_proxy_cny"].sum())
        assert abs(r["market_buy_cny"] - r["market_repay_implied_cny"] - r["market_balance_change_cny"]) < .01
        return r

    for date in weekly.loc[keep, "date"]:
        previous = weekly.loc[weekly.date.lt(date), "date"].iloc[-1]
        intervals.append(window(previous, date))
    for month in MONTHS:
        end = ends.loc[month]
        start = ends.loc[str(pd.Period(month, freq="M") - 1)]
        r = dict(month=month, **window(start, end))
        positions = np.flatnonzero(market.date.gt(end))
        idx = int(positions[1])
        r.update(statistical_date=end, assumed_available_date=market.date.iloc[positions[0]], next_entry_date=market.date.iloc[idx], historical_first_publication_verified=False)
        for horizon in protocol["event_windows"]:
            clock = {"event_id": "SNAPSHOT_" + month, "title": month + "完整区间末次快照", "entry_idx": idx}
            event = execution.fixed_event_window(market, dividends, engine, clock, horizon)
            r[f"next{horizon}_net_return"] = event["net_return"]
            events.append(event)
        comparison.append(r)
    for event in EVENTS:
        idx = int(np.flatnonzero(market.date.gt(event["date"]))[0])
        for horizon in protocol["event_windows"]:
            output = execution.fixed_event_window(market, dividends, engine, dict(event, entry_idx=idx), horizon)
            output.update(announcement_date=event["date"], source_ids=event["source_ids"], policy_causal_effect_identified=False)
            events.append(output)
    for name, frame in [("monthly_comparison", pd.DataFrame(comparison)), ("weekly_comparison", pd.DataFrame(intervals)), ("weekly_etf_components", shares.loc[shares.date.between("2022-09-01", "2023-03-31")]), ("daily_participation", daily.loc[daily.date.between("2022-09-30", "2023-03-31")])]:
        frame.to_parquet(OUT / (name + ".parquet"), index=False)
        frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")
    save(OUT / "event_returns.json", events)
    save(OUT / "calculation_checks.json", {"same_pool_proxy_reproduction_max_cny": max_pool_error, "monthly_intervals": len(comparison), "weekly_intervals": len(intervals), "event_windows": len(events), "account_identity_checks": "PASS_SHARED_EVENT_EXECUTION_ENGINE", "missing_rows_filled": 0, "historical_first_publication_verified": False, "daily_share_nontrading_rows": "不进入实际交易日历，不人工平移", "etf_margin_max_daily_identity_residual_cny": float(daily.loc[daily.date.between("2022-10-01", "2023-03-31"), "etf_identity_residual"].abs().max())})
    save(OUT / "input_receipts.json", [{"path": path, "sha256": digest(ROOT / path)} for path in FILES.values()])
    print(pd.DataFrame(comparison)[["month", "start", "end", "csi300_price_return", "same_index_pool_proxy_cny", "etf_balance_change_cny", "market_balance_change_cny", "next20_net_return"]].to_string(index=False))


def resolve():
    """仅记录已核对内容、网页工具证据与两日官方差额；不修订源数据。"""
    manifest = read(OUT / "source_manifest.json")
    required = {
        "sse_margin_definition": ["本日融资余额", "本日直接还款额", "本日融资强制平仓额"],
        "sse_expansion": ["2022-10-21", "800只扩大到1000只", "2022年10月24日"],
        "sse_etf_arbitrage": ["2022年06月23日", "使用一篮子股票申购ETF份额", "然后在二级市场卖出ETF"],
        "fund_prospectus_2022": ["华泰柏瑞", "2022年第1号", "退补现金替代适用于深交所上市的成份股"],
        "csf_rate_cut_relay": ["2022-10-2020:25", "40个基点", "正常经营性调整"],
    }
    web_facts = [
        {"id": "szse_expansion", "url": SOURCES[2]["url"], "page_date": "2022-10-21", "source_type": "原交易所网页，网页搜索工具提供完整正文；本机抓取失败", "facts": ["10月24日起，深市注册制股票以外的标的数量由800只增至1200只。", "新纳入400只中，流通市值低于100亿元的股票超过七成；扩围后深市沪深300成分全覆盖。"], "historical_first_release_verified": False},
        {"id": "csf_pilot_relay", "url": SOURCES[6]["url"], "page_date": "2023-02-21", "source_type": "第一财经当日报道转述中证金融，网页工具正文可读；本机抓取失败", "facts": ["2023年2月21日市场化转融资业务试点上线，以竞价和灵活期限改善证券公司的筹资安排。", "此前启动试点与当天系统上线是两个阶段，均不能等同于客户实际增持股票。"], "historical_first_release_verified": False},
    ]
    for record in manifest:
        key = record["id"]
        if key in required:
            body = "".join((OUT / "sources" / (key + ".txt")).read_text(encoding="utf-8").split())
            for token in required[key]:
                assert "".join(token.split()) in body, (key, token)
            if "error" in record:
                record["initial_error_preserved"] = record.pop("error")
            path = OUT / "sources" / (key + (".pdf" if key == "fund_prospectus_2022" else ".html"))
            record.update(status="CONTENT_REVIEWED", raw_path=rel(path), sha256=digest(path), bytes=path.stat().st_size)
        else:
            record.update(status="WEB_TEXT_REVIEWED_NATIVE_FETCH_FAILED", web_evidence_path=rel(OUT / "web_evidence.json"))
    save(OUT / "source_manifest.json", manifest)
    save(OUT / "web_evidence.json", web_facts)
    local = pd.read_parquet(OUT / "daily_participation.parquet").set_index("date")
    comparisons = []
    for day in ["20221213", "20221214"]:
        path = OUT / "sources" / f"sse_margin_detail_{day}.json"
        url = "https://query.sse.com.cn/marketdata/tradedata/queryMargin.do"
        params = {"isPagination": "true", "tabType": "mxtype", "detailsDate": day, "stockCode": "510300", "beginDate": "", "endDate": "", "pageHelp.pageSize": "50", "pageHelp.pageCount": "1", "pageHelp.pageNo": "1", "pageHelp.beginPage": "1", "pageHelp.cacheSize": "1", "pageHelp.endPage": "1"}
        if not path.exists():
            response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.sse.com.cn/"}, timeout=(8, 20))
            response.raise_for_status()
            path.write_bytes(response.content)
        raw = read(path)
        row = next(item for item in raw["result"] if item["stockCode"] == "510300")
        d = local.loc[pd.Timestamp(day)]
        assert row["rzye"] == d.etf_balance and row["rzmre"] == d.etf_buy and row["rzche"] == d.etf_repay_reported
        comparisons.append({"date": pd.Timestamp(day), "official_row": row, "local_values_equal": True, "raw_path": rel(path), "sha256": digest(path), "url": url, "params": params, "verified_at": now(), "historical_first_publication_verified": False})
    a, b = (item["official_row"] for item in comparisons)
    residual = b["rzye"] - a["rzye"] - b["rzmre"] + b["rzche"]
    assert residual == 21842338
    differences = local.loc[local.etf_identity_residual.abs().gt(1), ["etf_buy", "etf_repay_reported", "etf_balance", "etf_identity_residual"]].reset_index()
    differences.to_csv(OUT / "融资明细恒等式差额.csv", index=False, encoding="utf-8-sig")
    save(OUT / "margin_source_difference.json", {"status": "OFFICIAL_DETAIL_CONFIRMED_IDENTITY_DIFFERENCE_UNEXPLAINED", "checks": comparisons, "max_day": "2022-12-14", "official_identity_residual_cny": residual, "decision": "官方明细同样存在差额，不归咎于供应商导入，不据此认定强平或直接还款金额。原报偿还与按余额推算的偿还并列，源表不改写。", "difference_days": len(differences)})
    print("七项来源内容已处理；最大融资差额获两日官方明细确认，原因仍未知。")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 10})
    d = pd.read_parquet(OUT / "monthly_comparison.parquet")
    labels = [f"{r.start:%m.%d}—{r.end:%m.%d}" for r in d.itertuples()]
    x = np.arange(len(d))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"hspace": .30})
    fig.suptitle("同一阶段：沪深300、ETF份额与两市融资出现分化", fontsize=18, y=.985, fontweight="bold")
    fig.text(.5, .937, "2022年9月30日至2023年3月31日｜固定三只沪深300ETF｜统一周快照边界", ha="center", color="#525E68", fontsize=11)
    colors = np.where(d.csi300_price_return.ge(0), "#1B7261", "#BB574B")
    bars = axes[0].bar(x, d.csi300_price_return * 100, color=colors, width=.58)
    axes[0].bar_label(bars, labels=[f"{v:+.2f}%" for v in d.csi300_price_return*100], padding=4)
    axes[0].set_title("同期沪深300价格变化", loc="left", fontsize=12)
    axes[0].set_ylabel("%")
    axes[0].set_ylim(-9.5, 11)
    positive, negative = np.zeros(len(d)), np.zeros(len(d))
    for symbol, color in zip(POOL, ["#266B83", "#78A7B4", "#B5D0D6"]):
        values = d[f"share_proxy_{symbol[:6]}_cny"].to_numpy() / 1e8
        bottom = np.where(values >= 0, positive, negative)
        axes[1].bar(x, values, bottom=bottom, color=color, width=.58, label=symbol[:6])
        positive += np.maximum(values, 0)
        negative += np.minimum(values, 0)
    axes[1].set_title("三只ETF份额转换的市值代理（净份额变化×各自当周收盘价）", loc="left", fontsize=12)
    axes[1].set_ylabel("亿元")
    axes[1].legend(ncol=3, loc="upper right", frameon=False)
    for i, value in enumerate(d.same_index_pool_proxy_cny/1e8):
        axes[1].text(i, value+(7 if value >= 0 else -8), f"{value:+.1f}", ha="center", va="bottom" if value >= 0 else "top")
    axes[1].set_ylim(-200, 255)
    values = d.market_balance_change_cny / 1e8
    bars = axes[2].bar(x, values, width=.58, color=np.where(values.ge(0), "#946A37", "#B7A187"))
    axes[2].bar_label(bars, labels=[f"{v:+.1f}" for v in values], padding=4)
    axes[2].set_title("沪深两市全标的融资余额变化（包含更广的股票及基金范围）", loc="left", fontsize=12)
    axes[2].set_ylabel("亿元")
    axes[2].set_ylim(-380, 850)
    axes[2].set_xticks(x, labels)
    for ax in axes:
        ax.axhline(0, color="#819099", lw=.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.12)
        ax.set_axisbelow(True)
    fig.subplots_adjust(top=.885, bottom=.13, left=.09, right=.97)
    fig.text(.09, .065, "份额代理不是实收现金；融资余额变化不是全部股票净买入。两者存在范围差异和重叠，不能相加。\n资料：上交所官方份额、中证指数、既有两市融资数据。图中为同期描述，非政策因果效应或买入规则。", fontsize=10, color="#525E68", linespacing=1.8)
    fig.savefig(OUT / "指数需求_份额与融资分化.png", dpi=160, facecolor="white")
    plt.close(fig)
    print("已生成统一时间边界的三层对照图。")


def publish():
    d = pd.read_parquet(OUT / "monthly_comparison.parquet")
    events = read(OUT / "event_returns.json")
    checks = read(OUT / "calculation_checks.json")
    source_diff = read(OUT / "margin_source_difference.json")
    manifest = read(OUT / "source_manifest.json")
    assert all(item["status"] in ["CONTENT_REVIEWED", "WEB_TEXT_REVIEWED_NATIVE_FETCH_FAILED"] for item in manifest)
    rowtable = ["| 实际区间 | 沪深300同期价格 | 三只ETF份额代理/亿元 | 两市融资余额变化/亿元 | 510300融资余额变化/亿元 | 保守时钟后20日净收益 |", "|---|---:|---:|---:|---:|---:|"]
    for r in d.itertuples():
        rowtable.append(f"| {r.start:%Y-%m-%d}→{r.end:%Y-%m-%d} | {r.csi300_price_return:+.2%} | {r.same_index_pool_proxy_cny/1e8:+.2f} | {r.market_balance_change_cny/1e8:+.2f} | {r.etf_balance_change_cny/1e8:+.2f} | {r.next20_net_return:+.2%} |")
    turnover = ["| 区间标签（边界同上） | 两市日均融资买入/亿元 | 两市日均隐含偿还/亿元 | 510300融资买入占本ETF成交额 |", "|---|---:|---:|---:|"]
    for r in d.itertuples():
        turnover.append(f"| {r.month} | {r.market_buy_mean_daily_cny/1e8:.2f} | {r.market_repay_mean_daily_cny/1e8:.2f} | {r.etf_financing_buy_turnover_ratio:.2%} |")
    policytable = ["| 制度节点 | 公告后下一开盘 | 5日净收益 | 20日净收益 |", "|---|---|---:|---:|"]
    for event in EVENTS:
        a = next(r for r in events if r["event_id"] == event["event_id"] and r["horizon"] == 5)
        b = next(r for r in events if r["event_id"] == event["event_id"] and r["horizon"] == 20)
        policytable.append(f"| {event['date']} {event['title']} | {a['entry_date'][:10]} | {a['net_return']:+.2%} | {b['net_return']:+.2%} |")
    discovery = "已识别融资供给成本、标的范围与实际需求的不同层次；2023-01-20至02-24两市融资余额增加677.63亿元，但沪深300跌2.88%、固定三只ETF份额代理为-148.60亿元。份额转换和融资净变动不能代替指数持续买入。最大偿还恒等式差额获官方明细确认但原因未识别。无新增策略或合格账户。"
    next_question = "沿用2022Q4至2023Q1完整日历，先核既有中证500/1000与沪深300风格比较，判断全市场融资扩张是否对应更广市场的重新配置；只做指数层面的范围与权重解释，不将本轮正负月份改为买入或反向规则。"
    sources_md = "\n".join(f"- [{r['id']}：{r['role']}]({r['url']})。{r['status']}。" for r in manifest)
    text = f"""# 历史发现：指数份额转换与融资需求

本轮有新的机制证据，没有形成夏普1.2的交易方案。固定比较2022Q4至2023Q1的六个区间、24个周区间及三个融资制度节点。沪深300整体仍是研究对象；510300和另外两只同指数ETF只是观察需求渠道，其他资产没有交易授权。

**核心发现：融资供给更便利、全市场融资余额增加、沪深300ETF份额增加，分属不同环节，不能合并为一个“流动性利好”信号。**

![指数份额与融资需求](指数需求_份额与融资分化.png)

{chr(10).join(rowtable)}

三只基金为原研究固定的510300、510310、510330，未新增或更换池子，也不代表全部沪深300ETF。份额代理为各周净份额变化乘以该基金当周未复权收盘价，再求和；不是实收现金。月标签实际采用上月末次与本月末次官方周快照之间的区间，因此1月终点是春节前1月20日，不能写成整个自然月。所有资金和同期价格采用同一起止日期。

右列为统计日后第一交易日收盘假定可用、第二交易日开盘进入510300的固定20交易日比较；原始首次发布时间未证明，属于保守日期假设下的历史计算。下一开盘日期及5日结果均保存在明细。成本为双边佣金万4、最低5元、每边千1滑点，按0.001元报价、100份整手和T+1执行，分红按权益日处理。各窗用10万元名义资金算已投入资金净收益，存在重叠，不是20万元连续全账户，不相加、不年化夏普。

2022年12月30日至2023年1月20日，沪深300上涨8.00%，三只ETF份额代理增加38.84亿元，两市融资余额却减少257.38亿元。随后1月20日至2月24日，两市融资余额增加677.63亿元，沪深300下跌2.88%，三只ETF份额代理减少148.60亿元。前一区间说明上涨可以与全市场去融资同时发生；后一区间说明全市场加融资也不必对应沪深300走强。这些是该阶段的事实，尚不能推断交易者动机或因果贡献。

融资余额是存量结果，要先拆到借入和归还。下表把交易天数不同造成的总量差异换成日均值：

{chr(10).join(turnover)}

这里“两市隐含偿还”只是融资买入减余额变化所得的代数量。交易所说明偿还包含直接还款、卖券还款、强平以及权益调整，不能把它全部看成主动卖出或被迫抛售。510300融资额属于ETF二级市场交易，既包含于两市两融，也与ETF其他交易环节可能重叠；它们不能累加成“总流入”。510300融资买入占自身成交额约4.8%—7.6%，只能说明可观测的这一种下单方式，不能将剩余成交额判为机构或无杠杆资金。

上游原因中，能核实的是融资供给条件的变化：

- 2022年10月20日，中国证券报当日报道中证金融将转融资费率下调40个基点，并转述其依据资金市场利率调整的说明。这降低的是证券公司的一个筹资渠道报价；没有客户实际融资利率、额度使用和对冲头寸，不能假定投资者融资需求同比例上升。[当日报道]({SOURCES[5]['url']})
- 10月21日，两交易所公告扩围，10月24日实施。沪市主板从800只增至1000只，深市注册制股票以外从800只增至1200只；深市新增400只中，七成以上流通市值低于100亿元。后续融资余额的变化同时可能受融资范围和需求影响，不能单独代表沪深300风险偏好。此次研究没有逐项分离新标的贡献。[上交所]({SOURCES[1]['url']})、[深交所]({SOURCES[2]['url']})
- 2023年2月21日市场化转融资试点上线，改善期限与竞价安排。试点启动、系统上线、证券公司实际借钱、客户买入是四件事；聚合数据无法将它们归并成确定的资金传导金额。[当日报道]({SOURCES[6]['url']})

{chr(10).join(policytable)}

三个节点按事前选定的融资成本、标的范围、系统上线分类，不是全部政策事件样本。10月两窗高度重叠，同时还有其他宏观和市场信息；不能当作三次独立实验，也不能将窗口收益归因于该政策。三个5日净收益均负，两个10月事件的20日净收益为小幅正，2月上线后的20日为负，说明“约束放松后马上买指数”在这些节点并非确定性。

ETF份额变动还需要往前追一层：谁交付了什么？2022年招募说明书规定，申购对价可以包含组合证券、现金替代和现金差额；深市成分采用退补现金替代，沪市存在实物及不同现金替代安排。交易所还列出了买入股票申购ETF后卖出ETF的套利过程。因此，新增份额可能来自新的风险需求，也可能来自已有股票的转换和套利配套，净份额表不能识别现金与实物的实际占比、最终持有人、对冲，或尚未完成的买单。[基金原说明书副本]({SOURCES[4]['url']})、[交易所机制说明]({SOURCES[3]['url']})

这不意味着ETF申购没有价格影响，而是本轮数据不足以计量哪类申购、何时产生了多少价格影响。应把“参与渠道变化”与“新增预期差”分开。前三个区间末次快照后的20日净收益为正，后三个为负；1月区间份额代理仍为正，后20日却为-3.90%。此前份额及溢价规则已有冻结失败，本轮不据这六个结果改方向、择时点或拼接参数。

有一处影响解释的数据差额需保留：2022年12月14日，510300融资余额由6,931,055,515元变为6,952,381,844元，融资买入180,510,959元，原报偿还181,026,968元。余额变化与买入减偿还相差21,842,338元。两日上交所原始明细与本地值逐项一致，因此不是本轮导入问题；差额的业务原因仍未识别，不能硬归于直接还款或强平。九个差额日期及原报、隐含值均保留。两市偿还是按余额推算，不宣称取得了独立完整的偿还分类。

本轮只做必要核对：固定三只基金周份额代理与原保存结果逐点一致，24周和6区间没有补行，18个事件窗口通过原成交与分红恒等式。官方份额的历史首次发布时间仍未知；本轮没有重开已关闭的时钟搜寻，也没有新增长周期参数检验。基金说明书本机已保存；深交所公告和第一财经报道由网页工具读到，原生抓取失败分别保留，未把转述升级为官网原件。

接下来的可检验问题是：同一阶段更广市场的融资扩张，是否对应沪深300以外的指数风格变化。它需要指数之间的范围比较。尚未计算出新的完整账户，净夏普、CAGR和独立有效性均未取得新结果，目标继续保持未达成。

可复核文件：`monthly_comparison.csv`、`weekly_comparison.csv`、`daily_participation.parquet`、`event_returns.json`、`margin_source_difference.json`。完整来源如下：

{sources_md}
"""
    REPORT.write_text(text, encoding="utf-8")
    result = {"study_id": read(OUT / "protocol.json")["study_id"], "status": "COMPLETED_HISTORICAL_MECHANISM_STUDY_NO_TRADING_CANDIDATE", "classification": "PROGRESS_INDEX_PARTICIPATION_SCOPE_AND_FINANCING_SUPPLY_SEPARATION", "completed_at": now(), "report": rel(REPORT), "figure": rel(OUT / "指数需求_份额与融资分化.png"), "report_status": "PENDING_VISUAL_REVIEW", "figure_visually_reviewed": False, "discovery": discovery, "months": 6, "weekly_intervals": 24, "constraint_events": 3, "event_windows": 18, "new_candidates": 0, "new_full_accounts": 0, "net_sharpe": None, "net_cagr": None, "max_drawdown": None, "goal_achieved": False, "new_prospective_tasks": 0, "orders_authorized": False, "independent_validation": False, "next_historical_question": next_question, "calculation_checks": checks, "official_detail_discrepancy_status": source_diff["status"]}
    save(OUT / "result.json", result)
    print(discovery)


def finalize():
    result = read(OUT / "result.json")
    result.update(report_status="REVIEWED", figure_visually_reviewed=True, updated_at=now())
    save(OUT / "result.json", result)
    save(OUT / "research_receipt.json", {"generated_at": now(), "script_sha256": digest(Path(__file__)), "output_hashes": {rel(p): digest(p) for p in OUT.iterdir() if p.is_file() and p.name != "research_receipt.json"}, "goal_achieved": False})
    path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    config = read(path)
    config.update(latest_completed_study=rel(OUT / "result.json"), latest_report=result["report"], current_study=rel(OUT / "protocol.json"), latest_result_summary=result["discovery"], next_historical_question=result["next_historical_question"], updated_at=now(), goal_achieved=False)
    save(path, config)
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    config = read(path)
    config.update(latest_historical_index_participation_channels=rel(OUT / "result.json"), latest_historical_report=result["report"], latest_historical_diagnostic_at=now(), current_driver_continuation_classification=result["classification"], current_driver_consecutive_blocked_goal_turns=0, next_research_question=result["next_historical_question"], local_goal_work_status="ACTIVE_HISTORICAL_ONLY", goal_achieved=False)
    save(path, config)
    print("研究及图表完成，指数主线继续，夏普目标未达成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="固定半年指数交易需求研究")
    parser.add_argument("action", choices=["prepare", "collect", "build", "resolve", "plot", "publish", "finalize"])
    args = parser.parse_args()
    globals()[args.action]()
