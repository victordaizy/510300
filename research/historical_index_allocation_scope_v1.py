"""固定半年三个规模指数的重估、成交与基金份额范围比较。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
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
from research.historical_index_participation_channels_v1 import digest, now, read, rel, save
from research import historical_index_reopening_constraints_v1 as execution

OUT = ROOT / "reports/research/510300_historical_index_allocation_scope_v1"
REPORT = OUT / "历史发现_指数规模分化与融资范围.md"
PREVIOUS = ROOT / "reports/research/510300_historical_index_participation_channels_v1"
SECTOR = ROOT / "reports/research/510300_historical_index_sector_repricing_v1"
URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
INDEXES = {"000300": "沪深300", "000905": "中证500", "000852": "中证1000"}
POOLS = {"000300": ["510300.SH", "510310.SH", "510330.SH"], "000905": ["510500.SH", "512500.SH"]}
DOCS = [
    {"id": "csi500_method_2022", "url": "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000904_Index_Methodology_cn.pdf", "role": "中证200/500/700/800编制方案，检索标注2022年5月版，下载后核正文"},
    {"id": "csi1000_method_archive", "url": "https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/000852_Index_Methodology_cn.pdf", "role": "官方通用路径历史方案，版本日期须核正文，不用新版回填旧规则"},
    {"id": "rebalance_20221125_relay", "url": "https://www.cs.com.cn/gppd/gsyj/202211/t20221125_6310359.html", "role": "中国证券报转述中证指数当时定期调整公告，非官网原件"},
]


def prepare():
    p = {"study_id": "510300_HISTORICAL_INDEX_ALLOCATION_SCOPE_V1", "created_at": now(),
         "previous_goal_turn": "PROGRESS_INDEX_PARTICIPATION_SCOPE_AND_FINANCING_SUPPLY_SEPARATION",
         "question": "全市场融资增加与沪深300走弱并存时，500/1000是否表现不同；相对定价、成交活动和ETF份额是否支持同一解释？",
         "calendar": ["2022-09-30", "2023-03-31"], "indexes": INDEXES, "etf_pools": POOLS,
         "main_units": "六个完整自然月；另按上一轮的六组周快照边界和24周做范围衔接，边界不依结果变化",
         "known_history": "已见300及其行业、份额、融资结果，本轮不是独立盲测",
         "deduplication": {"old_cross_etf": "五条总份额规则失败，禁止救援", "old_etf_migration": "已有单300ETF相对同篮子份额迁移表达，不重跑", "old_cross_asset_rotation": "八价格因子跨资产轮动代码与当前仅510300/现金范围不同，不启用", "new_information": "两条官方规模指数的历史价格、PE和成交额，与原融资及固定ETF池作相同边界比较"},
         "data_request": "000905、000852各一次，20220930至20230331；000300复用已保存官方原文",
         "relative_states": "300回报严格高于另两者为300领先，严格低于两者为300落后，其余居中或持平；不预设该分类构成择时",
         "valuation": "log(P1/P0)=log(PE1/PE0)+log((P1/PE1)/(P0/PE0))，只作恒等式；隐含分母不是可比公司EPS，也不是前瞻盈利修正，PE项不是纯风险溢价",
         "activity": "对三只指数不重叠样本组的官方成交额求和份额；范围仅这三个指数，不称全A股，不称净流入。调样可能改变构成。",
         "share_proxy": "原固定10ETF集合内300三只与500两只，周净份额变化乘各自未复权周收盘价；1000基金份额本轮未取得，不新拼产品池",
         "causality": "相对上涨、成交份额提高和两市融资增加不能合称融资流入该指数；没有按当期成员汇总的融资交易及对手方证据",
         "comparison_clock": "所有六个月末T统计，保守按T+1收盘形成可用上界、T+2开盘看510300后5/20个交易日；不宣称证明了历史第一次可得时间",
         "horizons": [5,20], "cost": execution.COST, "market_end": "2023-05-31",
         "new_candidates": 0, "new_full_accounts": 0, "goal_achieved": False, "orders_authorized": False,
         "new_prospective_tasks": 0, "stop": "不把相对领先、成交占比或观测的赢家月份拼成买入规则；不能证明的资金归属和纯规模效应保留未知。"}
    save(OUT / "protocol.json", p, exclusive=True)
    save(OUT / "source_plan.json", DOCS, exclusive=True)
    print("固定三个指数、六个自然月和原周边界；本轮不运行轮动策略。")


def collect():
    if (OUT / "source_manifest.json").exists():
        raise RuntimeError("已保存请求结果，先读取状态，不重复采集。")
    folder = OUT / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    def fetch(code):
        params = {"indexCode": code, "startDate": "20220930", "endDate": "20230331"}
        record = {"id": code, "url": URL, "params": params, "retrieved_at": now()}
        try:
            response = requests.get(URL, params=params, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.csindex.com.cn/"}, timeout=(10,30))
            path = folder / (code + ".json")
            path.write_bytes(response.content)
            response.raise_for_status()
            rows = response.json()["data"]
            assert isinstance(rows, list) and rows and all(r["indexCode"] == code for r in rows)
            record.update(status="RETRIEVED", rows=len(rows), path=rel(path), sha256=digest(path))
        except Exception as exc:
            record.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc)[:350])
        return record
    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(fetch, ["000905", "000852"]))
    records.append({"id": "000300", "url": URL, "status": "REUSED", "path": rel(SECTOR / "sources/000300.json"), "sha256": digest(SECTOR / "sources/000300.json")})
    save(OUT / "source_manifest.json", records)
    print(json.dumps(records, ensure_ascii=False))


def context():
    folder = OUT / "sources"
    records = []
    for item in DOCS:
        record = dict(item, retrieved_at=now())
        try:
            response = requests.get(item["url"], headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.csindex.com.cn/"}, timeout=(10,30))
            response.raise_for_status()
            pdf = response.content.startswith(b"%PDF")
            path = folder / (item["id"] + (".pdf" if pdf else ".html"))
            path.write_bytes(response.content)
            if pdf:
                import pdfplumber
                with pdfplumber.open(path) as doc:
                    text = "\n".join(f"【PDF第{i+1}页】\n{page.extract_text() or ''}" for i,page in enumerate(doc.pages))
            else:
                response.encoding = response.apparent_encoding
                soup = BeautifulSoup(response.text, "html.parser")
                for tag in soup(["script", "style"]): tag.decompose()
                text = soup.get_text("\n", strip=True)
            (folder / (item["id"] + ".txt")).write_text(text, encoding="utf-8")
            record.update(status="SAVED_PENDING_REVIEW", path=rel(path), sha256=digest(path), characters=len(text))
        except Exception as exc:
            record.update(status="FAILED", error=type(exc).__name__ + ": " + str(exc)[:350])
        records.append(record)
    save(OUT / "context_source_manifest.json", records)
    print(json.dumps(records, ensure_ascii=False))


def compute():
    protocol = read(OUT / "protocol.json")
    frames = []
    for record in read(OUT / "source_manifest.json"):
        assert record["status"] in ["RETRIEVED", "REUSED"], record
        frame = pd.DataFrame(read(ROOT / record["path"])["data"])
        frame["date"] = pd.to_datetime(frame.tradeDate.astype(str), format="%Y%m%d")
        frame = frame.loc[frame.date.between(*protocol["calendar"])].sort_values("date")
        assert frame.date.is_unique and frame.indexCode.eq(record["id"]).all()
        for field in ["close", "peg", "tradingValue"]:
            frame[field] = pd.to_numeric(frame[field], errors="raise")
            assert frame[field].gt(0).all()
        frame["source_sha256"] = record["sha256"]
        frames.append(frame)
    all_daily = pd.concat(frames, ignore_index=True)
    prices = all_daily.pivot(index="date", columns="indexCode", values="close").sort_index()
    pe = all_daily.pivot(index="date", columns="indexCode", values="peg").reindex(prices.index)
    activity = all_daily.pivot(index="date", columns="indexCode", values="tradingValue").reindex(prices.index)
    assert prices.notna().all().all() and prices.index[0] == pd.Timestamp("2022-09-30") and prices.index[-1] == pd.Timestamp("2023-03-31")
    prevdaily = pd.read_parquet(PREVIOUS / "daily_participation.parquet").set_index("date")
    assert prices.index.equals(prevdaily.index), "三个指数与既有交易日历不一致"
    week_source = pd.read_parquet(ROOT / "data/raw/flow/510300_cross_etf_forced_flow_sse_weekly_v1.parquet")
    week_source["date"] = pd.to_datetime(week_source.date)
    tickers = [s for pool in POOLS.values() for s in pool]
    week_source = week_source.loc[week_source.ts_code.isin(tickers), ["date", "ts_code", "fund_shares"]]
    p = pd.read_parquet(ROOT / "data/raw/all_etf_momentum_v1r/etf_total_return_panel_tushare_adj.parquet", columns=["date", "con_code", "raw_close"])
    p["date"] = pd.to_datetime(p.date)
    p = p.loc[p.con_code.isin(tickers)].rename(columns={"con_code": "ts_code"})
    shares = week_source.merge(p, on=["date", "ts_code"], how="left", validate="one_to_one").sort_values(["ts_code", "date"])
    shares["share_change"] = shares.groupby("ts_code").fund_shares.diff()
    shares["proxy_cny"] = shares.share_change * shares.raw_close
    shares["index_code"] = shares.ts_code.map({s: code for code,pool in POOLS.items() for s in pool})
    shares = shares.loc[shares.date.between("2022-10-01", "2023-03-31")]
    assert shares[["raw_close", "share_change"]].notna().all().all()

    def interval(start, end, kind, label):
        ids = prices.index[(prices.index > start) & (prices.index <= end)]
        r = prices.loc[end] / prices.loc[start] - 1
        log_p = np.log(prices.loc[end] / prices.loc[start])
        log_pe = np.log(pe.loc[end] / pe.loc[start])
        log_d = np.log((prices.loc[end]/pe.loc[end]) / (prices.loc[start]/pe.loc[start]))
        assert np.max(np.abs(log_p - log_pe - log_d)) < 1e-12
        trading_share = activity.loc[ids].sum() / activity.loc[ids].to_numpy().sum()
        block = prevdaily.loc[ids]
        state = "300_LEADS" if (r["000300"] > r[["000905", "000852"]]).all() else "300_LAGS" if (r["000300"] < r[["000905", "000852"]]).all() else "MIXED_OR_TIED"
        row = {"kind": kind, "label": label, "start": start, "end": end, "trading_days": len(ids), "relative_state": state,
               "market_margin_balance_change_cny": float(prevdaily.at[end,"market_rzye"] - prevdaily.at[start,"market_rzye"]),
               "market_buy_mean_daily_cny": float(block.market_rzmre.mean()), "market_repay_mean_daily_cny": float(block.market_repay_implied.mean())}
        for code in INDEXES:
            row.update({f"return_{code}": float(r[code]), f"log_price_pp_{code}": float(log_p[code]*100),
                        f"log_pe_pp_{code}": float(log_pe[code]*100), f"log_denominator_pp_{code}": float(log_d[code]*100),
                        f"pe_start_{code}": float(pe.at[start,code]), f"pe_end_{code}": float(pe.at[end,code]),
                        f"trading_share_{code}": float(trading_share[code])})
        row["return_500_minus_300_pp"] = float((r["000905"] - r["000300"])*100)
        row["return_1000_minus_300_pp"] = float((r["000852"] - r["000300"])*100)
        if kind != "CALENDAR_MONTH":
            for code in POOLS:
                row[f"etf_proxy_{code}_cny"] = float(shares.loc[shares.index_code.eq(code) & shares.date.gt(start) & shares.date.le(end), "proxy_cny"].sum())
            row["etf_proxy_000852_cny"] = None
        return row

    monthly, snapshot, weekly = [], [], []
    start = pd.Timestamp("2022-09-30")
    for month in pd.period_range("2022-10", "2023-03", freq="M"):
        end = prices.index[prices.index.to_period("M") == month][-1]
        monthly.append(interval(start, end, "CALENDAR_MONTH", str(month)))
        start = end
    for row in pd.read_parquet(PREVIOUS / "monthly_comparison.parquet").itertuples():
        snapshot.append(interval(row.start, row.end, "SNAPSHOT_MONTH", row.month))
    for row in pd.read_parquet(PREVIOUS / "weekly_comparison.parquet").itertuples():
        weekly.append(interval(row.start, row.end, "WEEK", str(row.end.date())))
    old = pd.read_parquet(PREVIOUS / "monthly_comparison.parquet")
    assert np.max(np.abs(pd.DataFrame(snapshot).etf_proxy_000300_cny-old.same_index_pool_proxy_cny)) < 1e-4
    base_protocol = read(execution.OUT / "protocol.json")
    base_protocol["account_calendar"][1] = protocol["market_end"]
    market, _, dividends, engine, _ = execution.inputs(base_protocol)
    event_rows = []
    for row in monthly:
        positions = np.flatnonzero(market.date.gt(row["end"]))
        idx = int(positions[1])
        row["available_date_assumption"] = market.date.iloc[positions[0]]
        row["next_entry_date"] = market.date.iloc[idx]
        for h in protocol["horizons"]:
            event = execution.fixed_event_window(market, dividends, engine, {"event_id": row["label"], "title": "自然月末三个指数及融资观察", "entry_idx": idx}, h)
            row[f"next{h}_net_return"] = event["net_return"]
            event_rows.append(event)
    for name, frame in [("official_index_daily", all_daily), ("monthly_comparison", pd.DataFrame(monthly)), ("snapshot_comparison", pd.DataFrame(snapshot)), ("weekly_comparison", pd.DataFrame(weekly)), ("weekly_etf_components", shares)]:
        frame.to_parquet(OUT / (name+".parquet"), index=False)
        frame.to_csv(OUT / (name+".csv"), index=False, encoding="utf-8-sig")
    save(OUT / "event_returns.json", event_rows)
    save(OUT / "calculation_checks.json", {"index_series": 3, "dates_per_index": len(prices), "calendar_months": 6, "snapshot_intervals": len(snapshot), "weekly_intervals": len(weekly), "new_event_windows": len(event_rows), "filled_observations": 0, "same_300_share_proxy": "PASS", "log_decomposition_identity": "PASS", "historical_first_publication_verified": False, "csi1000_etf_share_proxy": "NOT_COMPUTED", "margin_by_index_constituents": "NOT_COMPUTED"})
    print(pd.DataFrame(monthly)[["label","return_000300","return_000905","return_000852","market_margin_balance_change_cny","relative_state","next20_net_return"]].to_string(index=False))


def rebalance_context():
    """解释原方案中观察到的分母跳变，不改收益窗口和分类。"""
    manifest = read(OUT / "context_source_manifest.json")
    folder = OUT / "sources"
    definitions = "".join((folder / "csi500_method_2022.txt").read_text(encoding="utf-8").split())
    assert "2022年5月更新" in definitions and "剔除沪深300指数样本" in definitions
    late_method = "".join((folder / "csi1000_method_archive.txt").read_text(encoding="utf-8").split())
    assert "2023年9月更新" in late_method
    relay = "".join((folder / "rebalance_20221125_relay.txt").read_text(encoding="utf-8").split())
    for term in ["12月9日收市后", "中证500指数更换50只样本", "11月23日收盘价", "11.32倍和17.84倍"]:
        assert term in relay, term
    for item in manifest:
        item["status"] = "LATER_VERSION_CONTEXT_ONLY" if item["id"] == "csi1000_method_archive" else "CONTENT_REVIEWED"
    path = folder / "csi1000_fund_20220803.pdf"
    extract_path = folder / "csi1000_fund_20220803_extracts.json"
    url = "https://static.cninfo.com.cn/finalpage/2022-08-03/1214207059.PDF"
    if not path.exists():
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=(10,30))
        response.raise_for_status()
        path.write_bytes(response.content)
    if not extract_path.exists():
        import pdfplumber
        with pdfplumber.open(path) as doc:
            snippets = [{"pdf_page": i+1, "text": doc.pages[i].extract_text() or ""} for i in [0,3]]
        save(extract_path, {"url": url, "retrieved_at": now(), "pages": snippets})
    excerpts = read(extract_path)
    words = "".join("".join(row["text"] for row in excerpts["pages"]).split())
    assert "2022年8月更新" in words and "剔除中证800指数" in words
    manifest.append({"id": "csi1000_fund_20220803", "url": url, "role": "南方基金2022年8月原说明书，巨潮承载；第4页说明1000指数与800的样本区别", "status": "CONTENT_REVIEWED", "path": rel(path), "sha256": digest(path), "extract_path": rel(extract_path), "reason_added": "官方通用路径实际为2023年9月版，补充期间之前的基本范围说明；原比较规则不变"})
    save(OUT / "context_source_manifest.json", manifest)
    all_daily = pd.read_parquet(OUT / "official_index_daily.parquet").sort_values(["indexCode", "date"])
    all_daily["implied_denominator"] = all_daily.close / all_daily.peg
    all_daily["price_change"] = all_daily.groupby("indexCode").close.pct_change(fill_method=None)
    all_daily["pe_change"] = all_daily.groupby("indexCode").peg.pct_change(fill_method=None)
    all_daily["denominator_change"] = all_daily.groupby("indexCode").implied_denominator.pct_change(fill_method=None)
    event = all_daily.loc[all_daily.date.between("2022-12-08", "2022-12-13")].copy()
    event.to_csv(OUT / "调样前后_估值与分母.csv", index=False, encoding="utf-8-sig")
    reference = all_daily.loc[all_daily.date.eq(pd.Timestamp("2022-11-23"))].set_index("indexCode")
    proforma = [{"index_code": code, "same_price_date": "2022-11-23", "historical_old_sample_pe": float(reference.at[code,"peg"]), "announced_new_sample_pe": pe_value, "pe_ratio_change": pe_value/reference.at[code,"peg"]-1, "source_id": "rebalance_20221125_relay", "pure_causal_effect_identified": False} for code, pe_value in [("000300", 11.32), ("000905", 17.84)]]
    csi500 = event.loc[event.indexCode.eq("000905") & event.date.eq(pd.Timestamp("2022-12-12"))].iloc[0]
    monthly = pd.read_parquet(OUT / "monthly_comparison.parquet")
    m = monthly.loc[monthly.label.eq("2022-12")].iloc[0]
    jump = float(np.log1p(csi500.denominator_change)*100)
    save(OUT / "rebalance_evidence.json", {"reviewed_at": now(), "followup_type": "POST_RESULT_EXPLANATORY_NO_SIGNAL_CHANGE", "reason": "固定月度比较显示500的12月隐含分母变化较大，按既有定期调样日核对；不删除该日，不重跑策略", "announcement_date": "2022-11-25", "effective_after_close": "2022-12-09", "first_new_sample_trading_day": "2022-12-12", "csi500_dec12": {"price_change": float(csi500.price_change), "pe_change": float(csi500.pe_change), "implied_denominator_change": float(csi500.denominator_change), "log_denominator_pp": jump}, "december_log_denominator_pp": float(m.log_denominator_pp_000905), "other_december_days_log_denominator_pp": float(m.log_denominator_pp_000905-jump), "same_price_date_comparison": proforma, "identified": "时间重合与当时调样测算共同支持样本组成影响，不能将整月隐含分母变化称为同一群公司的盈利变化", "not_identified": "未保持逐股利润与权重不变重算，不能断言12月12日全部变化仅由调样造成"})
    rows = []
    for state in ["300_LEADS", "300_LAGS", "MIXED_OR_TIED"]:
        block = monthly.loc[monthly.relative_state.eq(state)]
        rows.append({"state": state, "months": block.label.tolist(), "n": len(block), "next5_mean": float(block.next5_net_return.mean()), "next20_mean": float(block.next20_net_return.mean()), "next20_positive": int(block.next20_net_return.gt(0).sum()), "strategy_status": "NOT_A_TRADING_RULE"})
    save(OUT / "descriptive_groups.json", rows)
    print("调样后首日500市盈率上升5.61%，价格下降0.61%；仅解释口径，不改变原结果。")


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.sans-serif": ["Microsoft YaHei"], "axes.unicode_minus": False, "font.size": 10})
    d = pd.read_parquet(OUT / "monthly_comparison.parquet")
    x = np.arange(len(d))
    colors = {"000300": "#276C84", "000905": "#C59855", "000852": "#8E665E"}
    fig, axes = plt.subplots(2, 1, figsize=(12,8), sharex=True, gridspec_kw={"hspace": .30})
    fig.suptitle("同样的融资环境，三个规模指数出现不同定价", fontsize=18, fontweight="bold", y=.98)
    fig.text(.5,.927,"六个完整自然月｜沪深300为交易研究对象，500与1000只作观察",ha="center",color="#58656D")
    for offset, code in zip([-.25,0,.25], INDEXES):
        values = d[f"return_{code}"]*100
        bars = axes[0].bar(x+offset, values, width=.23, color=colors[code], label=INDEXES[code])
        axes[0].bar_label(bars, labels=[f"{v:+.1f}" for v in values], padding=3, fontsize=9)
    axes[0].axhline(0,color="#85939C",lw=.7)
    axes[0].set_ylabel("价格回报 / %")
    axes[0].set_ylim(-11,14)
    axes[0].legend(frameon=False,ncol=3,loc="upper right")
    axes[0].set_title("相对领先与绝对上涨分别记录",loc="left",fontsize=12)
    bottom = np.zeros(len(d))
    for code in INDEXES:
        values = d[f"trading_share_{code}"].to_numpy()*100
        axes[1].bar(x,values,bottom=bottom,color=colors[code],width=.63)
        for i,v in enumerate(values): axes[1].text(i,bottom[i]+v/2,f"{v:.1f}%",ha="center",va="center",color="white",fontsize=11)
        bottom += values
    axes[1].set_ylim(0,100)
    axes[1].set_ylabel("三指数观察范围内成交额占比")
    axes[1].set_title("成交更活跃不等于净买入；分母仅为这三个指数",loc="left",fontsize=12)
    axes[1].set_xticks(x,d.label)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y",alpha=.13)
        ax.set_axisbelow(True)
    fig.subplots_adjust(top=.855,bottom=.14,left=.10,right=.97)
    fig.text(.10,.058,"资料：中证指数官方历史数据。月末相对强弱不构成买入信号，未识别全市场融资的指数归属。\n指数样本和行业权重不同，收益差不能全部归因于市值大小；未使用当前成分重建历史。",fontsize=10,color="#58656D",linespacing=1.8)
    fig.savefig(OUT / "指数规模_定价与成交.png",dpi=160,facecolor="white")
    plt.close(fig)
    daily = pd.read_parquet(OUT / "official_index_daily.parquet")
    v = daily.loc[daily.indexCode.eq("000905") & daily.date.between("2022-12-08","2022-12-13")].sort_values("date").set_index("date")
    v["implicit"] = v.close/v.peg
    base = v.loc["2022-12-09"]
    fig,ax = plt.subplots(figsize=(10,5.3))
    for field,label,color in [("close","指数点位","#276C84"),("peg","市盈率","#C59855"),("implicit","点位/市盈率隐含分母","#8E665E")]:
        ax.plot(np.arange(4),v[field]/base[field]*100,marker="o",lw=2,label=label,color=color)
    ax.axvline(1.5,ls="--",lw=1,color="#A1ABB1")
    ax.axhline(100,lw=.6,color="#CAD1D5")
    ax.set_xticks(np.arange(4),[f"{z:%m-%d}" for z in v.index])
    ax.set_ylabel("2022-12-09 = 100")
    ax.set_ylim(92,109)
    ax.set_title("中证500：调样后的估值跳变不能直接写成盈利恶化",loc="left",fontsize=15,pad=17)
    ax.text(1.57,107.2,"12月9日收市后调样\n12日开始使用新样本",fontsize=10,color="#58656D")
    ax.legend(frameon=False,ncol=3,loc="lower left")
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=.12)
    fig.subplots_adjust(bottom=.21,top=.85,left=.09,right=.97)
    fig.text(.09,.07,"12月12日：指数 -0.61%，PE +5.61%，隐含分母 -5.88%。\n同期制度变化支持样本组成的解释；未识别该日变化的全部独立因果贡献。",fontsize=10,color="#58656D",linespacing=1.8)
    fig.savefig(OUT / "指数调样_市盈率与分母.png",dpi=160,facecolor="white")
    plt.close(fig)
    print("已生成规模指数对照与调样口径两张图。")


def publish():
    d = pd.read_parquet(OUT / "monthly_comparison.parquet")
    snapshot = pd.read_parquet(OUT / "snapshot_comparison.parquet")
    rebalance = read(OUT / "rebalance_evidence.json")
    groups = read(OUT / "descriptive_groups.json")
    rows = ["| 自然月 | 沪深300 | 中证500 | 中证1000 | 两市融资余额变化/亿元 | T+2开盘后20日510300净收益 |", "|---|---:|---:|---:|---:|---:|"]
    for r in d.itertuples():
        rows.append(f"| {r.label} | {r.return_000300:+.2%} | {r.return_000905:+.2%} | {r.return_000852:+.2%} | {r.market_margin_balance_change_cny/1e8:+.2f} | {r.next20_net_return:+.2%} |")
    etf_rows = ["| 周快照实际区间 | 300三ETF代理/亿元 | 500两ETF代理/亿元 | 中证500价格变化 |", "|---|---:|---:|---:|"]
    for r in snapshot.itertuples():
        etf_rows.append(f"| {r.start:%Y-%m-%d}→{r.end:%Y-%m-%d} | {r.etf_proxy_000300_cny/1e8:+.2f} | {r.etf_proxy_000905_cny/1e8:+.2f} | {r.return_000905:+.2%} |")
    discovery = "六个自然月确认规模指数分化：2023年2月两市融资增加429.45亿元，300跌2.10%，500涨1.09%，1000涨2.21%。但同周边界观察的500两ETF份额代理也为负，未证明融资或ETF资金搬家。2022年12月12日500价格跌0.61%而PE升5.61%，隐含分母跌5.88%，与调样生效衔接，不能解释成同一群企业单日盈利恶化。无新策略账户。"
    next_question = "先核2022Q4至2023Q1已有汇率、北向资金与国内约束放松研究的覆盖；以指数共同重估和规模分化为对照，查当时可知的配置驱动变化。仅有价格排名或聚合资金同向不新增策略，不复活既有跨境吸收及份额迁移失败表达。"
    state_names = {"300_LEADS":"300同时领先500和1000", "300_LAGS":"300同时落后500和1000", "MIXED_OR_TIED":"相对位置居中或持平"}
    group_rows = ["| 事先定义的描述类别 | 月份 | 20日正收益数 | 20日平均净收益 |", "|---|---|---:|---:|"]
    for g in groups:
        group_rows.append(f"| {state_names[g['state']]} | {'、'.join(g['months'])} | {g['next20_positive']}/{g['n']} | {g['next20_mean']:+.2%} |")
    text = f"""# 历史发现：指数规模分化与融资范围

本轮完成三条官方指数各120个日期、六个完整自然月、原六组周快照边界及24周的比较。新取得中证500和中证1000历史数据，沪深300、两市融资与ETF份额复用已保存资料。研究对象是沪深300整体；500和1000仅用于观察。没有新增轮动策略、全账户或前瞻任务，夏普1.2仍未达成。

**新发现是两类：全市场融资增强不能代表300需求增强；指数PE的变化还会受样本组成影响。** 它们修正了因子解释，但没有自动产生可交易优势。

![规模与成交](<{(OUT / '指数规模_定价与成交.png').as_posix()}>)

{chr(10).join(rows)}

上表使用每个自然月的最后交易日，价格回报不含指数分红。2023年2月两市融资余额增加429.45亿元，300跌2.10%，500和1000分别涨1.09%、2.21%。这支持当期存在范围分化，不能继续用“融资增加→所有股票风险偏好同步上升→买300”代替传导分析。2023年3月融资继续增加248.63亿元，三指数又全部下跌，融资与广泛上涨也没有固定一一关系。

成交活动提供另一条观察：300在这三个指数合计成交额中的占比从2023年1月44.98%降到2月39.87%；1000从30.85%升到34.73%。这与中小市值样本的相对活跃相符。成交额同时包含买卖双方，样本的市值、换手率与调样也不同，不能把占比变化当作资金净迁移，更未识别新增融资买到了哪些指数样本。

ETF渠道还有一个反例。采用上一轮相同周快照边界时，两只500ETF也出现份额净减少：

{chr(10).join(etf_rows)}

300池固定为510300、510310、510330；500池固定为原十ETF集合内的510500、512500。每周净份额变化乘各自当周未复权收盘价后求和，叫份额转换市值代理，不叫实收现金。1000的份额代理本轮没有取得，保留缺失。两个池均非全品种覆盖，也不涵盖直接股票、衍生品和其他资金渠道。

2023年1月20日至2月24日，500指数上涨1.43%，其两ETF代理却为-186.14亿元。因此，300ETF份额减少和500指数上涨，并不足以证明资金从300ETF搬到500ETF。实物转换、套利、其他产品及直接股票需求仍是未分开的解释。相对强弱是真实价格观察，资金归属尚未识别。

另一个发现来自指数估值本身。以同一公式分解每个月的指数点位和PE后，500在2022年12月的隐含分母出现-5.9641个对数百分点变化。这里的“点位/PE”是统计分母，不能当作同一组公司可比EPS。按早已公布的调样边界核对，12月12日的分母变动为-6.0632个对数百分点，其余12月交易日合计+0.0991个对数百分点。

![调样与估值](<{(OUT / '指数调样_市盈率与分母.png').as_posix()}>)

12月9日至12日，中证500点位6192.30→6154.81，PE从17.12→18.08：价格下降0.61%，PE上升5.61%，隐含分母下降5.88%。当时报道已经明确12月9日收市后定期调整生效，中证500更换50个样本。按同一个11月23日收盘价，公告对新样本测算的PE为17.84；保存的旧样本历史PE为16.92。新旧统计对象的变化可以在没有对应价格上涨的情况下改变PE。[当时调样报道]({DOCS[2]['url']})

**因此，这个跳变不能直接写成“企业盈利单日恶化5.88%”，PE抬升也不能全部写成“投资者提高了估值”。** 调样时间和当时同价格日期的测算支持组成影响的解释；本轮未保持逐股盈利、权重和计算版本恒定重算，未证明12月12日全部变化都由调样造成。也没有删除该日后重跑失败策略。这一核对由固定月度结果触发，属于事后解释，已单独记录。

指数与个股的区别在这里很具体：指数是会调整样本和权重的组合；不同规模指数还带有不同的行业暴露。500与1000跑赢300，不能直接称为“纯市值因子有效”。要把公司经营变化、组合结构变化和价格变化分开，才能进一步讨论哪种预期差尚未反映。

历史范围依据也保留版本：500编制方案为2022年5月版，明确在300之外选样；1000官方通用下载路径实际为2023年9月版，没有拿它回填2022年。补充的2022年8月南方基金原说明书第4页明确1000剔除800样本，说明三个指数的基本范围不同。指数收益直接来自官方当期历史点位，没有使用今天的成分倒算。[500当期方案]({DOCS[0]['url']})、[1000期间之前的说明书](https://static.cninfo.com.cn/finalpage/2022-08-03/1214207059.PDF)

相对领先是否提供300的剩余收益，按原方案只做以下描述，不按结果选方向：

{chr(10).join(group_rows)}

各类仅两个自然月。300领先的两次后20日一正一负，落后的两次也一正一负；居中两次均负不能据此反向建模。这些窗以月末T统计、T+1收盘的保守可用假设、T+2开盘进入510300，沿用5/20交易日、每边万4佣金/最低5元/千1滑点、整手、T+1与分红处理。10万元名义事件窗口的投入资金净收益不是20万元完整账户，窗口重叠，不能累加或年化夏普。历史数据后来获取，保守延迟也不证明首发时刻。

本轮复用了原成本与成交实现，检查只覆盖必要算术：三指数交易日期一致、无补行、份额代理和上一轮完全对上、PE分解恒等式、12个新增收益窗口的成交与分红。既有份额、溢价和迁移规则的失败记录保持不变。新增的价格范围与调样事实没有被包装成策略通过。

后续问题回到上游：已有的汇率、北向资金与国内活动约束证据，能否解释这段历史里共同重估和规模分化为何交替出现。若只有资金与价格同向，仍不足以形成买入条件；优先查未覆盖的机制及当时信息，避免继续堆相关指标。

原始两条指数请求位于`sources/000905.json`和`sources/000852.json`，300原始文件路径在`source_manifest.json`。自然月、原周边界、逐周比较分别为`monthly_comparison.csv`、`snapshot_comparison.csv`、`weekly_comparison.csv`；调样解释与完整算术分别见`rebalance_evidence.json`、`调样前后_估值与分母.csv`。报告中的“新样本PE”属于当时报道的测算，和实际生效后PE分别保存。
"""
    REPORT.write_text(text, encoding="utf-8")
    save(OUT / "result.json", {"study_id": read(OUT / "protocol.json")["study_id"], "completed_at": now(), "status": "COMPLETED_HISTORICAL_SCOPE_AND_MEASUREMENT_DISCOVERY", "classification": "PROGRESS_INDEX_SIZE_SCOPE_AND_REBALANCE_DENOMINATOR", "report": rel(REPORT), "figures": [rel(OUT / "指数规模_定价与成交.png"),rel(OUT / "指数调样_市盈率与分母.png")], "figure_visually_reviewed": False, "report_status": "PENDING_REVIEW", "discovery": discovery, "next_historical_question": next_question, "new_index_series": 2, "days_per_series": 120, "new_candidates": 0, "new_full_accounts": 0, "new_prospective_tasks": 0, "net_sharpe": None, "net_cagr": None, "max_drawdown": None, "goal_achieved": False, "orders_authorized": False, "independent_validation": False, "new_event_windows": 12, "limitations": ["未取得按当期指数样本归属的融资买入", "未识别纯规模效应", "两组ETF覆盖不完整", "调样因果贡献没有逐项恒定重建", "历史第一发布时间未证明"]})
    print(discovery)


def finalize():
    result = read(OUT / "result.json")
    result.update(figure_visually_reviewed=True, report_status="REVIEWED", updated_at=now())
    save(OUT / "result.json", result)
    save(OUT / "research_receipt.json", {"updated_at": now(), "script_sha256": digest(Path(__file__)), "output_hashes": {rel(p): digest(p) for p in OUT.iterdir() if p.is_file() and p.name != "research_receipt.json"}, "goal_achieved": False})
    path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    config = read(path)
    config.update(latest_completed_study=rel(OUT / "result.json"), latest_report=result["report"], current_study=rel(OUT / "protocol.json"), latest_result_summary=result["discovery"], next_historical_question=result["next_historical_question"], updated_at=now(), goal_achieved=False)
    save(path, config)
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    config = read(path)
    config.update(latest_historical_index_allocation_scope=rel(OUT / "result.json"), latest_historical_report=result["report"], latest_historical_diagnostic_at=now(), current_driver_continuation_classification=result["classification"], current_driver_consecutive_blocked_goal_turns=0, next_research_question=result["next_historical_question"], local_goal_work_status="ACTIVE_HISTORICAL_ONLY", goal_achieved=False)
    save(path, config)
    print("指数范围与调样研究完成；原账户目标和风险参数不变。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="指数范围与融资需求的历史比较")
    parser.add_argument("action", choices=["prepare", "collect", "context", "compute", "rebalance_context", "plot", "publish", "finalize"])
    args = parser.parse_args()
    globals()[args.action]()
