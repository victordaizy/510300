"""固定2019年第二季度，比较利率构成、会议内重定价和贸易消息时序。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
from io import StringIO
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import pdfplumber
import requests

from historical_index_liquidity_transmission_v1 import clean
from historical_index_us_rates_policy_v1 import curve, asof_rate, DKW_FIELDS
from historical_price_gap_causes_v1 import round_trip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_index_us_rates_trade_q2_v1"
OLD = ROOT / "reports/research/510300_historical_index_us_rates_policy_v1"
FLOW = ROOT / "reports/research/510300_historical_index_credit_flow_composition_v1"
FOMC = ROOT / "reports/research/510300_historical_fomc_transmission_v1"
EVENTS = ["2019-05-01", "2019-06-19"]
REPORT = "历史发现_2019Q2利率缓和与贸易冲击.md"
SOURCES = [
    {"id": "fed_20190501", "name": "fed_20190501.html", "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20190501a.htm"},
    {"id": "fed_20190619", "name": "fed_20190619.html", "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20190619a.htm"},
    {"id": "press_20190501", "name": "press_20190501.pdf", "url": "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20190501.pdf"},
    {"id": "press_20190619", "name": "press_20190619.pdf", "url": "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20190619.pdf"},
    {"id": "trump_20190505", "name": "trump_20190505.html", "url": "https://www.presidency.ucsb.edu/documents/tweets-may-5-2019"},
    {"id": "china_tariff_20190513", "name": "china_tariff_20190513.html", "url": "https://gss.mof.gov.cn/gzdt/zhengcefabu/201905/t20190513_3256787.htm"},
    {"id": "g20_20190629", "name": "g20_20190629.html", "url": "https://www.fmprc.gov.cn/web/gjhdq_676201/gj_676203/bmz_679954/1206_680528/xgxw_680534/201906/t20190629_9360943.shtml"},
]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("本轮方案已存在，不覆盖。")
    old = read(OLD / "protocol.json")
    existing = read(FOMC / "result.json")
    save("protocol.json", {
        "study_id": "510300_HISTORICAL_INDEX_US_RATES_TRADE_Q2_V1", "recorded_at": now(),
        "previous_goal_turn_classification": "PROGRESS", "previous_result": (OLD / "result.json").relative_to(ROOT).as_posix(),
        "mode": "HISTORICAL_DISCOVERY_ONLY", "research_unit": "沪深300整体与510300，不扩展个股",
        "question": "相邻2019Q2中，美债下降的构成和政策回应能否解释指数阶段差异？会议内预期路径重定价、交易前等待与新贸易消息是否改变解释？",
        "calendar": ["2019-04-01", "2019-06-30"], "baseline_month": "2019-03", "events": EVENTS,
        "selection": "季度及全部两次会议承接上轮写明的下一步；历史价格与部分美国事件反应已见，不是独立盲测。",
        "horizons": [5, 20], "costs": old["costs"],
        "same_return_contract": "完整会议窗口结束之后第一中国交易日开盘买入，入场后5/20交易日开盘卖出；固定窗口，完整保留亏损，不按关税消息撤单重选样本。",
        "price_context": "此前20/60交易日、前收至开盘缺口、入场后净收益分别计算；不将假期缺口归为纯FOMC效应，也不将已跌部分记作空仓收益。",
        "monthly_rates": "按中国月末21:00沿旧保守美国日终时钟对齐；同日名义/TIPS内连接，另保留延迟一条美国观测。季度端点及当日PE仅作事后描述。",
        "intraday": "复用USMPD现有文件，固定2018Q4—2019Q2全部六次预定会议，保留声明、记者会和完整事件三个窗口；OIS及债券百分点乘100转bp，美股/美元原字段是百分比。",
        "intraday_limits": "2026版历史重建，非2019年数据库下载记录；采用原始报价变化，不拟合PCA、不把单一OIS变化叫纯政策冲击、不设新分组交易规则。",
        "nearest_old_study": {"path": FOMC.relative_to(ROOT).as_posix(), "period": "2022—2025全部32次会议",
                              "status": existing["status"], "difference": "新工作是已声明的2019相邻季度机制对照及交易日等待解释；不复活旧利率降股涨买入规则。"},
        "dkw": "沿已有2025-11-12重估版只作历史构成解释，禁止当成当年输入；保留模型拟合残差与Treasury曲线差异。",
        "trade_context": "5月5日公开推文、5月10日美方实施、5月13日中方公告、6月29日G20声明；材料时间与经济生效时间分开。已知新闻与持有期新消息不混用。",
        "sources": SOURCES, "existing_inputs": old["input_paths"],
        "usmpd": (FOMC / "sources/USMPD_20260930.xlsx").relative_to(ROOT).as_posix(),
        "anti_overfit": ["不延长到多年筛选", "不凭季度结果调方向", "不以新消息删掉亏损", "不调期限或阈值", "不把事件平均转成年化夏普"],
        "full_account_runs": 0, "parameters_fitted": 0, "goal_achieved": False, "orders_authorized": False,
    })
    print("已保存固定2019Q2及两次会议；旧32次会议失败结果继续保留。", flush=True)


def fetch_one(source):
    path = OUT / "sources" / source["name"]
    if path.exists():
        raise RuntimeError("来源已经保存，不重复请求：" + source["id"])
    receipt = {**source, "retrieved_at": now(), "historical_download_receipt": False}
    try:
        response = requests.get(source["url"], timeout=(15, 50), headers={"User-Agent": "Mozilla/5.0"})
        receipt.update(http_status=response.status_code, final_url=response.url)
        response.raise_for_status()
        raw = response.content
        path.write_bytes(raw)
        receipt.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        if path.suffix == ".pdf":
            with pdfplumber.open(path) as document:
                text = "\n".join(f"第{i + 1}页\n{page.extract_text() or ''}" for i, page in enumerate(document.pages))
        else:
            response.encoding = response.apparent_encoding
            text = BeautifulSoup(response.text, "html.parser").get_text("\n", strip=True)
        path.with_suffix(".txt").write_text(text, encoding="utf-8")
        receipt["status"] = "SAVED"
    except Exception as error:
        receipt.update(status="ERROR", error=str(error))
    save("sources/" + source["name"] + ".receipt.json", receipt)
    return receipt


def fetch():
    (OUT / "sources").mkdir(exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(fetch_one, source) for source in SOURCES]):
            row = future.result()
            rows.append(row)
            print(row["id"], row["status"], flush=True)
    save("source_manifest.json", sorted(rows, key=lambda row: row["id"]))
    if any(row["status"] == "ERROR" for row in rows):
        raise RuntimeError("存在来源缺口，请核对具体回执。")


def build_intraday(protocol):
    path = ROOT / protocol["usmpd"]
    full = pd.read_excel(path, sheet_name="Monetary Events")
    statements = pd.read_excel(path, sheet_name="Statements").set_index("Date")
    conferences = pd.read_excel(path, sheet_name="Press Conferences").set_index("Date")
    chosen = full.loc[full.Date.between("2018-10-01", "2019-06-30") & full.Unscheduled.eq(0)].copy()
    assert len(chosen) == 6 and not chosen.Date.duplicated().any()
    records = []
    for row in chosen.to_dict("records"):
        date = pd.Timestamp(row["Date"])
        statement = statements.loc[date]
        pc = conferences.loc[date] if row["PC"] else None
        end = (pd.Timestamp(pc.date_time) + pd.Timedelta(minutes=60)) if pc is not None else (pd.Timestamp(row["date_time"]) + pd.Timedelta(minutes=20))
        end_cn = end.tz_localize("America/New_York").tz_convert("Asia/Shanghai")
        record = {"event_id": date.strftime("%Y-%m-%d"), "full_end_cn": end_cn,
                  "statement_at_cn": pd.Timestamp(row["date_time"]).tz_localize("America/New_York").tz_convert("Asia/Shanghai"),
                  "has_press_conference": bool(row["PC"])}
        for prefix, data in [("full", row), ("statement", statement), ("conference", pc)]:
            for field in ["OIS1Y", "UST10Y", "TIPS10Y", "SP500", "DXY"]:
                scale, suffix = (100, "bp") if field in ["OIS1Y", "UST10Y", "TIPS10Y"] else (1, "pct")
                record[f"{prefix}_{field}_{suffix}"] = None if data is None or pd.isna(data[field]) else float(data[field] * scale)
        if pc is not None:
            assert abs(record["full_OIS1Y_bp"] - record["statement_OIS1Y_bp"] - record["conference_OIS1Y_bp"]) < 1e-7
            compounded = ((1 + record["statement_SP500_pct"] / 100) * (1 + record["conference_SP500_pct"] / 100) - 1) * 100
            assert abs(compounded - record["full_SP500_pct"]) < 1e-7
        record["statement_full_ois_sign_reversal"] = record["statement_OIS1Y_bp"] * record["full_OIS1Y_bp"] < 0
        record["pure_policy_shock_identified"] = False
        records.append(record)
    save("six_meetings_intraday.json", records)
    return records


def calculate():
    if (OUT / "events_and_returns.json").exists():
        raise RuntimeError("结果已存在，不改变窗口重算。")
    protocol = read(OUT / "protocol.json")
    inputs = protocol["existing_inputs"]
    intra = build_intraday(protocol)
    nominal = pd.concat([curve(ROOT / inputs[f"nominal{year}"]) for year in [2018, 2019]])
    tips = pd.concat([curve(OLD / f"sources/tips_{year}.xml", real=True) for year in [2018, 2019]])
    rates = nominal.merge(tips, on="date", validate="one_to_one").sort_values("date").reset_index(drop=True)
    rates["inflation_compensation"] = rates.nominal10 - rates.tips10
    rates["known_at_assumed"] = (rates.date + pd.Timedelta(hours=23, minutes=59, seconds=59)).dt.tz_localize("America/New_York").dt.tz_convert("Asia/Shanghai")
    rates["known_at_delay1_assumed"] = rates.known_at_assumed.shift(-1)
    for field in ["nominal10", "tips10", "inflation_compensation"]:
        rates[field + "_20obs_bp"] = rates[field].diff(20) * 100
    rates.loc[rates.date.between("2019-02-01", "2019-07-31")].to_parquet(OUT / "rates.parquet", index=False)
    market = pd.read_parquet(ROOT / inputs["market"]).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).dt.tz_localize(None).dt.normalize()
    calendar = pd.DatetimeIndex(market.date)
    opens = calendar.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=9, minutes=30)
    price = pd.read_parquet(ROOT / inputs["price"]).sort_values("date").set_index("date")
    pe = pd.read_parquet(ROOT / inputs["pe"])
    pe["date"] = pd.to_datetime(pe.observation_date)
    pe["available_at"] = pd.to_datetime(pe.available_at, utc=True).dt.tz_convert("Asia/Shanghai")
    panel = price.reset_index()[["date", "close"]].merge(pe[["date", "available_at", "original_pe_official"]], on="date", validate="one_to_one").sort_values("date").set_index("date")
    dividends = pd.read_csv(ROOT / inputs["dividends"])
    dividends = dividends.loc[dividends.symbol.eq("510300.SH")].copy()
    dividends["record_date"] = pd.to_datetime(dividends.record_date)
    states, trades = [], []
    for date in EVENTS:
        window = next(row for row in intra if row["event_id"] == date)
        entry_pos = int(opens.searchsorted(window["full_end_cn"], side="right"))
        entry, before = market.iloc[entry_pos], market.iloc[entry_pos - 1]
        last_price = price.loc[price.index < entry.date].iloc[-1]
        pos = price.index.get_loc(last_price.name)
        last_pe = panel.loc[panel.available_at.le(opens[entry_pos])].iloc[-1]
        state = {"event_id": date, "published_cn": window["statement_at_cn"], "full_window_end_cn": window["full_end_cn"],
                 "entry_at": opens[entry_pos], "entry_date": entry.date.strftime("%Y-%m-%d"), "wait_hours": (opens[entry_pos] - window["full_end_cn"]).total_seconds() / 3600,
                 "previous_china_close_date": before.date, "etf_gap_return": float((entry.open + entry.dividend) / before.close - 1),
                 "index_close_before": last_price.close, "index_close_before_date": last_price.name,
                 "lookbacks": [{"horizon": h, "start_date": price.index[pos - h], "end_date": last_price.name,
                                "price_return": float(last_price.close / price.iloc[pos - h].close - 1)} for h in [20, 60]],
                 "pe": last_pe.original_pe_official, "pe_date": last_pe.name, "pe_available_at": last_pe.available_at,
                 "curve_at_entry": asof_rate(rates, opens[entry_pos]), "delay1_curve_at_entry": asof_rate(rates, opens[entry_pos], True),
                 "uncontaminated_fomc_to_etf_effect_identified": False}
        assert window["full_end_cn"] < opens[entry_pos] and last_pe.available_at <= opens[entry_pos]
        states.append(state)
        for horizon in protocol["horizons"]:
            end = market.iloc[entry_pos + horizon]
            div = float(dividends.loc[dividends.record_date.ge(entry.date) & dividends.record_date.lt(end.date), "cash_dividend_per_share"].sum())
            trade = round_trip(float(entry.open), float(end.open), div, protocol["costs"])
            assert trade["paid_cny"] <= protocol["costs"]["illustrative_event_budget_cny"]
            assert end.date > entry.date
            error = (trade["sell_price"] - trade["buy_price"]) * trade["shares"] + trade["dividend_entitlement_cny"] - trade["commissions_cny"] - trade["net_pnl_cny"]
            assert abs(error) < 1e-7
            trades.append({"event_id": date, "horizon": horizon, "entry_date": entry.date.strftime("%Y-%m-%d"), "exit_date": end.date.strftime("%Y-%m-%d"),
                           "entry_open": float(entry.open), "exit_open": float(end.open), "dividend_per_share": div,
                           "gross_return": float((end.open + div) / entry.open - 1), **trade})
    monthends = []
    for month in pd.period_range("2019-03", "2019-06", freq="M"):
        date = calendar[calendar.to_period("M") == month][-1]
        cutoff = date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=21)
        row = panel.loc[date]
        monthends.append({"cn_date": date, "cutoff": cutoff, "index_close": row.close,
                          "same_day_pe_retrospective": row.original_pe_official, "pe_available_at": row.available_at,
                          "rates": asof_rate(rates, cutoff), "rates_delay1": asof_rate(rates, cutoff, True)})
    a, b = monthends[0], monthends[-1]
    quarter = {"period": "2019Q2", "cn_start": a["cn_date"], "cn_end": b["cn_date"],
               "us_start": a["rates"]["us_observation_date"], "us_end": b["rates"]["us_observation_date"],
               "index_price_return": b["index_close"] / a["index_close"] - 1,
               "same_day_pe_change_retrospective": b["same_day_pe_retrospective"] / a["same_day_pe_retrospective"] - 1}
    for field in ["nominal10", "tips10", "inflation_compensation", "nominal2"]:
        quarter[field + "_bp"] = 100 * (b["rates"][field] - a["rates"][field])
        quarter[field + "_delay1_bp"] = 100 * (b["rates_delay1"][field] - a["rates_delay1"][field])
    assert abs(quarter["nominal10_bp"] - quarter["tips10_bp"] - quarter["inflation_compensation_bp"]) < 1e-10
    monthly_changes = []
    for a, b in zip(monthends[:-1], monthends[1:]):
        monthly_changes.append({"cn_start": a["cn_date"], "cn_end": b["cn_date"], "us_start": a["rates"]["us_observation_date"],
                                "us_end": b["rates"]["us_observation_date"], "index_price_return": b["index_close"] / a["index_close"] - 1,
                                **{field + "_bp": 100 * (b["rates"][field] - a["rates"][field]) for field in ["nominal10", "tips10", "inflation_compensation"]}})
    raw = (ROOT / inputs["dkw"]).read_text(encoding="utf-8")
    header = next(i for i, line in enumerate(raw.splitlines()) if line.startswith('"date",'))
    dkw = pd.read_csv(StringIO(raw), skiprows=header, parse_dates=["date"]).set_index("date")
    delta = (dkw.loc[quarter["us_end"], DKW_FIELDS] - dkw.loc[quarter["us_start"], DKW_FIELDS]) * 100
    assert abs(sum(delta[key] for key in DKW_FIELDS[:4]) - delta["nominal.yield.fitted.10"]) < 1e-3
    assert abs(delta["exp.inflation.10"] + delta["inflation.risk.prem.10"] - delta["tips.liq.prem.10"] - delta["ic.fitted.10"]) < 1e-3
    decomposition = {"period": "2019Q2", "us_start": quarter["us_start"], "us_end": quarter["us_end"], "changes_bp": delta.to_dict(),
                     "nominal_residual_change_bp": delta["nominal.yield.raw.10"] - delta["nominal.yield.fitted.10"],
                     "ic_residual_change_bp": delta["ic.raw.10"] - delta["ic.fitted.10"],
                     "nominal_cmt_minus_dkw_raw_change_bp": quarter["nominal10_bp"] - delta["nominal.yield.raw.10"],
                     "historical_entry_input_admitted": False, "model_refit_through": "2025-11-12"}
    old_events = {row["id"]: row for row in read(OLD / "event_states.json")}
    for window in intra:
        if window["event_id"] in old_events:
            pos = int(opens.searchsorted(window["full_end_cn"], side="right"))
            assert market.iloc[pos].date.strftime("%Y-%m-%d") == old_events[window["event_id"]]["entry_date"]
    save("event_states.json", states)
    save("events_and_returns.json", trades)
    save("monthends.json", monthends)
    save("monthly_changes.json", monthly_changes)
    save("quarter_comparison.json", [*read(OLD / "quarter_comparison.json"), quarter])
    save("dkw_retrospective_decomposition.json", [*read(OLD / "dkw_retrospective_decomposition.json"), decomposition])
    market.loc[market.date.between("2019-03-29", "2019-07-18")].to_parquet(OUT / "etf_context.parquet", index=False)
    price.loc["2019-03-29":"2019-07-18"].to_parquet(OUT / "index_context.parquet")
    save("calculation_checks.json", {"recorded_at": now(), "intraday_events": len(intra), "new_event_nodes": len(states), "new_cost_windows": len(trades),
                                    "intraday_additive_rates_and_compounded_stocks": "PASS", "old_four_entry_dates_unchanged": True,
                                    "curve_and_model_identities": "PASS", "account_cost_identity": "PASS", "entry_clock": "PASS",
                                    "new_full_accounts": 0, "goal_achieved": False})
    used = {**inputs, "usmpd": protocol["usmpd"], "tips2019": (OLD / "sources/tips_2019.xml").relative_to(ROOT).as_posix()}
    save("input_receipts.json", [{"id": key, "path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()} for key, path in used.items()])
    print(pd.DataFrame([quarter])[["period", "nominal10_bp", "tips10_bp", "inflation_compensation_bp", "index_price_return"]].to_string(index=False), flush=True)
    print(pd.DataFrame(trades)[["event_id", "horizon", "entry_date", "exit_date", "net_return"]].to_string(index=False), flush=True)
    print(pd.DataFrame(states)[["event_id", "wait_hours", "etf_gap_return"]].to_string(index=False), flush=True)


def supplement():
    source = {"id": "call_20190618", "name": "call_20190618.html",
              "url": "https://www.mfa.gov.cn/web/zyxw/201906/t20190618_346718.shtml"}
    receipt = fetch_one(source)
    save("supplement_source.json", {"recorded_at": now(), "role": "EXPLANATION_AFTER_RETURN_REVIEW_NO_RULE_CHANGE",
                                   "reason": "检查6月盈利的共同消息时，补核会前中美通话；不改变两次会议的入场或退出。", "receipt": receipt})
    if receipt["status"] != "SAVED":
        raise RuntimeError("6月18日通话来源仍有缺口。")
    print("已补存6月18日会前通话原文，只用于解释共同信息。", flush=True)


def evidence():
    specs = [
        ("fed_20190501", ["will be patient", "below 2 percent"], "2019-05-01", "维持2.25%—2.50%目标区间并延续耐心；总体和核心通胀低于2%。"),
        ("press_20190501", ["transitory factors", "a strong case for moving in either direction", "technical adjustment"], "2019-05-01", "认为部分通胀偏弱来自暂时因素，暂无向任一方向调整的充分理由；IOER下调属于技术实施调整，不代表目标政策立场改变。"),
        ("fed_20190619", ["uncertainties about this outlook have increased", "will act as appropriate"], "2019-06-19", "目标区间仍不变，但展望不确定性增大，指引转为将采取适当行动维持扩张。"),
        ("press_20190619", ["Apparent progress on trade turned to greater uncertainty", "Risk sentiment in financial markets has deteriorated"], "2019-06-19", "上游原因是贸易谈判从进展变为不确定、全球增长信息转弱、企业信心与金融风险偏好走弱；政策缓和与经济担忧同时存在。"),
        ("trump_20190505", ["16:08:46", "The 10% will go up to 25% on Friday"], "2019-05-05", "公开表示将相关2000亿美元商品税率从10%提高至25%；只记录宣布内容，不把有关成本承担者的政治表述当作经济测量。"),
        ("china_tariff_20190513", ["600亿美元", "2019年6月1日0时", "25%、20%或10%"], "2019-05-13", "中方宣布6月1日起对600亿美元清单中部分美国商品提高加征税率；公告日与生效日分开。"),
        ("g20_20190629", ["2019-06-29 13:38", "重启经贸磋商", "不再对中国产品加征新的关税"], "2019-06-29", "大阪会晤后同意重启磋商，美方不再加征新的关税；这不等于撤销已有税率。"),
        ("call_20190618", ["2019年6月18日", "大阪峰会", "经贸团队"], "2019-06-18", "美联储6月会议之前，中美元首已通话并同意在大阪会晤；贸易接触改善不是6月20日开仓以后才出现的信息。"),
    ]
    sources = {row["id"]: row for row in SOURCES}
    sources["call_20190618"] = read(OUT / "supplement_source.json")["receipt"]
    rows = []
    for source_id, needles, date, fact in specs:
        path = OUT / "sources" / (source_id + ".txt")
        if not path.exists():
            path = OUT / "sources" / (source_id + ".web.txt")
        compact = " ".join(path.read_text(encoding="utf-8").split())
        for needle in needles:
            assert needle in compact, (source_id, needle)
        rows.append({"source_id": source_id, "public_event_date": date, "url": sources[source_id]["url"], "fact": fact,
                     "role": "PUBLIC_HISTORICAL_EXPLANATION_NOT_CAUSAL_IDENTIFICATION"})
    reused = read(FLOW / "supplement_sources.json")
    ustr = next(row for row in reused if row["id"] == "ustr_may10")
    raw_ustr = (FLOW / "sources/ustr_may10.txt").read_text(encoding="utf-8")
    assert "10 percent to 25 percent" in raw_ustr and "$200 billion" in raw_ustr
    rows.append({"source_id": "ustr_may10", "public_event_date": "2019-05-10", "url": ustr["url"],
                 "fact": "USTR确认当日已对约2000亿美元中国进口商品将税率从10%提高到25%，其他商品程序尚在启动。", "role": "REUSED_PRIMARY_DOCUMENT"})
    tweet_shanghai_candidates = [pd.Timestamp("2019-05-05 16:08:46", tz=zone).tz_convert("Asia/Shanghai") for zone in ["UTC", "America/New_York"]]
    bound = max(tweet_shanghai_candidates)
    may_state = read(OUT / "event_states.json")[0]
    assert bound < pd.Timestamp(may_state["entry_at"])
    timeline = [
        {"source_id": "trump_20190505", "phase_for_may_entry": "BEFORE_ENTRY_UNDER_BOTH_CLOCK_INTERPRETATIONS", "known_at_upper_assumed": bound,
         "clock_note": "档案16:08:46未标时区，UTC/美东两种解释都早于5月6日开盘；取较晚时刻作保守上界，不声称精确时区已核实。"},
        {"source_id": "ustr_may10", "known_at_assumed": "2019-05-11T11:59:59+08:00", "phase_for_may_entry": "DURING_BOTH_5_AND_20_DAY_WINDOWS", "clock_note": "美国公布日终保守假设。"},
        {"source_id": "china_tariff_20190513", "known_at_assumed": "2019-05-13T23:59:59+08:00", "phase_for_may_entry": "AFTER_5_DAY_EXIT_INSIDE_20_DAY_WINDOW", "clock_note": "仅核实公布日；5日窗口在5月13日09:30已退出，不能将此消息倒填。"},
        {"source_id": "call_20190618", "known_at": "2019-06-18T22:50:00+08:00", "phase_for_june_entry": "BEFORE_FOMC_AND_ENTRY", "clock_note": "外交部页面标注公开时间。"},
        {"source_id": "g20_20190629", "known_at": "2019-06-29T13:38:00+08:00", "phase_for_june_entry": "AFTER_5_DAY_EXIT_INSIDE_20_DAY_WINDOW", "clock_note": "外交部页面公开时间。5日窗口6月27日已结束；20日窗口持续至7月18日。"},
    ]
    price = pd.read_parquet(OUT / "index_context.parquet")
    quarter = price.loc[:"2019-06-28"]
    drawdown = quarter.close / quarter.close.cummax() - 1
    trough = drawdown.idxmin()
    peak = quarter.loc[:trough].close.idxmax()
    save("index_path_description.json", {"recorded_at": now(), "role": "DESCRIPTIVE_RISK_AFTER_RETURN_REVIEW_NO_RULE_CHANGE",
                                         "max_close_price_drawdown": drawdown.min(), "peak_date": peak, "trough_date": trough,
                                         "not_full_account_risk": True, "no_order_or_parameter_changes": True})
    save("mechanism_evidence.json", {"recorded_at": now(), "facts": rows, "coevent_timeline": timeline,
        "interpretation": "增长或贸易风险上升可能同时压低权益估值和债券收益率，并促使央行缓和；这些同源反应不能当成独立利好票数。",
        "limits": "收益窗口为事后实现值，不识别新闻各自贡献；OIS为窗口内重定价，不是纯政策冲击。"})
    print("政策原文与共同消息时序已核对；季度指数价格最大回撤：", float(drawdown.min()), flush=True)


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False, "font.size": 10})
    prices = pd.read_parquet(OUT / "index_context.parquet")
    intra = [row for row in read(OUT / "six_meetings_intraday.json") if row["event_id"] in EVENTS]
    trades = read(OUT / "events_and_returns.json")
    fig = plt.figure(figsize=(14, 9), facecolor="#fbfcfe")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.52, wspace=0.24)
    top, left, right = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])
    top.plot(prices.index, prices.close, color="#235789", linewidth=2)
    top.axvspan(pd.Timestamp("2019-04-30"), pd.Timestamp("2019-05-06"), color="#b77e39", alpha=0.1)
    top.axvspan(pd.Timestamp("2019-06-29"), prices.index[-1], color="#71869b", alpha=0.06)
    for date, label, height in [("2019-05-06", "假期后首个开盘\n已出现新关税消息", 3990), ("2019-06-20", "6月会议后\n首个开盘", 4110), ("2019-07-01", "G20会晤后\n首个开盘", 3990)]:
        top.axvline(pd.Timestamp(date), color="#8b98a7", linewidth=0.8, linestyle="--")
        top.text(pd.Timestamp(date), height, label, ha="center", va="top", fontsize=9)
    top.set_title("沪深300路径：季度跌幅不大，期间风险和新消息不能省略", loc="left", fontweight="bold")
    top.set_ylabel("指数收盘点位")
    top.xaxis.set_major_locator(mdates.MonthLocator())
    top.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    x = np.arange(2)
    for offset, key, label, color in [(-0.2, "statement_OIS1Y_bp", "声明窗口", "#869aae"), (0.2, "full_OIS1Y_bp", "声明＋记者会", "#235789")]:
        bars = left.bar(x + offset, [row[key] for row in intra], width=0.38, color=color, label=label)
        left.bar_label(bars, fmt="%+.2f", padding=4)
    left.set_ylim(-15, 6)
    left.set_xticks(x, ["2019-05-01", "2019-06-19"])
    left.set_ylabel("1年OIS变动（基点）")
    left.set_title("措辞相似，市场的利率路径重定价不同", loc="left", fontweight="bold")
    left.legend(loc="lower left", frameon=False)
    for offset, horizon, label, color in [(-0.2, 5, "5交易日", "#869aae"), (0.2, 20, "20交易日", "#235789")]:
        values = [next(row["net_return"] * 100 for row in trades if row["event_id"] == event and row["horizon"] == horizon) for event in EVENTS]
        bars = right.bar(x + offset, values, width=0.38, color=color, label=label)
        right.bar_label(bars, fmt="%+.2f%%", padding=4)
    right.set_ylim(-5.2, 4.1)
    right.set_xticks(x, ["5月6日入场", "6月20日入场"])
    right.set_ylabel("510300事件资金净收益（%）")
    right.set_title("A股真正可成交之后的收益", loc="left", fontweight="bold")
    right.legend(loc="lower right", frameon=False)
    for ax in [top, left, right]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#dfe5ec", alpha=0.7)
        ax.set_axisbelow(True)
    for ax in [left, right]:
        ax.axhline(0, color="#758497", linewidth=0.8)
    fig.suptitle("2019年第二季度：利率缓和不能覆盖所有指数风险", x=0.065, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.065, 0.045, "5月会议完整窗口至A股开盘相隔102小时，6月相隔6小时；空档和持有期内的新消息均保留。\n事件收益计入10万元预算、双边各0.1%滑点、各0.04%佣金及整手约束。日内数据为现存历史重建；未识别独立政策因果效应。", fontsize=10, color="#475569")
    fig.subplots_adjust(left=0.075, right=0.97, top=0.88, bottom=0.16)
    fig.savefig(OUT / "2019Q2指数与政策消息时序.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def report():
    protocol = read(OUT / "protocol.json")
    q = read(OUT / "quarter_comparison.json")
    monthly = read(OUT / "monthly_changes.json")
    intra = read(OUT / "six_meetings_intraday.json")
    events = read(OUT / "event_states.json")
    trades = read(OUT / "events_and_returns.json")
    model = read(OUT / "dkw_retrospective_decomposition.json")[-1]
    path = read(OUT / "index_path_description.json")
    proof = read(OUT / "mechanism_evidence.json")
    previous_trades = read(OLD / "events_and_returns.json")
    lines = ["# 历史发现：2019Q2利率缓和与贸易冲击", "",
             "相邻季度给出了反例：2019Q1与Q2按同一时钟对齐的10年美债名义收益率都下降38基点，但沪深300分别上涨28.62%和下跌1.21%。Q2的TIPS也下降23基点，因此仅把名义换成实际利率，仍不足以解释指数机会。", "",
             "| 固定季度 | 名义10年 | TIPS 10年 | 通胀补偿代理 | 沪深300价格 |", "|---|---:|---:|---:|---:|"]
    for row in q:
        lines.append(f"| {row['period']} | {row['nominal10_bp']:+.0f}bp | {row['tips10_bp']:+.0f}bp | {row['inflation_compensation_bp']:+.0f}bp | {row['index_price_return']:+.2%} |")
    lines += ["", "Q2中国端点为3月29日、6月28日；对应按中国21:00及旧美国日终时钟可得的美国曲线为3月28日、6月27日。不是美国自然季度最后一天。季度分组仅用于复盘，不作为当时已知的牛熊标签。", "",
              f"季度累计−1.21%还掩盖了路径：本段沪深300收盘价格最大回撤为{path['max_close_price_drawdown']:.2%}，峰值日{path['peak_date'][:10]}、低点日{path['trough_date'][:10]}。这是指数价格风险描述，不是新策略完整账户最大回撤。", "",
              "| Q2全部日历月 | 名义10年变化 | TIPS变化 | 通胀补偿变化 | 沪深300价格 |", "|---|---:|---:|---:|---:|"]
    for row in monthly:
        lines.append(f"| {row['cn_end'][:7]} | {row['nominal10_bp']:+.0f}bp | {row['tips10_bp']:+.0f}bp | {row['inflation_compensation_bp']:+.0f}bp | {row['index_price_return']:+.2%} |")
    lines += ["", "5月和6月三项利率构成都同向下降，指数月度表现却相反。因此，增加‘名义、实际、通胀补偿三项正负号’仍未自动得到可靠买卖条件。收益率本身是增长信息、政策回应与资产供求共同作用的价格，需继续辨认新增消息。", "",
              "**先看政策为什么变，再看市场是否已经预期。**", "",
              "5月1日仍维持目标区间并称保持耐心，记者会却指出部分低通胀可能是暂时因素，当时没有向任一方向调整的充分理由。IOER的技术调整也不能误称为目标区间降息。[5月记者会，第2—6页](https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20190501.pdf)。", "",
              "6月19日目标区间仍未变，但指引转为将采取适当行动。鲍威尔解释，全球增长数据转弱、贸易进展变成不确定性、企业信心和风险偏好承压，增强了更宽松政策的理由。这里有经济风险，也有政策缓冲，不能把二者当作互不相关的两份利好。[6月声明](https://www.federalreserve.gov/newsevents/pressreleases/monetary20190619a.htm)、[6月记者会，第1—3页](https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20190619.pdf)。", "",
              "复用旧研究已保存的USMPD原始日内变化；声明窗口为发布前10分钟至后20分钟，记者会窗口为开始前10分钟至后60分钟。完整会议涵盖两者；2018年11月无记者会，完整窗口就是声明窗口。利率百分点乘100转为基点，美股收益保留原百分比。[旧金山联储数据定义](https://www.frbsf.org/research-and-insights/data-and-indicators/us-monetary-policy-event-study-database/)。", "",
              "| 美国会议日 | 声明OIS1年 | 记者会OIS1年 | 完整会议OIS1年 | 完整会议标普500 | 510300后5日净收益 | 后20日净收益 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in intra:
        pool = previous_trades if row["event_id"] not in EVENTS else trades
        r5 = next(item for item in pool if item["event_id"] == row["event_id"] and item["horizon"] == 5)
        r20 = next(item for item in pool if item["event_id"] == row["event_id"] and item["horizon"] == 20)
        pc = "无" if row["conference_OIS1Y_bp"] is None else f"{row['conference_OIS1Y_bp']:+.2f}bp"
        lines.append(f"| {row['event_id']} | {row['statement_OIS1Y_bp']:+.2f}bp | {pc} | {row['full_OIS1Y_bp']:+.2f}bp | {row['full_SP500_pct']:+.2f}% | {r5['net_return']:+.2%} | {r20['net_return']:+.2%} |")
    lines += ["", "5月声明窗口OIS下降4.00bp，但记者会回升6.90bp，完整窗口最终上升2.90bp，方向发生逆转。6月完整窗口则下降11.85bp。这是比‘政策说了什么’更接近预期修正的观察，但OIS变化仍可能混合政策、经济信息及风险定价，未隔离纯政策冲击。没有拟合PCA或用这些正负号补救旧32次会议买入实验。", "",
              "**美国消息结束，到A股能成交，还有一个信息窗口。**", "",
              "| 会议日 | 中国入场日 | 完整会议后等待 | 前次中国收盘至开盘缺口 | 5日退出 | 20日退出 |", "|---|---|---:|---:|---|---|"]
    for row in events:
        r5 = next(item for item in trades if item["event_id"] == row["event_id"] and item["horizon"] == 5)
        r20 = next(item for item in trades if item["event_id"] == row["event_id"] and item["horizon"] == 20)
        lines.append(f"| {row['event_id']} | {row['entry_date']} | {row['wait_hours']:.0f}小时 | {row['etf_gap_return']:+.2%} | {r5['exit_date']} | {r20['exit_date']} |")
    lines += ["", "5月会议结束后恰逢中国假期，102小时后才能在5月6日开盘买入。期间5月5日特朗普已公开表示提高相关关税，故开盘缺口−3.03%不能归为单一FOMC反应，也不能计入开盘后新买入的损益。[原始推文档案](https://www.presidency.ucsb.edu/documents/tweets-may-5-2019)。该档案16:08:46未标时区，按UTC或美东解读均早于5月6日开盘；保存两种解释，未宣称秒级时区已核实。", "",
              "5月6日开盘后仍亏损：5日−2.75%，20日−3.81%，均已扣成本。没有因后来找到关税解释而删除亏损或重选入场。5月10日USTR确认实施提高税率，5月13日中方宣布6月1日起提高部分商品加征税率；两者的公告与生效日分别保存。中方5月13日消息不能倒填到该日上午已经结束的5日收益窗口。[USTR原文](https://ustr.gov/about-us/policy-offices/press-office/press-releases/2019/may/statement-us-trade-representative)、[中方公告](https://gss.mof.gov.cn/gzdt/zhengcefabu/201905/t20190513_3256787.htm)。", "",
              "6月盈利也按同样标准处理：6月18日中美元首通话并约定大阪会晤，早于美联储会议和6月20日入场；6月29日实际会晤后的重启磋商消息，晚于6月27日5日窗口退出，但位于20日窗口内。它不能解释已经实现的全部5日利润，也不能被原美联储信号提前认领。[会前通话](https://www.mfa.gov.cn/web/zyxw/201906/t20190618_346718.shtml)、[6月29日会晤](https://www.fmprc.gov.cn/web/gjhdq_676201/gj_676203/bmz_679954/1206_680528/xgxw_680534/201906/t20190629_9360943.shtml)。", "",
              "**构成的更下一层只作解释。**", "",
              "已有2025年重估DKW模型对Q2的分解，依次为预期实际短端利率、预期通胀、实际期限溢价、通胀风险溢价；TIPS流动性溢价另列。它不是当年可得的交易输入，也不是对指数因果贡献的估计。", "",
              "| 模型Q2变化，单位bp | 数值 |", "|---|---:|"]
    for key, label in [("exp.real.short.rate.10", "预期实际短端利率"), ("exp.inflation.10", "预期通胀"), ("real.term.prem.10", "实际期限溢价"), ("inflation.risk.prem.10", "通胀风险溢价"), ("tips.liq.prem.10", "TIPS流动性溢价，另列"), ("nominal.yield.raw.10", "模型输入名义零息收益率"), ("nominal.yield.fitted.10", "四项拟合名义收益率")]:
        lines.append(f"| {label} | {model['changes_bp'][key]:+.2f} |")
    lines += ["", f"名义拟合残差变化为{model['nominal_residual_change_bp']:+.2f}bp，Treasury名义变化与模型零息输入变化之差为{model['nominal_cmt_minus_dkw_raw_change_bp']:+.2f}bp；详细通胀补偿恒等式和残差保存在数据文件。没有把模型流动性溢价作为名义四项之外的第五项相加。", "",
              "**本轮研究取舍。**停止继续扩展‘美债下降或FOMC偏宽松便买入指数’这一单独规则。保留三项有用信息：政策为什么回应、市场在完整沟通后如何修正预期、A股真正可交易之前与持有期内是否出现改变原假设的新信息。下一步把已有国内信用、资金价格、外部政策和指数定价归并到同一条历史时间线，先检查哪种状态变化有可重复的剩余收益，避免继续堆因子名称。", "",
              "本轮新增的是两次固定事件的四个成本窗口，旧四次收益直接复用；六次日内变化为已有文件的新提取，没有重新下载数据或拟合参数。交易资金预算10万元，单边滑点0.1%、佣金0.04%且最低5元，100份整手、0.001元价位；收益分母为实际买入支付资金，不是20万元完整账户。T+1时间关系、费用身份式、日内利率加总与股价复利、曲线及模型恒等式均已核对。", "",
              "这些历史已被研究，不能称独立验证。USMPD是2026版历史重建，不是2019年的采集回执；最终记者会转录文件也不声称当日已经下载。未计算新账户夏普，完整账户净夏普1.2及年化10%的目标仍未实现。", "",
              "图：2019Q2指数与政策消息时序.png。计算：quarter_comparison.json、monthly_changes.json、six_meetings_intraday.json、event_states.json、events_and_returns.json、mechanism_evidence.json。代码：research/historical_index_us_rates_trade_q2_v1.py。", ""]
    (OUT / REPORT).write_text("\n".join(lines), encoding="utf-8")
    draw()
    save("result.json", {"study_id": protocol["study_id"], "completed_at": now(),
        "classification": "PROGRESS_INDEX_Q2_COUNTEREXAMPLE_AND_INFORMATION_TIMING", "new_event_nodes": 2, "new_return_rows": 4,
        "intraday_events": 6, "reused_cost_event_nodes": 4, "quarter": q[-1], "index_path": path,
        "discovery": "2019Q2名义/TIPS/通胀补偿变化-38/-23/-15bp，沪深300-1.21%；5、6月三项同降但指数一跌一涨。5月声明OIS-4bp经记者会后变+2.9bp，6月完整窗口-11.85bp。5月A股等待102小时已夹杂关税新消息，开盘后5/20日仍亏2.75%/3.81%；6月盈利2.01%/2.38%。",
        "research_action": "STOP_EXTENDING_RATE_DIRECTION_OR_FOMC_TONE_STANDALONE_BUY_RULE",
        "next_historical_question": "将2018Q4—2019Q2现有国内信用、资金价格、完整政策沟通和指数已发生定价归并为当时可知的状态时间线；先查既有最接近的状态研究，只检验明确机制增量，不再堆美债单因子变体。",
        "new_source_documents": 8, "new_native_document_files": 6, "web_resolved_primary_documents": 2, "reused_ustr_document": 1,
        "new_parameters_fitted": 0, "new_full_accounts": 0, "net_sharpe": None, "goal_achieved": False,
        "orders_authorized": False, "independent_validation": False, "report": REPORT})
    print("已生成反例报告、完整消息时序和图；没有新参数或新完整账户。", flush=True)


def record_progress():
    result = read(OUT / "result.json")
    assert (OUT / REPORT).exists()
    prefix = OUT.relative_to(ROOT).as_posix() + "/"
    stamp, changes = now(), []
    for name in ["510300_historical_cause_discovery_v1.json", "510300_existing_data_training_mandate_v1.json"]:
        path = ROOT / "config" / name
        cfg = read(path)
        before = dict(cfg)
        if name == "510300_historical_cause_discovery_v1.json":
            cfg.update(current_study=prefix + "protocol.json", latest_completed_study=prefix + "result.json", latest_report=prefix + REPORT, updated_at=stamp)
        else:
            cfg.update(current_round=result["study_id"], latest_progress_receipt=prefix + "result.json",
                       latest_historical_index_us_rates_trade_q2=prefix + "result.json", latest_continuation_report=prefix + REPORT,
                       latest_historical_report=prefix + REPORT, latest_continuation_classification=result["classification"],
                       current_driver_continuation_classification=result["classification"], current_driver_consecutive_blocked_goal_turns=0,
                       latest_historical_diagnostic_at=stamp, latest_goal_service_status="active", latest_goal_service_status_observed_at=stamp,
                       goal_status="active", goal_achieved=False, local_goal_work_status="ACTIVE_HISTORICAL_ONLY",
                       last_research_result=result["discovery"], last_source_result="补齐两次政策声明与记者会、贸易公告及会前通话，复用六次日内事件数据。",
                       next_research_question=result["next_historical_question"])
        changes.append({"path": path.relative_to(ROOT).as_posix(), "fields": {key: {"before": before.get(key), "after": value} for key, value in cfg.items() if before.get(key) != value}})
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save("authority_update.json", {"recorded_at": stamp, "previous_goal_turn_classification": "PROGRESS",
                                  "current_goal_turn_classification": result["classification"], "changes": changes,
                                  "goal_achieved": False, "orders_authorized": False})
    print("已登记第二季度反例及停止单独利率买入规则扩展的研究动作。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="2019Q2指数、利率与贸易消息历史对照")
    parser.add_argument("mode", choices=["prepare", "fetch", "calculate", "supplement", "evidence", "report", "record-progress"])
    mode = parser.parse_args().mode
    {"prepare": prepare, "fetch": fetch, "calculate": calculate, "supplement": supplement,
     "evidence": evidence, "report": report, "record-progress": record_progress}[mode]()


if __name__ == "__main__":
    main()
