"""固定2018年下半年公开月，对照信用消息、统计范围和指数整体定价。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from credit_recency_structure_v23 import Ledger
from historical_index_credit_constraint_relief_v1 import assign_clusters
from historical_index_credit_price_decomposition_v1 import macro_context
from historical_index_liquidity_transmission_v1 import clean
from historical_price_gap_causes_v1 import round_trip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_index_credit_regime_compare_v1"
PRIOR = ROOT / "reports/research/510300_historical_index_valuation_repricing_v1"
FLOWS = ROOT / "reports/research/510300_historical_index_credit_flow_composition_v1"
EXPECT = ROOT / "reports/research/510300_historical_index_credit_expectation_gap_v1"
STUDY = "510300_HISTORICAL_INDEX_CREDIT_REGIME_COMPARE_V1"
INPUTS = {
    "originals": "reports/research/510300_macro_transmission_context_v4/inputs/loan_originals.json",
    "market": "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet",
    "price": "reports/research/510300_macro_earnings_pricing_bridge_v5/inputs/index_price.parquet",
    "pe": "reports/research/510300_macro_earnings_pricing_bridge_v5/inputs/pe.parquet",
    "curve": "data/raw/macro/china_government_bond_yields_daily.parquet",
    "funding": "data/raw/macro/510300_macro_stress_2015_v2/fdr007_daily_2015_2026.parquet",
}
SOURCES = [
    {"id": "reuters_june_recall", "date": "2018-07-13", "month": "2018-06", "role": "RELEASE_REPORT_RECALL", "url": "https://www.business-standard.com/amp/article/reuters/china-june-new-loans-jump-to-1-84-trillion-yuan-above-forecasts-118071300551_1.html", "required": ["1.6 trillion yuan", "8.3 percent"]},
    {"id": "reuters_july_recall", "date": "2018-08-13", "month": "2018-07", "role": "RELEASE_REPORT_RECALL", "url": "https://finance.yahoo.com/news/china-july-loans-stronger-expected-120641332.html", "required": ["1.2 trillion yuan", "8.2 percent"]},
    {"id": "reuters_august_pre", "date": "2018-09-07", "month": "2018-08", "role": "PRE_RELEASE_PUBLICATION", "url": "https://www.investing.com/news/economy-news/chinas-august-bank-lending-seen-lower-but-shift-to-credit-easing-intact-reuters-poll-1601663", "required": ["1.3 trillion yuan", "8.5 percent", "35 economists"]},
    {"id": "reuters_september_pre", "date": "2018-10-05", "month": "2018-09", "role": "PRE_RELEASE_PUBLICATION", "url": "https://www.investing.com/news/economy-news/chinas-sept-new-loans-seen-rising-as-policymakers-seek-to-underpin-growth-1634407", "required": ["1.35 trillion yuan", "8.3 percent", "26 analysts"]},
    {"id": "tsf_june", "date": "2018-07-13", "month": "2018-06", "role": "央行原报告同期转载", "url": "https://finance.sina.com.cn/roll/2018-07-13/doc-ihfhfwmu7932378.shtml", "required": ["9.1万亿元", "1.18万亿元", "8008亿元"]},
    {"id": "tsf_july", "date": "2018-08-13", "month": "2018-07", "role": "央行原报告及当时回溯表", "url": "https://xining.pbc.gov.cn/diaochatongjisi/116219/116225/060ccf383b7145a0a4367fcb83bd36c0/index.html", "required": ["10415", "13922", "贷款核销"]},
    {"id": "tsf_august", "date": "2018-09-12", "month": "2018-08", "role": "央行原报告同期转载", "url": "https://www.nbd.com.cn/articles/2018-09-12/1254375.html", "required": ["1.52万亿元", "1207亿元", "688亿元"]},
    {"id": "tsf_september", "date": "2018-10-17", "month": "2018-09", "role": "央行原报告及当时回溯表同期转载", "url": "https://finance.sina.com.cn/roll/2018-10-17/doc-ifxeuwws5281212.shtml", "required": ["22054", "7389", "19286"]},
    {"id": "cbirc_early_july", "date": "2018-08-11", "month": "2018-07", "role": "银保监会原文同期转载，早于完整金融统计报告", "url": "https://m.21jingji.com/article/20180811/herald/8a21bfb96f22dc12d294980af6699fdd.html", "required": ["1.45万亿元", "1724亿元", "尽职免责"]},
    {"id": "pboc_september_briefing", "date": "2018-10-17", "month": "2018-09", "role": "记者直接报道央行当日发布会，预期调查来源未明", "url": "https://m.21jingji.com/article/20181017/herald/92ed1978e0787b892db9d6aa48dca63f.html", "required": ["7389亿元", "1.55万亿元", "0.17个百分点"]},
]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("已固定范围，不覆盖。")
    prior = read(PRIOR / "result.json")
    assert prior["classification"] == "PROGRESS_INDEX_VALUATION_CLOCK_AND_REPRICING"
    events = [{"id": r["stat_month"], "date": r["conservative_known_at"][:10], "loan_source": r["source_url"]}
              for r in read(ROOT / INPUTS["originals"]) if "2018-07" <= r["conservative_known_at"][:7] <= "2018-12"]
    assert len(events) == 6
    save("protocol.json", {
        "study_id": STUDY, "recorded_at": now(), "previous_goal_turn_classification": "PROGRESS",
        "previous_completed_receipt": str((PRIOR / "result.json").relative_to(ROOT)),
        "mode": "HISTORICAL_DISCOVERY_ONLY", "research_unit": "沪深300整体，不扩展公司案例",
        "question": "2018年下半年政策支持、信用消息与此前指数重估的组合，是否重复2019年初的剩余收益；统计扩围和提前披露改变什么？",
        "calendar_release_months": ["2018-07", "2018-12"], "events": events,
        "selection": "承接上轮已写明的六个公开月，全部保留；不是按涨跌划牛熊，不是独立验证。",
        "sources": SOURCES, "existing_inputs": INPUTS,
        "returns": "10/11统计月直接复用上一轮成本后5/20日结果；其余四次沿同一算法补成本。V4既有E0采用第20日收盘，本轮沿近期事件合同采用入场后20日开盘，两者不可混称同一收益。",
        "entry": "原金融统计报告保守可用日结束后下一交易日开盘；5/20个交易日后开盘退出。即使发现提前披露，保留原窗口，只标注消息已先公开，不据收益移动入场。",
        "costs": read(FLOWS / "protocol.json")["costs"], "horizons": [5, 20],
        "valuation": "沿上轮20/60交易日描述；PE按原available_at不晚于公开日21:00取值，价格使用同日000300，P/PE不叫真实盈利。",
        "rates": "固定2018年6—12月月末，同期FDR007及近20次均值、1/10年国债只用于描述，不转换仓位。",
        "actuals": "贷款优先同一原文直接单月；否则使用原V23累计差账本，保留舍入界。M2提取同一原文。社融初始公布总量与后来扩围表分别保留。",
        "expectations": "路透调查为主，公布前正文与公布后回述分开；缺社融调查就缺失。其他报道的未标明来源预期只用于口径演示，不与路透拼接。",
        "mechanism_before_new_results": "新信息与此前定价结合可能解释阶段差异；不预设2018年一定亏损或一定相反。特别检查银行贷款增加是否被其他融资收缩抵消。",
        "statistical_change": "不能拿扩围后的社融减旧口径预测，或把新披露回溯历史当成旧日期已知。",
        "checks_only": ["日期先后", "金额单位和同日价格", "现金费用恒等式", "保留全部固定事件"],
        "candidate_action": "本轮不拟合参数，不用六个事件定义买卖门槛，不修改MACRO-02或VAL04失败。若普通信用/估值叙事仍无法区分收益，结束这条单消息方向性研究，转向独立的指数历史机会机制。",
        "new_parameters_fitted": 0, "new_full_accounts": 0, "net_sharpe": None, "goal_achieved": False, "orders_authorized": False,
    })
    print("已固定六个公开月、原成本与窗口；新旧统计口径及提前披露分别记录。", flush=True)


def fetch_one(source):
    item = dict(source)
    try:
        response = requests.get(source["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=(8, 25))
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        body = unicodedata.normalize("NFKC", soup.get_text(" ", strip=True))
        compact = re.sub(r"\s+", "", body).lower()
        missing = [v for v in source["required"] if re.sub(r"\s+", "", v).lower() not in compact]
        path = OUT / "sources" / (source["id"] + ".html")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        path.with_suffix(".txt").write_text(body, encoding="utf-8")
        item.update(status="RETRIEVED" if not missing else "BODY_CHECK_REQUIRED", missing=missing,
                    file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(response.content).hexdigest(), retrieved_at=now())
    except Exception as exc:
        item.update(status="MISSING_SOURCE", error=str(exc), retrieved_at=now())
    return item


def fetch():
    if (OUT / "source_manifest.json").exists():
        raise RuntimeError("已请求来源，不重复。")
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch_one, source) for source in SOURCES]
        for future in as_completed(futures):
            item = future.result()
            records.append(item)
            print("来源完成：" + item["id"] + "，状态：" + item["status"], flush=True)
    save("source_manifest.json", sorted(records, key=lambda r: r["id"]))


def build_local():
    if (OUT / "events_and_returns.json").exists():
        raise RuntimeError("已有事件结果，不改窗口重算。")
    protocol = read(OUT / "protocol.json")
    originals = {r["stat_month"]: r for r in read(ROOT / INPUTS["originals"])}
    ledger = Ledger()
    market = pd.read_parquet(ROOT / INPUTS["market"]).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market["date"]).dt.tz_localize(None).dt.normalize()
    calendar = pd.DatetimeIndex(market.date)
    dividends = pd.read_csv(ROOT / "data/reference/510300_dividends.csv")
    dividends = dividends.loc[dividends.symbol.eq("510300.SH")].copy()
    dividends["record_date"] = pd.to_datetime(dividends["record_date"])
    old_returns = {(r["stat_month"], r["horizon"]): r for r in read(FLOWS / "events_and_returns.json")}
    prices = pd.read_parquet(ROOT / INPUTS["price"])
    pe = pd.read_parquet(ROOT / INPUTS["pe"])
    prices["date"] = pd.to_datetime(prices["date"])
    pe["date"] = pd.to_datetime(pe["observation_date"])
    pe["available_at"] = pd.to_datetime(pe["available_at"], utc=True).dt.tz_convert("Asia/Shanghai")
    panel = prices[["date", "close"]].merge(pe[["date", "available_at", "original_pe_official"]], on="date", validate="one_to_one").sort_values("date").set_index("date")
    returns, states, loans = [], [], []
    for event in protocol["events"]:
        date, month = pd.Timestamp(event["date"]), event["id"]
        cutoff = date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=21)
        entry_pos = calendar.searchsorted(date, side="right")
        before_pos = calendar.searchsorted(date, side="left") - 1
        entry, before = market.iloc[entry_pos], market.iloc[before_pos]
        for horizon in protocol["horizons"]:
            if (month, horizon) in old_returns:
                row = {**old_returns[(month, horizon)], "return_origin": "REUSED_SAVED_COST_AFTER_RESULT"}
            else:
                end = market.iloc[entry_pos + horizon]
                div = float(dividends.loc[(dividends.record_date >= entry.date) & (dividends.record_date < end.date), "cash_dividend_per_share"].sum())
                trade = round_trip(float(entry.open), float(end.open), div, protocol["costs"])
                row = {"stat_month": month, "date": event["date"], "horizon": horizon,
                       "entry_date": entry.date.strftime("%Y-%m-%d"), "exit_date": end.date.strftime("%Y-%m-%d"),
                       "entry_open": float(entry.open), "exit_open": float(end.open), "dividend_per_share": div,
                       "gross_return": float((end.open + div) / entry.open - 1), **trade,
                       "return_origin": "SAME_FIXED_EVENT_CONTRACT_COSTS_ADDED"}
            assert pd.Timestamp(row["entry_date"]) > date
            assert row["paid_cny"] <= protocol["costs"]["illustrative_event_budget_cny"]
            error = (row["sell_price"] - row["buy_price"]) * row["shares"] + row["dividend_entitlement_cny"] - row["commissions_cny"] - row["net_pnl_cny"]
            assert abs(error) < 1e-7
            returns.append(row)
        actual_loans = [ledger.row(month, field, "1") for field in ["rmb_total", "corporate_long", "corporate_short", "bills", "household_long"]]
        loans.extend(actual_loans)
        raw = originals[month]
        body = unicodedata.normalize("NFKC", BeautifulSoup((ROOT / raw["raw_path"]).read_bytes(), "html.parser").get_text(" ", strip=True))
        compact = re.sub(r"\s+", "", body)
        match = re.search(r"广义货币\(M2\)余额[\d.]+万亿元,同比增长([\d.]+)%", compact)
        if not match:
            raise ValueError("未定位原报告M2：" + month)
        latest = panel.loc[panel.available_at.le(cutoff)].iloc[-1]
        pos = panel.index.get_loc(latest.name)
        lookbacks = []
        for horizon in [20, 60]:
            a = panel.iloc[pos - horizon]
            lp = float(100 * np.log(latest.close / a.close))
            lm = float(100 * np.log(latest.original_pe_official / a.original_pe_official))
            lookbacks.append({"horizon": horizon, "start": a.name, "end": latest.name,
                              "price_return": float(latest.close / a.close - 1),
                              "pe_change": float(latest.original_pe_official / a.original_pe_official - 1),
                              "price_log_points": lp, "pe_log_points": lm, "implied_denominator_log_points_not_profit": lp - lm})
        states.append({"stat_month": month, "release_date": event["date"], "cutoff": cutoff,
                       "pe_date": latest.name, "pe_available_at": latest.available_at, "pe": latest.original_pe_official,
                       "index_close": latest.close, "lookbacks": lookbacks, "m2_actual": float(match[1]),
                       "loan_actual": actual_loans[0]["value_yi"], "loan_actual_method": actual_loans[0]["method"],
                       "loan_actual_rounding_half_yi": actual_loans[0]["rounding_half_yi"],
                       "loan_source_url": raw["source_url"], "loan_raw_path": raw["raw_path"]})
    curve = pd.read_parquet(ROOT / INPUTS["curve"])
    curve["date"] = pd.to_datetime(curve["date"])
    funding = pd.read_parquet(ROOT / INPUTS["funding"])
    funding["date"] = pd.to_datetime(funding["date"])
    funding = funding.sort_values("date")
    funding["fdr007"] = pd.to_numeric(funding["first_release_value"])
    funding["fdr20"] = funding.fdr007.rolling(20, min_periods=20).mean()
    daily = panel.reset_index().merge(curve[["date", "cgb_1y", "cgb_10y"]], on="date", how="left", validate="one_to_one")
    daily = daily.merge(funding[["date", "fdr007", "fdr20"]], on="date", how="left", validate="one_to_one").set_index("date")
    daily.loc["2018-06-01":"2019-01-15"].to_parquet(OUT / "daily_context.parquet")
    monthends = []
    for month in pd.period_range("2018-06", "2018-12", freq="M"):
        dates = calendar[calendar.to_period("M") == month]
        stamp = dates[-1]
        monthends.append({"date": stamp, **daily.loc[stamp].to_dict()})
    clusters = [assign_clusters(returns, h) for h in protocol["horizons"]]
    save("events_and_returns.json", returns)
    save("event_states.json", states)
    save("loan_components.json", loans)
    save("monthends.json", monthends)
    save("known_macro_context.json", macro_context(protocol["events"]))
    save("input_receipts.json", [{"id": key, "path": path, "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest()} for key, path in INPUTS.items()])
    save("local_build.json", {"recorded_at": now(), "events": len(states), "return_rows": len(returns), "overlap": clusters,
                              "new_cost_windows": sum(r["return_origin"] != "REUSED_SAVED_COST_AFTER_RESULT" for r in returns),
                              "reused_cost_windows": sum(r["return_origin"] == "REUSED_SAVED_COST_AFTER_RESULT" for r in returns),
                              "new_full_accounts": 0, "goal_achieved": False})
    print(pd.DataFrame([{**{k: r[k] for k in ["stat_month", "release_date", "loan_actual", "m2_actual", "pe"]},
                         "此前60日涨幅": r["lookbacks"][1]["price_return"],
                         "20日净收益": next(t["net_return"] for t in returns if t["stat_month"] == r["stat_month"] and t["horizon"] == 20)} for r in states]).to_string(index=False), flush=True)
    print(pd.DataFrame(monthends)[["date", "close", "original_pe_official", "fdr20", "cgb_10y"]].to_string(index=False), flush=True)


def build_evidence():
    if (OUT / "expectation_comparison.json").exists():
        raise RuntimeError("已保存预期和构成，不覆盖。")
    sources = read(OUT / "source_manifest.json")
    for source in sources:
        if source["status"] == "MISSING_SOURCE":
            web = OUT / "sources" / (source["id"] + ".web.txt")
            if web.exists():
                body = web.read_text(encoding="utf-8")
                compact = re.sub(r"\s+", "", body).lower()
                if not all(re.sub(r"\s+", "", v).lower() in compact for v in source["required"]):
                    raise ValueError("网页正文关键值不齐：" + source["id"])
                source.update(status="RETRIEVED_WEB_EXTRACT", original_error=source.get("error"),
                              file=str(web.relative_to(ROOT)), sha256=hashlib.sha256(web.read_bytes()).hexdigest(),
                              note="浏览工具可读正文，保留原HTTP失败；不是原始HTML。")
    if any(not r["status"].startswith("RETRIEVED") for r in sources):
        raise ValueError("仍有来源正文待定位。")
    save("source_resolution.json", {"recorded_at": now(), "sources": sources})
    by_id = {r["id"]: r for r in sources}
    specs = [
        ("reuters_june_recall", "rmb_loan_flow", 16000, "1.6 trillion yuan", None, None, None),
        ("reuters_june_recall", "m2_yoy", 8.3, "8.3 percent", None, None, None),
        ("reuters_july_recall", "rmb_loan_flow", 12000, "1.2 trillion yuan", None, None, None),
        ("reuters_july_recall", "m2_yoy", 8.2, "8.2 percent", None, None, None),
        ("reuters_august_pre", "rmb_loan_flow", 13000, "1.3 trillion yuan", 35, None, None),
        ("reuters_august_pre", "m2_yoy", 8.5, "8.5 percent", 35, None, None),
        ("reuters_september_pre", "rmb_loan_flow", 13500, "1.35 trillion yuan", 26, 7000, 16000),
        ("reuters_september_pre", "m2_yoy", 8.3, "8.3 percent", 24, None, None),
    ]
    forecasts = []
    for source_id, series, value, anchor, count, low, high in specs:
        source = by_id[source_id]
        body_file = OUT / "sources" / (source_id + (".web.txt" if source["status"] == "RETRIEVED_WEB_EXTRACT" else ".txt"))
        body = body_file.read_text(encoding="utf-8")
        assert re.sub(r"\s+", "", anchor).lower() in re.sub(r"\s+", "", body).lower()
        forecasts.append({"source_id": source_id, "stat_month": source["month"], "series": series,
                          "expected": value, "source_url": source["url"], "published_date": source["date"],
                          "evidence_tier": source["role"], "anchor": anchor, "article_survey_n": count,
                          "forecast_min": low, "forecast_max": high,
                          "aggregation": "M2中位数" if source_id == "reuters_september_pre" and series == "m2_yoy" else "调查代表值，未另推均值/中位数",
                          "provider": "路透自身调查或署名原报道"})
    forecasts.extend(r for r in read(EXPECT / "expectations.json")["observations"] if r["stat_month"] in ["2018-10", "2018-11"])

    # 取最内层数据表，避免将网页布局表中的嵌套正文重复解析。
    vintages = []
    for source_id in ["tsf_july", "tsf_september"]:
        soup = BeautifulSoup((OUT / "sources" / (source_id + ".html")).read_bytes(), "html.parser")
        tables = [t for t in soup.find_all("table") if "当月增量" in t.get_text() and "2018年" in t.get_text()]
        table = min(tables, key=lambda t: len(t.get_text()))
        months = []
        current = {}
        for tr in table.find_all("tr"):
            cells = [unicodedata.normalize("NFKC", c.get_text("", strip=True)).replace(" ", "") for c in tr.find_all(["td", "th"], recursive=False)]
            if not cells:
                continue
            if cells[0] == "月份":
                months = []
                for cell in cells[1:]:
                    found = re.fullmatch(r"(\d{4})年(\d{1,2})月", cell)
                    months.append(f"{found[1]}-{int(found[2]):02d}" if found else None)
                continue
            field = ("total" if cells[0].startswith("当月增量") else
                     "special_bonds" if "地方政府专项债券" in cells[0] else
                     "bank_abs" if "存款类金融机构资产支持证券" in cells[0] else
                     "writeoffs" if "贷款核销" in cells[0] else None)
            if not field:
                continue
            for month, cell in zip(months, cells[1:]):
                if month:
                    current.setdefault(month, {"stat_month": month, "source_id": source_id,
                                              "published_date": by_id[source_id]["date"], "unit": "亿元"})[field] = float(cell.replace(",", ""))
        vintages.extend(current.values())
    sept = {r["stat_month"]: r for r in vintages if r["source_id"] == "tsf_september"}
    july = {r["stat_month"]: r for r in vintages if r["source_id"] == "tsf_july"}
    assert sept["2018-09"]["total"] == 22054 and sept["2018-09"]["special_bonds"] == 7389
    assert sept["2018-08"]["total"] == 19286 and sept["2018-08"]["special_bonds"] == 4106
    assert july["2018-06"]["total"] == 13922 and july["2018-07"]["total"] == 10415
    same_vintage = {"published_date": "2018-10-17", "comparison": "9月减8月，均取10月17日同一原文表",
                    "total_change_yi": sept["2018-09"]["total"] - sept["2018-08"]["total"],
                    "special_bond_change_yi": sept["2018-09"]["special_bonds"] - sept["2018-08"]["special_bonds"],
                    "other_tsf_september_yi": sept["2018-09"]["total"] - sept["2018-09"]["special_bonds"],
                    "other_tsf_august_yi": sept["2018-08"]["total"] - sept["2018-08"]["special_bonds"]}
    same_vintage["other_tsf_change_yi"] = same_vintage["other_tsf_september_yi"] - same_vintage["other_tsf_august_yi"]
    assert same_vintage["total_change_yi"] == same_vintage["special_bond_change_yi"] + same_vintage["other_tsf_change_yi"]
    same_vintage["limits"] = "其他社融含多个渠道和主体，不等于私人融资；本表是同口径月度差，不是预期差或政策因果贡献。"
    old_headlines = {r["stat_month"]: r["total_yi"] for r in read(FLOWS / "composition.json")["monthly_tsf_headlines"]}
    actual_tsf = {"2018-06": 11800, "2018-07": 10415, "2018-08": 15200, "2018-09": 22054,
                  "2018-10": old_headlines["2018-10"], "2018-11": old_headlines["2018-11"]}
    for source_id, literal in [("tsf_june", "1.18万亿元"), ("tsf_august", "1.52万亿元")]:
        assert literal in (OUT / "sources" / (source_id + ".txt")).read_text(encoding="utf-8")
    states = read(OUT / "event_states.json")
    returns = read(OUT / "events_and_returns.json")
    comparisons = []
    for state in states:
        month = state["stat_month"]
        for series, actual in [("rmb_loan_flow", state["loan_actual"]), ("m2_yoy", state["m2_actual"]), ("tsf_flow", actual_tsf[month])]:
            found = [f for f in forecasts if f["stat_month"] == month and f["series"] == series]
            assert len(found) <= 1
            forecast = found[0] if found else {"expected": None, "evidence_tier": "MISSING_EXPECTATION", "source_id": None}
            if forecast["evidence_tier"] == "PRE_RELEASE_PUBLICATION":
                assert forecast["published_date"] < state["release_date"]
            expected = forecast["expected"]
            comparisons.append({**forecast, "stat_month": month, "release_date": state["release_date"], "series": series,
                                "actual": actual, "surprise": None if expected is None else actual - expected,
                                "unit": "百分点" if series == "m2_yoy" else "亿元",
                                "loan_was_preannounced": month == "2018-07" and series == "rmb_loan_flow",
                                "preannouncement_date": "2018-08-11" if month == "2018-07" and series == "rmb_loan_flow" else None,
                                "net_return_5d": next(r["net_return"] for r in returns if r["stat_month"] == month and r["horizon"] == 5),
                                "net_return_20d": next(r["net_return"] for r in returns if r["stat_month"] == month and r["horizon"] == 20)})
    save("expectations.json", {"recorded_at": now(), "observations": forecasts})
    save("expectation_comparison.json", comparisons)
    save("tsf_vintages.json", {"recorded_at": now(), "rows": vintages, "same_release_monthly_decomposition": same_vintage,
         "initial_headlines_yi": actual_tsf,
         "june_vintages": [{"published_date": "2018-07-13", "value_yi": 11800, "precision": "原文1.18万亿元，四舍五入"},
                            {"published_date": "2018-08-13", "value_yi": july["2018-06"]["total"], "precision": "原表亿元"},
                            {"published_date": "2018-10-17", "value_yi": sept["2018-06"]["total"], "precision": "原表亿元"}],
         "vintage_note": "扩围表也可能含修订，差额不能全部强归因于新增字段；8月或10月表不回填7月决策。",
         "reported_expectation_scope_demo": {"source_id": "pboc_september_briefing", "reported_forecast_yi": 15500,
             "forecast_provider_and_scope_verified": False, "new_total_minus_reported_forecast_yi": 22054 - 15500,
             "excluding_special_bonds_minus_reported_forecast_yi": 22054 - 7389 - 15500,
             "valid_surprise_signal": False, "purpose": "仅说明混合口径足以翻转符号，不认定调查原本排除了专项债。"}})
    save("mechanism_evidence.json", {"recorded_at": now(),
         "jul_early_publication": {"source_id": "cbirc_early_july", "published_date": "2018-08-11", "loan_yi": 14500,
             "full_report_date": "2018-08-13", "old_entry_date": "2018-08-14", "entry_moved": False,
             "interpretation": "总贷款数字在完整报告前已公开；8月13日整包报告不等于所有字段都是新消息。"},
         "july_shadow_channels": {"source_id": "tsf_july", "entrusted_yi": -950, "trust_yi": -1192,
             "undiscounted_bills_yi": -2744, "sum_yi": -4886, "real_economy_rmb_loans_yi": 12900,
             "bank_total_loans_yi": 14500, "interpretation": "两种贷款范围不同，不相加；银行贷款增加同时存在表外三渠道收缩。"},
         "august_shadow_channels": {"source_id": "tsf_august", "entrusted_yi": -1207, "trust_yi": -688,
             "undiscounted_bills_yi": -779, "sum_yi": -2674},
         "upstream_constraints": {"source_id": "cbirc_early_july",
             "bank_capital": "提高利润留存、补充一级资本，关系到放贷能力。",
             "balance_sheet": "不良处置核销及债转股盘活既有资产，不能把操作金额等同新增股票购买力。",
             "incentives": "尽职免责与小微不良容忍度，影响承担信用风险的意愿。",
             "credit_demand": "公告提出保障有效融资需求，并不证明所有企业需求或盈利已改善。"},
         "september_cost_counterevidence": {"source_id": "pboc_september_briefing", "small_loan_limit_cny": 5000000,
             "new_loan_rate_percent": 6.25, "change_vs_first_half_pp": -.17,
             "interpretation": "同期发布会已报告部分小微贷款价格下降，不能把2018年概括为所有宽松传导均无效；该口径也不是沪深300整体融资成本。"},
         "causal_contribution_identified": False, "new_rules_admitted": 0})
    print("已连接13项调查观测与六个固定窗口，保留5项社融预期缺失。", flush=True)
    print(json.dumps(same_vintage, ensure_ascii=False, indent=2), flush=True)


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10})
    events = read(OUT / "event_states.json")
    returns = read(OUT / "events_and_returns.json")
    parts = read(OUT / "tsf_vintages.json")["same_release_monthly_decomposition"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), gridspec_kw={"width_ratios": [1.4, 1]})
    fig.patch.set_facecolor("#f6f6ef")
    for ax in axes:
        ax.set_facecolor("#f6f6ef")
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="#737970", lw=.8)
        ax.grid(axis="y", color="#daddd3", alpha=.8)
        ax.set_axisbelow(True)
    x = np.arange(len(events))
    for horizon, shift, color in [(5, -.18, "#9b9d88"), (20, .18, "#337c86")]:
        values = [100 * next(r["net_return"] for r in returns if r["stat_month"] == e["stat_month"] and r["horizon"] == horizon) for e in events]
        bars = axes[0].bar(x + shift, values, .33, color=color, label=f"{horizon}日事件净收益")
        for bar, value in zip(bars, values):
            axes[0].text(bar.get_x() + bar.get_width()/2, value + (.13 if value >= 0 else -.13), f"{value:+.2f}",
                         ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    axes[0].set_xticks(x, [e["release_date"][5:] for e in events])
    axes[0].set_ylim(-8, 5)
    axes[0].set_ylabel("成本后事件收益（%）")
    axes[0].set_xlabel("2018年金融数据公开日")
    axes[0].set_title("全部六次保留：20日窗口仅一次为正", loc="left", fontsize=13, pad=12)
    axes[0].legend(frameon=False, loc="upper left", ncol=2)
    values = [parts["special_bond_change_yi"], parts["other_tsf_change_yi"], parts["total_change_yi"]]
    bars = axes[1].bar(np.arange(3), values, .55, color=["#b78943", "#b56549", "#337c86"])
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x()+bar.get_width()/2, value + (100 if value >= 0 else -100), f"{value:+,.0f}",
                     ha="center", va="bottom" if value >= 0 else "top", fontsize=12)
    axes[1].set_xticks(np.arange(3), ["地方专项债", "剔除专项债的\n其他社融合计", "社融合计"])
    axes[1].set_ylim(-1150, 4250)
    axes[1].set_ylabel("9月比8月变化（亿元）")
    axes[1].set_title("同一发布版本：社融增加来自哪里", loc="left", fontsize=13, pad=12)
    axes[1].text(.03, .95, "两个月均取2018年10月17日原表\n3,283 − 515 = 2,768", transform=axes[1].transAxes,
                 va="top", fontsize=10, color="#434e51")
    fig.suptitle("指数历史对照：融资总量、资金传导与收益需要分别解释", x=.06, ha="left", fontsize=17, y=.98)
    fig.text(.06, .064, "左图沿用原10万元事件预算及费用；保持次日开盘进入和固定开盘退出。窗口重叠，不是六次独立试验或完整账户。", fontsize=9, color="#4e595a")
    fig.text(.06, .028, "右图是同口径月度构成，不能叫作预期差；其他社融包含多种主体。8月13日完整报告前，7月贷款总额已在8月11日公开。", fontsize=9, color="#4e595a")
    fig.subplots_adjust(left=.065, right=.98, top=.84, bottom=.205, wspace=.25)
    fig.savefig(OUT / "信用构成与指数阶段对照.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def report():
    states = read(OUT / "event_states.json")
    returns = read(OUT / "events_and_returns.json")
    comparisons = read(OUT / "expectation_comparison.json")
    monthends = read(OUT / "monthends.json")
    vintages = read(OUT / "tsf_vintages.json")
    contexts = read(OUT / "known_macro_context.json")
    sources = {r["id"]: r for r in read(OUT / "source_resolution.json")["sources"]}
    first, last = monthends[0], monthends[-1]
    period = {"start": first["date"], "end": last["date"],
              "index_price_return": last["close"] / first["close"] - 1,
              "pe_change": last["original_pe_official"] / first["original_pe_official"] - 1,
              "fdr20_change_bp": 100 * (last["fdr20"] - first["fdr20"]),
              "cgb10_change_bp": 100 * (last["cgb_10y"] - first["cgb_10y"])}
    summary = []
    for horizon in [5, 20]:
        values = np.array([r["net_return"] for r in returns if r["horizon"] == horizon])
        summary.append({"horizon": horizon, "n": len(values), "positive_count": int((values > 0).sum()),
                        "mean_net_return": float(values.mean()), "median_net_return": float(np.median(values)),
                        "minimum_net_return": float(values.min()), "maximum_net_return": float(values.max())})
    main = next(r for r in summary if r["horizon"] == 20)
    positive_loan = [r for r in comparisons if r["series"] == "rmb_loan_flow" and r["surprise"] is not None and r["surprise"] > 0]
    positive_summary = {"observations": len(positive_loan), "positive_20d": sum(r["net_return_20d"] > 0 for r in positive_loan),
                        "mean_20d": float(np.mean([r["net_return_20d"] for r in positive_loan])),
                        "role": "自然正负号的描述，包括回述调查和提前披露；不是新注册策略或独立胜率估计。"}
    local = lambda p, label: f"[{label}](<{p.as_posix()}>)"
    external = lambda source_id, label: f"[{label}]({sources[source_id]['url']})"
    lines = ["# 510300历史发现：2018年下半年信用传导与指数定价对照", "",
        "**结论：此前下跌、整体PE下降、贷款好于调查值，并未在这段历史中形成稳定的公告后指数收益。保留宏观信息作为背景，结束对单条信用消息买入规则的继续包装。**", "",
        "本轮承接上一轮已指定的2018年7—12月六个公开月，对应2018年6—11月统计期，全部保留。研究对象为沪深300整体；没有另找公司证明故事，也没有按后续涨跌重新划分阶段。2019年1—4月作为已经看过的对照，并非样本外验证。", "",
        "**六个公开节点与原费用合同**", "",
        "| 金融报告公开日 | 人民币贷款实际/调查值（亿元） | M2实际/调查值（%） | 此前60日指数涨幅 | 5日事件净收益 | 20日事件净收益 |",
        "|---|---:|---:|---:|---:|---:|"]
    for state in states:
        m = state["stat_month"]
        loan = next(r for r in comparisons if r["stat_month"] == m and r["series"] == "rmb_loan_flow")
        money = next(r for r in comparisons if r["stat_month"] == m and r["series"] == "m2_yoy")
        lookback = next(r for r in state["lookbacks"] if r["horizon"] == 60)
        lines.append(f"| {state['release_date']} | {loan['actual']:,.0f} / {loan['expected']:,.0f} | {money['actual']:.1f} / {money['expected']:.1f} | {lookback['price_return']:+.2%} | {loan['net_return_5d']:+.2%} | {loan['net_return_20d']:+.2%} |")
    lines += ["",
        f"六个20日窗口有{main['positive_count']}个为正，简单均值{main['mean_net_return']:+.2%}，中位数{main['median_net_return']:+.2%}。四次贷款高于所取得调查代表值，其中三次20日净收益为负。所有发布前60日指数均下跌；这些现象反对把‘还没涨、估值便宜、贷款超预期’直接当作买入依据。它们不证明信用越差越值得买，也不证明宏观信息没有任何条件性作用。", "",
        "这批贷款和M2预期共有12项，加上11月社融1项，共13项观测；其中6、7月贷款/M2仅取得发布当日报道回述，8、9、10、11月有公布前路透调查原文。其余5个月社融调查值缺失，未以同比或前值填补。媒体调查也不是按股票资金权重测得的市场共识。", "",
        "前四个公开节点原V4只有另一套E0毛收益窗口，本轮按近期已经使用的事件合同补算5/20日开盘至开盘成本后收益；最后两个公开节点直接复用已保存结果。旧E0以第20日收盘为终点，不能与本轮第20个交易日后开盘的结果混称一次复算。预算10万元，单边滑点0.1%、佣金0.04%且最低5元、0.001元价位、100份整手与分红均沿用。收益分母为实际投入，不是20万元全账户；重叠窗口也不构成独立试验。", "",
        "**信息本身为什么容易被看错**", "",
        "一是，同一整包报告的字段，未必同时成为新信息。银保监会8月11日原文已披露7月人民币贷款1.45万亿元；完整金融报告在8月13日发布。原8月14日入场窗口继续保留，但不能把这1.45万亿元都解释成13日晚市场第一次获知的意外。" + external("cbirc_early_july", "8月11日监管原文同期转载") + "。", "",
        "二是，银行贷款增加与其他融资渠道收缩可以同时发生。7月银行人民币贷款合计1.45万亿元，社融中的实体人民币贷款为1.29万亿元；两者范围不同。同期委托贷款、信托贷款、未贴现承兑汇票合计减少4886亿元，8月这三项仍合计减少2674亿元。贷款支持确实存在，但不能据此认定所有融资约束已解除。" + external("tsf_july", "7月央行报告") + "、" + external("tsf_august", "8月央行报告同期转载") + "。", "",
        "三是，总量改善可能集中在特定渠道。10月17日发布的同一张表中，9月社融22054亿元、8月19286亿元，相差2768亿元；地方专项债分别7389亿元和4106亿元，相差3283亿元。因此，剔除专项债后，其他社融合计从15180亿元降至14665亿元，减少515亿元。这里‘其他社融’包括不同主体和融资方式，不能进一步改叫私人融资。" + external("tsf_september", "央行前三季度报告及原表同期转载") + "。", "",
        "| 同一10月17日版本 | 8月（亿元） | 9月（亿元） | 9月减8月（亿元） |",
        "|---|---:|---:|---:|",
        "| 社融合计 | 19,286 | 22,054 | +2,768 |",
        "| 地方专项债 | 4,106 | 7,389 | +3,283 |",
        "| 剔除专项债的其他社融 | 15,180 | 14,665 | −515 |", "",
        "这次报告才将地方专项债纳入社融；更早的7月报告已纳入存款类机构ABS和贷款核销。同一6月社融，7月13日初报为约11800亿元，8月13日回溯表为13922亿元，10月17日表又为14852亿元。既有融资因统计范围或版本改变而重新进入指标，不等于后来的月份又向企业支付同样多的新现金；各版本差额也不能全部强归因于新增字段。", "",
        "10月17日同期报道回述社融预期约15500亿元，但未说明调查对象及专项债口径。用新总量相减得到+6554亿元，剔除专项债后相减则为−835亿元，符号能够翻转。这里只演示混合口径的问题，**没有证据认定原调查一定排除了专项债，因此两个数字均未进入有效预期差信号**。" + external("pboc_september_briefing", "当日发布会直接报道") + "。", "",
        "**政策与融资变化背后的约束**", "",
        "8月11日监管原文提出三类具体措施：提高利润留存和补充资本以支持放贷能力；处置不良、债转股及盘活存量以腾出资产负债表空间；尽职免责和适度提高小微不良容忍度以影响放贷意愿。这些措施说明资金数量之外，还有资本、信用风险和机构激励约束。公告提出保障有效融资需求，并不能证明全社会需求或指数利润已经改善。", "",
        "同样要保留传导有效的反证：10月17日发布会已报告，9月新发放500万元以下小微贷款平均利率6.25%，比上半年低0.17个百分点。部分借款人成本确有下降，不能用指数下跌倒推所有宽松均无效；这一利率也不能代表沪深300全部公司的平均融资成本。", "",
        "**整体估值与资金价格并未给出简单答案**", "",
        f"2018年6月末至12月末，沪深300价格下跌{-period['index_price_return']:.2%}，原中证PE由{first['original_pe_official']:.2f}降至{last['original_pe_official']:.2f}；FDR007最近20次均值下降{-period['fdr20_change_bp']:.2f} bp，10年国债收益率下降{-period['cgb10_change_bp']:.2f} bp。这些固定月末的共同变化表明，市场资金价格降低并不自动保证指数上涨；不能将因子下行统一解释成利好。", "",
        f"宏观背景也在变：在六次发布前已经可见的制造业新订单读数，从6月的{contexts[0]['pmi_new_orders']:.1f}逐步降到11月的{contexts[-1]['pmi_new_orders']:.1f}，均未低于50却持续走弱。因此，‘订单仍在扩张区间’也不等于边际增长改善。这是既有初次发布记录的描述，不是为本轮亏损新增一个PMI过滤器。", "",
        "2018年12月节点PE约10.96、20日事件净收益−3.54%；2019年1月节点PE约10.84、同合同净收益+9.42%。相近估值水平可以对应不同后续收益。2019年1—4月的三个正窗口又大多观察同一次上涨，不能凭它们认定该阶段已经找到稳定规律。详见" + local(PRIOR / "历史发现_指数整体估值与已发生定价.md", "上一轮指数整体定价") + "。", "",
        "**这轮改变了后续研究动作**", "",
        "停止将单月贷款超预期、社融总量增加或低PE直接包装为独立买入规则；不增加阈值去挽救2018年的亏损节点，也不反向使用失败结果。宏观数据继续用于解释约束和竞争机制，缺失的5项社融预期不再无界搜索。MACRO-02与VAL04原失败保持原状。", "",
        "下一项只补现有2018年第四季度至2019年第一季度指数时间线中的外部条件：复用已有美债拆解方法，核对同日名义/TIPS利率、通胀补偿代理和当时美联储声明，区分增长担忧与政策路径变化。这与已做过的2023年、2020—2021年公司案例不同；研究对象保持指数整体，不将美债下降直接当买入信号，也不把当前修订模型当成当时可得。", "",
        "本轮完成12个固定成本窗口、13项调查观测和两次统计扩围的历史对照；没有新增完整账户或拟合规则。夏普1.2、年化10%的完整账户目标仍未达到。", "",
        "文件：" + "、".join([local(OUT / "events_and_returns.json", "全部成本后窗口"), local(OUT / "expectation_comparison.json", "实际与调查值"),
            local(OUT / "tsf_vintages.json", "统计版本与同口径构成"), local(OUT / "mechanism_evidence.json", "提前披露与传导约束"),
            local(OUT / "信用构成与指数阶段对照.png", "历史图"), local(ROOT / "research/historical_index_credit_regime_compare_v1.py", "研究脚本")]) + "。", ""]
    name = "历史发现_2018下半年信用与指数对照.md"
    (OUT / name).write_text("\n".join(lines), encoding="utf-8")
    draw()
    result = {"study_id": STUDY, "completed_at": now(), "classification": "PROGRESS_INDEX_CREDIT_STAGE_COUNTEREVIDENCE",
              "events": len(states), "return_rows": len(returns), "expectation_observations": 13,
              "missing_tsf_expectations": 5, "summary": summary, "positive_loan_context": positive_summary,
              "calendar_changes": period, "same_release_tsf_decomposition": vintages["same_release_monthly_decomposition"],
              "research_action": "STOP_STANDALONE_CREDIT_HEADLINE_BUY_RULE_NO_THRESHOLD_RESCUE",
              "discovery": "2018年下半年六个发布窗口20日净收益仅一正，前期下跌与贷款高于调查值仍不足以形成优势；7月贷款已提前披露，9月同口径社融增加2768亿元由专项债增加3283亿元覆盖，其他渠道减少515亿元。",
              "next_historical_question": "固定2018Q4至2019Q1指数历史，复用现有美债拆解方法补名义/TIPS同日数据及Fed当时声明，区分全球增长担忧和政策路径变化；保持指数层面，不复活旧跨境方向策略或用后来修订模型倒填。",
              "report": name, "new_parameters_fitted": 0, "new_full_accounts": 0, "net_sharpe": None,
              "goal_achieved": False, "orders_authorized": False, "independent_validation": False}
    save("result.json", result)
    print(json.dumps({"六个固定事件": main, "贷款高于调查值的描述": positive_summary, "下一动作": result["research_action"]}, ensure_ascii=False, indent=2), flush=True)


def record_progress():
    result = read(OUT / "result.json")
    assert (OUT / result["report"]).is_file()
    prefix = str(OUT.relative_to(ROOT)).replace("\\", "/") + "/"
    stamp, changes = now(), []
    for name in ["510300_historical_cause_discovery_v1.json", "510300_existing_data_training_mandate_v1.json"]:
        path = ROOT / "config" / name
        cfg = read(path)
        before = dict(cfg)
        if name == "510300_historical_cause_discovery_v1.json":
            cfg.update(current_study=prefix + "protocol.json", latest_completed_study=prefix + "result.json", latest_report=prefix + result["report"], updated_at=stamp)
        else:
            cfg.update(current_round=STUDY, latest_progress_receipt=prefix + "result.json",
                       latest_historical_index_credit_regime_compare=prefix + "result.json",
                       latest_continuation_report=prefix + result["report"], latest_historical_report=prefix + result["report"],
                       latest_continuation_classification=result["classification"], current_driver_continuation_classification=result["classification"],
                       current_driver_consecutive_blocked_goal_turns=0, latest_historical_diagnostic_at=stamp,
                       latest_goal_service_status="active", latest_goal_service_status_observed_at=stamp, goal_status="active", goal_achieved=False,
                       local_goal_work_status="ACTIVE_HISTORICAL_ONLY", last_research_result=result["discovery"],
                       last_source_result="新增10份历史来源，连接13项调查观测；核对社融扩围原表和银保监会提前披露。",
                       next_research_question=result["next_historical_question"])
        changes.append({"path": str(path.relative_to(ROOT)), "fields": {k: {"before": before.get(k), "after": v} for k, v in cfg.items() if before.get(k) != v}})
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save("authority_update.json", {"recorded_at": stamp, "previous_goal_turn_classification": "PROGRESS",
                                  "current_goal_turn_classification": "PROGRESS_COMPLETED_INDEX_CREDIT_STAGE_COMPARISON",
                                  "changes": changes, "goal_achieved": False, "orders_authorized": False})
    print("已登记历史阶段反证及停止单消息买入规则研究的动作；完整账户目标仍未实现。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="指数信用信息的固定历史阶段对照")
    parser.add_argument("mode", choices=["prepare", "fetch", "build-local", "evidence", "report", "record-progress"])
    args = parser.parse_args()
    {"prepare": prepare, "fetch": fetch, "build-local": build_local, "evidence": build_evidence,
     "report": report, "record-progress": record_progress}[args.mode]()


if __name__ == "__main__":
    main()
