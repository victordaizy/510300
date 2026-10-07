"""沪深300历史信息顺序：复用固定收益，分开当时状态与后来消息。"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from credit_recency_structure_v23 import Ledger
from historical_index_liquidity_transmission_v1 import clean
from historical_index_us_rates_policy_v1 import curve, asof_rate
from historical_price_gap_causes_v1 import round_trip

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_index_joint_information_clock_v1"
BASE = ROOT / "reports/research"
FLOW = BASE / "510300_historical_index_credit_flow_composition_v1"
COMPARE = BASE / "510300_historical_index_credit_regime_compare_v1"
GAP = BASE / "510300_historical_index_credit_expectation_gap_v1"
US = BASE / "510300_historical_index_us_rates_policy_v1"
Q2 = BASE / "510300_historical_index_us_rates_trade_q2_v1"
TZ = ZoneInfo("Asia/Shanghai")
INPUTS = {
    "market": "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet",
    "index": "reports/research/510300_macro_earnings_pricing_bridge_v5/inputs/index_price.parquet",
    "pe": "reports/research/510300_macro_earnings_pricing_bridge_v5/inputs/pe.parquet",
    "funding": "data/raw/macro/510300_macro_stress_2015_v2/fdr007_daily_2015_2026.parquet",
    "cgb": "data/raw/macro/china_government_bond_yields_daily.parquet",
    "pmi": "data/raw/macro/510300_macro_stress_2015_v2/pmi_new_orders_release_vintage_2015_2026.parquet",
    "originals": "reports/research/510300_macro_transmission_context_v4/inputs/loan_originals.json",
    "ledger_originals": "reports/research/510300_credit_recency_structure_v23/inputs/originals.json",
    "ledger_monthly": "reports/research/510300_credit_recency_structure_v23/inputs/monthly.csv",
    "dividends": "data/reference/510300_dividends.csv",
}
SOURCES = [
    {"id": "politburo_20190419", "url": "https://politics.people.com.cn/n1/2019/0419/c1024-31040111.html",
     "known_at": "2019-04-19T18:29:00+08:00", "anchors": ["结构性去杠杆", "松紧适度"],
     "label": "4月19日经济工作会议", "role": "新华社会议原文同期转载",
     "fact": "判断一季度总体平稳、好于预期，同时保留国内下行压力；强调结构性去杠杆、货币松紧适度。未宣布政策利率上调，不直接编码为紧缩冲击。"},
    {"id": "rrr_20190506", "url": "https://www.gov.cn/xinwen/2019-05/06/content_5389151.htm",
     "known_at": "2019-05-06T19:29:00+08:00", "anchors": ["2800", "民营和小微"],
     "label": "县域银行定向降准", "role": "新华社当日原报道，采用该网页时间上界",
     "fact": "县域农商行较低准备金率安排用于民营和小微贷款；公告日和实施日分开。该网页不足以认定开盘前已知，也不意味着资金流入沪深300。"},
    {"id": "bank_risk_20190524", "url": "https://finance.sina.com.cn/roll/2019-05-24/doc-ihvhiqay1132737.shtml",
     "known_at": "2019-05-24T17:49:00+08:00", "anchors": ["严重信用风险", "接管期限"],
     "label": "银行信用风险处置", "role": "中国人民银行署名公告同期转载",
     "fact": "监管机关因严重信用风险接管包商银行；这里观察金融体系信用分层，不研究该银行股票。不能由一条FDR定盘值推断所有融资主体风险同步下降。"},
    {"id": "reuters_april_recall", "url": "https://www.yahoo.com/news/china-banks-temper-april-lending-042257523.html",
     "known_at": "2019-05-09T15:40:00+08:00", "anchors": ["1.02 trillion", "1.2 trillion"],
     "role": "Reuters署名原报道，公布后回述调查，不是公布前调查存档"},
    {"id": "reuters_may_recall", "url": "https://finance.yahoo.com/news/1-china-may-loans-rebound-085024983.html",
     "known_at": "2019-06-12T23:59:59+08:00", "anchors": ["1.18 trillion", "1.225"],
     "role": "Reuters署名原报道，公布后回述调查，采用中国当日末上界"},
]


def now():
    return datetime.now(TZ).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fetch_one(source):
    destination = OUT / "sources" / (source["id"] + ".html")
    if destination.exists():
        raise RuntimeError("来源已存在，停止重复请求：" + source["id"])
    receipt = {**source, "retrieved_at": now(), "historical_receipt_available": False}
    try:
        response = requests.get(source["url"], timeout=(12, 35), headers={"User-Agent": "Mozilla/5.0"})
        receipt.update(http_status=response.status_code, final_url=response.url)
        response.raise_for_status()
        raw = response.content
        destination.write_bytes(raw)
        body = BeautifulSoup(raw, "html.parser").get_text("\n", strip=True)
        destination.with_suffix(".txt").write_text(body, encoding="utf-8")
        receipt.update(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
        compact = re.sub(r"\s+", " ", body)
        receipt["anchor_checks"] = {s: s in compact for s in source["anchors"]}
        if not all(receipt["anchor_checks"].values()):
            raise ValueError("来源未包含全部预定事实定位词")
        receipt["status"] = "SAVED_PRIMARY_OR_SIGNED_ORIGINAL"
    except Exception as error:
        receipt.update(status="ERROR", error=str(error))
    save("sources/" + source["id"] + ".receipt.json", receipt)
    return receipt


def fetch():
    save("source_plan.json", SOURCES)
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(fetch_one, SOURCES))
    save("source_receipts.json", rows)
    for row in rows:
        print(row["id"], row["status"], row.get("error", ""), flush=True)


def load_market():
    market = pd.read_parquet(ROOT / INPUTS["market"]).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).dt.tz_localize(None).dt.normalize()
    opens = pd.DatetimeIndex(market.date).tz_localize(TZ) + pd.Timedelta(hours=9, minutes=30)
    return market, opens


def expectations():
    rows = read(COMPARE / "expectations.json")["observations"] + read(GAP / "expectations.json")["observations"]
    rows = list({(r["stat_month"], r["series"], r["source_id"]): r for r in rows}.values())
    resolutions = read(OUT / "source_resolution.json") if (OUT / "source_resolution.json").exists() else {"resolved_ids": []}
    receipts = {r["id"]: r for r in read(OUT / "source_receipts.json")}
    for month, source_id, series, expected in [
        ("2019-04", "reuters_april_recall", "rmb_loan_flow", 12000),
        ("2019-05", "reuters_may_recall", "rmb_loan_flow", 12250),
        ("2019-05", "reuters_may_recall", "m2_yoy", 8.6),
    ]:
        if receipts[source_id]["status"] != "ERROR" or source_id in resolutions["resolved_ids"]:
            source = next(r for r in SOURCES if r["id"] == source_id)
            rows.append({"stat_month": month, "source_id": source_id, "series": series, "expected": expected,
                         "evidence_tier": "RELEASE_REPORT_RECALL", "source_url": source["url"],
                         "published_date": source["known_at"][:10], "aggregation": "报道回述代表值，不额外推断分布"})
    return rows


def interval_groups(returns, horizon):
    rows = sorted([r for r in returns if r["horizon"] == horizon], key=lambda r: r["entry_date"])
    groups = []
    # 退出开盘与新进入开盘相同时，前一个持仓已经结束，不算同时占用。
    for row in rows:
        if not groups or row["entry_date"] >= groups[-1]["end_exclusive"]:
            groups.append({"start": row["entry_date"], "end_exclusive": row["exit_date"], "events": [row["event_key"]]})
        else:
            groups[-1]["end_exclusive"] = max(groups[-1]["end_exclusive"], row["exit_date"])
            groups[-1]["events"].append(row["event_key"])
    dates = sorted(set(r[k] for r in rows for k in ["entry_date", "exit_date"]))
    counts = [{"date": d, "simultaneous_windows": sum(r["entry_date"] <= d < r["exit_date"] for r in rows)} for d in dates]
    return {"horizon": horizon, "events": len(rows), "connected_groups": groups,
            "max_simultaneous_windows": max(r["simultaneous_windows"] for r in counts), "not_independent_sample_count": True}


def calculate():
    if (OUT / "events_and_returns.json").exists():
        raise RuntimeError("结果已经保存，不调整规则重跑")
    protocol = read(OUT / "protocol.json")
    receipts = read(OUT / "source_receipts.json")
    resolved = read(OUT / "source_resolution.json")["resolved_ids"] if (OUT / "source_resolution.json").exists() else []
    required = [r for r in receipts if "label" in r]
    if any(r["status"] == "ERROR" and r["id"] not in resolved for r in required):
        raise RuntimeError("政策消息原文尚未定位，先处理来源")
    originals = {r["stat_month"]: r for r in read(ROOT / INPUTS["originals"])}
    ledger = Ledger()
    market, opens = load_market()
    calendar = pd.DatetimeIndex(market.date)
    dividends = pd.read_csv(ROOT / INPUTS["dividends"])
    dividends = dividends.loc[dividends.symbol.eq("510300.SH")].copy()
    dividends["record_date"] = pd.to_datetime(dividends.record_date)
    index = pd.read_parquet(ROOT / INPUTS["index"]).sort_values("date")
    index["date"] = pd.to_datetime(index.date)
    index = index.set_index("date")
    pe = pd.read_parquet(ROOT / INPUTS["pe"])
    pe["available"] = pd.to_datetime(pe.available_at, utc=True).dt.tz_convert(TZ)
    funding = pd.read_parquet(ROOT / INPUTS["funding"]).sort_values("date")
    funding["known"] = pd.to_datetime(funding.available_at, utc=True).dt.tz_convert(TZ)
    funding["mean20"] = funding.first_release_value.rolling(20, min_periods=20).mean()
    funding["mean20_change_bp"] = (funding.mean20 - funding.mean20.shift(20)) * 100
    cgb = pd.read_parquet(ROOT / INPUTS["cgb"]).sort_values("date")
    cgb["date"] = pd.to_datetime(cgb.date)
    cgb["change20_bp"] = cgb.cgb_10y.diff(20) * 100
    pmi = pd.read_parquet(ROOT / INPUTS["pmi"]).sort_values("available_at")
    pmi["known"] = pd.to_datetime(pmi.available_at, utc=True).dt.tz_convert(TZ)
    pmi["change"] = pmi.first_release_value.diff()
    nominal = pd.concat([curve(BASE / f"510300_rmb_residual_state_v1/sources/treasury_{y}.xml") for y in [2018, 2019]])
    tips = pd.concat([curve(US / f"sources/tips_{y}.xml", real=True) for y in [2018, 2019]])
    rates = nominal.merge(tips, on="date", validate="one_to_one").sort_values("date").reset_index(drop=True)
    rates["inflation_compensation"] = rates.nominal10 - rates.tips10
    rates["known_at_assumed"] = (rates.date + pd.Timedelta(hours=23, minutes=59, seconds=59)).dt.tz_localize("America/New_York").dt.tz_convert(TZ)
    rates["known_at_delay1_assumed"] = rates.known_at_assumed.shift(-1)
    for f in ["nominal10", "tips10", "inflation_compensation"]:
        rates[f + "_20obs_bp"] = rates[f].diff(20) * 100
    intraday = read(Q2 / "six_meetings_intraday.json")
    forecasts = expectations()
    save("expectations.json", forecasts)
    credit = []
    for month in protocol["stat_months"]:
        raw = originals[month]
        body = unicodedata.normalize("NFKC", BeautifulSoup((ROOT / raw["raw_path"]).read_bytes(), "html.parser").get_text(" ", strip=True))
        match = re.search(r"广义货币\(M2\)余额[\d.]+万亿元,同比增长([\d.]+)%", re.sub(r"\s+", "", body))
        if not match:
            raise ValueError("M2原文缺失：" + month)
        components = {f: ledger.row(month, f, "1") for f in ["rmb_total", "corporate_long", "corporate_short", "bills", "household_long"]}
        comparisons = []
        for row in forecasts:
            if row["stat_month"] == month and row["series"] in ["rmb_loan_flow", "m2_yoy"]:
                actual = components["rmb_total"]["value_yi"] if row["series"] == "rmb_loan_flow" else float(match[1])
                comparisons.append({**row, "actual": actual, "actual_minus_expected": actual - row["expected"],
                                    "actual_rounding_half": components["rmb_total"]["rounding_half_yi"] if row["series"] == "rmb_loan_flow" else 0.05})
        credit.append({"stat_month": month, "known_at": raw["conservative_known_at"], "published_at": raw["published_at"],
                       "source_url": raw["source_url"], "raw_path": raw["raw_path"], "m2_yoy": float(match[1]),
                       "components": components, "expectations": comparisons})
    save("credit_releases.json", credit)
    nodes = [{"event_key": "CREDIT_" + r["stat_month"], "kind": "CREDIT", "event_id": r["stat_month"], "known_at": r["known_at"]} for r in credit]
    nodes += [{"event_key": "FOMC_" + r["event_id"], "kind": "FOMC", "event_id": r["event_id"], "known_at": r["full_end_cn"]} for r in intraday]
    nodes.sort(key=lambda r: r["known_at"])
    old_credit = {(r["stat_month"], r["horizon"]): (r, str(path.relative_to(ROOT))) for path in [COMPARE / "events_and_returns.json", FLOW / "events_and_returns.json"] for r in read(path)}
    old_fomc = {(r["event_id"], r["horizon"]): (r, str(path.relative_to(ROOT))) for path in [US / "events_and_returns.json", Q2 / "events_and_returns.json"] for r in read(path)}
    states, returns, errors = [], [], []
    for node in nodes:
        entry_pos = int(opens.searchsorted(pd.Timestamp(node["known_at"]), side="right"))
        entry = market.iloc[entry_pos]
        cutoff = opens[entry_pos]
        ix = index.loc[index.index < entry.date].iloc[-1]
        ix_pos = index.index.get_loc(ix.name)
        latest_pe = pe.loc[pe.available.le(cutoff)].sort_values("available").iloc[-1]
        fund = funding.loc[funding.known.le(cutoff)].iloc[-1]
        bond = cgb.loc[cgb.date.lt(entry.date)].iloc[-1]
        orders = pmi.loc[pmi.known.le(cutoff)].iloc[-1]
        latest_credit = [r for r in credit if pd.Timestamp(r["known_at"]) < cutoff][-1]
        latest_fed = [r for r in intraday if pd.Timestamp(r["full_end_cn"]) < cutoff]
        state = {**node, "entry_date": entry.date.strftime("%Y-%m-%d"), "entry_at": cutoff,
                 "credit_stat_month": latest_credit["stat_month"], "credit_age_days": (cutoff - pd.Timestamp(latest_credit["known_at"])).total_seconds() / 86400,
                 "index_close_date": ix.name, "index_close": ix.close,
                 "index_prior20": float(ix.close / index.iloc[ix_pos - 20].close - 1),
                 "index_prior60": float(ix.close / index.iloc[ix_pos - 60].close - 1),
                 "pe": float(latest_pe.original_pe_official), "pe_date": str(latest_pe.observation_date), "pe_available_at": latest_pe.available,
                 "funding_date": str(fund.date), "funding_available_at": fund.known, "fdr007": float(fund.first_release_value),
                 "fdr20": float(fund.mean20), "fdr20_vs_previous20_bp": float(fund.mean20_change_bp),
                 "cgb_date": bond.date, "cgb10": float(bond.cgb_10y), "cgb10_change20_bp": float(bond.change20_bp),
                 "cgb_known_rule": "原观测后首个ETF开盘才可用，选取严格早于入场日的最后观测",
                 "pmi_month": orders.reference_period, "pmi_known_at": orders.known, "pmi_new_orders": float(orders.first_release_value),
                 "pmi_new_orders_change": float(orders.change), "us_curve": asof_rate(rates, cutoff), "us_curve_delay1": asof_rate(rates, cutoff, True),
                 "last_fomc": latest_fed[-1] if latest_fed else None,
                 "last_fomc_age_days": (cutoff - pd.Timestamp(latest_fed[-1]["full_end_cn"])).total_seconds() / 86400 if latest_fed else None,
                 "last_fomc_scope_note": "仅该段六次会议，首条之前未在本表检索，不等于无政策通信"}
        assert fund.known <= cutoff and orders.known <= cutoff and latest_pe.available <= cutoff and ix.name < entry.date
        assert pd.Timestamp(latest_credit["known_at"]) < cutoff
        states.append(state)
        for horizon in protocol["horizons"]:
            end = market.iloc[entry_pos + horizon]
            old = old_credit if node["kind"] == "CREDIT" else old_fomc
            if (node["event_id"], horizon) in old:
                prior, origin = old[(node["event_id"], horizon)]
                row = {**prior, "event_key": node["event_key"], "kind": node["kind"], "event_id": node["event_id"],
                       "origin": origin, "newly_computed": False}
                assert row["entry_date"] == entry.date.strftime("%Y-%m-%d") and row["exit_date"] == end.date.strftime("%Y-%m-%d")
            else:
                assert node["kind"] == "CREDIT" and node["event_id"] in ["2019-04", "2019-05"]
                div = float(dividends.loc[dividends.record_date.ge(entry.date) & dividends.record_date.lt(end.date), "cash_dividend_per_share"].sum())
                trade = round_trip(float(entry.open), float(end.open), div, protocol["costs"])
                row = {**node, "stat_month": node["event_id"], "horizon": horizon, "entry_date": entry.date.strftime("%Y-%m-%d"),
                       "exit_date": end.date.strftime("%Y-%m-%d"), "entry_open": float(entry.open), "exit_open": float(end.open),
                       "dividend_per_share": div, "gross_return": (float(end.open) + div) / float(entry.open) - 1,
                       **trade, "origin": "原成本和5/20日合约新增", "newly_computed": True}
            error = (row["sell_price"] - row["buy_price"]) * row["shares"] + row["dividend_entitlement_cny"] - row["commissions_cny"] - row["net_pnl_cny"]
            assert abs(error) < 1e-7 and row["paid_cny"] <= protocol["costs"]["illustrative_event_budget_cny"]
            errors.append(abs(error))
            returns.append(row)
    save("event_states.json", states)
    save("events_and_returns.json", returns)
    save("overlap.json", [interval_groups(returns, h) for h in protocol["horizons"]])
    save("calculation_checks.json", {"events": len(states), "return_rows": len(returns), "new_return_rows": sum(r["newly_computed"] for r in returns),
                                     "max_cash_identity_error_cny": max(errors), "checked_known_times": True, "old_return_clocks_match": True})
    inputs = [ROOT / s for s in INPUTS.values()] + [COMPARE / "events_and_returns.json", FLOW / "events_and_returns.json", US / "events_and_returns.json", Q2 / "events_and_returns.json", Q2 / "six_meetings_intraday.json"]
    inputs += [BASE / f"510300_rmb_residual_state_v1/sources/treasury_{y}.xml" for y in [2018, 2019]]
    inputs += [US / f"sources/tips_{y}.xml" for y in [2018, 2019]]
    save("input_receipts.json", [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs])
    market.loc[market.date.between("2018-09-28", "2019-07-18")].to_parquet(OUT / "etf_context.parquet", index=False)
    index.loc["2018-09-28":"2019-07-18"].to_parquet(OUT / "index_context.parquet")
    print(pd.DataFrame(returns)[["event_key", "horizon", "entry_date", "exit_date", "net_return", "newly_computed"]].to_string(index=False), flush=True)


def timeline():
    if (OUT / "information_timeline.json").exists():
        raise RuntimeError("时间线已保存，停止覆盖")
    states = read(OUT / "event_states.json")
    returns = read(OUT / "events_and_returns.json")
    nodes = [{"id": r["event_key"], "known_at": r["known_at"], "kind": r["kind"],
              "label": ("信用公布" if r["kind"] == "CREDIT" else "FOMC完整通信") + r["event_id"]} for r in states]
    old = read(Q2 / "mechanism_evidence.json")
    facts = {r["source_id"]: r for r in old["facts"]}
    for row in old["coevent_timeline"]:
        stamp = row.get("known_at", row.get("known_at_assumed", row.get("known_at_upper_assumed")))
        nodes.append({"id": row["source_id"], "kind": "TRADE", "known_at": stamp, "label": facts[row["source_id"]]["fact"],
                      "source_url": facts[row["source_id"]]["url"], "clock_note": row["clock_note"]})
    nodes.extend({"id": r["id"], "kind": "DOMESTIC_POLICY_OR_RISK", "known_at": r["known_at"], "label": r["label"],
                  "fact": r["fact"], "source_url": r["url"], "clock_note": r["role"]} for r in SOURCES if "label" in r)
    nodes.sort(key=lambda r: r["known_at"])
    save("information_timeline.json", nodes)
    mappings = []
    for row in returns:
        entry = pd.Timestamp(row["entry_date"], tz=TZ) + pd.Timedelta(hours=9, minutes=30)
        end = pd.Timestamp(row["exit_date"], tz=TZ) + pd.Timedelta(hours=9, minutes=30)
        new = [n for n in nodes if entry < pd.Timestamp(n["known_at"]) < end]
        mappings.append({"event_key": row["event_key"], "horizon": row["horizon"], "entry_at": entry, "exit_at": end,
                         "new_information_ids": [n["id"] for n in new], "new_information_count_lower_bound": len(new),
                         "not_causal_contribution": True})
    save("holding_window_new_information.json", mappings)
    market, _ = load_market()
    market = market.set_index("date")
    # 固定消息日期前后的价格分段，不产生额外交易收益或选择退出点。
    landmarks = [("2019-04-15", "open", "原信用窗口进入"), ("2019-04-19", "close", "政策会议公开前最后收盘"),
                 ("2019-04-22", "open", "会议消息后首个开盘"), ("2019-04-30", "close", "五一休市前收盘"),
                 ("2019-05-06", "open", "贸易升级后首个开盘"), ("2019-05-16", "open", "原20日固定退出")]
    points = [{"date": d, "field": f, "label": label, "price": float(market.loc[pd.Timestamp(d), f])} for d, f, label in landmarks]
    pieces = [{"from": a, "to": b, "price_return": b["price"] / a["price"] - 1,
               "log_return_points": float(np.log(b["price"] / a["price"]) * 100)} for a, b in zip(points[:-1], points[1:])]
    entire = points[-1]["price"] / points[0]["price"] - 1
    compound = float(np.prod([1 + r["price_return"] for r in pieces]) - 1)
    assert abs(compound - entire) < 1e-12
    original = next(r for r in returns if r["event_key"] == "CREDIT_2019-03" and r["horizon"] == 20)
    assert abs(entire - original["gross_return"]) < 1e-12
    save("april_credit_path.json", {"points": points, "pieces": pieces, "entire_gross_price_return": entire,
                                    "original_net_return": original["net_return"], "compound_identity_error": abs(compound - entire),
                                    "not_exit_strategy": True, "no_causal_attribution": True})
    print("已保存23个已核实节点的先后关系及原4月信用窗口价格分段。", flush=True)


def reconcile_headlines():
    if (OUT / "monthly_headline_reconciliation.json").exists():
        raise RuntimeError("原文总额对照已保存，不覆盖")
    rows = []
    for row in read(OUT / "credit_releases.json"):
        body = unicodedata.normalize("NFKC", BeautifulSoup((ROOT / row["raw_path"]).read_bytes(), "html.parser").get_text(" ", strip=True))
        compact = re.sub(r"\s+", "", body)
        month_number = int(row["stat_month"][-2:])
        pattern = rf"(?<!\d){month_number}月(?:份|当月)?,?人民币贷款增加([\d.]+)(万亿|亿)元"
        matches = list(dict.fromkeys(re.findall(pattern, compact)))
        if len(matches) != 1:
            raise ValueError("当月总量原文不唯一：" + row["stat_month"])
        literal, unit = matches[0]
        factor = 10000 if unit == "万亿" else 1
        decimals = len(literal.split(".")[1]) if "." in literal else 0
        value = float(literal) * factor
        half = 0.5 * factor * 10 ** (-decimals)
        old = row["components"]["rmb_total"]
        difference = value - old["value_yi"]
        assert abs(difference) <= half + old["rounding_half_yi"] + 1e-8
        gaps = [{**r, "headline_actual_yi": value, "headline_surprise_yi": value - r["expected"],
                 "headline_rounding_half_yi": half} for r in row["expectations"] if r["series"] == "rmb_loan_flow"]
        rows.append({"stat_month": row["stat_month"], "literal": literal + unit + "元", "direct_monthly_yi": value,
                     "direct_monthly_rounding_half_yi": half, "ledger_yi": old["value_yi"],
                     "ledger_rounding_half_yi": old["rounding_half_yi"], "direct_minus_ledger_yi": difference,
                     "source_url": row["source_url"], "raw_path": row["raw_path"], "expectations": gaps})
    save("monthly_headline_reconciliation.json", {"recorded_at": now(),
          "reason": "核对发现累计端点差的月度总量与同份原文直接公布的月度值存在舍入差；保留原Ledger结果，另列原文值用于解读调查预期差。", 
          "after_return_calculation": True, "entry_exit_return_or_parameters_changed": False, "rows": rows})
    print("已将9份原报告的当月总量与累计差分对照；2018-12和2019-03差异在原显示舍入范围内。", flush=True)


def draw():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False,
                         "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    states = read(OUT / "event_states.json")
    returns = read(OUT / "events_and_returns.json")
    index = pd.read_parquet(OUT / "index_context.parquet")
    index.index = pd.to_datetime(index.index)
    colors = {"CREDIT": "#1b6d85", "FOMC": "#ba7444"}
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={"height_ratios": [2.1, 2.9, 1.5]})
    fig.patch.set_facecolor("#f7f5f0")
    for ax in axes:
        ax.set_facecolor("#f7f5f0")
    normalized = index.close / index.close.iloc[0] * 100
    axes[0].plot(index.index, normalized, color="#263b44", linewidth=2)
    for kind, marker in [("CREDIT", "o"), ("FOMC", "s")]:
        subset = [r for r in states if r["kind"] == kind]
        dates = pd.to_datetime([r["entry_date"] for r in subset])
        axes[0].scatter(dates, normalized.loc[dates], color=colors[kind], marker=marker, s=36, zorder=4)
    axes[0].set_title("沪深300整体：同一条价格路径，接受不同时间到达的信息", loc="left", fontsize=15, fontweight="bold")
    axes[0].set_ylabel("价格指数，2018-09-28＝100")
    axes[0].legend(handles=[Line2D([0], [0], color="#263b44", label="沪深300价格指数"),
                            Line2D([0], [0], color=colors["CREDIT"], marker="o", linestyle="", label="信用数据后进入日"),
                            Line2D([0], [0], color=colors["FOMC"], marker="s", linestyle="", label="FOMC后进入日")], loc="upper left", frameon=False, ncol=3)
    selected = [r for r in returns if r["horizon"] == 20]
    labels = []
    for pos, row in enumerate(selected):
        start, end = pd.Timestamp(row["entry_date"]), pd.Timestamp(row["exit_date"])
        axes[1].plot([start, end], [pos, pos], color=colors[row["kind"]], linewidth=7, solid_capstyle="butt")
        axes[1].text(end + pd.Timedelta(days=2), pos, f'{row["net_return"]:+.2%}', va="center", fontsize=9)
        labels.append(("信用 " if row["kind"] == "CREDIT" else "FOMC ") + row["event_id"])
    axes[1].set_yticks(range(len(selected)), labels)
    axes[1].invert_yaxis()
    axes[1].set_title("15个固定20日窗口大量交叠：收益不能相加，事件数不等于独立机会数", loc="left", fontsize=13)
    axes[1].set_xlim(pd.Timestamp("2018-09-28"), pd.Timestamp("2019-08-08"))
    for ax in axes[:2]:
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.grid(axis="x", alpha=0.15)
    pieces = read(OUT / "april_credit_path.json")["pieces"]
    names = ["4/15开 → 4/19收", "4/19收 → 4/22开", "4/22开 → 4/30收", "4/30收 → 5/6开", "5/6开 → 5/16开"]
    values = [r["price_return"] * 100 for r in pieces]
    axes[2].bar(names, values, color=["#ab583f" if x > 0 else "#3a8073" for x in values], width=0.62)
    for pos, value in enumerate(values):
        axes[2].text(pos, value + (0.22 if value >= 0 else -0.22), f"{value:+.2f}%", ha="center", va="bottom" if value >= 0 else "top")
    axes[2].axhline(0, color="#444444", linewidth=0.7)
    axes[2].set_ylim(-6.7, 3.1)
    axes[2].set_ylabel("各段未扣费价格收益")
    axes[2].set_title("4月强信用窗口：贸易升级前已出现回落；日期分段不是因果归因或新退出规则", loc="left", fontsize=13)
    fig.suptitle("2018Q4—2019Q2 · 指数信息顺序与可成交窗口", x=0.08, ha="left", fontsize=20, fontweight="bold", y=0.985)
    fig.text(0.08, 0.02, "中图沿用100,000元单次预算、双边0.04%佣金（最低5元）、各边0.1%滑点、整手与价格取整。事件收益不是完整账户收益。", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0.02, 0.045, 0.99, 0.965), h_pad=2.2)
    fig.savefig(OUT / "指数信息时序与交叠.png", dpi=160)
    plt.close(fig)
    print("已生成指数时序、窗口交叠与价格分段图。", flush=True)


def report():
    states = read(OUT / "event_states.json")
    returns = read(OUT / "events_and_returns.json")
    credit = {r["stat_month"]: r for r in read(OUT / "credit_releases.json")}
    headlines = {r["stat_month"]: r for r in read(OUT / "monthly_headline_reconciliation.json")["rows"]}
    overlap = read(OUT / "overlap.json")
    updates = read(OUT / "holding_window_new_information.json")
    path = read(OUT / "april_credit_path.json")
    index_rows = [r for r in states if r["kind"] == "CREDIT"]
    twenty = [r for r in updates if r["horizon"] == 20]
    n_updated = sum(r["new_information_count_lower_bound"] > 0 for r in twenty)
    def ret(key, h):
        return next(r["net_return"] for r in returns if r["event_key"] == key and r["horizon"] == h)
    def local(label, name):
        return f"[{label}](<{(OUT / name).as_posix()}>)"
    lines = [
        "# 历史发现：指数的驱动状态会被后续消息更新", "",
        "本轮把2018年四季度至2019年二季度的全部九次月度信用公布和六次FOMC会议放回同一条沪深300价格路径。直接结论是：这组历史里的20日收益不能作为原起点某个因子的单独贡献，且多个看似成功的因子窗口共享同一段行情。联合状态更接近指数研究，但尚未形成达到夏普1.2的可执行规则。", "",
        f"15个原20日窗口全部跨过了至少一次本表记录的后续消息更新（{n_updated}/15）。月度数据和定期会议本来就会落进不少20日窗口；这个计数说明归因存在混杂，不证明原判断每次都失效，也不能独自证明或否定预测能力。表内只有23个节点，国内政策及贸易部分是定向补充；这是已核实的下界，不是完整新闻库，也没有识别各消息的因果系数。", "",
        "研究只观察指数整体：资金价格、信用期限用途、订单、海外利率构成和指数此前定价。银行风险处置在这里是金融体系信用分层的证据，不延伸为个股研究。", "",
        "## 一、先看全部九次信用公布", "",
        "预期差取当月人民币贷款原文总额减调查值，单位亿元；`前`为已经找到公布前调查原文，`后述`为公布后报道回述，后者不能作为已证明的公布前预期档案。部分回述甚至发表于原进入日，未核实它早于开盘，因此这里只作历史解释，不被认定为该入场的可用信号。2019年2月保留两家调查，不按收益选择。经济学家调查也不等于边际股票资金的真实定价。", "",
        "|统计月／原入场日|贷款预期差|此前60日指数涨幅|最近PMI新订单|资金均值变化bp|5日净收益|20日净收益|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in index_rows:
        gaps = headlines[row["event_id"]]["expectations"]
        labels = [f'{r["headline_surprise_yi"]:+,.2f}' .rstrip("0").rstrip(".") + (" 前" if r["evidence_tier"] == "PRE_RELEASE_PUBLICATION" else " 后述") for r in gaps]
        lines.append(f'|{row["event_id"]}／{row["entry_date"]}|{"；".join(labels) or "缺失"}|{row["index_prior60"]:+.2%}|{row["pmi_new_orders"]:.1f}|{row["fdr20_vs_previous20_bp"]:+.2f}|{ret(row["event_key"],5):+.2%}|{ret(row["event_key"],20):+.2%}|')
    lines += ["", "资金均值变化是截至入场前最近20个FDR007有效观测的均值，相对再前20个观测的变化；它是上午定盘序列，不是全天DR007。负值只说明这一资金价格回落，不证明所有借款主体融资约束缓和。此前涨幅取入场前最后收盘，不用本次入场开盘以后价格。", "",
              "以上均为单次100,000元预算的510300事件窗口，按原规则在开盘进入，5或20个交易日后的开盘退出，计入双边0.04%佣金、单边最低5元、各边0.1%滑点、整手、0.001元价格取整及持有期分红权益。百分比分母为实际买入支出；不等于20万元完整账户收益。复用26条旧收益，仅新增最后两次信用事件的四条收益。", "",
              "同份原文的月度总额与累计端点差偶有舍入差：2018年12月直接为10,800亿元，旧Ledger为10,830±150.5亿元；2019年3月直接为16,900亿元，旧Ledger为17,000±100亿元。两者在显示精度范围内相容，本表解读预期差用直接公布值，保留全部旧计算和原交易结果。", "",
              "## 二、把‘宽松’拆成同时存在的不同变化", "",
              "|统计月／进入日|FDR007近20观测均值|中债10年近20观测变化bp|美债名义／TIPS／通胀补偿近20观测变化bp|可用官方PE|",
              "|---|---:|---:|---|---:|" ]
    for row in index_rows:
        us = row["us_curve"]
        rates_label = "／".join(f'{us[k]:+.0f}' for k in ["nominal10_20obs_bp", "tips10_20obs_bp", "inflation_compensation_20obs_bp"])
        lines.append(f'|{row["event_id"]}／{row["entry_date"]}|{row["fdr20"]:.4f}%|{row["cgb10_change20_bp"]:+.2f}|{rates_label}|{row["pe"]:.2f}|')
    lines += ["", "2019年2月18日，信用超预期、国内资金均价回落与海外实际收益率下降同时出现，指数此前60日仅涨4.17%。到4月15日，强信用公布时指数此前60日已涨29.81%，国内资金均价和国债收益率都在上行。这里的实际历史状态不同；这不证明‘涨过某个阈值就应该卖’，本轮未搜索该阈值。", "",
              "两次新增反例继续限制简单推论：5月10日和6月13日入场都对应贷款低于回述调查，但原20日净收益分别为−1.81%和＋3.57%。6月13日可知的订单已低于50、海外名义和实际收益率同步下降；随后持有期又经历6月18日中美元首通话、6月FOMC以及大阪会晤。不能把后来的上涨认作‘弱信用天然利好’，也不能让6月13日提前知道这些后续消息。", "",
              "美债主表沿用美东观测日日终才可知的保守时钟，并在JSON中保留额外延迟一个美国观测日的对照。名义减TIPS是通胀补偿，含风险和流动性成分，不能直接命名为纯通胀预期；美债下降也可能与增长或贸易担忧同源。中债估值曲线沿用下一ETF开盘可用的假设。全部是今天重建的历史数据，未把下载日期伪装成当年接收记录。", "",
              "## 三、4月强信用窗口：回落并非全部发生在贸易升级之后", "",
              "4月12日公布的是3月信用，原窗口4月15日开盘进入、5月16日开盘退出。期间4月19日会议既肯定一季度表现，也强调结构性问题和政策适度；它没有宣布加息，因此不直接给它贴‘收紧’交易标签。来源：[新华社原文同期转载](http://politics.people.com.cn/n1/2019/0419/c1024-31040111.html)。", "",
              "|固定消息边界之间的价格段|未扣费价格收益|",
              "|---|---:|" ]
    for piece in path["pieces"]:
        a, b = piece["from"], piece["to"]
        lines.append(f'|{a["date"]}{"开" if a["field"]=="open" else "收"} → {b["date"]}{"开" if b["field"]=="open" else "收"}|{piece["price_return"]:+.2%}|')
    before_trade = path["points"][3]["price"] / path["points"][0]["price"] - 1
    lines += ["", f"到4月30日收盘，相对原4月15日开盘已经回落{abs(before_trade):.2%}。5月6日开盘又相对休市前收盘低开3.03%，其后至原退出点再跌1.88%。各段以乘法衔接，合计未扣费−8.33%，原完整窗口净收益仍是−8.62%。4月22日开盘与4月19日收盘相同，不能把随后几天的−5.44%直接当成会议即时冲击。", "",
              "5月初贸易公告、5月10日关税实施确认、5月13日中方公告均按原Q2来源保留；宣布和生效不作为两个独立预期差。5月6日县域银行准备金安排指向民营和小微贷款，政策金额不等于指数新增买盘。[定向准备金安排](https://www.gov.cn/xinwen/2019-05/06/content_5389151.htm)。", "",
              "这段分解说明价格在信息不断变化时如何走过来；没有把退出日改到局部高点，也没有计算或推荐新的消息退出策略。统计上的原20日亏损真实存在，归因时则必须保留这几个不同时间段。", "",
              "## 四、多个因子可能重复描述同一段指数行情", "",
              f'15个20日窗口最多有{overlap[1]["max_simultaneous_windows"]}个同时占用同一交易日，按严格交叠连接形成{len(overlap[1]["connected_groups"])}组；相同开盘时一笔退出、另一笔进入不算同时持仓。这不是三个独立样本，也不是可以执行的三笔交易。', "",
              "2019年1月FOMC与12月、1月信用事件的20日收益共享一部分1—3月上涨行情；合在一起的证据不会凭空增加独立机会或可用本金。20万元、最高50%敞口的完整账户必须处理重叠信号和退出，不能把每个100,000元事件的收益相加。", "",
              "5月24日的银行接管公告也保留在持有期时间线中。公告证明发生了信用风险处置，但本轮没有完整测量同业分层价差，不能因资金定盘下降就宣布金融体系风险已经消失。[央行、银保监会原公告](https://finance.sina.com.cn/roll/2019-05-24/doc-ihvhiqay1132737.shtml)。", "",
              "## 五、对目标的处理", "",
              "本轮有用的发现是信息必须随新公告更新，且指数的同一段重估不能被多个因子重复算作机会。尚未证明组合后预测更准，也没有因果识别、独立样本验证或完整账户新增结果。夏普1.2、年化10%及既有风险约束仍未达成；净夏普保留NOT_COMPUTED，不能用事件平均收益替代。", "",
              "旧V4/V23已经做过广义宏观联合描述，旧全球流动性研究使用过2019年作选择段；本轮不称这段历史为未见样本。MACRO-02与其他已终止分支保持原结论。只保留时钟、成本现金恒等式、窗口对齐和价格分段复合恒等式这几项直接影响结论的检查。", "",
              "下一项值得回答的是：在这15个固定起点上，除了已经公布的因子水平，当时发生了什么新的机制变化，足以改变整项指数持仓的依据；若找不到可事先判定的触发和失效条件，就不能把本表拼成一个看起来达标的策略。现有窗口显示单起点归因不足；是否存在交易优势，还需要按事先写定、当时能执行的规则重放完整账户，不能仅由后续消息的存在作结论。", "",
              "## 文件", "",
              "- " + local("原固定口径", "protocol.json"),
              "- " + local("15个时点的联合状态", "event_states.json"),
              "- " + local("23个信息节点", "information_timeline.json"),
              "- " + local("原30条事件净收益及来源", "events_and_returns.json"),
              "- " + local("月度总量直接值与差分值", "monthly_headline_reconciliation.json"),
              "- " + local("每个持有期的新信息", "holding_window_new_information.json"),
              "- " + local("4月固定窗口价格分段", "april_credit_path.json"),
              "- " + local("计算核对", "calculation_checks.json"), "",
              f"![指数信息顺序](<{(OUT / '指数信息时序与交叠.png').as_posix()}>)", ""]
    (OUT / "历史发现_指数驱动的信息更新与重复暴露.md").write_text("\n".join(lines), encoding="utf-8")
    save("result.json", {"study_id": read(OUT / "protocol.json")["study_id"], "completed_at": now(),
                         "classification": "PROGRESS_INDEX_JOINT_INFORMATION_CLOCK_AND_OVERLAPPING_EXPOSURE",
                         "event_nodes": len(states), "information_nodes": len(read(OUT / "information_timeline.json")),
                         "return_rows": len(returns), "new_return_rows": sum(r["newly_computed"] for r in returns),
                         "twenty_day_windows_with_subsequent_information": n_updated,
                         "max_simultaneous_20d_windows": overlap[1]["max_simultaneous_windows"],
                         "new_credit_20d_net_returns": {m: ret("CREDIT_" + m, 20) for m in ["2019-04", "2019-05"]},
                         "april_credit_pre_may5_price_return": before_trade,
                         "historical_description_is_prediction_edge": False, "new_parameters_fitted": 0, "new_full_accounts": 0,
                         "net_sharpe": None, "goal_achieved": False, "orders_authorized": False,
                         "next_question": "同一历史信息时间线上，有无事先可判定的指数持仓依据变化和失效条件；禁止将已见赢家状态拼成买卖阈值。"})
    print("已生成历史发现报告，目标保持未达成。", flush=True)


def record_progress():
    result = read(OUT / "result.json")
    result_path = str((OUT / "result.json").relative_to(ROOT)).replace("\\", "/")
    report_path = str((OUT / "历史发现_指数驱动的信息更新与重复暴露.md").relative_to(ROOT)).replace("\\", "/")
    protocol_path = str((OUT / "protocol.json").relative_to(ROOT)).replace("\\", "/")
    path = ROOT / "config/510300_historical_cause_discovery_v1.json"
    current = read(path)
    current.update(latest_completed_study=result_path, latest_report=report_path, current_study=protocol_path,
                   updated_at=now(), goal_achieved=False)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    current = read(path)
    current.update(latest_historical_index_joint_information_clock=result_path,
                   latest_historical_report=report_path, latest_historical_diagnostic_at=now(),
                   next_research_question=result["next_question"], goal_achieved=False)
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    save("completion.json", {"recorded_at": now(), "classification": result["classification"],
                             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "goal_service_status": "ACTIVE_NOT_ACHIEVED", "orders_authorized": False})
    print("进度已登记；保留历史研究和指数整体范围。", flush=True)


def main():
    parser = argparse.ArgumentParser(description="指数历史信息顺序研究")
    parser.add_argument("action", choices=["fetch", "calculate", "timeline", "reconcile", "draw", "report", "record"])
    args = parser.parse_args()
    {"fetch": fetch, "calculate": calculate, "timeline": timeline, "reconcile": reconcile_headlines,
     "draw": draw, "report": report, "record": record_progress}[args.action]()


if __name__ == "__main__":
    main()
