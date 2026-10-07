"""五次MSCI纳入比例提升：固定日历窗口、事前联合状态及事后资金核对。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import pdfplumber
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_msci_inclusion_demand_v1"
STUDY = "510300_HISTORICAL_MSCI_INCLUSION_DEMAND_V1"
SOURCES = {
    "price_state": "reports/research/510300_multidim_financing_composition_v1/data_correction/daily_inputs_corrected_20240808.parquet",
    "funding": "reports/research/510300_factor96_funding_relief_v1/daily_features_lag1.parquet",
    "northbound": "data/raw/flow/northbound_legacy_net_flow_v1.parquet",
    "dividends": "data/reference/510300_dividends.csv",
}
EVENTS = [
    {"id": "201805", "review_date": "2018-05-14", "implementation": "2018-05-31", "iif_before": 0., "iif_after": .025,
     "weight_em_planned": .0039, "roadmap_date": "2017-06-20", "roadmap_id": "roadmap2017", "scope": "首次纳入大盘A股", "pdf": "MSCI_May18_SAIRPR.pdf", "date_text": "May 31, 2018", "factor_text": "2.5%"},
    {"id": "201808", "review_date": "2018-08-13", "implementation": "2018-08-31", "iif_before": .025, "iif_after": .05,
     "weight_em_planned": .0075, "roadmap_date": "2017-06-20", "roadmap_id": "roadmap2017", "scope": "大盘A股第二步", "pdf": "MSCI_Aug18_QIRPR.pdf", "date_text": "August 31, 2018", "factor_text": "2.5% to 5%"},
    {"id": "201905", "review_date": "2019-05-13", "implementation": "2019-05-28", "iif_before": .05, "iif_after": .10,
     "weight_em_planned": .0176, "roadmap_date": "2019-02-28", "roadmap_id": "roadmap2019", "scope": "大盘A股提升并纳入创业板大盘股", "pdf": "MSCI_May19_QIRPR.pdf", "date_text": "May 28, 2019", "factor_text": "0.05 to 0.10"},
    {"id": "201908", "review_date": "2019-08-07", "implementation": "2019-08-27", "iif_before": .10, "iif_after": .15,
     "weight_em_planned": .0246, "roadmap_date": "2019-02-28", "roadmap_id": "roadmap2019", "scope": "大盘A股继续提升", "pdf": "MSCI_Aug19_QIRPR.pdf", "date_text": "August 27, 2019", "factor_text": "0.10 to 0.15"},
    {"id": "201911", "review_date": "2019-11-07", "implementation": "2019-11-26", "iif_before": .15, "iif_after": .20,
     "weight_em_planned": .041, "roadmap_date": "2019-02-28", "roadmap_id": "roadmap2019", "scope": "大盘A股提升并纳入中盘股", "pdf": "MSCI_Nov19_QIRPR.pdf", "date_text": "November 26, 2019", "factor_text": "0.15 to 0.20"},
]
DOCUMENTS = [
    {"id": e["id"], "file": e["pdf"], "url": "https://app2.msci.com/eqb/pressreleases/archive/" + e["pdf"], "type": "pdf"}
    for e in EVENTS
] + [
    {"id": "roadmap2017", "file": "roadmap2017.html", "type": "html",
     "url": "https://app2.msci.com/webapp/index_ann/DocGet?format=html&lang=en&pub_key=qmTkbZqRZxA%3D"},
    {"id": "roadmap2019", "file": "roadmap2019.html", "type": "html",
     "url": "https://app2.msci.com/webapp/index_ann/DocGet?format=html&lang=en&pub_key=3wTKvvfPyTs%3D"},
]


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare():
    if (OUT / "protocol.json").exists():
        raise RuntimeError("本轮设定已保存，不重复覆盖。")
    correction = read(ROOT / "config/510300_financing_source_correction_20240808_v1.json")
    if sha(ROOT / SOURCES["price_state"]) != correction["corrected_daily_sha256"]:
        raise ValueError("日线状态不是已纠正的固定副本。")
    save("protocol.json", {
        "study_id": STUDY, "frozen_at": now(), "previous_turn_classification": "PROGRESS_LOAN_SURPRISE_JOINT_SCORE_AND_FOUR_ACCOUNTS",
        "question": "提前明确的A股纳入比例提升，是否在临近实施阶段提供510300方向收益；真实资金实现与指数回报是否相同？",
        "new_evidence": "五份MSCI当期审议公告及两份路线决定原文；用基准配置变化解释潜在需求来源，不用ETF份额变化代替被动净需求。",
        "universe": "2018年首次纳入两步及2019年提升至20%的三步，五次全部保留。共同来自两项路线决定，不当成五个独立政策冲击。",
        "events": EVENTS, "documents": DOCUMENTS,
        "mechanism": "基准A股权重提高，使严格复制且规模不变的基金目标A股持仓增加；但管理规模、实际持仓、提前建仓、衍生品替代、其他国家权重变化和主动交易仍会影响实际现货需求。",
        "scope_limit": "MSCI A股部分并非沪深300，未取得各期交集权重、被动规模及实际实施剩余量。纳入因子是自由流通调整市值的乘数，不是MSCI新兴市场指数内的A股权重，也不是未来实际资金流。",
        "information_clock": "每份日期级审议PDF保守视为其标注日期后第二个日历日北京时间00:00可用；路线HTML保留原文时刻。所有事前状态取入场前一交易日16:00。现有资金数据保持原滞后一交易日及已生效政策时钟。",
        "primary_window": "设实施收盘日为E，固定在E-4交易日开盘进入，在E+1交易日开盘退出，共五个开盘间隔。E-5收盘决策，审议原文必须先于决策。不是收盘竞价成交策略。",
        "diagnostic_window": "仅用于说明实施后延续或回吐：E+1开盘到E+6开盘，独立计费，不作方向反转或备选持有期。",
        "label_cost": {"quantity": 10000, "commission": .0004, "minimum": 5., "slippage": .001, "tick": .001, "dividend": "登记日处于入场至退出前，计入每份股息权益。标签不是完整现金到账账户。"},
        "joint_state": "E-5已知20日510300总回报（正/非正）与已知资金利率政策差的5交易日变化（收窄/不收窄）组成固定四格；同时展示资金差水平、订单、融资及波动。均保留连续值，缺失独立保留，不用事后收益分组。",
        "northbound": "旧披露口径日净买入只用于事后核对实施日及五日窗口实际全部北向净买入，既非事前特征也非MSCI被动流量或沪深300专项资金。沿用第三方历史副本，并非官方逐笔流量。",
        "score_policy": "本轮只确定新因子的机制和五个事件联合状态，不在五个同源事件上拟合加权评分、阈值或概率。一个已知配置方向不能直接赋予指数上涨高分。",
        "comparison_policy": "一组完整日历案例和四格描述；零阈值搜索、零新模型、零账户。主窗口无正成本后均值则封存该固定表达，不修改入场日或增加过滤救援。",
        "context": "本轮2018—2019是独立主题的有限阶段发现；不替代2024—2025主要账户结果，不与其拼接收益。",
        "legacy_boundaries": ["沪深300内部换样无法仅凭调入推出指数整体净需求", "跨ETF份额二元方向旧家族保持拒绝", "北向流量旧二元家族保持原判，不重跑或改阈值", "旧个股调入收益数据门不重开"],
        "sources": {k: {"path": v, "sha256": sha(ROOT / v)} for k, v in SOURCES.items()},
        "history_is_independent_validation": False, "goal_achieved": False, "current_view": "NO_VIEW", "orders_authorized": False,
    })
    print("已固定五次纳入比例提升、五日窗口及四格事前状态，未计算收益。", flush=True)


def acquire_one(item):
    # 官网二进制下载返回403时，只接纳已逐项核对的同一官网网页阅读事实。
    web_evidence = OUT / "source_web_evidence.json"
    if web_evidence.exists():
        evidence = next(d for d in read(web_evidence)["documents"] if d["id"] == item["id"])
        if not evidence["all_terms_verified"] or evidence["url"] != item["url"]:
            raise ValueError("官网阅读事实与固定来源不符。")
        return {**item, "local_path": str(web_evidence.relative_to(ROOT)).replace("\\", "/"),
                "retrieved_at": evidence["retrieved_utc"], "sha256": sha(web_evidence),
                "source_mode": "OFFICIAL_WEB_READER_FACTS_ONLY", "original_binary_obtained": False,
                "evidence": evidence["matched_terms"], "reader_reference": evidence["reader_reference"]}
    p = OUT / "raw" / item["file"]
    p.parent.mkdir(parents=True, exist_ok=True)
    retrieved = now()
    response = requests.get(item["url"], timeout=40)
    response.raise_for_status()
    if item["type"] == "pdf" and not response.content.startswith(b"%PDF"):
        raise ValueError("官方地址没有返回PDF：" + item["file"])
    if p.exists() and p.read_bytes() != response.content:
        raise ValueError("已保存来源变化，不覆盖：" + item["file"])
    if not p.exists():
        p.write_bytes(response.content)
    if item["type"] == "pdf":
        with pdfplumber.open(p) as pdf:
            pages = len(pdf.pages)
            content = pdf.pages[0].extract_text() or ""
    else:
        pages = None
        content = BeautifulSoup(response.content, "html.parser").get_text(" ", strip=True)
    flat = re.sub(r"\s+", " ", content)
    (OUT / "raw" / (item["id"] + "_text.txt")).write_text(content, encoding="utf-8")
    if item["id"].isdigit():
        e = next(x for x in EVENTS if x["id"] == item["id"])
        if e["date_text"] not in flat or e["factor_text"] not in flat:
            raise ValueError("日期或纳入比例原文未匹配：" + item["id"])
        evidence = {"implementation_date_match": e["date_text"], "inclusion_factor_match": e["factor_text"]}
    else:
        required = ["May 2018", "August 2018", "June 20, 2017"] if item["id"] == "roadmap2017" else ["February 28, 2019", "5% to 10%", "10% to 15%", "15% to 20%"]
        if not all(x in flat for x in required):
            raise ValueError("路线决定原文未匹配：" + item["id"])
        evidence = {"matched_terms": required}
    return {**item, "local_path": str(p.relative_to(ROOT)).replace("\\", "/"), "retrieved_at": retrieved,
            "http_status": response.status_code, "final_url": response.url, "sha256": sha(p), "bytes": p.stat().st_size,
            "pdf_pages": pages, "review_page": 1 if pages else None, "evidence": evidence}


def sources():
    read(OUT / "protocol.json")
    if (OUT / "source_result.json").exists():
        raise RuntimeError("本轮来源已完成，不重复采集。")
    if (OUT / "source_web_evidence.json").exists():
        save("收益计算前的来源访问说明.json", {
            "recorded_at": now(), "observed_direct_download_status": "HTTP_403",
            "observed_requests": [
                "https://app2.msci.com/eqb/pressreleases/archive/MSCI_May18_SAIRPR.pdf",
                "https://www.msci.com/eqb/pressreleases/archive/MSCI_May18_SAIRPR.pdf",
                "https://www.msci.com/downloads/documents/indexes/quarterly-index-reviews/historical/MSCI_Aug18_QIRPR.pdf"],
            "fallback": "网页阅读工具可以读取同一官方归档；仅保存日期、比例、来源地址及核对凭据，没有下载到七份原文副本。",
            "fixed_events_dates_windows_unchanged": True, "first_vintage_authenticated": False,
            "return_evaluations_before_change": 0,
        })
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(acquire_one, DOCUMENTS))
    save("source_result.json", {"completed_at": now(), "documents": receipts, "event_count": 5,
                                "official_documents": 7, "original_binaries_downloaded": sum(bool(r.get("original_binary_obtained", True)) for r in receipts),
                                "first_vintage_authenticated": False, "return_evaluations": 0})
    print("已核对五份审议公告和两份路线决定，日期及比例与官网阅读结果相符。", flush=True)


def label(market, div, first, last):
    a, b = market.iloc[first], market.iloc[last]
    buy = np.ceil(float(a.open) * 1.001 * 1000 - 1e-9) / 1000
    sell = np.floor(float(b.open) * .999 * 1000 + 1e-9) / 1000
    entitlement = float(div.loc[div.record_date.ge(a.date) & div.record_date.lt(b.date), "cash_dividend_per_share"].sum())
    commission_in = max(5., 10000 * buy * .0004)
    commission_out = max(5., 10000 * sell * .0004)
    cash_in = 10000 * buy + commission_in
    cash_out = 10000 * (sell + entitlement) - commission_out
    gross_pnl = 10000 * (float(b.open) - float(a.open) + entitlement)
    net_pnl = cash_out - cash_in
    return {"entry_date": a.date.strftime("%Y-%m-%d"), "exit_date": b.date.strftime("%Y-%m-%d"), "entry_open": a.open,
            "exit_open": b.open, "buy_with_slippage": buy, "sell_with_slippage": sell,
            "dividend_entitlement_per_share": entitlement, "commission_in": commission_in, "commission_out": commission_out,
            "gross_return": (float(b.open) + entitlement) / float(a.open) - 1, "net_return": cash_out / cash_in - 1,
            "net_profit_cny": net_pnl, "gross_profit_cny": gross_pnl, "friction_cny": gross_pnl - net_pnl,
            "cash_in": cash_in, "cash_out": cash_out, "open_intervals": last - first}


def summarize(frame, key):
    values = frame[key].to_numpy(float)
    return {"n": len(values), "mean": np.mean(values) if len(values) else None, "median": np.median(values) if len(values) else None,
            "positive": int((values > 0).sum()), "win_rate": float((values > 0).mean()) if len(values) else None,
            "minimum": values.min() if len(values) else None, "maximum": values.max() if len(values) else None}


def run():
    if (OUT / "result.json").exists():
        raise RuntimeError("五次事件已计算，不重选窗口。")
    protocol = read(OUT / "protocol.json")
    source = read(OUT / "source_result.json")
    for item in list(protocol["sources"].values()) + [{"path": d["local_path"], "sha256": d["sha256"]} for d in source["documents"]]:
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise ValueError("固定输入已变化：" + item["path"])
    price_columns = ["date", "open", "close", "wealth", "订单水平", "订单月度变化", "融资五日净变化", "融资买入活跃度", "短长波动比", "成交活跃度"]
    market = pd.read_parquet(ROOT / SOURCES["price_state"], columns=price_columns).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).dt.tz_localize(None)
    funding = pd.read_parquet(ROOT / SOURCES["funding"])
    funding["date"] = pd.to_datetime(funding.date).dt.tz_localize(None)
    funding = funding.set_index("date")
    north = pd.read_parquet(ROOT / SOURCES["northbound"], columns=["date", "north_net_buy_100m_cny", "usable_as_net_flow", "northbound_semantics"])
    north["date"] = pd.to_datetime(north.date).dt.tz_localize(None)
    north = north.set_index("date")
    div = pd.read_csv(ROOT / SOURCES["dividends"])
    div["record_date"] = pd.to_datetime(div.record_date)
    rows, windows, flow_rows, state_rows = [], [], [], []
    for event in EVENTS:
        matches = market.index[market.date.eq(pd.Timestamp(event["implementation"]))]
        if len(matches) != 1:
            raise ValueError("实施日不是唯一交易日。")
        e = int(matches[0])
        decision, entry, exit_day = e - 5, e - 4, e + 1
        state = market.iloc[decision]
        decision_at = state.date.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=16)
        conservative_available = pd.Timestamp(event["review_date"], tz="Asia/Shanghai") + pd.Timedelta(days=2)
        if conservative_available > decision_at:
            raise ValueError("审议公告未先于决策。")
        f = funding.loc[state.date]
        f_before = funding.loc[market.iloc[decision - 5].date]
        for frow in [f, f_before]:
            if not bool(frow.fund_known) or pd.Timestamp(frow.available_at) > decision_at:
                raise ValueError("资金状态在当时未知。")
            if pd.Timestamp(frow.policy_known_at) > decision_at:
                raise ValueError("政策利率取自后续公告。")
        momentum = float(state.wealth / market.iloc[decision - 20].wealth - 1)
        gap_change = float(f.gap_pp - f_before.gap_pp)
        trend_group = "此前20日上涨" if momentum > 0 else "此前20日非上涨"
        funding_group = "资金差收窄" if gap_change < 0 else "资金差未收窄"
        row = {**event, "review_available_at_upper": conservative_available, "decision_at": decision_at,
               "review_lead_days": (state.date - pd.Timestamp(event["review_date"])).days,
               "roadmap_lead_days": (state.date - pd.Timestamp(event["roadmap_date"])).days,
               "pre20_total_return": momentum, "funding_gap_pp": float(f.gap_pp), "funding_gap_change5_pp": gap_change,
               "dr007": float(f.dr007), "policy_rate": float(f.rate), "fund_stat_date": f.fund_stat_date, "funding_available_at": f.available_at,
               "trend_group": trend_group, "funding_group": funding_group, "joint_state": trend_group + " / " + funding_group,
               "orders_index": float(state["订单水平"] + 50), "orders_change_pp": float(state["订单月度变化"]),
               "margin_change5": float(state["融资五日净变化"]), "margin_activity": float(state["融资买入活跃度"]),
               "log_vol5_vol60": float(state["短长波动比"]), "log_volume_activity": float(state["成交活跃度"]),
               "remaining_passive_demand_cny": None, "nonlinear_score": None, "score_status": "NOT_COMPUTED_FIVE_SAME_PROGRAM_EVENTS_NO_FIT"}
        primary = label(market, div, entry, exit_day)
        diagnostic = label(market, div, exit_day, e + 6)
        row.update({"primary_" + k: v for k, v in primary.items()})
        row.update({"post_" + k: v for k, v in diagnostic.items()})
        gross_to_close = float(market.iloc[e].close / market.iloc[entry].open - 1)
        overnight = float(market.iloc[exit_day].open / market.iloc[e].close - 1)
        row.update(implementation_close=market.iloc[e].close, implementation_day_total_return=float(market.iloc[e].wealth / market.iloc[e-1].wealth - 1),
                   entry_to_implementation_close_price_return=gross_to_close, implementation_to_exit_open_return=overnight)
        if abs(primary["dividend_entitlement_per_share"]) < 1e-12 and abs((1 + gross_to_close) * (1 + overnight) - 1 - primary["gross_return"]) > 1e-12:
            raise ValueError("日线收盘与次开盘分解不一致。")
        dates = market.iloc[entry:e + 1].date.to_list()
        flow = north.reindex(dates)
        admitted = flow.usable_as_net_flow.eq(True) & flow.northbound_semantics.eq("LEGACY_DAILY_NET_BUY")
        flow_values = flow.north_net_buy_100m_cny.where(admitted)
        row["realized_northbound_window_rows"] = int(flow_values.notna().sum())
        row["realized_northbound_5d_yi"] = float(flow_values.sum(min_count=5))
        row["realized_northbound_implementation_yi"] = float(flow_values.iloc[-1])
        for dt, value in flow_values.items():
            flow_rows.append({"id": event["id"], "date": dt, "northbound_net_yi": value, "role": "事后全部北向资金核对，不作入场特征"})
        for window_name, first, last in [("实施前五日", entry, exit_day), ("实施后五日说明", exit_day, e+6)]:
            for i in range(first, last + 1):
                day = market.iloc[i]
                windows.append({"id": event["id"], "window": window_name, "date": day.date, "offset_from_entry": i-first, "open": day.open, "close": day.close,
                                "role": "退出日开盘后不再持有" if i == last else "持有期日线"})
        for name, frow in [("决策日", f), ("五个交易日前", f_before)]:
            state_rows.append({"id": event["id"], "state_role": name, **frow.to_dict()})
        rows.append(row)
    events = pd.DataFrame(rows)
    events.to_csv(OUT / "五次纳入的事前状态与历史结果.csv", index=False, encoding="utf-8-sig")
    events.to_parquet(OUT / "五次纳入的事前状态与历史结果.parquet", index=False)
    pd.DataFrame(windows).to_csv(OUT / "十个固定窗口日线.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(flow_rows).to_csv(OUT / "实施窗口全部北向净买入.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(state_rows).to_csv(OUT / "事前资金状态取值.csv", index=False, encoding="utf-8-sig")
    cells = []
    for trend in ["此前20日上涨", "此前20日非上涨"]:
        for fund in ["资金差收窄", "资金差未收窄"]:
            subset = events[events.trend_group.eq(trend) & events.funding_group.eq(fund)]
            cells.append({"trend": trend, "funding": fund, "ids": subset.id.to_list(), **summarize(subset, "primary_net_return")})
    save("四格联合状态.json", cells)
    primary = summarize(events, "primary_net_return")
    result = {
        "study_id": STUDY, "completed_at": now(), "status": "COMPLETED_FIVE_INCLUSION_EVENTS_JOINT_STATE_DISCOVERY",
        "candidate_status": "REJECTED_FIXED_FIVE_DAY_EXPRESSION_NO_PARAMETER_RESCUE" if primary["mean"] <= 0 else "POSITIVE_SMALL_EVENT_DESCRIPTION_NOT_QUALIFIED_STRATEGY",
        "official_documents": 7, "event_count": 5, "roadmap_programs": 2,
        "primary_net": primary, "primary_gross": summarize(events, "primary_gross_return"), "post_net_diagnostic": summarize(events, "post_net_return"),
        "implementation_northbound_positive": int(events.realized_northbound_implementation_yi.gt(0).sum()),
        "primary_window_northbound_positive": int(events.realized_northbound_5d_yi.gt(0).sum()),
        "positive_implementation_flow_with_negative_primary_return": int((events.realized_northbound_implementation_yi.gt(0) & events.primary_net_return.lt(0)).sum()),
        "positive_implementation_flow_with_negative_same_day_total_return": int((events.realized_northbound_implementation_yi.gt(0) & events.implementation_day_total_return.lt(0)).sum()),
        "missing_northbound_observations": int(25-events.realized_northbound_window_rows.sum()),
        "all_review_available_before_decision": True, "joint_cells": cells,
        "new_model_fits": 0, "new_full_accounts": 0, "new_parameter_searches": 0, "new_candidates_admitted": 0,
        "estimated_passive_flow_cny": "NOT_COMPUTED_MISSING_PASSIVE_AUM_ACTUAL_POSITIONS_AND_INDEX_OVERLAP",
        "net_sharpe": "NOT_COMPUTED_EVENT_LABELS_NOT_FULL_ACCOUNT", "goal_achieved": False,
        "source_limitations": ["官网归档可查，不证明当年实时接收回执或完整初始版本", "北向为第三方旧净买入历史副本，不能分离被动和主动，也不限定沪深300", "不能证明其余参与者的反事实交易或期望已完全计价", "五次实施仅对应两项路线，不是五个独立政策冲击"],
    }
    save("result.json", result)
    print(json.dumps(clean({"事件数": 5, "主窗口": primary, "状态": result["candidate_status"], "新拟合": 0, "新账户": 0}), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "sources", "run"])
    args = parser.parse_args()
    globals()[args.stage]()
