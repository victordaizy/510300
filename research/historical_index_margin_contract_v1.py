"""历史融资保证金合同变化：原因、适用存量、实际融资活动及指数回报。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_index_margin_contract_v1"
STUDY = "510300_HISTORICAL_INDEX_MARGIN_CONTRACT_V1"
FILES = {
    "market": "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet",
    "margin": "reports/research/510300_factor96_crowding_overlay_v1_0_1/inputs/margin.parquet",
    "dividends": "data/reference/510300_dividends.csv",
}
SOURCES = [
    {"id": "rule_2015", "url": "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20151113_4012108.shtml", "publication_date": "2015-11-13", "required": ["100%", "2015年11月23日", "展期"]},
    {"id": "reason_2015", "url": "https://www.csrc.gov.cn/csrc/c100029/c1000218/content.shtml", "publication_date": "2015-11-13", "required": ["1.14万亿元", "9041亿元", "追加保证金", "预先缴款"]},
    {"id": "rule_2023", "url": "https://www.sse.com.cn/lawandrules/sselawsrules2025/repeal/rules/c/c_20250616_10805180.shtml", "publication_date": "2023-08-27", "required": ["80%", "收市后", "可用余额", "尚未了结"]},
    {"id": "reason_2023", "url": "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20230827_5725661.shtml", "publication_date": "2023-08-27", "required": ["125万元", "约定", "存", "80%"]},
    {"id": "rule_2026", "url": "https://www.sse.com.cn/lawandrules/sselawsrules2025/trade/specific/margin/c/c_20260114_10805174.shtml", "publication_date": "2026-01-14", "required": ["100%", "2026年1月19日", "展期"], "local": "reports/research/510300_participant_identity_clock_v1/raw/margin_rule_20260114.html"},
    {"id": "reason_2026", "url": "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260114_10805178.shtml", "publication_date": "2026-01-14", "required": ["80%", "100%", "相对充裕", "新开"], "local": "reports/research/510300_participant_identity_clock_v1/raw/margin_reason_20260114.html"},
]
EVENTS = [
    {"event_id": "R2015", "label": "2015提高", "announcement": "2015-11-13", "effective": "2015-11-23", "effective_first_session": "2015-11-23", "old_margin": .5, "new_margin": 1., "existing_contracts": "存量及展期沿用原条件，无本次规则强制追加或平仓要求", "reason": "融资交易及规模快速回升，逆周期控制新增杠杆", "source_ids": ["rule_2015", "reason_2015"]},
    {"event_id": "R2023", "label": "2023降低", "announcement": "2023-08-27", "effective": "2023-09-08收市后", "effective_first_session": "2023-09-11", "old_margin": 1., "new_margin": .8, "existing_contracts": "券商可依约下调存量保证金要求，非自动或全体统一兑现", "reason": "促进融资业务功能并支持合理交易需求", "source_ids": ["rule_2023", "reason_2023"]},
    {"event_id": "R2026", "label": "2026提高", "announcement": "2026-01-14", "effective": "2026-01-19", "effective_first_session": "2026-01-19", "old_margin": .8, "new_margin": 1., "existing_contracts": "存量及展期沿用原条件，无本次规则统一强制降仓要求", "reason": "融资交易活跃、流动性相对充裕，逆周期降低新增杠杆", "source_ids": ["rule_2026", "reason_2026"]},
]


def now() -> str:
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


def save(name: str, value) -> None:
    (OUT / name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        raise RuntimeError("本研究已登记，不能覆盖原设定。")
    (OUT / "sources").mkdir(exist_ok=True)
    protocol = {
        "study_id": STUDY, "recorded_at": now(), "previous_goal_turn_classification": "PROGRESS_FROZEN_INDEX_STATE_INCREMENT_RESOLVED",
        "question": "融资保证金调整为何发生，改变新增还是存量合同，实际融资活动和指数价格是否按同一方向变化？",
        "research_mode": "HISTORICAL_ONLY", "events": EVENTS, "sources": SOURCES,
        "sample_selection": "由原2026年案例追溯同一最低融资保证金比例的2015提高、2023降低、2026提高三个已确认调整。公告与实施各一锚点，共三条政策链，不声称穷尽所有融资监管措施。",
        "novelty": "不是重做融资收缩/偿还量过滤器；比较公告与预先已知的生效日、新旧合同适用范围、机械融资容量与实际买入/偿还的区别。",
        "duplicate_evidence": ["research/factor96_margin_repair_v1.py", "reports/research/510300_historical_collateral_relief_cases_v1/result.json", "reports/research/510300_participant_identity_clock_v1/研究结论.md", "reports/research/510300_daily_supply_test_v1/研究结论.md"],
        "return_windows_sessions": [5, 20], "primary_horizon": 5,
        "entry_clock": "公告仅有日期，按日期结束后的下一交易日开盘；实施日早已公告，按首个适用新规则的交易日开盘。2023年9月8日收市后生效，首个适用交易日为9月11日。",
        "entry_quantity": 10000,
        "entry_quantity_rule": "所有事件固定10000份，数量不按实现开盘价决定；没有行情过滤器。只是事件收益测量，不是完整账户、融资交易或实际成交证明。",
        "exit_clock": "入场后第5或第20交易日开盘，保持原固定期限，不优化。",
        "costs": {"commission_each_side": .0004, "slippage_each_side": .001, "minimum_commission_cny": 5., "tick": .001},
        "dividends": "入场日及以后至退出前的权益登记日产生应收；退出时未到账单列应收，不当成已经可用现金，不重投资。",
        "financing_windows": "分别对每个锚点取前5日与从锚点开始5日；需要前一期余额以计算余额差。后5日只解释已发生传导，不用作同次入场信号。",
        "financing_scope": "统一比较沪市汇总；2015/2023复用已验证文件的沪市列，2026定向请求1月官方汇总。沪市不等于两市或沪深300成分融资。",
        "implied_repayment": "融资买入额减融资余额变化；包括直接还款、卖券还款、强平及权益调整，不能识别纯强平。",
        "missing_data": "若2026官方汇总不可得则保留缺失，不用新闻估数填日线。最多一次初始请求和一次相同参数技术重试。",
        "source_snapshot_semantics": "今日回取的历史官方档案；保存原始文件及来源。公告日期不冒充精确首次分钟，统计日不冒充历史公开日。",
        "independent_validation": False, "historical_prices_previously_seen": True,
        "new_strategy_accounts": 0, "parameter_grids": 0, "no_parameter_rescue": True,
        "causal_effect_identified": False, "orders_authorized": False, "goal_achieved": False,
        "files": {key: {"path": value, "sha256": digest(ROOT / value)} for key, value in FILES.items()},
        "stop": "完成三条政策链的原因、合同范围、融资量与固定回报比较；仅靠同向不新增账户，不把六锚点当六个独立事件。",
    }
    save("protocol.json", protocol)
    print("三条历史政策链已登记，公告与实施分列；尚未计算本研究收益。")


def collect_source(spec: dict) -> dict:
    path = OUT / "sources" / (spec["id"] + ".html")
    if path.exists():
        raw, origin = path.read_bytes(), "已保存副本"
    elif spec.get("local") and (ROOT / spec["local"]).exists():
        raw, origin = (ROOT / spec["local"]).read_bytes(), spec["local"]
        path.write_bytes(raw)
    else:
        response = requests.get(spec["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=(10, 35))
        response.raise_for_status()
        raw, origin = response.content, "本次公开历史源请求"
        path.write_bytes(raw)
    html = raw.decode("utf-8", errors="replace")
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    missing = [word for word in spec["required"] if word not in text]
    if missing:
        raise ValueError(f"{spec['id']}正文缺少核对词：{missing}")
    path.with_suffix(".txt").write_text(text, encoding="utf-8")
    return dict(spec, fetched_at=now(), origin=origin, local_path=str(path.relative_to(ROOT)), sha256=digest(path), bytes=len(raw), status="RETRIEVED_AND_CONTENT_MATCHED")


def collect() -> None:
    if not (OUT / "protocol.json").exists():
        raise RuntimeError("应先登记研究范围。")
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts = list(pool.map(collect_source, SOURCES))
    save("source_receipts.json", receipts)
    raw_path = OUT / "sources/sse_margin_202601.json"
    url = "https://query.sse.com.cn/marketdata/tradedata/queryMargin.do"
    params = {"isPagination": "true", "beginDate": "20260101", "endDate": "20260131", "tabType": "", "stockCode": "", "pageHelp.pageSize": "5000", "pageHelp.pageNo": "1", "pageHelp.beginPage": "1", "pageHelp.cacheSize": "1", "pageHelp.endPage": "5"}
    if not raw_path.exists():
        try:
            response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.sse.com.cn/"}, timeout=(10, 35))
            response.raise_for_status()
            response.json()
            raw_path.write_bytes(response.content)
        except Exception as exc:
            save("sse_margin_source_receipt.json", {"status": "MISSING_AFTER_REQUEST", "url": url, "params": params, "error": str(exc), "recorded_at": now()})
            print("规则原文已保存；2026沪市汇总暂缺，不回填。")
            return
    body = json.loads(raw_path.read_text(encoding="utf-8-sig"))
    save("sse_margin_source_receipt.json", {"status": "RETRIEVED_RAW", "url": url, "params": params, "recorded_at": now(), "sha256": digest(raw_path), "rows": len(body.get("result", [])), "sample_keys": list(body.get("result", [{}])[0]) if body.get("result") else []})
    print("历史规则正文已保存：", len(receipts), "份；2026沪市汇总行数：", len(body.get("result", [])))
    if body.get("result"):
        print("沪市汇总首行字段：", json.dumps(body["result"][0], ensure_ascii=False))


def event_return(market: pd.DataFrame, dividends: pd.DataFrame, entry: int, horizon: int, protocol: dict) -> dict:
    first, last = market.iloc[entry], market.iloc[entry + horizon]
    d = protocol["costs"]
    tick, slip = Decimal(str(d["tick"])), Decimal(str(d["slippage_each_side"]))
    price_buy = (Decimal(str(first.open)) * (1 + slip) / tick).to_integral_value(rounding=ROUND_CEILING) * tick
    price_sell = (Decimal(str(last.open)) * (1 - slip) / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
    q = Decimal(protocol["entry_quantity"])
    buy_fee = max(Decimal(str(d["minimum_commission_cny"])), q * price_buy * Decimal(str(d["commission_each_side"])))
    sell_fee = max(Decimal(str(d["minimum_commission_cny"])), q * price_sell * Decimal(str(d["commission_each_side"])))
    eligible = dividends[dividends.record_date.ge(first.date) & dividends.record_date.lt(last.date)]
    entitlement = float(eligible.cash_dividend_per_share.sum())
    paid_div = float(eligible.loc[eligible.payment_date.lt(last.date), "cash_dividend_per_share"].sum())
    money_in = q * price_buy + buy_fee
    money_out = q * price_sell - sell_fee + q * Decimal(str(entitlement))
    raw_gross = (float(last.open) + entitlement) / float(first.open) - 1
    raw_price = float(last.open) / float(first.open) - 1
    return {"horizon": horizon, "entry_date": first.date.strftime("%Y-%m-%d"), "exit_date": last.date.strftime("%Y-%m-%d"), "quantity": int(q), "entry_open": float(first.open), "exit_open": float(last.open), "entry_fill_assumption": float(price_buy), "exit_fill_assumption": float(price_sell), "entry_fee_cny": float(buy_fee), "exit_fee_cny": float(sell_fee), "entry_cost_cny": float(money_in), "dividend_per_share": entitlement, "dividend_cash_received_before_exit_cny": paid_div * int(q), "dividend_receivable_at_exit_open_cny": (entitlement - paid_div) * int(q), "raw_price_return": raw_price, "gross_total_return": raw_gross, "net_event_return": float(money_out / money_in - 1), "net_pnl_cny": float(money_out - money_in)}


def analyse() -> None:
    if (OUT / "result.json").exists():
        raise RuntimeError("本研究已完成，不更改事件、期限或重测。")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    for record in protocol["files"].values():
        if digest(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("原输入在登记后变化。")
    market = pd.read_parquet(ROOT / FILES["market"]).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date)
    dividends = pd.read_csv(ROOT / FILES["dividends"])
    for column in ["record_date", "ex_date", "payment_date"]:
        dividends[column] = pd.to_datetime(dividends[column])
    source = pd.read_parquet(ROOT / FILES["margin"])
    financing = source[["date", "rzye_sse", "rzmre_sse"]].rename(columns={"rzye_sse": "balance_cny", "rzmre_sse": "buy_cny"})
    financing["scope"] = "SSE"
    financing["source"] = FILES["margin"]
    raw_path = OUT / "sources/sse_margin_202601.json"
    if raw_path.exists():
        raw = pd.DataFrame(json.loads(raw_path.read_text(encoding="utf-8-sig"))["result"])
        extra = raw[["opDate", "rzye", "rzmre"]].rename(columns={"opDate": "date", "rzye": "balance_cny", "rzmre": "buy_cny"})
        extra["date"] = pd.to_datetime(extra.date.astype(str), format="%Y%m%d")
        extra[["balance_cny", "buy_cny"]] = extra[["balance_cny", "buy_cny"]].apply(pd.to_numeric)
        extra["reported_repayment_cny"] = pd.to_numeric(raw["rzche"])
        extra["scope"], extra["source"] = "SSE", str(raw_path.relative_to(ROOT))
        financing = pd.concat([financing, extra], ignore_index=True)
    financing["date"] = pd.to_datetime(financing.date)
    financing = financing.sort_values("date").reset_index(drop=True)
    if financing.date.duplicated().any():
        raise ValueError("融资日期重复。")
    financing["balance_change_cny"] = financing.balance_cny.diff()
    financing["implied_repayment_cny"] = financing.buy_cny - financing.balance_change_cny
    financing["repayment_identity_difference_cny"] = financing.implied_repayment_cny - financing.reported_repayment_cny
    positions = market.set_index("date").assign(market_idx=market.index)["market_idx"]
    financing["market_idx"] = financing.date.map(positions)
    nonconsecutive = financing.market_idx.diff().ne(1)
    financing.loc[nonconsecutive, ["balance_change_cny", "implied_repayment_cny"]] = np.nan
    return_rows, observation_rows, flow_rows, phase_paths, facts = [], [], [], [], []
    for event in EVENTS:
        announce_idx = int(market.date.searchsorted(pd.Timestamp(event["announcement"]), side="right"))
        effective_idx = int(market.date.searchsorted(pd.Timestamp(event["effective_first_session"]), side="left"))
        if market.date.iloc[effective_idx].strftime("%Y-%m-%d") != event["effective_first_session"]:
            raise ValueError("固定生效交易日不在日历。")
        facts.append(dict(event, financing_capacity_per_100_margin_before=100 / event["old_margin"], financing_capacity_per_100_margin_after=100 / event["new_margin"], relative_capacity_change=event["old_margin"] / event["new_margin"] - 1))
        for node, idx in [("ANNOUNCEMENT", announce_idx), ("EFFECTIVE", effective_idx)]:
            previous_close = float(market.close.iloc[idx - 1])
            ex_dividend = float(dividends.loc[dividends.ex_date.eq(market.date.iloc[idx]), "cash_dividend_per_share"].sum())
            for horizon in protocol["return_windows_sessions"]:
                return_rows.append(dict(event_id=event["event_id"], label=event["label"], node=node, previous_close=previous_close,
                                        entry_gap_raw=float(market.open.iloc[idx]) / previous_close - 1,
                                        entry_gap_with_ex_dividend=(float(market.open.iloc[idx]) + ex_dividend) / previous_close - 1,
                                        **event_return(market, dividends, idx, horizon, protocol)))
            for window, lo, hi in [("BEFORE5", idx - 5, idx), ("AFTER5", idx, idx + 5)]:
                dates = market.date.iloc[lo:hi]
                block = financing.set_index("date").reindex(dates).reset_index()
                cols = ["balance_cny", "buy_cny", "balance_change_cny", "implied_repayment_cny"]
                complete = len(block) == 5 and block[cols].notna().all().all()
                flow_rows.append({"event_id": event["event_id"], "label": event["label"], "node": node, "window": window, "start": dates.iloc[0].strftime("%Y-%m-%d"), "end": dates.iloc[-1].strftime("%Y-%m-%d"), "complete": complete, "scope": "SSE", "buy_daily_cny": float(block.buy_cny.mean()) if complete else None, "repayment_daily_cny": float(block.implied_repayment_cny.mean()) if complete else None, "net_five_day_cny": float(block.balance_change_cny.sum()) if complete else None, "last_balance_cny": float(block.balance_cny.iloc[-1]) if complete else None})
                observation_rows.extend(block.assign(event_id=event["event_id"], node=node, window=window).to_dict("records"))
        base = announce_idx - 1
        for idx in range(max(0, announce_idx - 5), min(len(market), effective_idx + 21)):
            row = market.iloc[idx]
            cash_div = float(dividends.loc[dividends.record_date.ge(market.date.iloc[base]) & dividends.ex_date.le(row.date) & dividends.ex_date.gt(market.date.iloc[base]), "cash_dividend_per_share"].sum()) if idx >= base else 0.0
            gross = (float(row.close) + cash_div) / float(market.close.iloc[base]) - 1
            phase_paths.append({"event_id": event["event_id"], "date": row.date, "sessions_from_first_announcement_open": idx - announce_idx, "first_effective_offset": effective_idx - announce_idx, "close_total_return_from_pre_announcement_close": gross, "raw_close": float(row.close)})
    returns = pd.DataFrame(return_rows)
    flows = pd.DataFrame(flow_rows)
    for frame, name in [(returns, "固定事件收益.csv"), (flows, "融资前后五日.csv"), (pd.DataFrame(observation_rows), "融资逐日观察.csv"), (pd.DataFrame(facts), "合同与机械融资容量.csv"), (pd.DataFrame(phase_paths), "固定事件价格路径.csv")]:
        frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    save("result.json", {"study_id": STUDY, "completed_at": now(), "status": "HISTORICAL_CONTRACT_SCOPE_AND_FINANCING_TRANSMISSION_MEASURED", "classification": "PROGRESS_HISTORICAL_INDEX_MARGIN_CONTRACT_AND_CAPACITY", "policy_chains": 3, "announcement_and_effective_nodes": 6, "fixed_event_windows": len(returns), "events_are_independent": False, "official_rule_documents": 6, "complete_five_day_financing_windows": int(flows.complete.sum()), "financing_scope": "SSE_ONLY", "new_fitted_models": 0, "new_full_accounts": 0, "parameter_grids": 0, "net_sharpe": None, "goal_achieved": False, "orders_authorized": False, "new_prospective_forecasts_enabled": False, "return_rows": returns.to_dict("records"), "mechanical_capacity_is_realized_cash_flow": False, "causal_effect_identified": False, "report_status": "NUMBERS_READY_INTERPRETATION_PENDING", "report": str((OUT / "历史发现_融资能力与存量合同.md").relative_to(ROOT)).replace("\\", "/")})
    print(returns[["label", "node", "horizon", "entry_date", "exit_date", "net_event_return", "dividend_per_share"]].to_string(index=False))
    printable = flows.copy()
    for col in ["buy_daily_cny", "repayment_daily_cny", "net_five_day_cny"]:
        printable[col] /= 1e8
    print(printable[["label", "node", "window", "complete", "buy_daily_cny", "repayment_daily_cny", "net_five_day_cny"]].to_string(index=False))


def finish() -> None:
    """读取固定收益结果，保存原因解释和必要复算，不追加策略或事件。"""
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    returns = pd.read_csv(OUT / "固定事件收益.csv")
    flows = pd.read_csv(OUT / "融资前后五日.csv")
    daily = pd.read_csv(OUT / "融资逐日观察.csv")
    paths = pd.read_csv(OUT / "固定事件价格路径.csv")
    capacity = pd.read_csv(OUT / "合同与机械融资容量.csv")
    q = returns.quantity
    paid = q * returns.entry_fill_assumption + returns.entry_fee_cny
    final = q * returns.exit_fill_assumption - returns.exit_fee_cny + q * returns.dividend_per_share
    if not np.allclose(final / paid - 1, returns.net_event_return, atol=1e-12, rtol=0):
        raise ValueError("事件收益的保存金额复算不一致。")
    if len(returns) != 12 or len(flows) != 12 or not flows.complete.all():
        raise ValueError("固定窗口数或融资覆盖不完整。")
    if not (pd.to_datetime(returns.exit_date) > pd.to_datetime(returns.entry_date)).all():
        raise ValueError("事件交易时序不成立。")
    if not np.allclose(returns.dividend_cash_received_before_exit_cny + returns.dividend_receivable_at_exit_open_cny, q * returns.dividend_per_share, atol=1e-8, rtol=0):
        raise ValueError("股息现金与应收合计不一致。")

    decompositions = []
    for (event_id, node), block in flows.groupby(["event_id", "node"], sort=False):
        b = block.set_index("window")
        decompositions.append({"event_id": event_id, "node": node,
                               "buy_relative_change": b.loc["AFTER5", "buy_daily_cny"] / b.loc["BEFORE5", "buy_daily_cny"] - 1,
                               "implied_repayment_relative_change": b.loc["AFTER5", "repayment_daily_cny"] / b.loc["BEFORE5", "repayment_daily_cny"] - 1,
                               "five_day_balance_change_before_cny": b.loc["BEFORE5", "net_five_day_cny"],
                               "five_day_balance_change_after_cny": b.loc["AFTER5", "net_five_day_cny"]})
    pd.DataFrame(decompositions).to_csv(OUT / "固定融资量分解.csv", index=False, encoding="utf-8-sig")
    observed = daily[daily.event_id.eq("R2026")].drop_duplicates("date")
    residual_row = observed.loc[observed.repayment_identity_difference_cny.abs().idxmax()]
    reported = []
    for (node, window), block in daily[daily.event_id.eq("R2026")].groupby(["node", "window"]):
        reported.append({"node": node, "window": window, "mean_reported_repayment_cny": float(block.reported_repayment_cny.mean()), "sum_identity_residual_cny": float(block.repayment_identity_difference_cny.sum())})
    save("necessary_checks.json", {"completed_at": now(), "event_amount_recomputation": "PASS", "fixed_windows": 12,
                                  "financing_five_day_coverage": "12_OF_12", "dividend_cash_and_receivable": "PASS",
                                  "sse_reported_vs_implied_repayment": "NONZERO_RESIDUAL_RETAINED",
                                  "max_daily_residual_date": residual_row.date,
                                  "max_daily_residual_cny": float(residual_row.repayment_identity_difference_cny),
                                  "residual_cause": "NOT_IDENTIFIED；官方定义已经把权益调整纳入偿还，不把这个额外差额擅自命名为权益调整或强平。",
                                  "reported_repayment_windows": reported,
                                  "market_fill_proof": "NOT_ESTABLISHED_BY_EVENT_LABELS"})

    font_file = Path("C:/Windows/Fonts/msyh.ttc")
    if font_file.exists():
        fontManager.addfont(str(font_file))
        plt.rcParams["font.family"] = FontProperties(fname=str(font_file)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 7.7), gridspec_kw={"height_ratios": [1.35, 1]})
    for j, event in enumerate(EVENTS):
        eid = event["event_id"]
        p = paths[paths.event_id.eq(eid)]
        a = returns[(returns.event_id == eid) & (returns.node == "ANNOUNCEMENT") & (returns.horizon == 5)].iloc[0]
        e = returns[(returns.event_id == eid) & (returns.node == "EFFECTIVE") & (returns.horizon == 5)].iloc[0]
        x = p.sessions_from_first_announcement_open
        y = p.close_total_return_from_pre_announcement_close * 100
        axes[0, j].plot(x, y, color="#275d83", linewidth=1.8)
        axes[0, j].scatter([0], [a.entry_gap_with_ex_dividend * 100], marker="D", color="#c36f28", s=35, label="公告后首次开盘")
        axes[0, j].axhline(0, color="#999999", linewidth=.8)
        axes[0, j].axvline(0, color="#999999", linestyle=":", linewidth=.9)
        axes[0, j].axvline(int(p.first_effective_offset.iloc[0]), color="#a83d35", linestyle="--", linewidth=1.0, label="新规则首个交易日")
        axes[0, j].set(title=f"{event['label']}：{event['old_margin']:.0%} → {event['new_margin']:.0%}", xlabel="相对公告后首个交易日", ylabel="相对公告前收盘的总回报（%）")
        axes[0, j].legend(frameon=False, fontsize=8)
        axes[0, j].grid(axis="y", alpha=.18)
        b = flows[(flows.event_id == eid) & (flows.node == "EFFECTIVE")].set_index("window")
        values_buy = [b.loc[w, "buy_daily_cny"] / 1e8 for w in ["BEFORE5", "AFTER5"]]
        values_repay = [b.loc[w, "repayment_daily_cny"] / 1e8 for w in ["BEFORE5", "AFTER5"]]
        positions = np.arange(2)
        axes[1, j].bar(positions - .17, values_buy, width=.34, color="#275d83", label="日均融资买入")
        axes[1, j].bar(positions + .17, values_repay, width=.34, color="#c36f28", label="日均隐含偿还")
        axes[1, j].set_xticks(positions, ["实施前5日", "实施起5日"])
        axes[1, j].set(ylabel="沪市融资活动（亿元/日）", title=f"实施日起5日事件净回报：{e.net_event_return:+.2%}", ylim=(0, max(values_buy + values_repay) * 1.32))
        axes[1, j].grid(axis="y", alpha=.18)
        axes[1, j].legend(frameon=False, fontsize=8)
    fig.suptitle("融资合同改变了可用空间，实际融资和指数收益仍需分别观察", fontsize=16, y=.99)
    fig.text(.5, .02, "上图为含除息权益的价格路径，未扣成本；下图收益是固定10000份、压力成本的事件测量。六个锚点来自三条政策链。", ha="center", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, .05, 1, .95))
    figure = OUT / "保证金规则_融资活动与指数回报.png"
    fig.savefig(figure, dpi=150, facecolor="white")
    plt.close(fig)

    contract_rows, return_rows, finance_rows = [], [], []
    for event in EVENTS:
        c = capacity[capacity.event_id.eq(event["event_id"])].iloc[0]
        contract_rows.append(f"| {event['label']} | {event['old_margin']:.0%}→{event['new_margin']:.0%} | {c.financing_capacity_per_100_margin_before:.0f}→{c.financing_capacity_per_100_margin_after:.0f} | {event['existing_contracts']} |")
        for node, label in [("ANNOUNCEMENT", "公告后"), ("EFFECTIVE", "实施起")]:
            b = returns[(returns.event_id == event["event_id"]) & (returns.node == node)].set_index("horizon")
            return_rows.append(f"| {event['label']}·{label} | {b.loc[5,'entry_date']} | {b.loc[5,'entry_gap_with_ex_dividend']:+.2%} | {b.loc[5,'net_event_return']:+.2%} | {b.loc[20,'net_event_return']:+.2%} |")
        f = flows[(flows.event_id == event["event_id"]) & (flows.node == "EFFECTIVE")].set_index("window")
        finance_rows.append(f"| {event['label']} | {f.loc['BEFORE5','buy_daily_cny']/1e8:.2f}→{f.loc['AFTER5','buy_daily_cny']/1e8:.2f} | {f.loc['BEFORE5','repayment_daily_cny']/1e8:.2f}→{f.loc['AFTER5','repayment_daily_cny']/1e8:.2f} | {f.loc['BEFORE5','net_five_day_cny']/1e8:+.2f}→{f.loc['AFTER5','net_five_day_cny']/1e8:+.2f} |")
    report = f"""# 510300历史发现：融资能力、存量合同与实际价格

**确定的是合同改变了什么；本次尚未找到能把这种变化稳定兑现为510300净收益的规则。** 三次最低融资保证金调整的原因和存量适用范围并不相同。2023年放松后融资确实扩张，指数ETF仍可下跌；两次收紧后融资买入和隐含偿还同时下降，不能把收紧直接解释成全体存量强平。

本研究固定2015、2023、2026三条政策链的公告与生效节点，共六锚点、两个固定期限、12个事件收益。沿用原日线及股息数据，并补取2026年1月20个沪市官方融资汇总日。六个节点不是六次独立试验，历史已被观察，本次不拟合参数、不新增策略账户。

## 从调整的原因读规则

2015年提高比例发生在融资快速回升之后。证监会当时披露，11月10日融资余额约1.14万亿元，比9月30日低点增长26%；监管目的包括控制新开仓杠杆。存量合约金额、期限及展期条件保留，不要求因这次调整立即平仓或追加保证金。[证监会当时发布会](https://www.csrc.gov.cn/csrc/c100029/c1000218/content.shtml)

2026年的说明同样把融资活跃、流动性较充裕列为逆周期提高比例的背景，明确仅作用于新开融资合约。存量及展期沿用原规定。[上交所2026年说明](https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260114_10805178.shtml)

2023年则为支持合理交易需求而降低比例，除了新增合约，还允许券商依约降低存量合约的保证金要求。这可以减少存量合同占用的保证金；是否实际降低、客户是否继续借入和买入，仍取决于券商及客户。[上交所2023年说明](https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20230827_5725661.shtml)

| 历史调整 | 最低比例 | 每100单位保证金对应的新增融资上限 | 存量合同处理 |
|---|---:|---:|---|
{chr(10).join(contract_rows)}

表内是静态融资容量，不是账户总资产杠杆，也不是等额现金流入股市。券商可以设置更高比例，担保物折算、信用额度、证券范围等也会约束实际买入。提高比例的两次均未统一要求存量退出；降低比例的一次允许释放存量保证金空间，因此不能把正反调整机械看成完全对称的冲击。

这里还有一个原因识别问题：政策本身会回应融资活跃或需求不足的状态。政策后的表现同时反映原来的市场状态、政策、同期消息和价格反应；本研究不声称识别了政策净因果效应。

## 容量有没有转成实际融资

统一使用沪市汇总。下表比较首个生效交易日前5日与生效起5日；融资买入、隐含偿还为日均亿元，余额变化为五日累计亿元。后5日数据只用于历史解释，绝不提前用于生效日入场。

| 调整 | 日均融资买入：前→后 | 日均隐含偿还：前→后 | 五日融资余额变化：前→后 |
|---|---:|---:|---:|
{chr(10).join(finance_rows)}

2015年实施后，买入降幅约21.18%，隐含偿还降幅约8.77%，余额由增转减；这可以由买入比偿还降得更多形成，不要求偿还突然激增。2026年买入降约23.68%，隐含偿还降约19.27%，余额仍略增。聚合数据不排除局部强平，却不能证明全体融资客户遭到强平。

2023年实施后，沪市五日余额净增204.07亿元，融资买入日均增长14.91%，融资活动确有扩张。但这不是沪深300成分股的净买入，更不是510300的特定资金需求；其间其他投资者的卖出、配置范围及风险预期仍可能抵消影响。

隐含偿还定义为买入减余额变化。上交所说明将直接还款、卖券还款、强平和权益调整都计入偿还，故即便直接披露的偿还额也不能全部称为卖压。[官方字段定义](https://www.sse.com.cn/market/othersdata/margin/sum/)

2026年额外取得了直接披露的偿还字段：实施前后日均由1651.63亿元降至1333.65亿元，同样下降。1月22日的反推值与披露值差约1.68亿元，原因尚未核实；保留两列及差额，**不把这个额外残差擅自命名为权益调整、真实资金或强平**。该差额不改变这次前后平均方向。

## 从公告到能成交的价格，还剩多少收益

公告按公开日期结束后的下一交易日开盘评价；生效时间此前已公告，因此用首个适用新规则交易日开盘。2023年9月8日是收市后生效，实施起点为9月11日，不能把9月8日开盘当新规则已经执行。

所有测量固定10000份，数量不由实现开盘价决定。买卖均按开盘加减千分之一滑点、单边万分之四佣金、最低5元及0.001元刻度计算；5/20日后开盘退出。它们是事件测量，未模拟盘口及全部风险预算，不是实际成交证明或完整账户业绩。

| 调整及锚点 | 测量入场日 | 前收至入场开盘含权价差 | 后5日压力净回报 | 后20日压力净回报 |
|---|---|---:|---:|---:|
{chr(10).join(return_rows)}

2023年公告后的首次开盘已高于前收盘约6.00%，随后5日为-3.31%、20日为-5.40%。生效后的融资净扩张并未使该固定5日回报转正（-1.13%）。这能反驳“释放融资空间必然带来随后指数上涨”的充分条件说法，却不能证明政策没有托底作用，也不能据此反向建立卖空规则。

那次开盘包含同一周末其他政策消息。财政部及税务总局同时宣布从8月28日起减半征收证券交易印花税，所以6%的跳空不能全部归因于融资保证金调整。[当时印花税公告](https://www.mof.gov.cn/jrttts/202308/t20230828_3904235.htm) 2015年的发布会还同时谈及恢复IPO及旧预缴款安排；这些竞争解释保留，不拆成已识别的贡献比例。

2026年1月15日入场、1月22日退出，原价回报约-2.23%，但持有人跨过1月16日登记日，享有每份0.123元分红；含权益毛回报约+0.31%，压力净回报约+0.02%。该笔在退出开盘仍有1230元应收，1月27日才支付，未当作已经可再投资现金。1月19日除息日新买入则不享有这次分红。忽略权益归属，会制造一次虚假的明显下跌或一笔虚假的额外收益。

![合同变化后的融资与价格](<{figure.as_posix()}>)

## 对后续研究的作用

本次把原因链拆成：监管为什么调整 → 改变谁的合同与可用空间 → 借款和偿还是否实际变化 → 这些活动是否涉及指数需求 → 首次可成交价格之后还有无净收益。历史证据支持前面几层存在差别，但没有补齐指数需求归属，也未建立可重复的净收益规律。

12个事件金额和股息现金/应收已从保存数据复算；三条政策链不足以评价当前年度交易次数，未计算策略夏普或年化，也不因稀少而放宽次数。目标保持未实现。这个分支完成后，不将这些结果变成事后挑年份、挑节点或改期限的交易规则。

研究设定：[protocol.json](protocol.json)；固定结果：[result.json](result.json)；逐笔测量：[固定事件收益.csv](固定事件收益.csv)；融资观察：[融资前后五日.csv](融资前后五日.csv)；必要复核：[necessary_checks.json](necessary_checks.json)。
"""
    report_path = OUT / "历史发现_融资能力与存量合同.md"
    report_path.write_text(report, encoding="utf-8")
    result.update(report_status="WRITTEN_FIGURE_PENDING_VISUAL_REVIEW", report=str(report_path.relative_to(ROOT)).replace("\\", "/"),
                  official_documents_total=8, new_sse_daily_observations=20,
                  distinct_financing_dates_observed=int(daily.date.nunique()),
                  direct_repayment_residual_status="RETAINED_UNEXPLAINED", direct_repayment_max_daily_residual_cny=float(residual_row.repayment_identity_difference_cny),
                  previous_goal_turn_classification="PROGRESS_FROZEN_INDEX_STATE_INCREMENT_RESOLVED", consecutive_blocked_goal_turns=0,
                  findings=["提高比例只约束新增合约，两次都不要求存量及展期统一追加或平仓；降低比例还允许存量释放保证金空间。",
                            "2015与2026实施后沪市融资买入和隐含偿还均下降，不能由收紧直接推断强平潮。",
                            "2023实施后沪市五日融资余额净增204.07亿元，而固定五日510300事件净回报为-1.13%。",
                            "2023公告后首次开盘已经跳升约6.00%；之后5日和20日净回报为负，同期印花税等政策不能单独归因。",
                            "2026公告五日原价下降2.23%，计入应收股息后的压力净回报约+0.02%；除息不能当作卖压。"],
                  branch_boundary="完成三条合同变化的历史比较；不根据结果挑节点、反号、改窗口或恢复旧融资过滤器。",
                  next_historical_question="如继续，须先去重并取得融资需求对沪深300的实际归属或独立新增约束证据；本批聚合余额及已见回报不能构成新交易条件。")
    save("result.json", result)
    print("合同范围、融资量、价格与股息的历史结论已写入；无新策略账户或参数。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="历史融资合同变化与指数传导，固定三条政策链。")
    parser.add_argument("command", choices=["prepare", "collect", "analyse", "finish"])
    args = parser.parse_args()
    {"prepare": prepare, "collect": collect, "analyse": analyse, "finish": finish}[args.command]()
