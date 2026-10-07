"""只用已有政策公告，计算固定20日规则的510300完整账户。"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

import sparse_event_training_core_v1 as core

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_equity_support_policy_event_v1"
PARENT = ROOT / "reports/research/510300_policy_information_clock_v1"


def freeze(out):
    core.require(not (out / "freeze.json").exists(), "本轮输入已保存，不覆盖。")
    cfg = core.load(out / "protocol.json")
    core.require(cfg["new_data_downloads"] == 0, "本轮只能使用已有本地数据。")
    (out / "inputs/raw").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(exist_ok=True)
    for destination, source in {
        "market.parquet": ROOT / "reports/research/510300_private_manager_intent_v1/inputs/market.parquet",
        "original_chain.csv": PARENT / "results/跨通道政策链_完整事实与时钟.csv",
        "original_catalog.csv": PARENT / "results/全部官方目录记录.csv",
        "parent_protocol.json": PARENT / "protocol.json",
        "mandate.json": ROOT / "config/510300_existing_data_training_mandate_v1.json",
    }.items():
        shutil.copy2(source, out / "inputs" / destination)
    original = pd.read_csv(out / "inputs/original_chain.csv").fillna("")
    selected = original[original.chain == "资本市场工具"].to_dict("records")
    core.require([r["node_id"] for r in selected] == [f"C{i:02d}" for i in range(1, 7)], "原政策链节点范围变化。")
    template = original[original.node_id == "R03"].iloc[0].to_dict()
    template.update({
        "node_id": "C07", "chain": "资本市场工具", "channel": "资本市场", "stage": "额度打通与条款优化",
        "source_available_upper": "2025-05-07T10:50:54+08:00",
        "time_precision": "TRANSCRIPT_SEGMENT_END_UPPER_BOUND",
        "title": "两项工具8000亿元额度打通及条款优化",
        "new_or_confirmed_information": "5000亿互换便利与3000亿回购增持再贷款额度打通；公告介绍期限和自有资金比例优化，并重申支持汇金必要时增持指数基金。",
        "previously_known": "两项工具及合计8000亿元总额度已于2024年宣布；部分细则此前已调整，不将合并额度当作新增8000亿元。",
        "still_unknown_at_this_node": "没有逐日净股票购买路径；不把宣示、可用额度和已申请贷款当作当日实际买入。",
        "amount": 8000, "amount_unit": "亿元：两项既有额度合计并打通", "underlying_execution_date": "",
        "source_checks": json.dumps(["2025-05-0710:50:54", "一是将两项工具总额度8000亿元合并使用", "自有资金比例要求从30%下降到10%"], ensure_ascii=False),
        "review_close": "2025-05-07", "research_execution_open": "2025-05-08",
        "first_daily_open_after_source_upper": "2025-05-08",
        "scope": "LOCAL_EXISTING_RAW_ADDITIONAL_FACT_NO_NEW_DOWNLOAD",
    })
    selected.append(template)
    for row in selected:
        source = PARENT / row["source_path"]
        core.require(core.digest(source) == row["source_sha256"], f"原始公告身份变化：{row['node_id']}")
        destination = out / "inputs/raw" / source.name
        shutil.copy2(source, destination)
        row["local_source_path"] = destination.relative_to(out).as_posix()
        normalized = re.sub(r"\s+", "", BeautifulSoup(source.read_bytes(), "html.parser").get_text("", strip=True))
        for check in json.loads(row["source_checks"]):
            core.require(re.sub(r"\s+", "", check) in normalized, f"公告保存内容未匹配：{row['node_id']} / {check}")
    core.export(out / "inputs/policy_facts.csv", pd.DataFrame(selected))
    catalog = pd.read_csv(out / "inputs/original_catalog.csv").fillna("")
    relevant = catalog[catalog.text.str.contains("股票|互换便利|capital market", case=False)].copy()
    relevant["admission_note"] = "年度目录为覆盖线索，不能直接按目录经济日期交易；交易仅依据policy_facts中已有原文的公告上界。"
    core.export(out / "inputs/catalog_equity_leads.csv", relevant)
    for source in [Path(__file__), Path(core.__file__)]:
        shutil.copy2(source, out / "code" / source.name)
    paths = [out / "protocol.json", *[p for p in (out / "inputs").rglob("*") if p.is_file()], *list((out / "code").iterdir())]
    core.save(out / "freeze.json", {"frozen_at": core.now(), "before_new_event_returns_and_accounts": True,
        "historical_market_previously_seen": True, "new_downloads": 0,
        "policy_nodes": len(selected), "announcement_dates": len({r["economic_event_date"] for r in selected}),
        "identities": {p.relative_to(out).as_posix(): core.digest(p) for p in paths}})
    print("本地7条政策事实与两种固定时点已保存；尚未计算本轮事件收益。", flush=True)


def events_for(out, cfg, market):
    facts = pd.read_csv(out / "inputs/policy_facts.csv").fillna("")
    dates = pd.DatetimeIndex(market.date).tz_localize("Asia/Shanghai")
    closes, decisions = dates + pd.Timedelta(hours=15), dates + pd.Timedelta(hours=9)
    events = []
    for publication_date, group in facts.groupby("economic_event_date", sort=True):
        known = max(pd.Timestamp(t) for t in group.source_available_upper)
        for timing in cfg["timings"]:
            if timing == "REVIEW_CLOSE_NEXT_OPEN_PRIMARY":
                review_i = int(closes.searchsorted(known, side="right"))
                entry_i = review_i + 1
                core.require(str(market.iloc[entry_i].date.date()) in group.research_execution_open.tolist(), "原研究确认后开盘时点不同。")
            else:
                entry_i = int(decisions.searchsorted(known, side="right"))
            exit_i = entry_i + cfg["horizon_sessions"] - 1
            core.require(entry_i > 20 and exit_i < len(market), "事件缺少价格历史或未来20日标签。")
            last, block = market.iloc[entry_i - 1], market.iloc[entry_i:exit_i + 1]
            core.require(known < decisions[entry_i], "事实晚于决策时间。")
            core.require(str(block.date.iloc[-1].date()) <= cfg["account_end"], "事件标签超出评价范围。")
            events.append({"timing": timing, "event_id": publication_date,
                "node_ids": "|".join(group.node_id), "title": "；".join(group.title),
                "known_at": known.isoformat(), "decision_at": decisions[entry_i].isoformat(),
                "entry_i": entry_i, "exit_i": exit_i,
                "entry_date": str(block.date.iloc[0].date()), "exit_date": str(block.date.iloc[-1].date()),
                "reference_close": float(last.close), "vol20": float(last.vol20),
                "weight": min(1.0, cfg["account"]["vol_target_annual"] / (last.vol20 * math.sqrt(cfg["annual_days"]))),
                "entry_open": float(block.open.iloc[0]), "exit_close": float(block.close.iloc[-1]),
                "dividend_per_share": float(block.dividend.iloc[1:].sum()),
                "gross_return20": (float(block.close.iloc[-1]) + float(block.dividend.iloc[1:].sum())) / float(block.open.iloc[0]) - 1})
    return pd.DataFrame(events).sort_values(["timing", "entry_i"]).reset_index(drop=True)


def account(timing, scenario, events, market, cfg):
    params = cfg["account"]
    slip = params["slippage_per_side"][scenario]
    signals = {int(r.entry_i): r for r in events[events.timing == timing].itertuples()}
    eligible = market[(market.date >= cfg["account_start"]) & (market.date <= cfg["account_end"])]
    start_i, end_i = int(eligible.index.min()), int(eligible.index.max())
    cash = peak = previous_equity = float(cfg["capital_cny"])
    quantity, position, halted, pending_brake = 0, None, False, False
    nav, trades, logs = [], [], []
    for i in range(start_i, end_i + 1):
        day = market.iloc[i]
        old_quantity = quantity
        dividend = old_quantity * float(day.dividend)
        cash += dividend
        if position:
            position["dividends_cny"] += dividend
        cost, exposed = 0.0, quantity > 0

        def sell(price, reason):
            nonlocal cash, quantity, position, cost
            core.require(position is not None and i > position["entry_i"], "退出必须满足T+1。")
            value = quantity * price * (1 - slip)
            fee = core.commission(value, cfg)
            cash += value - fee
            cost += fee + quantity * price * slip
            position.update({"exit_date": str(day.date.date()), "exit_i": i, "exit_raw_price": price,
                "exit_fill_price": price * (1 - slip), "exit_commission_cny": fee, "exit_reason": reason,
                "net_pnl_cny": value - fee + position["dividends_cny"] - position["entry_cash_debit"],
                "holding_sessions": i - position["entry_i"] + 1})
            trades.append(position)
            quantity, position = 0, None

        if pending_brake and quantity:
            sell(float(day.open), "DRAWDOWN_BRAKE_NEXT_OPEN")
            pending_brake = False
        signal = signals.get(i)
        desired = 0
        event_id = "BENCHMARK"
        planned_exit = end_i
        if timing == "BUY_AND_HOLD" and i == start_i:
            desired = core.quantity_for_budget(cash, cash, float(day.open), slip, cfg)
        elif signal is not None:
            decision = {"timing": timing, "scenario": scenario, "event_id": signal.event_id,
                "node_ids": signal.node_ids, "entry_date": signal.entry_date, "decision": "CASH"}
            if halted:
                decision["decision"] = "HALTED_AFTER_DRAWDOWN"
            elif quantity:
                decision["decision"] = "SKIP_ALREADY_HOLDING"
            else:
                budget = cash * signal.weight
                planned_q = core.quantity_for_budget(budget, cash, signal.reference_close, params["slippage_per_side"]["STRESS"], cfg)
                affordable_q = core.quantity_for_budget(budget, cash, float(day.open), slip, cfg)
                desired = min(planned_q, affordable_q)
                event_id, planned_exit = signal.event_id, int(signal.exit_i)
                decision.update({"weight": signal.weight, "shares": desired,
                    "decision": "BUY" if desired > 0 else "BELOW_ONE_LOT"})
            logs.append(decision)
        if desired > 0:
            core.require(quantity == 0, "不允许持仓期间加仓。")
            quantity = desired
            fill = float(day.open) * (1 + slip)
            value = quantity * fill
            fee = core.commission(value, cfg)
            cash -= value + fee
            cost += fee + quantity * float(day.open) * slip
            position = {"timing": timing, "scenario": scenario, "event_id": event_id,
                "entry_date": str(day.date.date()), "entry_i": i, "entry_raw_price": float(day.open),
                "entry_fill_price": fill, "shares": quantity, "entry_commission_cny": fee,
                "entry_cash_debit": value + fee, "dividends_cny": 0.0, "planned_exit_i": planned_exit}
            exposed = True
        if quantity and (i == position["planned_exit_i"] or i == end_i):
            sell(float(day.close), "FIXED_20_SESSION_CLOSE" if timing != "BUY_AND_HOLD" else "BENCHMARK_END_CLOSE")
        equity = cash + quantity * float(day.close)
        peak = max(peak, equity)
        drawdown = equity / peak - 1
        if timing in cfg["timings"] and drawdown <= -params["drawdown_brake"] and not halted:
            halted = True
            pending_brake = quantity > 0
        core.require(cash >= -1e-7 and quantity >= 0 and quantity % params["lot_size"] == 0, "现金或份额约束失效。")
        nav.append({"timing": timing, "scenario": scenario, "date": str(day.date.date()),
            "cash_cny": cash, "shares": quantity, "equity_cny": equity,
            "daily_return": equity / previous_equity - 1, "drawdown": drawdown,
            "dividends_cny": dividend, "cost_cny": cost, "held_at_start": old_quantity,
            "any_exposure": exposed, "halted": halted})
        previous_equity = equity
    core.require(quantity == 0, "期末必须清仓。")
    frame = pd.DataFrame(nav)
    r = frame.daily_return.to_numpy(dtype=float)
    years, deviation = len(r) / cfg["annual_days"], r.std(ddof=1)
    metric = {"timing": timing, "scenario": scenario, "start": frame.date.iloc[0], "end": frame.date.iloc[-1],
        "trading_days": len(frame), "ending_equity_cny": cash, "net_profit_cny": cash - cfg["capital_cny"],
        "net_sharpe": float(r.mean() / deviation * math.sqrt(cfg["annual_days"])) if deviation > 1e-14 else None,
        "max_drawdown": float(-frame.drawdown.min()), "annualized_return": (cash / cfg["capital_cny"]) ** (1 / years) - 1,
        "opportunities": len(trades), "opportunities_per_year": len(trades) / years,
        "exposure_day_fraction": float(frame.any_exposure.mean()), "cost_cny": float(frame.cost_cny.sum()),
        "profitable_opportunities": sum(t["net_pnl_cny"] > 0 for t in trades), "halted": halted}
    metric["meets_numeric_targets"] = metric["net_sharpe"] is not None and metric["net_sharpe"] >= cfg["target_net_sharpe"] and metric["max_drawdown"] <= cfg["target_max_drawdown_magnitude"]
    return frame, trades, logs, metric


def report(out, cfg, summary, events, metrics):
    lines = ["# 资本市场支持工具公告：固定规则检验", "", summary["conclusion"], "",
        "仅使用本地保存的7条政策事实，合并为6个公告日期。原24条跨通道政策链中C01—C06全部纳入；从同一份已存2025年5月7日原始公告补录C07，使用10:50:54发言段落结束作为信息上界。全年目录只用于显示来源范围，不能用后编年表的经济日期代替当时公开日期。", "",
        "本轮没有模型拟合。直接检验“政策已公开后买入”这一固定事件规则；不将同轮政策的公告、细则和操作结果当作独立学习样本。先前LPR和私募实际训练结果保留。", "",
        "主方案沿用原政策时钟：信息上界之后首个15:00收盘确认，次交易日开盘入场；时点对照允许信息在09:00之前已可知时，当日开盘入场。两者均持有含入场日在内20个交易日；持仓中忽略新事件，不延长退出。", "",
        "账户期为2024年首个交易日至2025年末，与原政策目录两年范围一致。2026年行情虽存在，但没有同口径公告覆盖，因此未把2026年当作没有事件的空仓期。实际公告覆盖不完整；例如目录中的2024年10月10日SFISF创设、2025年1月23日中长期资金入市发布会没有在本轮已有政策链中取得相应直接来源，保持缺口。ETF增持的完整独立公告集也不存在。", "",
        "只允许510300与现金。20万元、不融资、100份整手；20日历史波动确定10%年化目标波动仓位且上限100%；佣金单边万2且最低5元，基础/压力滑点万5/千1，现金和无风险收益均为0，242日年化。收盘账户回撤达到8%时，次开盘清仓并停止。该规则不能保证未来跳空回撤仍小于10%。", "",
        "|规则|成本|净夏普|最大回撤|年化收益|完整机会|在场天数占比|期末资金|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in metrics.itertuples():
        sh = "未定义" if pd.isna(r.net_sharpe) else f"{r.net_sharpe:.3f}"
        lines.append(f"|{r.timing}|{r.scenario}|{sh}|{r.max_drawdown:.2%}|{r.annualized_return:.2%}|{r.opportunities}|{r.exposure_day_fraction:.2%}|{r.ending_equity_cny:,.2f}|")
    lines += ["", "逐事件20日收益如下。事件窗口可能重叠，这些行不能相加成组合，也不代表全部实际成交。", "",
        "|公告日期|节点|主方案入场|计划退出|20日标的总回报|", "|---|---|---|---|---:|"]
    for r in events[events.timing == "REVIEW_CLOSE_NEXT_OPEN_PRIMARY"].itertuples():
        lines.append(f"|{r.event_id}|{r.node_ids}|{r.entry_date}|{r.exit_date}|{r.gross_return20:.2%}|")
    lines += ["", "额度打通没有新增8000亿元，互换操作与已申请回购贷款也不等于同日净买入510300。所有来源为已有事后保存网页，历史首版未认证；历史行情已在前轮研究中看过。即使某组历史数值达标，也不能认定有确定机会或独立验证。", "",
        "按已保存成交逐笔列出损益及首笔贡献，帮助识别2024年9月行情集中度；不删除亏损周期，不重置回撤停止状态，不用改窗口或挑公告挽救结果。最小价位、涨跌停队列与真实订单成交没有模拟。", ""]
    (out / "研究结果.md").write_text("\n".join(lines), encoding="utf-8")


def run(out):
    core.require(not (out / "summary.json").exists(), "结果已存在，不覆盖。")
    for name, expected in core.load(out / "freeze.json")["identities"].items():
        core.require(core.digest(out / name) == expected, f"固定输入变化：{name}")
    cfg = core.load(out / "protocol.json")
    market = core.read_market(out)
    events = events_for(out, cfg, market)
    navs, trades, logs, metrics = [], [], [], []
    for timing in [*cfg["timings"], "CASH", "BUY_AND_HOLD"]:
        for scenario in cfg["account"]["slippage_per_side"]:
            nav, trade, log, metric = account(timing, scenario, events, market, cfg)
            navs.append(nav)
            trades.extend(trade)
            logs.extend(log)
            metrics.append(metric)
    accounts = pd.DataFrame(metrics)
    for name, frame in {"事件与20日标签": events, "完整逐日账户": pd.concat(navs, ignore_index=True),
            "机会成交账簿": pd.DataFrame(trades), "逐事件判断": pd.DataFrame(logs), "账户指标": accounts}.items():
        core.export(out / "results" / f"{name}.csv", frame)
    primary = accounts[accounts.timing == "REVIEW_CLOSE_NEXT_OPEN_PRIMARY"]
    diagnostics = []
    for timing in cfg["timings"]:
        for scenario in cfg["account"]["slippage_per_side"]:
            subset = [t for t in trades if t["timing"] == timing and t["scenario"] == scenario]
            diagnostics.append({"timing": timing, "scenario": scenario, "complete_cycles": len(subset),
                "first_cycle_net_pnl_cny": subset[0]["net_pnl_cny"] if subset else None,
                "later_cycles_net_pnl_cny": sum(t["net_pnl_cny"] for t in subset[1:]),
                "interpretation": "保存成交的损益归因；不是删除首笔之后重跑或可执行的新账户。"})
    core.export(out / "results/首笔与后续成交损益归因.csv", pd.DataFrame(diagnostics))
    summary = {"study_id": cfg["study_id"], "completed_at": core.now(),
        "status": "FIXED_RULE_EVALUATED_NO_VALIDATED_STRATEGY", "fact_nodes": 7,
        "announcement_dates": 6, "account_scenarios": len(accounts), "model_fits": 0, "new_data_downloads": 0,
        "market_data_end": str(market.date.iloc[-1].date()), "account_start": str(primary.start.iloc[0]), "account_end": str(primary.end.iloc[0]),
        "primary_passes_both_costs_numerically": bool(primary.meets_numeric_targets.all()),
        "goal_achieved": False, "independent_validation_established": False, "orders_authorized": False,
        "conclusion": "资本市场支持工具公告后的固定20日账户已完成。" + ("主方案历史数值达标，但事件太少、覆盖不全，尚未形成独立验证策略。" if primary.meets_numeric_targets.all() else "主方案没有同时达到成本后夏普1.5、最大回撤不超过10%的目标。")}
    core.save(out / "summary.json", summary)
    report(out, cfg, summary, events, accounts)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(accounts.to_string(index=False), flush=True)


def verify(out):
    cfg = core.load(out / "protocol.json")
    for name, expected in core.load(out / "freeze.json")["identities"].items():
        core.require(core.digest(out / name) == expected, f"输入身份不同：{name}")
    events = pd.read_csv(out / "results/事件与20日标签.csv")
    nav = pd.read_csv(out / "results/完整逐日账户.csv")
    trades = pd.read_csv(out / "results/机会成交账簿.csv")
    metrics = pd.read_csv(out / "results/账户指标.csv")
    market = pd.read_parquet(out / "inputs/market.parquet").sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).dt.strftime("%Y-%m-%d")
    by_date = market.set_index("date")
    for r in events.itertuples():
        block = market.iloc[int(r.entry_i):int(r.exit_i) + 1]
        expected = (block.close.iloc[-1] + block.dividend.iloc[1:].sum()) / block.open.iloc[0] - 1
        core.require(abs(expected - r.gross_return20) < 1e-12, "事件标签复算不同。")
        core.require(pd.Timestamp(r.known_at) < pd.Timestamp(r.decision_at), "公告时间晚于决策。")
    for metric in metrics.itertuples():
        block = nav[(nav.timing == metric.timing) & (nav.scenario == metric.scenario)]
        equity = block.equity_cny.to_numpy(float)
        returns = equity / np.r_[cfg["capital_cny"], equity[:-1]] - 1
        peak = np.maximum.accumulate(np.r_[cfg["capital_cny"], equity])[1:]
        dd = equity / peak - 1
        core.require(np.allclose(returns, block.daily_return, rtol=0, atol=1e-12), "逐日收益不同。")
        core.require(np.allclose(dd, block.drawdown, rtol=0, atol=1e-12), "逐日回撤不同。")
        prices = by_date.loc[block.date, "close"].to_numpy(float)
        core.require(np.allclose(block.cash_cny + block.shares * prices, equity, rtol=0, atol=1e-7), "现金加持仓市值不能对账。")
        selected = trades[(trades.timing == metric.timing) & (trades.scenario == metric.scenario)]
        core.require(abs(selected.net_pnl_cny.sum() - (equity[-1] - cfg["capital_cny"])) < 1e-7, "成交损益不能对账。")
        core.require(abs(-dd.min() - metric.max_drawdown) < 1e-12, "最大回撤不同。")
        if returns.std(ddof=1) > 1e-14:
            sh = returns.mean() / returns.std(ddof=1) * math.sqrt(cfg["annual_days"])
            core.require(abs(sh - metric.net_sharpe) < 1e-10, "全账户夏普不同。")
        else:
            core.require(pd.isna(metric.net_sharpe), "零波动夏普应未定义。")
        for t in selected.itertuples():
            core.require(t.exit_i > t.entry_i, "交易不满足T+1。")
            block_prices = market.iloc[int(t.entry_i) + 1:int(t.exit_i) + 1]
            dividend = float(block_prices.dividend.sum()) * t.shares
            expected = t.shares * t.exit_fill_price - t.exit_commission_cny + dividend - t.entry_cash_debit
            core.require(abs(expected - t.net_pnl_cny) < 1e-7, "保存交易现金流不同。")
    print(json.dumps({"status": "PASS_SAVED_EVENT_AND_ACCOUNT_RECOMPUTATION", "event_timing_rows": len(events),
        "account_scenarios": len(metrics), "complete_transactions": len(trades), "new_model_fits": 0,
        "new_accounts": 0, "new_downloads": 0}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="已有资本市场工具公告的固定事件账户。")
    parser.add_argument("action", choices=["freeze", "run", "verify"])
    parser.add_argument("--root", type=Path, default=OUT)
    args = parser.parse_args()
    {"freeze": freeze, "run": run, "verify": verify}[args.action](args.root)


if __name__ == "__main__":
    main()
