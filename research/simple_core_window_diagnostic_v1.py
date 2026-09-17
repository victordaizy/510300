"""固定三窗口简单核心诊断：原价格状态、单层ETF波动缩放、独立资金账本。"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from research.strategy_review_diagnostics_v1 import ROOT, INPUTS, read, write, digest, inputs, metrics, cycles
from research.session_window_neighborhood_v1 import factor_rule, PERIODS, WINDOWS
from research.post_selection_continuous_accounts_v1 import simulate_rearmed_exit, simulate_indexed_request_account
from research.post_selection_continuous_factors_v1 import ordinary_multiplier, align_decisions
from research.adaptive_allocation_v1 import target_request

OUT = ROOT / "reports/research/510300_simple_core_window_diagnostic_v1"
OLD = ROOT / "reports/research/510300_strategy_review_diagnostics_v1"
COSTS = ["BASE", "STRESS"]


def csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def settings():
    original = read(ROOT / "config/510300_entry_vintage_exit_v1.json")
    cfg, data, dividends = inputs()
    cfg = {k: cfg[k] for k in ["initial_capital", "lot", "tick", "limit_fraction", "annual_days", "costs", "dividends"]}
    cfg.update(weight_band=0.1, cash_annual_rate_assumption=0.0)
    return cfg, original["specification"], data, dividends


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "protocol.json").exists():
        return check_freeze()
    cfg, spec, data, _ = settings()
    paths = [Path(__file__), ROOT / "research/post_selection_continuous_accounts_v1.py",
             ROOT / "research/post_selection_continuous_factors_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/session_window_neighborhood_v1.py",
             ROOT / "research/strategy_review_diagnostics_v1.py", ROOT / "config/510300_entry_vintage_exit_v1.json",
             ROOT / "config/510300_incremental_selected_intent_mix_v1.json", ROOT / "config/510300_research_authority_v6.json",
             INPUTS / "candidate_features.parquet", INPUTS / "candidate_prices.parquet", ROOT / cfg["dividends"],
             OUT / "用户评审原文.md"]
    for model in ["BUY_HOLD", "ETF_VOL10"]:
        for cost in COSTS:
            paths.extend(OLD / "counterfactuals" / model / "main" / cost / name for name in ["ledger.parquet", "decisions.parquet", "checkpoint.json"])
    protocol = {
        "study_id": "510300_SIMPLE_CORE_WINDOW_DIAGNOSTIC_V1", "registered_at": datetime.now().astimezone().isoformat(),
        "research_question": "原价格入场核心在移除学习退出和内部风险放大后，三邻近窗口是否仍有一致经济改善迹象",
        "windows": WINDOWS, "periods": PERIODS, "costs": COSTS, "settings": cfg, "price_specification": spec,
        "score": "SUM_N(intraday_log-overnight_log)/(STDEV.S_N(intraday_log-overnight_log)*SQRT(N))",
        "entry": "连续两日分数严格大于1；保留原feature_valid；必须满足原价格参考实际成交状态的空仓、再入场许可及冷却期",
        "rearm": "价格参考实际清仓后入场模式先归零才重置许可；入场成交后关闭许可；冷却=决策索引-实际退出索引>=2",
        "price_reference": "直接复用simulate_rearmed_exit(controller=None)。每个价格周期只买入一次和全部退出；含分红成本收益/高点口径不改。参考账本仅产生0/1，不以其收益或波动决定倍率。",
        "funded_account": "唯一资金账户目标=同日价格参考的reference_weight乘ordinary_multiplier；目标映射不额外滞后；次日开盘成交。再入场状态来自原价格参考，不把波动调仓视为新价格周期。",
        "volatility": "ETF含分红日收益20日样本标准差乘SQRT(242)；倍率=min(1,0.10/vol20)；无效值沿用前值、初始1。",
        "execution": "沿用target_request，正目标与当前收盘仓位差小于10个百分点保留份额；零目标不受带宽阻挡；100份整手、现金、T+1及原涨跌停成交约束。",
        "terminal": "两个区间均末日收盘估值，不人工末日开盘清仓；末日决定仅保存待下一交易日执行。",
        "removed": ["学习退出", "内部账户波动风险放大", "1.15倍乘数", "多计划及辅助信号合成", "额外两次归零等待"],
        "planned_primary_accounts": 12, "planned_price_reference_accounts": 12,
        "planned_new_comparison_accounts": 4, "planned_reused_main_comparison_accounts": 4,
        "comparators": "BUY_HOLD和ETF_VOL10；主历史4份直接复用；早期4份仅对齐末日收盘重新计算，并核对前1218日与旧账本一致。",
        "concentration": "实际资金账户已完成持仓周期的最大/前5净盈利对总净利润；另列2024年9月25日至10月8日固定事件窗口盈亏，及未完成周期。总净利润<=0不报告比例。",
        "decision_rule": "不选最佳窗口：一致改善才值得后续逐模块验证；只有60突出或主要依赖特殊大赢家则停止本路线扩展；不确定则保留观察而不增加过滤器。",
        "strict_forward_evidence_days": 0, "new_model_fits": 0, "random_samples": 0, "network_requests": 0,
        "historical_data_previously_observed": True, "candidate_returns_read_before_freeze": False,
        "current_strategy_disposition": "HIGH_OVERFITTING_RISK_LOCAL_ROBUSTNESS_FAILED_FROZEN_RESEARCH_CONTROL",
        "automatic_strategy_promotion": False, "goal_achieved": False, "position_impact": 0,
        "frozen_files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p)} for p in paths]
    }
    write(OUT / "protocol.json", protocol)
    columns = ["date", "previous_close", "open", "close", "dividend", "overnight_log", "intraday_log", "total_simple", "vol20", "variance60", "feature_valid"]
    csv(OUT / "signal_inputs.csv", data[columns])
    print("固定方案已登记：12份主账户、12份原价格参考、4份早期对齐对照；不训练、不选优。", flush=True)
    return protocol


def check_freeze():
    protocol = read(OUT / "protocol.json")
    for item in protocol["frozen_files"]:
        if digest(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("登记后的来源发生变化：" + item["path"])
    return protocol


def save_account(folder, run):
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in [("ledger", run[0]), ("decisions", run[1])]:
        frame.to_parquet(folder / (name + ".parquet"), index=False)
        csv(folder / (name + ".csv"), frame)
    if len(run) == 4:
        csv(folder / "price_cycles.csv", run[2])
    write(folder / "checkpoint.json", run[-1])


def target_account(data, dividends, cfg, cost, start, next_date, model, targets):
    return simulate_indexed_request_account(data, dividends, cfg, cfg["costs"][cost], start, model,
        targets=targets, event_mask=np.ones(len(data), bool), next_execution_date=next_date,
        request_policy=lambda a, p, v, c, m, t: target_request(a, p, v, c))


def run_accounts():
    check_freeze()
    cfg, spec, all_data, dividends = settings()
    for period, (start, end, next_date) in PERIODS.items():
        data = all_data[all_data.date.le(end)].reset_index(drop=True)
        multiplier = ordinary_multiplier(data)
        for window in WINDOWS:
            score, rule = factor_rule(data, window)
            for cost in COSTS:
                label = f"{period}_N{window}_{cost}"
                folder = OUT / "primary_accounts" / label
                if (folder / "completed.json").exists():
                    continue
                price = simulate_rearmed_exit(data, dividends, cfg, cfg["costs"][cost], start, rule, spec,
                                              controller=None, next_execution_date=next_date)
                save_account(OUT / "price_references" / label, price)
                binary = align_decisions(data, price[1])
                target = binary * multiplier
                run = target_account(data, dividends, cfg, cost, start, next_date, f"SIMPLE_CORE_N{window}", target)
                indices = run[1].origin_index.to_numpy(int)
                run[1]["price_reference_weight"] = binary[indices]
                run[1]["etf_volatility_multiplier"] = multiplier[indices]
                run[1]["score"] = score.iloc[indices].to_numpy(float)
                run[1]["price_entry_rearmed"] = price[1].entry_rearmed.to_numpy(bool)
                np.testing.assert_allclose(run[1].reference_weight, price[1].reference_weight * multiplier[indices], atol=1e-14, rtol=0)
                save_account(folder, run)
                write(folder / "completed.json", {"status": "COMPLETED", "period": period, "window": window, "cost": cost, **metrics(run[0])})
                print("已保存简单核心主账户：" + label, flush=True)
        for model in ["BUY_HOLD", "ETF_VOL10"]:
            for cost in COSTS:
                folder = OUT / "comparators" / f"{period}_{model}_{cost}"
                if (folder / "completed.json").exists():
                    continue
                old = OLD / "counterfactuals" / model / ("main" if period == "main" else "earlier") / cost
                old_ledger = pd.read_parquet(old / "ledger.parquet")
                if period == "main":
                    run = (old_ledger, pd.read_parquet(old / "decisions.parquet"), read(old / "checkpoint.json"))
                    checks = {"reused_saved_account": True, "source": old.relative_to(ROOT).as_posix()}
                else:
                    targets = np.ones(len(data)) if model == "BUY_HOLD" else multiplier
                    run = target_account(data, dividends, cfg, cost, start, next_date, model, targets)
                    columns = ["equity", "cash", "shares", "net_return", "filled_quantity", "commission", "dividend_receivable"]
                    np.testing.assert_allclose(run[0][columns].iloc[:-1], old_ledger[columns].iloc[:-1], atol=1e-7, rtol=0)
                    checks = {"reused_saved_account": False, "earlier_prefix_rows_identical": len(old_ledger)-1,
                              "old_terminal_clock": old_ledger.mark_clock.iloc[-1], "new_terminal_clock": "CLOSE",
                              "old_end_equity": old_ledger.equity.iloc[-1], "new_end_equity": run[0].equity.iloc[-1]}
                save_account(folder, run)
                write(folder / "completed.json", {"status": "COMPLETED", "period": period, "model": model, "cost": cost, **checks, **metrics(run[0])})
                print("已保存对照：" + folder.name, flush=True)


def summarize():
    check_freeze()
    rows, cycle_rows, concentration, yearly, event_rows = [], [], [], [], []
    for category in ["primary_accounts", "comparators"]:
        for folder in sorted((OUT / category).iterdir()):
            item = read(folder / "completed.json")
            ledger = pd.read_parquet(folder / "ledger.parquet")
            identity = {"account": folder.name, "category": category, "period": item["period"], "window": item.get("window"), "model": item.get("model", "SIMPLE_CORE"), "cost": item["cost"]}
            rows.append({**identity, **metrics(ledger), "ending_shares": int(ledger.shares.iloc[-1]), "cash_days": int(ledger.shares.eq(0).sum())})
            completed, active = cycles(ledger)
            completed["account"] = folder.name
            if len(completed):
                cycle_rows.extend(completed.to_dict("records"))
            profit = float(ledger.equity.iloc[-1] - 200000)
            winners = completed[completed.profit > 0].sort_values("profit", ascending=False) if len(completed) else completed
            best = winners.iloc[0].to_dict() if len(winners) else {}
            current_profit = float(ledger.equity.iloc[-1] - active["start_equity"]) if active else 0.0
            realized = float(completed.profit.sum()) if len(completed) else 0.0
            np.testing.assert_allclose(realized + current_profit, profit, atol=1e-7, rtol=0)
            event = ledger[ledger.date.between("2024-09-25", "2024-10-08")]
            fixed_pnl = float(event.pnl.sum()) if len(event) else None
            concentration.append({**identity, "completed_cycles": len(completed), "total_net_profit": profit,
                "top1_profit": best.get("profit", 0.0), "top1_entry": best.get("entry"), "top1_exit": best.get("exit"),
                "top1_share": best.get("profit", 0.0) / profit if profit > 0 else None,
                "top5_share": float(winners.profit.head(5).sum()) / profit if profit > 0 and len(winners) else None,
                "unfinished_cycle_profit": current_profit, "unfinished_entry": active["entry"] if active else None,
                "fixed_2024_event_pnl": fixed_pnl, "fixed_2024_event_share": fixed_pnl / profit if profit > 0 and fixed_pnl is not None else None})
            event_rows.extend([{**identity, **row} for row in event.to_dict("records")])
            previous = 200000.0
            for year, piece in ledger.groupby(ledger.date.dt.year):
                yearly.append({**identity, "year": int(year), "days": len(piece), "return": float(piece.equity.iloc[-1]/previous-1)})
                previous = float(piece.equity.iloc[-1])
    frame = pd.DataFrame(rows)
    comparisons = []
    for row in rows:
        if row["category"] != "primary_accounts":
            continue
        for model in ["BUY_HOLD", "ETF_VOL10"]:
            control = frame[(frame.period == row["period"]) & (frame.cost == row["cost"]) & (frame.model == model)].iloc[0]
            comparisons.append({"account": row["account"], "period": row["period"], "window": row["window"], "cost": row["cost"], "comparator": model,
                **{name + "_difference": row[name]-float(control[name]) for name in ["annual_return", "sharpe", "volatility", "max_drawdown", "mean_exposure"]}})
    for name, values in [("account_metrics", rows), ("cycle_concentration", concentration), ("completed_cycles", cycle_rows),
                         ("yearly_returns", yearly), ("fixed_2024_event_ledger", event_rows), ("comparison_differences", comparisons)]:
        csv(OUT / (name + ".csv"), pd.DataFrame(values))
    write(OUT / "result.json", {"status": "COMPLETED_FIXED_THREE_WINDOW_SIMPLE_CORE_DIAGNOSTIC", "completed_at": datetime.now().astimezone().isoformat(),
        "primary_accounts": 12, "price_reference_accounts": 12, "comparison_accounts": 8, "new_comparison_accounts": 4,
        "reused_comparison_accounts": 4, "new_model_fits": 0, "strict_forward_evidence_days": 0,
        "selected_window": None, "goal_achieved": False, "position_impact": 0,
        "multi_module_removal_cannot_identify_single_module_causality": True, "metrics": rows})
    print(frame[frame.cost == "STRESS"][["account", "annual_return", "sharpe", "volatility", "max_drawdown", "mean_exposure"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="登记、运行或汇总固定三窗口简单核心诊断")
    parser.add_argument("stage", choices=["freeze", "run", "summarize"])
    args = parser.parse_args()
    {"freeze": freeze, "run": run_accounts, "summarize": summarize}[args.stage]()
