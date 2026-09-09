"""使用现有行情直接检验简单进出场；所有判断在收盘后产生。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import (
    Account, affordable_quantity, execute_order, fill_price, normalize_dividends,
    require, save_account, summarize,
)
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_simple_price_entry_exit_v1"
CONFIG = ROOT / "config/510300_simple_price_entry_exit_v1.json"

NAMES = {
    "T1_BAND60": "六十日均线缓冲与追踪退出",
    "T2_CHANNEL20": "二十日突破与十日低点退出",
    "T3_EMA10_40": "双指数均线缓冲",
    "T4_PULLBACK20": "长期上涨中的二十日均线收复",
    "R1_RSI2_BULL": "上涨趋势内两日超跌",
    "R2_Z_CONFIRM": "偏离均值后的首日回升",
    "R3_SHOCK_RECOVERY": "五日急跌后的收盘转强",
    "R4_BAND_REENTRY": "跌出波动带后的重新收复",
    "S1_TREND_REBOUND": "趋势突破与震荡反弹切换",
    "S2_PULLBACK_SHOCK": "上涨回调与弱市急跌切换",
    "D1_DRAWDOWN_RECOVERY": "回撤退出与短期趋势恢复",
    "R5_PULLBACK_DOUBLE": "长期上涨中的双条件回调",
}


def modes(loss=.05, trail=None, take=None, days=None):
    return {"loss": loss, "trail": trail, "take": take, "days": days}


def specifications():
    return {
        "T1_BAND60": {"cooldown": 3, "modes": {1: modes(.08, .08)}},
        "T2_CHANNEL20": {"cooldown": 3, "modes": {1: modes(.06, .06)}},
        "T3_EMA10_40": {"cooldown": 2, "modes": {1: modes(.08, .08)}},
        "T4_PULLBACK20": {"cooldown": 2, "modes": {1: modes(.05, .06, days=60)}},
        "R1_RSI2_BULL": {"cooldown": 1, "modes": {1: modes(.04, days=5)}},
        "R2_Z_CONFIRM": {"cooldown": 1, "modes": {1: modes(.05, days=10)}},
        "R3_SHOCK_RECOVERY": {"cooldown": 1, "modes": {1: modes(.04, take=.04, days=5)}},
        "R4_BAND_REENTRY": {"cooldown": 1, "modes": {1: modes(.05, days=10)}},
        "S1_TREND_REBOUND": {"cooldown": 2, "modes": {1: modes(.08, .08), 2: modes(.05, days=10)}},
        "S2_PULLBACK_SHOCK": {"cooldown": 1, "modes": {1: modes(.05, .06, days=20), 2: modes(.04, take=.04, days=5)}},
        "D1_DRAWDOWN_RECOVERY": {"cooldown": 3, "modes": {1: modes(.06, .08)}},
        "R5_PULLBACK_DOUBLE": {"cooldown": 1, "modes": {1: modes(.04, days=10)}},
    }


def signals(data):
    """仅引用当日收盘及更早记录，返回入场模式及各模式退出条件。"""
    w = data.wealth
    up = w > w.shift(1)
    bull = data.sma120 > 0
    strong = bull & (data.efficiency20 > .3)
    above60 = data.sma60 > .01
    below60 = data.sma60 < -.01
    below20 = data.sma20 < 0
    ema = w.ewm(span=10, adjust=False).mean() / w.ewm(span=40, adjust=False).mean() - 1
    high20 = w.shift(1).rolling(20).max()
    low10 = w.shift(1).rolling(10).min()
    breakout = w > high20
    lowexit = w < low10
    zentry = (data.z20 < -1.5) & up
    shock = (data.mom5.shift(1) < np.log(.97)) & up & (data.close_location >= .6)
    rules = {}

    def add(key, entry, exit_flag, second_entry=None, second_exit=None):
        entry_mode = np.where(entry.fillna(False), 1, 0)
        exits = {1: exit_flag.fillna(False).to_numpy(bool)}
        if second_entry is not None:
            entry_mode = np.where(second_entry.fillna(False) & (entry_mode == 0), 2, entry_mode)
            exits[2] = second_exit.fillna(False).to_numpy(bool)
        valid = data.feature_valid.fillna(False).to_numpy(bool)
        entry_mode = np.where(valid, entry_mode, 0)
        rules[key] = {"entry": entry_mode, "exit": exits}

    add("T1_BAND60", above60 & above60.shift(1, fill_value=False) & (data.mom20 > 0),
        below60 & below60.shift(1, fill_value=False))
    add("T2_CHANNEL20", breakout, lowexit)
    add("T3_EMA10_40", (ema > .005) & (ema.shift(1) > .005), (ema < -.005) & (ema.shift(1) < -.005))
    add("T4_PULLBACK20", bull & (data.sma20 > 0) & (data.sma20.shift(1) <= 0),
        (~bull) | (below20 & below20.shift(1, fill_value=False)))
    add("R1_RSI2_BULL", bull & (data.rsi2 < 10), (~bull) | (data.rsi2 > 70))
    add("R2_Z_CONFIRM", zentry, data.z20 >= 0)
    add("R3_SHOCK_RECOVERY", shock, data.mom5 > 0)
    add("R4_BAND_REENTRY", (data.z20.shift(1) < -2) & (data.z20 >= -2), data.z20 >= -.2)
    add("S1_TREND_REBOUND", strong & breakout, (~strong) | lowexit,
        (~strong) & zentry, strong | (data.z20 >= 0))
    add("S2_PULLBACK_SHOCK", bull & (data.mom5 < np.log(.99)) & up, (~bull) | (data.rsi2 > 80),
        (~bull) & shock, bull | (data.mom5 > 0))
    add("D1_DRAWDOWN_RECOVERY", (data.sma20 > 0) & (data.mom5 > 0),
        (data.dd20 <= -.06) | (below20 & below20.shift(1, fill_value=False)))
    add("R5_PULLBACK_DOUBLE", bull & (data.rsi2 < 20) & (data.z20 < 0), (~bull) | (data.rsi2 > 80))
    return rules


def simulate_policy(data, dividends, config, cost, start, rule, spec):
    """按实际成交状态维护持仓周期、退出待成交和再入场等待期。"""
    first = int(np.flatnonzero(data.date >= pd.Timestamp(start))[0])
    last, anchor = len(data) - 1, first - 1
    dates = pd.DatetimeIndex(data.date)
    op, cl = data.open.to_numpy(float), data.close.to_numpy(float)
    previous, distribution = data.previous_close.to_numpy(float), data.dividend.to_numpy(float)
    require(len(rule["entry"]) == len(data), "信号长度不符")
    record_events, ex_events, pay_events = {}, {}, {}
    for key, event in enumerate(dividends.itertuples()):
        for field, mapping in [("record_date", record_events), ("ex_date", ex_events)]:
            when = getattr(event, field)
            if when in dates:
                mapping.setdefault(dates.get_loc(when), []).append((key, event.cash_dividend_per_share))
        idx = dates.searchsorted(event.payment_date)
        if idx < len(dates):
            pay_events.setdefault(idx, []).append((key, event.payment_date == dates[idx]))
    account = Account(config["initial_capital"])
    previous_nav, previous_mark = config["initial_capital"], cl[anchor]
    records, decisions, cycles = [], [], []
    cycle, last_exit, cycle_number = None, -1000000, 0
    pending_reasons, peak_value, cycle_dividends = [], np.nan, 0.

    def decide(t):
        nonlocal pending_reasons
        mode = int(rule["entry"][t])
        reasons, qty, selected_mode = [], 0, 0
        if account.shares:
            selected_mode = cycle["mode"]
            s = spec["modes"].get(selected_mode, spec["modes"].get(str(selected_mode)))
            current_value = account.shares * cl[t] + cycle_dividends
            cycle_return = current_value / cycle["entry_cost_cny"] - 1
            if not pending_reasons:
                if rule["exit"][selected_mode][t]:
                    reasons.append("本持仓模式的价格退出条件成立")
                if s["loss"] is not None and cycle_return <= -s["loss"]:
                    reasons.append("持仓含分红收益触及固定止损")
                if s["trail"] is not None and current_value / peak_value - 1 <= -s["trail"]:
                    reasons.append("含分红持仓价值从周期高点回落至追踪退出线")
                if s["take"] is not None and cycle_return >= s["take"]:
                    reasons.append("持仓含分红收益达到预定止盈")
                if s["days"] is not None and t - cycle["entry_index"] + 1 >= s["days"]:
                    reasons.append("预定最长持有交易日到期")
                pending_reasons = reasons
            if pending_reasons:
                qty, reasons = -account.shares, pending_reasons
                action = "请求全部退出，受阻后逐日继续请求"
            else:
                action = "保持已有份额，收盘检查退出"
        elif mode and t - last_exit >= spec["cooldown"]:
            qty = affordable_quantity(account.cash, fill_price(cl[t], 1, cost, config["tick"]), cost, config["lot"])
            selected_mode, action = mode, "入场条件成立，下一开盘请求买入"
        elif t - last_exit < spec["cooldown"]:
            action = "退出后的等待期尚未结束"
        else:
            action = "空仓等待新的入场条件"
        row = {"origin": dates[t], "execution_date": dates[t + 1] if t + 1 < len(dates) else pd.NaT,
               "origin_index": t, "requested_quantity": int(qty), "entry_mode": selected_mode,
               "exit_reasons": "；".join(reasons), "action": action,
               "reference_weight": 0. if qty < 0 or (not account.shares and not qty) else 1.,
               "simulation_only": True}
        decisions.append(row)
        return row

    pending = decide(anchor)
    for day in range(first, last + 1):
        old_shares, recognized, paid = account.shares, 0., 0.
        for key, amount in ex_events.get(day, []):
            value = account.entitlements.get(key, 0) * amount
            account.receivables[key] = value
            recognized += value
        if old_shares:
            cycle_dividends += recognized
        for key, same in pay_events.get(day, []):
            if not same:
                value = account.receivables.pop(key, 0.)
                paid += value
                account.cash += value
        terminal = day == last
        quantity = -account.shares if terminal else int(pending["requested_quantity"])
        before = account.value(op[day])
        execution = execute_order(account, quantity, op[day], previous[day], distribution[day], day, cost, config)
        mark = op[day] if terminal else cl[day]
        if old_shares == 0 and account.shares > 0:
            cycle_number += 1
            cycle_dividends, pending_reasons = 0., []
            cycle = {"cycle_id": cycle_number, "mode": int(pending["entry_mode"]), "entry_date": dates[day], "entry_index": day,
                     "entry_origin": pending["origin"], "entry_quantity": account.shares,
                     "entry_cost_cny": execution["notional"] + execution["commission"]}
            peak_value = cycle["entry_cost_cny"]
        elif old_shares > 0 and account.shares == 0:
            proceeds = execution["notional"] - execution["commission"]
            cycle.update({"exit_date": dates[day], "exit_origin": pending["origin"] if not terminal else dates[day],
                          "exit_reasons": pending["exit_reasons"] if not terminal else "研究终点统一开盘退出",
                          "holding_intervals": day - cycle["entry_index"], "dividend_cny": cycle_dividends,
                          "net_profit_cny": proceeds + cycle_dividends - cycle["entry_cost_cny"]})
            cycles.append(cycle)
            cycle, last_exit, pending_reasons, peak_value, cycle_dividends = None, day, [], np.nan, 0.
        elif account.shares and execution["filled_quantity"]:
            raise ValueError("本轮仅单次买入及全部退出，出现非预定份额变化")
        if not terminal:
            for key, same in pay_events.get(day, []):
                if same:
                    value = account.receivables.pop(key, 0.)
                    paid += value
                    account.cash += value
            for key, amount in record_events.get(day, []):
                account.entitlements[key] = account.shares
        if account.shares:
            peak_value = max(peak_value, account.shares * mark + cycle_dividends)
        nav = account.value(mark)
        price_pnl = old_shares * (op[day] - previous_mark) + account.shares * (mark - op[day])
        error = nav - previous_nav - price_pnl - recognized + execution["commission"] + execution["slippage_cost"]
        require(abs(error) < 1e-6, "价格策略账户财富恒等式不成立")
        account.assert_valid()
        records.append({"date": dates[day], "open": op[day], "mark": mark, "mark_clock": "OPEN_TERMINAL" if terminal else "CLOSE",
                        "cash": account.cash, "shares": account.shares, "dividend_receivable": account.receivable(), "equity": nav,
                        "net_return": nav / previous_nav - 1, "pnl": nav - previous_nav, "price_pnl": price_pnl,
                        "dividend_recognized": recognized, "dividend_paid": paid, "exposure": account.shares * mark / nav,
                        "accounting_error": error, "origin": dates[day - 1], "terminal_unliquidated": bool(terminal and account.shares),
                        "turnover": execution["notional"] / before, "cycle_id": cycle["cycle_id"] if cycle else None,
                        "mode": cycle["mode"] if cycle else 0, "execution_reasons": pending["exit_reasons"] if not terminal else "研究终点统一退出",
                        **execution})
        previous_nav, previous_mark = nav, mark
        if not terminal:
            pending = decide(day)
    if cycle:
        cycles.append({**cycle, "exit_date": None, "exit_reasons": "研究终点退出未成交"})
    return pd.DataFrame(records), pd.DataFrame(decisions), pd.DataFrame(cycles)


def freeze():
    require(not CONFIG.exists(), "本轮已登记，禁止覆盖参数")
    cfg = json.loads((ROOT / "config/510300_adaptive_allocation_v1.json").read_text(encoding="utf-8"))
    cfg = {k: cfg[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                               "cash_annual_rate_assumption", "high_sharpe_target", "costs", "weight_band"]}
    cfg.update({"study_id": "510300_SIMPLE_PRICE_ENTRY_EXIT_V1", "round": 23, "registered_at": now(), "primary": "S1_TREND_REBOUND",
                "candidate_names": NAMES, "candidate_specs": specifications(), "candidate_count": 12,
                "features": "reports/research/510300_adaptive_allocation_v1/features.parquet", "dividends": "data/reference/510300_dividends.csv",
                "rules": "docs/510300_SIMPLE_PRICE_ENTRY_EXIT_V1.md", "position_impact": 0,
                "evidence_class": "RESEARCH_SCREEN_ON_PREVIOUSLY_OBSERVED_HISTORY", "independent_validation": "NOT_ESTABLISHED"})
    paths = [Path(__file__), ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / "research/intraday_overnight_increment_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第23轮12种简单进出场规则已登记，尚未读取候选账户收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "本轮登记后的输入或规则发生变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    rules = signals(data)
    pd.DataFrame({"date": data.date, **{k: r["entry"] for k, r in rules.items()}}).to_parquet(OUT / "entry_modes.parquet", index=False)
    metrics, yearly, eras = [], [], []
    for cost_id, cost in cfg["costs"].items():
        accounts = {}
        for i, key in enumerate(cfg["candidate_specs"], 1):
            ledger, decisions, cycles = simulate_policy(data, dividends, cfg, cost, cfg["evaluation_start"], rules[key], cfg["candidate_specs"][key])
            save_account(OUT / "evaluation" / cost_id, key, ledger, decisions)
            cycles.to_csv(OUT / "evaluation" / cost_id / f"{key}_cycles.csv", index=False, encoding="utf-8-sig")
            accounts[key] = ledger
            print(f"{cost_id} 简单策略 {i}/12：{NAMES[key]}，净夏普 {summarize(ledger,cfg)['net_sharpe']:.4f}", flush=True)
        benchmark, decisions = simulate_event_account(data, dividends, cfg, cost, cfg["evaluation_start"], "BUY_HOLD", event_mask=np.ones(len(data), bool))
        save_account(OUT / "evaluation" / cost_id, "BUY_HOLD", benchmark, decisions)
        accounts["BUY_HOLD"] = benchmark
        base = summarize(benchmark, cfg)
        for key, ledger in accounts.items():
            m = {"cost": cost_id, "model": key, "name": NAMES.get(key, "买入持有"), **summarize(ledger, cfg)}
            m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
            m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
            metrics.append(m)
            for y, g in ledger.groupby(ledger.date.dt.year):
                yearly.append({"cost": cost_id, "model": key, "year": int(y), **summarize(g, cfg)})
            for label, start, end in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                g = ledger[(ledger.date >= start) & (ledger.date <= end)]
                eras.append({"cost": cost_id, "model": key, "era": label, **summarize(g, cfg)})
        pd.DataFrame({"date": benchmark.date, **{k: a.net_return.to_numpy() for k, a in accounts.items()}}).to_parquet(OUT / f"{cost_id}_returns.parquet", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eras).to_csv(OUT / "era_metrics.csv", index=False, encoding="utf-8-sig")
    best = max((m for m in metrics if m["cost"] == "BASE" and m["model"] != "BUY_HOLD"), key=lambda m: m["net_sharpe"] or -999)
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "COMPLETED_HISTORICAL_SCREEN", "candidate_configurations": 12,
              "evaluation_accounts": 26, "new_accounts_generated": 26, "trained_models": 0, "all_metrics": metrics,
              "primary": [m for m in metrics if m["model"] == cfg["primary"]], "post_selected_best_base": best,
              "historical_point_target_met": any(m["meets_point_target"] for m in metrics if m["model"] != "BUY_HOLD"),
              "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY", "goal_achieved": False, "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "本轮最高基础夏普": best["net_sharpe"], "方法": best["name"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
