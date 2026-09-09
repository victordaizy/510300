"""简单切换策略的有限局部改动及仅按历史表现选规则。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.simple_price_entry_exit_v1 import simulate_policy, modes
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_simple_regime_local_v1"
CONFIG = ROOT / "config/510300_simple_regime_local_v1.json"
ORIGINAL = "L01_FAST_QUICK_ALL_Z15"


def candidates():
    result = []
    for regime in ["FAST", "SLOW"]:
        for exit_mode in ["QUICK", "PATIENT"]:
            for rebound_gate in ["ALL", "DRAWDOWN", "VOLATILITY"]:
                for z in [1.5, 2.0]:
                    number = len(result) + 1
                    key = f"L{number:02}_{regime}_{exit_mode}_{rebound_gate}_Z{int(z*10)}"
                    result.append({"id": key, "regime": regime, "exit_mode": exit_mode, "rebound_gate": rebound_gate, "z": z})
    return result


def specification():
    return {"cooldown": 2, "modes": {1: modes(.08, .08), 2: modes(.05, days=10)}}


def make_rule(data, p):
    w = data.wealth
    bull = data.sma120 > 0
    if p["regime"] == "FAST":
        strong = bull & (data.efficiency20 > .3)
    else:
        strong = (w.rolling(60).mean() > w.rolling(120).mean()) & (data.mom120 > 0)
    high20 = w.shift(1).rolling(20).max()
    gate = pd.Series(True, index=data.index)
    if p["rebound_gate"] == "DRAWDOWN":
        gate = data.dd120 > -.15
    elif p["rebound_gate"] == "VOLATILITY":
        gate = data.vol20 < data.vol60
    trend_entry = strong & (w > high20)
    rebound_entry = (~strong) & gate & (data.z20 < -p["z"]) & (w > w.shift(1))
    if p["exit_mode"] == "QUICK":
        trend_exit = (~strong) | (w < w.shift(1).rolling(10).min())
    else:
        trend_exit = (~bull) | (w < w.shift(1).rolling(20).min())
    rebound_exit = strong | (data.z20 >= 0)
    entry = np.where(trend_entry.fillna(False), 1, np.where(rebound_entry.fillna(False), 2, 0))
    entry = np.where(data.feature_valid.fillna(False), entry, 0)
    return {"entry": entry, "exit": {1: trend_exit.fillna(False).to_numpy(bool), 2: rebound_exit.fillna(False).to_numpy(bool)}}


def rolling_selection(data, shadow, rules, cfg, window):
    dates = pd.DatetimeIndex(data.date)
    n, ids = len(data), list(rules)
    anchor = int(np.flatnonzero(data.date >= cfg["evaluation_start"])[0]) - 1
    quarter = dates.to_period("Q")
    entry = np.zeros(n, int)
    exits = {1: np.zeros(n, bool), 2: np.zeros(n, bool)}
    selected = None
    receipts = []
    for t in range(anchor, n - 1):
        if t == anchor or quarter[t] != quarter[t + 1]:
            eligible = []
            start = t - window + 1
            for key in ids:
                frame = shadow[key]
                r = frame["returns"][start:t + 1]
                trades = int(frame["trades"][start:t + 1].sum())
                if len(r) != window or not np.isfinite(r).all() or trades < 6 or np.std(r, ddof=1) <= 1e-12:
                    continue
                score = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(cfg["annual_days"]))
                if score > 0:
                    eligible.append((score, key))
            eligible.sort(key=lambda item: (-item[0], item[1]))
            selected = eligible[0][1] if eligible else None
            receipts.append({"origin": dates[t], "next_execution_date": dates[t + 1], "window": window,
                             "first_training_mark": dates[start], "last_training_mark": dates[t],
                             "selected": selected or "CASH", "past_net_sharpe": eligible[0][0] if eligible else None,
                             "eligible_candidates": len(eligible), "source_cost": "BASE"})
        if selected is None:
            entry[t] = 0
            exits[1][t] = exits[2][t] = True
        else:
            entry[t] = rules[selected]["entry"][t]
            for mode in [1, 2]:
                exits[mode][t] = rules[selected]["exit"][mode][t]
    return {"entry": entry, "exit": exits}, pd.DataFrame(receipts)


def freeze():
    require(not CONFIG.exists(), "局部改动已经登记，不覆盖")
    cfg = json.loads((ROOT / "config/510300_simple_price_entry_exit_v1.json").read_text(encoding="utf-8"))
    for key in ["candidate_names", "candidate_specs", "frozen_files"]:
        cfg.pop(key, None)
    cfg.update({"study_id": "510300_SIMPLE_REGIME_LOCAL_V1", "round": 24, "registered_at": now(), "primary": "A1_PAST756",
                "shadow_start": "2015-01-05", "candidates": candidates(), "common_specification": specification(),
                "candidate_count": 26, "new_candidate_configurations": 25, "reused_prior_candidate_count": 1,
                "selector_windows": {"A1_PAST756": 756, "A2_PAST504": 504}, "rules": "docs/510300_SIMPLE_REGIME_LOCAL_V1.md"})
    paths = [Path(__file__), ROOT / "research/simple_price_entry_exit_v1.py", ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"]]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第24轮已登记24个局部设置，其中1个为原策略对照，另加2个仅用过去表现的季度选择器。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for p in cfg["frozen_files"]:
        require(digest(ROOT / p["path"]) == p["sha256"], "局部改动输入或规则发生变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    div = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    rules = {p["id"]: make_rule(data, p) for p in cfg["candidates"]}
    dates = pd.DatetimeIndex(data.date)
    shadow, accounts = {}, {"BASE": {}, "STRESS": {}}
    spec = cfg["common_specification"]
    for i, key in enumerate(rules, 1):
        ledger, decisions, cycles = simulate_policy(data, div, cfg, cfg["costs"]["BASE"], cfg["shadow_start"], rules[key], spec)
        save_account(OUT / "shadow" / "BASE", key, ledger, decisions)
        returns, trades = np.full(len(data), np.nan), np.zeros(len(data), int)
        locations = dates.get_indexer(ledger.date)
        returns[locations] = ledger.net_return
        trades[locations] = (ledger.filled_quantity != 0).astype(int)
        shadow[key] = {"returns": returns, "trades": trades}
        for cost_id, cost in cfg["costs"].items():
            if key == ORIGINAL:
                old = ROOT / "reports/research/510300_simple_price_entry_exit_v1/evaluation" / cost_id
                ledger = pd.read_parquet(old / "S1_TREND_REBOUND_ledger.parquet")
                decisions = pd.read_parquet(old / "S1_TREND_REBOUND_decisions.parquet")
                cycles = pd.read_csv(old / "S1_TREND_REBOUND_cycles.csv")
            else:
                ledger, decisions, cycles = simulate_policy(data, div, cfg, cost, cfg["evaluation_start"], rules[key], spec)
            save_account(OUT / "evaluation" / cost_id, key, ledger, decisions)
            cycles.to_csv(OUT / "evaluation" / cost_id / f"{key}_cycles.csv", index=False, encoding="utf-8-sig")
            accounts[cost_id][key] = ledger
        if i % 4 == 0:
            print(f"局部设置已完成 {i}/24：基础训练账户及两种费用评价账户已保存。", flush=True)
    for key, window in cfg["selector_windows"].items():
        rule, receipts = rolling_selection(data, shadow, rules, cfg, window)
        receipts.to_csv(OUT / f"{key}_selection.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"date": data.date, "entry_mode": rule["entry"], "exit_trend": rule["exit"][1], "exit_rebound": rule["exit"][2]}).to_parquet(OUT / f"{key}_signals.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            ledger, decisions, cycles = simulate_policy(data, div, cfg, cost, cfg["evaluation_start"], rule, spec)
            save_account(OUT / "evaluation" / cost_id, key, ledger, decisions)
            cycles.to_csv(OUT / "evaluation" / cost_id / f"{key}_cycles.csv", index=False, encoding="utf-8-sig")
            accounts[cost_id][key] = ledger
        print(f"季度选择器已完成：{key}，所有选择仅用当日及更早已产生收益。", flush=True)
    metrics, yearly, eras = [], [], []
    for cost_id in cfg["costs"]:
        old = ROOT / "reports/research/510300_simple_price_entry_exit_v1/evaluation" / cost_id
        benchmark = pd.read_parquet(old / "BUY_HOLD_ledger.parquet")
        accounts[cost_id]["BUY_HOLD"] = benchmark
        base = summarize(benchmark, cfg)
        for key, ledger in accounts[cost_id].items():
            m = {"cost": cost_id, "model": key, **summarize(ledger, cfg)}
            m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
            m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
            metrics.append(m)
            for y, group in ledger.groupby(ledger.date.dt.year):
                yearly.append({"cost": cost_id, "model": key, "year": int(y), **summarize(group, cfg)})
            for label, start, end in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                group = ledger[(ledger.date >= start) & (ledger.date <= end)]
                eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
        pd.DataFrame({"date": benchmark.date, **{key: ledger.net_return.to_numpy() for key, ledger in accounts[cost_id].items()}}).to_parquet(OUT / f"{cost_id}_returns.parquet", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(yearly).to_csv(OUT / "yearly_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(eras).to_csv(OUT / "era_metrics.csv", index=False, encoding="utf-8-sig")
    best = max((m for m in metrics if m["cost"] == "BASE" and m["model"] != "BUY_HOLD"), key=lambda m: m["net_sharpe"] or -999)
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "COMPLETED_LOCAL_SCREEN_AND_PAST_ONLY_SELECTION",
              "candidate_configurations": 25, "total_candidates_evaluated": 26, "evaluation_accounts": 54, "new_accounts_generated": 50,
              "reused_control_accounts": 4, "shadow_accounts": 24, "trained_models": 0,
              "primary": [m for m in metrics if m["model"] == cfg["primary"]], "all_metrics": metrics, "post_selected_best_base": best,
              "historical_point_target_met": any(m["meets_point_target"] for m in metrics if m["model"] != "BUY_HOLD"),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "最高基础夏普": best["net_sharpe"], "设置": best["model"], "预定滚动主方案": result["primary"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
