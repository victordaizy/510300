"""以现有日线中的日内和隔夜分解直接决定简单进出场。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.simple_price_entry_exit_v1 import simulate_policy, modes
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_simple_session_divergence_v1"
CONFIG = ROOT / "config/510300_simple_session_divergence_v1.json"


def specifications():
    records = []
    phase_names = {"PP": "隔夜日内同时上涨", "PN": "隔夜上涨日内下跌", "NP": "隔夜下跌日内上涨", "NN": "隔夜日内同时下跌"}
    for window in [5, 20, 60]:
        for phase, name in phase_names.items():
            records.append({"id": f"P{window}_{phase}", "window": window, "kind": "PHASE", "phase": phase, "name": f"{window}日{name}"})
    for window in [20, 60]:
        for direction in [1, -1]:
            name = "日内强于隔夜" if direction == 1 else "隔夜强于日内"
            records.append({"id": f"D{window}_{'INTRA' if direction==1 else 'NIGHT'}", "window": window,
                            "kind": "DIFFERENCE", "direction": direction, "name": f"{window}日{name}的标准化差异"})
    return records


def make_rule(data, item):
    window = item["window"]
    night = data.overnight_log.rolling(window).sum()
    intraday = data.intraday_log.rolling(window).sum()
    if item["kind"] == "PHASE":
        phase = item["phase"]
        first = night > 0 if phase[0] == "P" else night < 0
        second = intraday > 0 if phase[1] == "P" else intraday < 0
        state = first & second
        entry = state & state.shift(1, fill_value=False)
        exit_flag = (~state) & (~state.shift(1, fill_value=False))
        factor = intraday - night
    else:
        daily_difference = data.intraday_log - data.overnight_log
        scale = daily_difference.rolling(60).std(ddof=1) * np.sqrt(window)
        factor = daily_difference.rolling(window).sum() / scale.replace(0, np.nan)
        oriented = factor * item["direction"]
        entry = (oriented > 1) & (oriented.shift(1) > 1)
        exit_flag = (oriented < 0) & (oriented.shift(1) < 0)
    entry = entry.fillna(False) & data.feature_valid.fillna(False)
    return {"entry": entry.to_numpy(int), "exit": {1: exit_flag.fillna(False).to_numpy(bool)}}, factor


def freeze():
    require(not CONFIG.exists(), "日内隔夜简单策略已登记，不覆盖")
    old = json.loads((ROOT / "config/510300_simple_price_entry_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: v for k, v in old.items() if k not in ["candidate_specs", "candidate_names", "frozen_files"]}
    cfg.update({"study_id": "510300_SIMPLE_SESSION_DIVERGENCE_V1", "round": 25, "registered_at": now(), "primary": "D20_INTRA",
                "candidates": specifications(), "candidate_count": 16, "common_specification": {"cooldown": 2, "modes": {1: modes(.06, .08, days=60)}},
                "rules": "docs/510300_SIMPLE_SESSION_DIVERGENCE_V1.md"})
    paths = [Path(__file__), ROOT / "research/simple_price_entry_exit_v1.py", ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"]]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第25轮16个日内隔夜条件已登记。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "日内隔夜策略登记后的输入发生变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    rules = {}
    factors = {"date": data.date}
    for item in cfg["candidates"]:
        rules[item["id"]], factors[item["id"]] = make_rule(data, item)
    pd.DataFrame(factors).to_parquet(OUT / "factors.parquet", index=False)
    metrics, yearly, eras = [], [], []
    for cost_id, cost in cfg["costs"].items():
        accounts = {}
        for i, item in enumerate(cfg["candidates"], 1):
            key = item["id"]
            ledger, decisions, cycles = simulate_policy(data, dividends, cfg, cost, cfg["evaluation_start"], rules[key], cfg["common_specification"])
            save_account(OUT / "evaluation" / cost_id, key, ledger, decisions)
            cycles.to_csv(OUT / "evaluation" / cost_id / f"{key}_cycles.csv", index=False, encoding="utf-8-sig")
            accounts[key] = ledger
            print(f"{cost_id} 日内隔夜条件 {i}/16：{item['name']}，净夏普 {summarize(ledger,cfg)['net_sharpe']:.4f}", flush=True)
        benchmark = pd.read_parquet(ROOT / "reports/research/510300_simple_price_entry_exit_v1/evaluation" / cost_id / "BUY_HOLD_ledger.parquet")
        accounts["BUY_HOLD"] = benchmark
        base = summarize(benchmark, cfg)
        names = {item["id"]: item["name"] for item in cfg["candidates"]}
        for key, ledger in accounts.items():
            m = {"cost": cost_id, "model": key, "name": names.get(key, "买入持有"), **summarize(ledger, cfg)}
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
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "COMPLETED_SESSION_DIVERGENCE_HISTORICAL_SCREEN",
              "candidate_configurations": 16, "evaluation_accounts": 34, "new_accounts_generated": 32, "reused_control_accounts": 2,
              "trained_models": 0, "primary": [m for m in metrics if m["model"] == cfg["primary"]], "all_metrics": metrics,
              "post_selected_best_base": best, "historical_point_target_met": any(m["meets_point_target"] for m in metrics if m["model"] != "BUY_HOLD"),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED_ALREADY_OBSERVED_HISTORY", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "最高基础夏普": best["net_sharpe"], "方法": best["name"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
