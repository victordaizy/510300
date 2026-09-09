"""以成交量、突破和急跌收复直接检验六种简单策略。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.simple_price_entry_exit_v1 import simulate_policy, modes
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_simple_volume_reversal_v1"
CONFIG = ROOT / "config/510300_simple_volume_reversal_v1.json"
NAMES = {"V1_CLIMAX_RECOVERY": "急跌后放量收强", "V2_QUIET_PULLBACK": "上涨趋势中缩量回调转升",
         "V3_VOLUME_BREAKOUT": "放量突破二十日高点", "V4_SELLING_ABSORBED": "放量大跌后缩量回升",
         "V5_FAILED_BREAKDOWN": "跌破二十日低点后当日收复", "V6_PANIC_RECOVERY": "短期高波动急跌后回升"}


def specifications():
    return {"V1_CLIMAX_RECOVERY": {"cooldown": 2, "modes": {1: modes(.04, take=.06, days=10)}},
            "V2_QUIET_PULLBACK": {"cooldown": 2, "modes": {1: modes(.04, days=10)}},
            "V3_VOLUME_BREAKOUT": {"cooldown": 2, "modes": {1: modes(.06, .06, days=60)}},
            "V4_SELLING_ABSORBED": {"cooldown": 2, "modes": {1: modes(.03, days=5)}},
            "V5_FAILED_BREAKDOWN": {"cooldown": 2, "modes": {1: modes(.03, take=.05, days=5)}},
            "V6_PANIC_RECOVERY": {"cooldown": 2, "modes": {1: modes(.04, take=.06, days=10)}}}


def make_rules(data):
    w = data.wealth
    up = w > w.shift(1)
    volume = data.volume / data.volume.shift(1).rolling(20).mean()
    prior_low = w.shift(1).rolling(20).min()
    low = w * (data.low + data.dividend) / (data.close + data.dividend)
    r = data.total_simple
    bull = data.sma120 > 0
    rules = {}
    def add(key, entry, exit_flag):
        rules[key] = {"entry": (entry.fillna(False) & data.feature_valid.fillna(False)).to_numpy(int),
                      "exit": {1: exit_flag.fillna(False).to_numpy(bool)}}
    add("V1_CLIMAX_RECOVERY", (data.mom5.shift(1) < np.log(.97)) & up & (volume >= 1.5) & (data.close_location >= .75), data.z20 >= 0)
    add("V2_QUIET_PULLBACK", bull & (volume < .7) & (data.mom5 < np.log(.98)) & up, (~bull) | (data.rsi2 > 80))
    add("V3_VOLUME_BREAKOUT", (w > w.shift(1).rolling(20).max()) & (volume >= 1.5), w < w.shift(1).rolling(10).min())
    add("V4_SELLING_ABSORBED", (r.shift(1) <= -.02) & (volume.shift(1) >= 1.5) & up & (data.volume < data.volume.shift(1)), data.rsi2 > 70)
    add("V5_FAILED_BREAKDOWN", (low < prior_low) & (w > prior_low) & (data.close_location >= .6), data.z20 >= 0)
    add("V6_PANIC_RECOVERY", (data.mom5 < np.log(.95)) & up & (data.close_location >= .6) & (data.vol5 > data.vol60 * 1.5), data.z20 >= 0)
    return rules, volume, low


def freeze():
    require(not CONFIG.exists(), "量价策略已登记，不覆盖")
    old = json.loads((ROOT / "config/510300_simple_price_entry_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: v for k, v in old.items() if k not in ["candidate_specs", "candidate_names", "frozen_files"]}
    cfg.update({"study_id": "510300_SIMPLE_VOLUME_REVERSAL_V1", "round": 26, "registered_at": now(), "primary": "V5_FAILED_BREAKDOWN",
                "candidate_count": 6, "candidate_specs": specifications(), "candidate_names": NAMES, "rules": "docs/510300_SIMPLE_VOLUME_REVERSAL_V1.md"})
    paths = [Path(__file__), ROOT / "research/simple_price_entry_exit_v1.py", ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"]]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第26轮6个量价和低点收复规则已登记。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "量价策略输入发生变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    div = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    rules, volume, low = make_rules(data)
    pd.DataFrame({"date": data.date, "volume_vs_prior20": volume, "intraday_low_total_return_scale": low}).to_parquet(OUT / "factors.parquet", index=False)
    metrics, yearly, eras = [], [], []
    for cost_id, cost in cfg["costs"].items():
        accounts = {}
        for key, rule in rules.items():
            ledger, decisions, cycles = simulate_policy(data, div, cfg, cost, cfg["evaluation_start"], rule, cfg["candidate_specs"][key])
            save_account(OUT / "evaluation" / cost_id, key, ledger, decisions)
            cycles.to_csv(OUT / "evaluation" / cost_id / f"{key}_cycles.csv", index=False, encoding="utf-8-sig")
            accounts[key] = ledger
            print(f"{cost_id}：{NAMES[key]}，净夏普 {summarize(ledger,cfg)['net_sharpe']:.4f}", flush=True)
        benchmark = pd.read_parquet(ROOT / "reports/research/510300_simple_price_entry_exit_v1/evaluation" / cost_id / "BUY_HOLD_ledger.parquet")
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
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "COMPLETED_VOLUME_REVERSAL_HISTORICAL_SCREEN",
              "candidate_configurations": 6, "evaluation_accounts": 14, "new_accounts_generated": 12, "reused_control_accounts": 2,
              "trained_models": 0, "primary": [m for m in metrics if m["model"] == cfg["primary"]], "all_metrics": metrics, "post_selected_best_base": best,
              "historical_point_target_met": any(m["meets_point_target"] for m in metrics if m["model"] != "BUY_HOLD"),
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
