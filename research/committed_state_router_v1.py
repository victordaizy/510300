"""每次实际买入锁定基础策略，直到其退出，再重新选择。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.simple_price_entry_exit_v1 import simulate_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_committed_state_router_v1"
CONFIG = ROOT / "config/510300_committed_state_router_v1.json"
P39 = ROOT / "reports/research/510300_market_state_signal_router_v1"
PRIMARY = "COMMITTED_ROUTER"


def make_rule(states):
    a, b = states.trend_state.to_numpy(float), states.learned_state.to_numpy(float)
    strong, known = states.strong_market.to_numpy(bool), states.selector_available.to_numpy(bool)
    selected = np.where(strong, a, b)
    entry = np.where(known & np.isfinite(selected) & (selected == 1), np.where(strong, 1, 2), 0)
    return {"entry": entry, "exit": {1: np.isfinite(a) & (a == 0), 2: np.isfinite(b) & (b == 0)}}


def freeze():
    old = json.loads((ROOT / "config/510300_market_state_signal_router_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    cfg.update(study_id="510300_COMMITTED_STATE_ROUTER_V1", round=40, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
               specification={"cooldown": 0, "modes": {i: {"loss": None, "trail": None, "take": None, "days": None} for i in [1, 2]}},
               rules="docs/510300_COMMITTED_STATE_ROUTER_V1.md", new_model_fits=0, position_impact=0)
    paths = [Path(__file__), ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"],
             ROOT / "docs/510300_MARKET_STATE_SIGNAL_ROUTER_V1.md", ROOT / "tests/test_committed_state_router_v1.py"]
    paths += [P39 / f"{period}_states.parquet" for period in ["evaluation", "earlier_diagnostic"]]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第40轮一个持仓周期锁定策略的方案已登记，不新增模型训练。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "周期锁定策略登记内容变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    main, early, yearly, eras, mode_counts = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        states = pd.read_parquet(P39 / f"{period}_states.parquet")
        require(pd.DatetimeIndex(states.date).equals(pd.DatetimeIndex(frame.date)), "锁定策略状态与价格日期不同")
        rule = make_rule(states)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_policy(frame, dividends, cfg, cost, start, rule, cfg["specification"])
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "锁定策略账户结算失败")
            if len(cycles):
                require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "锁定策略违反买入次日可卖")
            for mode in [1, 2]:
                selected = cycles[cycles["mode"] == mode]
                mode_counts.append({"period": period, "cost": cost_id, "mode": mode, "name": "趋势基础策略" if mode == 1 else "日内强弱学习退出基础策略",
                                    "completed_cycles": len(selected), "actual_cycle_net_profit_cny": float(selected.net_profit_cny.sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "每次进入选定策略，退出后再选"}
            for key, source_key, name in [("STATE_ROUTER", "STATE_ROUTER", "每天按市场状态改选策略"), ("REARM_RIDGE", "REARM_RIDGE", "原线性退出＋等待新机会"),
                                          ("TREND_ONLY", "TREND_ONLY", "原趋势与震荡反弹"), ("BUY_HOLD", "BUY_HOLD", "买入持有")]:
                saved = pd.read_parquet(P39 / period / cost_id / f"{source_key}_ledger.parquet")
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "锁定策略评价日不完整")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                dest.append(m)
                if period == "evaluation":
                    for year, group in saved.groupby(saved.date.dt.year):
                        yearly.append({"cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[(saved.date >= left) & (saved.date <= right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：周期锁定方案及四个对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("mode_cycle_results.csv", mode_counts)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "COMMITTED_STATE_ROUTER_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": 10, "new_accounts_generated": 2, "reused_control_accounts": 8, "earlier_diagnostic_accounts": 10,
              "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8, "new_model_fits": 0, "all_metrics": main, "earlier_diagnostics": early,
              "primary": [m for m in main if m["model"] == PRIMARY], "mode_cycle_results": mode_counts,
              "post_selected_best_base": next(m for m in main if m["model"] == PRIMARY and m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] == PRIMARY),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "新方案": result["primary"], "较早同规则": [m for m in early if m["model"] == PRIMARY], "各模式实际周期": mode_counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
