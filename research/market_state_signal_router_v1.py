"""按当天可知的市场状态选择一个既有信号，重算完整账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.simple_signal_blend_v1 import decision_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_market_state_signal_router_v1"
CONFIG = ROOT / "config/510300_market_state_signal_router_v1.json"
P31 = ROOT / "reports/research/510300_learned_cycle_exit_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P34 = ROOT / "reports/research/510300_trend_learned_equal_blend_v1"
PRIMARY = "STATE_ROUTER"


def freeze():
    old = json.loads((ROOT / "config/510300_simple_signal_blend_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                               "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]}
    cfg.update({"study_id": "510300_MARKET_STATE_SIGNAL_ROUTER_V1", "round": 39, "registered_at": now(),
                "candidate_configurations": 1, "primary": PRIMARY, "selector_sma120_threshold": 0., "selector_efficiency20_threshold": .3, "state_cost": "BASE", "new_model_fits": 0,
                "rules": "docs/510300_MARKET_STATE_SIGNAL_ROUTER_V1.md", "position_impact": 0})
    paths = [Path(__file__), ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/simple_signal_blend_v1.py",
             ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"]]
    for period in ["evaluation", "earlier_diagnostic"]:
        paths += [P31 / period / "BASE/S1_TREND_REBOUND__CLOSE_NEXT_OPEN_decisions.parquet", P32 / period / "BASE/REARM_RIDGE_decisions.parquet"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第39轮一个市场状态选择方案已登记，不新增训练或搜索状态阈值。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for p in cfg["frozen_files"]:
        require(digest(ROOT / p["path"]) == p["sha256"], "市场状态选择登记文件变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    div = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    main, early, yearly, eras, state_counts = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        a = decision_state(frame, pd.read_parquet(P31 / period / "BASE/S1_TREND_REBOUND__CLOSE_NEXT_OPEN_decisions.parquet"))
        b = decision_state(frame, pd.read_parquet(P32 / period / "BASE/REARM_RIDGE_decisions.parquet"))
        valid = np.isfinite(frame[["sma120", "efficiency20"]].to_numpy(float)).all(axis=1)
        strong = (frame.sma120.to_numpy(float) > cfg["selector_sma120_threshold"]) & (frame.efficiency20.to_numpy(float) > cfg["selector_efficiency20_threshold"])
        target = np.where(valid, np.where(strong, a, b), np.nan)
        anchor = int(np.flatnonzero(frame.date >= pd.Timestamp(start))[0]) - 1
        require(np.isfinite(target[anchor:-1]).all(), "组合所需收盘状态缺失")
        require(set(np.unique(target[anchor:-1])).issubset({0., 1.}), "选择账户目标超出预定零或一")
        pd.DataFrame({"date": frame.date, "trend_state": a, "learned_state": b, "strong_market": strong, "selector_available": valid, "target": target}).to_parquet(OUT / f"{period}_states.parquet", index=False)
        for value in [0., 1.]:
            state_counts.append({"period": period, "target": value, "decision_days": int((target[anchor:-1] == value).sum())})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions = simulate_event_account(frame, div, cfg, cost, start, PRIMARY, targets=target, event_mask=np.ones(len(frame), bool))
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "组合账户结算不完整")
            accounts = {PRIMARY: ledger}
            names = {PRIMARY: "按强趋势及其余状态选择已有策略"}
            controls = [("EQUAL_HALF", P34 / period / cost_id / "TREND_LEARNED_HALF_ledger.parquet", "原趋势与学习退出各半"),
                        ("REARM_RIDGE", P32 / period / cost_id / "REARM_RIDGE_ledger.parquet", "新线性退出＋等待新机会"),
                        ("TREND_ONLY", P31 / period / cost_id / "S1_TREND_REBOUND__CLOSE_NEXT_OPEN_ledger.parquet", "原趋势与震荡策略"),
                        ("BUY_HOLD", P32 / period / cost_id / "BUY_HOLD_ledger.parquet", "买入持有")]
            for key, path, name in controls:
                saved = pd.read_parquet(path)
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "组合完整评价日期不同")
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
            pd.DataFrame({"date": ledger.date, **{key: saved.net_return.to_numpy() for key, saved in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：市场状态选择及四个原样对照已完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("state_counts.csv", state_counts)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "MARKET_STATE_SIGNAL_ROUTER_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": 10, "new_accounts_generated": 2, "reused_control_accounts": 8, "earlier_diagnostic_accounts": 10,
              "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8, "new_model_fits": 0,
              "all_metrics": main, "earlier_diagnostics": early, "state_counts": state_counts,
              "primary": [m for m in main if m["model"] == PRIMARY],
              "post_selected_best_base": next(m for m in main if m["model"] == PRIMARY and m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] == PRIMARY),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "新状态选择": result["primary"], "较早同规则": [m for m in early if m["model"] == PRIMARY]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
