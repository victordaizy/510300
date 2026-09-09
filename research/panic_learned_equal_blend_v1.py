"""原学习退出和原急跌回升两个独立状态各占一半预算。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.simple_signal_blend_v1 import decision_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
CONFIG = ROOT / "config/510300_panic_learned_equal_blend_v1.json"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PRIMARY = "PANIC_LEARNED_HALF"


def panic_folder(period):
    return ROOT / ("reports/research/510300_simple_volume_reversal_v1/evaluation" if period == "evaluation"
                   else "reports/research/510300_simple_panic_earlier_history_v1")


def freeze():
    old = json.loads((ROOT / "config/510300_trend_learned_equal_blend_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]}
    cfg.update(study_id="510300_PANIC_LEARNED_EQUAL_BLEND_V1", round=46, registered_at=now(), candidate_configurations=1,
               primary=PRIMARY, fixed_weights=[.5, .5], state_cost="BASE", new_model_fits=0,
               rules="docs/510300_PANIC_LEARNED_EQUAL_BLEND_V1.md", position_impact=0)
    paths = [Path(__file__), ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/simple_signal_blend_v1.py",
             ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"]]
    for period in ["evaluation", "earlier_diagnostic"]:
        paths += [panic_folder(period) / "BASE/V6_PANIC_RECOVERY_decisions.parquet", P32 / period / "BASE/REARM_RIDGE_decisions.parquet"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第46轮一个急跌回升与原学习退出各半的组合已登记，权重固定。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "固定急跌组合登记内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    main, early, yearly, eras, counts = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        panic = decision_state(frame, pd.read_parquet(panic_folder(period) / "BASE/V6_PANIC_RECOVERY_decisions.parquet"))
        learned = decision_state(frame, pd.read_parquet(P32 / period / "BASE/REARM_RIDGE_decisions.parquet"))
        target = .5 * panic + .5 * learned
        anchor = int(np.flatnonzero(frame.date >= pd.Timestamp(start))[0]) - 1
        require(np.isfinite(target[anchor:-1]).all(), "固定急跌组合状态不完整，不能把缺失状态当空仓")
        require(set(np.unique(target[anchor:-1])).issubset({0., .5, 1.}), "固定急跌组合超出三档预算")
        pd.DataFrame({"date": frame.date, "panic_state": panic, "learned_state": learned, "target": target}).to_parquet(OUT / f"{period}_states.parquet", index=False)
        for a in [0., 1.]:
            for b in [0., 1.]:
                counts.append({"period": period, "panic_state": a, "learned_state": b, "target": .5 * (a + b),
                               "decision_days": int(((panic[anchor:-1] == a) & (learned[anchor:-1] == b)).sum())})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, PRIMARY, targets=target, event_mask=np.ones(len(frame), bool))
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "固定急跌组合账户结算失败")
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "急跌回升与原学习退出各半"}
            for key, path, name in [("REARM_RIDGE", P32 / period / cost_id / "REARM_RIDGE_ledger.parquet", "原线性退出及等待新机会"),
                                    ("PANIC_ONLY", panic_folder(period) / cost_id / "V6_PANIC_RECOVERY_ledger.parquet", "原短期高波动急跌后回升"),
                                    ("BUY_HOLD", P32 / period / cost_id / "BUY_HOLD_ledger.parquet", "买入持有")]:
                saved = pd.read_parquet(path)
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "固定急跌组合的对照账户日历不完整")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                destination.append(m)
                if period == "evaluation":
                    for year, group in saved.groupby(saved.date.dt.year):
                        yearly.append({"cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[(saved.date >= left) & (saved.date <= right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：一个固定组合和三个原样对照已完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("state_counts.csv", counts)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "PANIC_LEARNED_EQUAL_BLEND_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": 8, "new_accounts_generated": 2, "reused_control_accounts": 6, "earlier_diagnostic_accounts": 8,
              "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6, "new_model_fits": 0, "new_reference_accounts": 0,
              "all_metrics": main, "earlier_diagnostics": early, "state_counts": counts, "primary": [m for m in main if m["model"] == PRIMARY],
              "post_selected_best_base": next(m for m in main if m["model"] == PRIMARY and m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] == PRIMARY),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": result["primary"], "较早": [m for m in early if m["model"] == PRIMARY], "共同状态": counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
