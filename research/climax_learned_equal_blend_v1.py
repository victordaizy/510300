"""放量收强回升与原学习退出各半，所有原信号参数保持。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.simple_price_entry_exit_v1 import simulate_policy
from research.simple_signal_blend_v1 import decision_state
from research.simple_volume_reversal_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_climax_learned_equal_blend_v1"
CONFIG = ROOT / "config/510300_climax_learned_equal_blend_v1.json"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P28 = ROOT / "reports/research/510300_simple_signal_blend_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
KEY, NAME = "CLIMAX_LEARNED_HALF", "放量收强回升与原学习退出各半"


def climax_path(period, cost, artifact="ledger"):
    if period == "evaluation":
        parent = ROOT / "reports/research/510300_simple_volume_reversal_v1/evaluation" / cost
    else:
        parent = P28 / "earlier_experts" if cost == "BASE" else OUT / "new_earlier_control" / cost
    return parent / f"V1_CLIMAX_RECOVERY_{artifact}.parquet"


def freeze():
    old = json.loads((ROOT / "config/510300_panic_learned_equal_blend_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]}
    old28 = json.loads((ROOT / "config/510300_simple_signal_blend_v1.json").read_text(encoding="utf-8"))
    require(all(cfg[k] == old28[k] for k in cfg), "较早已保存信号来源的共同账户或日期口径不同")
    volume = json.loads((ROOT / "config/510300_simple_volume_reversal_v1.json").read_text(encoding="utf-8"))
    cfg.update(study_id="510300_CLIMAX_LEARNED_EQUAL_BLEND_V1", round=51, registered_at=now(), candidate_configurations=1,
               primary=KEY, names={KEY: NAME}, fixed_weights=[.5, .5], state_cost="BASE", new_model_fits=0,
               volume_specification=volume["candidate_specs"]["V1_CLIMAX_RECOVERY"],
               rules="docs/510300_CLIMAX_LEARNED_EQUAL_BLEND_V1.md", position_impact=0)
    paths = [Path(__file__), ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/simple_signal_blend_v1.py",
             ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/simple_volume_reversal_v1.py",
             ROOT / "config/510300_simple_signal_blend_v1.json", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"]]
    for period in ["evaluation", "earlier_diagnostic"]:
        paths += [climax_path(period, "BASE", "decisions"), P32 / period / "BASE/REARM_RIDGE_decisions.parquet", P46 / f"{period}_states.parquet"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第51轮一个固定各半组合已登记，只补一条较早压力费用原对照。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "放量回升组合登记内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    earlier = data[data.date <= cfg["earlier_terminal"]].copy()
    stress, stress_decisions, stress_cycles = simulate_policy(earlier, dividends, cfg, cfg["costs"]["STRESS"], cfg["earlier_start"],
                                                            make_rules(earlier)[0]["V1_CLIMAX_RECOVERY"], cfg["volume_specification"])
    save_account(OUT / "new_earlier_control/STRESS", "V1_CLIMAX_RECOVERY", stress, stress_decisions)
    stress_cycles.to_csv(OUT / "new_earlier_control/STRESS/V1_CLIMAX_RECOVERY_cycles.csv", index=False, encoding="utf-8-sig")
    require(stress.accounting_error.abs().max() < 1e-6 and not stress.terminal_unliquidated.iloc[-1], "较早放量对照未完成结算")
    main, early, yearly, eras, counts, source_stats = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", earlier, cfg["earlier_start"], early)]:
        climax = decision_state(frame, pd.read_parquet(climax_path(period, "BASE", "decisions")))
        learned = decision_state(frame, pd.read_parquet(P32 / period / "BASE/REARM_RIDGE_decisions.parquet"))
        existing = pd.read_parquet(P46 / f"{period}_states.parquet")
        require(pd.DatetimeIndex(existing.date).equals(pd.DatetimeIndex(frame.date)), "原组合状态日历不同")
        panic = existing.panic_state.to_numpy(float)
        target = .5 * climax + .5 * learned
        anchor = int(np.flatnonzero(frame.date >= pd.Timestamp(start))[0]) - 1
        require(np.isfinite(np.column_stack([climax, learned, panic])[anchor:-1]).all(), "比较所需完整状态缺失")
        require(np.isin(target[anchor:-1], [0., .5, 1.]).all(), "固定各半组合预算越界")
        pd.DataFrame({"date": frame.date, "climax_state": climax, "learned_state": learned, "old_panic_state": panic,
                      "target": target}).to_parquet(OUT / f"{period}_states.parquet", index=False)
        for a in [0., 1.]:
            for b in [0., 1.]:
                for c in [0., 1.]:
                    counts.append({"period": period, "climax_state": a, "learned_state": b, "old_panic_state": c,
                                   "target": .5 * (a + b), "decision_days": int(((climax[anchor:-1] == a) & (learned[anchor:-1] == b) & (panic[anchor:-1] == c)).sum())})
        for cost_id, cost in cfg["costs"].items():
            dest = OUT / period / cost_id
            ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, KEY, targets=target, event_mask=np.ones(len(frame), bool))
            save_account(dest, KEY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "放量学习组合账户未完成结算")
            accounts, names = {KEY: ledger}, {KEY: NAME}
            for key, path, name in [
                ("REARM_RIDGE", P32 / period / cost_id / "REARM_RIDGE_ledger.parquet", "原学习退出及等待新机会"),
                ("CLIMAX_ONLY", climax_path(period, cost_id), "原急跌后放量收强单策略"),
                ("OLD_PANIC_BLEND", P46 / period / cost_id / "PANIC_LEARNED_HALF_ledger.parquet", "原高波动急跌与学习退出各半"),
                ("OLD_SESSION_VOLUME", P28 / period / cost_id / "B4_SESSION_VOLUME_ledger.parquet", "原未加学习的日内强弱与放量回升各半"),
                ("BUY_HOLD", P32 / period / cost_id / "BUY_HOLD_ledger.parquet", "买入持有")]:
                saved = pd.read_parquet(path)
                saved.to_parquet(dest / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            source = accounts["CLIMAX_ONLY"]
            require((source.filled_quantity.gt(0).sum() == source.filled_quantity.lt(0).sum()) and not source.terminal_unliquidated.iloc[-1], "放量源周期未完整结束")
            source_stats.append({"period": period, "cost": cost_id, "source_completed_cycles": int(source.filled_quantity.gt(0).sum()),
                                 "source_trade_count": int(source.filled_quantity.ne(0).sum()),
                                 "climax_only_outside_old_two_signal_decision_days": int(((climax[anchor:-1] == 1) & (learned[anchor:-1] == 0) & (panic[anchor:-1] == 0)).sum())})
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "放量组合对照完整日历不同")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                intervals = [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"),
                             ("2024—终点", "2024-01-01", cfg["data_cutoff"])] if period == "evaluation" else [
                             ("2015—2016", "2015-01-01", "2016-12-31"), ("2017—2019", "2017-01-01", cfg["earlier_terminal"])]
                for label, left, right in intervals:
                    group = saved[(saved.date >= left) & (saved.date <= right)]
                    eras.append({"period": period, "cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：固定放量回升组合和五个对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("state_counts.csv", counts), ("source_statistics.csv", source_stats)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "CLIMAX_LEARNED_EQUAL_BLEND_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": 12, "new_accounts_generated": 2, "reused_control_accounts": 10,
              "earlier_diagnostic_accounts": 12, "new_earlier_diagnostic_accounts": 3, "new_earlier_candidate_accounts": 2,
              "new_earlier_control_accounts": 1, "reused_earlier_accounts": 9, "new_model_fits": 0, "new_reference_accounts": 0,
              "all_metrics": main, "earlier_diagnostics": early, "state_counts": counts, "source_statistics": source_stats,
              "primary": [m for m in main if m["model"] == KEY],
              "post_selected_best_base": next(m for m in main if m["model"] == KEY and m["cost"] == "BASE"),
              "goal_achieved": False, "independent_validation": False, "position_impact": 0}
    write_json(OUT / "result.json", result)
    print(json.dumps({"主评价": result["primary"], "较早历史": [m for m in early if m["model"] in [KEY, "CLIMAX_ONLY"]],
                      "来源事件统计": source_stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in {"freeze", "run"}:
        raise SystemExit("用法：python -m research.climax_learned_equal_blend_v1 freeze 或 run")
    freeze() if sys.argv[1] == "freeze" else run()
