"""冻结价格确认分批进入和同执行满仓对照，复用原参考信号。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.simple_signal_blend_v1 import decision_state
from research.staged_entry_inputs_v1 import reference_entry_frame, staged_targets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_staged_entry_v1"
CONFIG = ROOT / "config/510300_staged_entry_v1.json"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
P76 = ROOT / "reports/research/510300_two_policy_risk_budget_v1"
PRIMARY = "PRICE_CONFIRMED_STAGED_ENTRY"
EXECUTION_CONTROL = "FULL_REFERENCE_EXECUTION"
CONTROLS = {"REARM_RIDGE": (P32, "原学习退出账户"), "PANIC_LEARNED_HALF": (P46, "原急跌与学习各半"),
    "TWO_POLICY_RISK_BUDGET": (P76, "第76轮风险预算"), "BUY_HOLD": (P32, "买入持有")}
COEFFICIENTS = ROOT / "deliverables/510300下午提前进入_第75轮_20260908/沿用的每月八项模型中文规则.md"


def freeze():
    require(not CONFIG.exists(), "分批进入已冻结，不覆盖")
    parent = json.loads((ROOT / "config/510300_panic_learned_equal_blend_v1.json").read_text(encoding="utf-8"))
    cfg = {k: parent[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]}
    cfg.update(study_id="510300_PRICE_CONFIRMED_STAGED_ENTRY_V1", round=81, registered_at=now(), primary=PRIMARY,
        candidate_configurations=1, initial_target=.5, confirmed_target=1., reference_state_cost="BASE", reference_exits="UNCHANGED_SAVED_BASE_PATH",
        new_model_fits=0, new_reference_accounts=0, rules="docs/510300_STAGED_ENTRY_V1.md", execution_control=EXECUTION_CONTROL,
        previous_goal_turn_classification="PROGRESS_ROUND80_COMPLETED_AND_DELIVERED", independent_validation="NOT_ESTABLISHED",
        position_impact=0, goal_achieved=False)
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0, "分批进入关键测试未通过")
    paths = [Path(__file__), ROOT / "research/staged_entry_inputs_v1.py", ROOT / "research/event_clock_account_v1.py",
        ROOT / "research/simple_signal_blend_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"], ROOT / "tests/test_staged_entry_v1.py",
        ROOT / "tests/test_median_continuation_v1.py", OUT / "tests_receipt.json", COEFFICIENTS,
        ROOT / "config/510300_research_authority_v6.json", ROOT / "config/510300_rearmed_session_exit_v1.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        paths.append(P32 / period / "BASE/REARM_RIDGE_decisions.parquet")
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{model}_ledger.parquet" for model, (parent, name) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in dict.fromkeys(paths)]
    write_json(CONFIG, cfg, exclusive=True)
    print("第81轮一个分批进入设置及同执行满仓对照已冻结，尚无新账户收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "分批进入冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    main, early, yearly, coverage, stages = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        first = int(np.flatnonzero(frame.date >= pd.Timestamp(start))[0])
        reference = pd.read_parquet(P32 / period / "BASE/REARM_RIDGE_ledger.parquet")
        state = decision_state(frame, pd.read_parquet(P32 / period / "BASE/REARM_RIDGE_decisions.parquet"))
        f = staged_targets(reference_entry_frame(frame, reference, first), state, first)
        f.to_parquet(OUT / f"{period}_factors.parquet", index=False)
        f.to_csv(OUT / f"{period}_factors.csv", index=False, encoding="utf-8-sig")
        used = f.iloc[first-1:-1]
        stages.append({"period": period, "half_target_days": int(used.target.eq(.5).sum()), "full_target_days": int(used.target.eq(1).sum()),
            "zero_target_days": int(used.target.eq(0).sum()), "no_view_days": int(used.target.isna().sum()),
            "reference_actual_buys": int(reference.filled_quantity.gt(0).sum()), "upgrades": int(used.price_upgrade.sum())})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            accounts, names = {}, {}
            for model, target, name in [(PRIMARY, f.target.to_numpy(float), "先半仓、回升确认后补足"),
                (EXECUTION_CONTROL, state, "同执行规则满仓参考对照")]:
                ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, model, targets=target, event_mask=np.ones(len(frame), bool))
                decisions = decisions.merge(f.rename(columns={"date": "origin", "target": "staged_target"}), on="origin", how="left", validate="one_to_one")
                save_account(folder, model, ledger, decisions)
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "分批或执行对照账户结算失败")
                coverage.append({"period": period, "cost": cost_id, "model": model, "holding_closes": int(ledger.shares.gt(0).sum()),
                    "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                    "additional_buy_trades": int((ledger.filled_quantity.gt(0) & ledger.shares_before.gt(0)).sum()),
                    "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()), "mean_exposure": float(ledger.exposure.mean())})
                accounts[model], names[model] = ledger, name
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts[PRIMARY].date)), "分批进入和对照账户日期不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"]-bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                dest.append(m)
                if model in [PRIMARY, EXECUTION_CONTROL]:
                    for year, group in saved.groupby(saved.date.dt.year):
                        yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
            print(f"{period}／{cost_id}：分批净夏普{summarize(accounts[PRIMARY], cfg)['net_sharpe']}；同执行满仓{summarize(accounts[EXECUTION_CONTROL], cfg)['net_sharpe']}。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("account_coverage.csv", coverage), ("stage_coverage.csv", stages)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    write_json(OUT / "result.json", {"study_id": cfg["study_id"], "completed_at": now(), "status": "STAGED_ENTRY_ACCOUNTS_COMPLETE",
        "candidate_configurations": 1, "evaluation_accounts": 12, "new_accounts_generated": 4, "new_execution_control_accounts": 2, "reused_control_accounts": 8,
        "earlier_diagnostic_accounts": 12, "new_earlier_diagnostic_accounts": 4, "new_earlier_execution_control_accounts": 2, "reused_earlier_accounts": 8,
        "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": early, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"), "account_coverage": coverage, "stage_coverage": stages,
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    print(json.dumps({"阶段": stages, "账户覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
