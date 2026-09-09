"""复用原规则和已成熟模型，连续积累参考历史后评价新资金账户。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.two_policy_min_variance_inputs_v1 import budget_frame
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.panic_learned_equal_blend_v1 import P32
from research.simple_signal_blend_v1 import decision_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_continuous_interaction_budget_v1"
CONFIG = ROOT / "config/510300_continuous_interaction_budget_v1.json"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P102 = ROOT / "reports/research/510300_profit_drawdown_interaction_v1"
PRIMARY = "CONTINUOUS_INTERACTION_BUDGET"
NAME = "九项交互学习参考与原急跌的最小方差预算"
CONTROLS = {"CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "原第91轮连续风险预算"), "PROFIT_DRAWDOWN_INTERACTION_EXIT": (P102, "原第102轮九项单策略"), "BUY_HOLD": (P32, "买入持有")}
from research.profit_drawdown_interaction_inputs_v1 import InteractionExitController, FEATURES
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules as learned_rules


def reference_factors(data, references, first, cfg):
    returns=np.full((len(data),2),np.nan);states=[]
    for column,model in enumerate(["PANIC_ONLY","INTERACTION_REFERENCE"]):
        ledger,decisions=references[model]
        saved=ledger[ledger.date.le(data.date.iloc[-1])]
        chosen=decisions[decisions.origin.lt(data.date.iloc[-1])]
        require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(data.date.iloc[first:])),"连续参考账户完整日历缺失")
        require(pd.DatetimeIndex(chosen.origin).equals(pd.DatetimeIndex(data.date.iloc[first-1:-1])),"连续参考判断日期不完整")
        require(pd.DatetimeIndex(chosen.execution_date).equals(pd.DatetimeIndex(data.date.iloc[first:])),"连续参考意向不是下一开盘")
        returns[first:,column]=saved.net_return.to_numpy(float)
        states.append(decision_state(data,chosen))
    f=budget_frame(data.date,returns,np.column_stack(states),first,cfg["risk_window"])
    f["panic_reference_return"],f["learned_reference_return"]=returns[:,0],returns[:,1]
    return f


def read_models(path):
    models = json.loads(Path(path).read_text(encoding="utf-8"))["models"]
    require(isinstance(models, list) and len(models) > 0, "九项保存模型必须是按时间排列的列表")
    indexes = [m["fit_index"] for m in models]
    require(indexes == sorted(set(indexes)), "九项保存模型日程重复或不递增")
    for m in models:
        require(m["latest_exit_index"] is None or m["latest_exit_index"] <= m["fit_index"], "模型使用未来成熟样本")
        if m["status"] == "FIT_COMPLETE":
            require(m["model"]["kind"] == "PROFIT_DRAWDOWN_INTERACTION_RIDGE" and m["model"]["features"] == FEATURES, "没有读取冻结的九项交互模型")
    return models


def freeze():
    require(not CONFIG.exists(), "连续交互预算已登记，不重复冻结")
    old = json.loads((ROOT / "config/510300_continuous_reference_min_variance_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days", "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band", "risk_window", "reference_start", "learned_spec", "confirmation_days"]}
    cfg.update(study_id="510300_CONTINUOUS_INTERACTION_BUDGET_V1", round=103, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
        saved_models=str((P102 / "saved_models.json").relative_to(ROOT)), risk_clock="MONTH_FIRST_COMPLETE_CLOSE_NEXT_OPEN", state_cost="BASE", initial_budgets=[.5, .5], new_model_fits=0, new_reference_accounts=1,
        rules="docs/510300_CONTINUOUS_INTERACTION_BUDGET_V1.md", previous_goal_turn_classification="PROGRESS_ROUND102_COMPLETED_AND_DELIVERED", goal_achieved=False, independent_validation="NOT_ESTABLISHED", position_impact=0)
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 4, "连续九项交互必要测试未通过")
    paths = [Path(__file__), ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"],
        ROOT / "research/profit_drawdown_interaction_inputs_v1.py", ROOT / "research/rearmed_cycle_exit_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py", ROOT / "research/simple_price_entry_exit_v1.py",
        ROOT / "research/two_policy_min_variance_inputs_v1.py", ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/simple_signal_blend_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "config/510300_continuous_reference_min_variance_v1.json", ROOT / "config/510300_profit_drawdown_interaction_v1.json", ROOT / "tests/test_continuous_interaction_budget_v1.py", ROOT / "tests/test_profit_drawdown_interaction_v1.py", OUT / "tests_receipt.json"]
    paths.extend(P91 / "continuous_references/BASE" / f"PANIC_ONLY_{kind}.parquet" for kind in ["ledger", "decisions"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{model}_ledger.parquet" for model, (parent, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第103轮一个连续交互风险预算已冻结，尚无新参考或评价收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "风险预算冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    stored = read_models(ROOT / cfg["saved_models"])
    controller=InteractionExitController(data,stored,cfg["confirmation_days"])
    learned,learned_decisions,learned_cycles=simulate_rearmed_exit(data,dividends,cfg,cfg["costs"]["BASE"],cfg["reference_start"],learned_rules(data)["D60_INTRA"],cfg["learned_spec"],controller)
    panic = pd.read_parquet(P91 / "continuous_references/BASE/PANIC_ONLY_ledger.parquet")
    panic_decisions = pd.read_parquet(P91 / "continuous_references/BASE/PANIC_ONLY_decisions.parquet")
    references = {"PANIC_ONLY": (panic, panic_decisions), "INTERACTION_REFERENCE": (learned, learned_decisions)}
    require(learned.accounting_error.abs().max() < 1e-6 and not learned.terminal_unliquidated.iloc[-1], "新连续学习参考未完整结算")
    save_account(OUT / "continuous_references/BASE", "INTERACTION_REFERENCE", learned, learned_decisions)
    learned_cycles.to_csv(OUT / "continuous_references/BASE/INTERACTION_REFERENCE_cycles.csv", index=False, encoding="utf-8-sig")
    reference_first=int(np.flatnonzero(data.date>=pd.Timestamp(cfg["reference_start"]))[0])
    print("一个新连续学习参考完成；原急跌参考与九项模型均复用，没有新拟合。",flush=True)
    main, earlier, yearly, eras, updates, coverage = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        first = int(np.flatnonzero(frame.date >= pd.Timestamp(start))[0])
        factors=reference_factors(frame,references,reference_first,cfg)
        require(np.isfinite(factors.target.iloc[first-1:-1]).all(),"新评价期间连续参考意向缺失")
        factors.to_parquet(OUT / f"{period}_factors.parquet", index=False)
        factors.to_csv(OUT / f"{period}_factors.csv", index=False, encoding="utf-8-sig")
        for row in factors[factors.risk_update_scheduled].to_dict("records"):
            updates.append({"period": period, **row})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, PRIMARY,
                targets=factors.target.to_numpy(float), event_mask=np.ones(len(frame), bool))
            decisions = decisions.merge(factors.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "风险预算完整账户结算失败")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()),
                "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                "no_view_target_origins": int(decisions.reference_weight.isna().sum()), "mean_exposure": float(ledger.exposure.mean())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "新旧账户日历不一致")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：连续历史新账户和三个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras),
        ("risk_update_records.csv", updates), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    write_json(OUT / "result.json", {"study_id": cfg["study_id"], "completed_at": now(), "status": "CONTINUOUS_INTERACTION_BUDGET_ACCOUNTS_COMPLETE",
        "candidate_configurations": 1, "evaluation_accounts": 8, "new_accounts_generated": 2, "reused_control_accounts": 6,
        "earlier_diagnostic_accounts": 8, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6,
        "new_model_fits": 0, "new_reference_accounts": 1, "all_metrics": main, "earlier_diagnostics": earlier, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"), "risk_update_count": len(updates), "account_coverage": coverage,
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    print(json.dumps({"主结果": primary, "较早结果": [m for m in earlier if m["model"]==PRIMARY], "账户覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
