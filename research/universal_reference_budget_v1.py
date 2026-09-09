"""仅复用保存的连续参考，计算一个通用混合预算及四个实际账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.universal_reference_budget_inputs_v1 import budget_frame
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_universal_reference_budget_v1"
CONFIG = ROOT / "config/510300_universal_reference_budget_v1.json"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P84 = ROOT / "reports/research/510300_two_policy_wealth_budget_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
PRIMARY = "UNIVERSAL_REFERENCE_BUDGET"
NAME = "全部固定比例增长汇总预算"
CONTROLS = {"CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "原第91轮连续最小方差"), "TWO_POLICY_WEALTH_BUDGET": (P84, "原第84轮单策略净值比例"), "PANIC_LEARNED_HALF": (P46, "原两条策略各半"), "BUY_HOLD": (P46, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "通用混合预算已经登记")
    old = json.loads((ROOT / "config/510300_continuous_reference_min_variance_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days", "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band", "reference_start"]}
    cfg.update(study_id="510300_UNIVERSAL_REFERENCE_BUDGET_V1", round=97, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
        initial_mixture_measure="UNIFORM_OVER_ALL_TWO_POLICY_CONSTANT_REBALANCED_WEIGHTS", growth_clock="EACH_OBSERVED_CONTINUOUS_REFERENCE_CLOSE_NEXT_OPEN",
        numerical_method="POSITIVE_BERNSTEIN_COEFFICIENTS_EXACT_INTEGRALS_COMMON_RESCALING", state_cost="BASE", missing_growth_rule="NO_VIEW_INCOMPLETE_CUMULATIVE_HISTORY",
        rules="docs/510300_UNIVERSAL_REFERENCE_BUDGET_V1.md", new_model_fits=0, new_reference_accounts=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED", position_impact=0)
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "通用混合预算必要测试未通过")
    paths = [Path(__file__), ROOT / "research/universal_reference_budget_inputs_v1.py", ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "tests/test_universal_reference_budget_v1.py", OUT / "tests_receipt.json", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"], ROOT / "docs/510300_TWO_POLICY_RISK_BUDGET_V1.md",
        ROOT / "config/510300_continuous_reference_min_variance_v1.json", P91 / "evaluation_factors.parquet"]
    for model in ["PANIC_ONLY", "REARM_RIDGE"]:
        paths.extend(P91 / "continuous_references/BASE" / f"{model}_{kind}.parquet" for kind in ["ledger", "decisions"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{model}_ledger.parquet" for model, (parent, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第97轮单一通用混合设置已冻结，不新增预测训练或参考账户。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "通用混合冻结输入改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    source = pd.read_parquet(P91 / "evaluation_factors.parquet")
    require(pd.DatetimeIndex(source.date).equals(pd.DatetimeIndex(data.date)), "连续参考因子与评价价格日历不同")
    first_ref = int(np.flatnonzero(data.date.ge(cfg["reference_start"]))[0])
    returns = source[["panic_reference_return", "learned_reference_return"]].to_numpy(float)
    states = source[["panic_state", "learned_state"]].to_numpy(float)
    factors = budget_frame(data.date, returns, states, first_ref)
    factors.to_parquet(OUT / "continuous_budget_factors.parquet", index=False)
    factors.to_csv(OUT / "连续参考增长与全部每日预算.csv", index=False, encoding="utf-8-sig")
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main), ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"], earlier)]:
        local = factors.iloc[:len(frame)]
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, PRIMARY, targets=local.target.to_numpy(float), event_mask=np.ones(len(frame), bool))
            decisions = decisions.merge(local.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "通用预算账户未完整结算")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()), "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()), "no_view_target_origins": int(decisions.reference_weight.isna().sum()), "mean_exposure": float(ledger.exposure.mean()),
                "minimum_panic_budget": float(decisions.panic_budget.min()), "maximum_panic_budget": float(decisions.panic_budget.max()), "mean_panic_budget": float(decisions.panic_budget.mean())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "新旧完整账户日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"]-bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：新完整账户及四个保存对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "UNIVERSAL_REFERENCE_BUDGET_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": 10, "new_accounts_generated": 2, "reused_control_accounts": 8, "earlier_diagnostic_accounts": 10, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8,
        "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier, "primary": primary, "account_coverage": coverage,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"), "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主结果": primary, "较早结果": [m for m in earlier if m["model"] == PRIMARY], "覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
