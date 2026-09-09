"""复用连续参考，条件回撤预算只计算一次并运行四个独立账户。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.conditional_drawdown_budget_inputs_v1 import budget_frame, prefix_factors
from research.intraday_overnight_increment_v1 import require, now, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_conditional_drawdown_budget_v1"
CONFIG = ROOT / "config/510300_conditional_drawdown_budget_v1.json"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P114 = ROOT / "reports/research/510300_within_cycle_exit_v1"
PRIMARY = "CONDITIONAL_DRAWDOWN_BUDGET"
NAME = "连续回撤资金预算"
CONTROLS = {"CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "第91轮连续参考最小方差"),
    "WITHIN_CYCLE_EXIT": (P114, "第114轮周期内退出"), "REARM_RIDGE": (P32, "原平均继续收益退出"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "条件回撤方案已经登记，不重复冻结")
    parent_path = ROOT / "config/510300_continuous_reference_min_variance_v1.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    cfg = {k: parent[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
        "weight_band", "risk_window", "reference_start", "panic_spec", "learned_spec", "confirmation_days", "saved_models"]}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 12, "条件回撤十二项必要测试未通过")
    cfg.update(study_id="510300_CONDITIONAL_DRAWDOWN_BUDGET_V1", round=124, registered_at=now(), primary=PRIMARY,
        candidate_configurations=1, confidence=.95, initial_budgets=[.5, .5], state_cost="BASE",
        risk_clock="MONTH_FIRST_COMPLETE_CLOSE_NEXT_OPEN", risk_path="UNCOMPOUNDED_SUM_WITH_ZERO_ANCHOR",
        tie_rule="NEAREST_PREVIOUS_WITHIN_OPTIMAL_INTERVAL", rules="docs/510300_CONDITIONAL_DRAWDOWN_BUDGET_V1.md",
        input_receipt="reports/research/510300_conditional_drawdown_budget_preflight_20260909/result.json",
        source_factors={period: str((P91 / f"{period}_factors.parquet").relative_to(ROOT)) for period in ["evaluation", "earlier_diagnostic"]},
        shared_prefix_optimized_once=True, new_model_fits=0, new_reference_accounts=0, source_budget_cny=0,
        independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUND123_COMPLETED_EIGHT_ACCOUNTS_AND_124_SOURCE_REVIEW")
    paths = [Path(__file__), parent_path, ROOT / "research/conditional_drawdown_budget_inputs_v1.py",
        ROOT / "research/conditional_drawdown_budget_optimizer_v1.py", ROOT / "research/two_policy_tail_loss_optimizer_v1.py",
        ROOT / "research/event_clock_account_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "tests/test_conditional_drawdown_budget_v1.py",
        OUT / "tests_receipt.json", ROOT / "config/510300_research_authority_v6.json", ROOT / "docs/510300_PANIC_LEARNED_EQUAL_BLEND_V1.md"]
    paths.extend(ROOT / cfg[key] for key in ["rules", "input_receipt", "features", "dividends"])
    receipt = json.loads((ROOT / cfg["input_receipt"]).read_text(encoding="utf-8"))
    for item in receipt["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "预检后连续参考来源改变")
        paths.append(ROOT / item["path"])
    for period in cfg["source_factors"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{model}_ledger.parquet" for model, (parent, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第124轮单一条件回撤预算已冻结，尚未计算新预算或账户。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "条件回撤冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    began = time.perf_counter()
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    sources = {period: pd.read_parquet(ROOT / path, columns=["date", "panic_reference_return", "learned_reference_return", "panic_state", "learned_state"])
               for period, path in cfg["source_factors"].items()}
    source = sources["evaluation"]
    require(pd.DatetimeIndex(source.date).equals(pd.DatetimeIndex(data.date)), "连续参考和行情日期不一致")
    first_reference = int(np.flatnonzero(source.date.ge(cfg["reference_start"]))[0])
    factors, certificates = budget_frame(source.date, source[["panic_reference_return", "learned_reference_return"]].to_numpy(float),
        source[["panic_state", "learned_state"]].to_numpy(float), first_reference, cfg["risk_window"], cfg["confidence"])
    factors.to_parquet(OUT / "full_continuous_factors.parquet", index=False)
    write_json(OUT / "optimizer_certificates.json", {"solves": certificates}, exclusive=True)
    scheduled = factors[factors.risk_update_scheduled]
    scheduled.to_csv(OUT / "unique_risk_update_records.csv", index=False, encoding="utf-8-sig")
    risk_summary = {"unique_monthly_origins": len(scheduled), "optimizer_windows": len(certificates),
        "linear_programs": int(scheduled.linear_programs.sum()),
        "successful_windows": int(scheduled.risk_status.eq("CONDITIONAL_DRAWDOWN_BUDGET_AVAILABLE").sum()),
        "status_counts": scheduled.risk_status.value_counts().to_dict()}
    print(f"连续参考预算完成：{risk_summary['optimizer_windows']}个独立优化窗口，{risk_summary['linear_programs']}次线性规划。", flush=True)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        selected = prefix_factors(factors, sources[period])
        require(pd.DatetimeIndex(selected.date).equals(pd.DatetimeIndex(frame.date)), "裁切后预算日期不一致")
        selected.to_parquet(OUT / f"{period}_factors.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, PRIMARY,
                targets=selected.target.to_numpy(float), event_mask=np.ones(len(frame), bool))
            decisions = decisions.merge(selected.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "条件回撤账户经济核算或终点结算失败")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()),
                "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                "no_view_target_origins": int(decisions.reference_weight.isna().sum()), "mean_exposure": float(ledger.exposure.mean())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for model, (parent, label) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, label
            benchmark = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "条件回撤策略及保存对照日历不同")
                measured = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                measured["annualized_return_excess_vs_buy_hold"] = measured["annualized_return"]-benchmark["annualized_return"]
                measured["meets_point_target"] = measured["net_sharpe"] is not None and measured["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(measured)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：一个新账户和四个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [row for row in main if row["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "CONDITIONAL_DRAWDOWN_BUDGET_ACCOUNTS_COMPLETE",
        "candidate_configurations": 1, "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8,
        "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8,
        "new_model_fits": 0, "new_reference_accounts": 0, "risk_summary": risk_summary, "all_metrics": main, "earlier_diagnostics": earlier,
        "account_coverage": coverage, "primary": primary, "post_selected_best_base": next(row for row in primary if row["cost"] == "BASE"),
        "historical_point_target_met": any(row["meets_point_target"] for row in primary), "run_seconds": time.perf_counter()-began,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主结果": primary, "较早結果": [row for row in earlier if row["model"] == PRIMARY], "预算": risk_summary,
        "账户覆盖": coverage, "核心耗时": result["run_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
