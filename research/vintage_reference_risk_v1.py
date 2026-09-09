"""第131轮固定入场模型参考与普通波动乘数的单一完整账户组合。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.vintage_reference_risk_inputs_v1 import vintage_risk_targets
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_vintage_reference_risk_v1"
CONFIG = ROOT / "config/510300_vintage_reference_risk_v1.json"
PRIMARY = "VINTAGE_REFERENCE_RISK"
P128 = ROOT / "reports/research/510300_entry_vintage_exit_v1"
P109 = ROOT / "reports/research/510300_conditional_variance_budget_v1"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {"ENTRY_VINTAGE_EXIT": (P128, "第128轮固定入场模型参考"),
    "ROLLING_VARIANCE_BUDGET_CONTROL": (P109, "第109轮普通波动缩减旧组合"),
    "CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "第91轮连续两策略预算"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "固定版本参考风险组合已经冻结")
    old_path = ROOT / "config/510300_conditional_variance_budget_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band", "target_volatility"]
    cfg = {k: old[k] for k in keys}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 7, "组合七项必要测试未通过")
    cfg.update(study_id="510300_VINTAGE_REFERENCE_RISK_V1", round=131, primary=PRIMARY, candidate_configurations=1,
        registered_at=now(), new_model_fits=0, new_reference_accounts=0, reference_cost_matching="SAME_PERIOD_AND_SAME_COST",
        reference_state_owner="EXISTING_ROUND128_RESEARCH_REFERENCE_NOT_OUTER_ACCOUNT", outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE",
        rules="docs/510300_VINTAGE_REFERENCE_RISK_V1.md", input_receipt="reports/research/510300_vintage_reference_risk_preflight_20260909/result.json",
        source_budget_cny=0, independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUND130_FOUR_ACCOUNTS_SAVED_FRONTIER_AND_NEXT_INPUT_CLOCKS_COMPLETE")
    paths = [Path(__file__), old_path, ROOT / "research/vintage_reference_risk_inputs_v1.py", ROOT / "research/event_clock_account_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "tests/test_vintage_reference_risk_v1.py",
        OUT / "tests_receipt.json", ROOT / "config/510300_research_authority_v6.json", ROOT / cfg["rules"], ROOT / cfg["input_receipt"], ROOT / cfg["dividends"]]
    receipt = json.loads((ROOT / cfg["input_receipt"]).read_text(encoding="utf-8"))
    for item in receipt["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "已绑定参考或普通波动来源变化")
        paths.append(ROOT / item["path"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第131轮单一参考与风险乘数组合已冻结，尚未合成新目标或账户。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "组合冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    began = time.perf_counter()
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    main, earlier, yearly, eras, coverage, target_summaries = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        risk = pd.read_parquet(P109 / f"{period}_factors.parquet")
        references = {cost: pd.read_parquet(P128 / period / cost / "ENTRY_VINTAGE_EXIT_decisions.parquet") for cost in cfg["costs"]}
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            factors, target_summary = vintage_risk_targets(frame, risk, references, cfg, start, cost_id)
            target_summaries.append({"period": period, **target_summary})
            folder.mkdir(parents=True)
            factors.to_parquet(folder / "factors.parquet", index=False)
            ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, PRIMARY,
                targets=factors.target.to_numpy(float), event_mask=np.ones(len(frame), bool))
            decisions = decisions.merge(factors.rename(columns={"date": "origin", "origin_index": "factor_origin_index"}), on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "组合账户财富或终点清仓不符")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()),
                "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "unknown_target_origins": int(decisions.reference_weight.isna().sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()), "mean_exposure": float(ledger.exposure.mean())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "固定入场模型参考与普通波动仓位组合"}
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            benchmark = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "组合与保存对照日历不同")
                measured = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                measured["annualized_return_excess_vs_buy_hold"] = measured["annualized_return"]-benchmark["annualized_return"]
                measured["meets_point_target"] = measured["net_sharpe"] is not None and measured["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(measured)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：一个新组合实际账户及四个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage), ("target_coverage.csv", target_summaries)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "VINTAGE_REFERENCE_RISK_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8, "earlier_diagnostic_accounts": len(earlier),
        "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8, "new_model_fits": 0, "new_reference_accounts": 0,
        "all_metrics": main, "earlier_diagnostics": earlier, "account_coverage": coverage, "target_coverage": target_summaries,
        "primary": primary, "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "run_seconds": time.perf_counter()-began,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主历史": primary, "较早历史": [m for m in earlier if m["model"] == PRIMARY], "账户": coverage,
        "目标": target_summaries, "核心耗时": result["run_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
