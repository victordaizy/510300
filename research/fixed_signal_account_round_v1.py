"""复用固定信号研究的冻结、四个完整账户及既有对照，减少重复编写。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.joint_entry_exit_account_v1 import simulate_joint_policy
from research.intraday_overnight_increment_v1 import require, now, digest, write_json

PARENTS = {"WITHIN_CYCLE_EXIT": ("510300_within_cycle_exit_v1", "第114轮周期内退出"),
    "REARM_RIDGE": ("510300_rearmed_session_exit_v1", "原平均继续收益退出"),
    "CONTINUOUS_REFERENCE_MIN_VARIANCE": ("510300_continuous_reference_min_variance_v1", "第91轮连续风险预算"),
    "BUY_HOLD": ("510300_rearmed_session_exit_v1", "买入持有")}


def freeze_signal(root, out, config, additions, source_files, tests_required):
    require(not config.exists(), "固定信号研究已经冻结，不重复登记")
    parent_path = root / "config/510300_rearmed_session_exit_v1.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    cfg = {k: parent[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    tests = json.loads((out / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == tests_required, "固定信号必要测试尚未通过")
    cfg.update(candidate_configurations=1, new_model_fits=0, new_reference_accounts=0, registered_at=now(), source_budget_cny=0,
        goal_achieved=False, independent_validation="NOT_ESTABLISHED", position_impact=0, **additions)
    paths = [Path(__file__), parent_path, out / "tests_receipt.json", root / "config/510300_research_authority_v6.json",
        root / "research/joint_entry_exit_account_v1.py", root / "research/intraday_overnight_increment_v1.py", root / "research/adaptive_allocation_v1.py"]
    paths.extend(root / cfg[k] for k in ["rules", "features", "dividends", "input_receipt"])
    paths.extend(source_files)
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(root / "reports/research" / folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in PARENTS.items())
    cfg["frozen_files"] = [{"path": str(path.relative_to(root)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(config, cfg, exclusive=True)
    print(f"第{cfg['round']}轮单一固定信号已冻结，尚未计算新指标或账户。", flush=True)


def run_signal(root, out, config, build_factors, controller_class):
    cfg = json.loads(config.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(root / item["path"]) == item["sha256"], "固定信号冻结来源改变")
    write_json(out / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(config)}, exclusive=True)
    began = time.perf_counter()
    data = pd.read_parquet(root / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(root / cfg["dividends"]))
    factors, factor_summary = build_factors(data, cfg)
    require(pd.DatetimeIndex(factors.date).equals(pd.DatetimeIndex(data.date)), "固定信号没有完整保留行情日历")
    factors.to_parquet(out / "factors.parquet", index=False)
    write_json(out / "factor_summary.json", factor_summary, exclusive=True)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        require(np.isfinite(frame[["open", "close", "previous_close", "dividend"]].iloc[first-1:].to_numpy(float)).all(), "真实账户缺少必要价格，不能编造收益")
        for cost_id, cost in cfg["costs"].items():
            folder = out / period / cost_id
            ledger, decisions, cycles = simulate_joint_policy(frame, dividends, cfg, cost, start, controller_class(factors.iloc[:len(frame)], cfg))
            save_account(folder, cfg["primary"], ledger, decisions)
            cycles.to_csv(folder / f"{cfg['primary']}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "固定信号账户财富核算或终点清仓不符")
            require(cycles.empty or cycles.dropna(subset=["exit_date"]).holding_intervals.ge(1).all(), "固定信号实际卖出违反次日可卖")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()), "complete_cycles": len(cycles),
                "unknown_origins": int(decisions.decision_status.str.startswith("NO_VIEW").sum()), "locked_exit_origins": int(decisions.account_mode.eq(2).sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum())})
            accounts, names = {cfg["primary"]: ledger}, {cfg["primary"]: cfg["name"]}
            for model, (directory, label) in PARENTS.items():
                saved = pd.read_parquet(root / "reports/research" / directory / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, label
            benchmark = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "固定信号与保存对照日历不同")
                measured = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                measured["annualized_return_excess_vs_buy_hold"] = measured["annualized_return"]-benchmark["annualized_return"]
                measured["meets_point_target"] = measured["net_sharpe"] is not None and measured["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(measured)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：一个新固定信号账户及四个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(out / name, index=False, encoding="utf-8-sig")
    primary = [row for row in main if row["model"] == cfg["primary"]]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "FIXED_SIGNAL_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8, "earlier_diagnostic_accounts": len(earlier),
        "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8, "new_model_fits": 0, "new_reference_accounts": 0,
        "factor_summary": factor_summary, "all_metrics": main, "earlier_diagnostics": earlier, "account_coverage": coverage, "primary": primary,
        "post_selected_best_base": next(row for row in primary if row["cost"] == "BASE"), "historical_point_target_met": any(row["meets_point_target"] for row in primary),
        "run_seconds": time.perf_counter()-began, "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(out / "result.json", result, exclusive=True)
    print(json.dumps({"主历史": primary, "较早历史": [row for row in earlier if row["model"] == cfg["primary"]], "因子": factor_summary,
        "账户": coverage, "核心耗时": result["run_seconds"]}, ensure_ascii=False), flush=True)
