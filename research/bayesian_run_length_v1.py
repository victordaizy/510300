"""逐日行情持续时间预测及全历史对照，固定运行八个完整账户。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import require, now, digest, write_json
from research.bayesian_run_length_inputs_v1 import PRIMARY, CONTROL, sequential_forecasts, MeanConfirmationController
from research.joint_entry_exit_account_v1 import simulate_joint_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_bayesian_run_length_v1"
CONFIG = ROOT / "config/510300_bayesian_run_length_v1.json"
NAMES = {PRIMARY: "逐日行情持续时间预测", CONTROL: "始终累积全部历史的预测"}
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P114 = ROOT / "reports/research/510300_within_cycle_exit_v1"
CONTROLS = {"WITHIN_CYCLE_EXIT": (P114, "第114轮周期内退出"), "REARM_RIDGE": (P32, "原平均继续收益退出"),
            "CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "第91轮局部候选"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "逐日阶段方案已登记，不重复冻结")
    previous = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: previous[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 8, "逐日阶段八项必要测试未通过")
    cfg.update(study_id="510300_BAYESIAN_RUN_LENGTH_V1", round=123, registered_at=now(), primary=PRIMARY, candidate_configurations=2,
        methods=NAMES, hazard=1/242, prior={"mu": 0., "kappa": 1., "alpha": 2., "beta": .0001}, warmup_observations=252,
        observation="total_log", full_run_length_posterior=True, confirmation_days=2, fixed_control_hazard=0.,
        source_gap="NO_VIEW_SOURCE_GAP_REMAINDER_NO_RESET_OR_IMPUTATION", numeric_failure="NO_VIEW_NUMERICAL_REMAINDER_NO_SOLVER_RESCUE",
        missing_decision="NO_VIEW_KEEP_ACTUAL_SHARES_WITH_LOCKED_EXIT_CONTINUATION", zero_or_mixed="KEEP_CURRENT_ACTUAL_ASSET",
        terminal="LAST_DAY_OPEN_LIQUIDATION_PRIORITY", old_cooldown_or_rearm=False,
        rules="docs/510300_BAYESIAN_RUN_LENGTH_V1.md", input_receipt="reports/research/510300_bayesian_run_length_preflight_20260909/result.json",
        new_reference_accounts=0, source_budget_cny=0, position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED",
        previous_goal_turn_classification="PROGRESS_ROUND122_COMPLETED_FOUR_ACCOUNTS_AND_123_METHOD_REVIEW")
    paths = [Path(__file__), ROOT / "research/bayesian_run_length_inputs_v1.py", ROOT / "research/joint_entry_exit_account_v1.py",
        ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "tests/test_bayesian_run_length_v1.py",
        OUT / "tests_receipt.json", ROOT / "config/510300_research_authority_v6.json", ROOT / "config/510300_rearmed_session_exit_v1.json"]
    paths.extend(ROOT / cfg[k] for k in ["rules", "features", "dividends", "input_receipt"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{name}_ledger.parquet" for name, (parent, label) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第123轮逐日阶段预测与全历史对照已冻结，尚未计算新的真实行情预测或账户收益。", flush=True)


def run():
    began = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "逐日阶段冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    forecasts, final_states, forecast_coverage = {}, {}, []
    for method in NAMES:
        predicted, archive, final = sequential_forecasts(data, cfg, method)
        forecasts[method], final_states[method] = predicted, final
        predicted.to_parquet(OUT / f"{method}_forecasts.parquet", index=False)
        np.savez_compressed(OUT / f"{method}_posterior.npz", **archive)
        forecast_coverage.append({"method": method, "filter_status": final["status"], "updates": final["observations"],
            "available_forecasts": int(predicted.model_status.eq("PREDICTION_AVAILABLE").sum()),
            "unknown_forecasts": int(predicted.prediction.isna().sum()), "saved_log_probability_cells": int(len(archive["log_probabilities"]))})
        print(f"{NAMES[method]}：{final['observations']}次逐日更新，状态{final['status']}。", flush=True)
    write_json(OUT / "filter_final_states.json", final_states, exclusive=True)
    pd.DataFrame(forecast_coverage).to_csv(OUT / "forecast_coverage.csv", index=False, encoding="utf-8-sig")
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        require(np.isfinite(frame[["open", "close", "previous_close", "dividend"]].iloc[first-1:].to_numpy(float)).all(), "真实账户必要价格缺失，不能编造收益")
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            accounts, names = {}, dict(NAMES)
            for method in NAMES:
                controller = MeanConfirmationController(forecasts[method].iloc[:len(frame)])
                ledger, decisions, cycles = simulate_joint_policy(frame, dividends, cfg, cost, start, controller)
                save_account(folder, method, ledger, decisions)
                cycles.to_csv(folder / f"{method}_cycles.csv", index=False, encoding="utf-8-sig")
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "阶段预测账户终点或财富核算不符")
                require(cycles.empty or cycles.dropna(subset=["exit_date"]).holding_intervals.ge(1).all(), "联合账户退出违反次日可卖")
                coverage.append({"period": period, "cost": cost_id, "model": method, "holding_closes": int(ledger.shares.gt(0).sum()),
                    "completed_round_trips": len(cycles), "unknown_origins": int(decisions.decision_status.str.startswith("NO_VIEW").sum()),
                    "locked_exit_origins": int(decisions.account_mode.eq(2).sum()),
                    "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum())})
                accounts[method] = ledger
            for name, (parent, label) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{name}_ledger.parquet")
                saved.to_parquet(folder / f"{name}_ledger.parquet", index=False)
                accounts[name], names[name] = saved, label
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for name, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "联合策略及保存对照日历不同")
                measured = {"cost": cost_id, "model": name, "name": names[name], **summarize(saved, cfg)}
                measured["annualized_return_excess_vs_buy_hold"] = measured["annualized_return"]-bh["annualized_return"]
                measured["meets_point_target"] = measured["net_sharpe"] is not None and measured["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(measured)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": name, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": name, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            pd.DataFrame({"date": accounts[PRIMARY].date, **{name: saved.net_return.to_numpy() for name, saved in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：两个新账户和四个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    candidates = [m for m in main if m["model"] in NAMES]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "BAYESIAN_RUN_LENGTH_ACCOUNTS_COMPLETE", "candidate_configurations": 2,
        "evaluation_accounts": len(main), "new_accounts_generated": 4, "reused_control_accounts": 8, "earlier_diagnostic_accounts": len(earlier),
        "new_earlier_diagnostic_accounts": 4, "reused_earlier_accounts": 8, "new_model_fits": len(final_states), "new_reference_accounts": 0,
        "sequential_updates": sum(m["observations"] for m in final_states.values()), "forecast_coverage": forecast_coverage,
        "monthly_model_fits": 0, "full_length_online_filters": len(final_states),
        "all_metrics": main, "earlier_diagnostics": earlier, "account_coverage": coverage, "primary": [m for m in candidates if m["model"] == PRIMARY],
        "post_selected_best_base": max([m for m in candidates if m["cost"] == "BASE"], key=lambda m: m["net_sharpe"] if m["net_sharpe"] is not None else -np.inf),
        "historical_point_target_met": any(m["meets_point_target"] for m in candidates), "run_seconds": time.perf_counter()-began,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主历史": candidates, "较早历史": [m for m in earlier if m["model"] in NAMES], "耗时": result["run_seconds"], "覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
