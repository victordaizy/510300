"""第104轮单一同期误差混合设置：零新预测拟合、四个完整历史账户。"""
import json
import time
from pathlib import Path
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.paired_forecast_mse_exit_inputs_v1 import reconstruct_pairs, calibrate_months, PairedForecastExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_paired_forecast_mse_exit_v1"
CONFIG = ROOT / "config/510300_paired_forecast_mse_exit_v1.json"
P31 = ROOT / "reports/research/510300_learned_cycle_exit_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P102 = ROOT / "reports/research/510300_profit_drawdown_interaction_v1"
PRIMARY = "PAIRED_FORECAST_MSE_EXIT"
CONTROLS = {"REARM_RIDGE": (P32, "原八项退出"), "PROFIT_DRAWDOWN_INTERACTION_EXIT": (P102, "原九项交互退出"),
            "CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "第91轮局部候选"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "第104轮已登记，不重复冻结")
    old = json.loads((ROOT / "config/510300_profit_drawdown_interaction_v1.json").read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days", "cash_annual_rate_assumption",
            "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "confirmation_days", "specification",
            "recent_cycles", "minimum_cycles", "minimum_rows", "samples", "saved_models"]
    cfg = {k: old[k] for k in keys}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 7, "同期误差混合必要测试尚未通过")
    cfg.update(study_id="510300_PAIRED_FORECAST_MSE_EXIT_V1", round=104, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
               interaction_models=str((P102 / "saved_models.json").relative_to(ROOT)), rules="docs/510300_PAIRED_FORECAST_MSE_EXIT_V1.md",
               calibration_loss="UNCENTERED_CYCLE_EQUAL_WEIGHTED_PAIRED_FORECAST_MEAN_SQUARE_ERROR", initial_weights=[1., 0.],
               calibration_failure="NO_VIEW_KEEP_PREVIOUS_WEIGHTS", partial_pair_cycle="EXCLUDE_WHOLE_CYCLE",
               missing_selected_errors="NO_VIEW_WHOLE_CALIBRATION_NO_ROW_DROPS", identical_errors="NO_VIEW_KEEP_PREVIOUS_WEIGHTS",
               forecast_clock="LATEST_SAVED_MODEL_NOT_AFTER_ORIGIN_NO_CURRENT_REFERENCE_CYCLE_IN_TRAINING",
               calibration_clock="ORIGINAL_MONTH_FIRST_CLOSE_ONLY_FULLY_MATURE_COMPLETE_PAIRED_CYCLES",
               new_model_fits=0, new_reference_accounts=0, planned_calibration_supported_origins=66, expected_first_supported_origin="2021-03-01",
               source_budget_cny=0, position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED")
    paths = [Path(__file__), ROOT / "research/paired_forecast_mse_exit_inputs_v1.py", ROOT / "research/profit_drawdown_interaction_inputs_v1.py",
             ROOT / "research/rearmed_cycle_exit_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/simple_intraday_protection_v1.py",
             ROOT / "research/simple_session_divergence_v1.py", ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "tests/test_paired_forecast_mse_exit_v1.py", ROOT / "tests/test_median_continuation_v1.py",
             ROOT / "tests/test_profit_drawdown_interaction_v1.py", ROOT / "config/510300_research_authority_v6.json", OUT / "tests_receipt.json"]
    paths += [ROOT / cfg[k] for k in ["rules", "features", "dividends", "samples", "saved_models", "interaction_models"]]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths += [parent / period / cost / f"{model}_ledger.parquet" for model, (parent, _) in CONTROLS.items()]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第104轮唯一设置已冻结，尚未计算同期预测误差、新权重或新账户。", flush=True)


def run():
    start_clock = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "同期误差混合冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples = pd.read_parquet(ROOT / cfg["samples"])
    samples = samples[samples.signal.eq("D60_INTRA")].reset_index(drop=True)
    old = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    new = json.loads((ROOT / cfg["interaction_models"]).read_text(encoding="utf-8"))["models"]
    pairs = reconstruct_pairs(samples, old, new)
    pairs.to_parquet(OUT / "paired_reference_forecasts.parquet", index=False)
    calibrations, members = calibrate_months(pairs, old, cfg)
    ready = [r for r in calibrations if r["supports_minimum"]]
    require(len(ready) == cfg["planned_calibration_supported_origins"] and ready[0]["fit_origin"] == cfg["expected_first_supported_origin"], "完整成熟同期样本支持与已保存可用性不同")
    write_json(OUT / "saved_calibrations.json", {"calibrations": calibrations}, exclusive=True)
    members.to_parquet(OUT / "calibration_memberships.parquet", index=False)
    pd.DataFrame([{k: v for k, v in r.items() if k != "training_cycles"} for r in calibrations]).to_csv(OUT / "每月预测误差与混合权重.csv", index=False, encoding="utf-8-sig")
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
                                             ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        rule = make_rules(frame)["D60_INTRA"]
        pd.DataFrame({"date": frame.date, "entry_condition": rule["entry"], "original_price_exit": rule["exit"][1]}).to_parquet(OUT / f"{period}_entry_exit_conditions.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            controller = PairedForecastExitController(frame, old, new, calibrations, cfg["confirmation_days"])
            ledger, decisions, cycles = simulate_rearmed_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"], controller)
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "同期误差混合账户结算不符")
            require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "实际退出违反次日可卖")
            holding = decisions[decisions.learning_cycle_id.notna()]
            coverage.append({"period": period, "cost": cost_id, "holding_decisions": len(holding),
                             "prediction_available_rows": int(holding.learning_status.eq("PREDICTION_AVAILABLE").sum()),
                             "positive_new_weight_rows": int(holding.new_weight.gt(0).sum()), "mixed_weight_rows": int(holding.new_weight.between(0, 1, inclusive="neither").sum()),
                             "completed_round_trips": len(cycles), "learned_exit_cycles": int(cycles.exit_reasons.str.contains("学习条件", regex=False).sum()),
                             "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "同期成熟预测误差混合退出"}
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            benchmark = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "新旧账户完整交易日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"]-benchmark["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：一个新账户已完成，扣费夏普{destination[-5]['net_sharpe']}。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("model_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "PAIRED_FORECAST_MSE_EXIT_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8, "earlier_diagnostic_accounts": len(earlier),
              "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8, "new_reference_accounts": 0, "new_model_fits": 0,
              "new_calibration_estimates": sum(r["calibration_status"] == "CALIBRATION_COMPLETE" for r in calibrations),
              "no_view_calibration_origins": sum(r["calibration_status"] != "CALIBRATION_COMPLETE" for r in calibrations),
              "source_states": len(samples), "paired_states": int(pairs.both_available.sum()), "all_metrics": main, "earlier_diagnostics": earlier,
              "model_coverage": coverage, "primary": primary, "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False,
              "independent_validation": "NOT_ESTABLISHED", "position_impact": 0, "run_seconds": time.perf_counter()-start_clock}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"校准次数": result["new_calibration_estimates"], "耗时秒": result["run_seconds"], "持仓覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
