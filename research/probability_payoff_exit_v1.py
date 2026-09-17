"""拟合有限的概率幅度退出模型，保留月度时钟并计算四个真实资金账户。"""
import json
import time
from pathlib import Path
import pandas as pd
import sklearn
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.probability_payoff_exit_inputs_v1 import build_monthly_models, ProbabilityPayoffExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_probability_payoff_exit_v1"
CONFIG = ROOT / "config/510300_probability_payoff_exit_v1.json"
P114 = ROOT / "reports/research/510300_within_cycle_exit_v1"
PRIMARY = "PROBABILITY_PAYOFF_EXIT"
CONTROLS = {
    "ENTRY_VINTAGE_EXIT": (ROOT / "reports/research/510300_entry_vintage_exit_v1", "第128轮进入时固定原退出模型"),
    "VINTAGE_REFERENCE_RISK": (ROOT / "reports/research/510300_vintage_reference_risk_v1", "第131轮固定版本与普通波动预算"),
    "TREND_NOISE_REFERENCE_BLEND": (ROOT / "reports/research/510300_trend_noise_reference_blend_v1", "第143轮趋势波动连续预算"),
    "BUY_HOLD": (ROOT / "reports/research/510300_rearmed_session_exit_v1", "买入持有")}


def freeze():
    require(not CONFIG.exists(), "概率幅度退出策略已经登记")
    parent_path = ROOT / "config/510300_within_cycle_exit_v1.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
            "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
            "confirmation_days", "specification", "feature_columns", "feature_names", "feature_clip", "recent_cycles", "minimum_cycles", "minimum_rows"]
    cfg = {key: parent[key] for key in keys}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 9, "九项概率幅度退出必要测试尚未通过")
    cfg.update(study_id="510300_PROBABILITY_PAYOFF_EXIT_V1", round=158, registered_at=now(), primary=PRIMARY,
        candidate_configurations=1, source_models=str((P114 / "saved_models.json").relative_to(ROOT)),
        samples=str((P114 / "extended_reference_samples.parquet").relative_to(ROOT)),
        variance_smoothing=1e-9, class_definition="ORIGINAL_TARGET_STRICTLY_ABOVE_ZERO_VS_NONPOSITIVE",
        model_loss="CYCLE_EQUAL_WEIGHTED_GAUSSIAN_CLASS_MOMENTS_AND_CLASS_MEAN_NET_PAYOFFS",
        feature_normalization="CYCLE_EQUAL_GLOBAL_STANDARDIZATION_CLIP5_WITHOUT_WITHIN_CYCLE_CENTERING",
        zero_variance_rule="EXCLUDE_EXACT_GLOBAL_CONSTANT_FACTORS_KEEP_ALL_PARAMETERS_ALL_CONSTANT_USES_PRIORS",
        missing_class="NO_VIEW_BOTH_CLASSES_REQUIRED", solver="CLOSED_FORM_WEIGHTED_CLASS_MOMENTS",
        solver_fallback=False, sklearn_version=sklearn.__version__,
        model_selection_clock="FIRST_CLOSE_OF_ACTUAL_FILLED_ENTRY", model_reselection="ONLY_ON_NEW_ACTUAL_CYCLE",
        cache="ACTUAL_ORDERED_MEMBERS_FEATURES_TARGET_WEIGHTS_AND_SETTINGS_PAST_SUCCESS_OR_FAILURE",
        planned_distinct_fits=25, planned_monthly_records=141, planned_eligible_months=114,
        rules="docs/510300_PROBABILITY_PAYOFF_EXIT_V1.md", new_reference_accounts=0, source_budget_cny=0,
        independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUND157_COMPLETED_CLOSED_AND_DELIVERED")
    names = ["research/probability_payoff_exit_inputs_v1.py", "research/learned_cycle_exit_v1.py", "research/within_cycle_exit_inputs_v1.py",
        "research/rearmed_cycle_exit_account_v1.py", "research/simple_intraday_protection_v1.py", "research/simple_session_divergence_v1.py",
        "research/simple_price_entry_exit_v1.py", "research/adaptive_allocation_v1.py", "research/intraday_overnight_increment_v1.py",
        "tests/test_probability_payoff_exit_v1.py", "tests/test_median_continuation_v1.py", "config/510300_research_authority_v6.json",
        "docs/510300_PROBABILITY_PAYOFF_EXIT_NEXT_20260909.md"]
    paths = [Path(__file__), parent_path, OUT / "tests_receipt.json", *(ROOT / p for p in names),
             *(ROOT / cfg[key] for key in ["source_models", "samples", "rules", "features", "dividends"])]
    known = {}
    for name in ["510300_single_component_exit_v1"]:
        path = ROOT / "config" / (name + ".json")
        previous = json.loads(path.read_text(encoding="utf-8"))
        known.update({str(Path(item["path"])): item["sha256"] for item in previous["frozen_files"]})
        paths.append(path)
    for key in ["source_models", "samples", "features", "dividends"]:
        require(digest(ROOT / cfg[key]) == known[str(Path(cfg[key]))], "既有训练、价格或分红来源版本改变")
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            for model, (folder, _) in CONTROLS.items():
                path = folder / period / cost / f"{model}_ledger.parquet"
                require(digest(path) == known[str(path.relative_to(ROOT))], "保存对照已绑定版本改变")
                paths.append(path)
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第158轮一套概率幅度退出规则已冻结，尚未拟合新模型或计算新账户。", flush=True)


def run():
    started = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "概率幅度退出冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples = pd.read_parquet(ROOT / cfg["samples"])
    originals = json.loads((ROOT / cfg["source_models"]).read_text(encoding="utf-8"))["models"]
    require(all(pd.Timestamp(r["fit_time"]) == data.date.iloc[r["fit_index"]] + pd.Timedelta(hours=15, minutes=5) for r in originals), "月度训练时钟不对应完整交易日历")
    models, monthly, memberships, counts = build_monthly_models(samples, originals, cfg)
    require(counts["monthly_records"] == 141 and counts["eligible_monthly_records"] == 114 and counts["distinct_source_input_sets"] == 25, "原训练支持或缓存身份数量改变")
    require(counts["new_model_fits"] <= 25, "相同输入重复拟合")
    write_json(OUT / "saved_models.json", {"recorded_at": now(), "counts": counts, "models": models}, exclusive=True)
    monthly.to_csv(OUT / "monthly_fit_diagnostics.csv", index=False, encoding="utf-8-sig")
    memberships.to_parquet(OUT / "training_memberships.parquet", index=False)
    print(f"141个月度记录完成：{counts['new_model_fits']}次新拟合，{counts['reused_monthly_fits']}个月复用；失败{counts['failed_fits']}次。", flush=True)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        rule = make_rules(frame)["D60_INTRA"]
        pd.DataFrame({"date": frame.date, "entry_condition": rule["entry"], "original_price_exit": rule["exit"][1]}).to_parquet(OUT / f"{period}_entry_exit_conditions.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_rearmed_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"], ProbabilityPayoffExitController(frame, models, cfg["confirmation_days"]))
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "概率幅度退出账户未完整结算")
            require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "概率幅度退出违反次日可卖")
            holding = decisions[decisions.learning_cycle_id.notna()]
            coverage.append({"period": period, "cost": cost_id, "holding_decisions": len(holding),
                "model_available_rows": int(holding.learning_status.eq("PREDICTION_AVAILABLE").sum()),
                "no_model_rows": int(holding.learning_status.ne("PREDICTION_AVAILABLE").sum()),
                "learned_exit_cycles": int(cycles.exit_reasons.str.contains("学习条件", regex=False).sum()),
                "completed_round_trips": len(cycles), "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "原八因素估计概率与盈亏幅度后固定版本退出"}
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "概率幅度退出与对照完整日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                dest.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            pd.DataFrame({"date": ledger.date, **{k: a.net_return.to_numpy() for k, a in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：一个新概率幅度退出账户和四个保存对照已完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("model_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "PROBABILITY_PAYOFF_EXIT_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8,
        "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8,
        "new_reference_accounts": 0, "run_seconds": time.perf_counter() - started, **counts,
        "all_metrics": main, "earlier_diagnostics": earlier, "model_coverage": coverage, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False,
        "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": primary, "较早": [m for m in earlier if m["model"] == PRIMARY], "核心秒数": result["run_seconds"], "拟合统计": counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
