"""第144轮复用固定退出模型的继续持有预测幅度。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.continuation_strength_blend_inputs_v1 import continuation_strength_frames, current_close_signals, PRIMARY, MODELS
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_continuation_strength_blend_v1"
CONFIG = ROOT / "config/510300_continuation_strength_blend_v1.json"
P128 = ROOT / "reports/research/510300_entry_vintage_exit_v1"
P131 = ROOT / "reports/research/510300_vintage_reference_risk_v1"
P139 = ROOT / "reports/research/510300_model_support_reference_router_v1"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {MODELS[0]: (P131, "第131轮普通波动乘数策略"), MODELS[1]: (P143, "第143轮趋势波动连续预算"),
    "MODEL_SUPPORT_REFERENCE_ROUTER": (P139, "第139轮训练支持条件选择"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "继续预测预算已经冻结")
    preflight_path = ROOT / "reports/research/510300_continuation_strength_blend_preflight_20260909/result.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(preflight["status"] == "SAVED_CONTINUATION_PREDICTIONS_CLOCKS_STATES_AND_PARENT_TARGETS_READY" and
        preflight["new_predictions_or_budgets_or_accounts"] == 0, "原预测输入确认未完成")
    for source in preflight["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "原预测确认的来源改变")
    old_path = ROOT / "config/510300_trend_noise_reference_blend_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    original = json.loads((ROOT / "config/510300_entry_vintage_exit_v1.json").read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band", "target_volatility"]
    cfg = {key: old[key] for key in keys}
    require(original["specification"]["modes"]["1"]["days"] == 60, "原持仓期限改变")
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "继续预测预算六项必要测试未通过")
    cfg.update(study_id="510300_CONTINUATION_STRENGTH_BLEND_V1", round=144, primary=PRIMARY, candidate_configurations=1,
        decision_clock="15:05:00", combination="POSITIVE_CONTINUATION_OVER_CONTINUATION_PLUS_REMAINING_NOISE",
        maximum_holding_days=60, noise_window=20, saved_models=original["saved_models"], registered_at=now(), new_model_fits=0, new_reference_accounts=0,
        reference_cost_matching="SAME_PERIOD_AND_SAME_COST", parent_models=MODELS,
        outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
        rules="docs/510300_CONTINUATION_STRENGTH_BLEND_V1.md", source_budget_cny=0,
        independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUND143_FOUR_LOCAL_GAINS_INPUT144_READY_FULL_GOAL_NOT_MET")
    paths = [ROOT / item["path"] for item in preflight["sources"]]
    paths.extend([Path(__file__), preflight_path, ROOT / "scripts/preflight_continuation_strength_blend_20260909.py",
        ROOT / "research/continuation_strength_blend_inputs_v1.py", ROOT / "research/saved_parent_target_alignment_v1.py",
        ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "tests/test_continuation_strength_blend_v1.py", ROOT / "tests/test_trend_noise_reference_blend_v1.py",
        ROOT / "tests/test_trend_reference_router_v1.py", OUT / "tests_receipt.json", ROOT / cfg["rules"],
        ROOT / "config/510300_research_authority_v6.json", P131 / "saved_verification_receipt.json", P139 / "saved_verification_receipt.json"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    cfg["existing_input_checks"] = preflight["checks"]
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第144轮继续预测预算已冻结，尚未生成新目标或账户。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "继续预测预算冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    began = time.perf_counter()
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    records = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]
    main, earlier, yearly, eras, coverage, target_summaries = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        source_folders = [(MODELS[0], P131), (MODELS[1], P143)]
        parents_by_cost = {cost_id: {model: pd.read_parquet(parent_folder / period / cost_id / f"{model}_decisions.parquet").assign(source_cost=cost_id, source_model=model)
            for model, parent_folder in source_folders} for cost_id in cfg["costs"]}
        signals = {cost_id: current_close_signals(pd.read_parquet(P128 / period / cost_id / "ENTRY_VINTAGE_EXIT_decisions.parquet"),
            pd.read_parquet(P128 / period / cost_id / "ENTRY_VINTAGE_EXIT_ledger.parquet", columns=["date", "shares", "mark_clock"]),
            frame, start, cost_id) for cost_id in cfg["costs"]}
        factor_frames, summaries = continuation_strength_frames(frame, parents_by_cost, signals, records, cfg, start)
        target_summaries.extend({"period": period, **item} for item in summaries)
        for cost_id, cost in cfg["costs"].items():
            factors = factor_frames[cost_id]
            folder = OUT / period / cost_id
            folder.mkdir(parents=True)
            factors.to_parquet(folder / "factors.parquet", index=False)
            ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, PRIMARY,
                targets=factors.target.to_numpy(float), event_mask=np.ones(len(frame), bool))
            decisions = decisions.merge(factors.rename(columns={"date": "origin", "origin_index": "factor_origin_index", "model": "factor_model"}), on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "继续预测预算账户财富或终点清仓不符")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()),
                "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "unknown_target_origins": int(decisions.reference_weight.isna().sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()), "mean_exposure": float(ledger.exposure.mean())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "按保存继续持有预测幅度分配预算"}
            for model, (parent_folder, name) in CONTROLS.items():
                saved = pd.read_parquet(parent_folder / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            benchmark = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "继续预测预算与保存对照日历不同")
                measured = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                measured["annualized_return_excess_vs_buy_hold"] = measured["annualized_return"]-benchmark["annualized_return"]
                measured["meets_point_target"] = measured["net_sharpe"] is not None and measured["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(measured)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：一条新继续预测预算账户及四条保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage), ("target_coverage.csv", target_summaries)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "CONTINUATION_STRENGTH_BLEND_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8, "earlier_diagnostic_accounts": len(earlier),
        "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8, "new_model_fits": 0, "new_reference_accounts": 0,
        "all_metrics": main, "earlier_diagnostics": earlier, "account_coverage": coverage, "target_coverage": target_summaries,
        "primary": primary, "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "run_seconds": time.perf_counter()-began,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主历史": primary, "较早历史": [m for m in earlier if m["model"] == PRIMARY],
        "账户": coverage, "继续预测预算": target_summaries, "核心耗时": result["run_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
