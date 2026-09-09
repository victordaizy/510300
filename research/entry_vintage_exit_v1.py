"""复用原八项模型，单笔持仓固定首次收盘版本并计算四个实际账户。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.entry_vintage_exit_inputs_v1 import EntryVintageExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_entry_vintage_exit_v1"
CONFIG = ROOT / "config/510300_entry_vintage_exit_v1.json"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P114 = ROOT / "reports/research/510300_within_cycle_exit_v1"
PRIMARY = "ENTRY_VINTAGE_EXIT"
CONTROLS = {"WITHIN_CYCLE_EXIT": (P114, "第114轮持仓内按月更新模型"), "REARM_RIDGE": (P32, "原平均继续收益退出"),
    "CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "第91轮连续最小方差预算"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "首次持仓收盘版本策略已经登记")
    parent_path = ROOT / "config/510300_within_cycle_exit_v1.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
        "confirmation_days", "specification", "feature_columns", "feature_names", "feature_clip", "ridge_alpha"]
    cfg = {key: parent[key] for key in keys}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "固定版本六项必要测试未通过")
    cfg.update(study_id="510300_ENTRY_VINTAGE_EXIT_V1", round=128, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
        saved_models=str((P114 / "saved_models.json").relative_to(ROOT)), model_selection_clock="FIRST_CLOSE_OF_ACTUAL_FILLED_ENTRY",
        model_reselection="ONLY_ON_NEW_ACTUAL_CYCLE", first_unavailable_model="NO_LEARNING_PREDICTION_FOR_ENTIRE_CYCLE",
        missing_current_state="NO_PREDICTION_RESET_NEGATIVE_COUNT_KEEP_SELECTED_RECORD", new_model_fits=0, new_reference_accounts=0,
        planned_reused_model_records=141, planned_reused_available_models=114,
        input_receipt="reports/research/510300_entry_vintage_exit_preflight_20260909/result.json", rules="docs/510300_ENTRY_VINTAGE_EXIT_V1.md",
        source_budget_cny=0, independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUND127_COMPLETED_114_FITS_FOUR_ACCOUNTS_AND_128_VINTAGE_PREFLIGHT")
    paths = [Path(__file__), parent_path, ROOT / "research/entry_vintage_exit_inputs_v1.py", ROOT / "research/within_cycle_exit_inputs_v1.py",
        ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/rearmed_cycle_exit_account_v1.py", ROOT / "research/simple_intraday_protection_v1.py",
        ROOT / "research/simple_session_divergence_v1.py", ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/adaptive_allocation_v1.py",
        ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "tests/test_entry_vintage_exit_v1.py", ROOT / "tests/test_within_cycle_exit_v1.py",
        ROOT / "tests/test_median_continuation_v1.py", OUT / "tests_receipt.json", ROOT / "config/510300_research_authority_v6.json",
        ROOT / "docs/510300_WITHIN_CYCLE_EXIT_V1.md"]
    paths.extend(ROOT / cfg[key] for key in ["features", "dividends", "saved_models", "input_receipt", "rules"])
    receipt = json.loads((ROOT / cfg["input_receipt"]).read_text(encoding="utf-8"))
    for item in receipt["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "已经核对的原模型版本来源改变")
        paths.append(ROOT / item["path"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第128轮单一首次持仓收盘版本固定已冻结，尚未计算新预测或账户。", flush=True)

def run():
    started_clock = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "首次持仓收盘固定版本冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]
    require(len(models) == cfg["planned_reused_model_records"] and sum(m["status"] == "FIT_COMPLETE" for m in models) == cfg["planned_reused_available_models"], "原保存模型记录或成熟数量改变")
    write_json(OUT / "reused_models_receipt.json", {"recorded_at": now(), "source": cfg["saved_models"], "source_sha256": digest(ROOT / cfg["saved_models"]),
        "reused_model_records": len(models), "available_model_records": sum(m["status"] == "FIT_COMPLETE" for m in models), "new_model_fits": 0}, exclusive=True)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        rule = make_rules(frame)["D60_INTRA"]
        pd.DataFrame({"date": frame.date, "entry_condition": rule["entry"], "original_price_exit": rule["exit"][1]}).to_parquet(OUT / f"{period}_entry_exit_conditions.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_rearmed_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"], EntryVintageExitController(frame, models, cfg["confirmation_days"]))
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "首次持仓收盘固定版本账户结算不符")
            require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "首次持仓收盘固定版本退出违反次日可卖")
            holding = decisions[decisions.learning_cycle_id.notna()]
            coverage.append({"period": period, "cost": cost_id, "holding_decisions": len(holding),
                "model_available_rows": int(holding.learning_status.eq("PREDICTION_AVAILABLE").sum()), "no_model_rows": int(holding.learning_status.ne("PREDICTION_AVAILABLE").sum()),
                "learned_exit_cycles": int(cycles.exit_reasons.str.contains("学习条件", regex=False).sum()), "completed_round_trips": len(cycles),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                "waiting_for_new_condition_rows": int(decisions.action.str.contains("旧入场条件", regex=False).sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "首次持仓收盘固定版本继续收益退出"}
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "首次持仓收盘固定版本及对照完整日历不同")
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
            print(f"{period}／{cost_id}：一个新首次持仓收盘固定版本账户和四个保存对照已完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("model_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "ENTRY_VINTAGE_EXIT_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8, "earlier_diagnostic_accounts": len(earlier),
        "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8, "new_reference_accounts": 0,
        "run_seconds": time.perf_counter()-started_clock, "new_model_fits": 0, "reused_model_records": len(models), "reused_available_models": sum(m["status"] == "FIT_COMPLETE" for m in models),
        "reused_unavailable_models": sum(m["status"] != "FIT_COMPLETE" for m in models), "failed_fits": 0,
        "all_metrics": main, "earlier_diagnostics": earlier, "model_coverage": coverage, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": primary, "较早": [m for m in earlier if m["model"] == PRIMARY], "耗时": result["run_seconds"], "新训练次数": result["new_model_fits"], "求解失败": result["failed_fits"], "持仓覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
