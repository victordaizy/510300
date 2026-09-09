"""有限期限逐年龄倒推及同年龄固定自然目标的冻结、训练和独立账户。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import training_rows
from research.simple_intraday_protection_v1 import make_rules
from research.finite_horizon_exit_inputs_v1 import (FEATURES, CN, METHODS, PRIMARY, CONTROL, prepare_cashflows,
    fit_backward_month, FiniteHorizonExitController, simulate_finite_exit)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_finite_horizon_exit_v1"
CONFIG = ROOT / "config/510300_finite_horizon_exit_v1.json"
P31 = ROOT / "reports/research/510300_learned_cycle_exit_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P114 = ROOT / "reports/research/510300_within_cycle_exit_v1"
NAMES = {PRIMARY: "逐持仓日倒推后续退出", CONTROL: "同持仓日固定自然退出目标"}
CONTROLS = {"WITHIN_CYCLE_EXIT": (P114, "第114轮周期内退出"), "REARM_RIDGE": (P32, "原平均继续收益退出"),
            "CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "第91轮局部候选"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "逐年龄有限比較已登记，不重复冻结")
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    train = json.loads((ROOT / "config/510300_learned_cycle_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "specification", "saved_models"]}
    cfg.update({k: train[k] for k in ["recent_cycles", "minimum_cycles", "minimum_rows", "feature_clip", "ridge_alpha"]})
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 8, "逐年龄现金流、倒推和实际成交测试尚未通过")
    cfg.update(study_id="510300_FINITE_HORIZON_EXIT_V1", round=118, registered_at=now(), primary=PRIMARY, methods=METHODS,
        candidate_configurations=2, feature_columns=FEATURES, feature_names=CN, minimum_age_cycles=10, maximum_age=59,
        confirmation_days=1, fit_intercept=True, fit_cycle_intercepts=False, solver="JOINT_TWO_TARGET_NORMAL_EQUATIONS",
        planned_mature_months=114, planned_joint_design_fits=5685, planned_scalar_target_fits=11370,
        model_loss="EQUAL_CYCLE_PAIRED_RIDGE_UNPENALIZED_INTERCEPT", rules="docs/510300_FINITE_HORIZON_EXIT_V1.md",
        saved_market_samples="reports/research/510300_market_path_state_preflight_20260908/augmented_reference_samples.parquet",
        samples=str((P31 / "all_reference_samples.parquet").relative_to(ROOT)),
        reference_cycles=str((P31 / "reference/D60_INTRA_cycles.csv").relative_to(ROOT)),
        age_support_receipt="reports/research/510300_finite_horizon_age_support_20260908/result.json",
        training_cashflow="FIRST_FULLY_EXECUTABLE_OPEN_AFTER_REQUEST_LOCKED_UNTIL_FILL_BASE_COSTS_EARNED_DIVIDEND_ONCE",
        target_normalization="REFERENCE_QUANTITY_TIMES_IMMEDIATE_EXECUTABLE_RAW_OPEN",
        missing_training_features="WHOLE_MONTH_NO_VIEW_PRESERVE_ALL_ORIGINAL_ROWS",
        missing_age_model="NO_NEW_LEARNING_EXIT_KEEP_KNOWN_LATER_POLICY_AND_ORIGINAL_EXITS",
        fit_failure="NO_VIEW_NO_SOLVER_OR_PARAMETER_RESCUE", new_reference_accounts=0, source_budget_cny=0,
        position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED",
        environment={"numpy": np.__version__, "pandas": pd.__version__})
    paths = [Path(__file__), ROOT / "research/finite_horizon_exit_inputs_v1.py", ROOT / "research/market_path_exit_inputs_v1.py",
        ROOT / "research/market_path_state_v1.py", ROOT / "research/rearmed_cycle_exit_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
        ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py",
        ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py", ROOT / "research/simple_price_entry_exit_v1.py",
        ROOT / "config/510300_rearmed_session_exit_v1.json", ROOT / "config/510300_learned_cycle_exit_v1.json", ROOT / "config/510300_research_authority_v6.json",
        ROOT / "tests/test_finite_horizon_exit_v1.py", ROOT / "tests/test_median_continuation_v1.py", OUT / "tests_receipt.json"]
    paths.extend(ROOT / cfg[k] for k in ["rules", "features", "dividends", "saved_models", "samples", "reference_cycles", "age_support_receipt", "saved_market_samples"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{model}_ledger.parquet" for model, (parent, name) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第118轮两个逐年龄目标设置及可成交现金流规则已冻结，尚未拟合或读取新账户收益。", flush=True)


def train_models(data, samples, curves, originals, cfg):
    models, month_receipts, age_receipts, memberships = [], [], [], []
    for original in originals:
        t = int(original["fit_index"])
        rows, ids = training_rows(samples, t, cfg)
        require(ids == original["training_cycles"] and len(rows) == original["training_rows"], "逐年龄训练改变原完整周期成员")
        eligible = len(ids) >= cfg["minimum_cycles"] and len(rows) >= cfg["minimum_rows"]
        require(eligible == (original["status"] == "FIT_COMPLETE"), "逐年龄训练改变原成熟月份")
        missing = int((~np.isfinite(rows[FEATURES].to_numpy(float)).all(axis=1)).sum())
        stored, receipts, members = {}, [], []
        status = "NO_VIEW_MINIMUM_MATURE_CYCLES_OR_ROWS"
        if eligible and missing:
            status = "NO_VIEW_INCOMPLETE_TRAINING_FEATURES"
        elif eligible:
            stored, receipts, members = fit_backward_month(rows, curves, cfg)
            status = "MONTH_COMPLETE"
        record = {"fit_index": t, "fit_origin": str(data.date.iloc[t].date()),
                  "fit_time": data.date.iloc[t]+pd.Timedelta(hours=15, minutes=5), "status": status,
                  "training_cycles": ids, "training_cycle_count": len(ids), "training_rows": len(rows), "eligible_for_fit": eligible,
                  "latest_exit_index": int(rows.exit_index.max()) if len(rows) else None, "missing_feature_rows": missing,
                  "age_models": stored, "age_statuses": {str(r["holding_age"]): r["status"] for r in receipts}}
        require(not len(rows) or rows.exit_index.le(t).all(), "逐年龄回归读取未完成周期")
        models.append(record)
        month_receipts.append({k: v for k, v in record.items() if k not in ["training_cycles", "age_models", "age_statuses"]})
        age_receipts.extend({"fit_index": t, "fit_origin": record["fit_origin"], **r} for r in receipts)
        memberships.extend({"fit_index": t, **m} for m in members)
    mature = sum(r["eligible_for_fit"] for r in month_receipts)
    supported = sum(r["cycles"] >= cfg["minimum_age_cycles"] for r in age_receipts)
    require(mature == cfg["planned_mature_months"] and supported == cfg["planned_joint_design_fits"], "逐年龄成熟支持与登记统计不同")
    write_json(OUT / "saved_models.json", {"models": models, "methods": METHODS, "feature_names": CN}, exclusive=True)
    pd.DataFrame(month_receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(age_receipts).to_csv(OUT / "逐月逐持仓日模型支持.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memberships).to_parquet(OUT / "training_memberships.parquet", index=False)
    print(f"逐年龄共同求解完成{sum(r['status']=='FIT_COMPLETE' for r in age_receipts)}次；原成熟月份{mature}个；不足十周期的年龄单元{sum(r['cycles']<cfg['minimum_age_cycles'] for r in age_receipts)}个。", flush=True)
    return models, month_receipts, age_receipts


def run():
    started = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "逐年龄有限比较的冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    samples = pd.read_parquet(ROOT / cfg["saved_market_samples"])
    require(samples.signal.eq("D60_INTRA").all(), "逐年龄输入不是原D60参考")
    samples, curves, cashflows = prepare_cashflows(data, dividends, samples, pd.read_csv(ROOT / cfg["reference_cycles"]), cfg)
    samples.to_parquet(OUT / "extended_reference_samples.parquet", index=False)
    cashflows.to_parquet(OUT / "reference_liquidation_cashflows.parquet", index=False)
    originals = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    models, months, ages = train_models(data, samples, curves, originals, cfg)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        rule = make_rules(frame)["D60_INTRA"]
        pd.DataFrame({"date": frame.date, "entry_condition": rule["entry"], "original_price_exit": rule["exit"][1]}).to_parquet(OUT / f"{period}_entry_exit_conditions.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            folder, accounts, names = OUT / period / cost_id, {}, NAMES.copy()
            for method in METHODS:
                controller = FiniteHorizonExitController(frame, models, method)
                ledger, decisions, cycles = simulate_finite_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"], controller)
                save_account(folder, method, ledger, decisions)
                cycles.to_csv(folder / f"{method}_cycles.csv", index=False, encoding="utf-8-sig")
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "逐年龄真实账户结算不同")
                require(cycles.dropna(subset=["exit_date"]).holding_intervals.ge(1).all(), "逐年龄退出违反次日可卖")
                holding = decisions[decisions.learning_cycle_id.notna()]
                coverage.append({"period": period, "cost": cost_id, "model": method, "holding_decisions": len(holding),
                    "model_available_rows": int(holding.learning_status.eq("PREDICTION_AVAILABLE").sum()),
                    "no_model_rows": int(holding.learning_status.ne("PREDICTION_AVAILABLE").sum()),
                    "learned_exit_cycles": int(cycles.exit_reasons.str.contains("学习条件", regex=False).sum()),
                    "completed_round_trips": len(cycles), "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum())})
                accounts[method] = ledger
            for method, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{method}_ledger.parquet")
                saved.to_parquet(folder / f"{method}_ledger.parquet", index=False)
                accounts[method], names[method] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for method, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "逐年龄及保存对照完整日历不同")
                m = {"cost": cost_id, "model": method, "name": names[method], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"]-bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                dest.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": method, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": method, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            pd.DataFrame({"date": accounts[PRIMARY].date, **{k: a.net_return.to_numpy() for k, a in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：两个逐年龄真实账户及四个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("model_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    new_main = [m for m in main if m["model"] in METHODS]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "FINITE_HORIZON_EXIT_ACCOUNTS_COMPLETE", "candidate_configurations": 2,
        "evaluation_accounts": len(main), "new_accounts_generated": 4, "reused_control_accounts": 8, "earlier_diagnostic_accounts": len(earlier),
        "new_earlier_diagnostic_accounts": 4, "reused_earlier_accounts": 8, "new_reference_accounts": 0, "run_seconds": time.perf_counter()-started,
        "new_model_fits": sum(r["cycles"] >= cfg["minimum_age_cycles"] for r in ages), "scalar_target_models": 2*sum(r["status"] == "FIT_COMPLETE" for r in ages),
        "completed_fits": sum(r["status"] == "FIT_COMPLETE" for r in ages), "failed_fits": sum(r["status"] == "NO_VIEW_MODEL_FIT_FAILED" for r in ages),
        "mature_training_months": sum(r["eligible_for_fit"] for r in months), "no_view_fit_origins": sum(r["status"] != "MONTH_COMPLETE" for r in months),
        "no_view_age_cells": sum(r["status"] != "FIT_COMPLETE" for r in ages), "reference_sale_feasibility_rows": len(cashflows),
        "blocked_reference_sale_days": int(cashflows.filled_quantity.eq(0).sum()), "delayed_immediate_reference_exits": int(samples.immediate_exit_index.gt(samples.origin_index+1).sum()),
        "all_metrics": main, "earlier_diagnostics": earlier, "model_coverage": coverage, "primary": [m for m in new_main if m["model"] == PRIMARY],
        "post_selected_best_base": max((m for m in new_main if m["cost"] == "BASE"), key=lambda m: m["net_sharpe"] if m["net_sharpe"] is not None else -np.inf),
        "historical_point_target_met": any(m["meets_point_target"] for m in new_main), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": new_main, "较早": [m for m in earlier if m["model"] in METHODS], "耗时": result["run_seconds"],
                      "共同求解": result["new_model_fits"], "标量模型": result["scalar_target_models"], "求解失败": result["failed_fits"],
                      "成熟参考受阻日": result["blocked_reference_sale_days"], "成熟参考提前退出延后行": result["delayed_immediate_reference_exits"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
