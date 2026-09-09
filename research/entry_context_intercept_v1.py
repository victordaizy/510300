"""复用第一层模型，拟合买入前上下文并完成四个历史账户。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.entry_context_intercept_inputs_v1 import FEATURES, CN, CONTEXT_FEATURES, CONTEXT_CN, cycle_entry_context, select_context, fit_entry_context_intercept, EntryContextInterceptController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_entry_context_intercept_v1"
CONFIG = ROOT / "config/510300_entry_context_intercept_v1.json"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P114 = ROOT / "reports/research/510300_within_cycle_exit_v1"
PRIMARY = "ENTRY_CONTEXT_INTERCEPT"
CONTROLS = {"WITHIN_CYCLE_EXIT": (P114, "第114轮平均周期截距"), "REARM_RIDGE": (P32, "原平均继续收益学习退出"),
            "CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "第91轮局部候选"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "本轮已登记，不能重复冻结")
    old = json.loads((ROOT / "config/510300_within_cycle_exit_v1.json").read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days", "cash_annual_rate_assumption",
            "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "confirmation_days", "specification",
            "recent_cycles", "minimum_cycles", "minimum_rows", "feature_clip"]
    cfg = {key: old[key] for key in keys}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "两层分离及买入前时钟必要测试未通过")
    cfg.update(study_id="510300_ENTRY_CONTEXT_INTERCEPT_V1", round=115, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
               saved_models=str((P114 / "saved_models.json").relative_to(ROOT)), reference_cycles="reports/research/510300_learned_cycle_exit_v1/reference/D60_INTRA_cycles.csv",
               preflight_receipt="reports/research/510300_entry_context_intercept_preflight_20260908/result.json",
               context_ridge_alpha=1., context_fit_intercept=True, solver="svd", context_cycle_weight=1.,
               feature_columns=FEATURES.copy(), feature_names=CN.copy(), context_feature_columns=CONTEXT_FEATURES.copy(), context_feature_names=CONTEXT_CN.copy(),
               rules="docs/510300_ENTRY_CONTEXT_INTERCEPT_V1.md", new_reference_accounts=0, planned_model_fits=114, new_within_cycle_model_fits=0,
               missing_training_context="WHOLE_MONTH_NO_VIEW_NO_ROW_DROP", missing_prediction="NO_VIEW_NO_AVERAGE_INTERCEPT_FALLBACK",
               context_target="SAVED_MATURE_CYCLE_INTERCEPT", context_clock="ACTUAL_BUY_REQUEST_PREVIOUS_CLOSE",
               source_budget_cny=0, position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED",
               previous_goal_turn_classification="PROGRESS_ROUND114_LOCAL_GAIN_COMPLETED",
               environment={"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__})
    paths = [Path(__file__), ROOT / "research/entry_context_intercept_inputs_v1.py", ROOT / "research/within_cycle_exit_inputs_v1.py",
             ROOT / "research/rearmed_cycle_exit_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/simple_intraday_protection_v1.py",
             ROOT / "research/simple_session_divergence_v1.py", ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "config/510300_research_authority_v6.json", ROOT / "config/510300_within_cycle_exit_v1.json",
             ROOT / "tests/test_entry_context_intercept_v1.py", ROOT / "tests/test_within_cycle_exit_v1.py", ROOT / "tests/test_median_continuation_v1.py",
             OUT / "tests_receipt.json", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"], ROOT / cfg["reference_cycles"],
             ROOT / cfg["preflight_receipt"], ROOT / cfg["rules"]]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{model}_ledger.parquet" for model, (parent, name) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第115轮一个进入上下文截距设置已冻结，尚未拟合第二层或计算新账户。", flush=True)


def train_models(context, originals, cfg):
    models, receipts, memberships, coefficients = [], [], [], []
    for old in originals:
        eligible = old["status"] == "FIT_COMPLETE"
        model, failure, status, missing = None, None, old["status"], 0
        if eligible:
            selected = select_context(old, context)
            missing = int((~np.isfinite(selected[CONTEXT_FEATURES].to_numpy(float)).all(axis=1)).sum())
            status = "NO_VIEW_INCOMPLETE_TRAINING_CONTEXT" if missing else "FIT_COMPLETE"
            if not missing:
                try:
                    model = fit_entry_context_intercept(old, context, cfg)
                except (RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                    status, failure = "NO_VIEW_MODEL_FIT_FAILED", str(error)
            memberships.extend({"fit_index": old["fit_index"], "fit_origin": old["fit_origin"], "fit_status": status, **row}
                               for row in selected.reset_index(drop=True).to_dict("records"))
        record = {key: old[key] for key in ["fit_index", "fit_origin", "fit_time", "training_cycles", "training_cycle_count", "training_rows", "latest_exit_index", "latest_exit_date"]}
        record.update(status=status, eligible_for_fit=eligible, missing_context_cycles=missing, failure=failure, model=model)
        models.append(record)
        receipts.append({key: value for key, value in record.items() if key not in ["model", "training_cycles"]})
        if model:
            m = model["context_model"]
            coefficients.extend({"拟合收盘": old["fit_origin"], "成熟周期数": len(old["training_cycles"]), "因子": name, "训练均值": mean,
                                 "训练标准差": scale, "标准化系数": coefficient, "统一截距": m["intercept"]}
                                for name, mean, scale, coefficient in zip(CONTEXT_CN, m["mean"], m["scale"], m["coefficients"], strict=True))
    require(sum(r["eligible_for_fit"] for r in receipts) == cfg["planned_model_fits"], "原成熟拟合支持次数改变")
    write_json(OUT / "saved_models.json", {"models": models, "context_features": CONTEXT_FEATURES, "within_models_refitted": 0}, exclusive=True)
    pd.DataFrame(receipts).to_csv(OUT / "training_receipts.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(memberships).to_parquet(OUT / "training_memberships.parquet", index=False)
    pd.DataFrame(coefficients).to_csv(OUT / "每月进入上下文系数与标准化.csv", index=False, encoding="utf-8-sig")
    print(f"复用第一层：第二层完成{sum(r['status']=='FIT_COMPLETE' for r in receipts)}次，原支持不足或失败{sum(r['status']!='FIT_COMPLETE' for r in receipts)}次。", flush=True)
    return models, receipts


def run():
    clock = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "进入上下文冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    context = cycle_entry_context(data, pd.read_csv(ROOT / cfg["reference_cycles"]))
    context.reset_index(drop=True).to_parquet(OUT / "reference_entry_context.parquet", index=False)
    originals = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]
    models, receipts = train_models(context, originals, cfg)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
                                     ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        rule = make_rules(frame)["D60_INTRA"]
        pd.DataFrame({"date": frame.date, "entry_condition": rule["entry"], "original_price_exit": rule["exit"][1]}).to_parquet(OUT / f"{period}_entry_exit_conditions.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_rearmed_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"], EntryContextInterceptController(frame, models, cfg["confirmation_days"]))
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "两层退出账户未完整结算")
            require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "两层退出违反买入次日可卖")
            holding = decisions[decisions.learning_cycle_id.notna()]
            coverage.append({"period": period, "cost": cost_id, "holding_decisions": len(holding),
                             "model_available_rows": int(holding.learning_status.eq("PREDICTION_AVAILABLE").sum()),
                             "no_model_rows": int(holding.learning_status.ne("PREDICTION_AVAILABLE").sum()),
                             "learned_exit_cycles": int(cycles.exit_reasons.str.contains("学习条件", regex=False).sum()), "completed_round_trips": len(cycles),
                             "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                             "waiting_for_new_condition_rows": int(decisions.action.str.contains("旧入场条件", regex=False).sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "买入前上下文加周期内继续价值"}
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "两层退出及对照完整日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"]-bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                dest.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            pd.DataFrame({"date": ledger.date, **{key: a.net_return.to_numpy() for key, a in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：一个新两层退出账户和四个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("model_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "ENTRY_CONTEXT_INTERCEPT_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8, "earlier_diagnostic_accounts": len(earlier),
              "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8, "new_reference_accounts": 0, "new_within_cycle_model_fits": 0,
              "run_seconds": time.perf_counter()-clock, "new_model_fits": sum(r["eligible_for_fit"] for r in receipts),
              "completed_fits": sum(r["status"] == "FIT_COMPLETE" for r in receipts), "no_view_fit_origins": sum(r["status"] != "FIT_COMPLETE" for r in receipts),
              "failed_fits": sum(r["status"] == "NO_VIEW_MODEL_FIT_FAILED" for r in receipts), "all_metrics": main, "earlier_diagnostics": earlier,
              "model_coverage": coverage, "primary": primary, "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": primary, "较早": [m for m in earlier if m["model"] == PRIMARY], "耗时": result["run_seconds"], "训练次数": result["new_model_fits"], "求解失败": result["failed_fits"], "持仓覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
