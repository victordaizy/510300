"""第139轮按已形成的模型训练支持选择已有策略。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.model_support_reference_router_inputs_v1 import support_routed_frames, PRIMARY, MODELS
PARENT = MODELS[0]
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_model_support_reference_router_v1"
CONFIG = ROOT / "config/510300_model_support_reference_router_v1.json"
P131 = ROOT / "reports/research/510300_vintage_reference_risk_v1"
P137 = ROOT / "reports/research/510300_joint_downside_reference_pair_v1"
P132 = ROOT / "reports/research/510300_downside_reference_risk_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {MODELS[0]: (P131, "第131轮普通波动乘数策略"), MODELS[1]: (P137, "第137轮合成下行预算"),
    "DOWNSIDE_REFERENCE_RISK": (P132, "第132轮下行风险组合"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "模型训练支持选择已经冻结")
    old_path = ROOT / "config/510300_joint_downside_reference_pair_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band", "target_volatility"]
    cfg = {k: old[k] for k in keys}
    old_hashes = {str(Path(v["path"])): v["sha256"] for v in old["frozen_files"]}
    for key in ["features", "dividends"]:
        require(digest(ROOT / cfg[key]) == old_hashes[str(Path(cfg[key]))], "训练支持选择行情或分红不再等于已核对来源")
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "训练支持选择六项必要测试未通过")
    cfg.update(study_id="510300_MODEL_SUPPORT_REFERENCE_ROUTER_V1", round=139, primary=PRIMARY, candidate_configurations=1,
        decision_clock="15:05:00", minimum_training_cycles=10, minimum_training_rows=100, saved_models="reports/research/510300_within_cycle_exit_v1/saved_models.json", registered_at=now(), new_model_fits=0, new_reference_accounts=0,
        reference_cost_matching="SAME_PERIOD_AND_SAME_COST", parent_models=MODELS,
        outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
        rules="docs/510300_MODEL_SUPPORT_REFERENCE_ROUTER_V1.md", input_receipt="reports/research/510300_model_support_router_preflight_20260909/result.json",
        source_budget_cny=0, independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUND138_FOUR_ACCOUNTS_VERIFIED_CLOSED_AND_MODEL_SUPPORT_CLOCKS_READY")
    paths = [Path(__file__), old_path, ROOT / "research/model_support_reference_router_inputs_v1.py", ROOT / "research/event_clock_account_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "tests/test_model_support_reference_router_v1.py",
        OUT / "tests_receipt.json", ROOT / "config/510300_research_authority_v6.json", ROOT / cfg["features"], ROOT / cfg["dividends"],
        ROOT / cfg["rules"], ROOT / cfg["input_receipt"]]
    receipt = json.loads((ROOT / cfg["input_receipt"]).read_text(encoding="utf-8"))
    for item in receipt["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "训练支持选择已绑定的父目标或模型记录变化")
        paths.append(ROOT / item["path"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第139轮模型训练支持选择已冻结，尚未生成新目标或账户。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "训练支持选择冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    began = time.perf_counter()
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    main, earlier, yearly, eras, coverage, target_summaries = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        source_folders = [(MODELS[0], P131), (MODELS[1], P137)]
        records = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]
        require(len(records) == 141 and sum(r["status"] == "FIT_COMPLETE" for r in records) == 114, "原模型训练支持数量改变")
        parents_by_cost = {cost_id: {model: pd.read_parquet(parent_folder / period / cost_id / f"{model}_decisions.parquet").assign(source_cost=cost_id, source_model=model)
            for model, parent_folder in source_folders} for cost_id in cfg["costs"]}
        factor_frames, summaries = support_routed_frames(frame, records, parents_by_cost, cfg, start)
        target_summaries.extend({"period": period, **item} for item in summaries)
        base_factors = factor_frames["BASE"]
        base_factors[base_factors.selection_changed].to_csv(OUT / f"{period}_selection_changes.csv", index=False, encoding="utf-8-sig")
        for cost_id, cost in cfg["costs"].items():
            factors = factor_frames[cost_id]
            folder = OUT / period / cost_id
            folder.mkdir(parents=True)
            factors.to_parquet(folder / "factors.parquet", index=False)
            ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, PRIMARY,
                targets=factors.target.to_numpy(float), event_mask=np.ones(len(frame), bool))
            decisions = decisions.merge(factors.rename(columns={"date": "origin", "origin_index": "factor_origin_index", "model": "factor_model"}), on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "训练支持选择账户财富或终点清仓不符")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()),
                "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "unknown_target_origins": int(decisions.reference_weight.isna().sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()), "mean_exposure": float(ledger.exposure.mean())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "按退出模型的训练支持选择策略"}
            for model, (parent_folder, name) in CONTROLS.items():
                saved = pd.read_parquet(parent_folder / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            benchmark = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "训练支持选择与保存对照日历不同")
                measured = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                measured["annualized_return_excess_vs_buy_hold"] = measured["annualized_return"]-benchmark["annualized_return"]
                measured["meets_point_target"] = measured["net_sharpe"] is not None and measured["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(measured)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：一条新训练支持选择账户及四条保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage), ("target_coverage.csv", target_summaries)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "MODEL_SUPPORT_REFERENCE_ROUTER_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8, "earlier_diagnostic_accounts": len(earlier),
        "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8, "new_model_fits": 0, "new_reference_accounts": 0,
        "all_metrics": main, "earlier_diagnostics": earlier, "account_coverage": coverage, "target_coverage": target_summaries,
        "primary": primary, "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "run_seconds": time.perf_counter()-began,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主历史": primary, "较早历史": [m for m in earlier if m["model"] == PRIMARY],
        "账户": coverage, "训练支持选择": target_summaries, "核心耗时": result["run_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
