"""第146轮使用日内高低区间和隔夜风险约束143的目标。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.range_overnight_risk_inputs_v1 import range_overnight_frames, PRIMARY, MODELS
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_range_overnight_risk_v1"
CONFIG = ROOT / "config/510300_range_overnight_risk_v1.json"
P143 = ROOT / "reports/research/510300_trend_noise_reference_blend_v1"
P145 = ROOT / "reports/research/510300_trend_coherence_blend_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {MODELS[0]: (P143, "第143轮趋势波动连续预算"), "TREND_COHERENCE_BLEND": (P145, "第145轮趋势连贯预算"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "区间隔夜风险已经冻结")
    preflight_path = ROOT / "reports/research/510300_range_overnight_risk_preflight_20260909/result.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(preflight["status"] == "RANGE_OVERNIGHT_DIVIDEND_IDENTITIES_AND_PARENT_TARGET_CLOCKS_READY" and
        preflight["new_risk_multipliers_or_accounts"] == 0, "现有高低区间与隔夜口径未完成确认")
    for source in preflight["sources"]:
        require(digest(ROOT / source["path"]) == source["sha256"], "现有高低区间或保存目标来源改变")
    old_path = ROOT / "config/510300_trend_coherence_blend_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    keys = ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band", "target_volatility"]
    cfg = {key: old[key] for key in keys}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 5, "区间隔夜风险五项必要测试未通过")
    cfg.update(study_id="510300_RANGE_OVERNIGHT_RISK_V1", round=146, primary=PRIMARY, candidate_configurations=1,
        decision_clock="15:05:00", combination="PARENT143_TIMES_RANGE_PLUS_OVERNIGHT_RISK_MULTIPLIER", risk_window=20,
        registered_at=now(), new_model_fits=0, new_reference_accounts=0,
        reference_cost_matching="SAME_PERIOD_AND_SAME_COST", parent_models=MODELS,
        outer_exit_retry="RECOMPUTE_FROM_LATEST_TARGET_EACH_CLOSE", reentry="ANY_NEW_POSITIVE_TARGET_NO_ADDITIONAL_WAIT",
        rules="docs/510300_RANGE_OVERNIGHT_RISK_V1.md", source_budget_cny=0,
        independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUNDS144_145_COMPLETED_CLOSED_INPUT146_READY_FULL_GOAL_NOT_MET")
    paths = [ROOT / item["path"] for item in preflight["sources"]]
    paths.extend([Path(__file__), preflight_path, ROOT / "scripts/preflight_range_overnight_risk_20260909.py",
        ROOT / "research/range_overnight_risk_inputs_v1.py", ROOT / "research/saved_parent_target_alignment_v1.py",
        ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "tests/test_range_overnight_risk_v1.py", ROOT / "tests/test_trend_reference_router_v1.py", OUT / "tests_receipt.json",
        ROOT / cfg["rules"], ROOT / "config/510300_research_authority_v6.json"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    cfg["existing_input_checks"] = preflight["checks"]
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第146轮区间隔夜风险已冻结，尚未生成新目标或账户。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "区间隔夜风险冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    began = time.perf_counter()
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    main, earlier, yearly, eras, coverage, target_summaries = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        source_folders = [(MODELS[0], P143)]
        parents_by_cost = {cost_id: {model: pd.read_parquet(parent_folder / period / cost_id / f"{model}_decisions.parquet").assign(source_cost=cost_id, source_model=model)
            for model, parent_folder in source_folders} for cost_id in cfg["costs"]}
        factor_frames, summaries = range_overnight_frames(frame, parents_by_cost, cfg, start)
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
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "区间隔夜风险账户财富或终点清仓不符")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()),
                "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "unknown_target_origins": int(decisions.reference_weight.isna().sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()), "mean_exposure": float(ledger.exposure.mean())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "按日内区间和隔夜风险缩放143仓位"}
            for model, (parent_folder, name) in CONTROLS.items():
                saved = pd.read_parquet(parent_folder / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            benchmark = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "区间隔夜风险与保存对照日历不同")
                measured = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                measured["annualized_return_excess_vs_buy_hold"] = measured["annualized_return"]-benchmark["annualized_return"]
                measured["meets_point_target"] = measured["net_sharpe"] is not None and measured["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(measured)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：一条新区间隔夜风险账户及三条保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage), ("target_coverage.csv", target_summaries)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "RANGE_OVERNIGHT_RISK_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 6, "earlier_diagnostic_accounts": len(earlier),
        "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6, "new_model_fits": 0, "new_reference_accounts": 0,
        "all_metrics": main, "earlier_diagnostics": earlier, "account_coverage": coverage, "target_coverage": target_summaries,
        "primary": primary, "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "run_seconds": time.perf_counter()-began,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主历史": primary, "较早历史": [m for m in earlier if m["model"] == PRIMARY],
        "账户": coverage, "区间隔夜风险": target_summaries, "核心耗时": result["run_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
