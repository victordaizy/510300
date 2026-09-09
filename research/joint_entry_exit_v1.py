"""固定两个联合动作设置，复用历史时钟并运行八个完整账户。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import require, now, digest, write_json
from research.joint_state_action_inputs_v1 import market_state_frame
from research.joint_entry_exit_inputs_v1 import PRIMARY, CONTROL, make_cases, fit_month, JointController
from research.joint_entry_exit_account_v1 import simulate_joint_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_joint_entry_exit_v1"
CONFIG = ROOT / "config/510300_joint_entry_exit_v1.json"
NAMES = {PRIMARY: "联合进入持有退出", CONTROL: "只看单日奖励进出"}
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P114 = ROOT / "reports/research/510300_within_cycle_exit_v1"
CONTROLS = {"WITHIN_CYCLE_EXIT": (P114, "第114轮周期内退出"), "REARM_RIDGE": (P32, "原平均继续收益退出"),
            "CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "第91轮局部候选"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "联合策略已登记，不重复冻结")
    previous = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: previous[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 9, "联合状态必要测试未通过")
    cfg.update(study_id="510300_JOINT_ENTRY_EXIT_V1", round=121, registered_at=now(), primary=PRIMARY, candidate_configurations=2,
        methods=NAMES, training_days=504, minimum_market_observations=20, discount=.99, improvement_tolerance=1e-12,
        equation_residual_limit=1e-10, maximum_policy_iterations=50, model_dimensions=12,
        market_factors=["sma120严格为正", "mom5严格为正"], account_modes=["空仓", "可自由决策持仓", "已锁定待卖持仓"],
        shared_admission="BOTH_METHODS_NO_VIEW_IF_ANY_REQUIRED_WINDOW_OR_PRIMARY_SOLVE_FAILS",
        nominal_scenario_limit="名义单日状态未编码历史剩余现金、已有应收和净资产水平，连续实际账户完整保留",
        missing_decision="NO_VIEW_KEEP_ACTUAL_SHARES_WITH_LOCKED_EXIT_CONTINUATION",
        terminal="LAST_DAY_OPEN_LIQUIDATION_PRIORITY", old_cooldown_or_rearm=False,
        rules="docs/510300_JOINT_ENTRY_EXIT_V1.md",
        saved_clocks="reports/research/510300_learned_cycle_exit_v1/saved_models.json",
        saved_states="reports/research/510300_joint_state_action_support_20260908/market_states.parquet",
        support_receipt="reports/research/510300_joint_state_action_support_20260908/result.json",
        new_reference_accounts=0, source_budget_cny=0, position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED")
    paths = [Path(__file__), ROOT / "research/joint_entry_exit_inputs_v1.py", ROOT / "research/joint_entry_exit_account_v1.py",
        ROOT / "research/joint_state_action_inputs_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py",
        ROOT / "tests/test_joint_entry_exit_v1.py", OUT / "tests_receipt.json", ROOT / "config/510300_research_authority_v6.json",
        ROOT / "config/510300_rearmed_session_exit_v1.json"]
    paths.extend(ROOT / cfg[k] for k in ["rules", "features", "dividends", "saved_clocks", "saved_states", "support_receipt"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{name}_ledger.parquet" for name, (parent, label) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第121轮联合状态策略及单步对照已冻结，尚未读取新动作收益或账户业绩。", flush=True)


def run():
    began = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "联合策略冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    states = market_state_frame(data)
    pd.testing.assert_frame_equal(states, pd.read_parquet(ROOT / cfg["saved_states"]))
    states.to_parquet(OUT / "market_states.parquet", index=False)
    schedule = json.loads((ROOT / cfg["saved_clocks"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    clocks = [int(m["fit_index"]) for m in schedule]
    cases = make_cases(data, dividends, states, clocks, cfg)
    cases.to_parquet(OUT / "one_step_cases.parquet", index=False)
    models = []
    with threadpool_limits(limits=1):
        for original in schedule:
            model = fit_month(cases, int(original["fit_index"]), cfg)
            model["fit_origin"] = str(data.date.iloc[model["fit_index"]].date())
            models.append(model)
    write_json(OUT / "saved_models.json", {"models": models, "nominal_one_step_cases": len(cases)}, exclusive=True)
    complete = [m for m in models if m["status"] == "FIT_COMPLETE"]
    linear_solves = sum(m.get("solution", {}).get("linear_solves", 0) for m in models)
    pd.DataFrame([{"fit_index": m["fit_index"], "fit_origin": m["fit_origin"], "status": m["status"],
        "linear_solves": m.get("solution", {}).get("linear_solves", 0)} for m in models]).to_csv(OUT / "monthly_fit_coverage.csv", index=False, encoding="utf-8-sig")
    print(f"{len(cases)}个名义单步情景，{len(models)}个月模型，{len(complete)}个月共同可用，{linear_solves}次小方程求解。", flush=True)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        period_states = states.iloc[:len(frame)].copy()
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        require(np.isfinite(frame[["open", "close", "previous_close", "dividend"]].iloc[first-1:].to_numpy(float)).all(), "真实账户必要价格缺失，不能编造收益")
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            accounts, names = {}, dict(NAMES)
            for method in NAMES:
                controller = JointController(period_states, models, method)
                ledger, decisions, cycles = simulate_joint_policy(frame, dividends, cfg, cost, start, controller)
                save_account(folder, method, ledger, decisions)
                cycles.to_csv(folder / f"{method}_cycles.csv", index=False, encoding="utf-8-sig")
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "联合账户终点或财富核算不符")
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
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "JOINT_ENTRY_EXIT_ACCOUNTS_COMPLETE", "candidate_configurations": 2,
        "evaluation_accounts": len(main), "new_accounts_generated": 4, "reused_control_accounts": 8, "earlier_diagnostic_accounts": len(earlier),
        "new_earlier_diagnostic_accounts": 4, "reused_earlier_accounts": 8, "new_model_fits": len(models), "new_reference_accounts": 0,
        "nominal_one_step_cases": len(cases), "distinct_training_origins": int(cases.origin_index.nunique()), "complete_monthly_models": len(complete),
        "policy_tables_saved": len(complete)*2, "policy_evaluation_linear_solves": linear_solves,
        "all_metrics": main, "earlier_diagnostics": earlier, "account_coverage": coverage, "primary": [m for m in candidates if m["model"] == PRIMARY],
        "post_selected_best_base": max([m for m in candidates if m["cost"] == "BASE"], key=lambda m: m["net_sharpe"] if m["net_sharpe"] is not None else -np.inf),
        "historical_point_target_met": any(m["meets_point_target"] for m in candidates), "run_seconds": time.perf_counter()-began,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主历史": candidates, "较早历史": [m for m in earlier if m["model"] in NAMES], "耗时": result["run_seconds"], "覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
