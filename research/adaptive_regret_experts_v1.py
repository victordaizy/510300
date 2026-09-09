"""复用原连续参考，逐日自适应后悔比较只计算一次，再运行四个独立账户。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.adaptive_regret_experts_inputs_v1 import adaptive_regret_frame, prefix_factors
from research.intraday_overnight_increment_v1 import require, now, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_adaptive_regret_experts_v1"
CONFIG = ROOT / "config/510300_adaptive_regret_experts_v1.json"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P114 = ROOT / "reports/research/510300_within_cycle_exit_v1"
PRIMARY = "ADAPTIVE_REGRET_EXPERTS"
NAME = "不同起点自适应策略组合"
CONTROLS = {"CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "第91轮连续参考最小方差"),
    "WITHIN_CYCLE_EXIT": (P114, "第114轮周期内退出"), "REARM_RIDGE": (P32, "原平均继续收益退出"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "自适应组合已经登记，不重复冻结")
    parent_path = ROOT / "config/510300_continuous_reference_min_variance_v1.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    cfg = {k: parent[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
        "weight_band", "reference_start", "panic_spec", "learned_spec", "confirmation_days", "saved_models"]}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 11, "自适应比较十一项必要测试未通过")
    cfg.update(study_id="510300_ADAPTIVE_REGRET_EXPERTS_V1", round=126, registered_at=now(), primary=PRIMARY,
        candidate_configurations=1, experts=["PANIC_ONLY", "REARM_RIDGE", "CASH_CNY"], state_cost="BASE",
        method="ADANORMALHEDGE_WITH_MONTHLY_SLEEPING_COHORTS", potential_denominator=3., unnormalized_prior_per_record=1.,
        loss_transform="ONE_MINUS_NET_REFERENCE_RETURN_DIVIDED_BY_TWO", allowed_return_interval="(-1,1]",
        birth_clock="INITIAL_REFERENCE_ANCHOR_AND_MONTH_FIRST_COMPLETE_CLOSE", learning_clock="PREVIOUS_CLOSE_WEIGHTS_THEN_OBSERVE_UPDATE_THEN_BIRTH",
        initial_budgets=[1./3., 1./3., 1./3.], zero_weight_rule="EQUAL_ACTIVE_RECORDS", numerical_tolerance=1e-12,
        missing_rule="NO_VIEW_REMAINDER_NO_RESET_NO_CASH_SUBSTITUTION", rules="docs/510300_ADAPTIVE_REGRET_EXPERTS_V1.md",
        input_receipt="reports/research/510300_adaptive_regret_experts_preflight_20260909/result.json",
        source_factors={period: str((P91 / f"{period}_factors.parquet").relative_to(ROOT)) for period in ["evaluation", "earlier_diagnostic"]},
        shared_prefix_learned_once=True, new_model_fits=0, new_reference_accounts=0, source_budget_cny=0,
        independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        previous_goal_turn_classification="PROGRESS_ROUND125_COMPLETED_FOUR_ACCOUNTS_AND_126_METHOD_REVIEW")
    paths = [Path(__file__), parent_path, ROOT / "research/adaptive_regret_experts_inputs_v1.py",
        ROOT / "research/event_clock_account_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "tests/test_adaptive_regret_experts_v1.py",
        OUT / "tests_receipt.json", ROOT / "config/510300_research_authority_v6.json", ROOT / "docs/510300_PANIC_LEARNED_EQUAL_BLEND_V1.md"]
    paths.extend(ROOT / cfg[key] for key in ["rules", "input_receipt", "features", "dividends", "saved_models"])
    receipt = json.loads((ROOT / cfg["input_receipt"]).read_text(encoding="utf-8"))
    common = ROOT / receipt["reused_completeness_receipt"]
    require(digest(common) == receipt["reused_completeness_receipt_sha256"], "已核对的连续参考回执改变")
    paths.append(common)
    for item in json.loads(common.read_text(encoding="utf-8"))["sources"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "预检后连续参考来源改变")
        paths.append(ROOT / item["path"])
    for reference in ["PANIC_ONLY", "REARM_RIDGE"]:
        paths.extend(P91 / "continuous_references/BASE" / f"{reference}_{kind}.parquet" for kind in ["ledger", "decisions"])
    for period in cfg["source_factors"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{model}_ledger.parquet" for model, (parent, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(path.relative_to(ROOT)), "sha256": digest(path)} for path in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第126轮单一自适应组合已冻结，尚未计算新比较权重或实际账户。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "自适应组合冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    began = time.perf_counter()
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    sources = {period: pd.read_parquet(ROOT / path, columns=["date", "panic_reference_return", "learned_reference_return", "panic_state", "learned_state"])
               for period, path in cfg["source_factors"].items()}
    source = sources["evaluation"]
    require(pd.DatetimeIndex(source.date).equals(pd.DatetimeIndex(data.date)), "连续参考和行情日期不一致")
    first_reference = int(np.flatnonzero(source.date.ge(cfg["reference_start"]))[0])
    factors, records = adaptive_regret_frame(source.date, source[["panic_reference_return", "learned_reference_return"]].to_numpy(float),
        source[["panic_state", "learned_state"]].to_numpy(float), first_reference)
    factors.to_parquet(OUT / "full_continuous_factors.parquet", index=False)
    np.savez_compressed(OUT / "saved_comparison_records.npz", **records)
    factors[factors.birth_added].to_csv(OUT / "unique_birth_records.csv", index=False, encoding="utf-8-sig")
    learning_summary = {"observed_daily_updates": int(factors.updated_records.gt(0).sum()),
        "internal_comparison_groups": len(records["group_birth_index"]), "maximum_active_records": int(factors.active_groups.max())*3,
        "internal_record_updates": int(factors.updated_records.sum()), "saved_record_snapshots": len(records["weight"]),
        "zero_weight_fallback_days": int(factors.zero_weight_fallback.sum()), "status_counts": factors.learning_status.value_counts().to_dict()}
    print(f"连续参考比较完成：{learning_summary['observed_daily_updates']}个逐日更新，{learning_summary['internal_comparison_groups']}组内部记录。", flush=True)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        selected = prefix_factors(factors, sources[period])
        require(pd.DatetimeIndex(selected.date).equals(pd.DatetimeIndex(frame.date)), "裁切后权重日期不一致")
        selected.to_parquet(OUT / f"{period}_factors.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, PRIMARY,
                targets=selected.target.to_numpy(float), event_mask=np.ones(len(frame), bool))
            decisions = decisions.merge(selected.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "自适应组合账户经济核算或终点结算失败")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()),
                "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                "no_view_target_origins": int(decisions.reference_weight.isna().sum()), "mean_exposure": float(ledger.exposure.mean())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for model, (parent, label) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, label
            benchmark = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "自适应组合策略及保存对照日历不同")
                measured = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                measured["annualized_return_excess_vs_buy_hold"] = measured["annualized_return"]-benchmark["annualized_return"]
                measured["meets_point_target"] = measured["net_sharpe"] is not None and measured["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(measured)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：一个新账户和四个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [row for row in main if row["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "ADAPTIVE_REGRET_EXPERTS_ACCOUNTS_COMPLETE",
        "candidate_configurations": 1, "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8,
        "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8,
        "new_model_fits": 0, "new_reference_accounts": 0, "learning_summary": learning_summary, "all_metrics": main, "earlier_diagnostics": earlier,
        "account_coverage": coverage, "primary": primary, "post_selected_best_base": next(row for row in primary if row["cost"] == "BASE"),
        "historical_point_target_met": any(row["meets_point_target"] for row in primary), "run_seconds": time.perf_counter()-began,
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主结果": primary, "较早結果": [row for row in earlier if row["model"] == PRIMARY], "学习记录": learning_summary,
        "账户覆盖": coverage, "核心耗时": result["run_seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
