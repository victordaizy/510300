"""冻结单一云图规则并运行四个独立完整账户，无新模型拟合。"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import require, now, digest, write_json
from research.ichimoku_cloud_inputs_v1 import PERIODS, cloud_rule
from research.known_price_signal_account_v1 import simulate_known_price_exit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_ichimoku_cloud_v1"
CONFIG = ROOT / "config/510300_ichimoku_cloud_v1.json"
PRIMARY = "ICHIMOKU_CLOUD"
NAME = "固定云图进入与退出"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P114 = ROOT / "reports/research/510300_within_cycle_exit_v1"
CONTROLS = {"WITHIN_CYCLE_EXIT": (P114, "第114轮周期内退出"), "REARM_RIDGE": (P32, "原平均继续收益退出"),
            "CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "第91轮局部候选"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "云图规则已登记，不重复冻结")
    previous = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: previous[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 8, "云图及三值账户必要测试未通过")
    cfg.update(study_id="510300_ICHIMOKU_CLOUD_V1", round=120, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
        periods=PERIODS, floating_comparison_relative_tolerance=1e-12, confirmation_days=1,
        specification={"cooldown": 2, "modes": {"1": {"loss": None, "trail": None, "take": None, "days": None}}},
        entry_exit_conflict="KNOWN_EXIT_PREVENTS_NEW_ENTRY_WITHOUT_CONSUMING_ENTRY_RIGHT",
        missing_entry="NO_VIEW_NO_NEW_BUY_AND_NO_REARM", missing_exit="THREE_VALUED_OR_KEEP_LOCKED_EXIT",
        rules="docs/510300_ICHIMOKU_CLOUD_V1.md", saved_factors="reports/research/510300_cloud_definition_preflight_20260908/cloud_factors.parquet",
        definition_receipt="reports/research/510300_cloud_definition_preflight_20260908/result.json",
        new_model_fits=0, new_reference_accounts=0, source_budget_cny=0, position_impact=0, goal_achieved=False,
        previous_goal_turn_classification="PROGRESS_ROUNDS118_AND119_COMPLETED_TWELVE_NEW_ACCOUNTS", independent_validation="NOT_ESTABLISHED")
    paths = [Path(__file__), ROOT / "research/ichimoku_cloud_inputs_v1.py", ROOT / "research/known_price_signal_account_v1.py",
        ROOT / "research/rearmed_cycle_exit_account_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py",
        ROOT / "tests/test_ichimoku_cloud_v1.py", ROOT / "tests/test_median_continuation_v1.py", OUT / "tests_receipt.json",
        ROOT / "config/510300_research_authority_v6.json", ROOT / "config/510300_rearmed_session_exit_v1.json"]
    paths.extend(ROOT / cfg[k] for k in ["rules", "features", "dividends", "saved_factors", "definition_receipt"])
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{name}_ledger.parquet" for name, (parent, label) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第120轮单一固定云图及完整三值账户规则已冻结，尚未读取新策略收益。", flush=True)


def run():
    began = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "云图冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    factors = pd.read_parquet(ROOT / cfg["saved_factors"])
    require(factors.date.equals(data.date), "云图与价格完整日历不同")
    factors.to_parquet(OUT / "factors.parquet", index=False)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        states = factors.iloc[:len(frame)].copy()
        rule = cloud_rule(states)
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        require(np.isfinite(frame[["open", "close", "previous_close", "dividend"]].iloc[first-1:].to_numpy(float)).all(), "真实行动区间成交或估值来源缺失，不能编造收益")
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_known_price_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"])
            decisions = decisions.merge(states.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "云图账户终点或财富核算不符")
            require(cycles.empty or cycles.dropna(subset=["exit_date"]).holding_intervals.ge(1).all(), "云图退出违反次日可卖")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()), "completed_round_trips": len(cycles),
                "entry_unknown_origins": int(decisions.entry_signal.isna().sum()), "exit_unknown_origins": int(decisions.exit_signal.isna().sum()),
                "simultaneous_entry_exit_origins": int((decisions.entry_signal.eq(1) & decisions.exit_signal.eq(1)).sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                "waiting_for_new_condition_origins": int(decisions.action.str.contains("旧入场条件", regex=False).sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for name, (parent, label) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{name}_ledger.parquet")
                saved.to_parquet(folder / f"{name}_ledger.parquet", index=False)
                accounts[name], names[name] = saved, label
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for name, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "云图及保存对照日历不同")
                measured = {"cost": cost_id, "model": name, "name": names[name], **summarize(saved, cfg)}
                measured["annualized_return_excess_vs_buy_hold"] = measured["annualized_return"]-bh["annualized_return"]
                measured["meets_point_target"] = measured["net_sharpe"] is not None and measured["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(measured)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": name, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": name, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            pd.DataFrame({"date": ledger.date, **{name: saved.net_return.to_numpy() for name, saved in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：固定云图新账户及四个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "ICHIMOKU_CLOUD_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8, "earlier_diagnostic_accounts": len(earlier),
        "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8, "new_model_fits": 0, "new_reference_accounts": 0,
        "all_metrics": main, "earlier_diagnostics": earlier, "account_coverage": coverage, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"), "historical_point_target_met": any(m["meets_point_target"] for m in primary),
        "run_seconds": time.perf_counter()-began, "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主历史": primary, "较早历史": [m for m in earlier if m["model"] == PRIMARY], "耗时": result["run_seconds"], "账户覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
