"""有限观察状态均值与同口径合并对照，直接生成真实费用账户。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.observed_return_state_inputs_v1 import MODES, label_frame, estimate_schedule, ObservedStateController
from research.observed_return_state_account_v1 import simulate_observed_state_account
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_observed_return_state_v1"
CONFIG = ROOT / "config/510300_observed_return_state_v1.json"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PRIMARY = MODES[0]
NAMES = {MODES[0]: "当日涨跌状态均值进出场", MODES[1]: "不分状态的同口径均值对照"}
CONTROLS = {"CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "原连续参考最小方差"), "REARM_RIDGE": (P32, "原线性学习退出"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "观察状态方案已经登记")
    old_path = ROOT / "config/510300_conditional_variance_budget_v1.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days", "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "model_schedule", "planned_fit_origins"]}
    cfg.update(study_id="510300_OBSERVED_RETURN_STATE_V1", round=111, registered_at=now(), primary=PRIMARY, candidate_configurations=2, modes=MODES,
               training_window=242, minimum_state_rows=60, model_views_per_estimation=2, new_reference_accounts=0,
               rules="docs/510300_OBSERVED_RETURN_STATE_V1.md", label_clock="NEXT_OPEN_TO_SUBSEQUENT_OPEN_AND_DIVIDEND_RIGHTS_MATURED",
               state_rule="NEGATIVE_OR_NONNEGATIVE_CURRENT_TOTAL_RETURN", decision_price_proxy="CURRENT_KNOWN_CLOSE", actual_execution="NEXT_ACTUAL_OPEN",
               source_budget_cny=0, goal_achieved=False, position_impact=0, independent_validation="NOT_ESTABLISHED",
               environment={"numpy": np.__version__, "pandas": pd.__version__})
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 7, "观察状态必要测试未通过")
    paths = [Path(__file__), ROOT / "research/observed_return_state_inputs_v1.py", ROOT / "research/observed_return_state_account_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "tests/test_observed_return_state_v1.py",
             OUT / "tests_receipt.json", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"], ROOT / cfg["model_schedule"],
             ROOT / "config/510300_research_authority_v6.json", old_path]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(p / period / cost / f"{model}_ledger.parquet" for model, (p, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第111轮两个观察状态设置已冻结，尚未计算目标、均值或新账户收益。", flush=True)


def run():
    clock = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "观察状态冻结输入改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    schedule = json.loads((ROOT / cfg["model_schedule"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    require(len(schedule) == cfg["planned_fit_origins"], "原月度日程数量不同")
    labels = label_frame(data, dividends)
    labels.to_parquet(OUT / "open_interval_labels.parquet", index=False)
    records = estimate_schedule(data, labels, schedule, cfg)
    write_json(OUT / "saved_state_estimates.json", {"monthly_estimates": records, "model_views_per_estimation": 2}, exclusive=True)
    flattened = []
    for record in records:
        for state in ["0", "1", "pooled"]:
            flattened.append({k: v for k, v in record.items() if k not in {"counts", "estimates"}} | {"state": state, "negative_rows": record["counts"]["0"], "nonnegative_rows": record["counts"]["1"], **((record["estimates"] or {}).get(state, {}))})
    pd.DataFrame(flattened).to_csv(OUT / "每月成熟窗口与状态均值.csv", index=False, encoding="utf-8-sig")
    print(f"141个月度估计完成，可用{sum(r['status']=='ESTIMATION_COMPLETE' for r in records)}个，同时保存分状态与合并视角。", flush=True)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main), ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            accounts, names = {}, NAMES.copy()
            for model in MODES:
                controller = ObservedStateController(frame, records, cost, cfg, model)
                ledger, decisions = simulate_observed_state_account(frame, dividends, cfg, cost, start, model, controller)
                save_account(folder, model, ledger, decisions)
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "观察状态账户未完整结算")
                coverage.append({"period": period, "cost": cost_id, "model": model, "holding_closes": int(ledger.shares.gt(0).sum()), "buy_trades": int(ledger.filled_quantity.gt(0).sum()),
                                 "sell_trades": int(ledger.filled_quantity.lt(0).sum()), "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                                 "no_view_origins": int(decisions.signal_state.str.startswith("NO_VIEW").sum()), "entry_requests": int(decisions.requested_quantity.gt(0).sum()),
                                 "exit_requests": int(decisions.requested_quantity.lt(0).sum()), "mean_exposure": float(ledger.exposure.mean())})
                accounts[model] = ledger
            for model, (p, name) in CONTROLS.items():
                saved = pd.read_parquet(p / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "观察状态账户与对照日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"]-bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                dest.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：两个实际新账户及三个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "OBSERVED_RETURN_STATE_ACCOUNTS_COMPLETE", "candidate_configurations": 2,
              "evaluation_accounts": len(main), "new_accounts_generated": 4, "reused_control_accounts": 6, "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 4,
              "reused_earlier_accounts": 6, "new_model_fits": len(records), "model_views_per_estimation": 2, "completed_fits": sum(r["status"] == "ESTIMATION_COMPLETE" for r in records),
              "no_view_fit_origins": sum(r["status"] != "ESTIMATION_COMPLETE" for r in records), "new_reference_accounts": 0, "run_seconds": time.perf_counter()-clock,
              "all_metrics": main, "earlier_diagnostics": earlier, "primary": [m for m in main if m["model"] == PRIMARY], "account_coverage": coverage,
              "post_selected_best_base": max((m for m in main if m["model"] in MODES and m["cost"] == "BASE"), key=lambda m: float('-inf') if m["net_sharpe"] is None else m["net_sharpe"]),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] in MODES), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": [m for m in main if m["model"] in MODES], "较早": [m for m in earlier if m["model"] in MODES], "月度估计成功": result["completed_fits"], "耗时": result["run_seconds"], "覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
