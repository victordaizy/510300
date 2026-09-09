"""简易波动零线策略，零训练直接验证完整账户。"""
import json
import time
from pathlib import Path
import pandas as pd
from research.ease_of_movement_inputs_v1 import factor_frame, EaseOfMovementController, PRIMARY
from research.observed_return_state_account_v1 import simulate_observed_state_account
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_ease_of_movement_v1"
CONFIG = ROOT / "config/510300_ease_of_movement_v1.json"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONTROLS = {"CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "原连续参考最小方差"), "REARM_RIDGE": (P32, "原线性学习退出"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "简易波动策略已经登记")
    parent = ROOT / "config/510300_observed_return_state_v1.json"
    old = json.loads(parent.read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days", "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    cfg.update(study_id="510300_EASE_OF_MOVEMENT_V1", round=112, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
               averaging_window=14, volume_scale_shares=100000000, rules="docs/510300_EASE_OF_MOVEMENT_V1.md", new_model_fits=0, new_reference_accounts=0,
               source_budget_cny=0, goal_achieved=False, position_impact=0, independent_validation="NOT_ESTABLISHED")
    receipt = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(receipt["passed"] == 6 and receipt["exit_code"] == 0, "简易波动必要测试未完成")
    paths = [Path(__file__), ROOT / "research/ease_of_movement_inputs_v1.py", ROOT / "research/observed_return_state_account_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "tests/test_ease_of_movement_v1.py", OUT / "tests_receipt.json", ROOT / cfg["rules"],
             ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / "config/510300_research_authority_v6.json", parent]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(p / period / cost / f"{model}_ledger.parquet" for model, (p, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第112轮固定十四日简易波动零线设置已冻结，尚未生成新因子及新收益。", flush=True)


def run():
    clock = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "简易波动冻结输入改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    factors = factor_frame(data, cfg["averaging_window"])
    factors.to_parquet(OUT / "factors.parquet", index=False)
    factors.to_csv(OUT / "每日中点区间成交量因子.csv", index=False, encoding="utf-8-sig")
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main), ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            controller = EaseOfMovementController(frame, factors.iloc[:len(frame)], cost, cfg)
            ledger, decisions = simulate_observed_state_account(frame, dividends, cfg, cost, start, PRIMARY, controller)
            save_account(folder, PRIMARY, ledger, decisions)
            require(not ledger.terminal_unliquidated.iloc[-1] and ledger.accounting_error.abs().max() < 1e-6, "简易波动实际账户未完整结算")
            coverage.append({"period": period, "cost": cost_id, "model": PRIMARY, "holding_closes": int(ledger.shares.gt(0).sum()), "buy_trades": int(ledger.filled_quantity.gt(0).sum()),
                             "sell_trades": int(ledger.filled_quantity.lt(0).sum()), "unfilled_requests": int((ledger.requested_quantity.ne(0)&ledger.filled_quantity.eq(0)).sum()),
                             "no_view_origins": int(decisions.signal_state.str.startswith("NO_VIEW").sum()), "entry_requests": int(decisions.requested_quantity.gt(0).sum()), "exit_requests": int(decisions.requested_quantity.lt(0).sum()), "mean_exposure": float(ledger.exposure.mean())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "中点区间成交量简易波动零线"}
            for model, (p, name) in CONTROLS.items():
                saved = pd.read_parquet(p / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            benchmark = summarize(accounts["BUY_HOLD"], cfg)
            for model, account in accounts.items():
                require(pd.DatetimeIndex(account.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "简易波动和对照日历不同")
                measured = {"cost": cost_id, "model": model, "name": names[model], **summarize(account, cfg)}
                measured["annualized_return_excess_vs_buy_hold"] = measured["annualized_return"]-benchmark["annualized_return"]
                measured["meets_point_target"] = measured["net_sharpe"] is not None and measured["net_sharpe"] >= cfg["high_sharpe_target"]
                dest.append(measured)
                for year, group in account.groupby(account.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for era, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": era, **summarize(account[account.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：一个实际新账户和三个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "EASE_OF_MOVEMENT_ACCOUNTS_COMPLETE", "candidate_configurations": 1, "evaluation_accounts": len(main),
              "new_accounts_generated": 2, "reused_control_accounts": 6, "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6,
              "new_model_fits": 0, "new_reference_accounts": 0, "run_seconds": time.perf_counter()-clock, "all_metrics": main, "earlier_diagnostics": earlier, "primary": primary,
              "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"), "account_coverage": coverage, "historical_point_target_met": any(m["meets_point_target"] for m in primary),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": primary, "较早": [m for m in earlier if m["model"] == PRIMARY], "耗时": result["run_seconds"], "覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
