"""高低点时间信号直接驱动完整自然退出账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.aroon_time_entry_inputs_v1 import factor_frame, rules, attach_factor_context
from research.simple_price_entry_exit_v1 import simulate_policy
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_aroon_time_entry_v1"
CONFIG = ROOT / "config/510300_aroon_time_entry_v1.json"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PRIMARY = "AROON_TIME_ENTRY"
NAME = "高低点时间进入及明确自然退出"
CONTROLS = {"REARM_NONE": (P32, "原日内强弱进入及自然退出"), "CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "原第91轮局部候选"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "高低点时间策略已经登记")
    old = json.loads((ROOT / "config/510300_entry_payoff_gate_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days", "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "specification"]}
    cfg.update(study_id="510300_AROON_TIME_ENTRY_V1", round=100, registered_at=now(), primary=PRIMARY, candidate_configurations=1, extrema_period=25, entry_up_threshold=70., entry_down_threshold=30.,
        numerical_tie_tolerance=1e-12, price_basis="WINDOW_LOCAL_CASH_DIVIDEND_REINVESTED_WEALTH_FOR_SIGNAL_ONLY", entry_clock="COMPLETE_CLOSE_NEXT_OPEN", missing_rule="NO_NEW_BUY_PRICE_AND_TIME_EXIT_REMAIN_KNOWN",
        rules="docs/510300_AROON_TIME_ENTRY_V1.md", new_model_fits=0, new_reference_accounts=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED", position_impact=0)
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 5, "高低点时间必要测试未通过")
    paths = [Path(__file__), ROOT / "research/aroon_time_entry_inputs_v1.py", ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "tests/test_aroon_time_entry_v1.py", OUT / "tests_receipt.json", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"], ROOT / "config/510300_entry_payoff_gate_v1.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(parent / period / cost / f"{model}_ledger.parquet" for model, (parent, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第100轮单一高低点时间设置已冻结，尚无新策略账户收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "高低点时间冻结输入改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    factors = factor_frame(data, cfg["extrema_period"], cfg["entry_up_threshold"], cfg["entry_down_threshold"], cfg["numerical_tie_tolerance"])
    factors.to_parquet(OUT / "aroon_factors.parquet", index=False)
    factors.to_csv(OUT / "全部高低点时间因子.csv", index=False, encoding="utf-8-sig")
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main), ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"], earlier)]:
        local = factors.iloc[:len(frame)]
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_policy(frame, dividends, cfg, cost, start, rules(local), cfg["specification"])
            decisions = attach_factor_context(decisions, local)
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "高低点时间账户未完整结算")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()), "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()), "no_view_factor_origins": int(decisions.factor_status.ne("AROON_TIME_FACTORS_AVAILABLE").sum()),
                "mean_exposure": float(ledger.exposure.mean()), "completed_cycles": len(cycles), "raw_entry_origins": int(decisions.raw_entry.sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for model, (parent, name) in CONTROLS.items():
                saved = pd.read_parquet(parent / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "新旧完整账户日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"]-bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：一个新完整账户及三个保存对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "AROON_TIME_ENTRY_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": 8, "new_accounts_generated": 2, "reused_control_accounts": 6, "earlier_diagnostic_accounts": 8, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6,
        "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier, "primary": primary, "account_coverage": coverage,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"), "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主结果": primary, "较早结果": [m for m in earlier if m["model"] == PRIMARY], "覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
