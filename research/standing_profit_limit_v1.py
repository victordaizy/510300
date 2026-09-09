"""一个事前止盈策略、两成交假设与两个时期的真实账户。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from research.standing_profit_limit_account_v1 import simulate_profit_limit
from research.simple_volume_reversal_v1 import make_rules
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.panic_learned_equal_blend_v1 import panic_folder, P32
from research.intraday_overnight_increment_v1 import now, require, digest, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_standing_profit_limit_v1"
CONFIG = ROOT / "config/510300_standing_profit_limit_v1.json"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
PRIMARY = "PANIC_PROFIT_LIMIT_THROUGH"
ASSUMPTIONS = {PRIMARY: "THROUGH_PRICE", "PANIC_PROFIT_LIMIT_CLOSE_CONFIRM": "CLOSE_STILL_ABOVE"}
NAMES = {PRIMARY: "事前止盈限价：穿价情景", "PANIC_PROFIT_LIMIT_CLOSE_CONFIRM": "事前止盈限价：收盘仍在价上情景"}


def controls(period, cost):
    return {"PANIC_ONLY": (panic_folder(period) / cost / "V6_PANIC_RECOVERY_ledger.parquet", "原急跌回升收盘退出"),
            "CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91 / period / cost / "CONTINUOUS_REFERENCE_MIN_VARIANCE_ledger.parquet", "原91连续风险预算"),
            "BUY_HOLD": (P32 / period / cost / "BUY_HOLD_ledger.parquet", "买入持有")}


def freeze():
    require(not CONFIG.exists(), "事前止盈限价已经登记")
    old = json.loads((ROOT / "config/510300_directional_turnover_recovery_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days", "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    panic = json.loads((ROOT / "config/510300_simple_volume_reversal_v1.json").read_text(encoding="utf-8"))
    cfg.update(study_id="510300_STANDING_PROFIT_LIMIT_V1", round=110, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
               daily_volume_cap=.01, execution_assumptions=ASSUMPTIONS.copy(), specification=panic["candidate_specs"]["V6_PANIC_RECOVERY"],
               rules="docs/510300_STANDING_PROFIT_LIMIT_V1.md", new_model_fits=0, new_reference_accounts=0, goal_achieved=False, position_impact=0,
               independent_validation="NOT_ESTABLISHED", execution_evidence="DAILY_BAR_CONDITIONAL_PROXIES_NOT_TRADE_PROOF", previous_goal_turn_classification="PROGRESS_ROUNDS108_AND109_COMPLETED")
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 7, "事前止盈的必要测试未通过")
    paths = [Path(__file__), ROOT / "research/standing_profit_limit_account_v1.py", ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/simple_volume_reversal_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "tests/test_standing_profit_limit_v1.py", OUT / "tests_receipt.json",
             ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / "config/510300_simple_volume_reversal_v1.json", ROOT / "config/510300_research_authority_v6.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(p for p, _ in controls(period, cost).values())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第110轮一个原6%止盈设置、两成交假设已冻结，尚无新账户收益。", flush=True)


def run():
    clock = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "止盈限价冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main), ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        rule = make_rules(frame)[0]["V6_PANIC_RECOVERY"]
        pd.DataFrame({"date": frame.date, "raw_entry": rule["entry"], "original_price_exit": rule["exit"][1]}).to_parquet(OUT / f"{period}_entry_exit_conditions.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            accounts, names = {}, NAMES.copy()
            for model, assumption in ASSUMPTIONS.items():
                ledger, decisions, cycles = simulate_profit_limit(frame, dividends, cfg, cost, start, rule, cfg["specification"], assumption)
                save_account(folder, model, ledger, decisions)
                cycles.to_csv(folder / f"{model}_cycles.csv", index=False, encoding="utf-8-sig")
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "止盈账户未完整结算")
                standing = ledger.standing_limit_price.notna()
                fills = ledger.filled_quantity.lt(0) & ledger.execution_clock.isin(["OPEN_MARKETABLE_PROFIT_LIMIT", "INTRADAY_CONDITIONAL_PROFIT_LIMIT"])
                require((ledger.loc[fills, "fill_price"] >= ledger.loc[fills, "standing_limit_price"]-1e-10).all(), "实际卖出价突破限价下限")
                coverage.append({"period": period, "cost": cost_id, "model": model, "assumption": assumption, "completed_cycles": len(cycles), "standing_order_days": int(standing.sum()),
                                 "profit_limit_fills": int(fills.sum()), "intraday_profit_fills": int((fills & ledger.execution_clock.eq("INTRADAY_CONDITIONAL_PROFIT_LIMIT")).sum()),
                                 "holding_closes": int(ledger.shares.gt(0).sum()), "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                                 "unfilled_triggered_or_open_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                                 "no_view_execution_inputs": int(ledger.standing_limit_status.str.startswith("NO_VIEW").sum())})
                accounts[model] = ledger
            for model, (p, name) in controls(period, cost_id).items():
                saved = pd.read_parquet(p)
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "止盈及对照账户完整日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"]-bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                dest.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：两止盈情景账户和三保存对照已完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "STANDING_PROFIT_LIMIT_ACCOUNTS_COMPLETE", "candidate_configurations": 1, "execution_scenarios": 2,
              "evaluation_accounts": len(main), "new_accounts_generated": 4, "reused_control_accounts": 6, "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 4,
              "reused_earlier_accounts": 6, "new_model_fits": 0, "new_reference_accounts": 0, "run_seconds": time.perf_counter()-clock, "all_metrics": main, "earlier_diagnostics": earlier,
              "primary": primary, "account_coverage": coverage, "post_selected_best_base": max((m for m in main if m["model"] in ASSUMPTIONS and m["cost"] == "BASE"), key=lambda m: float('-inf') if m["net_sharpe"] is None else m["net_sharpe"]),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] in ASSUMPTIONS), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED",
              "execution_evidence": cfg["execution_evidence"], "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主结果": [m for m in main if m["model"] in ASSUMPTIONS or m["model"] == "PANIC_ONLY"], "较早": [m for m in earlier if m["model"] in ASSUMPTIONS or m["model"] == "PANIC_ONLY"], "耗时": result["run_seconds"], "覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
