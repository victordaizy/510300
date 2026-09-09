"""用原策略连续参考账户过去一年的净收益决定是否接受新进入。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import ExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.self_performance_entry_account_v1 import simulate_self_performance_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_self_performance_entry_v1"
CONFIG = ROOT / "config/510300_self_performance_entry_v1.json"
PARENT = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PRIMARY = "POSITIVE_SELF_YEAR"


def performance_flags(data, reference, window=242):
    require(not reference.date.duplicated().any(), "参考账户日期重复")
    available = reference.set_index("date")
    returns = data.date.map(available.net_return)
    closed = data.date.map(available.mark_clock).eq("CLOSE")
    good = closed & np.isfinite(returns) & (returns > -1)
    logs = np.log1p(returns.where(good))
    count = logs.rolling(window, min_periods=window).count()
    log_total = logs.rolling(window, min_periods=window).sum()
    valid = count.eq(window) & np.isfinite(log_total)
    total = np.expm1(log_total.where(valid))
    return pd.DataFrame({"date": data.date, "history_available": valid, "reference_year_net_return": total,
                         "positive_history": valid & (total > 0)})


class SelfPerformanceGate:
    def __init__(self, flags):
        self.flags = flags

    def __call__(self, t, account, mode, quantity):
        row = self.flags.iloc[t]
        valid, positive = bool(row.history_available), bool(row.positive_history)
        return {"entry_allowed": valid and positive, "self_performance_status": "HISTORY_AVAILABLE" if valid else "NO_VIEW_INCOMPLETE_REFERENCE_YEAR",
                "reference_year_net_return": float(row.reference_year_net_return) if valid else None,
                "entry_deferred_for_nonpositive_performance": valid and not positive}


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
                              "confirmation_days", "specification", "saved_models"]}
    cfg.update(study_id="510300_SELF_PERFORMANCE_ENTRY_V1", round=45, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
               reference_start="2013-06-03", reference_cost="BASE", history_trading_days=242, threshold=0., new_model_fits=0,
               rules="docs/510300_SELF_PERFORMANCE_ENTRY_V1.md", position_impact=0)
    paths = [Path(__file__), ROOT / "research/self_performance_entry_account_v1.py", ROOT / "research/rearmed_cycle_exit_account_v1.py",
             ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / cfg["rules"],
             ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"], ROOT / "tests/test_self_performance_entry_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第45轮登记一个按自身过去一年净收益过滤新进入的设置，原退出保持。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "自身表现入场登记内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    reference, reference_decisions, reference_cycles = simulate_rearmed_exit(data, dividends, cfg, cfg["costs"]["BASE"], cfg["reference_start"],
        make_rules(data)["D60_INTRA"], cfg["specification"], ExitController(data, models, cfg["confirmation_days"]))
    require(reference.accounting_error.abs().max() < 1e-6 and not reference.terminal_unliquidated.iloc[-1], "连续参考账户结算失败")
    save_account(OUT / "reference", "ORIGINAL_REARM_RIDGE", reference, reference_decisions)
    reference_cycles.to_csv(OUT / "reference/ORIGINAL_REARM_RIDGE_cycles.csv", index=False, encoding="utf-8-sig")
    flags = performance_flags(data, reference, cfg["history_trading_days"])
    flags.to_parquet(OUT / "self_performance_flags.parquet", index=False)
    flags.to_csv(OUT / "self_performance_flags.csv", index=False, encoding="utf-8-sig")
    main, early, yearly, eras, stats = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        local_flags = flags.iloc[:len(frame)]
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_self_performance_exit(frame, dividends, cfg, cost, start, make_rules(frame)["D60_INTRA"],
                cfg["specification"], ExitController(frame, models, cfg["confirmation_days"]), SelfPerformanceGate(local_flags))
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "自身表现过滤账户结算失败")
            require(not ((ledger.filled_quantity > 0) & (ledger.shares_before > 0)).any(), "自身表现过滤发生了持仓中追加")
            if len(cycles):
                require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "自身表现过滤违反次日可卖")
            checks = decisions[decisions.self_performance_status.notna()]
            stats.append({"period": period, "cost": cost_id, "eligible_entry_checks": len(checks), "positive_history_acceptances": int(checks.entry_allowed.sum()),
                          "nonpositive_deferrals": int(checks.entry_deferred_for_nonpositive_performance.sum()),
                          "no_view_history_checks": int(checks.self_performance_status.eq("NO_VIEW_INCOMPLETE_REFERENCE_YEAR").sum()), "completed_cycles": len(cycles)})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "原策略过去一年赚钱才接受新进入"}
            for key, name in [("REARM_RIDGE", "原线性退出及等待新机会"), ("BUY_HOLD", "买入持有")]:
                saved = pd.read_parquet(PARENT / period / cost_id / f"{key}_ledger.parquet")
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "自身表现过滤完整评价日期不一致")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                destination.append(m)
                if period == "evaluation":
                    for year, group in saved.groupby(saved.date.dt.year):
                        yearly.append({"cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[(saved.date >= left) & (saved.date <= right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：一个自身表现入场和两个原样对照已完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("entry_gate_statistics.csv", stats)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "SELF_PERFORMANCE_ENTRY_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": 6, "new_accounts_generated": 2, "reused_control_accounts": 4, "earlier_diagnostic_accounts": 6,
              "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 4, "new_reference_accounts": 1, "new_model_fits": 0,
              "reference_account_days": len(reference), "reference_is_continuous_across_2019_2020": True,
              "all_metrics": main, "earlier_diagnostics": early, "entry_gate_statistics": stats,
              "primary": [m for m in main if m["model"] == PRIMARY],
              "post_selected_best_base": next(m for m in main if m["model"] == PRIMARY and m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] == PRIMARY),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": result["primary"], "较早": [m for m in early if m["model"] == PRIMARY], "入场检查": stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
