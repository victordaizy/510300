"""用连续两个含分红收盘下跌替代学习退出，保留原进入许可。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.price_streak_exit_account_v1 import simulate_price_streak_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_price_streak_exit_v1"
CONFIG = ROOT / "config/510300_price_streak_exit_v1.json"
PARENT = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PRIMARY = "PRICE_STREAK_EXIT"


def price_flags(data):
    w = data.wealth.where(np.isfinite(data.wealth) & (data.wealth > 0))
    valid = w.notna() & w.shift(1).notna() & w.shift(2).notna()
    down = w < w.shift(1)
    return pd.DataFrame({"date": data.date, "prices_available": valid,
                         "one_day_down": down, "two_day_down": valid & down & down.shift(1, fill_value=False)})


class PriceEntryGate:
    def __init__(self, flags):
        self.flags = flags

    def __call__(self, t, account, mode, quantity):
        row = self.flags.iloc[t]
        valid, two_down = bool(row.prices_available), bool(row.two_day_down)
        return {"entry_allowed": valid and not two_down, "price_entry_check_status": "PRICE_AVAILABLE" if valid else "NO_VIEW_PRICE_HISTORY",
                "entry_deferred_for_two_down": valid and two_down}


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "specification"]}
    cfg.update(study_id="510300_PRICE_STREAK_EXIT_V1", round=42, registered_at=now(), primary=PRIMARY,
               candidate_configurations=1, consecutive_down_closes=2, new_model_fits=0,
               rules="docs/510300_PRICE_STREAK_EXIT_V1.md", position_impact=0)
    paths = [Path(__file__), ROOT / "research/price_streak_exit_account_v1.py", ROOT / "research/simple_intraday_protection_v1.py",
             ROOT / "research/simple_session_divergence_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"], ROOT / "tests/test_price_streak_exit_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第42轮一个连续下跌退出方案已登记，不使用学习退出模型。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "简单价格退出登记内容变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    flags = price_flags(data)
    flags.to_parquet(OUT / "price_flags.parquet", index=False)
    main, early, yearly, eras, stats = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        local_flags = flags.iloc[:len(frame)]
        rule = make_rules(frame)["D60_INTRA"]
        rule["exit"][1] = rule["exit"][1] | local_flags.two_day_down.to_numpy(bool)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_price_streak_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"],
                controller=None, entry_gate=PriceEntryGate(local_flags))
            decisions["two_day_down_at_close"] = local_flags.two_day_down.to_numpy()[decisions.origin_index.to_numpy(int)]
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "连续下跌退出账户结算失败")
            if len(cycles):
                require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "连续下跌退出违反买入次日可卖")
            checks = decisions[decisions.price_entry_check_status.notna()]
            exits = cycles[~cycles.exit_reasons.str.contains("研究终点", regex=False)].copy()
            exit_dates = pd.to_datetime(exits.exit_origin)
            flags_by_date = local_flags.set_index("date").two_day_down
            stats.append({"period": period, "cost": cost_id, "entry_checks": len(checks),
                          "entries_deferred_for_two_down": int(checks.entry_deferred_for_two_down.sum()),
                          "no_view_entry_checks": int((checks.price_entry_check_status != "PRICE_AVAILABLE").sum()),
                          "completed_cycles": len(cycles), "two_down_present_at_exit_origin": int(exit_dates.map(flags_by_date).sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "连续两个收盘下跌退出"}
            for key, name in [("REARM_RIDGE", "原线性退出＋等待新机会"), ("REARM_NONE", "原价格退出＋等待新机会"), ("BUY_HOLD", "买入持有")]:
                saved = pd.read_parquet(PARENT / period / cost_id / f"{key}_ledger.parquet")
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "连续下跌退出评价日期不完整")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                dest.append(m)
                if period == "evaluation":
                    for year, group in saved.groupby(saved.date.dt.year):
                        yearly.append({"cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[(saved.date >= left) & (saved.date <= right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：简单价格退出及三个原样对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("price_exit_statistics.csv", stats)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "PRICE_STREAK_EXIT_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": 8, "new_accounts_generated": 2, "reused_control_accounts": 6, "earlier_diagnostic_accounts": 8,
              "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6, "new_model_fits": 0, "all_metrics": main, "earlier_diagnostics": early,
              "price_exit_statistics": stats, "primary": [m for m in main if m["model"] == PRIMARY],
              "post_selected_best_base": next(m for m in main if m["model"] == PRIMARY and m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] == PRIMARY),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "新方案": result["primary"], "较早": [m for m in early if m["model"] == PRIMARY], "价格退出行为": stats}, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
