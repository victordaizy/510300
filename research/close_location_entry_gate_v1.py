"""以二十一日收盘位置确认原进入，直接比较成交量加权及等权版本。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import ExitController
from research.self_performance_entry_account_v1 import simulate_self_performance_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_close_location_entry_gate_v1"
CONFIG = ROOT / "config/510300_close_location_entry_gate_v1.json"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
PRIMARY = "VOLUME_LOCATION_ENTRY"
CANDIDATES = {PRIMARY: ("volume_location21", "成交量加权收盘位置确认进入"),
              "EQUAL_LOCATION_ENTRY": ("equal_location21", "等权收盘位置确认进入")}


def location_factors(data, window=21):
    fields = data[["high", "low", "close", "volume"]]
    finite = np.isfinite(fields).all(axis=1)
    valid = finite & data.low.gt(0) & data.high.ge(data.low) & data.close.between(data.low, data.high) & data.volume.ge(0)
    span = data.high - data.low
    multiplier = ((2 * data.close - data.high - data.low) / span.where(span.ne(0))).where(span.ne(0), 0.).where(valid)
    volume = data.volume.where(valid)
    volume_sum = volume.rolling(window, min_periods=window).sum()
    weighted = (multiplier * volume).rolling(window, min_periods=window).sum() / volume_sum.where(volume_sum.gt(0))
    equal = multiplier.rolling(window, min_periods=window).mean()
    available = valid.astype(int).rolling(window, min_periods=window).sum().eq(window) & volume_sum.gt(0)
    return pd.DataFrame({"date": data.date, "daily_location": multiplier, "volume_sum21": volume_sum,
                         "volume_location21": weighted.where(available), "equal_location21": equal.where(available),
                         "location_available": available})


class LocationGate:
    def __init__(self, factors, column, name):
        self.factors, self.column, self.name = factors, column, name

    def __call__(self, t, account, mode, quantity):
        row = self.factors.iloc[t]
        available = bool(row.location_available) and np.isfinite(row[self.column])
        value = float(row[self.column]) if available else None
        allowed = available and value > 0
        return {"entry_allowed": bool(allowed), "location_status": "因子完整" if available else "NO_VIEW_二十一日价格或成交量不完整",
                "location_gate_value": value,
                "action": (self.name + "大于零，原条件允许下一开盘请求买入") if allowed else
                          (self.name + "未大于零，保留原机会并等待") if available else "收盘位置无有效值，保留原机会且不增加买入"}


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
                              "confirmation_days", "specification", "saved_models"]}
    cfg.update(study_id="510300_CLOSE_LOCATION_ENTRY_GATE_V1", round=56, registered_at=now(), primary=PRIMARY,
               candidate_configurations=2, location_window=21, threshold=0., new_model_fits=0,
               rules="docs/510300_CLOSE_LOCATION_ENTRY_GATE_V1.md", position_impact=0)
    paths = [Path(__file__), ROOT / "research/self_performance_entry_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
             ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"],
             ROOT / "tests/test_close_location_entry_gate_v1.py"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend([P32 / period / cost / "REARM_RIDGE_ledger.parquet", P32 / period / cost / "BUY_HOLD_ledger.parquet",
                          P46 / period / cost / "PANIC_LEARNED_HALF_ledger.parquet"])
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第56轮已登记成交量加权及等权两个收盘位置进入设置，尚未运行账户。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "收盘位置登记内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    factors = location_factors(data, cfg["location_window"])
    factors.to_parquet(OUT / "all_daily_factors.parquet", index=False)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        local = factors.iloc[:len(frame)]
        first = int(np.flatnonzero(frame.date >= start)[0])
        require(local.location_available.iloc[first - 1:len(frame) - 1].all(), "实际评价因子不完整，不能用空仓补收益")
        rule = make_rules(frame)["D60_INTRA"]
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            accounts, names = {}, {}
            for key, (column, name) in CANDIDATES.items():
                ledger, decisions, cycles = simulate_self_performance_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"],
                    ExitController(frame, models, cfg["confirmation_days"]), LocationGate(local, column, name))
                save_account(folder, key, ledger, decisions)
                if len(cycles):
                    cycles["entry_location_factor"] = cycles.entry_origin.map(local.set_index("date")[column])
                    require(cycles.entry_location_factor.gt(0).all(), "实际进入未通过收盘位置条件")
                    require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "收盘位置账户违反次日可卖")
                cycles.to_csv(folder / f"{key}_cycles.csv", index=False, encoding="utf-8-sig")
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "收盘位置账户未完整结算")
                require(not ((ledger.filled_quantity > 0) & (ledger.shares_before > 0)).any(), "收盘位置账户发生持仓中追加")
                checks = decisions[decisions.location_status.notna()]
                coverage.append({"period": period, "cost": cost_id, "model": key, "eligible_entry_checks": len(checks),
                                 "positive_acceptances": int(checks.entry_allowed.sum()),
                                 "nonpositive_deferrals": int((checks.location_status.eq("因子完整") & ~checks.entry_allowed.astype(bool)).sum()),
                                 "no_view_checks": int(checks.location_status.str.startswith("NO_VIEW").sum()),
                                 "completed_cycles": len(cycles), "positive_cycles": int(cycles.net_profit_cny.gt(0).sum()) if len(cycles) else 0,
                                 "mean_holding_intervals": float(cycles.holding_intervals.mean()) if len(cycles) else None})
                accounts[key], names[key] = ledger, name
            for key, path, name in [("REARM_RIDGE", P32 / period / cost_id / "REARM_RIDGE_ledger.parquet", "原学习退出及等待新机会"),
                                    ("PANIC_LEARNED_HALF", P46 / period / cost_id / "PANIC_LEARNED_HALF_ledger.parquet", "原急跌回升及学习退出各半"),
                                    ("BUY_HOLD", P32 / period / cost_id / "BUY_HOLD_ledger.parquet", "买入持有")]:
                saved = pd.read_parquet(path)
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "收盘位置账户比较未使用相同完整日历")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[saved.date.between(left, right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：两个收盘位置进入设置及三个原样对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("entry_gate_statistics.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    best = max((m for m in main if m["model"] in CANDIDATES and m["cost"] == "BASE"), key=lambda m: m["net_sharpe"] if m["net_sharpe"] is not None else -999)
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "CLOSE_LOCATION_ENTRY_GATE_COMPLETE", "candidate_configurations": 2,
              "evaluation_accounts": len(main), "new_accounts_generated": 4, "reused_control_accounts": 6,
              "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 4, "reused_earlier_accounts": 6,
              "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier,
              "entry_gate_statistics": coverage, "primary": [m for m in main if m["model"] == PRIMARY], "post_selected_best_base": best,
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] in CANDIDATES),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价候选": [m for m in main if m["model"] in CANDIDATES], "较早候选": [m for m in earlier if m["model"] in CANDIDATES]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
