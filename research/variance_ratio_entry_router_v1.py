"""用重叠方差比选择突破或反弹进入，实际持仓沿用固定模式退出。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.simple_price_entry_exit_v1 import signals, simulate_policy, specifications

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_variance_ratio_entry_router_v1"
CONFIG = ROOT / "config/510300_variance_ratio_entry_router_v1.json"
PRIMARY = "VARIANCE_RATIO_ROUTER"
CONTROL = "UNGATED_TWO_MODE"
NAMES = {PRIMARY: "方差比选择突破或反弹进入", CONTROL: "相同规则不加方差条件"}
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P30 = ROOT / "reports/research/510300_simple_intraday_protection_v1"


def variance_ratio(values, aggregation=5):
    values = np.asarray(values, dtype=float)
    size = len(values)
    if size <= aggregation or aggregation < 2 or not np.isfinite(values).all():
        return np.nan
    centered = values - values.mean()
    daily = float(centered @ centered / (size - 1))
    tolerance = np.finfo(float).eps * max(1e-12, float(np.mean(values ** 2)))
    if daily <= tolerance:
        return np.nan
    blocks = np.convolve(centered, np.ones(aggregation), mode="valid")
    divisor = aggregation * (size - aggregation + 1) * (1 - aggregation / size)
    return float((blocks @ blocks / divisor) / daily)


def rolling_ratio(returns, window=60, aggregation=5):
    require(window > aggregation >= 2, "方差比窗口必须大于合计周期")
    return returns.rolling(window, min_periods=window).apply(lambda x: variance_ratio(x, aggregation), raw=True)


def make_rules(data, ratio):
    old = signals(data)
    trend = old["T2_CHANNEL20"]["entry"] > 0
    rebound = old["R2_Z_CONFIRM"]["entry"] > 0
    v = np.asarray(ratio, dtype=float)
    require(len(v) == len(data), "方差状态长度不符")
    entries = {
        PRIMARY: np.where((v > 1) & trend, 1, np.where((v < 1) & rebound, 2, 0)),
        CONTROL: np.where(trend, 1, np.where(rebound, 2, 0)),
    }
    exits = {1: old["T2_CHANNEL20"]["exit"][1], 2: old["R2_Z_CONFIRM"]["exit"][1]}
    return {key: {"entry": value, "exit": {mode: flags.copy() for mode, flags in exits.items()}} for key, value in entries.items()}, trend, rebound


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    specs = specifications()
    cfg.update(study_id="510300_VARIANCE_RATIO_ENTRY_ROUTER_V1", round=53, registered_at=now(), primary=PRIMARY,
               candidate_configurations=2, names=NAMES, window=60, aggregation=5,
               specification={"cooldown": 2, "modes": {1: specs["T2_CHANNEL20"]["modes"][1], 2: specs["R2_Z_CONFIRM"]["modes"][1]}},
               rules="docs/510300_VARIANCE_RATIO_ENTRY_ROUTER_V1.md", position_impact=0)
    paths = [Path(__file__), ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"],
             ROOT / "tests/test_variance_ratio_entry_router_v1.py"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend([P30 / period / cost / "S1_TREND_REBOUND__CLOSE_NEXT_OPEN_ledger.parquet",
                          P32 / period / cost / "REARM_RIDGE_ledger.parquet", P32 / period / cost / "BUY_HOLD_ledger.parquet"])
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第53轮已登记方差状态进入及不加状态的相同规则对照，共两个设置。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "方差状态进入登记内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    ratio = rolling_ratio(data.total_log, cfg["window"], cfg["aggregation"])
    rules, trend, rebound = make_rules(data, ratio)
    factors = pd.DataFrame({"date": data.date, "variance_ratio": ratio,
                            "regime": np.where(ratio.notna(), np.sign(ratio - 1), np.nan),
                            "state_label": np.select([ratio.gt(1), ratio.lt(1), ratio.eq(1)],
                                                     ["较偏同向累积", "较偏涨跌交替", "等于一不选新模式"], default="NO_VIEW_无有效方差状态"),
                            "trend_entry": trend, "rebound_entry": rebound,
                            **{k + "_entry": r["entry"] for k, r in rules.items()}})
    factors.to_parquet(OUT / "all_daily_factors.parquet", index=False)
    main, earlier, yearly, eras, counts, mode_stats = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        local_ratio = ratio.iloc[:len(frame)]
        local_rules, _, _ = make_rules(frame, local_ratio)
        first = int(np.flatnonzero(frame.date >= start)[0])
        usable_factors = factors.iloc[first - 1:len(frame) - 1].copy()
        require(usable_factors.variance_ratio.notna().all(), "本轮评价所需状态不完整，不能以空仓替代缺失后继续计算")
        for state, group in usable_factors.groupby("state_label", dropna=False):
            counts.append({"period": period, "state": state, "decision_days": len(group),
                           "raw_trend_entry_days": int(group.trend_entry.sum()), "raw_rebound_entry_days": int(group.rebound_entry.sum()),
                           "admitted_trend_days": int(group[PRIMARY + "_entry"].eq(1).sum()),
                           "admitted_rebound_days": int(group[PRIMARY + "_entry"].eq(2).sum())})
        for cost_id, cost in cfg["costs"].items():
            folder, accounts, names = OUT / period / cost_id, {}, dict(NAMES)
            for key in NAMES:
                ledger, decisions, cycles = simulate_policy(frame, dividends, cfg, cost, start, local_rules[key], cfg["specification"])
                decisions["variance_ratio"] = local_ratio.iloc[decisions.origin_index.to_numpy(int)].to_numpy()
                decisions["factor_used_for_entry"] = key == PRIMARY
                save_account(folder, key, ledger, decisions)
                cycles.to_csv(folder / f"{key}_cycles.csv", index=False, encoding="utf-8-sig")
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "方差状态策略账户未完成结算")
                require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "方差状态策略违反买入次日可卖")
                for mode in [1, 2]:
                    c = cycles[cycles["mode"].eq(mode)]
                    mode_stats.append({"period": period, "cost": cost_id, "model": key, "entry_mode": mode,
                                       "cycles": len(c), "positive_cycles": int(c.net_profit_cny.gt(0).sum()),
                                       "net_cycle_profit_cny": float(c.net_profit_cny.sum()),
                                       "mean_holding_intervals": float(c.holding_intervals.mean()) if len(c) else None})
                accounts[key] = ledger
            for key, name, source in [("OLD_REGIME", "原均线与趋势效率状态切换", P30 / period / cost_id / "S1_TREND_REBOUND__CLOSE_NEXT_OPEN_ledger.parquet"),
                                      ("REARM_RIDGE", "原学习退出及等待新机会", P32 / period / cost_id / "REARM_RIDGE_ledger.parquet"),
                                      ("BUY_HOLD", "买入持有", P32 / period / cost_id / "BUY_HOLD_ledger.parquet")]:
                accounts[key] = pd.read_parquet(source)
                accounts[key].to_parquet(folder / f"{key}_ledger.parquet", index=False)
                names[key] = name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, ledger in accounts.items():
                require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "方差状态策略没有保留完整评价日历")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(ledger, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in ledger.groupby(ledger.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = ledger[ledger.date.between(left, right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：两个进入设置及三个保存对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("state_counts.csv", counts), ("mode_statistics.csv", mode_stats)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "VARIANCE_RATIO_ENTRY_ROUTER_COMPLETE", "candidate_configurations": 2,
              "evaluation_accounts": len(main), "new_accounts_generated": 4, "reused_control_accounts": 6,
              "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 4, "reused_earlier_accounts": 6,
              "new_reference_accounts": 0, "new_model_fits": 0, "all_metrics": main, "earlier_diagnostics": earlier,
              "state_counts": counts, "mode_statistics": mode_stats, "primary": [m for m in main if m["model"] == PRIMARY],
              "post_selected_best_base": max((m for m in main if m["model"] in NAMES and m["cost"] == "BASE"), key=lambda m: m["net_sharpe"]),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] in NAMES),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": [m for m in main if m["model"] in NAMES], "较早诊断": [m for m in earlier if m["model"] in NAMES]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
