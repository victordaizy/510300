"""记录先前突破的固定位置，后来回踩守住才产生一次进入机会。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.simple_price_entry_exit_v1 import signals, simulate_policy, specifications

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_breakout_retest_entry_v1"
CONFIG = ROOT / "config/510300_breakout_retest_entry_v1.json"
PRIMARY = "BREAKOUT_RETEST"
P54 = ROOT / "reports/research/510300_compression_confirmed_entry_v1"


def low_in_wealth_units(wealth, low, close, dividend):
    return wealth * (low + dividend) / (close + dividend)


def detect_retests(dates, wealth, low, prior_high, first_breakout, valid, max_days=10):
    values, lows, levels = np.asarray(wealth, float), np.asarray(low, float), np.asarray(prior_high, float)
    new_events, available = np.asarray(first_breakout, bool), np.asarray(valid, bool)
    rows, active = [], None
    for t in range(len(values)):
        row = {"date": dates.iloc[t], "setup_event_index": None, "setup_event_date": pd.NaT, "fixed_breakout_level": np.nan,
               "setup_age": None, "setup_created": False, "candidate_entry": False, "setup_status": "空闲，等待新的突破首日"}
        if active is not None:
            origin, level = active
            age = t - origin
            row.update(setup_event_index=origin, setup_event_date=dates.iloc[origin], fixed_breakout_level=level, setup_age=age)
            if not available[t] or t == 0 or not np.isfinite([values[t], values[t - 1], lows[t]]).all():
                row["setup_status"], active = "NO_VIEW_数据缺失，取消待回踩机会", None
            elif age > max_days:
                row["setup_status"], active = "等待期限结束", None
            elif values[t] < level:
                row["setup_status"], active = "收盘跌破原突破位，取消机会", None
            elif age >= 1 and lows[t] <= level <= values[t] and values[t] > values[t - 1]:
                row["setup_status"], row["candidate_entry"], active = "后来回踩守住且收盘回升，发出一次信号", True, None
            else:
                row["setup_status"] = "等待后来回踩确认，原突破位固定"
        elif available[t] and new_events[t] and np.isfinite([values[t], levels[t], lows[t]]).all():
            active = (t, float(levels[t]))
            row.update(setup_event_index=t, setup_event_date=dates.iloc[t], fixed_breakout_level=float(levels[t]), setup_age=0,
                       setup_created=True, setup_status="突破首日收盘建立机会，当日不确认回踩")
        elif not available[t]:
            row["setup_status"] = "NO_VIEW_没有完整价格条件"
        rows.append(row)
    return pd.DataFrame(rows)


def build_factors(data, cfg):
    w = data.wealth
    level = w.shift(1).rolling(20, min_periods=20).max()
    low = low_in_wealth_units(w, data.low, data.close, data.dividend)
    raw = w.gt(level)
    available = np.isfinite(pd.DataFrame({"wealth": w, "previous": w.shift(1), "low": low, "level": level})).all(axis=1) & data.feature_valid.fillna(False)
    first = raw & ~raw.shift(1, fill_value=False) & available & available.shift(1, fill_value=False)
    result = detect_retests(data.date, w, low, level, first, available, cfg["max_retest_days"])
    result["ordinary_breakout"] = raw & available
    result["first_breakout"] = first
    result["low_total_return_scale"] = low
    result["prior20_high"] = level
    result["inputs_available"] = available
    return result


def freeze():
    old = json.loads((ROOT / "config/510300_compression_confirmed_entry_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    cfg.update(study_id="510300_BREAKOUT_RETEST_ENTRY_V1", round=55, registered_at=now(), primary=PRIMARY,
               candidate_configurations=1, max_retest_days=10, specification=specifications()["T2_CHANNEL20"],
               rules="docs/510300_BREAKOUT_RETEST_ENTRY_V1.md", position_impact=0)
    paths = [Path(__file__), ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"],
             ROOT / "tests/test_breakout_retest_entry_v1.py"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(P54 / period / cost / f"{k}_ledger.parquet" for k in ["ORDINARY_BREAKOUT", "AFTER_COMPRESSION_BREAKOUT", "REARM_RIDGE", "BUY_HOLD"])
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第55轮已登记一个后来回踩确认进入设置，正式账户尚未运行。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "回踩进入登记内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    factors = build_factors(data, cfg)
    factors.to_parquet(OUT / "all_daily_factors.parquet", index=False)
    main, earlier, yearly, eras, states, coverage = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        f = factors.iloc[:len(frame)]
        original = signals(frame)["T2_CHANNEL20"]
        rule = {"entry": f.candidate_entry.to_numpy(int), "exit": original["exit"]}
        first = int(np.flatnonzero(frame.date >= start)[0])
        requested = f.iloc[first - 1:len(frame) - 1]
        require(requested.inputs_available.all(), "评价日期价格不完整，不能按空仓填补")
        for status, group in requested.groupby("setup_status"):
            states.append({"period": period, "status": status, "decision_days": len(group), "entry_signals": int(group.candidate_entry.sum())})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_policy(frame, dividends, cfg, cost, start, rule, cfg["specification"])
            for column in ["candidate_entry", "setup_event_date", "fixed_breakout_level", "setup_age", "setup_status"]:
                decisions[column] = f[column].iloc[decisions.origin_index.to_numpy(int)].to_numpy()
            if len(cycles):
                ff = f.set_index("date")
                for column in ["setup_event_date", "fixed_breakout_level", "setup_age"]:
                    cycles[column] = cycles.entry_origin.map(ff[column])
                require(cycles.setup_event_date.notna().all() and (cycles.setup_event_date < cycles.entry_origin).all(), "实际买入没有先前突破事件")
                require(cycles.setup_age.between(1, cfg["max_retest_days"]).all(), "实际买入使用过期回踩信号")
                require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "回踩账户违反买入次日可卖")
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "回踩账户结算不完整")
            coverage.append({"period": period, "cost": cost_id, "candidate_signals": int(requested.candidate_entry.sum()),
                             "buy_requests": int(decisions.requested_quantity.gt(0).sum()), "actual_cycles": len(cycles),
                             "different_breakout_events": int(cycles.setup_event_date.nunique()) if len(cycles) else 0,
                             "positive_cycles": int(cycles.net_profit_cny.gt(0).sum()) if len(cycles) else 0,
                             "mean_holding_intervals": float(cycles.holding_intervals.mean()) if len(cycles) else None})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "突破之后等后来回踩守住进入", "ORDINARY_BREAKOUT": "原普通二十日突破",
                                                 "AFTER_COMPRESSION_BREAKOUT": "原压缩后等待突破", "REARM_RIDGE": "原学习退出及等待新机会", "BUY_HOLD": "买入持有"}
            for key in ["ORDINARY_BREAKOUT", "AFTER_COMPRESSION_BREAKOUT", "REARM_RIDGE", "BUY_HOLD"]:
                accounts[key] = pd.read_parquet(P54 / period / cost_id / f"{key}_ledger.parquet")
                accounts[key].to_parquet(folder / f"{key}_ledger.parquet", index=False)
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "回踩比较没有使用完整相同日历")
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
            print(f"{period}／{cost_id}：回踩进入及四个原样对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("state_statistics.csv", states), ("entry_statistics.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "BREAKOUT_RETEST_ENTRY_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8,
              "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8,
              "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier,
              "entry_statistics": coverage, "state_statistics": states, "primary": [m for m in main if m["model"] == PRIMARY],
              "post_selected_best_base": next(m for m in main if m["model"] == PRIMARY and m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] == PRIMARY),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": result["primary"], "较早诊断": [m for m in earlier if m["model"] == PRIMARY]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
