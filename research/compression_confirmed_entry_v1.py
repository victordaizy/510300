"""旧波动压缩条件仅产生准备窗口，后续实际突破才允许买入。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.simple_price_entry_exit_v1 import signals, simulate_policy, specifications
from research.volatility_compression_change_point import select_event_indices

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_compression_confirmed_entry_v1"
CONFIG = ROOT / "config/510300_compression_confirmed_entry_v1.json"
PRIMARY = "AFTER_COMPRESSION_BREAKOUT"
P23 = ROOT / "reports/research/510300_simple_price_entry_exit_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"


def preparation_window(events, days=20):
    return events.astype(float).shift(1).rolling(days, min_periods=1).max().eq(1)


def build_factors(data, cfg):
    result = pd.DataFrame({"date": data.date})
    for window in [5, 20, 60]:
        result[f"log_rv{window}"] = data.total_log.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(cfg["annual_days"])
    result["prior_rv20_low_quantile"] = result.log_rv20.shift(1).rolling(cfg["quantile_window"], min_periods=cfg["quantile_minimum"]).quantile(cfg["low_quantile"])
    result["compression_inputs_available"] = result[["log_rv5", "log_rv20", "log_rv60", "prior_rv20_low_quantile"]].notna().all(axis=1)
    result["compression_condition"] = (result.log_rv5.lt(result.log_rv20) & result.log_rv20.lt(result.log_rv60) &
                                       result.log_rv20.le(result.prior_rv20_low_quantile) & result.compression_inputs_available)
    selected = select_event_indices(result.compression_condition, cooldown=cfg["event_cooldown"])
    result["compression_event"] = False
    result.loc[selected, "compression_event"] = True
    result["prepared"] = preparation_window(result.compression_event, cfg["preparation_days"])
    result["preparation_inputs_available"] = result.compression_inputs_available.shift(1).rolling(cfg["preparation_days"], min_periods=cfg["preparation_days"]).min().eq(1)
    event_index = pd.Series(np.where(result.compression_event, np.arange(len(data)), np.nan), index=data.index).shift(1).ffill()
    result["preparing_event_index"] = event_index.where(result.prepared)
    result["preparing_event_date"] = [data.date.iloc[int(i)] if pd.notna(i) else pd.NaT for i in result.preparing_event_index]
    original = signals(data)["T2_CHANNEL20"]
    result["ordinary_breakout"] = original["entry"] > 0
    result["candidate_entry"] = result.prepared & result.preparation_inputs_available & result.ordinary_breakout
    return result


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    cfg.update(study_id="510300_COMPRESSION_CONFIRMED_ENTRY_V1", round=54, registered_at=now(), primary=PRIMARY,
               candidate_configurations=1, quantile_window=756, quantile_minimum=504, low_quantile=.2,
               event_cooldown=20, preparation_days=20, specification=specifications()["T2_CHANNEL20"],
               rules="docs/510300_COMPRESSION_CONFIRMED_ENTRY_V1.md", position_impact=0)
    paths = [Path(__file__), ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/volatility_compression_change_point.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / cfg["rules"],
             ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / "config/volatility_compression_change_point_v1.yaml",
             ROOT / "tests/test_compression_confirmed_entry_v1.py"]
    for cost in cfg["costs"]:
        paths.append(P23 / "evaluation" / cost / "T2_CHANNEL20_ledger.parquet")
        for period in ["evaluation", "earlier_diagnostic"]:
            paths.extend(P32 / period / cost / f"{k}_ledger.parquet" for k in ["REARM_RIDGE", "BUY_HOLD"])
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第54轮已登记一个压缩后实际突破的完整账户候选，未计算新账户。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "压缩后突破登记内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    factors = build_factors(data, cfg)
    factors.to_parquet(OUT / "all_daily_factors.parquet", index=False)
    main, earlier, yearly, eras, counts, cycles_info = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        f = factors.iloc[:len(frame)]
        original = signals(frame)["T2_CHANNEL20"]
        rule = {"entry": f.candidate_entry.to_numpy(int), "exit": original["exit"]}
        first = int(np.flatnonzero(frame.date >= start)[0])
        decisions_f = f.iloc[first - 1:len(frame) - 1]
        require(decisions_f.preparation_inputs_available.all(), "准备窗口输入不完整，不能把缺失变成空仓后继续评价")
        counts.append({"period": period, "decision_days": len(decisions_f), "compression_events": int(decisions_f.compression_event.sum()),
                       "prepared_days": int(decisions_f.prepared.sum()), "ordinary_breakout_days": int(decisions_f.ordinary_breakout.sum()),
                       "candidate_entry_days": int(decisions_f.candidate_entry.sum())})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_policy(frame, dividends, cfg, cost, start, rule, cfg["specification"])
            for column in ["prepared", "preparing_event_date", "ordinary_breakout", "candidate_entry"]:
                decisions[column] = f[column].iloc[decisions.origin_index.to_numpy(int)].to_numpy()
            cycles["preparing_event_date"] = cycles.entry_origin.map(f.set_index("date").preparing_event_date)
            require(cycles.preparing_event_date.notna().all(), "出现没有先前准备事件的实际买入")
            require((cycles.preparing_event_date < cycles.entry_origin).all(), "压缩当天即进入，违反后续突破要求")
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "压缩后突破账户结算失败")
            require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "压缩后突破违反买入次日可卖")
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "旧压缩事件之后等实际突破进入", "ORDINARY_BREAKOUT": "原普通二十日突破",
                                                          "REARM_RIDGE": "原学习退出及等待新机会", "BUY_HOLD": "买入持有"}
            if period == "evaluation":
                ordinary = pd.read_parquet(P23 / period / cost_id / "T2_CHANNEL20_ledger.parquet")
                ordinary.to_parquet(folder / "ORDINARY_BREAKOUT_ledger.parquet", index=False)
            else:
                ordinary, od, oc = simulate_policy(frame, dividends, cfg, cost, start, original, cfg["specification"])
                save_account(folder, "ORDINARY_BREAKOUT", ordinary, od)
                oc.to_csv(folder / "ORDINARY_BREAKOUT_cycles.csv", index=False, encoding="utf-8-sig")
            accounts["ORDINARY_BREAKOUT"] = ordinary
            for key in ["REARM_RIDGE", "BUY_HOLD"]:
                accounts[key] = pd.read_parquet(P32 / period / cost_id / f"{key}_ledger.parquet")
                accounts[key].to_parquet(folder / f"{key}_ledger.parquet", index=False)
            cycles_info.append({"period": period, "cost": cost_id, "candidate_cycles": len(cycles),
                                "different_preparing_events": int(cycles.preparing_event_date.nunique()),
                                "positive_cycles": int(cycles.net_profit_cny.gt(0).sum()),
                                "mean_holding_intervals": float(cycles.holding_intervals.mean())})
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "压缩后突破缺少完整评价日期")
                require(saved.accounting_error.abs().max() < 1e-6 and not saved.terminal_unliquidated.iloc[-1], "候选或对照结算不完整")
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
            print(f"{period}／{cost_id}：压缩后突破及三个对照账户完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("preparation_statistics.csv", counts), ("cycle_statistics.csv", cycles_info)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "COMPRESSION_CONFIRMED_ENTRY_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 6,
              "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 4,
              "new_earlier_candidate_accounts": 2, "new_earlier_control_accounts": 2, "reused_earlier_accounts": 4,
              "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier,
              "preparation_statistics": counts, "cycle_statistics": cycles_info,
              "primary": [m for m in main if m["model"] == PRIMARY],
              "post_selected_best_base": next(m for m in main if m["model"] == PRIMARY and m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] == PRIMARY),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": result["primary"], "较早诊断": [m for m in earlier if m["model"] == PRIMARY]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
