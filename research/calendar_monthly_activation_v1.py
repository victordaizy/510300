"""每月只凭旧日历策略过去一年的净收益，启停它的一半预算。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.calendar_learned_equal_blend_v1 import combine_states
from research.calendar_liquidity_timing_v1 import simulate_policy
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.self_performance_entry_v1 import performance_flags

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_calendar_monthly_activation_v1"
CONFIG = ROOT / "config/510300_calendar_monthly_activation_v1.json"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P57 = ROOT / "reports/research/510300_calendar_learned_equal_blend_v1"
PRIMARY = "CALENDAR_MONTHLY_ACTIVATION"


def monthly_activation(data, flags, start):
    require(pd.DatetimeIndex(data.date).equals(pd.DatetimeIndex(flags.date)), "策略收益与市场日期不一致")
    first = int(np.flatnonzero(data.date >= pd.Timestamp(start))[0])
    active, evidence_date, evidence_return, rows = None, pd.NaT, None, []
    for t in range(len(data)):
        row = {"market_date": data.date.iloc[t], "execution_date": data.execution_date.iloc[t], "decision_time": data.decision_time.iloc[t],
               "is_update": False, "reference_history_available": bool(flags.history_available.iloc[t]),
               "current_reference_year_return": float(flags.reference_year_net_return.iloc[t]) if flags.history_available.iloc[t] else None,
               "calendar_enabled": None, "calendar_budget": np.nan, "learned_budget": np.nan,
               "has_valid_activation_decision": False, "last_valid_evidence_date": pd.NaT, "last_valid_reference_year_return": None,
               "activation_status": "评价区间之外，未作本账户决定"}
        if first - 1 <= t < len(data) - 1:
            scheduled = t == first - 1 or data.month_session_ordinal.iloc[t] == 1
            if scheduled:
                row["is_update"] = True
                if flags.history_available.iloc[t]:
                    active = bool(flags.reference_year_net_return.iloc[t] > 0)
                    evidence_date, evidence_return = data.date.iloc[t], float(flags.reference_year_net_return.iloc[t])
                    status = "按过去一年净盈利，启用日历一半预算" if active else "过去一年未净盈利，全部预算交给原学习状态"
                else:
                    status = "NO_VIEW_参考收益不完整，保持上次分配；无上次决定则使用原学习状态"
            else:
                status = "保持本月已决定的预算，不因月中表现改变"
            enabled = active is True
            row.update(calendar_enabled=enabled, calendar_budget=.5 if enabled else 0., learned_budget=.5 if enabled else 1.,
                       has_valid_activation_decision=active is not None, last_valid_evidence_date=evidence_date,
                       last_valid_reference_year_return=evidence_return, activation_status=status)
        rows.append(row)
    return pd.DataFrame(rows)


def active_targets(states, activation):
    require(pd.DatetimeIndex(states.market_date).equals(pd.DatetimeIndex(activation.market_date)), "启停与状态日期不一致")
    return np.where(states.inputs_complete.to_numpy(bool),
                    states.calendar_state.to_numpy(float) * activation.calendar_budget.to_numpy(float) +
                    states.learned_state.to_numpy(float) * activation.learned_budget.to_numpy(float), np.nan)


def freeze():
    old = json.loads((ROOT / "config/510300_calendar_learned_equal_blend_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "calendar_features", "dividends",
                              "earlier_start", "earlier_terminal", "weight_band", "decision_clock", "calendar_first_delivery_proven"]}
    cfg.update(study_id="510300_CALENDAR_MONTHLY_ACTIVATION_V1", round=58, registered_at=now(), primary=PRIMARY,
               candidate_configurations=1, reference_start="2013-06-03", reference_cost="BASE", history_trading_days=242,
               activation_threshold=0., activation_clock="MONTH_FIRST_EXECUTION_DAY_09_00", state_cost="BASE",
               new_model_fits=0, rules="docs/510300_CALENDAR_MONTHLY_ACTIVATION_V1.md", position_impact=0)
    paths = [Path(__file__), ROOT / "research/calendar_learned_equal_blend_v1.py", ROOT / "research/calendar_liquidity_timing_v1.py",
             ROOT / "research/self_performance_entry_v1.py", ROOT / "research/simple_signal_blend_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / cfg["features"], ROOT / cfg["calendar_features"], ROOT / cfg["dividends"],
             ROOT / cfg["rules"], ROOT / "docs/510300_CALENDAR_LEARNED_EQUAL_BLEND_V1.md", ROOT / "tests/test_calendar_monthly_activation_v1.py"]
    for period in ["evaluation", "earlier_diagnostic"]:
        paths.append(P32 / period / "BASE/REARM_RIDGE_decisions.parquet")
        for cost in cfg["costs"]:
            paths.extend(P57 / period / cost / f"{key}_ledger.parquet" for key in ["CALENDAR_LEARNED_HALF", "K2_MONTH_EDGE", "REARM_RIDGE", "PANIC_LEARNED_HALF", "BUY_HOLD"])
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第58轮一个按过去一年净收益月度启停日历预算的方案已登记。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "月度启停登记内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["calendar_features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    reference_cfg = {**cfg, "evaluation_start": cfg["reference_start"]}
    reference, ref_decisions = simulate_policy(data, dividends, reference_cfg, cfg["costs"][cfg["reference_cost"]], "K2_REFERENCE", targets=data.month_edge.to_numpy(float))
    save_account(OUT / "reference", "K2_CONTINUOUS_BASE", reference, ref_decisions)
    require(reference.accounting_error.abs().max() < 1e-6 and not reference.terminal_unliquidated.iloc[-1], "日历连续参考账户未完整结算")
    flags = performance_flags(data, reference, cfg["history_trading_days"])
    flags.to_parquet(OUT / "reference_year_flags.parquet", index=False)
    main, earlier, yearly, eras, updates, counts = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        states = combine_states(frame, pd.read_parquet(P32 / period / "BASE/REARM_RIDGE_decisions.parquet"))
        activation = monthly_activation(frame, flags.iloc[:len(frame)], start)
        target = active_targets(states, activation)
        first = int(np.flatnonzero(frame.date >= start)[0])
        require(np.isfinite(target[first - 1:len(frame) - 1]).all(), "启停目标缺失，不能用空仓填补")
        changed = activation[activation.is_update]
        require(changed.reference_history_available.all() and changed.has_valid_activation_decision.all(), "实际月度启停缺少完整参考收益")
        require((changed.last_valid_evidence_date < changed.execution_date).all(), "启停读到了成交日尚未发生的收益")
        updates.extend({"period": period, **r} for r in changed.to_dict("records"))
        saved_states = states.copy()
        for column in activation.columns:
            if column not in saved_states:
                saved_states[column] = activation[column]
        saved_states["static_half_target"], saved_states["target"] = states.target, target
        saved_states.to_parquet(OUT / f"{period}_states.parquet", index=False)
        origins = saved_states.iloc[first - 1:len(frame) - 1]
        for (enabled, a, b), group in origins.groupby(["calendar_enabled", "calendar_state", "learned_state"]):
            counts.append({"period": period, "calendar_enabled": bool(enabled), "calendar_state": a, "learned_state": b,
                           "target": float(group.target.iloc[0]), "decision_days": len(group)})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions = simulate_policy(frame, dividends, {**cfg, "evaluation_start": start}, cost, PRIMARY, targets=target)
            for column in ["calendar_enabled", "calendar_budget", "learned_budget", "last_valid_evidence_date", "last_valid_reference_year_return"]:
                decisions[column] = decisions.origin.map(activation.set_index("market_date")[column])
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "月度启停账户未完整结算")
            require((decisions.decision_time == decisions.execution_date + pd.Timedelta(hours=9)).all(), "启停账户实际决策时钟不一致")
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "按过去一年净收益每月启停日历预算"}
            for key, name in [("CALENDAR_LEARNED_HALF", "原月内两端与学习退出固定各半"), ("K2_MONTH_EDGE", "原月内两端单独运行"),
                              ("REARM_RIDGE", "原学习退出及等待新机会"), ("PANIC_LEARNED_HALF", "原急跌回升与学习退出各半"), ("BUY_HOLD", "买入持有")]:
                saved = pd.read_parquet(P57 / period / cost_id / f"{key}_ledger.parquet")
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "启停账户比较日历不完整")
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
            print(f"{period}／{cost_id}：月度启停及五个保存对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras),
                           ("monthly_activation_updates.csv", updates), ("state_counts.csv", counts)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "CALENDAR_MONTHLY_ACTIVATION_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 10,
              "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 10,
              "new_model_fits": 0, "new_reference_accounts": 1, "reference_days": len(reference),
              "all_metrics": main, "earlier_diagnostics": earlier, "state_counts": counts,
              "activation_updates": [{"period": period, "updates": sum(r["period"] == period for r in updates),
                                      "enabled_updates": sum(r["period"] == period and r["calendar_enabled"] for r in updates)} for period in ["evaluation", "earlier_diagnostic"]],
              "primary": [m for m in main if m["model"] == PRIMARY],
              "post_selected_best_base": next(m for m in main if m["model"] == PRIMARY and m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] == PRIMARY),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": result["primary"], "较早": [m for m in earlier if m["model"] == PRIMARY], "月度更新": result["activation_updates"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
