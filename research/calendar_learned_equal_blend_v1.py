"""月内两端与原学习退出分别贡献一半预算，统一到成交日九点。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.calendar_liquidity_timing_v1 import simulate_policy
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.simple_signal_blend_v1 import decision_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_calendar_learned_equal_blend_v1"
CONFIG = ROOT / "config/510300_calendar_learned_equal_blend_v1.json"
P6 = ROOT / "reports/research/510300_calendar_liquidity_timing_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
PRIMARY = "CALENDAR_LEARNED_HALF"


def combine_states(data, learned_decisions):
    require(not learned_decisions.origin.duplicated().any(), "原学习状态的决策日期重复")
    ids = pd.DatetimeIndex(data.date).get_indexer(pd.to_datetime(learned_decisions.origin))
    require((ids >= 0).all() and (ids + 1 < len(data)).all(), "原学习状态没有对应下一开市日")
    require(pd.DatetimeIndex(learned_decisions.execution_date).equals(pd.DatetimeIndex(data.date.iloc[ids + 1])), "学习状态成交日被错移")
    learned = decision_state(data, learned_decisions)
    calendar = data.month_edge.to_numpy(float)
    complete = data.feature_valid.fillna(False).to_numpy(bool) & np.isfinite(calendar) & np.isfinite(learned)
    require(set(np.unique(calendar[np.isfinite(calendar)])).issubset({0., 1.}), "日历状态超出零或一")
    require(set(np.unique(learned[np.isfinite(learned)])).issubset({0., 1.}), "学习状态超出零或一")
    target = np.where(complete, .5 * calendar + .5 * learned, np.nan)
    result = pd.DataFrame({"market_date": data.date, "execution_date": data.execution_date, "decision_time": data.decision_time,
                           "month_first3": data.month_first3, "month_last5_calendar": data.month_last5_calendar,
                           "calendar_state": calendar, "learned_state": learned, "target": target, "inputs_complete": complete})
    available = result[result.execution_date.notna()]
    require((available.decision_time == available.execution_date + pd.Timedelta(hours=9)).all(), "组合决策不是执行日九点")
    require((available.market_date < available.execution_date).all(), "组合使用执行日的价格信息")
    return result


def freeze():
    old = json.loads((ROOT / "config/510300_panic_learned_equal_blend_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]}
    cfg.update(study_id="510300_CALENDAR_LEARNED_EQUAL_BLEND_V1", round=57, registered_at=now(), primary=PRIMARY,
               candidate_configurations=1, fixed_weights=[.5, .5], state_cost="BASE", new_model_fits=0,
               calendar_features=str((P6 / "features.parquet").relative_to(ROOT)),
               decision_clock="EXECUTION_DAY_09_00_ASIA_SHANGHAI", calendar_first_delivery_proven=False,
               rules="docs/510300_CALENDAR_LEARNED_EQUAL_BLEND_V1.md", position_impact=0)
    paths = [Path(__file__), ROOT / "research/calendar_liquidity_timing_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/simple_signal_blend_v1.py", ROOT / cfg["features"],
             ROOT / cfg["calendar_features"], ROOT / cfg["dividends"], ROOT / cfg["rules"], ROOT / "tests/test_calendar_learned_equal_blend_v1.py",
             ROOT / "tests/test_calendar_liquidity_timing_v1.py", ROOT / "config/510300_calendar_liquidity_timing_v1.json",
             ROOT / "docs/510300_CALENDAR_LIQUIDITY_TIMING_V1_PROTOCOL.md"]
    for period in ["evaluation", "earlier_diagnostic"]:
        paths.append(P32 / period / "BASE/REARM_RIDGE_decisions.parquet")
        for cost in cfg["costs"]:
            paths.extend([P32 / period / cost / "REARM_RIDGE_ledger.parquet", P32 / period / cost / "BUY_HOLD_ledger.parquet",
                          P46 / period / cost / "PANIC_LEARNED_HALF_ledger.parquet"])
            if period == "evaluation":
                paths.extend([P6 / period / cost / "K2_MONTH_EDGE_ledger.parquet", P6 / period / cost / "K2_MONTH_EDGE_decisions.parquet"])
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第57轮一个日历与学习状态各半方案已登记，较早日历旧对照待补算。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "日历组合登记内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["calendar_features"])
    original = pd.read_parquet(ROOT / cfg["features"])
    columns = ["date", "open", "high", "low", "close", "volume", "previous_close", "dividend", "wealth", "variance60"]
    require(data[columns].equals(original[columns]), "日历与原学习价格字段不一致")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    main, earlier, yearly, eras, counts, coverage = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        source_decisions = pd.read_parquet(P32 / period / "BASE/REARM_RIDGE_decisions.parquet")
        state = combine_states(frame, source_decisions)
        first = int(np.flatnonzero(frame.date >= start)[0])
        origins = state.iloc[first - 1:len(frame) - 1]
        require(origins.inputs_complete.all(), "组合缺少完整日历或原学习状态，不能填空仓")
        state.to_parquet(OUT / f"{period}_states.parquet", index=False)
        for (a, b), group in origins.groupby(["calendar_state", "learned_state"]):
            counts.append({"period": period, "calendar_state": a, "learned_state": b, "target": .5 * (a + b), "decision_days": len(group)})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            local_cfg = {**cfg, "evaluation_start": start}
            ledger, decisions = simulate_policy(frame, dividends, local_cfg, cost, PRIMARY, targets=state.target.to_numpy(float))
            for column in ["calendar_state", "learned_state", "target"]:
                decisions[column] = decisions.origin.map(state.set_index("market_date")[column])
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "日历组合账户未完整结算")
            require((decisions.decision_time == decisions.execution_date + pd.Timedelta(hours=9)).all(), "实际组合决策时钟不一致")
            require(decisions.reference_weight.isin([0., .5, 1.]).all(), "组合目标不是约定三档")
            accounts, names = {PRIMARY: ledger}, {PRIMARY: "月内两端与原学习退出各半"}
            if period == "evaluation":
                calendar = pd.read_parquet(P6 / period / cost_id / "K2_MONTH_EDGE_ledger.parquet")
                calendar_decisions = pd.read_parquet(P6 / period / cost_id / "K2_MONTH_EDGE_decisions.parquet")
                requested = calendar_decisions.reference_weight.to_numpy(float)
                require(np.array_equal(requested, origins.calendar_state.to_numpy(float)), "复用日历对照目标与本轮日历状态不一致")
            else:
                calendar, calendar_decisions = simulate_policy(frame, dividends, local_cfg, cost, "K2_MONTH_EDGE", targets=state.calendar_state.to_numpy(float))
                require(calendar.accounting_error.abs().max() < 1e-6 and not calendar.terminal_unliquidated.iloc[-1], "较早日历旧控制结算失败")
            save_account(folder, "K2_MONTH_EDGE", calendar, calendar_decisions)
            accounts["K2_MONTH_EDGE"], names["K2_MONTH_EDGE"] = calendar, "原月末五个自然日与月初三日"
            for key, path, name in [("REARM_RIDGE", P32 / period / cost_id / "REARM_RIDGE_ledger.parquet", "原学习退出及等待新机会"),
                                    ("PANIC_LEARNED_HALF", P46 / period / cost_id / "PANIC_LEARNED_HALF_ledger.parquet", "原急跌回升与学习退出各半"),
                                    ("BUY_HOLD", P32 / period / cost_id / "BUY_HOLD_ledger.parquet", "买入持有")]:
                saved = pd.read_parquet(path)
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            coverage.append({"period": period, "cost": cost_id, "decisions": len(decisions), "valid_decisions": int(decisions.view.eq("有效").sum()),
                             "buy_requests": int(decisions.requested_quantity.gt(0).sum()), "sell_requests": int(decisions.requested_quantity.lt(0).sum()),
                             "actual_buys": int(ledger.filled_quantity.gt(0).sum()), "actual_sells": int(ledger.filled_quantity.lt(0).sum()),
                             "complete_flattenings": int(((ledger.shares_before > 0) & ledger.shares.eq(0)).sum())})
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "组合与对照日历不完整或不同")
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
            print(f"{period}／{cost_id}：日历组合及四个对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras),
                           ("state_counts.csv", counts), ("execution_statistics.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "CALENDAR_LEARNED_EQUAL_BLEND_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8,
              "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "new_earlier_control_accounts": 2, "reused_earlier_accounts": 6,
              "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier,
              "state_counts": counts, "execution_statistics": coverage, "primary": [m for m in main if m["model"] == PRIMARY],
              "post_selected_best_base": next(m for m in main if m["model"] == PRIMARY and m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] == PRIMARY),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": result["primary"], "较早": [m for m in earlier if m["model"] in [PRIMARY, "K2_MONTH_EDGE"]], "共同状态": counts}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
