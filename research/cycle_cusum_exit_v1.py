"""原学习退出加持仓内累积转弱保护，使用固定入场前基准和完整账户。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.additional_cycle_exit_account_v1 import simulate_additional_exit
from research.cycle_cusum_controller_v1 import CycleCUSUMController
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import ExitController
from research.simple_session_divergence_v1 import make_rule, specifications

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_cycle_cusum_exit_v1"
CONFIG = ROOT / "config/510300_cycle_cusum_exit_v1.json"
PRIMARY = "RIDGE_CUSUM_EXIT"
NAME = "原学习退出加持仓累积转弱退出"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
CONTROLS = {"REARM_RIDGE": (P32, "原学习退出及等待新机会"), "REARM_NONE": (P32, "原自然退出及等待新机会"),
            "PANIC_LEARNED_HALF": (P46, "原急跌回升及学习退出各半"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
                              "confirmation_days", "specification", "saved_models"]}
    cfg.update(study_id="510300_CYCLE_CUSUM_EXIT_V1", round=60, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
               baseline_trading_days=60, allowance_sigma=.5, alarm_threshold=5., model_key="D60_INTRA__RIDGE",
               baseline_locked_at="CLOSE_BEFORE_ACTUAL_FILLED_ENTRY", first_day_return_use="EXCLUDE_BUY_DAY_CLOSE_TO_CLOSE_RETURN",
               missing_detection_policy="NO_VIEW_FOR_REMAINDER_OF_CYCLE_KEEP_PRIOR_EXIT_REQUEST", new_model_fits=0, new_reference_accounts=0,
               rules="docs/510300_CYCLE_CUSUM_EXIT_V1.md", method_source="https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc323.htm",
               pre_freeze_test_fixture_note="累计量恰为5的合成断言遇到4.999999999999999浮点结果，测试输入在冻结前改为明确超过阈值的5.2；正式阈值和控制器逻辑保持5，未读取新账户收益。",
               position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED")
    paths = [Path(__file__), ROOT / "research/cycle_cusum_controller_v1.py", ROOT / "research/additional_cycle_exit_account_v1.py",
             ROOT / "research/rearmed_cycle_exit_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
             ROOT / "research/simple_session_divergence_v1.py", ROOT / "research/simple_price_entry_exit_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
             ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"], ROOT / cfg["rules"],
             ROOT / "tests/test_cycle_cusum_exit_v1.py", ROOT / "config/510300_research_authority_v6.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(p / period / cost / f"{key}_ledger.parquet" for key, (p, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第60轮一个持仓内累积转弱退出设置已登记，尚未读取本轮账户收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "累积转弱退出登记文件发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"][cfg["model_key"]]
    item = next(r for r in specifications() if r["id"] == "D60_INTRA")
    main, earlier, yearly, eras, coverage, baselines, alarms = [], [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        rule, factor = make_rule(frame, item)
        pd.DataFrame({"date": frame.date, "d60_factor": factor, "raw_entry": rule["entry"], "raw_exit": rule["exit"][1],
                      "total_log": frame.total_log, "daily_information_complete": np.isfinite(frame.total_log)}).to_parquet(OUT / f"{period}_factors.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            original = ExitController(frame, models, cfg["confirmation_days"])
            controller = CycleCUSUMController(frame, original, cfg["baseline_trading_days"], cfg["allowance_sigma"], cfg["alarm_threshold"])
            ledger, decisions, cycles = simulate_additional_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"], controller)
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            finished = cycles.dropna(subset=["exit_date"])
            require((finished.holding_intervals >= 1).all() and ledger.accounting_error.abs().max() < 1e-6, "累积转弱账户或次日可卖约束不成立")
            require(not ledger.terminal_unliquidated.iloc[-1], "累积转弱账户终点未平仓")
            holding = decisions[decisions.cusum_cycle_id.notna()]
            coverage.append({"period": period, "cost": cost_id, "model": PRIMARY, "holding_decision_rows": len(holding),
                             "cusum_available_rows": int(holding.cusum_information_available.eq(True).sum()),
                             "cusum_no_view_rows": int(holding.cusum_information_available.eq(False).sum()),
                             "cusum_alarm_rows": int(holding.cusum_alarm.eq(True).sum()),
                             "cycles_with_cusum_alarm": int(holding.loc[holding.cusum_alarm.eq(True), "cusum_cycle_id"].nunique()),
                             "exits_with_cusum_reason": int(finished.exit_reasons.str.contains("累积转弱", regex=False).sum()),
                             "exits_with_learned_reason": int(finished.exit_reasons.str.contains("学习条件", regex=False).sum()),
                             "completed_round_trips": len(finished), "positive_cycles": int(finished.net_profit_cny.gt(0).sum()),
                             "mean_holding_intervals": float(finished.holding_intervals.mean()) if len(finished) else None})
            for cycle_id, group in holding.groupby("cusum_cycle_id"):
                first = group.iloc[0]
                require(first.cusum_baseline_end < first.origin, "入场前基准使用了买入当天收益")
                require(group.cusum_baseline_end.nunique() == 1 and group.cusum_baseline_mean.nunique() <= 1 and group.cusum_baseline_sigma.nunique() <= 1, "基准在持仓期间发生更新")
                baselines.append({"period": period, "cost": cost_id, "cycle_id": int(cycle_id), "entry_close": first.origin,
                                  "baseline_start": first.cusum_baseline_start, "baseline_end": first.cusum_baseline_end,
                                  "baseline_mean": first.cusum_baseline_mean, "baseline_sigma": first.cusum_baseline_sigma,
                                  "first_status": first.cusum_status, "maximum_cusum": float(group.cusum_value.max()) if group.cusum_value.notna().any() else None})
                fired = group[group.cusum_alarm.eq(True)]
                if len(fired):
                    alarm = fired.iloc[0]
                    alarms.append({"period": period, "cost": cost_id, "cycle_id": int(cycle_id), "first_alarm_origin": alarm.origin,
                                   "planned_exit_date": alarm.execution_date, "alarm_value": alarm.cusum_value,
                                   "learned_exit_same_origin": alarm.learned_exit_requested, "exit_reasons_at_first_alarm": alarm.exit_reasons})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for key, (source, name) in CONTROLS.items():
                saved = pd.read_parquet(source / period / cost_id / f"{key}_ledger.parquet")
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "累积转弱评价账户完整日历不一致")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[saved.date.between(left, right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            pd.DataFrame({"date": ledger.date, **{k: a.net_return.to_numpy() for k, a in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：累积转弱退出和四个旧对照账户完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("model_coverage.csv", coverage), ("cycle_fixed_baselines.csv", baselines)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    pd.DataFrame(alarms, columns=["period", "cost", "cycle_id", "first_alarm_origin", "planned_exit_date", "alarm_value", "learned_exit_same_origin", "exit_reasons_at_first_alarm"]).to_csv(OUT / "first_alarm_records.csv", index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "CYCLE_CUSUM_EXIT_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8,
              "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8,
              "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier,
              "model_coverage": coverage, "primary": primary, "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in primary),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": primary, "较早历史": [m for m in earlier if m["model"] == PRIMARY], "退出状态": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
