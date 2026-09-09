"""仅增加信号日已知的买入价格上限，原学习退出和重新进入不变。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.capped_entry_exit_account_v1 import simulate_capped_entry
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import ExitController
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_volatility_capped_entry_v1"
CONFIG = ROOT / "config/510300_volatility_capped_entry_v1.json"
KEY, NAME = "VOL_ONE_SIGMA_CAP", "下一开盘买价不超过信号收盘加一日波动幅度"


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                               "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
                               "confirmation_days", "specification", "saved_models"]}
    cfg.update(study_id="510300_VOLATILITY_CAPPED_ENTRY_V1", round=50, registered_at=now(), primary=KEY,
               candidate_configurations=1, names={KEY: NAME}, gap_sigma=1., saved_model_group="D60_INTRA__RIDGE",
               rules="docs/510300_VOLATILITY_CAPPED_ENTRY_V1.md", new_model_fits=0, new_reference_accounts=0,
               position_impact=0, evidence_class="NEW_PRESET_PRICE_CONSTRAINT_ON_PREVIOUSLY_OBSERVED_HISTORY")
    paths = [Path(__file__), ROOT / "research/capped_entry_exit_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
             ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py",
             ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "tests/test_capped_entry_exit_account_v1.py",
             ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"], ROOT / cfg["rules"]]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第50轮已登记一个波动幅度买价上限，复用原模型及全部退出规则。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "进入价格上限登记内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"][cfg["saved_model_group"]]
    main, earlier, yearly, eras, statistics = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        rule = make_rules(frame)["D60_INTRA"]
        anchor = int(np.flatnonzero(frame.date >= pd.Timestamp(start))[0]) - 1
        require(frame.feature_valid.iloc[anchor:-1].all(), "买入限价评价期的原完整性标记缺失")
        for cost_id, cost in cfg["costs"].items():
            dest = OUT / period / cost_id
            controller = ExitController(frame, models, cfg["confirmation_days"])
            ledger, decisions, cycles = simulate_capped_entry(frame, dividends, cfg, cost, start, rule, cfg["specification"], controller, cfg["gap_sigma"])
            save_account(dest, KEY, ledger, decisions)
            cycles.to_csv(dest / f"{KEY}_cycles.csv", index=False, encoding="utf-8-sig")
            ledger[ledger.requested_quantity.gt(0)].to_csv(dest / "all_entry_attempts.csv", index=False, encoding="utf-8-sig")
            completed = cycles.dropna(subset=["exit_date"])
            require((completed.holding_intervals >= 1).all() and not ledger.terminal_unliquidated.iloc[-1], "买入限价账户未完成合法结算")
            require(ledger.accounting_error.abs().max() < 1e-6, "买入限价账户财富恒等式失败")
            buys = ledger[ledger.filled_quantity.gt(0)]
            require((buys.fill_price <= buys.entry_price_ceiling_after_ex_adjustment + 1e-9).all(), "实际模拟买价突破事前上限")
            violations = 0
            rows = completed.to_dict("records")
            for left, right in zip(rows, rows[1:]):
                a = int(np.flatnonzero(frame.date == pd.Timestamp(left["exit_date"]))[0])
                b = int(np.flatnonzero(frame.date == pd.Timestamp(right["entry_origin"]))[0])
                violations += int(a <= b and np.all(rule["entry"][a:b + 1] == left["mode"]))
            require(violations == 0, "旧进入条件未消失便再次买入")
            statistics.append({"period": period, "cost": cost_id, "completed_cycles": len(completed),
                               "buy_requests": int(ledger.requested_quantity.gt(0).sum()), "filled_buy_orders": len(buys),
                               "rejected_by_price_cap": int(ledger.entry_price_cap_rejected.sum()),
                               "unfilled_other_reasons": int((ledger.requested_quantity.gt(0) & ledger.filled_quantity.eq(0) & ~ledger.entry_price_cap_rejected).sum()),
                               "original_reentry_rule_violations": violations,
                               "cap_quantity_view_missing": int(ledger.status.eq("UNFILLED_NO_VIEW_ENTRY_PRICE_CAP").sum())})
            accounts, names = {KEY: ledger}, {KEY: NAME}
            for key, slug, source_key, name in [
                ("REARM_RIDGE", "rearmed_session_exit", "REARM_RIDGE", "原进入不设价格上限的学习退出"),
                ("ORIGINAL_TWO", "panic_learned_equal_blend", "PANIC_LEARNED_HALF", "原急跌与学习信号各半组合"),
                ("BUY_HOLD", "rearmed_session_exit", "BUY_HOLD", "买入持有")]:
                saved = pd.read_parquet(ROOT / f"reports/research/510300_{slug}_v1" / period / cost_id / f"{source_key}_ledger.parquet")
                saved.to_parquet(dest / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "进入限价与原对照全日历不同")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                intervals = [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"),
                             ("2024—终点", "2024-01-01", cfg["data_cutoff"])] if period == "evaluation" else [
                             ("2015—2016", "2015-01-01", "2016-12-31"), ("2017—2019", "2017-01-01", cfg["earlier_terminal"])]
                for label, left, right in intervals:
                    group = saved[(saved.date >= left) & (saved.date <= right)]
                    eras.append({"period": period, "cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：进入限价账户及三个原对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("entry_cap_statistics.csv", statistics)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "VOLATILITY_CAPPED_ENTRY_ACCOUNTS_COMPLETE",
              "candidate_configurations": 1, "evaluation_accounts": 8, "new_accounts_generated": 2, "reused_control_accounts": 6,
              "earlier_diagnostic_accounts": 8, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6,
              "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier,
              "entry_cap_statistics": statistics, "primary": [m for m in main if m["model"] == KEY],
              "post_selected_best_base": next(m for m in main if m["model"] == KEY and m["cost"] == "BASE"),
              "goal_achieved": False, "independent_validation": False, "position_impact": 0}
    write_json(OUT / "result.json", result)
    print(json.dumps({"状态": result["status"], "主评价": [m for m in main if m["model"] == KEY],
                      "较早历史": [m for m in earlier if m["model"] == KEY], "价格上限统计": statistics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in {"freeze", "run"}:
        raise SystemExit("用法：python -m research.volatility_capped_entry_v1 freeze 或 run")
    freeze() if sys.argv[1] == "freeze" else run()
