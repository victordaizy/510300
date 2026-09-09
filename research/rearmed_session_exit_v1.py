"""复用已训练的日内强弱退出模型，只改变再次进入的状态规则。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import ExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_rearmed_session_exit_v1"
CONFIG = ROOT / "config/510300_rearmed_session_exit_v1.json"
PARENT = ROOT / "reports/research/510300_learned_cycle_exit_v1"
NAMES = {"NONE": "原退出＋等待新机会", "RIDGE": "线性退出＋等待新机会", "TREE": "两层树退出＋等待新机会"}


def freeze():
    old = json.loads((ROOT / "config/510300_learned_cycle_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                               "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "confirmation_days"]}
    cfg.update({"study_id": "510300_REARMED_SESSION_EXIT_V1", "round": 32, "registered_at": now(), "primary": "REARM_TREE",
                "candidate_configurations": 3, "names": NAMES, "specification": old["candidate_specs"]["D60_INTRA"],
                "rules": "docs/510300_REARMED_SESSION_EXIT_V1.md", "new_model_fits": 0, "position_impact": 0,
                "saved_models": str((PARENT / "saved_models.json").relative_to(ROOT))})
    paths = [Path(__file__), ROOT / "research/rearmed_cycle_exit_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py",
             ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"],
             ROOT / "tests/test_rearmed_cycle_exit_account_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第32轮已登记三个再次进入设置，全部复用现有模型，不重新训练。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "再次进入登记内容发生变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]
    main, earlier, yearly, eras, states = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        rule = make_rules(frame)["D60_INTRA"]
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            accounts, names = {}, {}
            for kind, name in NAMES.items():
                model_id = "REARM_" + kind
                controller = None if kind == "NONE" else ExitController(frame, models["D60_INTRA__" + kind], cfg["confirmation_days"])
                ledger, decisions, cycles = simulate_rearmed_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"], controller)
                save_account(folder, model_id, ledger, decisions)
                cycles.to_csv(folder / f"{model_id}_cycles.csv", index=False, encoding="utf-8-sig")
                completed = cycles.dropna(subset=["exit_date"])
                violations = 0
                records = completed.to_dict("records")
                for left, right in zip(records, records[1:]):
                    a = int(np.flatnonzero(frame.date == pd.Timestamp(left["exit_date"]))[0])
                    b = int(np.flatnonzero(frame.date == pd.Timestamp(right["entry_origin"]))[0])
                    violations += int(a <= b and np.all(rule["entry"][a:b + 1] == left["mode"]))
                require(violations == 0, "出现旧条件未消失却再次进入")
                require((completed.holding_intervals >= 1).all() and ledger.accounting_error.abs().max() < 1e-6, "再次进入账户或T+1状态不正确")
                require(not ledger.terminal_unliquidated.iloc[-1], "再次进入账户终点未平仓")
                states.append({"period": period, "cost": cost_id, "model": model_id,
                               "waiting_for_new_condition_rows": int(decisions.action.str.contains("旧入场条件", regex=False).sum()),
                               "completed_round_trips": len(completed), "reentries_without_original_condition_reset": violations,
                               "learned_exit_cycles": int(completed.exit_reasons.str.contains("学习条件", regex=False).sum())})
                accounts[model_id], names[model_id] = ledger, name
            for suffix, name in [("RIDGE", "原线性退出"), ("TREE", "原两层树退出"), ("CLOSE_NEXT_OPEN", "原日内强弱规则")]:
                model_id = "D60_INTRA__" + suffix
                ledger = pd.read_parquet(PARENT / period / cost_id / f"{model_id}_ledger.parquet")
                decisions = pd.read_parquet(PARENT / period / cost_id / f"{model_id}_decisions.parquet")
                save_account(folder, model_id, ledger, decisions)
                accounts[model_id], names[model_id] = ledger, name
            bh = pd.read_parquet(PARENT / period / cost_id / "BUY_HOLD_ledger.parquet")
            bh.to_parquet(folder / "BUY_HOLD_ledger.parquet", index=False)
            accounts["BUY_HOLD"], names["BUY_HOLD"] = bh, "买入持有"
            base = summarize(bh, cfg)
            for model_id, ledger in accounts.items():
                require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(bh.date)), "再次进入账户评价日期缺失")
                m = {"cost": cost_id, "model": model_id, "name": names[model_id], **summarize(ledger, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                dest.append(m)
                if period == "evaluation":
                    for year, group in ledger.groupby(ledger.date.dt.year):
                        yearly.append({"cost": cost_id, "model": model_id, "year": int(year), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = ledger[(ledger.date >= left) & (ledger.date <= right)]
                        eras.append({"cost": cost_id, "model": model_id, "era": label, **summarize(group, cfg)})
            pd.DataFrame({"date": bh.date, **{k: v.net_return.to_numpy() for k, v in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：三个等待新机会设置和四个对照已完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("entry_state_statistics.csv", states)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    best = max((m for m in main if m["cost"] == "BASE" and m["model"].startswith("REARM_")), key=lambda m: m["net_sharpe"] if m["net_sharpe"] is not None else -999)
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "REARMED_ENTRY_ACCOUNTS_COMPLETE", "candidate_configurations": 3,
              "evaluation_accounts": 14, "new_accounts_generated": 6, "reused_control_accounts": 8, "earlier_diagnostic_accounts": 14,
              "new_earlier_diagnostic_accounts": 6, "reused_earlier_accounts": 8, "new_model_fits": 0,
              "all_metrics": main, "earlier_diagnostics": earlier, "entry_state_statistics": states,
              "primary": [m for m in main if m["model"] == cfg["primary"]], "post_selected_best_base": best,
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"].startswith("REARM_")),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "基础费用结果": [{k: m[k] for k in ["model", "net_sharpe", "annualized_return", "max_drawdown", "trade_count"]} for m in main if m["cost"] == "BASE"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
