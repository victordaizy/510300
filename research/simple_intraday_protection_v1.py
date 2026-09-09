"""三种既有入场规则的盘中保护退出小规模比较。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.intraday_protective_exit_v1 import simulate_protective
from research.simple_price_entry_exit_v1 import signals, simulate_policy, specifications as price_specs
from research.simple_session_divergence_v1 import make_rule, specifications as session_specs

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_simple_intraday_protection_v1"
CONFIG = ROOT / "config/510300_simple_intraday_protection_v1.json"
NAMES = {"S1_TREND_REBOUND": "趋势突破与震荡反弹切换", "R2_Z_CONFIRM": "偏离均值后首日回升",
         "D60_INTRA": "六十日日内强于隔夜"}


def freeze():
    old = json.loads((ROOT / "config/510300_simple_price_entry_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction",
                               "annual_days", "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends"]}
    session = json.loads((ROOT / "config/510300_simple_session_divergence_v1.json").read_text(encoding="utf-8"))
    specs = price_specs()
    cfg.update({"study_id": "510300_SIMPLE_INTRADAY_PROTECTION_V1", "round": 30, "registered_at": now(),
                "primary": "S1_TREND_REBOUND__TRIGGER", "candidate_names": NAMES, "candidate_configurations": 3,
                "candidate_specs": {k: specs[k] if k in specs else session["common_specification"] for k in NAMES},
                "assumptions": ["TRIGGER", "DAY_LOW"], "earlier_start": "2015-01-05", "earlier_terminal": "2019-12-31",
                "rules": "docs/510300_INTRADAY_PROTECTIVE_EXIT_V1.md", "goal_achieved": False,
                "evidence_class": "PREVIOUSLY_OBSERVED_HISTORY_WITH_CONDITIONAL_INTRADAY_FILLS", "position_impact": 0})
    paths = [Path(__file__), ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"],
             ROOT / "research/intraday_protective_exit_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
             ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/simple_session_divergence_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "tests/test_intraday_protective_exit_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第30轮已登记：三个既有入场信号、两种盘中保护成交情景，尚未读取本轮账户收益。", flush=True)


def make_rules(frame):
    rules = signals(frame)
    item = next(x for x in session_specs() if x["id"] == "D60_INTRA")
    rules["D60_INTRA"], _ = make_rule(frame, item)
    return {k: rules[k] for k in NAMES}


def old_folder(key, cost):
    study = "510300_simple_session_divergence_v1" if key == "D60_INTRA" else "510300_simple_price_entry_exit_v1"
    return ROOT / "reports/research" / study / "evaluation" / cost


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "盘中保护登记文件发生变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    div = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    main, earlier, yearly, eras, execution_rows = [], [], [], [], []
    all_ledgers = {}
    for period, frame, start, destination in [
        ("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier),
    ]:
        rules = make_rules(frame)
        pd.DataFrame({"date": frame.date, **{k: v["entry"] for k, v in rules.items()}}).to_parquet(OUT / f"{period}_entry_modes.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            folder.mkdir(parents=True, exist_ok=True)
            accounts = {}
            metadata = {}
            for key, name in NAMES.items():
                spec = cfg["candidate_specs"][key]
                for assumption in cfg["assumptions"]:
                    model = key + "__" + assumption
                    ledger, decisions, cycles = simulate_protective(frame, div, cfg, cost, start, rules[key], spec, assumption)
                    save_account(folder, model, ledger, decisions)
                    cycles.to_csv(folder / f"{model}_cycles.csv", index=False, encoding="utf-8-sig")
                    finished = cycles.dropna(subset=["exit_date"])
                    require((finished.holding_intervals >= 1).all(), "出现违反次日可卖的保护交易")
                    require(ledger.accounting_error.abs().max() < 1e-6, "盘中保护账户财富不平")
                    require(not ledger.terminal_unliquidated.iloc[-1], "研究终点仍有未卖出份额")
                    accounts[model] = ledger
                    metadata[model] = {"signal": key, "name": name, "assumption": assumption, "reused": False}
                    execution_rows.append({"period": period, "cost": cost_id, "model": model,
                                           "intraday_filled_exits": int(ledger.conditional_intraday_fill.sum()),
                                           "open_protective_exits": int(((ledger.execution_clock == "OPEN_PROTECTIVE") & (ledger.filled_quantity < 0)).sum()),
                                           "blocked_protective_attempts": int(((ledger.execution_clock.isin(["OPEN_PROTECTIVE", "INTRADAY_CONDITIONAL"])) & (ledger.filled_quantity == 0)).sum()),
                                           "completed_round_trips": len(finished), "max_accounting_error": float(ledger.accounting_error.abs().max())})
                control = key + "__CLOSE_NEXT_OPEN"
                if period == "evaluation":
                    source = old_folder(key, cost_id)
                    ledger = pd.read_parquet(source / f"{key}_ledger.parquet")
                    decisions = pd.read_parquet(source / f"{key}_decisions.parquet")
                else:
                    ledger, decisions, cycles = simulate_policy(frame, div, cfg, cost, start, rules[key], spec)
                    cycles.to_csv(folder / f"{control}_cycles.csv", index=False, encoding="utf-8-sig")
                save_account(folder, control, ledger, decisions)
                accounts[control] = ledger
                metadata[control] = {"signal": key, "name": name, "assumption": "CLOSE_NEXT_OPEN", "reused": period == "evaluation"}
            if period == "evaluation":
                source = ROOT / "reports/research/510300_simple_price_entry_exit_v1/evaluation" / cost_id
            else:
                source = ROOT / "reports/research/510300_simple_signal_blend_v1/earlier_diagnostic" / cost_id
            bh = pd.read_parquet(source / "BUY_HOLD_ledger.parquet")
            accounts["BUY_HOLD"] = bh
            bh.to_parquet(folder / "BUY_HOLD_ledger.parquet", index=False)
            metadata["BUY_HOLD"] = {"signal": "BUY_HOLD", "name": "买入持有", "assumption": "BUY_HOLD", "reused": True}
            base = summarize(bh, cfg)
            expected_days = 1604 if period == "evaluation" else 1219
            for model, ledger in accounts.items():
                require(len(ledger) == expected_days and pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(bh.date)), "完整账户日期不一致")
                metric = {"cost": cost_id, "model": model, **metadata[model], **summarize(ledger, cfg)}
                metric["annualized_return_excess_vs_buy_hold"] = metric["annualized_return"] - base["annualized_return"]
                metric["meets_point_target"] = metric["net_sharpe"] is not None and metric["net_sharpe"] >= 1.2
                destination.append(metric)
                if period == "evaluation":
                    for year, group in ledger.groupby(ledger.date.dt.year):
                        yearly.append({"cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = ledger[(ledger.date >= left) & (ledger.date <= right)]
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(group, cfg)})
            all_ledgers[(period, cost_id)] = accounts
            pd.DataFrame({"date": bh.date, **{k: v.net_return.to_numpy() for k, v in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：六个盘中保护账户、三个原规则对照及买入持有已完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("execution_statistics.csv", execution_rows)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    candidates = [m for m in main if m["cost"] == "BASE" and m["assumption"] in cfg["assumptions"]]
    best = max(candidates, key=lambda m: m["net_sharpe"] if m["net_sharpe"] is not None else -999)
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "PROTECTIVE_EXIT_ACCOUNTS_COMPLETE",
              "candidate_configurations": 3, "execution_assumptions": 2, "evaluation_accounts": 20,
              "new_accounts_generated": 12, "reused_control_accounts": 8, "earlier_diagnostic_accounts": 20,
              "new_earlier_diagnostic_accounts": 18, "reused_earlier_accounts": 2,
              "all_metrics": main, "earlier_diagnostics": earlier,
              "primary": [m for m in main if m["model"] == cfg["primary"]], "post_selected_best_base": best,
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["assumption"] in cfg["assumptions"]),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "real_intraday_execution": "NOT_VERIFIED",
              "execution_statistics": execution_rows, "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    summary = [{k: m[k] for k in ["model", "net_sharpe", "annualized_return", "max_drawdown", "trade_count"]} for m in main if m["cost"] == "BASE"]
    print(json.dumps({"状态": result["status"], "基础费用完整结果": summary, "目标达到": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
