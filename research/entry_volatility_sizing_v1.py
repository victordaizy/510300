"""原学习退出及重新进入规则，只改变每次进入的预算比例。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.entry_sized_rearmed_account_v1 import simulate_entry_sized_exit
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import ExitController
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_entry_volatility_sizing_v1"
CONFIG = ROOT / "config/510300_entry_volatility_sizing_v1.json"
PARENT = ROOT / "reports/research/510300_rearmed_session_exit_v1"
NAMES = {"ENTRY_VOL10": "按二十日波动控制初始仓位", "ENTRY_HALF": "每次固定半仓进入"}


def entry_fraction(data, t, key, volatility_target=.10):
    if key == "ENTRY_HALF":
        return .5
    require(key == "ENTRY_VOL10", "未知初始仓位设置")
    volatility = float(data.vol20.iloc[t])
    return min(1., volatility_target / volatility) if np.isfinite(volatility) and volatility > 0 else None


def order_schedule(ledger):
    trades = ledger[ledger.filled_quantity != 0]
    return [(str(r.date.date()), int(np.sign(r.filled_quantity))) for r in trades.itertuples()]


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
                              "confirmation_days", "specification", "saved_models"]}
    cfg.update(study_id="510300_ENTRY_VOLATILITY_SIZING_V1", round=41, registered_at=now(), primary="ENTRY_VOL10", candidate_configurations=2,
               candidate_names=NAMES, volatility_target=.10, fixed_fraction=.5, rules="docs/510300_ENTRY_VOLATILITY_SIZING_V1.md",
               new_model_fits=0, position_impact=0)
    paths = [Path(__file__), ROOT / "research/entry_sized_rearmed_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
             ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / cfg["features"],
             ROOT / cfg["dividends"], ROOT / cfg["saved_models"], ROOT / cfg["rules"], ROOT / "tests/test_entry_volatility_sizing_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第41轮两个初始仓位设置已登记，不训练新模型或扫描风险目标。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "初始仓位登记内容发生变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    main, early, yearly, eras, sizing = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        rule = make_rules(frame)["D60_INTRA"]
        for cost_id, cost in cfg["costs"].items():
            folder, accounts, names = OUT / period / cost_id, {}, {}
            baseline = pd.read_parquet(PARENT / period / cost_id / "REARM_RIDGE_ledger.parquet")
            for key, name in NAMES.items():
                provider = lambda t, policy=key: entry_fraction(frame, t, policy, cfg["volatility_target"])
                ledger, decisions, cycles = simulate_entry_sized_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"],
                    ExitController(frame, models, cfg["confirmation_days"]), provider)
                save_account(folder, key, ledger, decisions)
                cycles.to_csv(folder / f"{key}_cycles.csv", index=False, encoding="utf-8-sig")
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "初始仓位账户结算失败")
                require(not ((ledger.filled_quantity > 0) & (ledger.shares_before > 0)).any(), "初始仓位规则发生了持仓中追加")
                if len(cycles):
                    require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "初始仓位违反买入次日可卖")
                fractions = cycles.entry_fraction if len(cycles) else pd.Series(dtype=float)
                sizing.append({"period": period, "cost": cost_id, "model": key, "completed_cycles": len(cycles),
                               "entry_fraction_min": float(fractions.min()) if len(fractions) else None,
                               "entry_fraction_median": float(fractions.median()) if len(fractions) else None,
                               "entry_fraction_max": float(fractions.max()) if len(fractions) else None,
                               "full_budget_cycles": int((fractions >= 1. - 1e-12).sum()),
                               "no_view_budget_decisions": int((decisions.entry_sizing_status == "NO_VIEW_ENTRY_SIZING_INPUT").sum()),
                               "same_order_dates_and_directions_as_original": order_schedule(ledger) == order_schedule(baseline)})
                accounts[key], names[key] = ledger, name
            baseline.to_parquet(folder / "REARM_RIDGE_ledger.parquet", index=False)
            accounts["REARM_RIDGE"], names["REARM_RIDGE"] = baseline, "原满仓进入的线性退出候选"
            bh = pd.read_parquet(PARENT / period / cost_id / "BUY_HOLD_ledger.parquet")
            bh.to_parquet(folder / "BUY_HOLD_ledger.parquet", index=False)
            accounts["BUY_HOLD"], names["BUY_HOLD"] = bh, "买入持有"
            base = summarize(bh, cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(bh.date)), "初始仓位评价日期不完整")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                dest.append(m)
                if period == "evaluation":
                    for year, group in saved.groupby(saved.date.dt.year):
                        yearly.append({"cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[(saved.date >= left) & (saved.date <= right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：两个初始仓位及两个原样对照已完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("entry_sizing_statistics.csv", sizing)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    best = max((m for m in main if m["model"] in NAMES and m["cost"] == "BASE"), key=lambda m: m["net_sharpe"] if m["net_sharpe"] is not None else -999)
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "ENTRY_VOLATILITY_SIZING_COMPLETE", "candidate_configurations": 2,
              "evaluation_accounts": 8, "new_accounts_generated": 4, "reused_control_accounts": 4, "earlier_diagnostic_accounts": 8,
              "new_earlier_diagnostic_accounts": 4, "reused_earlier_accounts": 4, "new_model_fits": 0,
              "all_metrics": main, "earlier_diagnostics": early, "entry_sizing_statistics": sizing,
              "primary": [m for m in main if m["model"] == cfg["primary"]], "post_selected_best_base": best,
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] in NAMES),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "新设置": [m for m in main if m["model"] in NAMES],
                      "较早": [m for m in early if m["model"] in NAMES], "进入仓位与日期": sizing}, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
