"""把同一套自身回撤控制固定应用于原91及92两个保存目标。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.own_cushion_account_v1 import simulate_cushion_account
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_own_cushion_risk_v1"
CONFIG = ROOT / "config/510300_own_cushion_risk_v1.json"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P92 = ROOT / "reports/research/510300_cycle_adverse_risk_v1"
PARENTS = {"OWN_CUSHION_CONTINUOUS": ("CONTINUOUS_REFERENCE_MIN_VARIANCE", P91, "第91轮加自身回撤控制"), "OWN_CUSHION_CYCLE_RISK": ("CYCLE_ADVERSE_RISK", P92, "第92轮加自身回撤控制")}
PRIMARY = "OWN_CUSHION_CONTINUOUS"
CONTROLS = {"CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "原第91轮连续历史风险预算"), "CYCLE_ADVERSE_RISK": (P92, "原第92轮周期风险预算"), "BUY_HOLD": (P91, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "自身回撤控制已登记，不能重复冻结")
    old = json.loads((ROOT / "config/510300_continuous_reference_min_variance_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days", "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]}
    cfg.update(study_id="510300_OWN_CUSHION_RISK_V1", round=94, registered_at=now(), primary=PRIMARY, candidate_configurations=2,
        candidates={key: {"parent": value[0], "parent_folder": str(value[1].relative_to(ROOT)), "name": value[2]} for key, value in PARENTS.items()},
        floor_fraction=.8, cushion_multiplier=5., high_water="OWN_ACTUAL_ACCOUNT_NONTERMINAL_CLOSE_AND_INITIAL_CAPITAL",
        target_method="PARENT_TARGET_TIMES_OWN_CUSHION_SCALE", reset_rule="NONE", rules="docs/510300_OWN_CUSHION_RISK_V1.md",
        new_model_fits=0, new_reference_accounts=0, independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0,
        source_note="Cont与Tankov组合保险机制提供设计参考；底线与倍数是本轮固定选择，非实证有效性保证。")
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0, "自身回撤控制必要测试未通过")
    paths = [Path(__file__), ROOT / "research/own_cushion_account_v1.py", ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "tests/test_own_cushion_account_v1.py", ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"], OUT / "tests_receipt.json", ROOT / "config/510300_continuous_reference_min_variance_v1.json", ROOT / "config/510300_cycle_adverse_risk_v1.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        paths.extend(folder / f"{period}_factors.parquet" for folder in [P91, P92])
        for cost in cfg["costs"]:
            paths.extend(folder / period / cost / f"{model}_ledger.parquet" for model, (folder, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG, cfg, exclusive=True)
    print("第94轮两个父策略使用相同自身回撤控制已冻结，尚无新账户收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "自身回撤控制冻结来源变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main), ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])], cfg["earlier_start"], earlier)]:
        first = int(np.flatnonzero(frame.date.ge(start))[0])
        targets = {}
        for model, (_, parent_folder, _) in PARENTS.items():
            parent = pd.read_parquet(parent_folder / f"{period}_factors.parquet")
            require(pd.DatetimeIndex(parent.date).equals(pd.DatetimeIndex(frame.date)), "原股票意向日历不一致")
            require(np.isfinite(parent.target.iloc[first-1:-1]).all(), "原完整目标缺失")
            targets[model] = parent.target.to_numpy(float)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            accounts, names = {}, {}
            for model, (_, _, name) in PARENTS.items():
                ledger, decisions = simulate_cushion_account(frame, dividends, cfg, cost, start, model, targets[model])
                save_account(folder, model, ledger, decisions)
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "自身回撤控制账户未完整结算")
                coverage.append({"period": period, "cost": cost_id, "model": model, "holding_closes": int(ledger.shares.gt(0).sum()),
                    "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                    "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                    "mean_exposure": float(ledger.exposure.mean()), "minimum_risk_scale": float(decisions.risk_scale.min()), "mean_risk_scale": float(decisions.risk_scale.mean()),
                    "zero_scale_origins": int(decisions.risk_scale.eq(0).sum()), "missing_parent_origins": int(decisions.parent_target.isna().sum()),
                    "closes_below_reference_floor": int(decisions.floor_breached_at_origin.sum()), "days_below_previous_decision_floor": int(ledger.below_previous_decision_floor.sum())})
                accounts[model], names[model] = ledger, name
            for model, (parent_folder, name) in CONTROLS.items():
                ledger = pd.read_parquet(parent_folder / period / cost_id / f"{model}_ledger.parquet")
                ledger.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = ledger, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, ledger in accounts.items():
                require(pd.DatetimeIndex(ledger.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])), "新旧账户完整评价日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(ledger, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"]-bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                destination.append(m)
                for year, group in ledger.groupby(ledger.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(ledger[ledger.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：两个自身回撤控制完整账户和三个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    candidates = [m for m in main if m["model"] in PARENTS]
    write_json(OUT / "result.json", {"study_id": cfg["study_id"], "completed_at": now(), "status": "OWN_CUSHION_RISK_ACCOUNTS_COMPLETE", "candidate_configurations": 2,
        "evaluation_accounts": 10, "new_accounts_generated": 4, "reused_control_accounts": 6, "earlier_diagnostic_accounts": 10, "new_earlier_diagnostic_accounts": 4, "reused_earlier_accounts": 6,
        "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier, "account_coverage": coverage,
        "primary": [m for m in candidates if m["model"] == PRIMARY], "post_selected_best_base": max((m for m in candidates if m["cost"] == "BASE"), key=lambda m: m["net_sharpe"] if m["net_sharpe"] is not None else -999),
        "historical_point_target_met": any(m["meets_point_target"] for m in candidates), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    print(json.dumps({"主结果": candidates, "较早结果": [m for m in earlier if m["model"] in PARENTS], "账户覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
