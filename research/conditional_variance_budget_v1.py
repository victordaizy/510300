"""有限条件方差估计与同执行口径波动对照，不重建原参考。"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from research.conditional_variance_budget_inputs_v1 import MODELS, train_schedule, forecast_frame, budget_frame
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import now, digest, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_conditional_variance_budget_v1"
CONFIG = ROOT / "config/510300_conditional_variance_budget_v1.json"
P91 = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PRIMARY = MODELS[0]
NAMES = {MODELS[0]: "条件方差预测缩减原预算", MODELS[1]: "二十日普通波动同口径对照"}
CONTROLS = {"CONTINUOUS_REFERENCE_MIN_VARIANCE": (P91, "原第91轮未缩减目标"), "REARM_RIDGE": (P32, "原线性学习退出"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    require(not CONFIG.exists(), "条件方差预算已经登记")
    old = json.loads((ROOT / "config/510300_continuous_reference_min_variance_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days", "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]}
    cfg.update(study_id="510300_CONDITIONAL_VARIANCE_BUDGET_V1", round=109, registered_at=now(), primary=PRIMARY, candidate_configurations=2,
               variance_structure="ZERO_MEAN_SYMMETRIC_GARCH_1_1", training_window=756, minimum_training_rows=242, maximum_iterations=200, target_volatility=.10,
               model_schedule="reports/research/510300_learned_cycle_exit_v1/saved_models.json", planned_fit_origins=141, new_reference_accounts=0,
               rules="docs/510300_CONDITIONAL_VARIANCE_BUDGET_V1.md", parent_rules="docs/510300_TWO_POLICY_RISK_BUDGET_V1.md",
               fit_failure="NO_VIEW_KEEP_LAST_EXPLICIT_RISK_MULTIPLIER_NO_RESTART", initial_risk_multiplier=1., source_budget_cny=0, goal_achieved=False, position_impact=0,
               independent_validation="NOT_ESTABLISHED", environment={"scipy": scipy.__version__, "numpy": np.__version__, "pandas": pd.__version__})
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0 and tests["passed"] == 6, "条件方差必要测试未通过")
    paths = [Path(__file__), ROOT / "research/conditional_variance_budget_inputs_v1.py", ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "tests/test_conditional_variance_budget_v1.py", OUT / "tests_receipt.json", ROOT / cfg["features"], ROOT / cfg["dividends"],
             ROOT / cfg["rules"], ROOT / cfg["parent_rules"], ROOT / cfg["model_schedule"], ROOT / "config/510300_research_authority_v6.json", ROOT / "config/510300_continuous_reference_min_variance_v1.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        paths.append(P91 / f"{period}_factors.parquet")
        for cost in cfg["costs"]:
            paths.extend(p / period / cost / f"{model}_ledger.parquet" for model, (p, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第109轮条件方差与普通波动两个设置已冻结，尚未拟合或生成新账户收益。", flush=True)


def run():
    clock = time.perf_counter()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "条件方差冻结输入改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    schedule = json.loads((ROOT / cfg["model_schedule"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    require(len(schedule) == cfg["planned_fit_origins"], "原月度日程数量不同")
    records = train_schedule(data, schedule, cfg)
    write_json(OUT / "saved_variance_models.json", {"models": records}, exclusive=True)
    pd.DataFrame([{k: v for k, v in r.items() if k not in {"model", "theta"}} | (r["model"] or {}) for r in records]).to_csv(OUT / "每月条件方差系数与拟合状态.csv", index=False, encoding="utf-8-sig")
    forecasts = forecast_frame(data, records, cfg["annual_days"])
    forecasts.to_parquet(OUT / "variance_forecasts.parquet", index=False)
    forecasts.to_csv(OUT / "每天已知波动与预测.csv", index=False, encoding="utf-8-sig")
    print(f"141个原月度日程完成，成功{sum(r['status']=='FIT_COMPLETE' for r in records)}，其余逐条保留无模型状态。", flush=True)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main), ("earlier_diagnostic", data[data.date.le(cfg["earlier_terminal"])].copy(), cfg["earlier_start"], earlier)]:
        parent = pd.read_parquet(P91 / f"{period}_factors.parquet")
        factors = budget_frame(parent, forecasts.iloc[:len(frame)].copy(), cfg["target_volatility"])
        factors.to_parquet(OUT / f"{period}_factors.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            accounts, names = {}, NAMES.copy()
            for model in MODELS:
                ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, model, targets=factors[model].to_numpy(float), event_mask=np.ones(len(frame), bool))
                decisions = decisions.merge(factors.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
                save_account(folder, model, ledger, decisions)
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "波动预算账户未完整结算")
                coverage.append({"period": period, "cost": cost_id, "model": model, "holding_closes": int(ledger.shares.gt(0).sum()), "buy_trades": int(ledger.filled_quantity.gt(0).sum()),
                                 "sell_trades": int(ledger.filled_quantity.lt(0).sum()), "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                                 "no_view_target_origins": int(decisions.reference_weight.isna().sum()), "no_new_variance_view_origins": int(decisions[f"{model}_status"].str.startswith("NO_VIEW").sum()), "mean_exposure": float(ledger.exposure.mean())})
                accounts[model] = ledger
            for model, (p, name) in CONTROLS.items():
                saved = pd.read_parquet(p / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "同规则波动账户日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"]-bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                dest.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：两个新实际账户及三个保存对照已完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "CONDITIONAL_VARIANCE_BUDGET_ACCOUNTS_COMPLETE", "candidate_configurations": 2,
              "evaluation_accounts": len(main), "new_accounts_generated": 4, "reused_control_accounts": 6, "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 4,
              "reused_earlier_accounts": 6, "new_model_fits": len(records), "completed_fits": sum(r["status"] == "FIT_COMPLETE" for r in records),
              "failed_fits": sum(r["status"] == "NO_VIEW_VARIANCE_FIT_FAILED" for r in records), "no_view_fit_origins": sum(r["status"] != "FIT_COMPLETE" for r in records),
              "new_reference_accounts": 0, "run_seconds": time.perf_counter()-clock, "all_metrics": main, "earlier_diagnostics": earlier, "primary": primary, "account_coverage": coverage,
              "post_selected_best_base": max((m for m in main if m["model"] in MODELS and m["cost"] == "BASE"), key=lambda m: float('-inf') if m["net_sharpe"] is None else m["net_sharpe"]),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] in MODELS), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": [m for m in main if m["model"] in MODELS], "较早": [m for m in earlier if m["model"] in MODELS], "拟合成功": result["completed_fits"], "耗时": result["run_seconds"], "覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
