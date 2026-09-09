"""登记连续趋势模型、保存季度估计并运行完整净值账户。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.continuous_trend_inputs_v1 import DEFAULT_SETTINGS, walk_forward_states
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_continuous_trend_filter_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
CONFIG = ROOT / "config/510300_continuous_trend_filter_v1.json"
PRIMARY = "CONTINUOUS_TREND_FILTER"
NAME = "季度估计噪声、每日更新连续趋势"
CONTROLS = {"REARM_RIDGE": "原学习退出及等待新机会", "PANIC_LEARNED_HALF": "原急跌反弹与学习各半",
    "PANIC_ONLY": "原急跌反弹单独策略", "BUY_HOLD": "买入持有"}


def freeze():
    old = json.loads((ROOT / "config/510300_atr_trend_bands_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]}
    cfg.update(study_id="510300_CONTINUOUS_TREND_FILTER_V1", round=67, registered_at=now(), primary=PRIMARY,
        candidate_configurations=1, model_settings=DEFAULT_SETTINGS.copy(), fit_schedule="FIRST_TRADING_CLOSE_OF_CALENDAR_QUARTER",
        prior_mean="FIRST_WINDOW_LOG_WEALTH_AND_ZERO_SLOPE", prior_covariance=[[1., 0.], [0., 1.]],
        new_fit_failure="KEEP_PRIOR_VALID_MODEL_OR_NO_VIEW", missing_observation="PREDICT_TIME_ONLY_PUBLISH_NO_VIEW",
        signal_update="DAILY_HYSTERESIS_TARGET_NEXT_OPEN_ACTUAL_ACCOUNT", new_reference_accounts=0,
        rules="docs/510300_CONTINUOUS_TREND_FILTER_V1.md", runtime_versions={"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
        previous_goal_turn_classification="PROGRESS_ROUND66_COMPLETE", position_impact=0, goal_achieved=False,
        independent_validation="NOT_ESTABLISHED", point_target_does_not_establish_stable_alpha=True)
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0, "连续趋势的必要测试未通过")
    paths = [Path(__file__), ROOT / "research/continuous_trend_inputs_v1.py", ROOT / "research/event_clock_account_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / cfg["features"], ROOT / cfg["dividends"],
        ROOT / cfg["rules"], ROOT / "tests/test_continuous_trend_filter_v1.py", OUT / "tests_receipt.json", ROOT / "config/510300_research_authority_v6.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(P46 / period / cost / f"{model}_ledger.parquet" for model in CONTROLS)
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第67轮一套连续趋势模型和完整进出场已冻结，尚未估计真实历史模型或读取新账户收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "连续趋势冻结来源改变")
    require(cfg["runtime_versions"] == {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__}, "冻结后的数值库版本改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))

    def progress(record):
        day = pd.Timestamp(record["origin"])
        write_json(OUT / "quarterly_fits" / f"{day:%Y%m%d}.json", record, exclusive=True)
        names = {"FIT_CONVERGED": "参数估计收敛", "NO_VIEW_INSUFFICIENT_HISTORY": "历史不足，保持无观点",
            "SKIPPED_INCOMPLETE_HISTORY": "窗口缺失，跳过新估计", "FIT_FAILED_PRIOR_MODEL_RETAINED": "新估计失败，保留原模型", "NO_VIEW_FIT_FAILED": "估计失败且没有原模型"}
        print(f"{day.date()}：{names[record['status']]}，优化迭代{record.get('optimizer_iterations', 0)}次。", flush=True)

    states, fits = walk_forward_states(data, cfg["model_settings"], progress=progress)
    states.to_parquet(OUT / "daily_model_states.parquet", index=False)
    write_json(OUT / "quarterly_models.json", {"completed_at": now(), "updates": fits}, exclusive=True)
    fits_count = sum(f["status"] == "FIT_CONVERGED" for f in fits)
    attempts = sum("optimizer_status" in f for f in fits)
    model_summary = {"scheduled_updates": len(fits), "fit_attempts": attempts, "successful_fits": fits_count,
        "failed_fits": attempts - fits_count, "insufficient_history_updates": sum(f["status"] == "NO_VIEW_INSUFFICIENT_HISTORY" for f in fits),
        "incomplete_history_updates": sum(f["status"] == "SKIPPED_INCOMPLETE_HISTORY" for f in fits),
        "fits_at_parameter_bounds": sum(f["status"] == "FIT_CONVERGED" and f.get("parameter_at_bound", False) for f in fits),
        "optimizer_objective_evaluations": sum(f.get("objective_evaluations", 0) for f in fits)}
    write_json(OUT / "model_summary.json", model_summary, exclusive=True)
    main, earlier, yearly, eras, coverage = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        period_states = states.iloc[:len(frame)].copy()
        targets = period_states.target.to_numpy(float)
        require((np.isnan(targets) | (targets == 0) | (targets == 1)).all(), "连续趋势目标不是零一或无观点")
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, PRIMARY, targets=targets, event_mask=np.ones(len(frame), bool))
            daily = period_states.rename(columns={"date": "origin"}).drop(columns=["origin_index"])
            decisions = decisions.merge(daily, on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "连续趋势账户结算失败")
            coverage.append({"period": period, "cost": cost_id, "buy_trades": int(ledger.filled_quantity.gt(0).sum()),
                "sell_trades": int(ledger.filled_quantity.lt(0).sum()), "holding_closes": int(ledger.shares.gt(0).sum()),
                "mean_exposure": float(ledger.exposure.mean()), "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                "zero_target_origins": int(decisions.reference_weight.eq(0).sum()), "full_target_origins": int(decisions.reference_weight.eq(1).sum()),
                "no_view_origins": int(decisions.reference_weight.isna().sum()), "old_model_after_failed_fit_origins": int(decisions.fit_update_status.eq("FIT_FAILED_PRIOR_MODEL_RETAINED").sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for model, name in CONTROLS.items():
                saved = pd.read_parquet(P46 / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "对照完整日历不一致")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            pd.DataFrame({"date": ledger.date, **{k: a.net_return.to_numpy() for k, a in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：连续趋势完整账户和四个保存对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "CONTINUOUS_TREND_FILTER_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8,
        "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8,
        "new_model_fits": fits_count, "new_reference_accounts": 0, "model_summary": model_summary,
        "all_metrics": main, "earlier_diagnostics": earlier, "account_coverage": coverage, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False,
        "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"模型": model_summary, "主评价": primary, "较早": [m for m in earlier if m["model"] == PRIMARY], "成交及持仓": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    try:
        {"freeze": freeze, "run": run}[sys.argv[1]]()
    except Exception as error:
        write_json(OUT / "RUN_FAILURE.json", {"recorded_at": now(), "stage": sys.argv[1], "exception": type(error).__name__, "message": str(error)}, exclusive=True)
        raise
