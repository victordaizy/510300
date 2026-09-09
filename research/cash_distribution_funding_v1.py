"""固定一个现金分红与资金利率条件，运行两段历史和两档费用。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.cash_distribution_funding_inputs_v1 import distribution_frame, known_funding
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_cash_distribution_funding_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
CONFIG = ROOT / "config/510300_cash_distribution_funding_v1.json"
PRIMARY = "CASH_DISTRIBUTION_FUNDING"
NAME = "过去现金分红收益率高于资金利率时持有"
RATES = "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2/dr007_daily_20150105_20260814.parquet"
RATE_RECEIPT = "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_2/dr007_tushare_source_acquisition_manifest.json"
DIVIDEND_RECEIPT = "reports/research/510300_adaptive_allocation_v1/frozen_inputs/dividend_coverage.json"
OLD_FUNDING = "reports/research/510300_total_reverse_repo_v2/features.parquet"
CONTROLS = {"REARM_RIDGE": "原学习退出及等待新机会", "PANIC_LEARNED_HALF": "原急跌反弹与学习各半",
    "PANIC_ONLY": "原急跌反弹单独策略", "BUY_HOLD": "买入持有"}


def freeze():
    require(not CONFIG.exists(), "本轮已冻结，不能重复登记")
    old = json.loads((ROOT / "config/510300_atr_trend_bands_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]}
    cov = json.loads((ROOT / DIVIDEND_RECEIPT).read_text(encoding="utf-8"))
    src = json.loads((ROOT / RATE_RECEIPT).read_text(encoding="utf-8"))
    require(cov["complete_history_confirmed"] and cov["distribution_file_sha256"] == digest(ROOT / cfg["dividends"]), "分红完整覆盖或账本身份不符")
    require(src["status"] == "PASS_DR007_SOURCE_ADMITTED_FULL_MARKET_SESSION_COVERAGE" and src["curated_artifact"]["sha256"] == digest(ROOT / RATES), "资金来源身份或既有状态不符")
    require(src["provider_value_field"] == "weight" and src["value_semantics"] == "WEIGHTED_AVERAGE_RATE_PERCENT" and not src["interpolation_performed"] and not src["substitute_used"], "资金口径不是已核对的加权平均百分率")
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0, "关键日期和账户测试尚未通过")
    data = pd.read_parquet(ROOT / cfg["features"])
    rates = pd.read_parquet(ROOT / RATES)
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    require(len(dividends) == cov["event_count"] and (dividends.record_date <= dividends.ex_date).all() and (dividends.ex_date <= dividends.payment_date).all(), "分红日期关系或数量错误")
    require(pd.Timestamp(cov["coverage_start"]) <= data.date.iloc[0] and pd.Timestamp(cov["coverage_end"]) >= data.date.iloc[-1], "分红覆盖不足本轮日期")
    funding = known_funding(data.date, rates, 7)
    prior = pd.read_parquet(ROOT / OLD_FUNDING, columns=["date", "dr_date", "dr_available_at", "dr007_known", "dr_age_days"])
    require(pd.DatetimeIndex(funding.date).equals(pd.DatetimeIndex(prior.date)), "原资金来源日历不同")
    for field in ["dr_date", "dr_available_at"]:
        pd.testing.assert_series_equal(funding[field].astype("datetime64[ns]").reset_index(drop=True),
            prior[field].astype("datetime64[ns]").reset_index(drop=True), check_names=False)
    np.testing.assert_allclose(funding.dr_age_days, prior.dr_age_days, equal_nan=True, atol=0, rtol=0)
    np.testing.assert_allclose(funding.dr_annual_rate, prior.dr007_known, equal_nan=True, atol=1e-15, rtol=0)
    receipt = {"checked_at": now(), "status": "EXISTING_DIVIDEND_AND_INDEPENDENT_FUNDING_CLOCK_CONFIRMED", "rate_rows": len(rates),
        "reconciled_calendar_rows": len(funding), "known_rate_rows": int(funding.dr_date.notna().sum()), "valid_rate_rows": int(funding.dr_valid.sum()),
        "dividend_events": len(dividends), "dividend_event_coverage_start": cov["coverage_start"], "dividend_event_coverage_end": cov["coverage_end"],
        "rate_source_first": str(rates.date.iloc[0]), "rate_source_last": str(rates.date.iloc[-1]), "rate_unit": "PERCENT_DIVIDED_BY_100",
        "rate_clock": "NEXT_STOCK_TRADING_DAY_OPEN_AFTER_RATE_DATE", "max_rate_age_natural_days": 7,
        "historical_vendor_first_delivery_proven": False, "old_joint_feature_gate_used": False,
        "new_market_data_downloaded": False, "new_strategy_returns_read": False, "security_audit_performed": False}
    write_json(OUT / "source_receipt.json", receipt, exclusive=True)
    cfg.update(study_id="510300_CASH_DISTRIBUTION_FUNDING_V1", round=68, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
        rates=RATES, max_rate_age_days=7, dividend_window="ONE_CALENDAR_YEAR_LEFT_EXCLUSIVE_EX_DATE_RIGHT_INCLUSIVE",
        dividend_coverage_start=cov["coverage_start"], dividend_coverage_end=cov["coverage_end"], threshold=0., entry="SPREAD_STRICTLY_POSITIVE", exit="SPREAD_NONPOSITIVE",
        missing="NO_VIEW_KEEP_ACTUAL_SHARES", signal_update="EVERY_CLOSE_NEXT_OPEN_ACTUAL_ACCOUNT", new_model_fits=0, new_reference_accounts=0,
        historical_vendor_first_delivery_proven=False, source_budget_cny=0,
        rules="docs/510300_CASH_DISTRIBUTION_FUNDING_V1.md", previous_goal_turn_classification="PROGRESS_ROUND67_COMPLETED",
        position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED")
    paths = [Path(__file__), ROOT / "research/cash_distribution_funding_inputs_v1.py", ROOT / "research/event_clock_account_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"], ROOT / "tests/test_cash_distribution_funding_v1.py",
        ROOT / RATES, ROOT / RATE_RECEIPT, ROOT / DIVIDEND_RECEIPT, ROOT / OLD_FUNDING,
        OUT / "tests_receipt.json", OUT / "source_receipt.json", OUT / "prefreeze_timestamp_comparison_note.json",
        ROOT / "config/510300_research_authority_v6.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(P46 / period / cost / f"{model}_ledger.parquet" for model in CONTROLS)
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第68轮现金分红与资金利率一个设置已冻结，尚无新策略收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "分红与资金比较冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    rates = pd.read_parquet(ROOT / cfg["rates"])
    factors = distribution_frame(data, dividends, rates, cfg["dividend_coverage_start"], cfg["dividend_coverage_end"], cfg["max_rate_age_days"])
    factors.to_parquet(OUT / "factors.parquet", index=False)
    main, earlier, yearly, eras, counts, coverage = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        states = factors.iloc[:len(frame)].copy()
        target = states.target.to_numpy(float)
        anchor = int(np.flatnonzero(frame.date >= pd.Timestamp(start))[0]) - 1
        states.to_parquet(OUT / f"{period}_states.parquet", index=False)
        for (state, value), group in states.iloc[anchor:-1].groupby(["source_state", "target"], dropna=False):
            counts.append({"period": period, "source_state": state, "target": value, "decision_origins": len(group)})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, PRIMARY, targets=target, event_mask=np.ones(len(frame), bool))
            decisions = decisions.merge(states.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "分红与资金比较完整账户结算失败")
            coverage.append({"period": period, "cost": cost_id, "buy_trades": int(ledger.filled_quantity.gt(0).sum()),
                "sell_trades": int(ledger.filled_quantity.lt(0).sum()), "holding_closes": int(ledger.shares.gt(0).sum()),
                "mean_exposure": float(ledger.exposure.mean()), "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                "zero_target_origins": int(decisions.reference_weight.eq(0).sum()), "full_target_origins": int(decisions.reference_weight.eq(1).sum()),
                "no_view_origins": int(decisions.reference_weight.isna().sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for model, name in CONTROLS.items():
                saved = pd.read_parquet(P46 / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "分红与资金比较对照日历不同")
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
            print(f"{period}／{cost_id}：分红与资金比较新账户与四个保存对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
        ("era_metrics.csv", eras), ("state_counts.csv", counts), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "CASH_DISTRIBUTION_FUNDING_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8,
        "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8,
        "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier,
        "state_counts": counts, "account_coverage": coverage, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False,
        "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": primary, "较早": [m for m in earlier if m["model"] == PRIMARY], "成交和持仓": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
