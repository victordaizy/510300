"""直接比较多数成分上涨与价格趋势、删除广度的同规则对照。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.breadth_majority_inputs_v1 import build_factors, rule_from_factors, MajorityExitController
from research.source_aware_cycle_account_v1 import simulate_source_aware_cycle
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_breadth_majority_trend_v1"
CONFIG = ROOT / "config/510300_breadth_majority_trend_v1.json"
PRIMARY = "BREADTH_MAJORITY_TREND"
VARIANTS = {PRIMARY: (True, "多数成分上涨与二十日价格趋势"), "PRICE20_ONLY": (False, "删除广度的二十日价格对照")}
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
CONTROLS = {"REARM_RIDGE": (P32, "原学习退出及等待新机会"),
            "PANIC_LEARNED_HALF": (P46, "原急跌回升及学习退出各半"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "specification"]}
    cfg.update(study_id="510300_BREADTH_MAJORITY_TREND_V1", round=62, registered_at=now(), primary=PRIMARY, candidate_configurations=2,
        new_primary_settings=1, required_price_only_ablations=1, lookback=20, majority_threshold=.5, required_member_coverage=.98,
        breadth="data/curated/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1/internal_F_T_features.parquet",
        source_scope="INDEPENDENT_RAW_BREADTH20_WITH_OWN_98_PERCENT_AND_DAILY_FOUR_STATE_GATE_NOT_F_T_COMPOSITE",
        entry="KNOWN_BREADTH_STRICTLY_ABOVE_HALF_AND_WEALTH_STRICTLY_ABOVE_CURRENT_20_CLOSE_MEAN",
        extra_exit="EITHER_VALID_BREADTH_AT_OR_BELOW_HALF_OR_VALID_WEALTH_AT_OR_BELOW_20_CLOSE_MEAN",
        missing="NO_NEW_ENTRY_NO_FALSE_REARM_KEEP_INDEPENDENT_KNOWN_EXITS_AND_PENDING_EXIT",
        new_model_fits=0, new_reference_accounts=0, rules="docs/510300_BREADTH_MAJORITY_TREND_V1.md",
        position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED")
    paths = [Path(__file__), ROOT / "research/breadth_majority_inputs_v1.py", ROOT / "research/source_aware_cycle_account_v1.py",
        ROOT / "research/additional_cycle_exit_account_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["breadth"], ROOT / cfg["rules"],
        ROOT / "tests/test_breadth_majority_trend_v1.py", OUT / "tests_receipt.json", ROOT / "config/510300_research_authority_v6.json",
        ROOT / "research/stress_transmission_hazard_v2.py", ROOT / "research/stress_transmission_hazard_v2_mft_features_v1.py",
        ROOT / "config/510300_csi300_pit_membership_weights_source_remediation_v1_0_1_clock_correction.json",
        ROOT / "config/510300_csi300_pit_membership_weights_source_remediation_v1_0_2_2015_extension_clock_addendum.json",
        ROOT / "data/curated/510300_csi300_pit_membership_weights_source_remediation_v1/000300_daily_pit_membership_20150101_20260814.parquet",
        ROOT / "reports/research/510300_stress_transmission_hazard_v2_g1_historical_remediation_status_v1.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(p / period / cost / f"{key}_ledger.parquet" for key, (p, _) in CONTROLS.items())
    receipt = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(receipt["exit_code"] == 0, "广度与来源状态测试尚未通过")
    cfg["pre_run_test_fixture_correction"] = receipt.get("pre_run_test_fixture_correction")
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第62轮主方案及删除广度的必要对照已登记，尚未读取本轮账户收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "广度策略登记文件发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    source = pd.read_parquet(ROOT / cfg["breadth"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    factors = build_factors(data, source)
    factors.to_parquet(OUT / "factors.parquet", index=False)
    main, earlier, yearly, eras, coverage, signal_counts = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        local = factors.iloc[:len(frame)]
        first = int(np.flatnonzero(frame.date >= start)[0])
        origins = local.iloc[first - 1:len(frame) - 1]
        for model, (use_breadth, name) in VARIANTS.items():
            prefix = "combined" if use_breadth else "price"
            signal_counts.append({"period": period, "model": model, "decision_origins": len(origins),
                "breadth_valid_origins": int(origins.breadth_valid.sum()), "entry_no_view_origins": int((~origins[f"{prefix}_entry_view"]).sum()),
                "eligible_entry_origins": int(origins[f"{prefix}_entry"].sum())})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            accounts, names = {}, {}
            for model, (use_breadth, name) in VARIANTS.items():
                rule = rule_from_factors(local, use_breadth)
                ledger, decisions, cycles = simulate_source_aware_cycle(frame, dividends, cfg, cost, start, rule, cfg["specification"], MajorityExitController(local, use_breadth))
                decisions = decisions.merge(local.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
                save_account(folder, model, ledger, decisions)
                cycles.to_csv(folder / f"{model}_cycles.csv", index=False, encoding="utf-8-sig")
                require(len(cycles) > 0 and cycles.exit_date.notna().all(), "广度账户无周期或终点未完成")
                require((cycles.holding_intervals >= 1).all() and ledger.accounting_error.abs().max() < 1e-6, "广度账户财富或次日可卖约束不成立")
                require(not ledger.terminal_unliquidated.iloc[-1], "广度账户终点未清算")
                holding = decisions[decisions.additional_exit_requested.notna()]
                coverage.append({"period": period, "cost": cost_id, "model": model, "completed_round_trips": len(cycles),
                    "positive_cycles": int(cycles.net_profit_cny.gt(0).sum()), "holding_decision_rows": len(holding),
                    "holding_breadth_no_view_rows": int((~holding.breadth_valid).sum()),
                    "breadth_exit_cycles": int(cycles.exit_reasons.str.contains("上涨比例", regex=False).sum()),
                    "price_exit_cycles": int(cycles.exit_reasons.str.contains("二十日平均", regex=False).sum()),
                    "mean_holding_intervals": float(cycles.holding_intervals.mean())})
                accounts[model], names[model] = ledger, name
            for model, (source_folder, name) in CONTROLS.items():
                saved = pd.read_parquet(source_folder / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "广度比较账户未保留相同完整日历")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            pd.DataFrame({"date": accounts[PRIMARY].date, **{k: a.net_return.to_numpy() for k, a in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：广度主方案、删除广度对照和三个原样对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
        ("era_metrics.csv", eras), ("entry_exit_coverage.csv", coverage), ("signal_counts.csv", signal_counts)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "BREADTH_MAJORITY_TREND_ACCOUNTS_COMPLETE", "candidate_configurations": 2,
        "evaluation_accounts": len(main), "new_accounts_generated": 4, "reused_control_accounts": 6,
        "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 4, "reused_earlier_accounts": 6,
        "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier,
        "entry_exit_coverage": coverage, "signal_counts": signal_counts, "primary": primary,
        "post_selected_best_base": max((m for m in main if m["cost"] == "BASE" and m["model"] in VARIANTS), key=lambda m: m["net_sharpe"]),
        "historical_point_target_met": any(m["meets_point_target"] for m in primary),
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": [m for m in main if m["model"] in VARIANTS], "较早历史": [m for m in earlier if m["model"] in VARIANTS], "覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
