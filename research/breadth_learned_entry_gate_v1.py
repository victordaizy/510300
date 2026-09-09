"""只用有效广度筛选原学习进入，缺失沿用原基线，完整比较两段两费用。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.breadth_majority_inputs_v1 import build_factors
from research.breadth_learned_gate_inputs_v1 import gate_rule
from research.learned_cycle_exit_v1 import ExitController
from research.simple_intraday_protection_v1 import make_rules
from research.simple_session_divergence_v1 import make_rule, specifications
from research.source_aware_cycle_account_v1 import simulate_source_aware_cycle
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_breadth_learned_entry_gate_v1"
CONFIG = ROOT / "config/510300_breadth_learned_entry_gate_v1.json"
PRIMARY = "BREADTH_GATED_REARM_RIDGE"
NAME = "有效广度筛选原学习进入，缺失沿用原基线"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
CONTROLS = {"REARM_RIDGE": (P32, "原学习退出及等待新机会"),
            "PANIC_LEARNED_HALF": (P46, "原急跌回升及学习退出各半"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "specification", "confirmation_days", "saved_models"]}
    cfg.update(study_id="510300_BREADTH_LEARNED_ENTRY_GATE_V1", round=63, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
        breadth="data/curated/510300_stress_transmission_hazard_v2_g1_historical_remediation_v1/internal_F_T_features.parquet",
        breadth_threshold=.5, coverage_threshold=.98, breadth_role="OPTIONAL_ENTRY_FILTER_ONLY",
        missing_breadth="PRESERVE_NO_VIEW_AND_USE_EXISTING_BASELINE_ENTRY_DECISION", rearm="ONLY_ORIGINAL_D60_ENTRY_CONDITION_DISAPPEARANCE",
        exit="ORIGINAL_D60_AND_RIDGE_PLUS_ORIGINAL_PROTECTIONS_NO_BREADTH_EXIT", new_model_fits=0, new_reference_accounts=0,
        saved_model_key="D60_INTRA__RIDGE", rules="docs/510300_BREADTH_LEARNED_ENTRY_GATE_V1.md",
        source_recomputation_receipt="reports/research/510300_breadth_majority_trend_v1/saved_verification_receipt.json",
        chinese_saved_models="deliverables/510300持仓累积转弱_第60轮_20260907/原学习模型逐月中文规则.md",
        position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED")
    paths = [Path(__file__), ROOT / "research/breadth_learned_gate_inputs_v1.py", ROOT / "research/breadth_majority_inputs_v1.py",
        ROOT / "research/source_aware_cycle_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/rearmed_cycle_exit_account_v1.py",
        ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["breadth"], ROOT / cfg["saved_models"], ROOT / cfg["rules"],
        ROOT / cfg["chinese_saved_models"], ROOT / cfg["source_recomputation_receipt"], ROOT / "tests/test_breadth_learned_entry_gate_v1.py",
        ROOT / "tests/test_cycle_cusum_exit_v1.py", OUT / "tests_receipt.json", ROOT / "config/510300_research_authority_v6.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(p / period / cost / f"{key}_ledger.parquet" for key, (p, _) in CONTROLS.items())
            paths.extend(P32 / period / cost / f"REARM_RIDGE_{kind}" for kind in ["decisions.parquet", "cycles.csv"])
    receipt = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(receipt["exit_code"] == 0, "广度附加进入测试未通过")
    cfg["pre_run_test_representation_correction"] = receipt.get("pre_run_test_representation_correction")
    cfg["previously_saved_baseline_verification_replays"] = 4
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第63轮一个附加进入设置已冻结，旧基线验证完成，尚未计算新策略收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "广度附加进入登记文件改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    breadth = build_factors(data, pd.read_parquet(ROOT / cfg["breadth"]))
    base_rule = make_rules(data)["D60_INTRA"]
    _, trace = gate_rule(base_rule, breadth)
    factors = breadth.merge(trace, on="date", how="left", validate="one_to_one")
    _, d60 = make_rule(data, next(x for x in specifications() if x["id"] == "D60_INTRA"))
    factors["original_d60_factor"] = d60.to_numpy()
    factors["original_d60_exit"] = base_rule["exit"][1]
    factors.to_parquet(OUT / "factors.parquet", index=False)
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"][cfg["saved_model_key"]]
    main, earlier, yearly, eras, coverage, signal_counts = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        local = factors.iloc[:len(frame)]
        rule, _ = gate_rule(make_rules(frame)["D60_INTRA"], local)
        first = int(np.flatnonzero(frame.date >= start)[0])
        origins = local.iloc[first - 1:len(frame) - 1]
        signal_counts.append({"period": period, "decision_origins": len(origins), "breadth_valid_origins": int(origins.breadth_valid.sum()),
            "breadth_missing_fallback_origins": int(origins.breadth_missing_baseline_fallback.sum()),
            "base_eligible_origins": int(origins.base_raw_entry.sum()), "effective_eligible_origins": int(origins.effective_raw_entry.sum()),
            "base_eligible_origins_rejected": int(origins.base_eligible_origin_rejected.sum()),
            "base_eligible_origins_with_missing_breadth": int(origins.base_eligible_origin_with_missing_breadth.sum())})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_source_aware_cycle(frame, dividends, cfg, cost, start, rule, cfg["specification"], ExitController(frame, models, cfg["confirmation_days"]))
            decisions = decisions.merge(local.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            position = ledger.set_index("date").shares.reindex(decisions.origin).fillna(0).to_numpy()
            decisions["flat_after_origin_execution"] = position == 0
            save_account(folder, PRIMARY, ledger, decisions)
            cycle_origins = cycles.entry_index.to_numpy(int) - 1
            cycles["entry_breadth_state"] = local.breadth_gate_state.iloc[cycle_origins].to_numpy()
            cycles["entry_breadth"] = local.breadth20.where(local.breadth_valid).iloc[cycle_origins].to_numpy()
            cycles["entry_used_missing_breadth_baseline_fallback"] = local.breadth_missing_baseline_fallback.iloc[cycle_origins].to_numpy()
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(cycles.exit_date.notna().all() and (cycles.holding_intervals >= 1).all(), "广度筛选账户周期或次日可卖异常")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "广度筛选完整账户异常")
            holding = decisions[decisions.learned_exit_requested.notna()]
            coverage.append({"period": period, "cost": cost_id, "model": PRIMARY, "completed_round_trips": len(cycles),
                "positive_cycles": int(cycles.net_profit_cny.gt(0).sum()), "holding_decision_rows": len(holding),
                "holding_with_prediction": int(holding.learned_prediction.notna().sum()), "holding_without_prediction": int(holding.learned_prediction.isna().sum()),
                "entry_cycles_with_missing_breadth_fallback": int(cycles.entry_used_missing_breadth_baseline_fallback.sum()),
                "learned_exit_cycles": int(cycles.exit_reasons.str.contains("学习条件", regex=False).sum()),
                "flat_rearmed_origins_rejected_by_breadth": int((decisions.flat_after_origin_execution & decisions.entry_rearmed & decisions.base_eligible_origin_rejected).sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for model, (source_folder, name) in CONTROLS.items():
                saved = pd.read_parquet(source_folder / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "广度学习比较缺少相同完整日历")
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
            print(f"{period}／{cost_id}：广度附加进入和三个原样对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
        ("era_metrics.csv", eras), ("entry_exit_coverage.csv", coverage), ("signal_counts.csv", signal_counts)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "BREADTH_LEARNED_ENTRY_GATE_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 6,
        "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6,
        "new_model_fits": 0, "new_reference_accounts": 0, "previously_saved_baseline_verification_replays": 4,
        "all_metrics": main, "earlier_diagnostics": earlier, "entry_exit_coverage": coverage, "signal_counts": signal_counts,
        "primary": primary, "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
        "historical_point_target_met": any(m["meets_point_target"] for m in primary),
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": primary, "较早": [m for m in earlier if m["model"] == PRIMARY], "覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
