"""冻结并运行原退出不变的一次下午进入局部检验。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.afternoon_entry_account_v1 import simulate_afternoon_entry
from research.afternoon_entry_inputs_v1 import EntryPreview, partial_entry
from research.afternoon_learned_exit_v1 import load_snapshots, MINUTE
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import ExitController, chinese_formula
from research.simple_session_divergence_v1 import make_rule

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_afternoon_entry_v1"
CONFIG = ROOT / "config/510300_afternoon_entry_v1.json"
PARENT = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PRIMARY = "AFTERNOON_ENTRY"
NAME = "原退出不变，增加一次下午进入"
SOURCE_RECEIPT = ROOT / "reports/research/510300_afternoon_entry_source_feasibility_v1/source_feasibility.json"


def freeze():
    require(not CONFIG.exists(), "下午进入已登记，不能重复冻结")
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
        "confirmation_days", "specification", "saved_models"]}
    cfg.update(study_id="510300_AFTERNOON_ENTRY_V1", round=75, primary=PRIMARY, registered_at=now(), candidate_configurations=1,
        minute_file=str(MINUTE.relative_to(ROOT)), signal_observation="14:29", signal_decision="14:31", execution_proxy="14:46_OPEN",
        minute_participation_cap=.1, rules="docs/510300_AFTERNOON_ENTRY_V1.md", new_model_fits=0, new_reference_accounts=0,
        earlier_evidence="ORIGINAL_DAILY_FALLBACK_ONLY_NO_MINUTE_VALIDATION", previous_goal_turn_classification="PROGRESS_ROUND74_COMPLETED_AND_DELIVERED",
        independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0)
    receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    require(digest(MINUTE) == receipt["minute_sha256"] and digest(ROOT / cfg["features"]) == receipt["daily_sha256"], "已核分钟或日线来源改变")
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0, "下午进入必要测试尚未通过")
    paths = [Path(__file__), ROOT / "research/afternoon_entry_account_v1.py", ROOT / "research/afternoon_entry_inputs_v1.py",
        ROOT / "research/afternoon_learned_exit_v1.py", ROOT / "research/learned_cycle_exit_v1.py", ROOT / "research/simple_session_divergence_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/rearmed_cycle_exit_account_v1.py",
        ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["saved_models"], ROOT / cfg["rules"], MINUTE, SOURCE_RECEIPT,
        ROOT / "tests/test_afternoon_entry_v1.py", OUT / "tests_receipt.json", ROOT / "config/510300_research_authority_v6.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(PARENT / period / cost / f"{model}_ledger.parquet" for model in ["REARM_RIDGE", "BUY_HOLD"])
            paths.append(PARENT / period / cost / "REARM_RIDGE_decisions.parquet")
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第75轮一个下午进入设置已冻结，尚未读取新策略收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "下午进入固定内容改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    div = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    _, snapshots = load_snapshots()
    snapshots.to_parquet(OUT / "minute_snapshots.parquet")
    snapshot_dict = snapshots.to_dict("index")
    partial_rows = []
    for t, date in enumerate(data.date):
        snapshot = snapshot_dict.get(date)
        if snapshot is not None:
            partial_rows.append({"date": date, **partial_entry(data, t, snapshot["signal_price"])})
    pd.DataFrame(partial_rows).to_csv(OUT / "all_partial_entry_factors.csv", index=False, encoding="utf-8-sig")
    main, early, yearly, eras, stats = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        rule, factors = make_rule(frame, {"kind": "DIFFERENCE", "window": 60, "direction": 1})
        pd.DataFrame({"date": frame.date, "d60_factor": factors, "entry_condition": rule["entry"], "original_price_exit": rule["exit"][1]}).to_csv(OUT / f"{period}_daily_factors.csv", index=False, encoding="utf-8-sig")
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles, afternoon = simulate_afternoon_entry(frame, div, cfg, cost, start, rule, cfg["specification"],
                ExitController(frame, models, cfg["confirmation_days"]), EntryPreview(frame), snapshot_dict)
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            afternoon.to_parquet(folder / "afternoon_decisions.parquet", index=False)
            afternoon.to_csv(folder / "afternoon_decisions.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "下午进入账户结算失败")
            require(not ((ledger.filled_quantity > 0) & (ledger.shares_before > 0)).any(), "下午进入账户发生追加持仓")
            require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "下午进入违反T+1")
            baseline = pd.read_parquet(PARENT / period / cost_id / "REARM_RIDGE_ledger.parquet")
            if period == "earlier_diagnostic":
                require(not snapshots.index.to_series().between(pd.Timestamp(start), pd.Timestamp(cfg["earlier_terminal"])).any(), "较早历史出现了未预定分钟数据")
                pd.testing.assert_frame_equal(ledger[baseline.columns], baseline)
                original_decisions = pd.read_parquet(PARENT / period / cost_id / "REARM_RIDGE_decisions.parquet")
                pd.testing.assert_frame_equal(decisions[original_decisions.columns], original_decisions)
            stats.append({"period": period, "cost": cost_id, "flat_day_checks": len(afternoon),
                "afternoon_requests": int(afternoon.requested_quantity.gt(0).sum()), "afternoon_fills": int(afternoon.filled_quantity.gt(0).sum()),
                "execution_statuses": afternoon.execution_status.value_counts().to_dict(), "factor_statuses": afternoon.factor_status.value_counts().to_dict(),
                "earlier_is_original_fallback": period == "earlier_diagnostic", "holding_closes": int(ledger.shares.gt(0).sum()), "cycles": len(cycles)})
            accounts, names = {PRIMARY: ledger, "REARM_RIDGE": baseline}, {PRIMARY: NAME, "REARM_RIDGE": "原日终进入和学习退出", "BUY_HOLD": "买入持有"}
            accounts["BUY_HOLD"] = pd.read_parquet(PARENT / period / cost_id / "BUY_HOLD_ledger.parquet")
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                if model != PRIMARY:
                    saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(baseline.date)), "下午进入账户日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：下午进入账户、原策略和买入持有完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("afternoon_statistics.csv", stats)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    write_json(OUT / "result.json", {"study_id": cfg["study_id"], "completed_at": now(), "status": "AFTERNOON_ENTRY_ACCOUNTS_COMPLETE",
        "candidate_configurations": 1, "evaluation_accounts": 6, "new_accounts_generated": 2, "reused_control_accounts": 4,
        "earlier_diagnostic_accounts": 6, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 4, "new_model_fits": 0, "new_reference_accounts": 0,
        "all_metrics": main, "earlier_diagnostics": early, "afternoon_statistics": stats, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"), "historical_point_target_met": any(m["meets_point_target"] for m in primary),
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    lines = ["# 本轮沿用的全部月度模型中文系数", "", "本轮没有新训练；下列模型只在其拟合日收盘后的判断中可用。", ""]
    for item in models:
        lines += [f"## {str(item['fit_time'])}：{'有成熟模型' if item['status']=='FIT_COMPLETE' else '无成熟模型'}", ""]
        if item["status"] == "FIT_COMPLETE":
            lines += chinese_formula(item["model"]) + [""]
        else:
            lines += ["成熟样本不足，保持无观点，原价格与时间退出继续有效。", ""]
    (OUT / "沿用的每月八项模型中文规则.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"主结果": primary, "下午行为": stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
