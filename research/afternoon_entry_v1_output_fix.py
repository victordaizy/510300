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
OUT = ROOT / "reports/research/510300_afternoon_entry_v1_output_fix"
ORIGINAL_OUT = ROOT / "reports/research/510300_afternoon_entry_v1"
CONFIG = ROOT / "config/510300_afternoon_entry_v1_output_fix.json"
PARENT = ROOT / "reports/research/510300_rearmed_session_exit_v1"
PRIMARY = "AFTERNOON_ENTRY"
NAME = "原退出不变，增加一次下午进入"
SOURCE_RECEIPT = ROOT / "reports/research/510300_afternoon_entry_source_feasibility_v1/source_feasibility.json"


def compare_saved_decisions(saved_path, original_path):
    current = pd.read_parquet(saved_path)
    original = pd.read_parquet(original_path)
    pd.testing.assert_frame_equal(current[original.columns], original)
    return len(current)


def freeze():
    import shutil
    require(not CONFIG.exists() and not OUT.exists(), "下午进入输出修正已登记")
    original_config = ROOT / "config/510300_afternoon_entry_v1.json"
    cfg = json.loads(original_config.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "原下午进入冻结内容改变")
    tests_path = ROOT / "reports/research/510300_afternoon_entry_saved_comparison_test.json"
    require(json.loads(tests_path.read_text(encoding="utf-8"))["exit_code"] == 0, "保存空值比较测试未通过")
    cfg.update(source_version="OUTPUT_FIX_PRESERVE_THREE_SAVED_ACCOUNTS", correction_registered_at=now(),
        original_config=str(original_config.relative_to(ROOT)), original_config_sha256=digest(original_config),
        correction="内存NaN与保存None比较失败；经济账户已一致。改为比较两份已保存Parquet，保留缺失，不填零、不改策略。",
        prior_actual_accounts_preserved=3, additional_accounts_to_run=1, evaluated_candidate_source_runs=2)
    original_files = [p for p in ORIGINAL_OUT.rglob("*") if p.is_file() and p.name != "RUN_STARTED.json"]
    cfg["preserved_files"] = [{"path": str(p.relative_to(ORIGINAL_OUT)), "sha256": digest(p)} for p in original_files]
    for path in [Path(__file__), original_config, tests_path, ROOT / "tests/test_afternoon_entry_saved_comparison.py"] + original_files:
        cfg["frozen_files"].append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
    OUT.mkdir(parents=True)
    for p in original_files:
        dest = OUT / p.relative_to(ORIGINAL_OUT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    write_json(CONFIG, cfg, exclusive=True)
    print("只修正保存格式比较，前三账户原字节保留，剩一个较早压力账户待算。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "下午进入固定内容改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    div = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    _, snapshots = load_snapshots()
    if not (OUT / "minute_snapshots.parquet").exists():
        snapshots.to_parquet(OUT / "minute_snapshots.parquet")
    snapshot_dict = snapshots.to_dict("index")
    partial_rows = []
    for t, date in enumerate(data.date):
        snapshot = snapshot_dict.get(date)
        if snapshot is not None:
            partial_rows.append({"date": date, **partial_entry(data, t, snapshot["signal_price"])})
    if not (OUT / "all_partial_entry_factors.csv").exists():
        pd.DataFrame(partial_rows).to_csv(OUT / "all_partial_entry_factors.csv", index=False, encoding="utf-8-sig")
    main, early, yearly, eras, stats = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        rule, factors = make_rule(frame, {"kind": "DIFFERENCE", "window": 60, "direction": 1})
        if not (OUT / f"{period}_daily_factors.csv").exists():
            pd.DataFrame({"date": frame.date, "d60_factor": factors, "entry_condition": rule["entry"], "original_price_exit": rule["exit"][1]}).to_csv(OUT / f"{period}_daily_factors.csv", index=False, encoding="utf-8-sig")
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            saved_path = folder / f"{PRIMARY}_ledger.parquet"
            if saved_path.exists():
                ledger = pd.read_parquet(saved_path)
                decisions = pd.read_parquet(folder / f"{PRIMARY}_decisions.parquet")
                cycles = pd.read_csv(folder / f"{PRIMARY}_cycles.csv")
                afternoon = pd.read_parquet(folder / "afternoon_decisions.parquet")
            else:
                require(period == "earlier_diagnostic" and cost_id == "STRESS", "修正轮只能补较早压力账户")
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
                compare_saved_decisions(folder / f"{PRIMARY}_decisions.parquet", PARENT / period / cost_id / "REARM_RIDGE_decisions.parquet")
            stats.append({"period": period, "cost": cost_id, "flat_day_checks": len(afternoon),
                "afternoon_requests": int(afternoon.requested_quantity.gt(0).sum()), "afternoon_fills": int(afternoon.filled_quantity.gt(0).sum()),
                "execution_statuses": afternoon.execution_status.value_counts().to_dict(), "factor_statuses": afternoon.factor_status.value_counts().to_dict(),
                "earlier_is_original_fallback": period == "earlier_diagnostic", "holding_closes": int(ledger.shares.gt(0).sum()), "cycles": len(cycles)})
            accounts, names = {PRIMARY: ledger, "REARM_RIDGE": baseline}, {PRIMARY: NAME, "REARM_RIDGE": "原日终进入和学习退出", "BUY_HOLD": "买入持有"}
            accounts["BUY_HOLD"] = pd.read_parquet(PARENT / period / cost_id / "BUY_HOLD_ledger.parquet")
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                if model != PRIMARY and not (folder / f"{model}_ledger.parquet").exists():
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
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0,
        "preserved_prior_actual_accounts": 3, "new_accounts_in_correction": 1, "evaluated_candidate_source_runs": 2}, exclusive=True)
    lines = ["# 本轮沿用的全部月度模型中文系数", "", "本轮没有新训练；下列模型只在其拟合日收盘后的判断中可用。", ""]
    for item in models:
        lines += [f"## {str(item['fit_time'])}：{'有成熟模型' if item['status']=='FIT_COMPLETE' else '无成熟模型'}", ""]
        if item["status"] == "FIT_COMPLETE":
            lines += chinese_formula(item["model"]) + [""]
        else:
            lines += ["成熟样本不足，保持无观点，原价格与时间退出继续有效。", ""]
    (OUT / "沿用的每月八项模型中文规则.md").write_text("\n".join(lines), encoding="utf-8")
    for item in cfg["preserved_files"]:
        require(digest(OUT / item["path"]) == item["sha256"], "原保存文件在修正过程中被重写")
    print(json.dumps({"主结果": primary, "下午行为": stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
