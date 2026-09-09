"""只用已完成周期的最大本金损失更新预算，原参考和模型直接复用。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.cycle_adverse_risk_inputs_v1 import cycle_risk_budget_frame
from research.adaptive_allocation_v1 import normalize_dividends,save_account,summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest,now,require,write_json
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/research/510300_cycle_adverse_risk_v1"
CONFIG=ROOT/"config/510300_cycle_adverse_risk_v1.json"
P91=ROOT/"reports/research/510300_continuous_reference_min_variance_v1"
P82=ROOT/"reports/research/510300_two_policy_min_variance_v1"
P46=ROOT/"reports/research/510300_panic_learned_equal_blend_v1"
PRIMARY="CYCLE_ADVERSE_RISK"
NAME="已完成周期最大本金损失预算"
CONTROLS={"CONTINUOUS_REFERENCE_MIN_VARIANCE":"原连续历史日风险预算","TWO_POLICY_MIN_VARIANCE":"原冷启动日风险预算","BUY_HOLD":"买入持有"}


def freeze():
    require(not CONFIG.exists(),"周期本金风险设置已登记，不能重复冻结")
    old=json.loads((ROOT/"config/510300_continuous_reference_min_variance_v1.json").read_text(encoding="utf-8"))
    cfg={k:old[k] for k in ["evaluation_start","data_cutoff","initial_capital","lot","tick","limit_fraction","annual_days","cash_annual_rate_assumption","high_sharpe_target","costs","features","dividends","earlier_start","earlier_terminal","weight_band","reference_start"]}
    cfg.update(study_id="510300_CYCLE_ADVERSE_RISK_V1",round=92,registered_at=now(),primary=PRIMARY,candidate_configurations=1,
        prior_loss_fractions={"PANIC_ONLY":old["panic_spec"]["modes"]["1"]["loss"],"REARM_RIDGE":old["learned_spec"]["modes"]["1"]["loss"]},
        prior_method="ONE_ORIGINAL_STOP_PSEUDO_CYCLE",cycle_memory="ALL_NATURAL_COMPLETED_MATURE_CYCLES_SINCE_ORIGINAL_REFERENCE_START",
        cycle_risks_source="reports/research/510300_cycle_risk_input_diagnostic_20260908/complete_reference_cycle_risks.parquet",
        rules="docs/510300_CYCLE_ADVERSE_RISK_V1.md",risk_clock="MONTH_FIRST_COMPLETE_CLOSE_NEXT_OPEN",state_cost="BASE",new_model_fits=0,new_reference_accounts=0,
        previous_goal_turn_classification="PROGRESS_ROUNDS90_91_COMPLETED_AND_DELIVERED",independent_validation="NOT_ESTABLISHED",goal_achieved=False,position_impact=0)
    tests=json.loads((OUT/"tests_receipt.json").read_text(encoding="utf-8"));require(tests["exit_code"]==0,"周期风险必要测试未通过")
    receipt=ROOT/"reports/research/510300_cycle_risk_input_diagnostic_20260908/result.json"
    require(json.loads(receipt.read_text(encoding="utf-8"))["cycle_inputs_sha256"]==digest(ROOT/cfg["cycle_risks_source"]),"周期风险输入与保存诊断不符")
    paths=[Path(__file__),ROOT/"research/cycle_adverse_risk_inputs_v1.py",ROOT/"tests/test_cycle_adverse_risk_v1.py",ROOT/cfg["rules"],OUT/"tests_receipt.json",ROOT/cfg["cycle_risks_source"],receipt,
        ROOT/"scripts/review_round74_saved.py",ROOT/"config/510300_continuous_reference_min_variance_v1.json",ROOT/"docs/510300_ACTIVE_RISK_BUDGET_V1.md",ROOT/"docs/510300_TWO_POLICY_TAIL_LOSS_V1.md",ROOT/"docs/510300_TWO_POLICY_WEALTH_BUDGET_V1.md"]
    for item in old["frozen_files"]:
        require(digest(ROOT/item["path"])==item["sha256"],"原连续历史冻结来源改变")
        paths.append(ROOT/item["path"])
    for model in ["PANIC_ONLY","REARM_RIDGE"]:
        for kind in ["ledger","decisions"]:paths.append(P91/"continuous_references/BASE"/f"{model}_{kind}.parquet")
    for period in ["evaluation","earlier_diagnostic"]:
        paths.append(P91/f"{period}_factors.parquet")
        for cost in cfg["costs"]:
            paths.extend((P91 if model=="CONTINUOUS_REFERENCE_MIN_VARIANCE" else P82 if model=="TWO_POLICY_MIN_VARIANCE" else P46)/period/cost/f"{model}_ledger.parquet" for model in CONTROLS)
    cfg["frozen_files"]=[{"path":str(p.relative_to(ROOT)),"sha256":digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG,cfg,exclusive=True)
    print("第92轮单项周期本金风险预算已冻结，尚无新策略收益。",flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "风险预算冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    cycles=pd.read_parquet(ROOT/cfg["cycle_risks_source"])
    reference_first=int(np.flatnonzero(data.date.ge(cfg["reference_start"]))[0])
    main, earlier, yearly, eras, updates, coverage = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        first = int(np.flatnonzero(frame.date >= pd.Timestamp(start))[0])
        parent=pd.read_parquet(P91/f"{period}_factors.parquet")
        require(pd.DatetimeIndex(parent.date).equals(pd.DatetimeIndex(frame.date)),"原连续参考意向日历不符")
        factors=cycle_risk_budget_frame(frame.date,cycles,parent[["panic_state","learned_state"]].to_numpy(),reference_first,cfg)
        require(np.isfinite(factors.target.iloc[first-1:-1]).all(),"新评价期间连续参考意向缺失")
        factors.to_parquet(OUT / f"{period}_factors.parquet", index=False)
        factors.to_csv(OUT / f"{period}_factors.csv", index=False, encoding="utf-8-sig")
        for row in factors[factors.risk_update_scheduled].to_dict("records"):
            updates.append({"period": period, **row})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, PRIMARY,
                targets=factors.target.to_numpy(float), event_mask=np.ones(len(frame), bool))
            decisions = decisions.merge(factors.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "风险预算完整账户结算失败")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()),
                "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                "no_view_target_origins": int(decisions.reference_weight.isna().sum()), "mean_exposure": float(ledger.exposure.mean())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for model, name in CONTROLS.items():
                saved = pd.read_parquet((P91 if model == "CONTINUOUS_REFERENCE_MIN_VARIANCE" else P82 if model == "TWO_POLICY_MIN_VARIANCE" else P46) / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "新旧账户日历不一致")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        eras.append({"cost": cost_id, "model": model, "era": label, **summarize(saved[saved.date.between(left, right)], cfg)})
            print(f"{period}／{cost_id}：周期风险新账户和三个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras),
        ("risk_update_records.csv", updates), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    write_json(OUT / "result.json", {"study_id": cfg["study_id"], "completed_at": now(), "status": "CYCLE_ADVERSE_RISK_ACCOUNTS_COMPLETE",
        "candidate_configurations": 1, "evaluation_accounts": 8, "new_accounts_generated": 2, "reused_control_accounts": 6,
        "earlier_diagnostic_accounts": 8, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6,
        "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"), "risk_update_count": len(updates), "account_coverage": coverage,
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    print(json.dumps({"主结果": primary, "较早结果": [m for m in earlier if m["model"]==PRIMARY], "账户覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
