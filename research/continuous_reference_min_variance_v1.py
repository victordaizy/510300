"""复用原规则和已成熟模型，连续积累参考历史后评价新资金账户。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.two_policy_min_variance_inputs_v1 import budget_frame
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.panic_learned_equal_blend_v1 import panic_folder, P32
from research.simple_signal_blend_v1 import decision_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_continuous_reference_min_variance_v1"
CONFIG = ROOT / "config/510300_continuous_reference_min_variance_v1.json"
P82 = ROOT / "reports/research/510300_two_policy_min_variance_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
PRIMARY = "CONTINUOUS_REFERENCE_MIN_VARIANCE"
NAME = "连续历史两策略最小方差预算"
CONTROLS = {"TWO_POLICY_MIN_VARIANCE":"原冷启动两策略预算","PANIC_LEARNED_HALF":"原两策略各半","BUY_HOLD":"买入持有"}
from research.learned_cycle_exit_v1 import ExitController
from research.rearmed_cycle_exit_account_v1 import simulate_rearmed_exit
from research.simple_price_entry_exit_v1 import simulate_policy
from research.simple_intraday_protection_v1 import make_rules as learned_rules
from research.simple_volume_reversal_v1 import make_rules as panic_rules


def reference_factors(data, references, first, cfg):
    returns=np.full((len(data),2),np.nan);states=[]
    for column,model in enumerate(["PANIC_ONLY","REARM_RIDGE"]):
        ledger,decisions=references[model]
        saved=ledger[ledger.date.le(data.date.iloc[-1])]
        chosen=decisions[decisions.origin.lt(data.date.iloc[-1])]
        require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(data.date.iloc[first:])),"连续参考账户完整日历缺失")
        require(pd.DatetimeIndex(chosen.origin).equals(pd.DatetimeIndex(data.date.iloc[first-1:-1])),"连续参考判断日期不完整")
        require(pd.DatetimeIndex(chosen.execution_date).equals(pd.DatetimeIndex(data.date.iloc[first:])),"连续参考意向不是下一开盘")
        returns[first:,column]=saved.net_return.to_numpy(float)
        states.append(decision_state(data,chosen))
    f=budget_frame(data.date,returns,np.column_stack(states),first,cfg["risk_window"])
    f["panic_reference_return"],f["learned_reference_return"]=returns[:,0],returns[:,1]
    return f


def freeze():
    require(not CONFIG.exists(),"连续历史风险预算已登记，不能重复冻结")
    old=json.loads((ROOT/"config/510300_two_policy_min_variance_v1.json").read_text(encoding="utf-8"))
    cfg={k:old[k] for k in ["evaluation_start","data_cutoff","initial_capital","lot","tick","limit_fraction","annual_days","cash_annual_rate_assumption","high_sharpe_target","costs","features","dividends","earlier_start","earlier_terminal","weight_band","risk_window"]}
    c31=json.loads((ROOT/"config/510300_learned_cycle_exit_v1.json").read_text(encoding="utf-8"))
    c32=json.loads((ROOT/"config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    c26=json.loads((ROOT/"config/510300_simple_volume_reversal_v1.json").read_text(encoding="utf-8"))
    cfg.update(study_id="510300_CONTINUOUS_REFERENCE_MIN_VARIANCE_V1",round=91,registered_at=now(),primary=PRIMARY,candidate_configurations=1,
        reference_start=c31["reference_start"],panic_features=c26["features"],panic_spec=c26["candidate_specs"]["V6_PANIC_RECOVERY"],learned_spec=c32["specification"],confirmation_days=c32["confirmation_days"],saved_models=c32["saved_models"],
        risk_clock="MONTH_FIRST_COMPLETE_CLOSE_NEXT_OPEN",state_cost="BASE",initial_budgets=[.5,.5],new_model_fits=0,new_reference_accounts=2,
        rules="docs/510300_CONTINUOUS_REFERENCE_MIN_VARIANCE_V1.md",previous_goal_turn_classification="PROGRESS_ROUND90_COMPLETED_AND_DELIVERED",independent_validation="NOT_ESTABLISHED",goal_achieved=False,position_impact=0)
    tests=json.loads((OUT/"tests_receipt.json").read_text(encoding="utf-8"));require(tests["exit_code"]==0,"连续历史必要测试未通过")
    paths=[Path(__file__),ROOT/cfg["rules"],ROOT/"tests/test_continuous_reference_min_variance_v1.py",OUT/"tests_receipt.json",
        ROOT/"config/510300_two_policy_min_variance_v1.json",ROOT/"config/510300_learned_cycle_exit_v1.json",ROOT/"config/510300_rearmed_session_exit_v1.json",ROOT/"config/510300_simple_volume_reversal_v1.json",ROOT/"reports/research/510300_continuous_history_boundary_20260908/result.json"]
    for parent in [old,c32,c26]:
        for item in parent["frozen_files"]:
            require(digest(ROOT/item["path"])==item["sha256"],"原冻结规则或来源改变")
            paths.append(ROOT/item["path"])
    for period in ["evaluation","earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend((P82 if model=="TWO_POLICY_MIN_VARIANCE" else P46)/period/cost/f"{model}_ledger.parquet" for model in CONTROLS)
    cfg["frozen_files"]=[{"path":str(p.relative_to(ROOT)),"sha256":digest(p)} for p in sorted(set(paths))]
    write_json(CONFIG,cfg,exclusive=True)
    print("第91轮连续参考历史单一设置已冻结，尚无新参考账户或评价收益。",flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "风险预算冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    panic_data=pd.read_parquet(ROOT/cfg["panic_features"])
    require(pd.DatetimeIndex(panic_data.date).equals(pd.DatetimeIndex(data.date)),"两参考原行情日历不一致")
    for name in ["open","close","dividend","mom5","mom20","sma120","vol20"]:
        np.testing.assert_allclose(panic_data[name],data[name],rtol=0,atol=1e-12,equal_nan=True)
    stored=json.loads((ROOT/cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    require(all(int(m["fit_index"])<=len(data)-1 and (m["latest_exit_index"] is None or int(m["latest_exit_index"])<=int(m["fit_index"])) for m in stored),"保存学习模型成熟时钟错误")
    controller=ExitController(data,stored,cfg["confirmation_days"])
    learned,learned_decisions,learned_cycles=simulate_rearmed_exit(data,dividends,cfg,cfg["costs"]["BASE"],cfg["reference_start"],learned_rules(data)["D60_INTRA"],cfg["learned_spec"],controller)
    panic,panic_decisions,panic_cycles=simulate_policy(panic_data,dividends,cfg,cfg["costs"]["BASE"],cfg["reference_start"],panic_rules(panic_data)[0]["V6_PANIC_RECOVERY"],cfg["panic_spec"])
    references={"PANIC_ONLY":(panic,panic_decisions),"REARM_RIDGE":(learned,learned_decisions)}
    for model,(ledger,decisions) in references.items():
        require(ledger.accounting_error.abs().max()<1e-6 and not ledger.terminal_unliquidated.iloc[-1],"连续参考账户结算失败")
        save_account(OUT/"continuous_references"/"BASE",model,ledger,decisions)
    learned_cycles.to_csv(OUT/"continuous_references/BASE/REARM_RIDGE_cycles.csv",index=False,encoding="utf-8-sig")
    panic_cycles.to_csv(OUT/"continuous_references/BASE/PANIC_ONLY_cycles.csv",index=False,encoding="utf-8-sig")
    reference_first=int(np.flatnonzero(data.date>=pd.Timestamp(cfg["reference_start"]))[0])
    print("两个连续基础参考账户完成；模型复用，没有新拟合。",flush=True)
    main, earlier, yearly, eras, updates, coverage = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        first = int(np.flatnonzero(frame.date >= pd.Timestamp(start))[0])
        factors=reference_factors(frame,references,reference_first,cfg)
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
                saved = pd.read_parquet((P82 if model == "TWO_POLICY_MIN_VARIANCE" else P46) / period / cost_id / f"{model}_ledger.parquet")
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
            print(f"{period}／{cost_id}：连续历史新账户和三个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras),
        ("risk_update_records.csv", updates), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    write_json(OUT / "result.json", {"study_id": cfg["study_id"], "completed_at": now(), "status": "CONTINUOUS_REFERENCE_MIN_VARIANCE_ACCOUNTS_COMPLETE",
        "candidate_configurations": 1, "evaluation_accounts": 8, "new_accounts_generated": 2, "reused_control_accounts": 6,
        "earlier_diagnostic_accounts": 8, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 6,
        "new_model_fits": 0, "new_reference_accounts": 2, "all_metrics": main, "earlier_diagnostics": earlier, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"), "risk_update_count": len(updates), "account_coverage": coverage,
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    print(json.dumps({"主结果": primary, "较早结果": [m for m in earlier if m["model"]==PRIMARY], "账户覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
