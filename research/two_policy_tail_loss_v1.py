"""冻结两原策略按已实现风险分预算，运行两段两费用真实账户。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.two_policy_tail_loss_inputs_v1 import budget_frame
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.panic_learned_equal_blend_v1 import panic_folder, P32
from research.simple_signal_blend_v1 import decision_state

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_two_policy_tail_loss_v1"
CONFIG = ROOT / "config/510300_two_policy_tail_loss_v1.json"
P82 = ROOT / "reports/research/510300_two_policy_min_variance_v1"
P76 = ROOT / "reports/research/510300_two_policy_risk_budget_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
PRIMARY = "TWO_POLICY_TAIL_LOSS"
NAME = "两条原策略最差百分之五日损失预算"
CONTROLS = {"PANIC_LEARNED_HALF": "原急跌与学习各半", "REARM_RIDGE": "原学习退出及等待新机会", "PANIC_ONLY": "原急跌回升单独策略", "BUY_HOLD": "买入持有", "TWO_POLICY_RISK_BUDGET": "第76轮倒数波动预算", "TWO_POLICY_MIN_VARIANCE": "第82轮最小方差预算"}


def freeze():
    require(not CONFIG.exists(), "两策略风险预算已登记，不能重复冻结")
    old = json.loads((ROOT / "config/510300_panic_learned_equal_blend_v1.json").read_text(encoding="utf-8"))
    for item in old["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "原各半参考规则来源改变")
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]}
    cfg.update(study_id="510300_TWO_POLICY_TAIL_LOSS_V1", round=83, objective="MINIMIZE_TWO_REFERENCE_EMPIRICAL_CVAR", tail_confidence=.95, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
        risk_window=242, risk_clock="MONTH_FIRST_COMPLETE_CLOSE_NEXT_OPEN", state_cost="BASE", initial_budgets=[.5, .5],
        missing_or_zero_risk="NO_VIEW_KEEP_PREVIOUS_BUDGET", optimizer_failure="NO_VIEW_KEEP_PREVIOUS_BUDGET", optimal_tie="PROJECT_PREVIOUS_BUDGET_ON_OPTIMAL_INTERVAL", rules="docs/510300_TWO_POLICY_TAIL_LOSS_V1.md",
        new_model_fits=0, new_reference_accounts=0, previous_goal_turn_classification="PROGRESS_ROUND82_COMPLETE",
        independent_validation="NOT_ESTABLISHED", goal_achieved=False, position_impact=0)
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0, "风险预算必要测试未通过")
    paths = [Path(__file__), ROOT / "research/two_policy_tail_loss_inputs_v1.py", ROOT / "research/two_policy_tail_loss_optimizer_v1.py", ROOT / "research/event_clock_account_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/panic_learned_equal_blend_v1.py",
        ROOT / "research/simple_signal_blend_v1.py", ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"], ROOT / "docs/510300_TWO_POLICY_RISK_BUDGET_V1.md",
        ROOT / "tests/test_two_policy_tail_loss_v1.py", OUT / "tests_receipt.json", ROOT / "config/510300_research_authority_v6.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        paths.extend([P46 / f"{period}_states.parquet", panic_folder(period) / "BASE/V6_PANIC_RECOVERY_decisions.parquet", P32 / period / "BASE/REARM_RIDGE_decisions.parquet"])
        for cost in cfg["costs"]:
            paths.extend((P82 if model == "TWO_POLICY_MIN_VARIANCE" else P76 if model == "TWO_POLICY_RISK_BUDGET" else P46) / period / cost / f"{model}_ledger.parquet" for model in CONTROLS)
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第83轮单项95%尾部损失预算已冻结，尚无新策略收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "风险预算冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    main, earlier, yearly, eras, updates, coverage = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        first = int(np.flatnonzero(frame.date >= pd.Timestamp(start))[0])
        old_states = pd.read_parquet(P46 / f"{period}_states.parquet")
        require(pd.DatetimeIndex(old_states.date).equals(pd.DatetimeIndex(frame.date)), "参考状态日历不同")
        mapped = np.column_stack([decision_state(frame, pd.read_parquet(panic_folder(period) / "BASE/V6_PANIC_RECOVERY_decisions.parquet")),
            decision_state(frame, pd.read_parquet(P32 / period / "BASE/REARM_RIDGE_decisions.parquet"))])
        np.testing.assert_allclose(mapped, old_states[["panic_state", "learned_state"]].to_numpy(), rtol=0, atol=0, equal_nan=True)
        reference_returns = np.full((len(frame), 2), np.nan)
        for column, model in enumerate(["PANIC_ONLY", "REARM_RIDGE"]):
            ref = pd.read_parquet(P46 / period / "BASE" / f"{model}_ledger.parquet")
            require(pd.DatetimeIndex(ref.date).equals(pd.DatetimeIndex(frame.date.iloc[first:])), "参考完整收益日历缺失或提前")
            reference_returns[first:, column] = ref.net_return.to_numpy(float)
        factors = budget_frame(frame.date, reference_returns, mapped, first, cfg["risk_window"], cfg["tail_confidence"])
        factors["panic_reference_return"], factors["learned_reference_return"] = reference_returns[:, 0], reference_returns[:, 1]
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
                saved = pd.read_parquet((P82 if model == "TWO_POLICY_MIN_VARIANCE" else P76 if model == "TWO_POLICY_RISK_BUDGET" else P46) / period / cost_id / f"{model}_ledger.parquet")
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
            print(f"{period}／{cost_id}：风险预算新账户和六个保存对照完成。", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras),
        ("risk_update_records.csv", updates), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    write_json(OUT / "result.json", {"study_id": cfg["study_id"], "completed_at": now(), "status": "TWO_POLICY_TAIL_LOSS_ACCOUNTS_COMPLETE",
        "candidate_configurations": 1, "evaluation_accounts": 14, "new_accounts_generated": 2, "reused_control_accounts": 12,
        "earlier_diagnostic_accounts": 14, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 12,
        "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"), "risk_update_count": len(updates), "account_coverage": coverage,
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    print(json.dumps({"主结果": primary, "较早结果": [m for m in earlier if m["model"]==PRIMARY], "账户覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
