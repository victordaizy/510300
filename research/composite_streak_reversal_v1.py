"""冻结三项短期反转规则，直接运行完整进出场账户。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.composite_streak_reversal_inputs_v1 import factor_frame, trading_rule
from research.simple_price_entry_exit_v1 import simulate_policy
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_composite_streak_reversal_v1"
CONFIG = ROOT / "config/510300_composite_streak_reversal_v1.json"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
P76 = ROOT / "reports/research/510300_two_policy_risk_budget_v1"
PRIMARY = "COMPOSITE_STREAK_REVERSAL"
NAME = "三项短期反转与独立退出"
CONTROLS = {"TWO_POLICY_RISK_BUDGET": "第76轮风险预算", "PANIC_LEARNED_HALF": "第46轮急跌与学习各半",
    "REARM_RIDGE": "第32轮原学习退出", "BUY_HOLD": "买入持有"}


def control_path(period, cost, model):
    return (P76 if model == "TWO_POLICY_RISK_BUDGET" else P46) / period / cost / f"{model}_ledger.parquet"


def freeze():
    require(not CONFIG.exists(), "三项短期反转已经冻结，不能覆盖")
    old = json.loads((ROOT / "config/510300_panic_learned_equal_blend_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    cfg.update(study_id="510300_COMPOSITE_STREAK_REVERSAL_V1", round=78, registered_at=now(), primary=PRIMARY,
        candidate_configurations=1, price_rsi_period=3, streak_rsi_period=2, prior_rank_window=100, trend_window=200,
        entry_score_strictly_below=10, exit_score_strictly_above=70,
        policy_spec={"cooldown": 1, "modes": {1: {"loss": .04, "trail": None, "take": None, "days": 5}}},
        rules="docs/510300_COMPOSITE_STREAK_REVERSAL_V1.md", new_model_fits=0, new_reference_accounts=0,
        previous_goal_turn_classification="PROGRESS_ROUND76_AND_77_COMPLETED_AND_DELIVERED",
        evidence_class="RESEARCH_SCREEN_ON_PREVIOUSLY_OBSERVED_HISTORY", independent_validation="NOT_ESTABLISHED",
        goal_achieved=False, position_impact=0)
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0, "三项短期反转必要测试未通过")
    paths = [Path(__file__), ROOT / "research/composite_streak_reversal_inputs_v1.py", ROOT / "research/simple_price_entry_exit_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / cfg["features"],
        ROOT / cfg["dividends"], ROOT / cfg["rules"], ROOT / "tests/test_composite_streak_reversal_v1.py", OUT / "tests_receipt.json",
        ROOT / "config/510300_research_authority_v6.json", ROOT / "reports/research/510300_adaptive_allocation_v1/frozen_inputs/dividend_coverage.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(control_path(period, cost, model) for model in CONTROLS)
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第78轮三项反转唯一设置已冻结，尚未运行新账户。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "三项反转冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    require(np.isfinite(data[["open", "close", "dividend", "wealth"]].to_numpy(float)).all(), "实际账户原始日线缺失，不可删日或填补")
    require(np.isfinite(data.loc[data.date >= cfg["earlier_start"], "previous_close"]).all(), "实际账户期前收盘缺失，不可填补")
    main, earlier, yearly, coverage = [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        f = factor_frame(frame)
        rule = trading_rule(f)
        f["entry_condition"], f["price_exit_condition"] = rule["entry"], rule["exit"][1]
        f.to_parquet(OUT / f"{period}_factors.parquet", index=False)
        f.to_csv(OUT / f"{period}_factors.csv", index=False, encoding="utf-8-sig")
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions, cycles = simulate_policy(frame, dividends, cfg, cost, start, rule, cfg["policy_spec"])
            decisions = decisions.merge(f.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "三项反转账户结算失败")
            coverage.append({"period": period, "cost": cost_id, "holding_closes": int(ledger.shares.gt(0).sum()),
                "completed_cycles": len(cycles), "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                "no_view_origins": int((~decisions.factor_valid).sum()), "mean_exposure": float(ledger.exposure.mean())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for model, name in CONTROLS.items():
                saved = pd.read_parquet(control_path(period, cost_id, model))
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(ledger.date)), "三项反转与对照账户日历不同")
                m = {"cost": cost_id, "model": model, "name": names[model], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                if model == PRIMARY:
                    for year, group in saved.groupby(saved.date.dt.year):
                        yearly.append({"period": period, "cost": cost_id, "model": model, "year": int(year), **summarize(group, cfg)})
            print(f"{period}／{cost_id}新三项反转净夏普：{summarize(ledger, cfg)['net_sharpe']}", flush=True)
    for name, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / name, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    write_json(OUT / "result.json", {"study_id": cfg["study_id"], "completed_at": now(), "status": "COMPOSITE_STREAK_REVERSAL_ACCOUNTS_COMPLETE",
        "candidate_configurations": 1, "evaluation_accounts": 10, "new_accounts_generated": 2, "reused_control_accounts": 8,
        "earlier_diagnostic_accounts": 10, "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8,
        "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier, "primary": primary,
        "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"), "account_coverage": coverage,
        "historical_point_target_met": any(m["meets_point_target"] for m in primary), "goal_achieved": False,
        "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}, exclusive=True)
    print(json.dumps({"本轮完成": cfg["study_id"], "新账户": 4, "保存对照": 16, "覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
