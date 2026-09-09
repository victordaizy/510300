"""按固定短长波动选择已有急跌或学习专家并运行完整账户。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.volatility_expert_router_inputs_v1 import routing_frame
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_volatility_expert_router_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
CONFIG = ROOT / "config/510300_volatility_expert_router_v1.json"
PRIMARY = "VOLATILITY_EXPERT_ROUTER"
NAME = "短期高波动选择急跌，其余选择原学习"
CONTROLS = {"PANIC_LEARNED_HALF": "原急跌回升与学习各半", "REARM_RIDGE": "原学习退出及等待新机会",
            "PANIC_ONLY": "原急跌回升单独策略", "BUY_HOLD": "买入持有"}


def freeze():
    old = json.loads((ROOT / "config/510300_panic_learned_equal_blend_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band", "state_cost"]}
    cfg.update(study_id="510300_VOLATILITY_EXPERT_ROUTER_V1", round=65, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
        target="PANIC_IF_VOL5_GREATER_THAN_1_5_VOL60_ELSE_LEARNED_WHEN_KNOWN", volatility_multiplier=1.5, source_folder=str(P46.relative_to(ROOT)),
        signal_update="EVERY_CLOSE_NEXT_OPEN_RECOMPUTE_FROM_CURRENT_ACTUAL_ACCOUNT", new_model_fits=0, new_reference_accounts=0,
        rules="docs/510300_VOLATILITY_EXPERT_ROUTER_V1.md", chinese_saved_models="deliverables/510300广度筛选学习进入_第63轮_20260907/原学习模型逐月中文规则.md",
        position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED")
    paths = [Path(__file__), ROOT / "research/volatility_expert_router_inputs_v1.py", ROOT / "research/event_clock_account_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
        ROOT / "research/simple_signal_blend_v1.py", ROOT / "research/panic_learned_equal_blend_v1.py", ROOT / cfg["features"], ROOT / cfg["dividends"],
        ROOT / cfg["rules"], ROOT / cfg["chinese_saved_models"], ROOT / "tests/test_volatility_expert_router_v1.py", OUT / "tests_receipt.json",
        ROOT / "config/510300_research_authority_v6.json", ROOT / "config/510300_panic_learned_equal_blend_v1.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        paths.append(P46 / f"{period}_states.parquet")
        for cost in cfg["costs"]:
            paths.extend(P46 / period / cost / f"{key}_ledger.parquet" for key in CONTROLS)
            paths.append(P46 / period / cost / "PANIC_LEARNED_HALF_decisions.parquet")
    receipt = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(receipt["exit_code"] == 0, "固定波动选择与实际账户测试尚未通过")
    cfg["previously_saved_half_allocation_verification_replays"] = 0
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第65轮一个固定波动状态选择设置已冻结，尚未计算新策略收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "固定波动选择冻结文件变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    main, earlier, yearly, eras, counts, coverage = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        states = pd.read_parquet(P46 / f"{period}_states.parquet")
        require(pd.DatetimeIndex(states.date).equals(pd.DatetimeIndex(frame.date)), "专家状态与完整市场日历不一致")
        states = routing_frame(frame, states, cfg["volatility_multiplier"])
        target = states.routed_target.to_numpy(float)
        anchor = int(np.flatnonzero(frame.date >= pd.Timestamp(start))[0]) - 1
        require(np.isfinite(target[anchor:-1]).all(), "实际评价原点专家状态缺失，保留无观点且停止运行")
        states.to_parquet(OUT / f"{period}_states.parquet", index=False)
        origins = states.iloc[anchor:-1]
        for (regime, selected, value), group in origins.groupby(["regime", "selected_expert", "routed_target"], dropna=False):
            counts.append({"period": period, "regime": regime, "selected_expert": selected,
                "routed_target": value, "decision_origins": len(group)})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, PRIMARY, targets=target, event_mask=np.ones(len(frame), bool))
            decisions = decisions.merge(states.rename(columns={"date": "origin"}), on="origin", how="left", validate="one_to_one")
            save_account(folder, PRIMARY, ledger, decisions)
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "固定波动选择结算不符")
            coverage.append({"period": period, "cost": cost_id, "buy_trades": int(ledger.filled_quantity.gt(0).sum()), "sell_trades": int(ledger.filled_quantity.lt(0).sum()),
                "holding_closes": int(ledger.shares.gt(0).sum()), "mean_exposure": float(ledger.exposure.mean()),
                "unfilled_requests": int((ledger.requested_quantity.ne(0) & ledger.filled_quantity.eq(0)).sum()),
                "zero_target_origins": int(decisions.reference_weight.eq(0).sum()), "full_target_origins": int(decisions.reference_weight.eq(1).sum()),
                "no_view_origins": int(decisions.reference_weight.isna().sum())})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for model, name in CONTROLS.items():
                saved = pd.read_parquet(P46 / period / cost_id / f"{model}_ledger.parquet")
                saved.to_parquet(folder / f"{model}_ledger.parquet", index=False)
                accounts[model], names[model] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for model, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "固定波动选择对照没有保留同一完整日历")
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
            print(f"{period}／{cost_id}：固定波动选择和四个原样对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
        ("era_metrics.csv", eras), ("state_counts.csv", counts), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "VOLATILITY_EXPERT_ROUTER_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
        "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8,
        "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8,
        "new_model_fits": 0, "new_reference_accounts": 0, "previously_saved_half_allocation_verification_replays": 0,
        "all_metrics": main, "earlier_diagnostics": earlier, "state_counts": counts, "account_coverage": coverage,
        "primary": primary, "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
        "historical_point_target_met": any(m["meets_point_target"] for m in primary),
        "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": primary, "较早": [m for m in earlier if m["model"] == PRIMARY], "成交及仓位": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
