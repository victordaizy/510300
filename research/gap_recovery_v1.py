"""冻结一个低开收复和日内转弱事件规则，复用完整账户。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.gap_recovery_inputs_v1 import recovery_frame
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_gap_recovery_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
CONFIG = ROOT / "config/510300_gap_recovery_v1.json"
PRIMARY = "GAP_RECOVERY"
NAME = "低开收复进入与日内转弱退出"
DIVIDEND_RECEIPT = "reports/research/510300_adaptive_allocation_v1/frozen_inputs/dividend_coverage.json"
CONTROLS = {"REARM_RIDGE": "原学习退出及等待新机会", "PANIC_LEARNED_HALF": "原急跌反弹与学习各半",
    "PANIC_ONLY": "原急跌反弹单独策略", "BUY_HOLD": "买入持有"}


def freeze():
    require(not CONFIG.exists(), "本轮已经冻结，不能重复登记")
    old = json.loads((ROOT / "config/510300_cash_distribution_funding_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
        "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]}
    tests = json.loads((OUT / "tests_receipt.json").read_text(encoding="utf-8"))
    require(tests["exit_code"] == 0, "关键边界和成交测试尚未通过")
    coverage = json.loads((ROOT / DIVIDEND_RECEIPT).read_text(encoding="utf-8"))
    require(coverage["complete_history_confirmed"] and coverage["distribution_file_sha256"] == digest(ROOT / cfg["dividends"]), "分红完整覆盖和事件账本身份不符")
    data = pd.read_parquet(ROOT / cfg["features"])
    require(pd.Timestamp(coverage["coverage_start"]) <= data.date.iloc[0] and pd.Timestamp(coverage["coverage_end"]) >= data.date.iloc[-1], "分红覆盖不足行情区间")
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    event_cash = data.date.map(dividends.groupby("ex_date").cash_dividend_per_share.sum()).fillna(0.).to_numpy()
    dividend_error = float(np.max(np.abs(data.dividend.to_numpy() - event_cash)))
    require(dividend_error < 1e-12, "信号每日分红与账户事件分红不一致")
    error = float(np.max(np.abs(data.previous_close.iloc[1:].to_numpy() - data.close.iloc[:-1].to_numpy())))
    require(error < 1e-12, "前收盘不是上一交易日未复权收盘")
    quote_errors = {}
    for name in ["open", "close", "previous_close", "dividend"]:
        values = data[name].dropna().to_numpy(float) / cfg["tick"]
        quote_errors[name] = float(np.max(np.abs(values - np.rint(values))))
        require(np.isfinite(values).all() and quote_errors[name] < 1e-7, "实际输入不能按声明单位表示")
    write_json(OUT / "source_receipt.json", {"checked_at": now(), "status": "EXISTING_PRICE_DIVIDEND_QUOTE_UNITS_CONFIRMED",
        "rows": len(data), "first": str(data.date.iloc[0]), "last": str(data.date.iloc[-1]),
        "missing_values": data[["open", "close", "previous_close", "dividend"]].isna().sum().to_dict(),
        "maximum_previous_close_error": error, "maximum_quote_unit_errors": quote_errors, "maximum_dividend_event_error": dividend_error,
        "new_market_data_downloaded": False, "new_strategy_returns_read": False, "source_budget_cny": 0}, exclusive=True)
    cfg.update(study_id="510300_GAP_RECOVERY_V1", round=69, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
        entry="OPEN_PLUS_EX_DIVIDEND_BELOW_PREVIOUS_CLOSE_AND_CLOSE_PLUS_EX_DIVIDEND_ABOVE_PREVIOUS_CLOSE",
        exit="CLOSE_AT_OR_BELOW_OPEN", initialization="FIRST_COMPLETE_ORIGIN_ENTRY_IF_EVENT_ELSE_ZERO",
        equality="NO_NEW_ENTRY_AT_PREVIOUS_CLOSE_AND_EXIT_AT_CURRENT_OPEN", integer_quote_units=True,
        missing="NO_VIEW_KEEP_INTENT_AND_ACTUAL_SHARES", signal_update="EVERY_CLOSE_NEXT_OPEN_ACTUAL_ACCOUNT",
        new_model_fits=0, new_reference_accounts=0, source_budget_cny=0, rules="docs/510300_GAP_RECOVERY_V1.md",
        previous_goal_turn_classification="PROGRESS_ROUNDS67_68_COMPLETED", position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED")
    paths = [Path(__file__), ROOT / "research/gap_recovery_inputs_v1.py", ROOT / "research/event_clock_account_v1.py",
        ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py", ROOT / cfg["features"], ROOT / cfg["dividends"],
        ROOT / DIVIDEND_RECEIPT, ROOT / cfg["rules"], ROOT / "tests/test_gap_recovery_v1.py", OUT / "tests_receipt.json", OUT / "source_receipt.json",
        ROOT / "config/510300_research_authority_v6.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(P46 / period / cost / f"{model}_ledger.parquet" for model in CONTROLS)
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第69轮一个低开收复进入、日内转弱退出设置已冻结，尚无新策略收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "低开收复冻结来源改变")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    factors = recovery_frame(data, cfg["tick"])
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
            require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "低开收复完整账户结算失败")
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
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "低开收复对照日历不同")
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
            print(f"{period}／{cost_id}：低开收复新账户与四个保存对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
        ("era_metrics.csv", eras), ("state_counts.csv", counts), ("account_coverage.csv", coverage)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "GAP_RECOVERY_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
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
