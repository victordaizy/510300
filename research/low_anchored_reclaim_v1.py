"""阶段新低后重新站上固定起点量价均值，直接检验完整进出场账户。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.additional_cycle_exit_account_v1 import simulate_additional_exit
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.low_anchored_price_v1 import build_factors, WeightedPriceInputs, LockedAnchorExitController

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_low_anchored_reclaim_v1"
CONFIG = ROOT / "config/510300_low_anchored_reclaim_v1.json"
PRIMARY = "LOW_ANCHORED_RECLAIM"
NAME = "阶段新低后重回固定起点量价价格"
P32 = ROOT / "reports/research/510300_rearmed_session_exit_v1"
P37 = ROOT / "reports/research/510300_volume_weighted_trend_v1"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
CONTROLS = {"VWMA_20_60": (P37, "原成交量加权二十日与六十日均线"), "REARM_RIDGE": (P32, "原学习退出及等待新机会"),
            "PANIC_LEARNED_HALF": (P46, "原急跌回升及学习退出各半"), "BUY_HOLD": (P32, "买入持有")}


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "specification"]}
    cfg.update(study_id="510300_LOW_ANCHORED_RECLAIM_V1", round=61, registered_at=now(), primary=PRIMARY, candidate_configurations=1,
               low_window_including_today=252, low_comparison="STRICTLY_BELOW_PREVIOUS_251_VALID_WEALTH_CLOSES",
               entry="CROSS_FROM_AT_OR_BELOW_TO_ABOVE_SAME_ANCHOR_WEIGHTED_PRICE", anchor_after_entry="LOCK_ENTRY_ORIGIN_ANCHOR_UNTIL_ACTUAL_EXIT",
               typical_price="MEAN_HIGH_LOW_CLOSE_TIMES_WEALTH_DIVIDED_BY_CLOSE", extra_exit="CLOSE_STRICTLY_BELOW_LOCKED_ANCHOR_WEIGHTED_PRICE",
               new_model_fits=0, new_reference_accounts=0, rules="docs/510300_LOW_ANCHORED_RECLAIM_V1.md",
               position_impact=0, goal_achieved=False, independent_validation="NOT_ESTABLISHED")
    paths = [Path(__file__), ROOT / "research/low_anchored_price_v1.py", ROOT / "research/additional_cycle_exit_account_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
             ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"], ROOT / "tests/test_low_anchored_price_v1.py",
             ROOT / "config/510300_research_authority_v6.json"]
    for period in ["evaluation", "earlier_diagnostic"]:
        for cost in cfg["costs"]:
            paths.extend(p / period / cost / f"{key}_ledger.parquet" for key, (p, _) in CONTROLS.items())
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第61轮一个固定低点量价回升进入设置已登记，尚未读取本轮账户收益。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "低点量价进入登记文件发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    factors, _ = build_factors(data, cfg["low_window_including_today"])
    factors.to_parquet(OUT / "factors.parquet", index=False)
    main, earlier, yearly, eras, coverage, signal_counts = [], [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        local = factors.iloc[:len(frame)]
        inputs = WeightedPriceInputs(frame)
        rule = {"entry": local.raw_entry.to_numpy(int), "exit": {1: np.zeros(len(frame), bool)}}
        first = int(np.flatnonzero(frame.date >= start)[0])
        origins = local.iloc[first - 1:len(frame) - 1]
        signal_counts.append({"period": period, "decision_origins": len(origins), "new_low_anchors": int(origins.new_anchor.sum()),
                              "entry_crosses": int(origins.entry_cross.sum()), "eligible_entry_origins": int(origins.raw_entry.sum()),
                              "range_no_view_origins": int(origins.range_status.str.startswith("NO_VIEW").sum()),
                              "different_signal_anchor_dates": int(origins.loc[origins.raw_entry.eq(1), "anchor_date"].nunique())})
        for cost_id, cost in cfg["costs"].items():
            folder = OUT / period / cost_id
            controller = LockedAnchorExitController(frame, inputs, local)
            ledger, decisions, cycles = simulate_additional_exit(frame, dividends, cfg, cost, start, rule, cfg["specification"], controller)
            if cycles.empty:
                cycles = pd.DataFrame(columns=["cycle_id", "entry_date", "entry_index", "entry_origin", "entry_quantity", "entry_cost_cny", "exit_date", "exit_reasons", "holding_intervals", "net_profit_cny"])
            holding = decisions[decisions.anchor_cycle_id.notna()] if "anchor_cycle_id" in decisions else pd.DataFrame()
            if len(holding):
                anchor_map = holding.groupby("anchor_cycle_id").locked_anchor_date.first()
                cycles["locked_anchor_date"] = cycles.cycle_id.map(anchor_map)
                for cycle_id, group in holding.groupby("anchor_cycle_id"):
                    require(group.locked_anchor_index.nunique() == 1, "实际持仓期间起算点被移动")
                    require((group.locked_anchor_date < group.origin).all(), "持仓使用了当日或未来起算点")
            save_account(folder, PRIMARY, ledger, decisions)
            cycles.to_csv(folder / f"{PRIMARY}_cycles.csv", index=False, encoding="utf-8-sig")
            finished = cycles.dropna(subset=["exit_date"])
            require((finished.holding_intervals >= 1).all() and ledger.accounting_error.abs().max() < 1e-6, "低点量价账户或次日可卖约束不成立")
            require(not ledger.terminal_unliquidated.iloc[-1], "低点量价账户终点未清算")
            coverage.append({"period": period, "cost": cost_id, "model": PRIMARY, "completed_round_trips": len(finished),
                             "positive_cycles": int(finished.net_profit_cny.gt(0).sum()), "holding_decision_rows": len(holding),
                             "locked_price_no_view_rows": int(holding.locked_range_status.str.startswith("NO_VIEW").sum()) if len(holding) else 0,
                             "holding_closes_with_different_market_anchor": int(holding.current_market_anchor_differs.sum()) if len(holding) else 0,
                             "exits_with_locked_price_reason": int(finished.exit_reasons.str.contains("固定起点", regex=False).sum()),
                             "different_actual_anchor_dates": int(cycles.locked_anchor_date.nunique()) if len(holding) else 0,
                             "mean_holding_intervals": float(finished.holding_intervals.mean()) if len(finished) else None})
            accounts, names = {PRIMARY: ledger}, {PRIMARY: NAME}
            for key, (source, name) in CONTROLS.items():
                saved = pd.read_parquet(source / period / cost_id / f"{key}_ledger.parquet")
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            bh = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "低点量价比较账户缺少相同完整日历")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - bh["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= cfg["high_sharpe_target"]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                if period == "evaluation":
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[saved.date.between(left, right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            pd.DataFrame({"date": ledger.date, **{k: a.net_return.to_numpy() for k, a in accounts.items()}}).to_parquet(OUT / f"{period}_{cost_id}_returns.parquet", index=False)
            print(f"{period}／{cost_id}：低点量价进入及四个原样对照账户完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("entry_exit_coverage.csv", coverage), ("signal_counts.csv", signal_counts)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    primary = [m for m in main if m["model"] == PRIMARY]
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "LOW_ANCHORED_RECLAIM_ACCOUNTS_COMPLETE", "candidate_configurations": 1,
              "evaluation_accounts": len(main), "new_accounts_generated": 2, "reused_control_accounts": 8,
              "earlier_diagnostic_accounts": len(earlier), "new_earlier_diagnostic_accounts": 2, "reused_earlier_accounts": 8,
              "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier,
              "entry_exit_coverage": coverage, "signal_counts": signal_counts,
              "primary": primary, "post_selected_best_base": next(m for m in primary if m["cost"] == "BASE"),
              "historical_point_target_met": any(m["meets_point_target"] for m in primary),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"主评价": primary, "较早历史": [m for m in earlier if m["model"] == PRIMARY], "覆盖": coverage}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
