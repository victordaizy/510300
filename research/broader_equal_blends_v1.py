"""两个预定第三信号的固定三等份组合，复用原规则和账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.event_clock_account_v1 import simulate_event_account
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.simple_price_entry_exit_v1 import simulate_policy
from research.simple_signal_blend_v1 import decision_state
from research.simple_volume_reversal_v1 import make_rules as volume_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_broader_equal_blends_v1"
CONFIG = ROOT / "config/510300_broader_equal_blends_v1.json"
P46 = ROOT / "reports/research/510300_panic_learned_equal_blend_v1"
NAMES = {"ADD_VOLUME_BREAKOUT": "加入放量突破，三个信号各三分之一", "ADD_REGULAR_REBOUND": "加入普通均值回升，三个信号各三分之一"}


def source_path(period, cost, component, artifact="ledger"):
    if component == "VOLUME":
        root = ROOT / "reports/research/510300_simple_volume_reversal_v1/evaluation" if period == "evaluation" else OUT / "new_earlier_control"
        return root / cost / f"V3_VOLUME_BREAKOUT_{artifact}.parquet"
    if period == "evaluation":
        return ROOT / "reports/research/510300_simple_price_entry_exit_v1/evaluation" / cost / f"R2_Z_CONFIRM_{artifact}.parquet"
    return ROOT / "reports/research/510300_learned_cycle_exit_v1/earlier_diagnostic" / cost / f"R2_Z_CONFIRM__CLOSE_NEXT_OPEN_{artifact}.parquet"


def freeze():
    old = json.loads((ROOT / "config/510300_panic_learned_equal_blend_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal", "weight_band"]}
    volume = json.loads((ROOT / "config/510300_simple_volume_reversal_v1.json").read_text(encoding="utf-8"))
    cfg.update(study_id="510300_BROADER_EQUAL_BLENDS_V1", round=47, registered_at=now(), candidate_configurations=2, primary="ADD_VOLUME_BREAKOUT",
               names=NAMES, fixed_weights=[1/3, 1/3, 1/3], rules="docs/510300_BROADER_EQUAL_BLENDS_V1.md",
               new_model_fits=0, position_impact=0, volume_specification=volume["candidate_specs"]["V3_VOLUME_BREAKOUT"])
    paths = [Path(__file__), ROOT / "research/event_clock_account_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/simple_signal_blend_v1.py",
             ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/simple_volume_reversal_v1.py",
             ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"]]
    for period in ["evaluation", "earlier_diagnostic"]:
        paths += [P46 / f"{period}_states.parquet", source_path(period, "BASE", "REBOUND", "decisions")]
    paths += [source_path("evaluation", "BASE", "VOLUME", "decisions")]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第47轮两个固定三等份组合已登记，只补算原放量突破缺少的较早对照账户。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "第三信号组合登记内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    earlier = data[data.date <= cfg["earlier_terminal"]].copy()
    v3_rule = volume_rules(earlier)[0]["V3_VOLUME_BREAKOUT"]
    for cost_id, cost in cfg["costs"].items():
        ledger, decisions, cycles = simulate_policy(earlier, dividends, cfg, cost, cfg["earlier_start"], v3_rule, cfg["volume_specification"])
        save_account(OUT / "new_earlier_control" / cost_id, "V3_VOLUME_BREAKOUT", ledger, decisions)
        cycles.to_csv(OUT / "new_earlier_control" / cost_id / "V3_VOLUME_BREAKOUT_cycles.csv", index=False, encoding="utf-8-sig")
        require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "原放量突破较早对照未完成结算")
    main, early, yearly, eras, counts = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", earlier, cfg["earlier_start"], early)]:
        existing = pd.read_parquet(P46 / f"{period}_states.parquet")
        require(pd.DatetimeIndex(existing.date).equals(pd.DatetimeIndex(frame.date)), "已有核心状态日历不同")
        volume = decision_state(frame, pd.read_parquet(source_path(period, "BASE", "VOLUME", "decisions")))
        rebound = decision_state(frame, pd.read_parquet(source_path(period, "BASE", "REBOUND", "decisions")))
        core = existing.panic_state.to_numpy(float) + existing.learned_state.to_numpy(float)
        targets = {"ADD_VOLUME_BREAKOUT": (core + volume) / 3, "ADD_REGULAR_REBOUND": (core + rebound) / 3}
        anchor = int(np.flatnonzero(frame.date >= pd.Timestamp(start))[0]) - 1
        for key, target in targets.items():
            require(np.isfinite(target[anchor:-1]).all(), "组合所需状态缺失，不能替代成现金")
            require(np.isin(target[anchor:-1], [0., 1/3, 2/3, 1.]).all(), "三等份组合超出预定预算档位")
            for value in [0., 1/3, 2/3, 1.]:
                counts.append({"period": period, "model": key, "target": value, "decision_days": int((target[anchor:-1] == value).sum())})
        pd.DataFrame({"date": frame.date, "panic_state": existing.panic_state, "learned_state": existing.learned_state,
                      "volume_state": volume, "rebound_state": rebound, **targets}).to_parquet(OUT / f"{period}_states.parquet", index=False)
        for cost_id, cost in cfg["costs"].items():
            dest, accounts, names = OUT / period / cost_id, {}, {}
            for key, target in targets.items():
                ledger, decisions = simulate_event_account(frame, dividends, cfg, cost, start, key, targets=target, event_mask=np.ones(len(frame), bool))
                save_account(dest, key, ledger, decisions)
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "三等份组合账户结算失败")
                accounts[key], names[key] = ledger, NAMES[key]
            controls = [("ORIGINAL_TWO", P46 / period / cost_id / "PANIC_LEARNED_HALF_ledger.parquet", "原两个信号各半组合"),
                        ("VOLUME_ONLY", source_path(period, cost_id, "VOLUME"), "原放量突破策略"),
                        ("REBOUND_ONLY", source_path(period, cost_id, "REBOUND"), "原普通均值回升策略"),
                        ("BUY_HOLD", P46 / period / cost_id / "BUY_HOLD_ledger.parquet", "买入持有")]
            for key, path, name in controls:
                saved = pd.read_parquet(path)
                saved.to_parquet(dest / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "第三信号组合全账户日期不一致")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                destination.append(m)
                if period == "evaluation":
                    for year, group in saved.groupby(saved.date.dt.year):
                        yearly.append({"cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[(saved.date >= left) & (saved.date <= right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：两个三等份组合和四个对照已完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras), ("state_counts.csv", counts)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "BROADER_EQUAL_BLENDS_COMPLETE", "candidate_configurations": 2,
              "evaluation_accounts": 12, "new_accounts_generated": 4, "reused_control_accounts": 8, "earlier_diagnostic_accounts": 12,
              "new_earlier_diagnostic_accounts": 6, "new_earlier_candidate_accounts": 4, "new_earlier_control_accounts": 2, "reused_earlier_accounts": 6,
              "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": early, "state_counts": counts,
              "primary": [m for m in main if m["model"] == cfg["primary"]],
              "post_selected_best_base": max((m for m in main if m["model"] in NAMES and m["cost"] == "BASE"), key=lambda m: m["net_sharpe"] if m["net_sharpe"] is not None else -999),
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] in NAMES),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"新主评价": [m for m in main if m["model"] in NAMES], "新较早": [m for m in early if m["model"] in NAMES]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"freeze": freeze, "run": run}[sys.argv[1]]()
