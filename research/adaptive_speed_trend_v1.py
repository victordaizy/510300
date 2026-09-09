"""逐日调整平滑速度与固定速度对照，使用相同的实际进出场账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.simple_price_entry_exit_v1 import simulate_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_adaptive_speed_trend_v1"
CONFIG = ROOT / "config/510300_adaptive_speed_trend_v1.json"
NAMES = {"ADAPTIVE_SPEED": "方向效率决定均线反应速度", "FIXED_NEUTRAL_SPEED": "效率固定为一半的同规则对照"}


def build_factors(data, window=10, fast=2, slow=30):
    """缺失打断递推；下次完整连续窗口重新以其前一收盘初始化。"""
    w = data.wealth.to_numpy(float)
    n = len(w)
    efficiency, adaptive_gain, adaptive_line, fixed_line = [np.full(n, np.nan) for _ in range(4)]
    fastest, slowest = 2. / (fast + 1), 2. / (slow + 1)
    neutral_gain = (slowest + .5 * (fastest - slowest)) ** 2
    previous_a, previous_f = np.nan, np.nan
    for t in range(window, n):
        local = w[t - window:t + 1]
        if not np.isfinite(local).all() or np.any(local <= 0):
            previous_a, previous_f = np.nan, np.nan
            continue
        path = float(np.abs(np.diff(local)).sum())
        er = float(abs(local[-1] - local[0]) / path) if path > 0 else 0.
        require(-1e-12 <= er <= 1 + 1e-12, "方向效率越界")
        er = float(np.clip(er, 0, 1))
        gain = (slowest + er * (fastest - slowest)) ** 2
        if not np.isfinite(previous_a):
            previous_a, previous_f = w[t - 1], w[t - 1]
        previous_a += gain * (w[t] - previous_a)
        previous_f += neutral_gain * (w[t] - previous_f)
        efficiency[t], adaptive_gain[t], adaptive_line[t], fixed_line[t] = er, gain, previous_a, previous_f
    available = np.isfinite(adaptive_line) & np.isfinite(fixed_line)
    return pd.DataFrame({"date": data.date, "wealth": w, "direction_efficiency10": efficiency,
                         "adaptive_gain": adaptive_gain, "fixed_neutral_gain": np.where(available, neutral_gain, np.nan),
                         "adaptive_line": adaptive_line, "fixed_line": fixed_line,
                         "factor_state": np.where(available, "VALUE_AVAILABLE", "NO_VIEW")})


def make_rule(factors, key):
    line = factors.adaptive_line if key == "ADAPTIVE_SPEED" else factors.fixed_line
    valid = factors.factor_state.eq("VALUE_AVAILABLE")
    above = valid & factors.wealth.gt(line)
    below = valid & factors.wealth.lt(line)
    entry = above & above.shift(1, fill_value=False)
    leave = below & below.shift(1, fill_value=False)
    return {"entry": entry.to_numpy(int), "exit": {1: leave.to_numpy(bool)}}


def freeze():
    old = json.loads((ROOT / "config/510300_broader_equal_blends_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    cfg.update(study_id="510300_ADAPTIVE_SPEED_TREND_V1", round=48, registered_at=now(), candidate_configurations=2,
               primary="ADAPTIVE_SPEED", names=NAMES, efficiency_window=10, fast_period=2, slow_period=30,
               neutral_efficiency=.5, confirmation_closes=2, rules="docs/510300_ADAPTIVE_SPEED_TREND_V1.md",
               specification={"cooldown": 0, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}},
               new_model_fits=0, position_impact=0, evidence_class="NEW_METHOD_ON_PREVIOUSLY_OBSERVED_HISTORY")
    paths = [Path(__file__), ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/adaptive_allocation_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "tests/test_adaptive_speed_trend_v1.py",
             ROOT / cfg["features"], ROOT / cfg["dividends"], ROOT / cfg["rules"]]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第48轮自适应均线及同规则固定速度对照已登记，共两个设置。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "反应速度研究登记文件发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    factors = build_factors(data, cfg["efficiency_window"], cfg["fast_period"], cfg["slow_period"])
    factors.to_parquet(OUT / "factors.parquet", index=False)
    main, earlier, yearly, eras, factor_stats = [], [], [], [], []
    for period, frame, start, destination in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], earlier)]:
        f = factors.iloc[:len(frame)]
        anchor = int(np.flatnonzero(frame.date >= pd.Timestamp(start))[0]) - 1
        active = f.iloc[anchor:-1]
        require(active.factor_state.eq("VALUE_AVAILABLE").all(), "评价所需因子出现缺失")
        factor_stats.append({"period": period, "decision_days": len(active), "valid_days": int(active.factor_state.eq("VALUE_AVAILABLE").sum()),
                             "mean_direction_efficiency": float(active.direction_efficiency10.mean()),
                             "mean_adaptive_gain": float(active.adaptive_gain.mean()),
                             "min_adaptive_gain": float(active.adaptive_gain.min()), "max_adaptive_gain": float(active.adaptive_gain.max()),
                             "fixed_neutral_gain": float(active.fixed_neutral_gain.iloc[0])})
        for cost_id, cost in cfg["costs"].items():
            dest, accounts, names, cycles_by_key = OUT / period / cost_id, {}, {}, {}
            for key, name in NAMES.items():
                ledger, decisions, cycles = simulate_policy(frame, dividends, cfg, cost, start, make_rule(f, key), cfg["specification"])
                save_account(dest, key, ledger, decisions)
                cycles.to_csv(dest / f"{key}_cycles.csv", index=False, encoding="utf-8-sig")
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "反应速度策略账户未完成结算")
                require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "反应速度策略违反次日可卖")
                accounts[key], names[key], cycles_by_key[key] = ledger, name, len(cycles)
            for key, slug, source_key, name in [
                ("REARM_RIDGE", "rearmed_session_exit", "REARM_RIDGE", "已有线性退出及等待新机会"),
                ("ORIGINAL_TWO", "panic_learned_equal_blend", "PANIC_LEARNED_HALF", "原急跌与学习信号各半"),
                ("BUY_HOLD", "rearmed_session_exit", "BUY_HOLD", "买入持有")]:
                saved = pd.read_parquet(ROOT / f"reports/research/510300_{slug}_v1" / period / cost_id / f"{source_key}_ledger.parquet")
                saved.to_parquet(dest / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "反应速度账户全日历不同")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                if key in cycles_by_key:
                    m["completed_cycles"] = cycles_by_key[key]
                destination.append(m)
                for year, group in saved.groupby(saved.date.dt.year):
                    yearly.append({"period": period, "cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                intervals = [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"),
                             ("2024—终点", "2024-01-01", cfg["data_cutoff"])] if period == "evaluation" else [
                             ("2015—2016", "2015-01-01", "2016-12-31"), ("2017—2019", "2017-01-01", cfg["earlier_terminal"])]
                for label, left, right in intervals:
                    group = saved[(saved.date >= left) & (saved.date <= right)]
                    eras.append({"period": period, "cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：自适应和固定速度账户完成，三个原对照复用。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", earlier), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("factor_statistics.csv", factor_stats)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    best = max((m for m in main if m["cost"] == "BASE" and m["model"] in NAMES), key=lambda m: m["net_sharpe"] if m["net_sharpe"] is not None else -np.inf)
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "ADAPTIVE_SPEED_TREND_ACCOUNTS_COMPLETE",
              "candidate_configurations": 2, "evaluation_accounts": 10, "new_accounts_generated": 4, "reused_control_accounts": 6,
              "earlier_diagnostic_accounts": 10, "new_earlier_diagnostic_accounts": 4, "reused_earlier_accounts": 6,
              "new_model_fits": 0, "new_reference_accounts": 0, "all_metrics": main, "earlier_diagnostics": earlier,
              "factor_statistics": factor_stats, "primary": [m for m in main if m["model"] == cfg["primary"]],
              "post_selected_best_base": best, "goal_achieved": False, "independent_validation": False, "position_impact": 0}
    write_json(OUT / "result.json", result)
    print(json.dumps({"状态": result["status"], "主评价": [{k: m[k] for k in ["model", "cost", "net_sharpe", "annualized_return", "max_drawdown", "trade_count"]} for m in main if m["model"] in NAMES],
                      "较早历史": [{k: m[k] for k in ["model", "cost", "net_sharpe", "annualized_return", "max_drawdown", "trade_count"]} for m in earlier if m["model"] in NAMES]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or sys.argv[1] not in {"freeze", "run"}:
        raise SystemExit("用法：python -m research.adaptive_speed_trend_v1 freeze 或 run")
    freeze() if sys.argv[1] == "freeze" else run()
