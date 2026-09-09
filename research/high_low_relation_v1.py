"""三个固定高低价关系因子及完整进出场账户。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.simple_price_entry_exit_v1 import simulate_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_high_low_relation_v1"
CONFIG = ROOT / "config/510300_high_low_relation_v1.json"
NAMES = {"H1_Z": "高低价斜率标准分", "H2_QUALITY": "拟合质量调整的标准分", "H3_RIGHT": "斜率与质量共同调整"}


def adjusted_levels(data):
    scale = data.wealth / (data.close + data.dividend)
    return (data.high + data.dividend) * scale, (data.low + data.dividend) * scale


def build_factors(data, window=20, history=242):
    high, low = adjusted_levels(data)
    n = len(data)
    beta, quality = np.full(n, np.nan), np.full(n, np.nan)
    if n >= window:
        x = np.lib.stride_tricks.sliding_window_view(low.to_numpy(float), window)
        y = np.lib.stride_tricks.sliding_window_view(high.to_numpy(float), window)
        xc, yc = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
        xx, yy, xy = (xc * xc).sum(axis=1), (yc * yc).sum(axis=1), (xc * yc).sum(axis=1)
        valid = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1) & (xx > 1e-20) & (yy > 1e-20)
        ids = np.flatnonzero(valid) + window - 1
        beta[ids] = xy[valid] / xx[valid]
        quality[ids] = np.clip(xy[valid] ** 2 / (xx[valid] * yy[valid]), 0., 1.)
    slope = pd.Series(beta, index=data.index)
    mean = slope.shift(1).rolling(history, min_periods=history).mean()
    std = slope.shift(1).rolling(history, min_periods=history).std(ddof=1)
    z = (slope - mean) / std.where(std > 1e-12)
    result = pd.DataFrame({"date": data.date, "adjusted_high": high, "adjusted_low": low, "slope": slope,
                           "fit_quality": quality, "previous_slope_mean": mean, "previous_slope_std": std,
                           "H1_Z": z, "H2_QUALITY": z * quality, "H3_RIGHT": z * quality * slope})
    result["factor_state"] = np.where(np.isfinite(result[list(NAMES)]).all(axis=1), "VALUE_AVAILABLE", "NO_VIEW_WARMUP_OR_DEGENERATE")
    return result


def make_rule(factors, key):
    score, beta = factors[key], factors.slope
    return {"entry": ((score > .7) & (beta > 0)).fillna(False).to_numpy(int),
            "exit": {1: ((score < -.7) | (beta <= 0)).fillna(False).to_numpy(bool)}}


def freeze():
    old = json.loads((ROOT / "config/510300_trend_learned_equal_blend_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                               "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    cfg.update({"study_id": "510300_HIGH_LOW_RELATION_V1", "round": 35, "registered_at": now(), "primary": "H2_QUALITY",
                "candidate_configurations": 3, "candidate_names": NAMES, "regression_window": 20, "normalization_history": 242,
                "entry_threshold": .7, "exit_threshold": -.7, "nonpositive_slope_exit": True, "new_predictive_model_fits": 0,
                "specification": {"cooldown": 0, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}},
                "rules": "docs/510300_HIGH_LOW_RELATION_V1.md", "position_impact": 0})
    paths = [Path(__file__), ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"],
             ROOT / "tests/test_high_low_relation_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第35轮三个高低价关系因子已登记，参数固定，不训练未来收益模型。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "高低价因子登记内容发生变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    div = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    factors = build_factors(data, cfg["regression_window"], cfg["normalization_history"])
    factors.to_parquet(OUT / "factors.parquet", index=False)
    main, early, yearly, eras = [], [], [], []
    parent = ROOT / "reports/research/510300_rearmed_session_exit_v1"
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        f = factors.iloc[:len(frame)]
        for cost_id, cost in cfg["costs"].items():
            folder, accounts, names = OUT / period / cost_id, {}, {}
            for key, name in NAMES.items():
                ledger, decisions, cycles = simulate_policy(frame, div, cfg, cost, start, make_rule(f, key), cfg["specification"])
                save_account(folder, key, ledger, decisions)
                cycles.to_csv(folder / f"{key}_cycles.csv", index=False, encoding="utf-8-sig")
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "高低价策略账户结算失败")
                if len(cycles):
                    require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "高低价策略违反次日可卖")
                accounts[key], names[key] = ledger, name
            for key, name in [("REARM_RIDGE", "已有线性退出＋等待新机会"), ("BUY_HOLD", "买入持有")]:
                saved = pd.read_parquet(parent / period / cost_id / f"{key}_ledger.parquet")
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "高低价策略评价日不完整")
                m = {"cost": cost_id, "model": key, "name": names[key], **summarize(saved, cfg)}
                m["annualized_return_excess_vs_buy_hold"] = m["annualized_return"] - base["annualized_return"]
                m["meets_point_target"] = m["net_sharpe"] is not None and m["net_sharpe"] >= 1.2
                dest.append(m)
                if period == "evaluation":
                    for year, group in saved.groupby(saved.date.dt.year):
                        yearly.append({"cost": cost_id, "model": key, "year": int(year), **summarize(group, cfg)})
                    for label, left, right in [("2020—2021", "2020-01-01", "2021-12-31"), ("2022—2023", "2022-01-01", "2023-12-31"), ("2024—终点", "2024-01-01", cfg["data_cutoff"])]:
                        group = saved[(saved.date >= left) & (saved.date <= right)]
                        eras.append({"cost": cost_id, "model": key, "era": label, **summarize(group, cfg)})
            print(f"{period}／{cost_id}：三个高低价关系账户和两个对照已完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    best = max((m for m in main if m["cost"] == "BASE" and m["model"] in NAMES), key=lambda x: x["net_sharpe"] if x["net_sharpe"] is not None else -999)
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "HIGH_LOW_RELATION_ACCOUNTS_COMPLETE", "candidate_configurations": 3,
              "evaluation_accounts": 10, "new_accounts_generated": 6, "reused_control_accounts": 4,
              "earlier_diagnostic_accounts": 10, "new_earlier_diagnostic_accounts": 6, "reused_earlier_accounts": 4,
              "rolling_factor_regressions": int(factors.slope.notna().sum()), "new_predictive_model_fits": 0,
              "first_valid_factor_date": factors.loc[factors.factor_state == "VALUE_AVAILABLE", "date"].min(),
              "all_metrics": main, "earlier_diagnostics": early, "primary": [m for m in main if m["model"] == cfg["primary"]],
              "post_selected_best_base": best, "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] in NAMES),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "基础费用结果": [{k: m[k] for k in ["model", "net_sharpe", "annualized_return", "max_drawdown", "trade_count"]} for m in main if m["cost"] == "BASE"],
                      "较早新设置": [{k: m[k] for k in ["model", "cost", "net_sharpe", "annualized_return"]} for m in early if m["model"] in NAMES]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
