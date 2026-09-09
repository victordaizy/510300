"""成交量加权趋势与同窗口普通均线的固定账户检验。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.simple_price_entry_exit_v1 import simulate_policy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_volume_weighted_trend_v1"
CONFIG = ROOT / "config/510300_volume_weighted_trend_v1.json"
NAMES = {"VWMA_20_60": "成交量加权二十日与六十日均线", "SMA_20_60": "普通二十日与六十日均线"}


def build_factors(data, windows=(20, 60)):
    price = data.wealth.where(np.isfinite(data.wealth) & (data.wealth > 0))
    volume = data.volume.where(np.isfinite(data.volume) & (data.volume >= 0))
    volume = volume.where(price.notna())
    result = pd.DataFrame({"date": data.date, "wealth": price, "volume": volume})
    for window in windows:
        denominator = volume.rolling(window, min_periods=window).sum()
        numerator = (price * volume).rolling(window, min_periods=window).sum()
        result[f"VWMA_{window}"] = numerator / denominator.where(denominator > 0)
        result[f"SMA_{window}"] = price.rolling(window, min_periods=window).mean()
    return result


def make_rule(factors, kind):
    price, fast, slow = factors.wealth, factors[f"{kind}_20"], factors[f"{kind}_60"]
    valid = np.isfinite(price) & np.isfinite(fast) & np.isfinite(slow)
    return {"entry": (valid & (price > fast) & (fast > slow)).to_numpy(int),
            "exit": {1: (valid & ((price <= fast) | (fast <= slow))).to_numpy(bool)}}


def freeze():
    old = json.loads((ROOT / "config/510300_high_low_relation_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal"]}
    cfg.update(study_id="510300_VOLUME_WEIGHTED_TREND_V1", round=37, registered_at=now(), primary="VWMA_20_60",
               candidate_configurations=2, candidate_names=NAMES, windows=[20, 60], new_model_fits=0,
               specification={"cooldown": 2, "modes": {1: {"loss": None, "trail": None, "take": None, "days": None}}},
               rules="docs/510300_VOLUME_WEIGHTED_TREND_V1.md", position_impact=0)
    paths = [Path(__file__), ROOT / "research/simple_price_entry_exit_v1.py", ROOT / "research/intraday_overnight_increment_v1.py",
             ROOT / "research/adaptive_allocation_v1.py", ROOT / cfg["rules"], ROOT / cfg["features"], ROOT / cfg["dividends"],
             ROOT / "tests/test_volume_weighted_trend_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第37轮两项均线设置已登记，窗口、进入退出和等待规则固定。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "成交量均线登记内容发生变化")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    factors = build_factors(data, cfg["windows"])
    factors.to_parquet(OUT / "factors.parquet", index=False)
    main, early, yearly, eras, increments = [], [], [], [], []
    parent = ROOT / "reports/research/510300_rearmed_session_exit_v1"
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        local = factors.iloc[:len(frame)]
        for cost_id, cost in cfg["costs"].items():
            folder, accounts, names = OUT / period / cost_id, {}, {}
            for key, name in NAMES.items():
                ledger, decisions, cycles = simulate_policy(frame, dividends, cfg, cost, start, make_rule(local, key.split("_")[0]), cfg["specification"])
                save_account(folder, key, ledger, decisions)
                cycles.to_csv(folder / f"{key}_cycles.csv", index=False, encoding="utf-8-sig")
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "均线账户结算失败")
                if len(cycles):
                    require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "均线账户违反买入次日可卖")
                accounts[key], names[key] = ledger, name
            for key, name in [("REARM_RIDGE", "原线性退出＋等待新机会"), ("BUY_HOLD", "买入持有")]:
                saved = pd.read_parquet(parent / period / cost_id / f"{key}_ledger.parquet")
                saved.to_parquet(folder / f"{key}_ledger.parquet", index=False)
                accounts[key], names[key] = saved, name
            base = summarize(accounts["BUY_HOLD"], cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(accounts["BUY_HOLD"].date)), "均线评价日不完整")
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
            a, b = accounts["VWMA_20_60"], accounts["SMA_20_60"]
            price_delta, dividends_delta = float(a.price_pnl.sum() - b.price_pnl.sum()), float(a.dividend_recognized.sum() - b.dividend_recognized.sum())
            fees_delta = float(a.commission.sum() + a.slippage_cost.sum() - b.commission.sum() - b.slippage_cost.sum())
            net_delta = float(a.equity.iloc[-1] - b.equity.iloc[-1])
            require(abs(net_delta - price_delta - dividends_delta + fees_delta) < 1e-6, "成交量增量不能与账户现金核对")
            increments.append({"period": period, "cost": cost_id, "terminal_equity_increment": net_delta,
                               "price_pnl_increment": price_delta, "dividend_increment": dividends_delta,
                               "extra_friction_cny": fees_delta, "net_sharpe_increment": summarize(a, cfg)["net_sharpe"] - summarize(b, cfg)["net_sharpe"]})
            print(f"{period}／{cost_id}：两种均线和两个原样对照已完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly),
                           ("era_metrics.csv", eras), ("volume_increment.csv", increments)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    best = max((m for m in main if m["cost"] == "BASE" and m["model"] in NAMES), key=lambda x: x["net_sharpe"] if x["net_sharpe"] is not None else -999)
    result = {"study_id": cfg["study_id"], "completed_at": now(), "status": "VOLUME_WEIGHTED_TREND_COMPLETE", "candidate_configurations": 2,
              "evaluation_accounts": 8, "new_accounts_generated": 4, "reused_control_accounts": 4, "earlier_diagnostic_accounts": 8,
              "new_earlier_diagnostic_accounts": 4, "reused_earlier_accounts": 4, "new_model_fits": 0,
              "all_metrics": main, "earlier_diagnostics": early, "primary": [m for m in main if m["model"] == cfg["primary"]],
              "post_selected_best_base": best, "volume_increment": increments,
              "historical_point_target_met": any(m["meets_point_target"] for m in main if m["model"] in NAMES),
              "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0}
    write_json(OUT / "result.json", result, exclusive=True)
    print(json.dumps({"状态": result["status"], "主评价": [{k: m[k] for k in ["model", "cost", "net_sharpe", "annualized_return", "max_drawdown", "trade_count"]} for m in main if m["model"] in NAMES],
                      "较早": [{k: m[k] for k in ["model", "cost", "net_sharpe", "annualized_return"]} for m in early if m["model"] in NAMES], "成交量增量": increments}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("请指定 freeze 或 run")
