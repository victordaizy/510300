"""用已有分钟价格提前检查一次原退出模型，不训练新模型。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from research.adaptive_allocation_v1 import normalize_dividends, save_account, summarize
from research.afternoon_exit_account_v1 import simulate_afternoon_exit
from research.intraday_overnight_increment_v1 import digest, now, require, write_json
from research.learned_cycle_exit_v1 import ExitController, FEATURES, predict
from research.simple_intraday_protection_v1 import make_rules

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_afternoon_learned_exit_v1"
CONFIG = ROOT / "config/510300_afternoon_learned_exit_v1.json"
PARENT = ROOT / "reports/research/510300_rearmed_session_exit_v1"
MINUTE = ROOT / "data/raw/market/510300_1m_tushare_raw.parquet"
KEY = "AFTERNOON_EXIT"
NAME = "原线性模型下午提前确认退出"


def partial_values(data, t, cycle, current_value, peak_value, price, annual_days=242):
    """今日只使用已知下午价和除息金额，历史窗口严格截至昨日。"""
    if t < 119 or not np.isfinite(price) or price <= 0:
        return np.full(len(FEATURES), np.nan)
    history = data.wealth.iloc[t - 119:t].to_numpy(float)
    past_returns = data.total_simple.iloc[t - 19:t].to_numpy(float)
    if not np.isfinite(history).all() or not (history > 0).all() or not np.isfinite(past_returns).all():
        return np.full(len(FEATURES), np.nan)
    previous = float(data.close.iloc[t - 1])
    partial_return = (price + float(data.dividend.iloc[t])) / previous - 1
    wealth = float(data.wealth.iloc[t - 1]) * (1 + partial_return)
    mean120 = (history.sum() + wealth) / 120
    returns = np.r_[past_returns, partial_return]
    return np.array([np.log1p(t - cycle["entry_index"] + 1), current_value / cycle["entry_cost_cny"] - 1,
                     current_value / peak_value - 1, cycle["mode"], np.log(wealth / data.wealth.iloc[t - 5]),
                     np.log(wealth / data.wealth.iloc[t - 20]), wealth / mean120 - 1,
                     np.std(returns, ddof=1) * np.sqrt(annual_days)], dtype=float)


class AfternoonPreview:
    def __init__(self, data, models, daily_controller, annual_days=242):
        self.data, self.models, self.daily, self.annual_days = data, models, daily_controller, annual_days
        self.fit_times = [pd.Timestamp(m["fit_time"]) for m in models]

    def __call__(self, t, cycle, value, peak, price, signal_time):
        available = [i for i, time in enumerate(self.fit_times) if time <= signal_time]
        stored = self.models[available[-1]] if available else None
        values = partial_values(self.data, t, cycle, value, peak, price, self.annual_days)
        estimate, status = None, "NO_VIEW_NO_MATURE_MODEL"
        if stored and stored["status"] == "FIT_COMPLETE" and np.isfinite(values).all():
            require(pd.Timestamp(stored["fit_time"]) <= signal_time and stored["latest_exit_index"] <= stored["fit_index"] <= t,
                    "下午预览使用了尚未完成的模型或未来持仓标签")
            estimate, status = predict(stored["model"], values), "PREDICTION_AVAILABLE"
        elif stored and stored["status"] == "FIT_COMPLETE":
            status = "NO_VIEW_INCOMPLETE_HOLDING_FEATURES"
        prior = self.daily.negative_count if self.daily.cycle_id == cycle["cycle_id"] else 0
        return {"learning_status": status, "continuation_prediction": estimate,
                "learning_fit_time": pd.Timestamp(stored["fit_time"]) if stored else pd.NaT,
                "previous_close_negative_count": prior,
                "learned_exit_requested": bool(estimate is not None and estimate < 0 and prior >= self.daily.confirmation_days - 1),
                **dict(zip(FEATURES, values))}


def load_snapshots():
    minute = pd.read_parquet(MINUTE)
    minute["trade_time"] = pd.to_datetime(minute.trade_time)
    require(not minute.trade_time.duplicated().any() and set(minute.ts_code) == {"510300.SH"}, "分钟来源时间重复或证券不符")
    require(minute.trade_time.is_monotonic_increasing, "分钟原始记录未按时间升序排列")
    minute["date"] = minute.trade_time.dt.normalize()
    clock = minute.trade_time.dt.strftime("%H:%M")
    signal = minute.loc[clock == "14:29", ["date", "close"]].set_index("date").rename(columns={"close": "signal_price"})
    execution = minute.loc[clock == "14:46", ["date", "open", "vol"]].set_index("date").rename(columns={"open": "execution_price", "vol": "execution_volume"})
    # 外连接保留信号已知但未来成交记录缺失的日期，禁止按未来数据删信号。
    snapshots = signal.join(execution, how="outer").sort_index()
    snapshots["signal_label"] = snapshots.index + pd.Timedelta(hours=14, minutes=29)
    snapshots["execution_label"] = snapshots.index + pd.Timedelta(hours=14, minutes=46)
    return minute, snapshots


def inventory():
    minute, snapshots = load_snapshots()
    data = pd.read_parquet(ROOT / "reports/research/510300_adaptive_allocation_v1/features.parquet")
    bad_prices = (~np.isfinite(minute[["open", "high", "low", "close"]]).all(axis=1) | (minute[["open", "high", "low", "close"]] <= 0).any(axis=1))
    require(not bad_prices.any(), "分钟行情价格缺失或不为正")
    daily = minute.groupby("date").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("vol", "sum"))
    joined = daily.join(data.set_index("date")[["open", "high", "low", "close", "volume"]], rsuffix="_daily", how="left")
    price_errors = {c: float((joined[c] - joined[c + "_daily"]).abs().max()) for c in ["open", "high", "low", "close"]}
    require(max(price_errors.values()) < .001000001, "分钟与既有日线价格未能在一跳内对齐")
    result = {"checked_at": now(), "status": "EXISTING_MINUTE_DATA_USABLE_FOR_LIMITED_HISTORICAL_PROXY_RESEARCH",
              "source": str(MINUTE.relative_to(ROOT)), "source_sha256": digest(MINUTE), "rows": len(minute), "days": minute.date.nunique(),
              "first": minute.trade_time.min(), "last": minute.trade_time.max(), "duplicate_timestamps": int(minute.trade_time.duplicated().sum()),
              "daily_counts": minute.groupby("date").size().value_counts().to_dict(), "daily_price_max_errors": price_errors,
              "signal_price_days": int(snapshots.signal_price.notna().sum()), "execution_price_days": int(snapshots.execution_price.notna().sum()),
              "main_days_without_minute": int(((data.date >= "2020-01-02") & ~data.date.isin(snapshots.index)).sum()),
              "earlier_minute_days": int(((snapshots.index >= "2015-01-05") & (snapshots.index <= "2019-12-31")).sum()),
              "official_definition_checked": "https://tushare.pro/document/2?doc_id=387", "volume_unit": "SHARES", "amount_unit": "CNY",
              "source_caveats": ["历史文件通过第三方代理取得，非直连官方来源。", "原接口名与当前官方ETF文档不同。",
                                 "交易时间没有柱首柱尾承诺；决策与价格代理之间保留分钟间隔。", "没有逐日历史发布快照、逐笔排队或真实成交证据。"],
              "new_downloads": 0, "new_accounts_generated": 0, "new_model_fits": 0}
    write_json(OUT / "existing_minute_inventory.json", result)
    print(f"现有分钟行情核对完成：{result['days']}日，主评价{result['main_days_without_minute']}日沿用原日线规则。", flush=True)


def freeze():
    old = json.loads((ROOT / "config/510300_rearmed_session_exit_v1.json").read_text(encoding="utf-8"))
    cfg = {k: old[k] for k in ["evaluation_start", "data_cutoff", "initial_capital", "lot", "tick", "limit_fraction", "annual_days",
                              "cash_annual_rate_assumption", "high_sharpe_target", "costs", "features", "dividends", "earlier_start", "earlier_terminal",
                              "confirmation_days", "specification", "saved_models"]}
    cfg.update(study_id="510300_AFTERNOON_LEARNED_EXIT_V1", round=43, registered_at=now(), primary=KEY, candidate_configurations=1,
               minute_file=str(MINUTE.relative_to(ROOT)), signal_observation="14:29", signal_decision="14:31", execution_proxy="14:46_OPEN",
               minute_participation_cap=.10, rules="docs/510300_AFTERNOON_LEARNED_EXIT_V1.md", new_model_fits=0, position_impact=0)
    paths = [Path(__file__), ROOT / "research/afternoon_exit_account_v1.py", ROOT / "research/learned_cycle_exit_v1.py",
             ROOT / "research/simple_intraday_protection_v1.py", ROOT / "research/simple_session_divergence_v1.py",
             ROOT / "research/intraday_overnight_increment_v1.py", ROOT / "research/adaptive_allocation_v1.py", ROOT / cfg["features"],
             ROOT / cfg["dividends"], ROOT / cfg["saved_models"], ROOT / cfg["rules"], MINUTE,
             OUT / "existing_minute_inventory.json", ROOT / "tests/test_afternoon_learned_exit_v1.py"]
    cfg["frozen_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": digest(p)} for p in paths]
    write_json(CONFIG, cfg, exclusive=True)
    print("第43轮一个下午提前退出设置已登记，不新增模型或扫描时间。", flush=True)


def run():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    for item in cfg["frozen_files"]:
        require(digest(ROOT / item["path"]) == item["sha256"], "下午退出冻结内容发生变化")
    write_json(OUT / "RUN_STARTED.json", {"started_at": now(), "config_sha256": digest(CONFIG)}, exclusive=True)
    data = pd.read_parquet(ROOT / cfg["features"])
    dividends = normalize_dividends(pd.read_csv(ROOT / cfg["dividends"]))
    models = json.loads((ROOT / cfg["saved_models"]).read_text(encoding="utf-8"))["models"]["D60_INTRA__RIDGE"]
    _, snapshots = load_snapshots()
    snapshots.to_parquet(OUT / "minute_snapshots.parquet")
    main, early, yearly, eras, stats = [], [], [], [], []
    for period, frame, start, dest in [("evaluation", data, cfg["evaluation_start"], main),
        ("earlier_diagnostic", data[data.date <= cfg["earlier_terminal"]].copy(), cfg["earlier_start"], early)]:
        for cost_id, cost in cfg["costs"].items():
            folder, accounts, names = OUT / period / cost_id, {}, {}
            folder.mkdir(parents=True, exist_ok=True)
            baseline = pd.read_parquet(PARENT / period / cost_id / "REARM_RIDGE_ledger.parquet")
            if period == "evaluation":
                daily = ExitController(frame, models, cfg["confirmation_days"])
                preview = AfternoonPreview(frame, models, daily, cfg["annual_days"])
                ledger, decisions, cycles, afternoon = simulate_afternoon_exit(frame, dividends, cfg, cost, start, make_rules(frame)["D60_INTRA"],
                    cfg["specification"], daily, preview, snapshots.to_dict("index"))
                save_account(folder, KEY, ledger, decisions)
                cycles.to_csv(folder / f"{KEY}_cycles.csv", index=False, encoding="utf-8-sig")
                afternoon.to_parquet(folder / "afternoon_decisions.parquet", index=False)
                afternoon.to_csv(folder / "afternoon_decisions.csv", index=False, encoding="utf-8-sig")
                require(ledger.accounting_error.abs().max() < 1e-6 and not ledger.terminal_unliquidated.iloc[-1], "下午退出账户结算失败")
                require(not ((ledger.filled_quantity > 0) & (ledger.shares_before > 0)).any(), "下午退出账户发生持仓中追加")
                require((cycles.dropna(subset=["exit_date"]).holding_intervals >= 1).all(), "下午退出违反买入次日可卖")
                signals = afternoon[afternoon.learned_exit_requested]
                require((pd.to_datetime(signals.learning_fit_time) <= pd.to_datetime(signals.signal_time)).all(), "下午信号使用了当日收盘之后的模型")
                stats.append({"period": period, "cost": cost_id, "holding_day_checks": len(afternoon), "afternoon_requests": len(signals),
                              "afternoon_fills": int((afternoon.filled_quantity < 0).sum()), "execution_statuses": afternoon.execution_status.value_counts().to_dict(),
                              "learning_statuses": afternoon.learning_status.value_counts().to_dict(), "earlier_is_original_fallback": False})
            else:
                require(not ((snapshots.index >= pd.Timestamp(start)) & (snapshots.index <= pd.Timestamp(cfg["earlier_terminal"]))).any(),
                        "较早历史出现了未预定的分钟数据，不能继续标记为纯回退")
                ledger = baseline.copy()
                ledger.to_parquet(folder / f"{KEY}_ledger.parquet", index=False)
                stats.append({"period": period, "cost": cost_id, "holding_day_checks": 0, "afternoon_requests": 0,
                              "afternoon_fills": 0, "earlier_is_original_fallback": True})
            accounts[KEY], names[KEY] = ledger, NAME
            accounts["REARM_RIDGE"], names["REARM_RIDGE"] = baseline, "原收盘学习退出候选"
            bh = pd.read_parquet(PARENT / period / cost_id / "BUY_HOLD_ledger.parquet")
            accounts["BUY_HOLD"], names["BUY_HOLD"] = bh, "买入持有"
            for key in ["REARM_RIDGE", "BUY_HOLD"]:
                accounts[key].to_parquet(folder / f"{key}_ledger.parquet", index=False)
            base = summarize(bh, cfg)
            for key, saved in accounts.items():
                require(pd.DatetimeIndex(saved.date).equals(pd.DatetimeIndex(bh.date)), "下午退出评价日历不完整")
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
            print(f"{period}／{cost_id}：下午退出或既定回退及两个原样对照完成。", flush=True)
    for filename, rows in [("metrics.csv", main), ("earlier_diagnostics.csv", early), ("yearly_metrics.csv", yearly), ("era_metrics.csv", eras)]:
        pd.DataFrame(rows).to_csv(OUT / filename, index=False, encoding="utf-8-sig")
    write_json(OUT / "result.json", {"study_id": cfg["study_id"], "completed_at": now(), "status": "AFTERNOON_LEARNED_EXIT_COMPLETE",
               "candidate_configurations": 1, "evaluation_accounts": 6, "new_accounts_generated": 2, "reused_control_accounts": 4,
               "earlier_diagnostic_accounts": 6, "new_earlier_diagnostic_accounts": 0, "reused_earlier_accounts": 6,
               "new_model_fits": 0, "all_metrics": main, "earlier_diagnostics": early, "afternoon_statistics": stats,
               "primary": [m for m in main if m["model"] == KEY],
               "post_selected_best_base": next(m for m in main if m["model"] == KEY and m["cost"] == "BASE"),
               "goal_achieved": False, "independent_validation": "NOT_ESTABLISHED", "position_impact": 0})
    print(json.dumps({"主评价": [{k: m[k] for k in ["cost", "net_sharpe", "annualized_return", "max_drawdown", "trade_count"]}
                                  for m in main if m["model"] == KEY], "下午行为": stats}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    import sys
    {"inventory": inventory, "freeze": freeze, "run": run}[sys.argv[1]]()
