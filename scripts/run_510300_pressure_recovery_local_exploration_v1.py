"""运行已登记的本地压力恢复探索；不发起网络请求，不生成交易指令。"""

import hashlib
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.pressure_recovery_local_exploration_v1 import (
    detect_price_episodes, hypothetical_price_cost, minute_measurements,
    next_day_proxies, quote_at, visible_order_quote, with_clock,
)
from research.pressure_recovery_v1 import commission, m1_measure, session

BASE = ROOT / "data/raw/510300_free_channels_v1/20261001"
OUT = ROOT / "reports/research/510300_pressure_recovery_v1/local_exploration_20261001"
CN = timezone(timedelta(hours=8))


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_json(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def stats(values):
    numbers = pd.Series(list(values), dtype=float).dropna()
    return {"n": len(numbers), "median": float(numbers.median()) if len(numbers) else None,
            "min": float(numbers.min()) if len(numbers) else None, "max": float(numbers.max()) if len(numbers) else None,
            "mean": float(numbers.mean()) if len(numbers) else None}


def export_table(name, rows, empty_columns):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty and len(frame.columns) == 0:
        frame = pd.DataFrame(columns=empty_columns)
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.12g")
    return frame


def add_scenarios(collection, identity, family, day, order, proxies):
    base = {"measurement_id": identity, "family": family, "trade_date": day,
            "nominal_budget_cny": order["nominal_budget_cny"], "quantity": order["quantity"],
            "hypothetical_entry_price_cny": order["buy_vwap_cny"],
            "next_trade_date": proxies["next_trade_date"], "actual_order": False,
            "verified_strategy_return": False, "scenario_status": "HYPOTHETICAL_PRICE_PATH_ONLY"}
    if not proxies["prices"]:
        collection.append(dict(base, scenario_status="NEXT_DAY_MINUTE_DATA_MISSING", proxy_label=None))
    for label, price in proxies["prices"].items():
        collection.append(dict(base, proxy_label=label, exit_minute_proxy_cny=price,
                               **hypothetical_price_cost(order["quantity"], order["buy_vwap_cny"], price)))


def make_chart(grid, events):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    days = sorted(grid.trade_date.unique())
    figure, axes = plt.subplots(3, len(days), figsize=(15, 9), sharex="col")
    settings = [("mid", 1, "中间价（元）", "#2563a6"),
                ("spread_ticks", 1, "买卖价差（最小价位数）", "#855723"),
                ("bid_depth_cny", 1e6, "下方10基点买盘（百万元）", "#27836b")]
    for column, day in enumerate(days):
        current = grid.loc[grid.trade_date.eq(day)]
        for row_number, (field, scale, label, color) in enumerate(settings):
            axis = axes[row_number, column]
            for _, part in current.groupby("session", sort=False):
                axis.plot(part.decision_at, part[field] / scale, linewidth=1.05, color=color)
            for event in events:
                if event["trade_date"] == day:
                    axis.axvspan(event["shock_at"], event["observation_end_at"], color="#b64f43", alpha=0.13)
                    axis.axvline(event["shock_at"], color="#b64f43", linewidth=0.55, alpha=0.7)
            axis.set_ylabel(label, fontsize=9)
            axis.grid(alpha=0.17)
            axis.xaxis.set_major_locator(mdates.HourLocator(interval=1, tz=CN))
            axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=CN))
            axis.tick_params(labelsize=8)
            if row_number == 0:
                axis.set_title(day, fontsize=12)
    figure.suptitle("510300 已存盘口：价格、价差与可见深度", fontsize=16)
    figure.text(0.5, 0.025, "红线：固定五分钟跌幅达到5基点的探索事件；浅红区：其后固定五分钟观察期。午休留空。均非M2交易信号。", ha="center", fontsize=10)
    figure.tight_layout(rect=(0, 0.055, 1, 0.95))
    figure.savefig(OUT / "三日盘口与固定观察窗口.png", dpi=150)
    plt.close(figure)


def main():
    if (OUT / "summary.json").exists():
        raise SystemExit("本批结果已存在，不能用新参数覆盖。")
    configuration = ROOT / "config/510300_pressure_recovery_local_exploration_v1.json"
    config = json.loads(configuration.read_text(encoding="utf-8"))
    registration = json.loads((OUT / "registration_receipt.json").read_text(encoding="utf-8"))
    for item in registration["files"]:
        if hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise SystemExit("已登记输入发生变化：" + item["path"])
    minute_prices = pd.read_parquet(BASE / "neigezhu/data/etf_1m/SH/510300.parquet")
    calendar = pd.read_csv(ROOT / "data/reference/sse_trade_calendar_2026.csv")
    trading_days = sorted(calendar.trade_date.astype(str).unique().tolist())
    grids, events, visible_orders, scenarios, closes, daily = [], [], [], [], [], []
    for compact_day in config["source_dates"]:
        day = f"{compact_day[:4]}-{compact_day[4:6]}-{compact_day[6:]}"
        folders = sorted((BASE / "venvoo_selected").glob(compact_day + "_*"))
        complete = [folder for folder in folders if all((folder / (name + ".parquet")).exists() for name in ("行情", "逐笔委托", "逐笔成交"))]
        if len(complete) != 1:
            raise SystemExit("目标日期没有唯一的已完成三流原值子集：" + compact_day)
        quotes, orders, trades = [with_clock(pd.read_parquet(complete[0] / (name + ".parquet"))) for name in ("行情", "逐笔委托", "逐笔成交")]
        grid = minute_measurements(quotes, orders, trades, day)
        grids.append(grid)
        current_events = detect_price_episodes(grid, threshold=config["exploratory_price_episode"]["shock_log_mid_return_bps_at_most"])
        proxies = next_day_proxies(day, minute_prices, trading_days)
        for row in grid.itertuples(index=False):
            raw = quote_at(quotes, row.decision_at)
            if raw is not None and row.measurement_status == "SOURCE_QUOTE_MEASUREMENT_ONLY":
                for budget in config["hypothetical_entry"]["nominal_budgets_cny"]:
                    visible_orders.append(dict(trade_date=day, decision_at=row.decision_at, **visible_order_quote(raw, budget)))
        for event in current_events:
            entry = quote_at(quotes, event["observation_end_at"], after=True) if event["observation_status"] == "OBSERVED_FIXED_5_MINUTES" else None
            event["hypothetical_entry_status"] = "NO_POST_OBSERVATION_QUOTE"
            if entry is not None:
                entry_mid = float((entry.bid_px1 + entry.ask_px1) / 20000)
                event.update(hypothetical_entry_status="QUOTE_AFTER_OBSERVATION_ONLY", hypothetical_entry_quote_at=entry._at, entry_mid_cny=entry_mid)
                for horizon in config["price_path_diagnostics"]["intraday_minutes_after_entry_quote"]:
                    target = entry._at + pd.Timedelta(minutes=horizon)
                    future = quote_at(quotes, target)
                    same_session = session(target) == session(entry._at) and target.date() == entry._at.date()
                    if future is not None and same_session:
                        future_mid = float((future.bid_px1 + future.ask_px1) / 20000)
                        event[f"post_entry_mid_path_{horizon}m_bps"] = (future_mid / entry_mid - 1) * 10000
                    else:
                        event[f"post_entry_mid_path_{horizon}m_bps"] = None
                for budget in config["hypothetical_entry"]["nominal_budgets_cny"]:
                    order = visible_order_quote(entry, budget)
                    if order["status"] == "VISIBLE_QUOTE_HYPOTHESIS_ONLY":
                        add_scenarios(scenarios, event["event_id"], "P0_EXPLORATORY_EPISODE", day, order, proxies)
            events.append(event)
        regular = quotes.loc[quotes.time.lt(150100000) & quotes.price.gt(0)]
        close = float(regular.iloc[-1].price / 10000)
        formal = m1_measure({"trade_date": day, "price": close})
        record = {"trade_date": day, "close_cny": close, "formal_m1_status": formal["status"],
                  "formal_m1_reasons": "|".join(formal["reasons"]), "reference_value_status": "MISSING_IOPV_ALL_ZERO",
                  "actual_order": False, "next_trade_date": proxies["next_trade_date"], "next_day_proxy_status": proxies["status"]}
        by_time = grid.set_index("decision_at")
        for clock in ("14:51", "14:56"):
            item = by_time.loc[pd.Timestamp(day + " " + clock, tz="Asia/Shanghai")]
            record[clock + "_spread_bps"] = item.spread_bps
            record[clock + "_bid_depth_cny"] = item.bid_depth_cny
        after = trades.loc[trades.time.ge(150600000) & trades.time.le(153000000)]
        record.update(after_1506_source_trade_rows=len(after), after_1506_source_volume=float(after.volume.sum()) if len(after) else None,
                      after_1506_coverage_status="SOURCE_RECORDS_PRESENT_NOT_COMPLETE_QUEUE" if len(after) else "NO_SOURCE_RECORDS_NOT_ZERO_MARKET_VOLUME",
                      after_1506_prices=sorted((after.price / 10000).unique().tolist()))
        for label, price in proxies["prices"].items():
            record[label + "_price_cny"] = price
            record[label + "_gross_price_path_bps"] = (price / close - 1) * 10000
        closes.append(record)
        for budget in config["hypothetical_entry"]["nominal_budgets_cny"]:
            quantity = int((budget - 5) // (close * 100)) * 100
            while quantity > 0 and close * quantity + commission(close * quantity) > budget:
                quantity -= 100
            add_scenarios(scenarios, "CLOSE_PATH_" + compact_day, "UNCONDITIONAL_CLOSE_PRICE_PATH_NOT_M1_SIGNAL", day,
                          {"nominal_budget_cny": budget, "quantity": quantity, "buy_vwap_cny": close}, proxies)
        day_orders = [item for item in visible_orders if item["trade_date"] == day and item["status"] == "VISIBLE_QUOTE_HYPOTHESIS_ONLY"]
        daily.append({"trade_date": day, "minute_grid_rows": len(grid), "valid_quote_rows": int(grid.measurement_status.eq("SOURCE_QUOTE_MEASUREMENT_ONLY").sum()),
                      "eligible_5m_price_windows": int(grid.log_mid_return_5m_bps.notna().sum()),
                      "threshold_hit_minutes": int(grid.log_mid_return_5m_bps.le(-5).sum()), "episodes": len(current_events),
                      "spread_bps": stats(grid.spread_bps), "bid_depth_cny": stats(grid.bid_depth_cny),
                      "quoted_cost_by_budget": {str(budget): {"immediate_roundtrip_cost_bps": stats(item["same_snapshot_roundtrip_cost_bps"] for item in day_orders if item["nominal_budget_cny"] == budget),
                                                              "next_day_proxy_required_rise_bps": stats(item["next_day_proxy_required_rise_from_entry_mid_bps"] for item in day_orders if item["nominal_budget_cny"] == budget)} for budget in (20000, 200000)}})
        print(f"已完成{day}：{len(grid)}个分钟时点、{len(current_events)}个固定价格压力事件。", flush=True)
    full_grid = pd.concat(grids, ignore_index=True)
    full_grid.to_parquet(OUT / "01_盘口与消息测量表.parquet", index=False)
    export_table("01_盘口与消息测量表.csv", full_grid, [])
    event_frame = export_table("02_固定观察事件表.csv", events, ["event_id", "observation_status", "formal_m2_status"])
    export_table("03_可见盘口数量与费用表.csv", visible_orders, ["trade_date", "status"])
    export_table("04_跨日价格费用假设表.csv", scenarios, ["measurement_id", "scenario_status", "verified_strategy_return"])
    export_table("收盘与次日价格路径.csv", closes, ["trade_date", "formal_m1_status"])
    summary = {"computed_at": datetime.now(CN).isoformat(), "status": "LOCAL_EXPLORATORY_MEASUREMENT_COMPLETED",
               "new_network_requests": 0, "source_days": len(daily), "minute_measurements": len(full_grid),
               "eligible_5m_price_windows": sum(item["eligible_5m_price_windows"] for item in daily),
               "exploratory_events": len(events), "observed_fixed_endpoints": sum(e["observation_status"] == "OBSERVED_FIXED_5_MINUTES" for e in events),
               "missing_fixed_endpoints": sum(e["observation_status"] != "OBSERVED_FIXED_5_MINUTES" for e in events),
               "observation_price_increases": sum(e.get("observation_price_change_bps", float("nan")) > 0 for e in events),
               "observation_price_nonincreases": sum(e.get("observation_price_change_bps", float("nan")) <= 0 for e in events),
               "spread_narrowed_events": sum(e.get("spread_label") == "收窄" for e in events),
               "bid_depth_increased_events": sum(e.get("bid_depth_ratio", 0) > 1 for e in events),
               "post_entry_5m_mid_price_path_bps": stats(e.get("post_entry_mid_path_5m_bps") for e in events),
               "post_entry_30m_mid_price_path_bps": stats(e.get("post_entry_mid_path_30m_bps") for e in events),
               "independent_day_count_for_episode_outcomes": len({e["trade_date"] for e in events}),
               "qualified_m1_signals": 0, "qualified_m2_signals": 0, "actual_orders": 0, "verified_fills": 0,
               "strategy_net_expectancy": "NOT_COMPUTED", "strategy_win_rate": "NOT_COMPUTED", "full_account_sharpe": "NOT_COMPUTED",
               "goal_achieved": False, "daily": daily, "close_paths": closes}
    save_json(OUT / "summary.json", summary)
    make_chart(full_grid, events)
    size = sum(path.stat().st_size for path in OUT.rglob("*") if path.is_file())
    if size > config["reporting"]["storage_of_new_results_limit_bytes"]:
        raise SystemExit("结果目录超过已登记体积上限，请检查原因；原始数据未删除。")
    print(json.dumps(clean({k: v for k, v in summary.items() if k not in ("daily", "close_paths")}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
