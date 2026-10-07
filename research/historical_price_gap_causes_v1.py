"""历史价差变化的原因分组与事件收益；不创建前瞻判断或完整账户。"""

from __future__ import annotations

import json
from datetime import datetime
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_price_gap_causes_v1"
PRICE = ROOT / "reports/research/510300_manufacturing_price_transmission_source_v1/released_manufacturing_price_transmission.parquet"
ORDERS = ROOT / "data/raw/macro/510300_macro_stress_2015_v2/pmi_new_orders_release_vintage_2015_2026.parquet"
MARKET = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet"
DIVIDENDS = ROOT / "data/reference/510300_dividends.csv"
LABELS = {
    "SELLING_STRENGTH": "出厂指数改善更强",
    "COST_DOWN_OUTPUT_NOT_DOWN": "购进回落、出厂不降",
    "BOTH_DOWN": "购进、出厂指数同降",
    "NO_GAP_IMPROVEMENT": "价差未改善",
}
PHASES = {
    "EARLIER_CONTEXT": "2022年5月—2023年",
    "PRIMARY": "2024年—2026年7月",
}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def bp_change(a, b):
    return float(Decimal(str(a)) - Decimal(str(b)))


def group_for(delta_output, delta_input):
    if Decimal(str(delta_output)) - Decimal(str(delta_input)) <= 0:
        return "NO_GAP_IMPROVEMENT"
    if delta_input >= 0:
        return "SELLING_STRENGTH"
    if delta_output >= 0:
        return "COST_DOWN_OUTPUT_NOT_DOWN"
    return "BOTH_DOWN"


def round_trip(entry_open, exit_open, dividends_per_share, costs):
    tick = Decimal(str(costs["tick"]))
    slip = Decimal(str(costs["slippage_fraction_each_side"]))
    commission = Decimal(str(costs["commission_rate"]))
    minimum = Decimal(str(costs["minimum_commission_cny"]))
    budget = Decimal(str(costs["illustrative_event_budget_cny"]))
    lot = costs["lot_size"]
    buy = (Decimal(str(entry_open)) * (1 + slip) / tick).to_integral_value(rounding=ROUND_CEILING) * tick
    sell = (Decimal(str(exit_open)) * (1 - slip) / tick).to_integral_value(rounding=ROUND_FLOOR) * tick
    shares = int((budget / buy / lot).to_integral_value(rounding=ROUND_FLOOR)) * lot
    while shares > 0 and buy * shares + max(minimum, buy * shares * commission) > budget:
        shares -= lot
    if shares <= 0:
        raise ValueError("事件预算不足一手。")
    entry_commission = max(minimum, buy * shares * commission)
    exit_commission = max(minimum, sell * shares * commission)
    paid = buy * shares + entry_commission
    dividend = Decimal(str(dividends_per_share)) * shares
    pnl = sell * shares - exit_commission + dividend - paid
    return {
        "buy_price": float(buy), "sell_price": float(sell), "shares": shares,
        "paid_cny": float(paid), "commissions_cny": float(entry_commission + exit_commission),
        "dividend_entitlement_cny": float(dividend), "net_pnl_cny": float(pnl),
        "net_return": float(pnl / paid),
    }


def describe(frame, horizon):
    values = frame[f"net_return{horizon}"].astype(float)
    n = len(values)
    if not n:
        return {"n": 0, "mean_net_pct": None, "median_net_pct": None, "positive_pct": None,
                "worst_net_pct": None, "best_net_pct": None, "mean_without_best_pct": None,
                "best_month": None, "worst_month": None, "mean_gross_pct": None}
    return {
        "n": n, "mean_net_pct": float(values.mean() * 100),
        "median_net_pct": float(values.median() * 100), "positive_pct": float((values > 0).mean() * 100),
        "worst_net_pct": float(values.min() * 100), "best_net_pct": float(values.max() * 100),
        "mean_without_best_pct": float(values.drop(values.idxmax()).mean() * 100) if n > 1 else None,
        "best_month": str(frame.loc[values.idxmax(), "stat_month"]),
        "worst_month": str(frame.loc[values.idxmin(), "stat_month"]),
        "mean_gross_pct": float(frame[f"gross_return{horizon}"].mean() * 100),
    }


def plot(events, summary):
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"axes.unicode_minus": False, "font.family": font.get_name(), "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.1, 1.35]})
    fig.patch.set_facecolor("#f5f3ed")
    main = events[events.phase.eq("PRIMARY")]
    colors = {True: "#177f77", False: "#b06c48"}
    for confirmed in [False, True]:
        x = main[main.order_confirmed.eq(confirmed)]
        axes[0].scatter(x.delta_input, x.delta_output, s=62, c=colors[confirmed],
                        edgecolors="white", linewidths=.7,
                        label="订单≥50且改善" if confirmed else "未获该订单条件确认")
    limits = [-8, 10]
    axes[0].plot(limits, limits, color="#85857f", lw=1, ls="--")
    axes[0].axhline(0, color="#bbb9b1", lw=.8)
    axes[0].axvline(0, color="#bbb9b1", lw=.8)
    for month in ["2024-06", "2024-08", "2026-03", "2026-06"]:
        row = main[main.stat_month.eq(month)].iloc[0]
        axes[0].annotate(month, (row.delta_input, row.delta_output), xytext=(7, 6), textcoords="offset points", fontsize=9)
    axes[0].set(xlabel="购进价格指数月度变化（点）", ylabel="出厂价格指数月度变化（点）",
                title="同样的价差改善，可以有不同来源")
    axes[0].text(.04, .96, "虚线上方：出厂减购进的价差改善", transform=axes[0].transAxes, va="top", fontsize=10)
    axes[0].legend(loc="lower right", fontsize=9, frameon=False)
    positions = np.arange(len(LABELS))
    for offset, phase, color in [(-.18, "EARLIER_CONTEXT", "#8fa5b5"), (.18, "PRIMARY", "#177f77")]:
        means, counts = [], []
        for group in LABELS:
            row = next(r for r in summary if r["phase"] == phase and r["group"] == group and r["horizon"] == 20)
            means.append(row["mean_net_pct"] if row["n"] else 0.)
            counts.append(row["n"])
        bars = axes[1].barh(positions + offset, means, height=.31, color=color, label=PHASES[phase])
        for bar, val, count in zip(bars, means, counts):
            axes[1].text(val + (.15 if val >= 0 else -.15), bar.get_y() + bar.get_height()/2,
                         f"{val:+.2f}% · n={count}" if count else "无样本", va="center",
                         ha="left" if val >= 0 else "right", fontsize=9)
    axes[1].set_yticks(positions, list(LABELS.values()))
    axes[1].axvline(0, color="#9f9d94", lw=.8)
    axes[1].set(xlabel="20个交易日事件净收益均值（%）", title="下一交易日开盘后：阶段差异")
    axes[1].margins(x=.45)
    axes[1].legend(loc="lower right", fontsize=9, frameon=False)
    for ax in axes:
        ax.set_facecolor("#f5f3ed")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#c3c0b7")
        ax.grid(axis="x", alpha=.12)
    fig.suptitle("510300 历史发现｜先解释变化，再看公布后的收益", fontsize=17, x=.04, ha="left")
    fig.text(.04, .025, "逐事件往返算例，含压力费用；非完整账户。分组样本少，20日窗口可能重叠；未按收益挑选阶段。", fontsize=10, color="#66645d")
    fig.tight_layout(rect=(0, .065, 1, .92))
    fig.savefig(OUT / "历史价差来源与阶段收益.png", dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    if (OUT / "event_results.json").exists():
        raise SystemExit("本轮历史结果已保存，不覆盖原结果。")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    prices = pd.read_parquet(PRICE).sort_values("stat_month").set_index("stat_month", drop=False)
    orders = pd.read_parquet(ORDERS).set_index("reference_period")
    market = pd.read_parquet(MARKET).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).dt.normalize()
    dividends = pd.read_csv(DIVIDENDS)
    dividends["record_date"] = pd.to_datetime(dividends.record_date)
    assert market.date.is_unique and prices.index.is_unique and orders.index.is_unique
    assert market.open.gt(0).all()
    start, end = protocol["sample_stat_months"]
    expected = [str(x) for x in pd.period_range(start, end, freq="M")]
    rows = []
    for month in expected:
        prior_month = str(pd.Period(month, freq="M") - 1)
        r, prev = prices.loc[month], prices.loc[prior_month]
        assert r.method_group == prev.method_group == "GB2017_31IND_3200FIRMS"
        known = pd.Timestamp(r.known_at)
        assert pd.Timestamp(orders.loc[month, "available_at"]) <= known
        assert pd.Timestamp(prev.known_at) < known
        known_date = known.tz_localize(None).normalize()
        entry_idx = int(market.date.searchsorted(known_date, side="right"))
        if entry_idx + max(protocol["horizons_trading_days"]) >= len(market):
            raise ValueError(f"历史月份尚无完整结果：{month}；不替换月份。")
        entry = market.iloc[entry_idx]
        delta_output = bp_change(r.output_price_diffusion, prev.output_price_diffusion)
        delta_input = bp_change(r.input_price_diffusion, prev.input_price_diffusion)
        delta_orders = bp_change(orders.loc[month, "first_release_value"], orders.loc[prior_month, "first_release_value"])
        group = group_for(delta_output, delta_input)
        row = {"stat_month": month, "year": int(month[:4]), "phase": "PRIMARY" if month >= protocol["primary_stat_months"][0] else "EARLIER_CONTEXT",
               "known_at": known.isoformat(), "source_url": r.url, "source_path": r.raw_path,
               "input_index": float(r.input_price_diffusion), "output_index": float(r.output_price_diffusion),
               "delta_input": delta_input, "delta_output": delta_output,
               "delta_gap": bp_change(delta_output, delta_input), "new_orders": float(orders.loc[month, "first_release_value"]),
               "delta_orders": delta_orders, "order_confirmed": bool(orders.loc[month, "first_release_value"] >= 50 and delta_orders > 0),
               "group": group, "group_name": LABELS[group], "entry_idx": entry_idx,
               "entry_date": entry.date.strftime("%Y-%m-%d"), "entry_open": float(entry.open)}
        for horizon in protocol["horizons_trading_days"]:
            exit_row = market.iloc[entry_idx + horizon]
            entitled = dividends[dividends.record_date.ge(entry.date) & dividends.record_date.lt(exit_row.date)]
            div = float(entitled.cash_dividend_per_share.sum())
            result = round_trip(entry.open, exit_row.open, div, protocol["costs"])
            row.update({f"exit_date{horizon}": exit_row.date.strftime("%Y-%m-%d"),
                        f"exit_open{horizon}": float(exit_row.open), f"dividend_per_share{horizon}": div,
                        f"gross_return{horizon}": (float(exit_row.open) + div) / float(entry.open) - 1})
            row.update({f"{key}{horizon}": value for key, value in result.items()})
            assert result["paid_cny"] <= protocol["costs"]["illustrative_event_budget_cny"]
            assert result["net_return"] <= row[f"gross_return{horizon}"] + 1e-12
        rows.append(row)
    events = pd.DataFrame(rows)
    assert events.stat_month.tolist() == expected
    events.to_csv(OUT / "全部历史事件.csv", index=False, encoding="utf-8-sig")
    summaries, cross, annual = [], [], []
    for phase in ["ALL", *PHASES]:
        sample = events if phase == "ALL" else events[events.phase.eq(phase)]
        for group in ["ALL", *LABELS]:
            selected = sample if group == "ALL" else sample[sample.group.eq(group)]
            for horizon in protocol["horizons_trading_days"]:
                summaries.append({"phase": phase, "group": group, "group_name": LABELS.get(group, "全部月份"), "horizon": horizon, **describe(selected, horizon)})
                if group != "ALL":
                    for confirmed in [False, True]:
                        part = selected[selected.order_confirmed.eq(confirmed)]
                        cross.append({"phase": phase, "group": group, "order_confirmed": confirmed,
                                      "horizon": horizon, **describe(part, horizon)})
    for year, sample in events.groupby("year"):
        for group in ["ALL", *LABELS]:
            selected = sample if group == "ALL" else sample[sample.group.eq(group)]
            annual.append({"year": int(year), "group": group, "horizon": 20, **describe(selected, 20)})
    pd.DataFrame(summaries).to_csv(OUT / "阶段与构成汇总.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cross).to_csv(OUT / "订单条件交叉汇总.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(annual).to_csv(OUT / "逐年描述.csv", index=False, encoding="utf-8-sig")
    overlap_pairs = sum(events.entry_idx.iloc[i] < events.entry_idx.iloc[i-1] + 20 for i in range(1, len(events)))
    selected_cases = events[events.stat_month.isin(["2024-06", "2026-06", "2024-08", "2026-03"])]
    result = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "study_id": protocol["study_id"], "status": "COMPLETED_HISTORICAL_EVENT_DISCOVERY_NOT_FULL_ACCOUNT",
        "events": len(events), "primary_events": int(events.phase.eq("PRIMARY").sum()),
        "earlier_context_events": int(events.phase.eq("EARLIER_CONTEXT").sum()),
        "market_cutoff": market.date.max().strftime("%Y-%m-%d"),
        "overlapping_adjacent_20d_windows": int(overlap_pairs),
        "summaries": summaries, "order_conditions": cross, "annual": annual,
        "selected_cases": selected_cases.to_dict("records"),
        "event_returns_are_strategy_performance": False, "historical_data_previously_used": True,
        "new_accounts": 0, "new_strategy_return_tests": 0, "new_historical_event_studies": 1,
        "new_fitted_parameters": 0, "new_forecast_cards": 0,
        "net_sharpe": None, "net_cagr": None, "goal_achieved": False,
    }
    save(OUT / "event_results.json", result)
    plot(events, summaries)
    print("历史事件研究完成：", len(events), "个月；主段", result["primary_events"], "个月。")
    table = pd.DataFrame(summaries)
    print(table[table.phase.eq("PRIMARY") & table.horizon.eq(20)][["group_name", "n", "mean_net_pct", "median_net_pct", "positive_pct", "mean_without_best_pct"]].to_string(index=False))
    print("预先选择的说明案例：")
    print(selected_cases[["stat_month", "delta_gap", "new_orders", "delta_orders", "entry_date", "exit_date20", "gross_return20", "net_return20"]].to_string(index=False))


if __name__ == "__main__":
    main()
