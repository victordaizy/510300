"""复盘七次历史降息的工具内容、公布时点与可成交收益。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

from historical_price_gap_causes_v1 import round_trip


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_historical_policy_constraint_events_v1"
RATE = ROOT / "data/curated/510300_asymmetric_stress_hazard_v1_source_remediation_v1_0_1/pboc_7d_reverse_repo_rate_changes_anchor_20150105_20260814.parquet"
CLOCK = ROOT / "reports/research/510300_policy_information_clock_v1"
OLD_POLICY = ROOT / "reports/research/510300_equity_support_policy_event_v1"
ORDERS = ROOT / "data/raw/macro/510300_macro_stress_2015_v2/pmi_new_orders_release_vintage_2015_2026.parquet"
MARKET = ROOT / "reports/research/510300_macro_dynamic_reframe_v1/inputs/market.parquet"
DIVIDENDS = ROOT / "data/reference/510300_dividends.csv"
TYPE_NAMES = {
    "GENERAL_RATE_CUT": "一般政策利率下调",
    "NEW_EQUITY_FINANCING_CHANNEL": "新设股票融资工具",
    "EXISTING_EQUITY_TOOL_TERMS_OPTIMIZED": "优化既有股票工具",
}


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def dividend_between(dividends, start, end):
    selected = dividends[dividends.record_date.ge(start) & dividends.record_date.lt(end)]
    return float(selected.cash_dividend_per_share.sum())


def summarize(frame, group, horizon):
    values = frame[f"net_return{horizon}"]
    return {
        "group": group, "horizon": horizon, "n": len(frame),
        "mean_net_pct": float(values.mean() * 100),
        "median_net_pct": float(values.median() * 100),
        "positive_count": int(values.gt(0).sum()),
        "mean_without_best_pct": float(values.drop(values.idxmax()).mean() * 100) if len(values) > 1 else None,
        "best_event": str(frame.loc[values.idxmax(), "announcement_date"]),
        "worst_event": str(frame.loc[values.idxmin(), "announcement_date"]),
    }


def plot(events):
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    plt.rcParams.update({"font.family": font.get_name(), "axes.unicode_minus": False, "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    background = "#f5f3ed"
    fig.patch.set_facecolor(background)
    y = np.arange(len(events))
    colors = ["#187b73" if value == "EQUITY_FINANCING_LINKED" else "#879ca9" for value in events.group]
    for ax, field, title in [
        (axes[0], "already_reflected_return", "公告前一收盘 → 下一开盘：已经发生"),
        (axes[1], "net_return20", "下一开盘 → 20个交易日后：费用后剩余收益"),
    ]:
        values = events[field].to_numpy() * 100
        ax.barh(y, values, height=.6, color=colors)
        for i, value in enumerate(values):
            ax.text(value + (.2 if value >= 0 else -.2), i, f"{value:+.2f}%", ha="left" if value >= 0 else "right", va="center")
        ax.axvline(0, color="#a29f96", lw=.8)
        ax.set(title=title, xlabel="区间收益（%）", facecolor=background)
        ax.margins(x=.3)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#c3c0b7")
        ax.grid(axis="x", alpha=.12)
    axes[0].set_yticks(y, [f"{r.announcement_date}  {r.policy_type_name}" for r in events.itertuples()])
    axes[0].invert_yaxis()
    fig.suptitle("510300 历史降息｜改变了什么约束，公告之后还剩多少收益", x=.035, ha="left", fontsize=17)
    fig.text(.035, .025, "7次既有历史事件；绿色为股票融资工具相关的2次。两段口径不同，不能相加；事件收益不代表账户夏普或政策因果效应。", fontsize=10, color="#68645b")
    fig.tight_layout(rect=(0, .065, 1, .92))
    fig.savefig(OUT / "七次政策的已发生反应与剩余收益.png", dpi=180, facecolor=background)
    plt.close(fig)


def main():
    if (OUT / "event_results.json").exists():
        raise SystemExit("已有本轮历史结果，不覆盖原结果。")
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    costs = json.loads((ROOT / protocol["costs_inherited_from"]).read_text(encoding="utf-8"))["costs"]
    rate = pd.read_parquet(RATE).sort_values("notice_date")
    rate["record_date_text"] = pd.to_datetime(rate.notice_date).dt.strftime("%Y-%m-%d")
    rate = rate[rate.record_date_text.between("2022-01-01", "2026-07-31") & rate.seven_day_rate_percent.lt(rate.previous_rate_percent)]
    assert rate.record_date_text.tolist() == protocol["expected_observed_change_dates"]
    facts = pd.read_csv(CLOCK / "results/跨通道政策链_完整事实与时钟.csv").set_index("node_id")
    child_facts = pd.read_csv(OLD_POLICY / "inputs/policy_facts.csv").set_index("node_id")
    orders = pd.read_parquet(ORDERS).sort_values("reference_period").reset_index(drop=True)
    orders["known"] = pd.to_datetime(orders.available_at, utc=True)
    market = pd.read_parquet(MARKET).sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date).dt.normalize()
    dividends = pd.read_csv(DIVIDENDS)
    dividends["record_date"] = pd.to_datetime(dividends.record_date)
    assert market.date.is_unique and orders.reference_period.is_unique and market.open.gt(0).all()
    rows, sources = [], []
    for r in rate.itertuples():
        record_date = r.record_date_text
        repair = protocol["announcement_date_repairs"].get(record_date)
        announcement = repair["announcement_date"] if repair else record_date
        policy_type = repair["policy_type"] if repair else "GENERAL_RATE_CUT"
        group = "EQUITY_FINANCING_LINKED" if repair else "GENERAL_RATE_CUT"
        decision_end = pd.Timestamp(announcement + "T23:59:59+08:00")
        sources.append({"event": announcement, "purpose": "利率记录", "url": r.source_url, "path": r.raw_path})
        source_upper = pd.Timestamp(r.published_at)
        if source_upper.tzinfo is None:
            source_upper = source_upper.tz_localize("Asia/Shanghai")
        if repair:
            nodes = [repair["rate_node"], repair["equity_node"]]
            source_times = []
            for node in nodes:
                f = child_facts.loc[node] if node == "C07" else facts.loc[node]
                source_times.append(pd.Timestamp(f.source_available_upper))
                source_file = (OLD_POLICY / f.local_source_path) if node == "C07" else (CLOCK / f.source_path)
                sources.append({"event": announcement, "purpose": node, "url": f.source_url, "path": source_file.relative_to(ROOT).as_posix()})
            source_upper = max(source_times)
            if "implementation_node" in repair:
                f = facts.loc[repair["implementation_node"]]
                sources.append({"event": announcement, "purpose": "实施日核对", "url": f.source_url, "path": (CLOCK / f.source_path).relative_to(ROOT).as_posix()})
        assert source_upper <= decision_end
        # 订单只取政策公布之前已经可用的月份，不能使用随后公布的当月值。
        known_orders = orders[orders.known.le(source_upper.tz_convert("UTC"))]
        current_order, prior_order = known_orders.iloc[-1], known_orders.iloc[-2]
        day = pd.Timestamp(announcement)
        prior_idx = int(market.date.searchsorted(day, side="left")) - 1
        entry_idx = int(market.date.searchsorted(day, side="right"))
        assert prior_idx >= 59 and entry_idx + max(protocol["horizons"]) < len(market)
        previous, entry = market.iloc[prior_idx], market.iloc[entry_idx]
        pre_dividend = dividend_between(dividends, previous.date, entry.date)
        row = {
            "announcement_date": announcement, "record_date": record_date,
            "implementation_date": repair["implementation_date"] if repair else record_date,
            "source_upper": source_upper.isoformat(), "decision_end": decision_end.isoformat(),
            "group": group, "policy_type": policy_type, "policy_type_name": TYPE_NAMES[policy_type],
            "old_rate_percent": float(r.previous_rate_percent), "new_rate_percent": float(r.seven_day_rate_percent),
            "cut_bp": round((r.previous_rate_percent - r.seven_day_rate_percent) * 100, 6),
            "prior_order_month": str(current_order.reference_period), "prior_order_available_at": str(current_order.available_at),
            "new_orders": float(current_order.first_release_value),
            "delta_orders": round(float(current_order.first_release_value - prior_order.first_release_value), 6),
            "prior_close_date": previous.date.strftime("%Y-%m-%d"), "prior_close": float(previous.close),
            "prior_20d_total_return": float(previous.wealth / market.iloc[prior_idx - 20].wealth - 1),
            "prior_60d_drawdown": float(previous.wealth / market.iloc[prior_idx - 59:prior_idx + 1].wealth.max() - 1),
            "entry_date": entry.date.strftime("%Y-%m-%d"), "entry_open": float(entry.open),
            "already_reflected_dividend_per_share": pre_dividend,
            "already_reflected_return": float((entry.open + pre_dividend) / previous.close - 1),
        }
        sources.append({"event": announcement, "purpose": "当时已公布订单", "url": current_order.source_url, "path": current_order.raw_path})
        for horizon in protocol["horizons"]:
            exit_row = market.iloc[entry_idx + horizon]
            div = dividend_between(dividends, entry.date, exit_row.date)
            result = round_trip(entry.open, exit_row.open, div, costs)
            row.update({f"exit_date{horizon}": exit_row.date.strftime("%Y-%m-%d"),
                        f"exit_open{horizon}": float(exit_row.open), f"dividend_per_share{horizon}": div,
                        f"gross_return{horizon}": float((exit_row.open + div) / entry.open - 1)})
            row.update({f"{key}{horizon}": value for key, value in result.items()})
            assert result["paid_cny"] <= costs["illustrative_event_budget_cny"]
            assert result["net_return"] <= row[f"gross_return{horizon}"]
        if repair:
            late_idx = int(market.date.searchsorted(pd.Timestamp(record_date), side="right"))
            late_entry = market.iloc[late_idx]
            exit_row = market.iloc[entry_idx + protocol["primary_horizon"]]
            late_div = dividend_between(dividends, late_entry.date, exit_row.date)
            late_result = round_trip(late_entry.open, exit_row.open, late_div, costs)
            row.update({"late_entry_date": late_entry.date.strftime("%Y-%m-%d"),
                        "late_entry_open": float(late_entry.open), "late_same_exit_date": exit_row.date.strftime("%Y-%m-%d"),
                        "late_gross_return": float((exit_row.open + late_div) / late_entry.open - 1),
                        "late_net_return": late_result["net_return"],
                        "late_remaining_sessions": entry_idx + protocol["primary_horizon"] - late_idx})
        rows.append(row)
    events = pd.DataFrame(rows)
    summaries = []
    for group in ["ALL", "GENERAL_RATE_CUT", "EQUITY_FINANCING_LINKED"]:
        selected = events if group == "ALL" else events[events.group.eq(group)]
        for horizon in protocol["horizons"]:
            summaries.append(summarize(selected, group, horizon))
    for source in sources:
        assert (ROOT / source["path"]).is_file(), f"缺少原文：{source['path']}"
    events.to_csv(OUT / "七次降息与剩余历史收益.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summaries).to_csv(OUT / "按政策内容汇总.csv", index=False, encoding="utf-8-sig")
    save(OUT / "来源路径.json", sources)
    result = {
        "recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "study_id": protocol["study_id"],
        "status": "COMPLETED_HISTORICAL_DIAGNOSTIC_NO_NEW_ACCOUNT", "events": len(events),
        "costs": costs, "summaries": summaries, "source_path_count": len(sources),
        "market_cutoff": market.date.max().strftime("%Y-%m-%d"),
        "event_returns_are_full_account_performance": False, "independent_validation": False,
        "new_accounts": 0, "new_fitted_parameters": 0, "new_forecast_cards": 0,
        "net_sharpe": None, "net_cagr": None, "goal_achieved": False,
        "limitations": [
            "7次事件属于现有利率记录链，不是全部政策的完整目录。",
            "两次股票融资相关事件已经用于旧研究；分组不是独立盲测。",
            "相邻数据可在当时发布日期使用，但未证明保存网页均为不可变首发版本。",
            "2024年9月26日及2025年5月12日还有后续政策；整段收益不能归于最初单一政策。",
            "没有同口径事前利率预期调查，不把实际降幅当作意外幅度。",
            "订单和既有价格只是状态坐标，不充分识别政策原因或现金流变化。",
            "每行10万元往返算例，不执行完整账户仓位及风险预算，不据此计算夏普。",
        ],
    }
    save(OUT / "event_results.json", result)
    plot(events)
    print("已完成7次历史政策内容与公布时点复盘；未创建账户或前瞻判断。")
    print(events[["announcement_date", "policy_type_name", "prior_order_month", "new_orders", "delta_orders", "already_reflected_return", "net_return5", "net_return20", "late_net_return"]].to_string(index=False))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
