"""绘制真实日线、货币预期差和政策背景；保留全部数据点，不画策略净值。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd


BLUE, GOLD, TEAL, RED, GREY = "#225e91", "#bb7c25", "#138774", "#bf525c", "#657588"


def configure() -> None:
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        plt.rcParams["font.family"] = FontProperties(fname=str(font)).get_name()
    else:
        plt.rcParams["font.family"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#bac3cc", "axes.labelcolor": "#344354",
        "text.color": "#233548", "xtick.color": GREY, "ytick.color": GREY,
        "figure.facecolor": "#ffffff", "axes.facecolor": "#ffffff", "svg.hashsalt": "510300_money_v2"})


def style(ax) -> None:
    ax.grid(axis="y", color="#e9edf0", linewidth=.8)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / (name + ".png"), dpi=160, facecolor="white")
    fig.savefig(out / (name + ".svg"), metadata={"Date": None})
    plt.close(fig)


def plot(study: Path, output: Path) -> None:
    configure()
    output.mkdir(parents=True, exist_ok=True)
    market = pd.read_parquet(study / "inputs/market_daily.parquet").sort_values("date")
    market["date"] = pd.to_datetime(market.date)
    money = pd.read_csv(study / "inputs/104个月共识选择.csv")
    money["at"] = pd.to_datetime(money.available_at_upper_bound, utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    money["actual_spread_pp"] = money.m1_yoy_pp - money.m2_yoy_pp
    money["spread_surprise_proxy_pp"] = money.spread_surprise_proxy_pp.mask(money.spread_surprise_proxy_pp.abs() < 1e-10, 0.)
    money = money.sort_values("at")
    policy = json.loads((study / "inputs/policy_context_events.json").read_text(encoding="utf-8"))
    breakpoint = money.loc[money.training_regime.eq("M1_NEW2025"), "at"].min()
    merged = pd.merge_asof(market.assign(decision_at=market.date + pd.Timedelta(hours=15)), money,
                           left_on="decision_at", right_on="at", direction="backward")
    merged["close_drawdown"] = merged.close / merged.close.cummax() - 1
    merged["total_return_drawdown"] = merged.wealth / merged.wealth.cummax() - 1
    merged.to_csv(study / "results/全部日线与当时已公布货币数据.csv", index=False, encoding="utf-8-sig")
    merged.to_parquet(study / "results/全部日线与当时已公布货币数据.parquet", index=False)
    money.to_csv(study / "results/全部月度实际与事前预期_绘图数据.csv", index=False, encoding="utf-8-sig")
    end = max(market.date.max(), money["at"].max()) + pd.Timedelta(days=12)
    fig, axs = plt.subplots(4, 1, figsize=(15.4, 11.6), sharex=True, gridspec_kw={"height_ratios": [1.7, 1.25, 1.25, .85]})
    fig.subplots_adjust(left=.075, right=.97, bottom=.10, top=.90, hspace=.20)
    fig.suptitle("510300、M1/M2 与市场预期：同一时间轴看走势", x=.075, ha="left", y=.965, fontsize=21, fontweight="bold")
    fig.text(.075, .928, "日线逐点保留；宏观数据在实际公布时刻更新。缺失预期留空，新旧 M1 口径分开。", color=GREY)
    visible = market[market.date.ge("2018-01-01")]
    axs[0].plot(visible.date, visible.close, color=BLUE, lw=1.25)
    axs[0].set_ylabel("510300 收盘价（元）")
    axs[0].set_title("价格走势", loc="left", fontsize=12)
    for ax in axs:
        style(ax)
        ax.axvline(breakpoint, color=RED, ls=":", lw=1.2)
    axs[0].text(breakpoint + pd.Timedelta(days=20), .95, "2025 新 M1 口径首次公布", transform=axs[0].get_xaxis_transform(), color=RED, fontsize=9, va="top")
    for j, (_, part) in enumerate(money.groupby("training_regime", sort=False)):
        dates = list(part["at"])
        stop = breakpoint if j == 0 else end
        dates.append(stop)
        for col, color, label in [("m1_yoy_pp", TEAL, "M1 同比"), ("m2_yoy_pp", BLUE, "M2 同比")]:
            axs[1].step(dates, [*part[col], part[col].iloc[-1]], where="post", color=color, lw=1.4, label=label if j == 0 else None)
        axs[2].step(dates, [*part.actual_spread_pp, part.actual_spread_pp.iloc[-1]], where="post", color=GOLD, lw=1.6, label="实际 M1−M2" if j == 0 else None)
    axs[1].set_ylabel("同比增速（%）")
    axs[1].legend(loc="upper left", frameon=False, ncol=2)
    admitted = money[money.consensus_admitted]
    axs[2].scatter(admitted["at"], admitted.expected_spread_proxy_pp, s=22, facecolors="white", edgecolors=BLUE, lw=1.1, label="事前预期之差（在对应公布日对照）", zorder=4)
    axs[2].axhline(0, color=GREY, lw=.7)
    axs[2].set_ylabel("剪刀差（百分点）")
    axs[2].legend(loc="lower left", frameon=False, ncol=2, fontsize=10)
    axs[3].bar(admitted["at"], admitted.spread_surprise_proxy_pp, width=17, color=np.where(admitted.spread_surprise_proxy_pp >= 0, TEAL, RED))
    axs[3].axhline(0, color=GREY, lw=.7)
    axs[3].set_ylabel("实际−预期\n（百分点）")
    axs[3].set_xlim(pd.Timestamp("2018-01-01"), end)
    axs[3].xaxis.set_major_locator(mdates.YearLocator())
    axs[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(.075, .048, "剪刀差 = M1 同比 − M2 同比。预期为同一报告中两个调查汇总值之差，不是受访者配对剪刀差的中位数。", fontsize=10, color=GREY)
    fig.text(.075, .022, "104 个月实际值；77 个月有配对预期；行情截至 2026-09-11。资料来自央行及 Bualuang / OCBC / NBG 原报告；属于历史重建。", fontsize=10, color=GREY)
    save(fig, output, "510300_M1M2_实际与预期走势对比")

    fig, axs = plt.subplots(3, 1, figsize=(15.4, 10.2), sharex=True, gridspec_kw={"height_ratios": [2.1, 1.1, .8]})
    fig.subplots_adjust(left=.075, right=.97, bottom=.23, top=.88, hspace=.18)
    fig.suptitle("政策与行情：标明发生时间，不把同步涨跌当作因果", x=.075, ha="left", y=.97, fontsize=20, fontweight="bold")
    fig.text(.075, .925, "8 项政策作为背景事件；清单不是完整政策样本，缺少一致的事前政策预期。", color=GREY)
    zoom = market[market.date.ge("2024-01-01")]
    axs[0].plot(zoom.date, zoom.close, color=BLUE, lw=1.5)
    axs[0].set_ylabel("510300 收盘价（元）")
    for i, rec in enumerate(policy, 1):
        date = pd.Timestamp(rec["date"])
        axs[0].axvline(date, color=GREY, alpha=.4, lw=.8)
        axs[0].text(date, 1.02 + ((i % 2) * .07), str(i), transform=axs[0].get_xaxis_transform(), ha="center", fontsize=10, color=GREY)
    for j, (_, part) in enumerate(money[money["at"].ge("2023-12-01")].groupby("training_regime", sort=False)):
        stop = breakpoint if j == 0 else end
        axs[1].step([*part["at"], stop], [*part.actual_spread_pp, part.actual_spread_pp.iloc[-1]], where="post", color=GOLD, lw=1.6)
    axs[1].scatter(admitted["at"], admitted.expected_spread_proxy_pp, s=24, facecolors="white", edgecolors=BLUE, label="事前预期")
    axs[1].set_ylabel("M1−M2（百分点）")
    axs[1].legend(frameon=False, fontsize=10)
    axs[2].bar(admitted["at"], admitted.spread_surprise_proxy_pp, width=12, color=np.where(admitted.spread_surprise_proxy_pp >= 0, TEAL, RED))
    axs[2].set_ylabel("预期差（百分点）")
    axs[2].axhline(0, lw=.8, color=GREY)
    for ax in axs:
        style(ax)
        ax.axvline(breakpoint, color=RED, ls=":", lw=1)
    axs[1].text(breakpoint + pd.Timedelta(days=7), .05, "M1 口径切换", transform=axs[1].get_xaxis_transform(), color=RED, fontsize=9)
    axs[2].set_xlim(pd.Timestamp("2024-01-01"), end)
    axs[2].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axs[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for i, rec in enumerate(policy):
        x = .075 if i < 4 else .54
        y = .159 - (i % 4) * .032
        fig.text(x, y, f"{i+1}  {rec['date']}  {rec['label']}", fontsize=10, color=GREY)
    fig.text(.075, .012, "政策宣布日、实施日与首次披露时间分开保存；未估计政策贡献百分比，也未把背景事件转为交易信号。", fontsize=10, color=GREY)
    save(fig, output, "510300_预期差与政策背景")

    pred = pd.read_csv(study / "results/A_B全部逐期预测.csv")
    cases = pd.read_csv(study / "results/全部准入事件_固定五日.csv")
    summary = json.loads((study / "results/summary.json").read_text(encoding="utf-8"))
    a = summary["adjudications"][0]
    fig, axs = plt.subplots(2, 1, figsize=(15.4, 9.3), gridspec_kw={"height_ratios": [1, 1.25]})
    fig.subplots_adjust(left=.075, right=.97, bottom=.12, top=.86, hspace=.53)
    fig.suptitle("超预期后的首轮反应，不等于之后五日仍有可预测收益", x=.075, ha="left", y=.97, fontsize=20, fontweight="bold")
    fig.text(.075, .916, f"旧口径 36 个起始训练事件、24 个逐期评价事件：B 加入预期差后，MSE 增加 {-a['relative_MSE_improvement']:.2%}。", color=RED, fontsize=12)
    x = np.arange(len(pred))
    gain = pred.MSE_improvement_A_minus_B * 10000
    axs[0].bar(x, gain, color=np.where(gain >= 0, TEAL, RED), width=.72)
    axs[0].axhline(0, color=GREY, lw=.8)
    axs[0].axvline(11.5, ls="--", color=GREY, lw=1)
    axs[0].set_xticks(x, pred.stat_month, rotation=45, ha="right", fontsize=9)
    axs[0].set_ylabel("A 平方误差 − B 平方误差\n（百分点平方；高于零才改善）")
    axs[0].set_title("全部 24 个评价事件都报告；前半、后半的平均改善均小于零", loc="left", fontsize=12)
    part = cases[cases.training_regime.eq("M1_NEW2025")]
    x = np.arange(len(part))
    axs[1].bar(x-.19, part.initial_response*100, width=.36, color=BLUE, label="公告前收盘 → 首完整交易日收盘")
    axs[1].bar(x+.19, part.residual_5d_gross_return*100, width=.36, color=GOLD, label="观察后次日开盘 → 第 5 日收盘")
    axs[1].set_xticks(x, part.stat_month, rotation=45, ha="right", fontsize=9)
    axs[1].axhline(0, lw=.8, color=GREY)
    axs[1].set_ylabel("510300 收益（%）")
    axs[1].legend(loc="lower left", frameon=False, ncol=2, fontsize=10)
    axs[1].set_title("新口径 16 个成熟历史案例：首轮与预期差同向 14 次；后续五日同向 8 次（仅描述，模型未运行）", loc="left", fontsize=12)
    for ax in axs:
        style(ax)
    fig.text(.075, .027, "两段收益的起止点不同，不可直接相减归因。旧口径增量门失败；新口径未达 60 个事件；均不是独立前向验证。", fontsize=10, color=GREY)
    save(fig, output, "预期差_增量检验与首轮后续对照")

    fig, axs = plt.subplots(2, 1, figsize=(15.4, 7.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    fig.subplots_adjust(left=.075, right=.97, top=.87, bottom=.12, hspace=.12)
    fig.suptitle("510300 完整日线：3476 个交易日逐点保留", x=.075, ha="left", y=.97, fontsize=21, fontweight="bold")
    axs[0].plot(market.date, market.close, lw=1.1, color=BLUE)
    axs[0].set_ylabel("未复权收盘价（元）")
    dd = market.wealth / market.wealth.cummax() - 1
    axs[1].fill_between(market.date, dd*100, 0, color=RED, alpha=.25)
    axs[1].plot(market.date, dd*100, color=RED, lw=.8)
    axs[1].set_ylabel("含分红总回报回撤（%）")
    trough = int(np.argmin(dd.to_numpy()))
    peak = int(np.argmax(market.wealth.iloc[:trough+1].to_numpy()))
    for index, label in [(peak, "回撤前高点"), (trough, "最大回撤谷底")]:
        axs[1].scatter(market.iloc[index].date, dd.iloc[index]*100, color=RED, s=22, zorder=4)
        axs[1].annotate(f"{label}\n{market.iloc[index].date:%Y-%m-%d}", (market.iloc[index].date, dd.iloc[index]*100), xytext=(14, 14), textcoords="offset points", fontsize=9, arrowprops={"arrowstyle": "-", "color": GREY})
    for ax in axs:
        style(ax)
    axs[1].xaxis.set_major_locator(mdates.YearLocator())
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.text(.075, .03, "2012-05-28 至 2026-09-11。下图为价格与分红序列的描述性回撤，无交易费用；本轮没有策略账户或策略净值。", fontsize=10, color=GREY)
    save(fig, output, "510300_全部历史日线与回撤")
    print("四张图及全部日频、月频绘图数据已生成。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制货币实际、预期与510300历史走势")
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    plot(args.study_dir.resolve(), args.output_dir or args.study_dir / "figures")
