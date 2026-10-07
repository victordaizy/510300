"""绘制实际价格、当时已知货币数据、事前预期快照及政策背景。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="绘制真实数据对比图")
    ap.add_argument("--study-dir", type=Path, required=True)
    args = ap.parse_args()
    study = args.study_dir.resolve()
    results = study / "results"
    figures = study / "figures"
    figures.mkdir(exist_ok=True)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    if font.exists():
        fm.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#aab3bb", "text.color": "#1e303c",
                         "axes.labelcolor": "#334852", "xtick.color": "#51606c", "ytick.color": "#51606c"})
    daily = pd.read_parquet(results / "510300与当时已知货币数据_全部日线.parquet")
    daily["date"] = pd.to_datetime(daily.date)
    events = pd.read_csv(results / "事前预期差与510300全部事件.csv", parse_dates=["release_date"])
    official = pd.read_csv(study / "inputs/official_releases.csv")
    official["day"] = pd.to_datetime(official.published_at).dt.tz_localize(None).dt.normalize()
    policy = json.loads((study / "inputs/policy_context_events.json").read_text(encoding="utf-8"))
    blue, green, orange, purple = "#195e99", "#008377", "#d9802f", "#7657a3"

    def draw_series(ax, frame, column, label, color, lw=1.6):
        for j, (_, part) in enumerate(frame.dropna(subset=[column]).groupby("training_regime", sort=False)):
            ax.step(part.date, part[column], where="post", color=color, lw=lw, label=label if j == 0 else None)

    cut = daily[daily.date >= "2024-01-01"].copy()
    fig, axes = plt.subplots(4, 1, figsize=(16, 13.4), sharex=True, gridspec_kw={"height_ratios": [2.8, 1.55, 1.65, 1.5]})
    fig.subplots_adjust(left=.075, right=.97, top=.88, bottom=.14, hspace=.20)
    fig.suptitle("510300：走势、M1/M2、预期差与政策背景", x=.075, y=.965, ha="left", fontsize=23, fontweight="bold")
    fig.text(.075, .925, "日线保留全部观察｜货币数据按实际公布后进入｜预期只有可核实快照，空缺不补造", fontsize=12, color="#536a77")
    ax = axes[0]
    ax.plot(cut.date, cut.close, color=blue, lw=1.8)
    ax.set_ylabel("510300 收盘价（元）")
    ax.set_title("01  价格与政策事件  ·  标记表示发生时间，不代表已经测出政策贡献", loc="left", fontsize=11, pad=9)
    ax.set_ylim(cut.close.min() * .965, cut.close.max() * 1.055)
    rows = [0.95, .80, .95, .66, .82, .95, .80, .95]
    for j, event in enumerate(policy):
        date = pd.Timestamp(event["date"])
        ax.axvline(date, lw=.75, color="#9cabb3", linestyle=(0, (3, 3)))
        ax.text(date, rows[j], chr(65 + j), transform=ax.get_xaxis_transform(), ha="center", va="center",
                bbox={"boxstyle": "circle,pad=.24", "facecolor": "#edf2f5", "edgecolor": "#a4b5bf"}, fontsize=9)
    axes[1].set_title("02  M1、M2 同比  ·  2025 年新口径单列，边界不连线", loc="left", fontsize=11, pad=8)
    draw_series(axes[1], cut, "m1_yoy_pp", "M1 同比", green)
    draw_series(axes[1], cut, "m2_yoy_pp", "M2 同比", orange)
    axes[1].set_ylabel("同比（%）")
    axes[1].axhline(0, color="#aab7bf", lw=.6)
    axes[1].legend(loc="lower right", frameon=False, ncol=2)
    axes[2].set_title("03  剪刀差 S = M1 − M2  ·  菱形为同份事前周报两项共识之差", loc="left", fontsize=11, pad=8)
    draw_series(axes[2], cut, "spread_pp", "当时已公布剪刀差", green)
    axes[2].scatter(events.release_date, events.expected_spread_proxy_pp, marker="D", s=44, color=purple,
                    edgecolor="white", linewidth=.8, zorder=5, label="公布前周报共识快照")
    axes[2].set_ylabel("剪刀差（百分点）")
    axes[2].legend(loc="lower right", frameon=False, ncol=2)
    axes[3].set_title("04  预期差 = 实际剪刀差 − 周报共识差  ·  仅 6 次事前配对，× 表示缺合格配对", loc="left", fontsize=11, pad=8)
    seen = set(events.stat_month)
    missing = official[(official.day >= cut.date.min()) & (official.day <= cut.date.max()) & ~official.stat_month.isin(seen)]
    axes[3].scatter(missing.day, np.zeros(len(missing)), marker="x", s=25, color="#abb4bb", linewidths=1, label="配对缺失")
    axes[3].bar(events.release_date, events.spread_surprise_proxy_pp, width=11,
                color=[green if x > 0 else "#bd5661" for x in events.spread_surprise_proxy_pp], zorder=3)
    for _, e in events.iterrows():
        axes[3].annotate(f"{e.spread_surprise_proxy_pp:+.1f}", (e.release_date, e.spread_surprise_proxy_pp),
                         xytext=(0, 7 if e.spread_surprise_proxy_pp > 0 else -15), textcoords="offset points", ha="center", fontsize=10)
    axes[3].axhline(0, lw=.7, color="#97a5af")
    axes[3].set_ylim(-3.3, 2.6)
    axes[3].set_ylabel("预期差（百分点）")
    for ax in axes:
        ax.grid(axis="y", color="#e7ebee", linewidth=.65)
    boundary = pd.Timestamp("2025-02-14")
    for ax in axes[1:3]:
        ax.axvline(boundary, color="#b0a3c6", lw=.9, linestyle="--")
    axes[1].annotate("新 M1 首次公布", (boundary, 8.5), xytext=(8, -4), textcoords="offset points", fontsize=9, color=purple)
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[-1].set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2026-09-16"))
    labels = [f"{chr(65+j)}  {e['date'][5:]} {e['label']}" for j, e in enumerate(policy)]
    fig.text(.075, .084, "   ｜   ".join(labels[:4]), fontsize=9.4)
    fig.text(.075, .060, "   ｜   ".join(labels[4:]), fontsize=9.4)
    fig.text(.075, .025, "来源：保存的 510300 行情、人民银行月报、WGC/Bloomberg 事前周报和所列官方政策公告。价格未复权；事件收益另用含分红序列。\n行情截至 2026-09-11。政策背景为有来源的示例目录；预期快照并非公告前最后一刻共识。", fontsize=9, color="#667780")
    for ext in ("png", "svg"):
        fig.savefig(figures / f"510300_货币预期与政策全景.{ext}", dpi=160, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), sharex=True)
    fig.subplots_adjust(left=.065, right=.975, top=.84, bottom=.12, hspace=.40, wspace=.27)
    fig.suptitle("6 次可核实的事前快照：消息方向与持有期限分开看", x=.065, y=.965, ha="left", fontsize=22, fontweight="bold")
    fig.text(.065, .913, "横轴为公告日后的交易日序号，0 为基准收盘｜全程含分红｜各子图纵轴分别缩放｜固定展示 1、5、20 日", fontsize=11, color="#546b76")
    market = pd.read_parquet(study / "inputs/market_daily.parquet").sort_values("date").reset_index(drop=True)
    market["date"] = pd.to_datetime(market.date)
    trajectories = []
    for ax, (_, e) in zip(axes.flat, events.iterrows()):
        first = int(pd.DatetimeIndex(market.date).searchsorted(e.release_date, side="right"))
        frame = market.iloc[first - 1:first + 20].copy()
        frame["event_session"] = np.arange(len(frame))
        frame["event_return"] = frame.wealth / frame.wealth.iloc[0] - 1
        frame["stat_month"] = e.stat_month
        trajectories.append(frame[["stat_month", "date", "event_session", "event_return"]])
        ax.plot(frame.event_session, frame.event_return * 100, color=blue, lw=2)
        ax.axhline(0, color="#a7b4bd", lw=.8)
        ax.set_title(f"{e.stat_month} 数据｜预期差 {e.spread_surprise_proxy_pp:+.1f} pp\n公布 {e.release_date:%Y-%m-%d}", loc="left", fontsize=11)
        ax.set_ylabel("累计收益（%）")
        for h, color in ((1, "#82979f"), (5, orange), (20, green)):
            val = float(frame.loc[frame.event_session == h, "event_return"].iloc[0] * 100)
            ax.scatter(h, val, s=26, zorder=4, color=color)
        ax.text(.02, .96, f"1 日 {e.return_1d:+.2%}  ·  5 日 {e.return_5d:+.2%}\n20 日 {e.return_20d:+.2%}", transform=ax.transAxes,
                va="top", fontsize=9, bbox={"facecolor":"white", "edgecolor":"none", "alpha":.8})
        ax.grid(color="#e8edef", lw=.6)
        ax.set_xlim(0, 20)
        ax.margins(y=.24)
        ax.set_xticks([0, 5, 10, 15, 20])
    for ax in axes[-1]:
        ax.set_xlabel("公告日后的交易日")
    fig.text(.065, .035, "这些是可取得的稀疏历史快照，不能当作随机完整样本；窗口内其他政策和数据可能影响价格。图中收益不是扣费账户收益，也不是单条消息的因果贡献。", fontsize=9, color="#657883")
    for ext in ("png", "svg"):
        fig.savefig(figures / f"六次预期差_公布后完整路径.{ext}", dpi=160, facecolor="white")
    plt.close(fig)
    pd.concat(trajectories).to_csv(results / "六次事件全部交易日路径.csv", index=False, encoding="utf-8-sig", float_format="%.12g")

    fig, axes = plt.subplots(3, 1, figsize=(15.5, 9), sharex=True, gridspec_kw={"height_ratios":[2,1,1]})
    fig.subplots_adjust(left=.08, right=.97, top=.86, bottom=.12, hspace=.18)
    fig.suptitle("510300 与 M1 / M2：2018—2026 年完整走势", x=.08, y=.96, ha="left", fontsize=21, fontweight="bold")
    fig.text(.08, .905, f"{len(daily):,} 个交易日全部保留；每次货币数据只能在实际公布后显示", fontsize=11)
    axes[0].plot(daily.date, daily.close, color=blue, lw=1.25)
    axes[0].set_ylabel("价格（元，未复权）")
    draw_series(axes[1], daily, "m1_yoy_pp", "M1 同比", green)
    draw_series(axes[1], daily, "m2_yoy_pp", "M2 同比", orange)
    axes[1].set_ylabel("同比（%）")
    axes[1].legend(ncol=2, loc="upper left", frameon=False)
    draw_series(axes[2], daily, "spread_pp", "M1 − M2", purple)
    axes[2].axhline(0, color="#a1afb8", lw=.7)
    axes[2].set_ylabel("剪刀差（百分点）")
    axes[2].legend(loc="upper left", frameon=False)
    for ax in axes:
        ax.grid(axis="y", color="#e7ebee", lw=.7)
    for ax in axes[1:]:
        ax.axvline(boundary, color="#b0a3c6", lw=.9, linestyle="--")
    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlim(daily.date.min(), daily.date.max())
    fig.text(.08, .037, "2025-02-14 首次公布新口径 M1，边界不连线；后来回溯的新口径数据没有提前放入历史。来源和逐日数据见配套文件。", fontsize=9.5, color="#657883")
    for ext in ("png", "svg"):
        fig.savefig(figures / f"510300_M1M2_完整历史走势.{ext}", dpi=160, facecolor="white")
    plt.close(fig)
    print("已生成三组真实数据图和全部事件路径。")


if __name__ == "__main__":
    main()
