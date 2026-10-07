"""把原边界、后补结果及当前量价背景放在同一张研究图上，不改候选规则。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as common
from research.adaptive_allocation_v1 import normalize_dividends

OUT = ROOT / "reports/research/510300_point_current_observation_20261001"
NAMES = {"CORE_AUXILIARY_DRAWDOWN_GATE": "回撤限制组合", "LAG_CONFIRMED_RUNS_AUXILIARY": "相关确认组合"}


def main():
    metrics = pd.read_parquet(OUT / "results/历史点位指标.parquet")
    extended = pd.read_parquet(ROOT / "reports/research/510300_point_history_extension_v1/results/更新历史指标.parquet")
    points = pd.read_parquet(OUT / "results/全部自然点位.parquet")
    prices = pd.read_parquet(OUT / "inputs/candidate_prices.parquet")
    dividends = normalize_dividends(pd.read_csv(OUT / "inputs/dividends.csv"))
    technical, _ = common.features(prices, dividends)
    recent = technical.loc[technical.date.ge("2026-07-15")].copy()
    recent["ema_raw"] = recent.ema20-recent.cash_shift
    new = points.loc[points.candidate.eq("CORE_AUXILIARY_DRAWDOWN_GATE") & points.status.eq("COMPLETE") & points.exit_date.ge("2026-08-14")]
    sensitivity = []
    for r in metrics.itertuples():
        # 仅问已有统计遇到一个假设损失如何变化，不能据此设定或优化止损。
        one_mean_loss_product = r.wins/(r.n+1)*r.b
        critical_loss = r.wins*r.mean_win*(r.losses+1)/(r.n+1)-r.losses*r.mean_loss
        sensitivity.append({"candidate": r.candidate, "actual_product": r.product, "actual_mean_loss": r.mean_loss,
                            "product_if_one_more_current_average_loss": one_mean_loss_product,
                            "single_extra_loss_at_product_equal_one": critical_loss,
                            "counterfactual_only_not_a_strategy": True})
    sensitivity = pd.DataFrame(sensitivity)
    sensitivity.to_csv(OUT / "results/达线余量说明.csv", index=False, encoding="utf-8-sig")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#f7f8fb", "axes.facecolor": "#ffffff"})
    fig, axes = plt.subplots(3, 1, figsize=(14, 10.5), gridspec_kw={"height_ratios": [1, 1.55, .85]})
    fig.subplots_adjust(top=.88, bottom=.095, left=.085, right=.97, hspace=.42)
    fig.suptitle("510300：历史点位仍达线，后补三笔削弱优势", x=.085, y=.965, ha="left", fontsize=22, fontweight="bold", color="#14263d")
    fig.text(.085, .925, "截至2026年9月30日｜日线与此前完整周｜固定份额、扣参考摩擦｜两条候选高度同源，不能合并样本", fontsize=11, color="#516173")
    colors = ["#176b87", "#aa5f13"]
    ax = axes[0]
    for (candidate, name), color in zip(NAMES.items(), colors):
        old = extended.loc[extended.candidate.eq(candidate)].iloc[0]
        current = metrics.loc[metrics.candidate.eq(candidate)].iloc[0]
        values = [old["old_product"], old["product"], current["product"]]
        ax.plot(range(3), values, "o-", color=color, linewidth=2.4, markersize=7, label=name)
        for x, value in enumerate(values):
            ax.annotate(f"{value:.4f}", (x, value), xytext=(7, 6), textcoords="offset points", color=color, fontsize=11)
    ax.axhline(1., color="#b3434a", linestyle="--", linewidth=1.4)
    ax.text(2.16, 1.005, "严格门槛 > 1", ha="right", color="#b3434a", fontsize=10)
    ax.set(xlim=(-.12, 2.22), ylim=(.985, 1.28), ylabel="胜率 × 实际净盈亏比")
    ax.set_xticks(range(3), ["原完整链\n8月14日", "补到既有行情\n9月16日", "补到最近收盘\n9月30日"])
    ax.legend(loc="upper right", frameon=False, ncol=2)
    ax.grid(axis="y", alpha=.15)
    ax = axes[1]
    ax.plot(recent.date, recent.close, color="#173751", linewidth=1.6, label="原始收盘价")
    ax.plot(recent.date, recent.ema_raw, color="#8c80af", linewidth=1.5, label="EMA20（当日原价坐标）")
    for i, r in enumerate(new.itertuples()):
        color = "#2d987d" if r.point_net_return > 0 else "#c56062"
        ax.axvspan(r.entry_date, r.exit_date, color=color, alpha=.11)
        ax.scatter([r.entry_date], [r.entry_raw], marker="^", color=color, s=64, zorder=5)
        ax.scatter([r.exit_date], [r.exit_raw], marker="v", color=color, s=64, zorder=5)
        middle = r.entry_date+(r.exit_date-r.entry_date)/2
        y = 4.98 if i == 0 else 4.84 if i == 1 else 4.84
        ax.text(middle, y, f"{r.entry_date:%m/%d}—{r.exit_date:%m/%d}\n{r.point_net_return:+.2%}", ha="center", va="center", fontsize=10, color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .87, "pad": 3})
    ax.set_title("后补三笔完整保留：+0.22%、−2.87%、−2.90%（两条候选日期相同）", loc="left", fontsize=12, pad=9)
    ax.set(ylabel="价格 / 元", ylim=(recent.low.min()-.05, max(recent.high.max()+.08, 5.06)))
    ax.legend(loc="lower left", frameon=False, ncol=2)
    ax.grid(axis="y", alpha=.15)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax = axes[2]
    ax.bar(recent.date, recent.daily_hist, color=np.where(recent.daily_hist >= 0, "#d69d59", "#779aa9"), width=1.1, alpha=.9, label="日MACD柱")
    ax.step(recent.date, recent.weekly_hist, where="post", color="#77639d", linewidth=1.5, label="此前完整周MACD柱")
    ax.axhline(0, color="#687582", linewidth=.8)
    ax.set_title("量价用于解释，未按事后亏损临时添加过滤条件", loc="left", fontsize=12, pad=9)
    ax.set_ylabel("MACD柱")
    ax.legend(loc="lower left", frameon=False, ncol=2)
    ax.grid(axis="y", alpha=.15)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    fig.text(.085, .035, "9月30日：两候选均为空仓等待；10月8日研究意向已登记。历史p×B达线 ≠ 已建立独立验证，前瞻完成点位仍为0。", fontsize=11, color="#516173")
    fig.savefig(OUT / "点位结果与最新量价.png", dpi=160)
    fig.savefig(OUT / "点位结果与最新量价.svg")
    plt.close(fig)
    lines = ["# 后补结果与达线余量", "",
             "**补齐后，回撤限制组合p×B为1.0951，相关确认组合为1.0289。两条仍过历史数值门槛，优势余量已经明显缩小。**", "",
             "原8月边界以后两条候选共有三组相同进出日期：8月7日至17日+0.2173%，8月18日至20日−2.8741%，9月17日至30日−2.8990%。三笔平均净收益约−1.85%。这些点位全部进入现有历史指标；两条候选的相同交易只能算同一组市场证据。", "",
             "| 候选 | 现有p×B | 假设再亏一笔当前平均亏损后的p×B | 单独新增一笔亏损使p×B恰等于1的亏损幅度 |", "|---|---:|---:|---:|"]
    for r in sensitivity.itertuples():
        lines.append(f"| {NAMES[r.candidate]} | {r.actual_product:.4f} | {r.product_if_one_more_current_average_loss:.4f} | {r.single_extra_loss_at_product_equal_one:.2%} |")
    lines += ["", "此表只是已有样本统计的代数敏感性，不预测下一笔，也不能把临界幅度反推为止损参数。相关确认组合仅再出现一笔当前平均大小的亏损，p×B就会落到1以下。因此当前只能称历史候选，尚不能称稳定优势。", "",
              "截至9月30日，两条候选都没有新多头进入意向；按原规则保存10月8日空仓等待记录。尚未有通过当前门槛的空头规则。历史补算、模型一致性与未来独立有效点位是不同证据。", ""]
    (OUT / "后补结果与达线余量.md").write_text("\n".join(lines), encoding="utf-8")
    common.save_json(OUT / "interpretation_receipt.json", {"at": common.now(), "new_strategy_evaluations": 0,
                     "new_point_dates": 0, "same_rules_preserved": True, "sensitivity_formula": "新增亏损x后pB=盈利总和*(亏损笔数+1)/((总笔数+1)*(亏损总额+x))",
                     "new_three_unique_points_mean": float(new.point_net_return.mean()), "sensitivity": sensitivity.to_dict("records"),
                     "figure": str((OUT / "点位结果与最新量价.png").relative_to(ROOT))})
    print("点位变化图与达线余量说明已保存，没有增加或筛除任何历史交易。", flush=True)


if __name__ == "__main__":
    main()
