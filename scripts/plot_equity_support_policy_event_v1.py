"""从已保存账户绘制净值与回撤，不生成新交易。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_equity_support_policy_event_v1"


def main():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    data = pd.read_csv(OUT / "results/完整逐日账户.csv")
    data["date"] = pd.to_datetime(data.date)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    colors = ["#126B87", "#D07735", "#929AA4"]
    specs = [("REVIEW_CLOSE_NEXT_OPEN_PRIMARY", "主方案（压力成本）"),
             ("FIRST_AVAILABLE_OPEN_DIAGNOSTIC", "提前入场对照（压力成本）"),
             ("BUY_AND_HOLD", "买入持有（压力成本）")]
    for (name, label), color in zip(specs, colors):
        group = data[(data.timing == name) & (data.scenario == "STRESS")]
        axes[0].plot(group.date, group.equity_cny / 10000, label=label, color=color, linewidth=1.7)
        axes[1].plot(group.date, group.drawdown * 100, color=color, linewidth=1.3)
    axes[0].axhline(20, color="#B2B8BF", linewidth=0.8, linestyle=":")
    axes[1].axhline(-10, color="#A14A40", linewidth=0.9, linestyle="--", label="回撤目标 −10%")
    axes[0].set_ylabel("账户资金（万元）")
    axes[1].set_ylabel("距账户高点回撤（%）")
    axes[0].legend(loc="upper left", frameon=False, fontsize=10)
    axes[1].legend(loc="lower left", frameon=False, fontsize=9)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("510300：支持政策公告后的稀疏交易仍未达到夏普1.5", fontsize=15, x=0.075, ha="left")
    fig.text(0.075, 0.923, "2024—2025年完整账户 · 20万元起始 · 计入全部空仓日与交易成本", fontsize=10, color="#58616B")
    fig.text(0.075, 0.027, "主方案压力夏普0.690，最大回撤7.20%，3次交易；仅已有公告历史检验，未建立独立验证。", fontsize=9, color="#58616B")
    fig.tight_layout(rect=[0.035, 0.06, 0.99, 0.91])
    path = OUT / "figures/账户净值与回撤.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"已由保存账簿生成图表：{path}")


if __name__ == "__main__":
    main()
