"""使用已有绘图库导出保存净值；不安装依赖、不运行账户。"""
from pathlib import Path
import csv
import datetime as dt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
FONT = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
POLICIES = ("BUY_HOLD", "STATIC50", "VOL10", "TREND", "REPAIR", "MIX50")
COLORS = {"BUY_HOLD": "#7a8697", "STATIC50": "#a69982", "VOL10": "#197c72",
          "TREND": "#bc5137", "REPAIR": "#a477c0", "MIX50": "#315db3"}


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor("#ffffff")
    for column, cost in enumerate(("BASE", "STRESS")):
        for policy in POLICIES:
            path = ROOT / "results_run1/accounts" / f"200000_{cost}_{policy}_account.csv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            dates = [dt.date.fromisoformat(row["date"]) for row in rows]
            equity = [float(row["equity"]) / 200000 for row in rows]
            peak, drawdown = 1.0, []
            for value in equity:
                peak = max(peak, value)
                drawdown.append(value / peak - 1.0)
            axes[0, column].plot(dates, equity, color=COLORS[policy], label=policy, linewidth=1.4)
            axes[1, column].plot(dates, drawdown, color=COLORS[policy], linewidth=1.2)
        axes[0, column].set_title("20万元 · " + ("基础费用" if cost == "BASE" else "压力费用"), fontproperties=FONT, fontsize=13, pad=12)
        axes[0, column].axhline(1, color="#c8cdd1", linestyle="--", linewidth=.7)
        axes[0, column].set_ylabel("净值（初始资金＝1）", fontproperties=FONT)
        axes[1, column].set_ylabel("完整账户回撤", fontproperties=FONT)
        axes[1, column].yaxis.set_major_formatter(PercentFormatter(1))
        for row in range(2):
            axes[row, column].grid(alpha=.17)
            axes[row, column].spines[["top", "right"]].set_visible(False)
        axes[0, column].legend(ncol=3, fontsize=8.5, loc="upper left")
    fig.suptitle("510300｜六个固定政策的完整账户净值与回撤", fontproperties=FONT, fontsize=19, x=.5, y=.98)
    fig.text(.5, .02, "2015-01-05—2026-08-14 · 含现金日、分红及费用 · 固定历史开发 · 日线开盘成交代理 · 未独立验证",
             ha="center", fontproperties=FONT, fontsize=10, color="#4b5563")
    fig.tight_layout(rect=(0, .05, 1, .94))
    path = ROOT / "analysis/20万元六政策净值与回撤.png"
    if path.exists():
        raise FileExistsError("图形已存在，不覆盖已有交付")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print("保存净值图已导出：" + str(path))


if __name__ == "__main__":
    main()
