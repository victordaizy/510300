"""从已保存净值绘制第二轮对比，不运行政策或安装依赖。"""
import csv
import datetime as dt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parent
FONT = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")


def main() -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for column, policies in enumerate((("TREND", "TREND_NO_SLOPE"), ("REPAIR", "REPAIR_SEQUENCE"))):
        for policy, color in zip(policies, ("#89939f", "#237b70" if column == 0 else "#ba6543")):
            path = ROOT / "results_run2/accounts" / f"200000_STRESS_{policy}_account.csv"
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            dates = [dt.date.fromisoformat(row["date"]) for row in rows]
            equity = [float(row["equity"]) / 200000 for row in rows]
            peak, drawdown = 1.0, []
            for value in equity:
                peak = max(peak, value)
                drawdown.append(value / peak - 1)
            axes[0, column].plot(dates, equity, label=policy, color=color, linewidth=1.45)
            axes[1, column].plot(dates, drawdown, color=color, linewidth=1.35)
        axes[0, column].set_title("趋势：删除斜率条件" if column == 0 else "修复：先超跌、后确认", fontproperties=FONT, fontsize=14, pad=12)
        axes[0, column].axhline(1, color="#ced2d5", linestyle="--", linewidth=.7)
        axes[0, column].set_ylabel("净值（初始资金＝1）", fontproperties=FONT)
        axes[1, column].set_ylabel("完整账户回撤", fontproperties=FONT)
        axes[1, column].yaxis.set_major_formatter(PercentFormatter(1))
        axes[0, column].legend(loc="upper left", fontsize=9)
        for row in range(2):
            axes[row, column].grid(alpha=.18)
            axes[row, column].spines[["top", "right"]].set_visible(False)
    figure.suptitle("第二轮结构比较｜20万元 · 压力费用", fontproperties=FONT, fontsize=18, y=.98)
    figure.text(.5, .015, "2015-01-05—2026-08-14 · 原规则与两个固定新版本 · 同一账户合同 · 历史开发，尚未独立验证",
                ha="center", fontproperties=FONT, fontsize=10, color="#4b5563")
    figure.tight_layout(rect=(0, .045, 1, .94))
    path = ROOT / "analysis/第二轮20万元压力净值与回撤.png"
    if path.exists():
        raise FileExistsError("图形已存在，不覆盖")
    figure.savefig(path, dpi=150, facecolor="white")
    plt.close(figure)
    print("保存净值图已导出：" + str(path))


if __name__ == "__main__":
    main()
