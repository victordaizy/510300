"""使用已保存的国家统计局全国数据计算经济结构解读依据并绘图。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def compact_text(run: Path, filename: str) -> str:
    return re.sub(r"\s+", "", (run / "release_text" / filename).read_text(encoding="utf-8"))


def source_fact(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"原稿中没有找到所需数据：{pattern}")
    return float(match.group(1).replace(",", ""))


def read_tables(run: Path, filename: str) -> list:
    return json.loads((run / "release_tables" / filename).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="计算国家统计局经济结构解读依据")
    parser.add_argument("--run-id", default="20261004_national_full")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "nbs_economic_interpretation_20261004")
    args = parser.parse_args()
    run = ROOT / "data" / "nbs_monthly" / "runs" / args.run_id
    comparison = ROOT / "data" / "nbs_monthly" / "analyses" / f"{args.run_id}_multiyear_v2"
    output = args.output.resolve()

    with (run / "官方发布索引.csv").open(encoding="utf-8-sig") as stream:
        release_index = list(csv.DictReader(stream))

    households = []
    for release in release_index:
        match = re.fullmatch(r"(202[1-6])年上半年居民收入和消费支出情况", release["标题"])
        if match is None:
            continue
        text = compact_text(run, Path(release["正文位置"]).name)
        income = source_fact(text, r"全国居民人均可支配收入([0-9,]+)元")
        spending = source_fact(text, r"全国居民人均消费支出([0-9,]+)元")
        households.append({
            "年份": int(match.group(1)), "统计期间": "上半年",
            "人均可支配收入元": income, "人均消费支出元": spending,
            "消费收入比%": spending / income * 100,
            "官方来源": release["官方链接"], "本地原稿": release["正文位置"],
        })
    households.sort(key=lambda row: row["年份"])
    household_by_year = {row["年份"]: row for row in households}
    ratio_change = household_by_year[2026]["消费收入比%"] - household_by_year[2025]["消费收入比%"]

    gdp_tables = read_tables(run, "t20260716_1964142.json")
    gdp_growth = []
    for row in gdp_tables[1]["grid"]:
        if not re.fullmatch(r"20\d{2}", str(row[0]).strip()):
            continue
        for quarter, value in enumerate(row[1:5], start=1):
            value = str(value).strip()
            if value:
                gdp_growth.append({"年份": int(row[0]), "季度": quarter, "实际GDP同比%": float(value)})
    gdp_mom = {}
    for row in gdp_tables[2]["grid"]:
        if not re.fullmatch(r"20\d{2}", str(row[0]).strip()):
            continue
        for quarter, value in enumerate(row[1:5], start=1):
            value = str(value).strip()
            if value:
                gdp_mom[(int(row[0]), quarter)] = float(value)
    for row in gdp_growth:
        row["季调GDP环比%"] = gdp_mom.get((row["年份"], row["季度"]))

    industry = pd.read_csv(comparison / "41行业月度生产盈利.csv", dtype={"数据月份": str})
    latest = industry.loc[industry["数据月份"] == "202608"].copy()
    if len(latest) != 41:
        raise ValueError("行业数据不完整，需要41个工业行业")

    with sqlite3.connect(f"file:{(run / '国家统计局月度.sqlite').resolve().as_posix()}?mode=ro", uri=True) as connection:
        def observation(indicator_id: str) -> float:
            row = connection.execute(
                "SELECT value FROM observations WHERE indicator_id=? AND period=? AND value IS NOT NULL",
                (indicator_id, "202608"),
            ).fetchone()
            if row is None:
                raise ValueError(f"缺少2026年8月数据：{indicator_id}")
            return float(row[0])

        profit_now = observation("5d98d24036dd4dbebf7f5c0a80753df6")
        profit_base = observation("cccb1653c99b45859f44c4acc07c32a2")
        revenue_now = observation("f3fbbb5fbb28401bb6351039a171df5a")
        revenue_base = observation("cdbcec91e4974de18314ac5db7c1271b")

    if not np.isclose(latest["利润累计亿元"].sum(), profit_now, atol=0.15, rtol=0):
        raise ValueError("行业利润合计与工业总额不一致")
    key_names = ["计算机、通信和其他电子设备制造业", "有色金属冶炼和压延加工业"]
    leaders = latest.loc[latest["行业"].isin(key_names)]
    if len(leaders) != 2 or leaders["利润可比基数亿元"].isna().any():
        raise ValueError("缺少电子或有色行业的可比利润基数")
    total_delta = profit_now - profit_base
    pair_delta = leaders["利润可比净增额亿元"].sum()
    concentration = pair_delta / total_delta * 100

    briefing = compact_text(run, "t20260915_1965307.txt")
    investment = {
        "全部投资": -source_fact(briefing, r"固定资产投资（不含农户）[0-9]+亿元，同比下降([0-9.]+)%"),
        "扣除房地产的投资": -source_fact(briefing, r"扣除房地产开发的固定资产投资下降([0-9.]+)%"),
        "制造业投资": -source_fact(briefing, r"制造业投资下降([0-9.]+)%"),
        "基础设施投资": -source_fact(briefing, r"基础设施投资同比下降([0-9.]+)%"),
        "房地产开发投资": -source_fact(briefing, r"房地产开发投资下降([0-9.]+)%"),
        "扣除房地产的民间投资": -source_fact(briefing, r"扣除房地产开发的民间投资下降([0-9.]+)%"),
    }
    profit_text = compact_text(run, "t20260928_1965425.txt")
    inventory_days = source_fact(profit_text, r"产成品存货周转天数为([0-9.]+)天")
    inventory_days_change = source_fact(profit_text, r"产成品存货周转天数为[0-9.]+天，同比增加([0-9.]+)天")
    receivable_days = source_fact(profit_text, r"应收账款平均回收期为([0-9.]+)天")
    receivable_days_change = source_fact(profit_text, r"应收账款平均回收期为[0-9.]+天，同比增加([0-9.]+)天")

    selected_names = ["计算机、通信和其他电子设备制造业", "有色金属冶炼和压延加工业", "汽车制造业", "电气机械和器材制造业"]
    industry_keys = ["行业", "增加值累计同比%", "营业收入累计同比%", "利润累计同比%", "营业收入利润率%", "利润率同比变化百分点", "应收账款期末同比%", "产成品库存期末同比%"]
    facts = {
        "数据快照": args.run_id,
        "季度背景说明": "GDP、居民收支和产能利用率使用已保存的季度发布稿，保持原频率，未构造月度GDP。",
        "GDP季度路径": gdp_growth,
        "历年上半年居民收支": households,
        "消费收入比2026较2025变化百分点": ratio_change,
        "消费收入比说明": "住户调查汇总消费与收入的比值，不是边际消费倾向，也不直接等于银行存款变化。",
        "工业利润本期亿元": profit_now,
        "工业利润可比基数亿元": profit_base,
        "工业利润可比净增额亿元": total_delta,
        "电子有色利润净增额亿元": pair_delta,
        "电子有色占工业利润净增额%": concentration,
        "净增额占比说明": "分母为含各行业增减相抵的净增加额，不是总利润占比或GDP增长贡献率。",
        "工业营业收入亿元": revenue_now,
        "工业营业收入可比基数亿元": revenue_base,
        "工业营业收入利润率计算值%": profit_now / revenue_now * 100,
        "工业可比上年营业收入利润率计算值%": profit_base / revenue_base * 100,
        "投资结构2026年1至8月同比%": investment,
        "存货周转天数": inventory_days,
        "存货周转天数同比增加": inventory_days_change,
        "应收账款回收期天数": receivable_days,
        "应收账款回收期同比增加": receivable_days_change,
        "重点行业2026年1至8月": json.loads(latest.loc[latest["行业"].isin(selected_names), industry_keys].to_json(orient="records", force_ascii=False)),
        "主要来源": [
            "https://www.stats.gov.cn/sj/zxfb/202607/t20260716_1964142.html",
            "https://www.stats.gov.cn/sj/zxfb/202607/t20260715_1964129.html",
            "https://www.stats.gov.cn/sj/zxfb/202609/t20260915_1965307.html",
            "https://www.stats.gov.cn/sj/zxfb/202609/t20260928_1965425.html",
            "https://data.stats.gov.cn/dg/website/page.html#/pc/national/monthData",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "经济解读计算依据.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    font_file = Path("C:/Windows/Fonts/msyh.ttc")
    if font_file.exists():
        font_manager.fontManager.addfont(str(font_file))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_file)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), layout="constrained")
    blue, teal, red, amber = "#2463A2", "#008778", "#BD4949", "#AD761C"

    gdp_plot = [row for row in gdp_growth if row["年份"] >= 2024]
    positions = np.arange(len(gdp_plot))
    axes[0, 0].plot(positions, [r["实际GDP同比%"] for r in gdp_plot], marker="o", color=blue, label="实际同比")
    axes[0, 0].plot(positions, [r["季调GDP环比%"] for r in gdp_plot], marker="s", color=teal, label="季调环比")
    axes[0, 0].set_xticks(positions, [f"{r['年份'] % 100}Q{r['季度']}" for r in gdp_plot])
    axes[0, 0].set_title("GDP保持增长，二季度增速低于一季度（%）", loc="left")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot([r["年份"] for r in households], [r["消费收入比%"] for r in households], marker="o", color=blue)
    for r in households:
        axes[0, 1].annotate(f"{r['消费收入比%']:.2f}%", (r["年份"], r["消费收入比%"]), xytext=(0, 10), textcoords="offset points", ha="center")
    axes[0, 1].set_xticks([r["年份"] for r in households])
    axes[0, 1].set_ylim(62.8, 66.5)
    axes[0, 1].set_title("历年上半年消费占收入比例（%，非零起点）", loc="left")

    names = list(investment)
    values = list(investment.values())
    bars = axes[1, 0].barh(names[::-1], values[::-1], color=blue)
    axes[1, 0].bar_label(bars, fmt="%.1f", padding=4)
    axes[1, 0].set_xlim(-23, 1)
    axes[1, 0].axvline(0, color="#666666", linewidth=0.8)
    axes[1, 0].set_title("投资下降超出房地产（2026年1—8月同比，%）", loc="left")

    electronic_delta = float(leaders.loc[leaders["行业"] == key_names[0], "利润可比净增额亿元"].iloc[0])
    nonferrous_delta = float(leaders.loc[leaders["行业"] == key_names[1], "利润可比净增额亿元"].iloc[0])
    bars = axes[1, 1].bar(["电子设备", "有色冶炼", "其他行业净额"], [electronic_delta, nonferrous_delta, total_delta - pair_delta], color=[blue, teal, "#8895A7"])
    axes[1, 1].bar_label(bars, fmt="%.1f", padding=4)
    axes[1, 1].set_ylim(0, max(electronic_delta, nonferrous_delta) * 1.2)
    axes[1, 1].set_title(f"利润净增额集中度（1—8月，亿元；两行业{concentration:.1f}%）", loc="left")

    labels = ["电子设备", "有色冶炼", "汽车", "电气机械"]
    for name, label, color in zip(selected_names, labels, [blue, teal, red, amber]):
        path = industry.loc[(industry["行业"] == name) & industry["数据月份"].str.endswith("08") & (industry["数据月份"] >= "202108")]
        axes[2, 0].plot(path["数据月份"].str[:4].astype(int), path["营业收入利润率%"], marker="o", color=color, label=label)
    axes[2, 0].set_xticks(range(2021, 2027))
    axes[2, 0].set_title("行业利润率路径（各年1—8月，%）", loc="left")
    axes[2, 0].legend(frameon=False, ncol=2)

    x = np.arange(2)
    previous = [inventory_days - inventory_days_change, receivable_days - receivable_days_change]
    current = [inventory_days, receivable_days]
    for offset, data, label, color in [(-0.18, previous, "上年可比同期", "#8895A7"), (0.18, current, "2026年8月末", blue)]:
        bars = axes[2, 1].bar(x + offset, data, width=0.36, label=label, color=color)
        axes[2, 1].bar_label(bars, fmt="%.1f", padding=4)
    axes[2, 1].set_xticks(x, ["存货周转天数", "应收账款平均回收期"])
    axes[2, 1].set_ylim(0, 87)
    axes[2, 1].set_title("周转与回款时间延长（天）", loc="left")
    axes[2, 1].legend(frameon=False, loc="upper left")

    for ax in axes.flat:
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
    fig.suptitle("全国经济结构与企业经营：按各指标实际统计期间比较\n2026年10月4日采集快照；季度数据保持原频率；历史为保存的公布版本", fontsize=16)
    fig.savefig(output / "经济结构与企业经营.png", dpi=170, facecolor="white")
    plt.close(fig)
    print(json.dumps({"计算依据": str(output / "经济解读计算依据.json"), "图表": str(output / "经济结构与企业经营.png"), "利润净增额亿元": round(total_delta, 2), "电子有色净增额占比%": round(concentration, 4), "消费收入比变化百分点": round(ratio_change, 4)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
