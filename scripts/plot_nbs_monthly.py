"""从本地全国月度快照生成经济与行业观察图；保留原口径和缺失月份。"""
import argparse
import json
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "nbs_monthly"
COLORS = ["#174B70", "#BD7C2D", "#287B73", "#926EAA"]


def main():
    parser = argparse.ArgumentParser(description="生成全国宏观与工业行业观察图，不联网。")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run = BASE / "runs" / args.run_id if args.run_id else Path(json.loads((BASE / "LATEST.json").read_text(encoding="utf-8"))["path"])
    scope = json.loads((run / "scope.json").read_text(encoding="utf-8"))
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    connection = sqlite3.connect(run / "国家统计局月度.sqlite")
    font_file = Path(r"C:\Windows\Fonts\msyh.ttc")
    if font_file.exists():
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_file)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#CBD2D7", "text.color": "#203645", "axes.labelcolor": "#526875"})
    periods = pd.period_range("2021-01", scope["end_request"], freq="M")
    dates = periods.to_timestamp()
    selections = [
        ("制造业：生产与订单", "%（扩散指数）", 50, [("a09aa989bdcf4cffa2021795722eb916", "制造业PMI", 0), ("4151df33b53f4d02ae9f51fe402f1a50", "新订单指数", 0)]),
        ("工业与服务业生产", "当月同比 %", 0, [("ef1b1765960d45a29b4d7c4ca91be916", "规模以上工业", 0), ("3fc5439f03ec46a5a76e8032036e8c17", "服务业生产", 0)]),
        ("社会消费品零售", "当月同比 %", 0, [("aaac57d54d2e465d91bc9f3ea1a8618e", "社会消费品零售总额", 0)]),
        ("投资与房地产", "累计同比 %", 0, [("7e570cf8071c4734a7d78d9f0a70fbe1", "固定资产投资（不含农户）", 0), ("205e08cba8c2409980db58c98da91b6f", "房地产开发投资", 0)]),
        ("工业企业盈利", "累计同比 %", 0, [("7f30595d047445e0a3880ab928795892", "工业企业利润总额", 0)]),
        ("居民与工业产品价格", "当月同比 %（同月基数100的指数减100）", 0, [("4ae9047687934a6390984c21d6ddab96", "CPI / 2021—2025口径", 100), ("53180dfb9c14411ba4b762307c85920c", "CPI / 2026起口径", 100), ("150633e52b9a470a9a9fd1b296dd6c5b", "PPI", 100)]),
        ("货币供应量", "官方同比 %；口径变化见原始定义", 0, [("e03f2232631f41cd9d754a7d7feb4a81", "M2", 0), ("640401d3351b4b868dea28f89f410a54", "M1", 0), ("db7891fb8f3c4eb2a4d71a9955eba8c7", "M0", 0)]),
    ]
    unemployment = [i for i in meta["indicators"] if i["category"] == "城镇调查失业率" and i["i_showname"].strip().startswith("全国城镇调查失业率")]
    selections.append(("就业", "%（调查失业率）", None, [(i["_id"], "全国城镇调查失业率", 0) for i in unemployment[:1]]))
    dump_rows, manifest = [], []
    fig, axes = plt.subplots(4, 2, figsize=(15, 15), facecolor="#F8FAFB")
    for ax, (title, unit, reference, selected) in zip(axes.flat, selections):
        ax.set_facecolor("white")
        latest_labels = []
        for index, (iid, label, subtract) in enumerate(selected):
            data = pd.read_sql_query("SELECT period,value FROM observations WHERE indicator_id=? AND status='有值' AND period>=? ORDER BY period", connection, params=[iid, "202101"])
            values = pd.Series(data.value.to_numpy(), index=pd.PeriodIndex(data.period, freq="M")).reindex(periods) - subtract
            ax.plot(dates, values, label=label, linewidth=1.6, color=COLORS[index % len(COLORS)])
            if values.notna().any():
                last = values.last_valid_index()
                latest_labels.append(f"{label}：{values.loc[last]:.1f} / {last}")
            manifest.append({"indicator_id": iid, "label": label, "transformation": f"原值减{subtract}", "window_start": "202101", "window_end": scope["end_request"]})
            dump_rows.extend({"indicator_id": iid, "label": label, "period": str(period), "plotted_value": value}
                             for period, value in values.items())
        if reference is not None:
            ax.axhline(reference, color="#96A4AE", linewidth=0.8, linestyle="--")
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel(unit, fontsize=9)
        ax.grid(axis="y", color="#E6EBEF", linewidth=0.6)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="best", frameon=False, fontsize=8)
        ax.text(0, -0.21, "；".join(latest_labels), transform=ax.transAxes, fontsize=8, color="#657988", wrap=True)
    fig.suptitle("国家统计局 · 全国月度经济观察", x=0.055, y=0.982, ha="left", fontsize=23, fontweight="bold")
    fig.text(0.055, 0.947, f"采集业务日期 {scope['as_of']} ｜数据库下载版本｜原始空值保持断线；不同价格口径分线显示", fontsize=11, color="#657988")
    fig.text(0.055, 0.012, "各指标最新月份不同。图中为当前官方数据库返回的历史版本；发布稿可能更早更新，详见官方发布索引。", fontsize=10, color="#657988")
    fig.subplots_adjust(top=0.91, bottom=0.075, left=0.065, right=0.985, hspace=0.55, wspace=0.18)
    fig.savefig(run / "全国经济观察.png", dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    pd.DataFrame(dump_rows).to_csv(run / "经济观察图对应数据.csv", index=False, encoding="utf-8-sig")
    (run / "经济观察图指标选择.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    industries = [i for i in meta["indicators"] if "分大类行业增加值增长速度" in i["catalogue_path"] and "2018" in i["catalogue_path"] and "增加值同比增长" in i["i_showname"]]
    industries.sort(key=lambda i: (int(i.get("ds_order") or 0), i["i_showname"]))
    months = pd.period_range(end=scope["end_request"], periods=24, freq="M")
    matrix, names = [], []
    for item in industries:
        data = pd.read_sql_query("SELECT period,value FROM observations WHERE indicator_id=? AND status='有值' ORDER BY period", connection, params=[item["_id"]])
        series = pd.Series(data.value.to_numpy(), index=pd.PeriodIndex(data.period, freq="M")).reindex(months)
        matrix.append(series.to_numpy())
        names.append(item["i_showname"].split("增加值")[0].strip())
    if matrix:
        values = np.array(matrix, dtype=float)
        finite = values[np.isfinite(values)]
        scale = max(abs(finite.min()), abs(finite.max()), 1) if len(finite) else 1
        fig, ax = plt.subplots(figsize=(15, max(10, len(names) * .27 + 2)))
        palette = plt.get_cmap("RdBu_r").copy()
        palette.set_bad("#E6E9EC")
        chart = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=palette, vmin=-scale, vmax=scale)
        ax.set_yticks(range(len(names)), labels=names, fontsize=9)
        ax.set_xticks(range(len(months)), labels=[str(m) for m in months], rotation=65, fontsize=8)
        ax.set_title("工业行业增加值 · 当月同比增速（%）", loc="left", fontsize=19, fontweight="bold", pad=28)
        ax.text(0, 1.015, "采用官方2018年至今行业分类目录；完整显示各行业；灰色表示空值或尚未公布", transform=ax.transAxes, fontsize=10, color="#657988")
        fig.colorbar(chart, ax=ax, fraction=.025, pad=.025, label="官方当月同比 %")
        fig.subplots_adjust(left=.30, right=.94, bottom=.12, top=.94)
        fig.savefig(run / "工业行业趋势.png", dpi=170, facecolor="white")
        plt.close(fig)
        pd.DataFrame(values, index=names, columns=[str(m) for m in months]).to_csv(run / "工业行业近24个月同比.csv", encoding="utf-8-sig")
    connection.close()
    print(f"已生成全国经济观察图及 {len(industries)} 个工业行业的历史同比观察图：{run}")


if __name__ == "__main__":
    main()
