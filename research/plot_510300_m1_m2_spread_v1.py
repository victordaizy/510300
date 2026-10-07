"""将 510300 日收盘价与已公布 M1/M2 同比剪刀差画在同一时间轴上。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/research/510300_m1_m2_comparison_charts_v1"
PRICE_PATH = ROOT / "reports/research/510300_post_selection_extension_inputs_v1/candidate_features.parquet"
MACRO_PATH = ROOT / "reports/research/510300_m1_m2_monthly_increment_v1/release_features.csv"


def configure_style() -> None:
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        plt.rcParams["font.family"] = ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "sans-serif"]
    plt.rcParams.update({
        "axes.unicode_minus": False, "font.size": 11,
        "figure.facecolor": "#f8fafc", "savefig.facecolor": "#f8fafc",
        "axes.facecolor": "white", "axes.edgecolor": "#d5dde5",
        "text.color": "#1c2d40", "axes.labelcolor": "#48586a",
        "xtick.color": "#647489", "ytick.color": "#647489",
        "axes.spines.top": False, "path.simplify": False,
    })


def segment_plot(ax, frame: pd.DataFrame, column: str, color: str, label: str,
                 final_day: pd.Timestamp, width: float = 2.0) -> None:
    # 旧、新 M1 口径分别绘制；不跨口径连线。月度值只在新公布后更新。
    for number, (_, group) in enumerate(frame.groupby("training_regime", sort=False)):
        x = group["published_time"]
        y = group[column]
        if group.index[-1] == frame.index[-1]:
            x = pd.concat([x, pd.Series([final_day])], ignore_index=True)
            y = pd.concat([y, pd.Series([y.iloc[-1]])], ignore_index=True)
        ax.step(x, y, where="post", color=color, lw=width,
                label=label if number == 0 else None, zorder=3)
        ax.scatter(group["published_time"], group[column], color=color, s=11, zorder=4, linewidths=0)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PRICE_PATH, columns=["date", "close", "symbol", "source"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.loc[prices["date"] >= "2018-01-01"].reset_index(drop=True)
    assert prices["symbol"].eq("510300.SH").all()
    assert prices["date"].is_monotonic_increasing and not prices["date"].duplicated().any()
    end = prices["date"].iloc[-1] + pd.Timedelta(hours=15)
    macro_all = pd.read_csv(MACRO_PATH)
    macro_all["published_time"] = pd.to_datetime(macro_all["available_at_upper_bound"], utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    macro = macro_all.loc[macro_all["published_time"] <= end].copy().reset_index(drop=True)
    assert macro["published_time"].is_monotonic_increasing
    np.testing.assert_allclose(macro["spread_pp"], macro["m1_yoy_pp"] - macro["m2_yoy_pp"], rtol=0, atol=1e-12)
    assert macro[["m1_yoy_pp", "m2_yoy_pp", "spread_pp"]].notna().all().all()
    boundary = macro.loc[macro["training_regime"].eq("M1_NEW2025"), "published_time"].iloc[0]
    latest = macro.iloc[-1]

    # 日度对照表按当日 15:00 已知信息合并；月度文件保留全部原始公布日期。
    prices["close_time"] = prices["date"] + pd.Timedelta(hours=15)
    aligned = pd.merge_asof(
        prices, macro[["published_time", "stat_month", "m1_yoy_pp", "m2_yoy_pp", "spread_pp", "definition_version"]],
        left_on="close_time", right_on="published_time", direction="backward", allow_exact_matches=True,
    )
    assert (aligned.loc[aligned["published_time"].notna(), "published_time"] <= aligned.loc[aligned["published_time"].notna(), "close_time"]).all()
    aligned.to_csv(OUTPUT / "510300与剪刀差_日度对照.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
    aligned.to_parquet(OUTPUT / "510300与剪刀差_日度对照.parquet", index=False)
    macro.to_csv(OUTPUT / "M1_M2逐月公布值.csv", index=False, encoding="utf-8-sig")

    configure_style()
    blue, orange, green, purple = "#236eb2", "#d58028", "#238875", "#796ca6"
    fig = plt.figure(figsize=(16, 10))
    fig.text(.06, .945, "510300 与 M1−M2 剪刀差：走势对比", fontsize=25, weight="bold")
    fig.text(.06, .904, "2018—2026  ·  510300 日收盘价 + 每月货币数据  ·  宏观按实际公布日更新", fontsize=12, color="#647489")
    ax = fig.add_axes([.075, .435, .835, .395])
    spread_ax = ax.twinx()
    components = fig.add_axes([.075, .18, .835, .165], sharex=ax)

    price_line, = ax.plot(prices["close_time"], prices["close"], color=blue, lw=1.5, zorder=2, label="510300 收盘价（左轴）")
    segment_plot(spread_ax, macro, "spread_pp", orange, "M1−M2 同比剪刀差（右轴）", end)
    ax.set_ylabel("510300 收盘价（元，未复权）", color=blue, labelpad=12)
    spread_ax.set_ylabel("M1同比 − M2同比（百分点）", color=orange, labelpad=13)
    ax.tick_params(axis="y", labelcolor=blue, length=0)
    spread_ax.tick_params(axis="y", labelcolor=orange, length=0)
    ax.set_ylim(prices["close"].min() - .28, prices["close"].max() + .3)
    spread_ax.set_ylim(macro["spread_pp"].min() - 2.2, macro["spread_pp"].max() + 2.2)
    spread_ax.axhline(0, color=orange, alpha=.25, lw=.9, ls="--", zorder=0)
    ax.grid(axis="y", color="#dfe5eb", lw=.7)
    handles2, labels2 = spread_ax.get_legend_handles_labels()
    fig.legend([price_line] + handles2, [price_line.get_label()] + labels2,
               loc="lower left", bbox_to_anchor=(.07, .848), ncol=2, frameon=False, fontsize=12)

    segment_plot(components, macro, "m1_yoy_pp", green, "M1 同比", end, 1.65)
    segment_plot(components, macro, "m2_yoy_pp", purple, "M2 同比", end, 1.65)
    components.set_ylabel("同比增速（%）", labelpad=15)
    components.axhline(0, color="#aebac7", lw=.75, ls="--")
    components.grid(axis="y", color="#dfe5eb", lw=.7)
    components.yaxis.set_major_locator(MultipleLocator(5))
    components.legend(loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0, 1.25), fontsize=11)
    components.set_ylim(min(-10, macro["m1_yoy_pp"].min() - 1), max(17, macro["m1_yoy_pp"].max() + 2))

    for target in [ax, components]:
        target.axvspan(boundary, end, color="#e9edf4", alpha=.44, zorder=0)
        target.axvline(boundary, color="#8f9bab", lw=1, ls="--")
        target.tick_params(axis="x", length=0)
    ax.text(boundary + pd.Timedelta(days=23), ax.get_ylim()[1] - .08,
            "M1 新口径\n2025年1月数据起", fontsize=10, color="#697789", va="top",
            bbox={"fc": "white", "ec": "none", "alpha": .85, "pad": 4})
    components.xaxis.set_major_locator(mdates.YearLocator())
    components.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    components.set_xlim(pd.Timestamp("2018-01-01"), end + pd.Timedelta(days=18))
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.tick_params(axis="x", bottom=False)

    fig.text(.075, .105,
             f"行情截至 {end:%Y-%m-%d}：510300 收盘 {prices['close'].iloc[-1]:.3f} 元。同期已知的最新宏观为 {latest['stat_month']}：M1 {latest['m1_yoy_pp']:.1f}%，M2 {latest['m2_yoy_pp']:.1f}%，剪刀差 {latest['spread_pp']:+.1f} 个百分点。",
             fontsize=10.5, color="#4e6074")
    fig.text(.075, .072,
             "橙线上行表示 M1 增速相对 M2 改善；阶梯中的每个圆点对应一次月度公布。新旧 M1 口径分段绘制，未使用事后回溯值拼接。",
             fontsize=10.5, color="#647489")
    fig.text(.075, .043,
             "来源：仓库已保存的 510300 行情、人民银行月度报告。左右轴单位不同，曲线重合不表示数值相等；本图用于观察走势。",
             fontsize=10.5, color="#647489")
    fig.savefig(OUTPUT / "510300与M1_M2剪刀差_走势对比.png", dpi=170)
    fig.savefig(OUTPUT / "510300与M1_M2剪刀差_走势对比.svg")
    plt.close(fig)

    metadata = {
        "title": "510300 与 M1-M2 剪刀差走势对比", "price_observations": len(prices),
        "macro_observations": len(macro), "start": str(prices['date'].iloc[0].date()), "end": str(end),
        "macro_last_stat_month": str(latest['stat_month']), "macro_last_published_at": str(latest['published_time']),
        "macro_definition": "M1同比增速减M2同比增速，单位百分点", "price_basis": "未复权收盘价，单位元",
        "macro_alignment": "按实际公布时间逐次更新；日度导出表仅合并当日15:00已经公布的信息",
        "M1_definition_boundary_at_release": str(boundary), "backcasts_used": False,
        "outside_price_cutoff_macro_months_excluded": macro_all.loc[macro_all['published_time'] > end, 'stat_month'].tolist(),
        "source_files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in [PRICE_PATH, MACRO_PATH]],
        "checks": ["价格日期唯一且有序", "剪刀差等于M1同比减M2同比", "日度合并无未来公布信息", "新旧M1口径不跨界连线"],
        "new_accounts": 0, "new_model_fits": 0,
    }
    (OUTPUT / "图表口径.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
