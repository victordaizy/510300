"""展示政策事件、510300 和当时已公布剪刀差的时间关系，不估计因果收益。"""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_macro_expectations_policy_context_v1"
PRICE = ROOT / "reports/research/510300_post_selection_extension_inputs_v1/candidate_features.parquet"
MACRO = ROOT / "reports/research/510300_m1_m2_monthly_increment_v1/release_features.csv"
EVENTS = [
    {
        "event_date": "2024-09-24", "time_precision": "DATE_ONLY_IN_THIS_CHART",
        "title": "金融支持经济高质量发展发布会",
        "content": "宣布降准、降低政策利率及支持资本市场的新工具",
        "source_url": "https://english.scio.gov.cn/pressroom/node_9013507.html",
        "alternative_source_url": "https://www.csrc.gov.cn/ningbo/c101607/c7509061/content.shtml",
        "pre_announcement_consensus": None, "surprise": None,
    },
    {
        "event_date": "2024-09-26", "time_precision": "DATE_ONLY_IN_THIS_CHART",
        "title": "中央政治局会议",
        "content": "部署加大财政货币政策逆周期调节力度、提振资本市场",
        "source_url": "https://cpc.people.com.cn/n1/2024/0926/c64094-40328825.html",
        "alternative_source_url": None,
        "pre_announcement_consensus": None, "surprise": None,
    },
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(PRICE, columns=["date", "close"])
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.loc[prices["date"].between("2024-06-01", "2024-12-31")].copy()
    prices["close_time"] = prices["date"] + pd.Timedelta(hours=15)
    macro = pd.read_csv(MACRO)
    macro["release_time"] = pd.to_datetime(macro["available_at_upper_bound"], utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    aligned = pd.merge_asof(prices, macro[["release_time", "stat_month", "m1_yoy_pp", "m2_yoy_pp", "spread_pp"]],
                            left_on="close_time", right_on="release_time", direction="backward")
    assert (aligned["release_time"] <= aligned["close_time"]).all()
    aligned.to_csv(OUT / "政策示例_逐日价格与已知剪刀差.csv", index=False, encoding="utf-8-sig")
    aligned.to_parquet(OUT / "政策示例_逐日价格与已知剪刀差.parquet", index=False)
    pd.DataFrame(EVENTS).to_csv(OUT / "政策示例_事件与来源.csv", index=False, encoding="utf-8-sig")

    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f8fafc", "savefig.facecolor": "#f8fafc",
                         "text.color": "#20334a", "axes.labelcolor": "#53687c", "path.simplify": False})
    fig = plt.figure(figsize=(15, 9))
    fig.text(.065, .942, "把政策日期放回 510300 与剪刀差走势图", fontsize=24, weight="bold")
    fig.text(.065, .895, "2024 年下半年事件展示  ·  真实历史数据  ·  本图未测量市场预期差或政策因果效应", fontsize=12, color="#6c7b8c")
    ax = fig.add_axes([.085, .455, .85, .365])
    bx = fig.add_axes([.085, .195, .85, .18], sharex=ax)
    blue, orange, purple = "#236eb2", "#d58028", "#7c63a3"
    ax.plot(prices["close_time"], prices["close"], color=blue, lw=2)
    ax.set_ylabel("510300 收盘价（元，未复权）", color=blue)
    ax.set_ylim(3.05, 4.78)
    ax.text(.015, .93, "510300", transform=ax.transAxes, color=blue, fontsize=13, weight="bold")
    bx.step(aligned["close_time"], aligned["spread_pp"], where="post", color=orange, lw=2)
    bx.set_ylabel("已公布 M1−M2（百分点）", color=orange)
    bx.set_ylim(-15.3, -7.9)
    for event in EVENTS:
        day = pd.Timestamp(event["event_date"])
        for target in [ax, bx]:
            target.axvline(day, color=purple, lw=1, alpha=.85, ls="--")
        row = prices.loc[prices["date"].eq(day)].iloc[0]
        ax.scatter(row["close_time"], row["close"], color=purple, s=55, edgecolors="white", zorder=6)
    ax.annotate("9月24日  金融政策发布会\n降准、降息及资本市场支持工具",
                xy=(pd.Timestamp("2024-09-24 15:00"), 3.427), xytext=(pd.Timestamp("2024-07-05"), 4.46),
                fontsize=11, color=purple,
                arrowprops={"arrowstyle": "-", "color": purple, "lw": 1.2},
                bbox={"fc": "white", "ec": "none", "pad": 5})
    ax.annotate("9月26日  中央政治局会议\n加大逆周期调节、提振资本市场",
                xy=(pd.Timestamp("2024-09-26 15:00"), 3.640), xytext=(pd.Timestamp("2024-10-25"), 3.4),
                fontsize=11, color=purple,
                arrowprops={"arrowstyle": "-", "color": purple, "lw": 1.2},
                bbox={"fc": "white", "ec": "none", "pad": 5})
    bx.annotate("9月下旬已知的最新值仍为 −13.6\n来自9月13日公布的8月数据",
                xy=(pd.Timestamp("2024-09-25"), -13.6), xytext=(pd.Timestamp("2024-07-17"), -9.3),
                fontsize=10.5, color="#a76320",
                arrowprops={"arrowstyle": "-", "color": orange, "lw": 1.2},
                bbox={"fc": "white", "ec": "none", "pad": 4})
    bx.annotate("10月14日公布的9月值：−14.2",
                xy=(pd.Timestamp("2024-10-15"), -14.2), xytext=(pd.Timestamp("2024-10-28"), -14.85),
                fontsize=10.5, color="#a76320",
                arrowprops={"arrowstyle": "-", "color": orange, "lw": 1.1},
                bbox={"fc": "white", "ec": "none", "pad": 3})
    for target in [ax, bx]:
        target.grid(axis="y", color="#dfe5eb", lw=.7)
        target.tick_params(length=0, labelcolor="#63758a")
        target.axvspan(pd.Timestamp("2024-09-24"), pd.Timestamp("2024-09-26 23:59"), color=purple, alpha=.06)
        target.spines["left"].set_visible(False)
        target.spines["bottom"].set_color("#dfe5eb")
    bx.xaxis.set_major_locator(mdates.MonthLocator())
    bx.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    bx.set_xlim(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-12-31"))
    plt.setp(ax.get_xticklabels(), visible=False)
    fig.text(.085, .117, "这段走势显示：价格已经明显上涨时，当时已公布的剪刀差仍偏弱。解释价格，需要继续考察政策及预期变化。", fontsize=11.2)
    fig.text(.085, .075, "说明：只标出两项政策事件作例示，不是完整事件样本；未把随后上涨倒推为“政策超预期”，也未计算政策贡献。", fontsize=10.4, color="#6c7b8c")
    fig.text(.085, .043, "数据：仓库 510300 日行情与人民银行当次月报。政策依据：国新办9月24日发布会、新华社9月26日会议报道。", fontsize=10.4, color="#6c7b8c")
    fig.savefig(OUT / "510300_剪刀差与政策事件_2024示例.png", dpi=165)
    fig.savefig(OUT / "510300_剪刀差与政策事件_2024示例.svg")
    plt.close(fig)
    (OUT / "示例范围.json").write_text(json.dumps({
        "status": "DESCRIPTIVE_ILLUSTRATION_ONLY", "period": ["2024-06-01", "2024-12-31"],
        "daily_rows": len(prices), "events": EVENTS, "complete_policy_event_universe": False,
        "period_chosen_after_observing_history": True, "policy_causal_effect_estimated": False,
        "market_expectation_surprise": "NOT_COMPUTED_MISSING_PRE_RELEASE_CONSENSUS",
        "new_accounts": 0, "new_fits": 0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存政策事件与剪刀差时间对照图，共 {len(prices)} 个日度观察；未估计政策因果收益。")


if __name__ == "__main__":
    main()
