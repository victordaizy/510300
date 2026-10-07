"""生成政策信息研究图表与中文报告；使用保存输入，不搜索卖点。"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from policy_information_clock_v1 import ROOT, OUT, save, now

FONT = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
plt.rcParams.update({"font.family": FONT.get_name(), "axes.unicode_minus": False, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "#f7f8fa", "axes.facecolor": "#ffffff"})
COLORS = {"利率与流动性": "#277c9b", "地产金融": "#ac6f27", "资本市场工具": "#784fa0", "财政化债": "#b34f4a", "消费与设备更新": "#54845e", "中美贸易政策": "#b27637"}


def output(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / "figures" / f"{name}.png", dpi=170, facecolor=fig.get_facecolor())
    fig.savefig(OUT / "figures" / f"{name}.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def data() -> tuple:
    m = pd.read_parquet(OUT / "inputs/market.parquet")
    m["date"] = pd.to_datetime(m.date)
    money = pd.read_csv(OUT / "inputs/money_104.csv")
    money["known"] = pd.to_datetime(money.available_at_upper_bound, utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    nodes = pd.read_parquet(OUT / "results/跨通道政策链_完整事实与时钟.parquet")
    nodes["known"] = pd.to_datetime(nodes.source_available_upper, utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None)
    return m, money, nodes


def overview(m: pd.DataFrame, money: pd.DataFrame, nodes: pd.DataFrame) -> None:
    window = m[(m.date >= "2024-01-01") & (m.date <= "2025-12-31")].copy()
    window["decision_time"] = window.date + pd.Timedelta(hours=15)
    aligned = pd.merge_asof(window.sort_values("decision_time"), money.sort_values("known"), left_on="decision_time", right_on="known", direction="backward")
    aligned["spread_pp"] = aligned.m1_yoy_pp - aligned.m2_yoy_pp
    aligned.to_csv(OUT / "results/两年全日频走势与当时已知货币数据.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(4, 1, figsize=(15.4, 13), sharex=True, gridspec_kw={"height_ratios": [3.2, 2.1, 2.1, 3]})
    fig.subplots_adjust(left=.085, right=.97, top=.91, bottom=.085, hspace=.18)
    fig.suptitle("510300 × 宏观数据 × 政策更新", x=.085, y=.975, ha="left", fontsize=23, weight="bold", color="#182c43")
    fig.text(.085, .938, "2024—2025 年完整日线｜货币数据按实际公布时间进入｜政策点是来源核实节点，不是买卖信号", fontsize=11, color="#56616f")
    ax[0].plot(window.date, window.close, color="#234d73", lw=1.7)
    ax[0].set_ylabel("510300 收盘价（元）")
    ax[0].set_ylim(window.close.min() - .30, window.close.max() + .20)
    for idx, label in ((window.close.idxmin(), "区间最低"), (window.close.idxmax(), "区间最高")):
        r = window.loc[idx]
        ax[0].scatter([r.date], [r.close], s=22, color="#234d73", zorder=4)
        ax[0].annotate(f"{label} {r.close:.3f}\n{r.date:%Y-%m-%d}", xy=(r.date, r.close), xytext=(8, -30) if label == "区间最低" else (-55, 13), textcoords="offset points", fontsize=9)
    regimes = [("OLD", "旧口径"), ("NEW2025", "新口径")]
    for i, (regime, label) in enumerate(regimes):
        part = aligned[aligned.training_regime.str.contains(regime, na=False)]
        for col, color, name in (("m1_yoy_pp", "#267f7a", "M1 同比"), ("m2_yoy_pp", "#c88331", "M2 同比")):
            ax[1].plot(part.date, part[col], color=color, lw=1.65, drawstyle="steps-post", label=name if i == 0 else None)
        ax[2].plot(part.date, part.spread_pp, color="#7055a1", lw=1.65, drawstyle="steps-post", label="实际 M1−M2（已公布值）" if i == 0 else None)
    first_new = money[money.training_regime.str.contains("NEW2025", na=False)].known.min()
    for a in ax[:3]:
        a.axvline(first_new, color="#8b8e95", ls=":", lw=1)
        a.grid(axis="y", alpha=.15)
    ax[1].annotate("新口径首次进入\n两侧分开画", (first_new, 7.8), xytext=(14, 7), textcoords="offset points", fontsize=9, color="#666a74")
    ax[1].legend(loc="lower left", ncol=2, frameon=False)
    ax[1].set_ylabel("同比增速（%）")
    admitted = money[(money.known >= window.date.min()) & (money.known <= window.date.max() + pd.Timedelta(days=1)) & money.consensus_admitted.eq(True)]
    ax[2].scatter(admitted.known, admitted.expected_spread_proxy_pp, s=25, facecolors="white", edgecolors="#b26630", label="事前调查预期（在实际公告时点对齐）", zorder=5)
    ax[2].set_ylabel("M1−M2（百分点）")
    ax[2].set_ylim(aligned.spread_pp.min() - .8, aligned.spread_pp.max() + 3.0)
    ax[2].legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
    ax[2].text(.01, .80, "调查预期 ≠ 公布前股价已隐含的预期", transform=ax[2].transAxes, ha="left", va="top", fontsize=9, color="#666a74")
    for j, chain in enumerate(COLORS):
        sub = nodes[nodes.chain == chain]
        yy = np.full(len(sub), float(j))
        for _, group in sub.reset_index(drop=True).groupby("known"):
            if len(group) > 1:
                yy[group.index] += np.linspace(-.09, .09, len(group))
        ax[3].scatter(sub.known, yy, color=COLORS[chain], s=32, zorder=4)
    ax[3].set_yticks(range(6), list(COLORS))
    ax[3].set_ylim(5.6, -.65)
    ax[3].grid(axis="y", alpha=.13)
    ax[3].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax[3].set_xlim(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31"))
    fig.text(.085, .035, f"上图保留全部 {len(window)} 个交易日；价格未复权，不是含费用账户净值。政策节点的含义见第二张图及完整 CSV。\n政策、货币与价格同向或背离只能描述时序；本图没有识别政策因果，也没有生成持仓。", color="#59616d", fontsize=10, linespacing=1.55)
    output(fig, "510300_货币数据与六类政策_完整走势")


def chain_diagram(nodes: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(19, 10.5))
    fig.subplots_adjust(left=.025, right=.985, bottom=.08, top=.86)
    fig.suptitle("政策从宣布到执行：哪些信息需要更新？", x=.035, y=.975, ha="left", fontsize=23, weight="bold", color="#182c43")
    fig.text(.035, .925, "六个通道 · 24 个来源节点｜箭头仅表示公布顺序，不表示资金累加；间距不代表时间｜回购与互换为不同工具", color="#5a6572", fontsize=11)
    ax.set_xlim(-1.05, 6.1)
    ax.set_ylim(-.6, 5.8)
    ax.axis("off")
    notes = {
        "R01": "拟降息20基点\n实施日期待确认", "R02": "目标利率已宣布\n本次确认生效", "R03": "拟再降息10基点\n同时存在贸易消息", "R04": "前一日已宣布\n当天操作确认",
        "H01": "3000亿央行额度\n不加上5000亿银行贷款", "H02": "调整资金支持比例\n不是新增3000亿",
        "C01": "5000亿计划额度\n尚无实际投放", "C02": "申请超2000亿\n仍不同于实际中标", "C03": "另一项3000亿工具\n不是重复新增额度", "C04": "当次操作500亿\n不等于当日股票成交", "C05": "首批投放超过90%\n机构池扩至40家", "C06": "当次操作550亿\n费率20→10基点",
        "F01": "方向已给出\n新增具体限额未给出", "F02": "每年2万亿 × 3年\n用途是置换存量债务", "F03": "回报12月18日已完成\n不能倒填为当日可知",
        "D01": "已有两新行动框架\n还没有7月的金额", "D02": "两新合计约3000亿\n消费部分约1500亿", "D03": "家电8类扩至12类\n增加数码购新支持",
        "T01": "拟4月10日实施\n尚未生效即被替代", "T02": "34%计划改为84%\n不把两者相加", "T03": "84%再改为125%\n不是相加的税率", "T04": "05/07 06:15已知\n尚无协议金额细节", "T05": "05/12 05:30已知\n尚无关税具体安排", "T06": "05/12 15:00公布\n部分暂停90天",
    }
    for row, chain in enumerate(COLORS):
        y = 5 - row
        part = nodes[nodes.chain == chain]
        color = COLORS[chain]
        ax.text(-1.03, y + .08, chain, fontsize=12, weight="bold", color=color, va="center")
        for col, r in enumerate(part.itertuples()):
            x = col
            card = FancyBboxPatch((x, y - .32), .90, .75, boxstyle="round,pad=.02,rounding_size=.04", facecolor="white", edgecolor=color, linewidth=1.05)
            ax.add_patch(card)
            date = r.economic_event_date[2:]
            if r.node_id == "F02":
                date += "（来源次日）"
            ax.text(x + .04, y + .32, f"{r.node_id}  {date}", fontsize=8.5, color=color, weight="bold")
            ax.text(x + .04, y + .15, r.title, fontsize=9.1, color="#273444")
            ax.text(x + .04, y - .04, notes[r.node_id], fontsize=8.3, color="#656e78", va="top", linespacing=1.6)
            if col < len(part) - 1:
                ax.annotate("", xy=(x + .99, y + .045), xytext=(x + .92, y + .045), arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2})
    fig.text(.035, .045, "所有节点均未取得可认证的事前政策共识，因此没有把政策变化填写成‘超预期分数’。来源上界可复核，但不声称已证明全球最早公开版本。", fontsize=10.5, color="#56616f")
    output(fig, "六条政策链_宣布细则执行进度")


def timing_chart(m: pd.DataFrame, nodes: pd.DataFrame) -> None:
    sub = m[(m.date >= "2025-05-06") & (m.date <= "2025-05-16")].copy()
    fig, ax = plt.subplots(2, 1, figsize=(14.6, 8.4), gridspec_kw={"height_ratios": [3.2, 2]})
    fig.subplots_adjust(left=.075, right=.96, top=.86, bottom=.095, hspace=.35)
    fig.suptitle("5 月 12 日：开盘前的进展消息，与下午的关税细节", x=.075, y=.965, ha="left", fontsize=20, weight="bold", color="#182c43")
    fig.text(.075, .914, "2025 年日内瓦会谈时钟案例｜仅说明信息与价格的先后顺序，不计算策略收益或政策因果", fontsize=11, color="#5a6572")
    ax[0].plot(sub.date, sub.close, color="#244f75", marker="o", ms=4, lw=1.6, label="每日收盘")
    ax[0].scatter(sub.date, sub.open, marker="D", s=27, facecolors="white", edgecolors="#b37438", label="每日开盘", zorder=5)
    ax[0].vlines(sub.date, sub.low, sub.high, color="#bac5ce", lw=2, zorder=1, label="当日高低范围")
    for day in ["2025-05-07", "2025-05-12", "2025-05-13", "2025-05-14"]:
        ax[0].axvline(pd.Timestamp(day), color="#bdc5ce", ls=":", lw=.9)
    ax[0].set_ylabel("510300 价格（元）")
    ax[0].grid(axis="y", alpha=.15)
    ax[0].legend(loc="upper left", frameon=False, ncol=3, fontsize=9)
    ax[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax[0].set_xticks(sub.date)
    ax[0].text(.015, .05, "日线只能显示开盘、收盘与高低范围；不能据此复原盘中消息反应。", transform=ax[0].transAxes, fontsize=9, color="#66707b")
    ax[1].axis("off")
    table_rows = []
    names = {"T04": "确认将举行会谈", "T05": "已取得实质性进展", "T06": "联合声明具体安排"}
    for identifier in names:
        r = nodes.set_index("node_id").loc[identifier]
        table_rows.append([names[identifier], r.source_available_upper[5:16].replace("T", " "), r.first_daily_open_after_source_upper[5:], r.review_close[5:], r.research_execution_open[5:]])
    table = ax[1].table(cellText=table_rows, colLabels=["新增可核对信息", "来源标注时间¹", "上界之后首个日频开盘", "固定复核收盘", "该规则下执行开盘"], cellLoc="center", loc="upper center", colWidths=[.24, .18, .22, .18, .18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#e1e5e9")
        cell.set_facecolor("#e6edf4" if row == 0 else "white")
        if row == 0:
            cell.set_text_props(weight="bold", color="#244f75")
    ax[1].text(0, -.02, "¹ 分钟精度按该分钟最后一秒处理。例如 15:00 → 15:00:59，不能视为当日收盘前已获得全文。\n首个日频开盘只是时间参照；固定规则先观察收盘再执行，会额外等待。两者必须分别评价。", transform=ax[1].transAxes, fontsize=9.5, color="#5a6572", va="top", linespacing=1.65)
    output(fig, "贸易政策_盘前消息与收盘细节时钟")


def catalog_chart() -> None:
    df = pd.read_csv(OUT / "results/全部官方目录记录.csv")
    lp = pd.read_csv(OUT / "results/全部24次LPR目录记录.csv")
    lp["date"] = pd.to_datetime(lp.catalog_event_date)
    fig, axes = plt.subplots(2, 1, figsize=(14.6, 8.4), gridspec_kw={"height_ratios": [2.3, 2.5]})
    fig.subplots_adjust(left=.08, right=.965, bottom=.10, top=.87, hspace=.44)
    fig.suptitle("完整保留目录记录，也保留‘政策没有改变’", x=.08, y=.968, ha="left", fontsize=20, weight="bold", color="#182c43")
    fig.text(.08, .918, "490 条目录记录不等于 490 个独立政策事件；同一政策可有多次发布，也可同时出现在多个目录。", fontsize=11, color="#5a6572")
    tab = df.groupby(["catalog_month", "kind"]).size().unstack(fill_value=0).sort_index()
    x = np.arange(len(tab))
    axes[0].bar(x, tab.SCIO_DIRECTORY_ENTRY, color="#547792", label="国新办年度目录入口")
    axes[0].bar(x, tab.PBOC_CHRONOLOGY_PARAGRAPH, bottom=tab.SCIO_DIRECTORY_ENTRY, color="#c19860", label="央行年度大事记条目")
    axes[0].set_xticks(x, tab.index, rotation=45, ha="right", fontsize=8.5)
    axes[0].set_ylabel("目录记录数")
    axes[0].set_ylim(0, tab.sum(axis=1).max() + 9)
    axes[0].legend(frameon=False, ncol=2, loc="upper right", fontsize=9)
    axes[0].grid(axis="y", alpha=.12)
    for col, label, color in (("one_year_percent", "1 年期 LPR", "#237f7b"), ("five_year_percent", "5 年期以上 LPR", "#a97834")):
        axes[1].step(lp.date, lp[col], where="post", color=color, label=label, lw=1.6)
        changed = lp[col].diff().lt(0)
        axes[1].scatter(lp.date[~changed], lp[col][~changed], s=24, facecolors="white", edgecolors=color, zorder=4)
        axes[1].scatter(lp.date[changed], lp[col][changed], s=34, color=color, marker="D", zorder=4)
    axes[1].set_ylabel("LPR（%）")
    axes[1].legend(frameon=False, loc="upper right", ncol=2)
    axes[1].grid(axis="y", alpha=.15)
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[1].text(.02, 1.07, "24 次月度记录：20 次两档均未变，4 次至少一档下降。\n空心圆 = 该期限未变；菱形 = 该期限下调。无事前共识，不能判断‘符合预期’。", transform=axes[1].transAxes, fontsize=9.5, color="#59616d", linespacing=1.6, va="bottom")
    fig.text(.08, .032, "两年目录覆盖范围固定，全部条目保留。2024 年英文年度目录只有186个入口，官方另称全年超过190场，故不宣称发布会全集。", fontsize=9.3, color="#5a6572")
    output(fig, "官方目录覆盖与全部LPR不变事件")


def report(nodes: pd.DataFrame) -> None:
    findings = """# 宏观政策信息、市场状态与持续更新：本阶段结论

**宏观研究已扩展到利率、地产、资本市场、财政化债、消费支持和贸易政策。公开资料支持把这些机制分别建账，也明确显示仅用M1/M2剪刀差不能代表全部政策信息；目前仍没有证据把这一结构直接转成可靠的510300未来方向判断。** 本阶段完成490条固定目录记录与六条政策链的24个来源节点，给出完整走势及信息时钟图。新模型、账户和卖点比较均为0，不据图形相关性宣布策略有效。

## 图表阅读

1. `figures/510300_货币数据与六类政策_完整走势.png`：2024—2025全部交易日，510300收盘价、当时已知M1/M2、实际与调查预期剪刀差、全部24个政策节点。新旧M1口径分开，月份数据按公布时点进入；政策点不代表买卖点。
2. `figures/六条政策链_宣布细则执行进度.png`：六个通道分别说明以前已知、此次新增和仍未知的内容。
3. `figures/贸易政策_盘前消息与收盘细节时钟.png`：确认会谈、公布进展、公布具体条款是三个不同的信息更新。日线不识别盘中因果。
4. `figures/官方目录覆盖与全部LPR不变事件.png`：490条目录记录按月全保留，24次LPR中的20次不变也全保留。

## 五个会改变后续研究的事实

**政策量不能直接加总为股票资金。** 互换便利计划额度5000亿元，10月18日披露的申请超2000亿元，10月21日的首次操作500亿元，是同一工具的不同阶段；12月31日所称实际投放超过90%，对应首批操作，不是5000亿元总额度。回购增持再贷款是另一项工具。必须分别研究计划、融资与实际投资，不能把额度、申请和操作相加。来源：[证监会启动公告](https://www.csrc.gov.cn/csrc/c100028/c7513113/content.shtml)、[央行首次操作](https://www.pbc.gov.cn/zhengcehuobisi/125207/125213/125431/5481510/5483261/index.html)、[证监会进度披露](https://www.csrc.gov.cn/csrc/c100028/c7529480/content.shtml)。

**财政用途决定传导路径。** 6万亿元化债额度用于置换存量隐性债务，分三年安排；它可能缓解利息和现金流约束，不能等同于当期6万亿元新增最终需求。2025年1月10日披露2024年2万亿元在12月18日发行完毕，是执行日期与信息公布日期不同的例子。该来源不能被倒填到12月18日。来源：[11月财政说明](https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/202411/t20241109_3947230.htm)、[1月执行进度](https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/202501/t20250110_3951525.htm)。

**消费政策的新增内容也会变化。** 2024年7月资金安排是在既有两新框架上加力；2025年1月则扩展家电与数码支持范围。应分别验证资金使用、消费数量、价格与盈利的传导，不能把所有发布会都记成一个同方向强度。国家发改委微信公众号发布的政策已通过政府网官方转载取得，保留来源和字节身份。来源：[发改委微信政策原文官方转载](https://app.www.gov.cn/govdata/gov/202407/25/517625/article.html)、[2025年扩围说明](https://www.ndrc.gov.cn/fzggw/wld/zcx/lddt/202501/t20250108_1395588.html)。

**市场可能在细则前已获得重要信息。** 2025年5月7日06:15已确认会谈；5月12日05:30已披露实质性进展；15:00公布具体关税安排。不能把5月12日全天行情都归给下午文本。反过来，也不能在5月7日预测中提前使用5月12日的关税降幅。来源：[会谈确认](https://www.mofcom.gov.cn/xwfb/xwfyrth/art/2025/art_89f7cb660ffe49e6a70bfb778ede6e22.html)、[盘前进展](https://www.mofcom.gov.cn/xwfb/ldrhd/art/2025/art_0adb7ac71bca40f2ab9af8e3d1eefcbb.html)、[条款说明](https://www.mofcom.gov.cn/syxwfb/art/2025/art_9748270381e54001bc291213ec6ee778.html)。

**没有政策变化，也可能有预期差。** 两年24次LPR记录里，20次两档均未变、4次至少一档下调。若市场事前期待下调，未变可能意味着失望；若原本期待不变，则不能仅凭不变得到同样结论。本轮政策节点未取得可认证的事前调查共识，预期与意外字段保留UNKNOWN，没有填成0。已有M1/M2调查预期仍单独保留，其五日失败不因这次整理而改变。

## 当时状态应怎样进入判断

每个节点保留复核收盘时已公开的M1/M2月份、口径、新订单月份、已采集操作利率记录，以及截至该收盘的价格涨幅、波动和过去60日价格回撤。这些是可观察上下文，没有被组合成评分或交易。利率的25条旧操作记录可能晚于宣布，不能取代本轮政策信息链。

例如，9月24日可用的货币信息仍是8月数据，旧口径M1为−7.3%、M2为6.3%、剪刀差−13.6个百分点，新订单48.9。当天新的政策改变了未来条件的信息，不能要求这些尚未更新的月度实际值先转好才承认消息出现。但这一事实本身不证明当天或随后应该买入。

收益判断需要区分盈利与增长、贴现与融资条件、风险溢价和资金约束；“超预期就涨、低预期就跌”应保留为有条件的假设。通胀数据可能同时影响成本、实际利率和政策空间，本轮尚未建立完整通胀预期与当次公布历史，不能用其他通道代替。

## 目前可以得出的研究结论与未完成项

可以确定：原来把研究收窄为一个货币预期差与剩余五日收益是不充分的；跨政策传导、重复公告、政策替代、实施和期限会改变信息解释。六条链提供可复核的数据组织和动态更新输入。

不能确定：这些新增字段是否对510300的20日收益、下行风险或持续持有决策有可靠正增量。490是目录记录数，不是独立事件数；24是人工核对的来源节点，不是完整政策母集。目录还需跨来源去重与多动作拆分，英文目录和年终大事记本身不能作为历史当时的交易输入。2024年年度英文页只有186项，而官方另称全年超过190场，范围差异明确保留。

当前信号为 **NO_VIEW_NO_VALIDATED_POLICY_RULE**。这表示没有通过验证的政策交易规则，不代表预测下跌，也不代表要卖出现有持仓。20万元主账户、2万元成本对照和上涨后回撤退出均为NOT_RUN。两个旧失败子实验及85/15终止状态保留。独立前向事件仍为0，完整账户年化10%/净夏普1.2未建立。

下一步必须完成的不是继续给这24个节点打分，而是：①按固定机构/公告系列定义可覆盖母集并追溯首发与更新版本；②把事前共识、实际政策参数和市场响应分开；③选择一项独立机制，预先规定收益或风险目标、简单基准与状态交互；④检验通过自己的门后，才比较每周与公告触发的连续账户，并固定入场比较回撤退出。资金模块的资料缺口不阻塞独立财政、利率或增长假设。不得根据这几段行情改方向、窗口或阈值。
"""
    (OUT / "研究结论.md").write_text(findings, encoding="utf-8")
    coverage = [
        ["货币数据", "104个月实际值；77个月成对调查预期，旧研究完整保留", "新旧M1不拼阈值；调查预期并非全部市场定价", "特定五日实现冻结失败"],
        ["增长状态", "139个月新订单；已完成20日收益/风险子实验", "不能把失败模型改窗口或缩尾救回", "两项固定模型各自冻结失败"],
        ["通胀", "政策目录导航尚不足以建立CPI/PPI当次值与共识序列", "缺完整原始发布时间与事前预期", "NOT_RUN_INDEPENDENT_CHANNEL"],
        ["利率与流动性", "137条央行年度目录；24次LPR；宣布实施链", "操作日非首发；未有完整事前政策概率", "DATA_PARTIAL_NO_VALIDATED_RULE"],
        ["财政与消费", "国新办目录；化债与两新事实及执行链", "额度、发行、实际使用与新增需求分开", "DATA_PARTIAL_NO_VALIDATED_RULE"],
        ["地产", "再贷款初始额度及资金支持比例变化", "缺完整收购与贷款进度、当时预期", "DATA_PARTIAL_NO_VALIDATED_RULE"],
        ["资本市场政策", "两工具宣布、细则、操作和进度", "操作金额不等于逐日净买股；母集待补", "DATA_PARTIAL_NO_VALIDATED_RULE"],
        ["外部与贸易", "税率替代、会谈预告、进展和具体安排时钟", "中美双方政策全集、有效期与后续延长尚未完整", "DATA_PARTIAL_NO_VALIDATED_RULE"],
        ["真实资金行为", "继承前轮份额等资料与其时钟缺口", "自己的缺口只约束资金模块", "NOT_RUN_OWN_DATA_GATE"],
        ["动态账户与退出", "固定下一开盘时钟，用户偏好涨幅+回撤", "新增模块尚无合格预测增量", "NOT_RUN_OWN_ACCEPTANCE_GATE"],
    ]
    pd.DataFrame(coverage, columns=["通道", "已经有的证据", "关键缺口或限制", "状态"]).to_csv(OUT / "results/宏观主线_证据与未完成项.csv", index=False, encoding="utf-8-sig")
    save(OUT / "results/chart_manifest.json", {"created_at": now(), "charts": [p.name for p in (OUT / "figures").glob("*.png")], "all_daily_points_retained": True, "plots_are_account_nav": False, "policy_event_returns_computed": 0, "source_nodes": len(nodes)})


if __name__ == "__main__":
    market, money, nodes = data()
    overview(market, money, nodes)
    chain_diagram(nodes)
    timing_chart(market, nodes)
    catalog_chart()
    report(nodes)
    print("四张图、完整日频对齐数据、跨通道结论与未完成项已生成。")
