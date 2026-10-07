"""把保存的日周线研究结果转为点位卡、图表和中文报告。"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fixed_weekly_points", ROOT / "research/weekly_daily_entry_locations_v1.py")
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
OUT = M.OUT
FIG = OUT / "figures"


def read(name):
    return pd.read_parquet(OUT / "results" / (name + ".parquet"))


def pct(x, digits=2):
    return "未计算" if pd.isna(x) else f"{x * 100:.{digits}f}%"


def number(x, digits=2):
    return "未计算" if pd.isna(x) else f"{x:.{digits}f}"


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def link(path, name):
    return f"[{name}](<{Path(path).resolve().as_posix()}>)"


def savefig(fig, name):
    fig.savefig(FIG / (name + ".png"), dpi=160)
    fig.savefig(FIG / (name + ".svg"))
    plt.close(fig)


def style():
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10, "axes.titlesize": 12, "axes.edgecolor": "#b9c6d1", "figure.facecolor": "#f7f9fb", "axes.facecolor": "white", "savefig.facecolor": "#f7f9fb", "text.color": "#1b3248", "axes.labelcolor": "#38546b", "xtick.color": "#52687a", "ytick.color": "#52687a"})


def candles(ax, frame, cols):
    for i, r in enumerate(frame.itertuples(index=False, name=None)):
        o, h, l, c = [float(r[frame.columns.get_loc(k)]) for k in cols]
        color = "#bc6554" if c >= o else "#4d857c"
        ax.vlines(i, l, h, color=color, lw=.85)
        ax.add_patch(Rectangle((i - .31, min(o, c)), .62, max(abs(c - o), .0004), facecolor=color, edgecolor=color, linewidth=.7))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.12)


def point_panels(cases, d, w, name):
    n = len(cases)
    height = 4.2 * n + 2.3
    fig, axes = plt.subplots(n, 2, figsize=(15.6, height), squeeze=False)
    fig.subplots_adjust(left=.065, right=.98, top=1 - 1.60 / height, bottom=.85 / height, hspace=.45, wspace=.15)
    fig.text(.045, 1 - .28 / height, "510300｜周线定位支撑，日线确认可接受的入场价", fontsize=20, weight="bold", va="top")
    fig.text(.045, 1 - .65 / height, "所有图仅使用日线与周线；价格统一到各次入场日的分红基准。横线在信号形成时已经确定。", fontsize=10, va="top", color="#52687a")
    for index, r in enumerate(cases.itertuples()):
        cash = float(d.cash_shift.iloc[int(r.entry_idx)])
        left = w[(w.last_date >= pd.Timestamp(r.support_pivot_date) - pd.Timedelta(weeks=22)) & (w.first_date <= r.exit_date)].copy().reset_index(drop=True)
        right = d.iloc[max(0, int(r.setup_idx) - 8):min(len(d), int(r.exit_idx) + 4)].copy().reset_index(drop=True)
        for col in ["open", "high", "low", "close"]:
            left[col] = left[col] - cash
        for col in ["ao", "ah", "al", "ac"]:
            right[col] = right[col] - cash
        a, b = axes[index]
        candles(a, left, ["open", "high", "low", "close"])
        candles(b, right, ["ao", "ah", "al", "ac"])
        for ax in [a, b]:
            ax.axhline(r.support_index - cash, color="#71899d", lw=.9, ls="--")
            ax.axhline(r.entry_target_raw, color="#558473", lw=1.1, ls="--")
            ax.axhline(r.entry_stop_raw, color="#b75c5c", lw=1., ls=":")
            ax.axhline(r.max_entry_raw_at_open, color="#c29147", lw=1., ls="-.")
            ax.set_ylabel("价格（元）")
        pivot = int(left.index[left.last_date.eq(r.support_pivot_date)][0])
        confirmed = int(left.index[left.last_date.eq(r.support_confirm_date)][0])
        a.scatter(pivot, r.support_index - cash, color="#7e709b", marker="v", s=50, zorder=5)
        a.axvline(confirmed, color="#7e709b", alpha=.55, ls=":")
        a.set_title(f"{r.entry_date:%Y-%m-%d}｜周线\n低点形成 {r.support_pivot_date:%m-%d} → 两周后确认 {r.support_confirm_date:%m-%d}", loc="left", pad=9)
        signal_i = int(right.index[right.date.eq(r.signal_date)][0])
        entry_i = int(right.index[right.date.eq(r.entry_date)][0])
        exit_i = int(right.index[right.date.eq(r.exit_date)][0])
        b.scatter(signal_i, float(right.ac.iloc[signal_i]), color="#3e617f", s=43, zorder=5)
        b.scatter(entry_i, r.entry_reference, color="#c58b3f", marker="*", s=115, edgecolors="white", linewidth=.4, zorder=6)
        b.scatter(exit_i, r.exit_reference, color="#3e617f", marker="D", s=42, zorder=6)
        b.axvline(signal_i, color="#3e617f", alpha=.35, ls="--")
        reason = "到期退出" if r.exit_reason == "TIME_20_SESSIONS" else "结构失效退出"
        b.set_title(f"日线｜计划净空间/风险 {r.planned_net_rr_at_open:.2f}倍\n实际压力净收益 {r.net_return:+.2%} · {reason}", loc="left", pad=9)
        b.text(.02, .96, f"最高开盘参考价 {r.max_entry_raw_at_open:.3f} · 入场原价 {r.entry_reference:.3f}\n失效线 {r.entry_stop_raw:.3f} · 阻力目标 {r.entry_target_raw:.3f}", transform=b.transAxes, fontsize=9, va="top", bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88})
        for ax, frame, col in [(a, left, "last_date"), (b, right, "date")]:
            ticks = np.unique(np.linspace(0, len(frame) - 1, 6).astype(int))
            ax.set_xticks(ticks)
            ax.set_xticklabels([pd.Timestamp(frame[col].iloc[j]).strftime("%y-%m-%d") for j in ticks], fontsize=9)
            ax.set_xlim(-1, len(frame))
    fig.text(.045, .42 / height, "横线：灰蓝＝周线支撑，绿色＝已知阻力目标，红色＝固定失效线，金色＝最高开盘参考价。● 日线确认  ★ 次日入场代理  ◆ 退出代理", fontsize=10)
    fig.text(.045, .14 / height, "失效与到目标均按收盘确认、下一可卖开盘处理；历史点位不代表当前报价或订单。计划2∶1不保证实际盈亏比或止损成交。", fontsize=9, color="#52687a")
    savefig(fig, name)


def summary_plot(stats, parent):
    x = stats[stats.cost.eq("STRESS") & stats.scope.eq("NONOVERLAPPING_CYCLES")].set_index("policy")
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.7), gridspec_kw={"width_ratios": [1, 1, 1.65]})
    fig.subplots_adjust(left=.055, right=.98, bottom=.23, top=.77, wspace=.30)
    fig.text(.055, .94, "高胜率与高实际盈亏比，需要同时成立", fontsize=21, weight="bold")
    fig.text(.055, .855, "日周线固定规则 · 压力成本 · 单仓不重叠事件 · 四项表达全部保留", fontsize=11, color="#52687a")
    labels = ["支撑确认\n19次", "加空间≥2\n4次", "加周MACD\n2次", "全部背景\n0次"]
    values = [x.loc[p, "win_rate"] * 100 for p in M.POLICIES]
    ax[0].bar(range(4), values, color=["#6588a0", "#bd7358", "#8a9eb1", "#dde3e8"])
    ax[0].axhline(55, color="#bdb0a0", ls="--")
    for i, value in enumerate(values):
        ax[0].text(i, value + 2 if np.isfinite(value) else 5, f"{value:.1f}%" if np.isfinite(value) else "未计算", ha="center", fontsize=9)
    ax[0].set(xticks=range(4), xticklabels=labels, ylim=(0, 90), ylabel="净收益为正的比例（%）", title="胜率")
    vals = [x.loc[p, "payoff_ratio"] for p in M.POLICIES]
    ax[1].bar(range(4), vals, color=["#6588a0", "#bd7358", "#8a9eb1", "#dde3e8"])
    ax[1].axhline(2, color="#bdb0a0", ls="--")
    for i, v in enumerate(vals):
        ax[1].text(i, v + .07 if np.isfinite(v) else .1, f"{v:.2f}" if np.isfinite(v) else "未计算", ha="center", fontsize=9)
    ax[1].set(xticks=range(4), xticklabels=labels, ylim=(0, 2.5), ylabel="平均盈利／平均亏损绝对值", title="实际盈亏比")
    parent = parent.sort_values("entry_date")
    ax[2].bar(range(len(parent)), parent.net_return * 100, color=np.where(parent.net_return.ge(0), "#618f85", "#bb705d"))
    ticks = list(range(0, len(parent), 3))
    ax[2].set(xticks=ticks, xticklabels=[parent.entry_date.iloc[i].strftime("%Y-%m") for i in ticks], ylabel="每次10万元预算的净收益（%）", title="全部19次：少数亏损幅度较大")
    ax[2].axhline(0, color="#bdc8d1", lw=.8)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", alpha=.12)
        a.tick_params(axis="x", labelsize=9)
    fig.text(.055, .085, "研究筛选线在收益计算前固定为胜率≥55%、实际盈亏比≥2、净均值>0；本轮没有同时通过者。", fontsize=10)
    fig.text(.055, .033, "19次父样本均值虽为正，95%块抽样区间仍跨零；4次空间组只用于案例发现。完整账户指标未计算。", fontsize=10, color="#52687a")
    savefig(fig, "胜率与实际盈亏比")


def write_report(result, stats, trades, cases, periods, backgrounds):
    main = stats[stats.cost.eq("STRESS") & stats.scope.eq("NONOVERLAPPING_CYCLES")].set_index("policy")
    p0, p1 = main.loc["P0_CONFIRM"], main.loc["P1_SPACE"]
    summary_rows = []
    for p in M.POLICIES:
        r = main.loc[p]
        summary_rows.append([M.NAMES[p], int(r.events), f"{int(r.wins)}/{int(r.losses)}", pct(r.win_rate, 1), pct(r.avg_win), pct(r.avg_loss), number(r.payoff_ratio), pct(r.mean_net)])
    point_rows, case_notes = [], []
    for r in cases.itertuples():
        target_distance = r.entry_target_raw / r.entry_reference - 1
        point_rows.append([f"{r.entry_date:%Y-%m-%d}", f"{r.signal_date:%m-%d}", f"{r.max_entry_raw_at_open:.3f}", f"{r.entry_reference:.3f}", f"{r.entry_stop_raw:.3f}", f"{r.entry_target_raw:.3f}", f"{r.planned_net_rr_at_open:.2f}", f"{r.exit_date:%Y-%m-%d}", pct(r.net_return)])
        case_notes.append(f"- **{r.entry_date:%Y-%m-%d}**：周低点形成于{r.support_pivot_date:%Y-%m-%d}，{r.support_confirm_date:%Y-%m-%d}完成右侧两周确认；日线确认在{r.signal_date:%Y-%m-%d}。目标距入场价{pct(target_distance)}，持有{int(r.holding_sessions)}个交易日后{'因结构失效' if r.exit_reason == 'CLOSE_BELOW_STRUCTURE' else '因预设20日期限'}退出，实际净收益{pct(r.net_return)}，为计划风险的{r.realized_multiple_of_planned_risk:+.2f}倍。")
    parent = trades[trades.policy.eq("P0_CONFIRM") & trades.cost.eq("STRESS") & trades.sequential_eligible].sort_values("entry_date")
    parent_rows = [[f"{r.signal_date:%Y-%m-%d}", f"{r.entry_date:%Y-%m-%d}", f"{r.entry_reference:.3f}", number(r.planned_net_rr_at_open), f"{r.exit_date:%Y-%m-%d}", {"TIME_20_SESSIONS": "20日到期", "CLOSE_REACHES_TARGET": "收盘到目标", "CLOSE_BELOW_STRUCTURE": "收盘失效"}[r.exit_reason], pct(r.net_return)] for r in parent.itertuples()]
    period_rows = []
    for p in ["P0_CONFIRM", "P1_SPACE"]:
        z = periods[periods.policy.eq(p) & periods.cost.eq("STRESS") & periods.scope.eq("NONOVERLAPPING_CYCLES")]
        for r in z.itertuples():
            period_rows.append(["支撑确认" if p == "P0_CONFIRM" else "加空间≥2", r.period, int(r.events), pct(r.win_rate, 1), number(r.payoff_ratio), pct(r.mean_net)])
    verification = json.loads((OUT / "saved_verification.json").read_text(encoding="utf-8"))
    text = f"""# 510300日周线入场点位：高胜率线索存在，尚未同时获得高实际盈亏比

已经按用户最新要求改为**只用日线、周线**。本轮没有读取分钟数据，没有沿用上一轮分钟FVG信号。研究对象是周线已确认支撑附近，日线测试后收回并转强的点位；进一步检验在入场前限制价格、要求计划净收益空间至少为结构风险两倍，能否得到高胜率与高实际盈亏比。

**结果：周线支撑＋日线确认有一条高胜率历史线索，但还没有找到同时满足两项目标的可靠点位规则。** 不重叠的19次父事件，压力成本后14赢5输，胜率{pct(p0.win_rate, 1)}；平均盈利{pct(p0.avg_win)}、平均亏损{pct(p0.avg_loss)}，实际盈亏比仅{number(p0.payoff_ratio)}。增加计划净空间/风险≥2后只剩4次，2赢2输、胜率50%、实际盈亏比{number(p1.payoff_ratio)}，样本不足以称规律。

主方案状态`REJECTED_FIXED_PRIMARY_JOINT_POINT_SCREEN`。完整账户夏普、年化和回撤未计算，项目盈利目标没有据此达成。潜在空间、真实收益、概率估计三者分开保存。

## 这次点位怎样确定

周线负责定位和背景，日线负责确认。周线低点必须严格低于左右各两周的最低价，等右侧两周全部完成后才确认；从下一周起可被日线使用。每个日线决策只读取上一自然交易周及更早周线，周五也不提前使用本周周线。没有把事后看见的底部回填成当时可买的日期。

最新已确认周低点作为支撑，最多保留52周。每个支撑锚点只启动一次：价格从支撑上方接近到0.5个前日ATR20范围内，开始观察。准备日及其后5个交易日内，收盘收回支撑、超过前一日最高且当日为阳线，形成日线确认；先跌破支撑2个准备ATR或超时则保留为失败。

确认时固定两条边界：失效线为本次测试至确认的最低价减0.5个准备ATR；上方目标为准备时已经确认、最近52周内且高于准备日收盘的周线高点中价格最低的一个，即最近的上方阻力。没有可识别上方阻力时不编造目标价。上述价格坐标只向前平移已发生现金分红，真实成交使用原始报价。

**先求最高开盘参考价，再评价次日开盘是否允许入场。** 价格上限按压力费用、滑点和价位取整计算，使“到目标的计划净盈利／到失效线的计划净亏损”至少为2。原始开盘价超过上限即放弃本次机会，不假设当日之后低价补买。这里约束的是模型输入的原始开盘参考价，成交代理另加不利滑点；原冻结文字中的“限价”应按此口径理解，并非券商限价委托或成交保证。2∶1是事前价格几何，不是概率或保证损失。

P0只要求结构区间内入场；P1加2倍计划净空间，是主研究方案；P2在P1上加周MACD柱回升；P3再加日MACD柱回升、相对低波动和相对放量。四项表达在本轮收益计算前固定，完整报告，没有挑出最好子组替换主方案。

## 日周线和指标的时点

| 信息 | 本轮口径 |
| --- | --- |
| 周MACD | 上一完整周的12/26/9柱值较前一完整周回升，至少130周预热 |
| 日MACD | 日线确认收盘后的12/26/9柱值较前日回升 |
| 波动率 | 确认日前一交易日RV20／此前252日中位数，至少126日；≤1为相对低波动 |
| 成交量 | 确认日量／此前20日量中位数；≥1为相对放量 |
| 入场 | 确认后的下一交易日开盘代理；涨停不假设买到 |
| 退出 | 收盘跌破固定失效线、收盘到达目标，或持有满20交易日，下一可卖开盘退出 |

股票ETF实行T+1；入场当日出现收盘失效，也最早在下一交易日开盘退出。跌停开盘无法按模型卖出时顺延。全程没有假设日内刚碰止损线就一定成交，也不需要分钟K线猜测高低点先后。[上交所股票ETF交易说明](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml)

20个交易日是本轮预先选定、约一个月的修复观察期限。用户指定了日周线观察，没有指定持有期限；这是本轮的实现选择。结果不能外推到持有数月的所有周线波段，也没有在看到到期退出较多后把期限改长。

## 全部四项结果

{table(['固定表达', '不重叠次数', '盈利/亏损次数', '胜率', '平均盈利', '平均亏损绝对值', '实际盈亏比', '平均净收益'], summary_rows)}

以上为每次10万元独立预算的压力费用后收益，不是20万元连续账户回报。单边压力佣金4bp、滑点10bp，基准为佣金2bp、滑点5bp；最低佣金5元，100份整数手，最小价位0.001元，现金分红按登记权益计入。完整两档结果见保存表。

本轮事先把“较高”操作化为压力净胜率至少55%、实际盈亏比至少2、平均净收益为正；这些是研究筛选线，用户没有指定55%或2的硬阈值。所有样本都报告，不以年度次数做门槛。P0胜率较高但实际盈亏比偏低；P1平均净收益{pct(p1.mean_net)}，但两项联合标准都未通过。去掉P1最好的一笔后，剩余均值为{pct(p1.mean_without_best)}。

P0的利润因子为{number(p0.profit_factor)}，它是“总盈利／总亏损”，不能替代实际盈亏比{number(p0.payoff_ratio)}。例如14次小盈利可以让总利润为正，同时每次平均亏损仍大于每次平均盈利。

P0胜率的描述性Wilson区间约[{pct(p0.win_rate_lower95, 1)}, {pct(p0.win_rate_upper95, 1)}]；净均值的20日块抽样区间约[{pct(p0.net_mean_lower95)}, {pct(p0.net_mean_upper95)}]，仍跨零。P1只有4次，胜率区间约[{pct(p1.win_rate_lower95, 1)}, {pct(p1.win_rate_upper95, 1)}]，按事先规则不报告均值和盈亏比抽样区间。这些都不构成独立盈利验证。

![胜率和实际盈亏比](<{(FIG / '胜率与实际盈亏比.png').as_posix()}>)

## 四个满足计划净空间要求的具体历史点位

以下都是历史研究点位。表中失效线是收盘判断边界，实际卖价可因下一开盘跳空更差；目标是当时已知的阻力价，不是已实现收益。价格展示到0.001元，计算保留全部精度。

{table(['入场日', '确认日', '最高开盘参考价', '入场原价', '失效线', '阻力目标', '计划净空间/风险', '实际退出日', '压力净收益'], point_rows)}

{chr(10).join(case_notes)}

**四次中没有一次按预设“收盘到阻力目标”退出**：三次在20日后退出，一次结构失效。2016年的阻力距离入场约27.6%，价格空间看起来很大，却没有在固定持有期兑现；2018-09这笔计划净空间/风险约4.02，实际失效退出损失达计划风险的约1.57倍。日周线研究必须把目标距离、实现时间和跳空风险放在一起看。

![四次点位的周线与日线对照](<{(FIG / '四次历史点位日周线对照.png').as_posix()}>)

## 三个指标背景提供了什么信息

在4次满足空间要求的事件里，4次都已出现日MACD柱回升和相对放量，所以这两项没有再区分赢家和输家。周MACD回升留下2次，一赢一输，实际盈亏比1.34；没有提高胜率。4次的先前RV20都高于各自历史中位数，加入“低波动”后直接变成0次，不能将无交易写成零收益或高安全性。

这提示一个受限的研究方向：价格修复时的较大上方空间，可能出现在之前波动较高的背景中；低波动并不自动适合所有反转点位。但4次远不足以据此反向改成“只做高波动”，本轮也没有这样调参。详细背景分组全部保存。

## 分段结果与完整母样本

{table(['表达', '固定时段', '次数', '胜率', '实际盈亏比', '平均净收益'], period_rows)}

所有历史截至2026-09-16，2026为不完整年度。分段按日历事先固定，不能把较好阶段拼接成新策略。P0在2024—2026只有2次、一赢一输，平均为负；没有证据说明早期高胜率已在近期稳定延续。

共重建47个周支撑测试过程：25次有已知目标的确认、1次确认但无可用阻力、13次超时、7次结构失败、1次资料终点仍在观察。26次确认中，20次具有可评价入场；5次下一开盘已不在固定失效与目标区间内、1次没有目标。20次事件有1次与已有持有期重叠，单仓顺序下留下19次，未删掉失败过程来提高胜率。

### 19次不重叠父事件逐笔结果

{table(['确认日', '入场日', '原始入场价', '计划净空间/风险', '退出日', '退出原因', '压力净收益'], parent_rows)}

少数P0计划净空间为负，表示虽然原始价格处在失效线和目标之间，但按目标价退出的狭小距离不足以覆盖成本；P0作为不加空间约束的对照保留它们。它们的后续开盘恰好跳过原目标并盈利，不代表信号日可以预知这段跳空。P1已在事前价格上限中处理这类问题。

## 对我们研究的实际结论

日周线这条路可以保留“已确认支撑附近的日线收复”作为待研究的高胜率线索，尚不能称高胜率、高盈亏比兼得的规则。已有结果明确了三个约束：日线确认往往使买入价离低点更远；远阻力不保证在持有期限内到达；收盘失效后的开盘卖出会暴露于跳空，实际亏损可能超过计划风险。

本轮主方案不晋升，不为了得到好看的结果延长持有、缩小止损、删除输家或挑选周线背景。该结论仅限这一个支撑点位定义和固定退出方式，不证明所有日周线方法无效。项目目标净夏普1.2、年化10%、最大回撤10%仍未达成，完整账户保持`NOT_COMPUTED_EVENT_POINT_RESEARCH`。

数据使用既有3,479条日线，从2012-05-28起用于指标预热，2015年起评价点位；价格早已被项目多轮研究使用。最后本地资料日2026-09-16不能代表今天行情，当前为`NO_VIEW_STALE_LOCAL_DATA`。没有新增分钟或其他市场数据采集，没有订单或自动交易。

## 文件与复核

8项针对性测试通过；保存的{verification['return_rows']}条两档成本事件收益复算误差为0，12个真实历史截断位置的周线锚点、日线确认、MACD、量和最高买价一致；日周时点、T+1与持有期重叠检查通过。研究输入在计算前单独快照，后续共享进度配置更新不改变本轮输入。

- {link(OUT / 'protocol.json', '固定研究定义')}、{link(OUT / 'freeze.json', '冻结记录')}、{link(OUT / 'saved_verification.json', '保存结果复算')}。
- {link(OUT / 'results/四次满足空间的历史点位.csv', '四次点位完整字段')}、{link(OUT / 'results/四组全部交易与重叠标记.csv', '全部事件交易与重叠标记')}。
- {link(OUT / 'results/全部准备过程.csv', '全部47个过程')}、{link(OUT / 'results/逐日观察与失败.csv', '逐日确认与失败路径')}、{link(OUT / 'results/全部入场与成熟状态.csv', '全部26次确认的入场状态')}。
- {link(OUT / 'results/胜率实际盈亏比分组统计.csv', '两档成本、全部事件与不重叠统计')}、{link(OUT / 'results/背景逐项观察.csv', '背景逐项观察')}。
- {link(ROOT / 'research/weekly_daily_entry_locations_v1.py', '完整研究代码')}、{link(ROOT / 'tests/test_weekly_daily_entry_locations_v1.py', '时序与收益测试')}、{link(Path(__file__), '报告图表实现')}。
"""
    path = OUT / "日周线点位研究结论.md"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    M.frozen_check()
    FIG.mkdir(exist_ok=True)
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    trades, d, w, stats, periods, backgrounds = [read(x) for x in ["四组全部交易与重叠标记", "日线及当时可用周背景", "周线与枢轴确认", "胜率实际盈亏比分组统计", "固定时段统计", "背景逐项观察"]]
    cases = trades[trades.policy.eq("P1_SPACE") & trades.cost.eq("STRESS") & trades.sequential_eligible].sort_values("entry_date").copy()
    parent = trades[trades.policy.eq("P0_CONFIRM") & trades.cost.eq("STRESS") & trades.sequential_eligible]
    M.save_table("四次满足空间的历史点位", cases)
    style()
    point_panels(cases, d, w, "四次历史点位日周线对照")
    point_panels(cases.tail(1), d, w, "最近一次历史点位日周线对照")
    summary_plot(stats, parent)
    report = write_report(result, stats, trades, cases, periods, backgrounds)
    M.save_json(OUT / "entry_ceiling_semantics.json", {"noted_at": M.now(), "scope": "文字澄清，不改变冻结代码、规则、输入或收益", "original_word": "限价", "implemented_semantics": "信号日计算原始开盘参考价的最高允许值；次日原始开盘通过后，用加滑点及价位取整的成交代理计算收益与计划风险。不是券商限价单成交证明。", "backtest_rerun": False})
    M.save_json(OUT / "report_receipt.json", {"generated_at": M.now(), "report": str(report), "report_sha256": M.digest(report), "result_sha256": M.digest(OUT / "result.json"), "report_code_sha256": M.digest(__file__), "historical_point_cards": len(cases), "minute_data_reads": 0, "new_strategy_runs": 0, "strategy_goal_achieved": False})
    print(json.dumps({"报告": str(report), "历史点位卡": len(cases), "图表": str(FIG)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
