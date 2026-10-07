"""交付上涨/下跌图谱与点位比较，按用户最新频率软目标解释。"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import upward_episode_anatomy_v1 as a
from research import directional_entry_timing_v1 as b
from research import weekly_anchor_bidirectional_v1 as c

OUT = ROOT / "reports/research/510300_directional_research_20261001"


def pct(value):
    return f"{value:.2%}" if np.isfinite(value) else "未定义"


def markdown(frame, columns, labels, formats=None):
    result = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(labels)) + "|"]
    formats = formats or {}
    for _, row in frame.iterrows():
        cells = [formats[key](row[key]) if key in formats else str(row[key]) for key in columns]
        result.append("| " + " | ".join(cells) + " |")
    return result


def figures(d, episodes, signals, point_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((name for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"] if name in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "figure.facecolor": "white"})
    chosen = []
    for direction in [1, -1]:
        chosen.append(episodes.loc[episodes.admitted & episodes.direction.eq(direction) & episodes.start_date.ge("2020-01-01") & episodes.status.eq("COMPLETE_RETROSPECTIVE")].iloc[0])
    fig, axes = plt.subplots(4, 2, figsize=(15, 11), sharex="col", gridspec_kw={"height_ratios": [3.1, 1.1, 1.3, 1.1]})
    for col, ep in enumerate(chosen):
        begin, end = int(ep.start_idx), int(ep.end_idx)
        g = d.iloc[max(0, begin - 20):min(len(d), end + 11)]
        ax = axes[0, col]
        ax.plot(g.date, g.close, color="#244c70", lw=1.8, label="510300原始收盘价")
        ax.plot(g.date, g.ema20 - g.cash_shift, color="#b98218", lw=1.2, label="EMA20（换回当日报价尺度）")
        ax.axvspan(ep.start_date, ep.end_date, alpha=.08, color="#217a60" if ep.direction == 1 else "#b44646")
        for idx, label, color in [(begin, "事后起点", "#707070"), (int(ep.confirm_idx), "5%确认", "#7c539b"), (end, "事后终点", "#707070")]:
            ax.axvline(d.date.iloc[idx], color=color, alpha=.7, linestyle="--", lw=1)
            ax.annotate(label, (d.date.iloc[idx], d.close.iloc[idx]), xytext=(0, 12 if label != "5%确认" else -22), textcoords="offset points", ha="center", fontsize=9, color=color)
        colors = {"MOMENTUM_TURN": "#14836b", "EMA20_CROSS": "#b98218", "RANGE20_BREAK": "#9f467d"}
        marks = {"MOMENTUM_TURN": "o", "EMA20_CROSS": "s", "RANGE20_BREAK": "D"}
        for family, name in b.FAMILIES.items():
            sig = signals.loc[signals.family.eq(family) & signals.direction.eq(ep.direction) & signals.signal_idx.between(begin, end)]
            if len(sig):
                first = sig.iloc[0]
                ax.scatter(first.signal_date, d.close.iloc[int(first.signal_idx)], s=46, marker=marks[family], color=colors[family], zorder=6, label=name+"首次信号")
        direction_name = "上涨" if ep.direction == 1 else "下跌"
        ax.set_title(f"{direction_name}案例：{ep.start_date:%Y-%m-%d} → {ep.end_date:%Y-%m-%d}\n事后含息幅度 {ep.directional_move:.1%}，{int(ep.sessions)}个交易日", loc="left", fontsize=12, pad=14)
        ax.set_ylabel("报价（元）")
        ax.legend(loc="best", fontsize=8, ncol=1, framealpha=.85)
        colors_daily = np.where(g.return1.ge(0), "#a75555", "#388171")
        axes[1, col].bar(g.date, g.relative_volume, color=colors_daily, width=1.6, alpha=.8)
        axes[1, col].axhline(1.5, ls="--", color="#666666", lw=1)
        axes[1, col].set_ylabel("相对成交量")
        axes[2, col].bar(g.date, g.hist_atr, color=np.where(g.daily_hist.ge(0), "#a75555", "#388171"), width=1.6, alpha=.85)
        axes[2, col].axhline(0, color="#555555", lw=.8)
        axes[2, col].set_ylabel("日MACD柱 / ATR")
        axes[3, col].step(g.date, g.weekly_hist, where="post", color="#6b4b90", lw=1.5)
        axes[3, col].axhline(0, color="#555555", lw=.8)
        axes[3, col].set_ylabel("上一完成周\nMACD柱")
        axes[3, col].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        axes[3, col].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        axes[3, col].tick_params(axis="x", rotation=25)
    fig.suptitle("先看波段，再检验信号：事后极值与当时可见确认分开", fontsize=17, x=.06, ha="left")
    fig.text(.06, .022, "案例按日期选取2020年起首段上涨与首段下跌；没有按收益择优。图上信号收盘后才可知，参考入场在下一开盘。\n5%用于事后整理波段；不是策略买点或保证跌幅。全样本49段上涨、49段下跌，另有1段末期尚未确认结束。", fontsize=10, color="#555555")
    fig.subplots_adjust(top=.89, bottom=.10, left=.07, right=.98, hspace=.17, wspace=.20)
    fig.savefig(OUT / "上涨下跌技术量价案例.png", dpi=150)
    fig.savefig(OUT / "上涨下跌技术量价案例.svg")
    plt.close(fig)

    # 同一规则的成功和失败均展示，按时间挑第一条，不按幅度择优。
    for direction in [1, -1]:
        group = point_rows.loc[point_rows.policy.eq("BOTH") & point_rows.status.eq("COMPLETE") & point_rows.direction.eq(direction) & point_rows.entry_date.ge("2020-01-01")]
        chosen_points = [group.loc[group.net_reference_return.gt(0)].iloc[0], group.loc[group.net_reference_return.le(0)].iloc[0]]
        fig, axes = plt.subplots(3, 2, figsize=(15, 9), sharex="col", gridspec_kw={"height_ratios": [3, 1.1, 1.3]})
        for col, point in enumerate(chosen_points):
            start, end = int(point.entry_idx), int(point.exit_idx)
            g = d.iloc[max(0, int(point.signal_idx) - 15):min(len(d), end + 8)]
            ax = axes[0, col]
            ax.plot(g.date, g.close, color="#244c70", lw=1.7, label="原始收盘价")
            period = g.loc[g.date.between(point.signal_date, point.exit_date)]
            ax.plot(period.date, point.stop_index - period.cash_shift, color="#a75555", ls="--", lw=1.3, label="当时固定的失效线")
            ax.plot(period.date, point.target_index - period.cash_shift, color="#388171", ls="--", lw=1.3, label="当时已知的目标")
            ax.scatter(point.entry_date, point.entry_fill, marker="^" if direction == 1 else "v", color="#a77717", s=85, zorder=6, label="入场参考成交")
            ax.scatter(point.exit_date, point.exit_fill, marker="X", color="#694a89", s=75, zorder=6, label="退出参考成交")
            ax.axvspan(point.entry_date, point.exit_date, alpha=.06, color="#244c70")
            ax.set_title(f"{'盈利' if col == 0 else '亏损'}对照：{point.entry_date:%Y-%m-%d} → {point.exit_date:%Y-%m-%d}\n扣统一摩擦后 {point.net_reference_return:+.2%}", loc="left", fontsize=12)
            ax.set_ylabel("报价（元）")
            ax.legend(fontsize=8, framealpha=.85)
            axes[1, col].bar(g.date, g.relative_volume, width=1.6, color="#7a8c9d")
            axes[1, col].axhline(1.5, ls="--", color="#777777", lw=1)
            axes[1, col].set_ylabel("相对量")
            axes[2, col].bar(g.date, g.hist_atr, width=1.6, color=np.where(g.daily_hist.ge(0), "#a75555", "#388171"))
            axes[2, col].axhline(0, color="#555555", lw=.8)
            axes[2, col].set_ylabel("日MACD柱 / ATR")
            axes[2, col].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
            axes[2, col].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            axes[2, col].tick_params(axis="x", rotation=25)
        name = "多头点位成功与失败" if direction == 1 else "空头点位成功与失败"
        fig.suptitle(name + "：同一周线锚点规则", fontsize=17, x=.06, ha="left")
        fig.text(.06, .025, "按2020年起第一笔盈利、第一笔亏损选图。目标和失效在收盘确认后，下一可平仓开盘退出，因此不保证按画线价格成交。\n这是510300标的方向的参考损益；没有选择期权合约，不代表认购或认沽期权收益。", fontsize=10, color="#555555")
        fig.subplots_adjust(top=.87, bottom=.13, left=.07, right=.98, hspace=.18, wspace=.18)
        fig.savefig(OUT / f"{name}.png", dpi=150)
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(a.OUT / "results/features.parquet")
    episodes = pd.read_parquet(b.OUT / "results/多空波段全集.parquet")
    signals = pd.read_parquet(b.OUT / "results/所有候选点位.parquet")
    points = pd.read_parquet(c.OUT / "results/点位逐笔与拒绝原因.parquet")
    dense = pd.read_parquet(b.OUT / "results/胜率实际盈亏比.parquet")
    sparse = pd.read_parquet(c.OUT / "results/胜率实际盈亏比.parquet")
    annual_dense = pd.read_parquet(b.OUT / "results/逐年次数.parquet")
    annual_sparse = pd.read_parquet(c.OUT / "results/逐年次数.parquet")
    confidence = pd.read_parquet(c.OUT / "results/不确定区间.parquet")
    frequency = pd.read_parquet(c.OUT / "results/频率与等待.parquet")
    rows = []
    for kind, table, annual, key, names in [("高频比较", dense, annual_dense, "family", b.FAMILIES), ("结构点位", sparse, annual_sparse, "policy", c.POLICIES)]:
        for policy, name in names.items():
            subset = table.loc[table[key].eq(policy) & table.era.eq("ALL") & table.direction.eq(0)]
            net = subset.loc[subset.cost.eq("REFERENCE_FRICTION")].iloc[0]
            gross = subset.loc[subset.cost.eq("GROSS")].iloc[0]
            years = annual.loc[annual[key].eq(policy) & annual.full_year, "cycles"]
            product = float(net.win_rate * net.payoff)
            loss_rate = float(net.losses / net.cycles)
            expected_units = product - loss_rate
            assert abs(expected_units * abs(net.mean_loss) - net.mean_return) < 1e-12
            rows.append({"kind": kind, "name": name, "cycles": int(net.cycles), "annual_mean": float(years.mean()),
                         "annual_min": int(years.min()), "annual_max": int(years.max()), "zero_years": int(years.eq(0).sum()),
                         "gross_win": gross.win_rate, "gross_payoff": gross.payoff, "gross_mean": gross.mean_return,
                         "net_win": net.win_rate, "net_payoff": net.payoff, "net_mean": net.mean_return,
                         "net_p_times_payoff": product, "net_expectancy_loss_units": expected_units,
                         "net_profit_factor": product / loss_rate, "user_strict_product_gate_pass": product > 1 and net.mean_return > 0})
    compare = pd.DataFrame(rows)
    compare.to_csv(OUT / "六组点位质量与频率.csv", index=False, encoding="utf-8-sig")
    annual_sparse.pivot(index="year", columns="policy", values="cycles").rename(columns=c.POLICIES).to_csv(OUT / "周线多空逐年次数.csv", encoding="utf-8-sig")
    complete_both = points.loc[points.policy.eq("BOTH") & points.status.eq("COMPLETE")].copy()
    complete_both["direction_name"] = complete_both.direction.map({1: "多头", -1: "空头"})
    selected_columns = ["direction_name", "setup_date", "signal_date", "entry_date", "exit_date", "entry_raw", "entry_fill", "entry_stop_raw", "entry_target_raw", "exit_raw", "exit_fill", "exit_reason", "gross_return", "net_reference_return", "planned_reference_rr", "realized_r"]
    complete_both[selected_columns].rename(columns={"direction_name": "方向", "setup_date": "准备日", "signal_date": "确认日", "entry_date": "入场日", "exit_date": "退出日", "entry_raw": "入场原价", "entry_fill": "入场含滑点", "entry_stop_raw": "入场时失效线", "entry_target_raw": "入场时目标", "exit_raw": "退出原价", "exit_fill": "退出含滑点", "exit_reason": "退出原因", "gross_return": "毛方向回报", "net_reference_return": "扣参考摩擦回报", "planned_reference_rr": "入场计划盈亏比", "realized_r": "实现R"}).to_csv(OUT / "37笔双向点位明细.csv", index=False, encoding="utf-8-sig")
    figures(d, episodes, signals, points)
    stage = pd.read_parquet(a.OUT / "results/各阶段特征出现率.parquet").set_index("stage")
    timing = pd.read_parquet(a.OUT / "results/指标时差汇总.parquet").set_index("feature")
    short_timing = pd.read_parquet(b.OUT / "results/下跌确认时差.parquet").groupby("feature").agg(n=("present_in_wave", "sum"), lag=("lag_sessions", "median"), move=("move_already", "median"))
    timing_rows = []
    for feature, name in [("daily_hist_rising", "MACD柱朝交易方向变化"), ("daily_hist_positive", "日MACD进入对应金叉/死叉侧"),
                           ("above_ema20", "收盘越过EMA20"), ("breakout20", "突破/跌破前20日收盘区间")]:
        up, down = timing.loc[feature], short_timing.loc[feature]
        timing_rows.append({"name": name, "up_count": f"{int(up.in_wave)}/49", "up_lag": up.median_lag,
                            "up_used": up.median_gain_already, "down_count": f"{int(down.n)}/49", "down_lag": down.lag, "down_used": down.move})
    timing_df = pd.DataFrame(timing_rows)
    timing_df.to_csv(OUT / "上涨下跌指标时差.csv", index=False, encoding="utf-8-sig")
    main_table = markdown(compare, ["name", "cycles", "annual_mean", "net_win", "net_payoff", "net_p_times_payoff", "net_mean"],
                          ["固定点位规则", "完成周期", "完整年平均次数", "扣摩擦胜率p", "实际盈亏比B", "p×B", "单笔平均参考回报"],
                          {"cycles": lambda x: str(int(x)), "annual_mean": lambda x: f"{x:.2f}", "net_win": pct, "net_payoff": lambda x: f"{x:.3f}", "net_p_times_payoff": lambda x: f"{x:.3f}", "net_mean": pct})
    time_table = markdown(timing_df, ["name", "up_count", "up_lag", "up_used", "down_count", "down_lag", "down_used"],
                          ["观察状态", "上涨段出现", "上涨中位时差", "已涨中位幅度", "下跌段出现", "下跌中位时差", "已跌中位幅度"],
                          {"up_lag": lambda x: f"{x:g}日", "down_lag": lambda x: f"{x:g}日", "up_used": pct, "down_used": pct})
    years = annual_sparse.pivot(index="year", columns="policy", values="cycles").reset_index()
    year_table = markdown(years, ["year", "LONG_ONLY", "SHORT_ONLY", "BOTH"], ["入场年", "单做多", "单做空", "双向单一活跃点位"],
                          {key: (lambda x: str(int(x))) for key in years.columns})
    era = sparse.loc[sparse.policy.eq("BOTH") & sparse.direction.eq(0) & sparse.cost.eq("REFERENCE_FRICTION") & ~sparse.era.eq("ALL")]
    era_table = markdown(era, ["era", "cycles", "win_rate", "payoff", "mean_return"], ["固定时期", "周期", "胜率", "实际盈亏比", "平均参考回报"],
                         {"cycles": lambda x: str(int(x)), "win_rate": pct, "payoff": lambda x: f"{x:.3f}", "mean_return": pct})
    top2 = complete_both.nlargest(2, "net_reference_return")
    contribution = float(top2.net_reference_return.sum() / complete_both.net_reference_return.sum())
    ci = confidence.loc[confidence.policy.eq("BOTH")].iloc[0]
    report = [
        "# 510300多空点位研究：先解释波段，再比较点位质量与次数", "",
        "**已按最新要求取消每年5次硬门槛，并将扣参考摩擦后的胜率×实际盈亏比>1设为质量硬门槛。六组点位当前全部不合格。** 本轮只研究510300标的多空点位，次数作为希望提高的软目标。允许未来用期权表达方向，但本报告没有选择期权合约，也没有把标的收益叫作期权收益。", "",
        "本轮完成49段上涨、49段下跌的技术量价图谱，并计算三个确认时点的398个完整点位记录；它们是三个独立规则的结果，不能加总为一条策略的交易。随后保持原周支撑多头不变，新增对称周阻力空头，得到37个不重叠的双向结构点位。", "",
        "周支撑/周阻力在历史上呈现较高胜率，但乘积仍不足。加入空头后，完整年份平均次数从原多头的1.64提高到3.00，样本胜率从73.68%降至64.86%，实际盈亏比约0.85，乘积约0.550，未过新门槛；单笔参考净回报均值约+0.65%，近期转负，均值区间也跨零。频率增加和表面高胜率都不能代替质量验收。", "",
        "## 数学口径与最新门槛", "",
        "记p为盈利笔数/总笔数，q为亏损笔数/总笔数，B为平均盈利收益率/平均亏损收益率绝对值。标准数学期望（以平均亏损为1单位）是E=p×B−q；没有平手时q=1−p。标准正期望要求E>0，等价于利润因子p×B/q>1（q>0）。用户最新指定的p×B>1，比标准正期望要求更严格，本报告按该额外硬门槛执行，不能将三者混为一谈。", "",
        "例如胜率60%、实际盈亏比1.5时，p×B=0.9，标准期望E=0.5个平均亏损单位：正期望成立，但不符合本次额外的p×B>1。胜率以小数计算，盈亏比使用实际兑现且扣参考摩擦的结果，不使用图上计划目标代替。原先55%胜率或盈亏比2的工作比较线，已不再作为通过标准。", "",
        "## 1. 上涨确实能用指标描述，但不等于提前识别", "",
        "数据为2012-05-28至2026-09-16的3479根日线；正式观察2015年起。以收盘含息总回报指数的5%方向变化统一整理波段，覆盖49段已完成上涨、49段已完成下跌，另有1段下跌尚未确认结束。全部样本保留，没有挑最好看的几段。低点/高点是事后标签，确认日期单列，信号计算不读取这些标签。", "",
        *time_table, "",
        "表中时差从事后起点算起，取波段内首次观察到该状态的日期；起点已经满足记0。中位数仅在该状态出现的波段中计算，不能解释为能提前知道极值。日MACD柱变化与“柱的斜率发生转折”也不是同一个条件，后面的早期点位规则要求真正发生转折。", "",
        f"日MACD柱回升在上涨低点前5日仅出现{stage.loc['PRE5','daily_hist_rising']:.1%}，在低点后5日出现{stage.loc['POST5','daily_hist_rising']:.1%}，而在首次涨到5%的确认日为{stage.loc['CONFIRM5','daily_hist_rising']:.1%}。这主要描述价格动能已恢复，不能只从成功上涨段倒推它的胜率。", "",
        "MACD本身来自价格的指数均线差，12/26/9是本轮固定参数。其金叉、死叉与震荡时反复交叉的含义可见[Fidelity指标说明](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/macd)。本轮用实际失败对照检查区分能力，没有把教材用语当作510300的胜率证据。", "",
        "![上涨下跌案例](上涨下跌技术量价案例.png)", "",
        "成交量提供的线索也不稳定：49段上涨首次涨到5%的当天，只有16段达到此前20日中位量的1.5倍；另外33段没有。上涨日/下跌日成交量比较只是量价代理，不等于主动买卖订单流。关于量配合趋势的常见解释见[Schwab成交量说明](https://www.schwab.com/learn/story/trading-volume-as-market-indicator)，能否增加预测价值仍以本地对照为准。", "",
        "本轮还从全部2826个成熟交易日原点看未来20日，并在186次上穿EMA20的相同价格事件上逐一加入MACD、周背景、成交量或波动率条件。所有单项净均值增量的区块95%区间都跨零；不能据此把多指标叠加直接升格为策略。20日原点有重叠，另存固定相位非重叠对照；从未把2826条当作2826次独立交易。", "",
        "## 2. 三个确认时点：次数多，但点位优势不足", "",
        "早期转折：多头在日MACD柱仍为负时，柱变化从不升变为上升；空头镜像。均线确认：收盘向交易方向越过EMA20。区间确认：首次收盘突破/跌破此前20日收盘区间。这三项各自形成多空序列，同族只允许一个活跃点位。", "",
        "三项共用过去5日极值加0.5个前日ATR的失效线，信号收盘到失效距离的2倍作固定目标；次日开盘参考入场，收盘确认失效/目标或持有20日后，下一可平仓开盘退出。目标距离为2倍风险，不保证实际平均盈亏比等于2。", "",
        *main_table, "",
        "这里的实际盈亏比＝平均盈利点位收益率÷平均亏损点位收益率绝对值；不是计划目标/止损比，也不是利润因子。参考摩擦按单边万四佣金、千一滑点、最低5元及报价刻度计算，股息对多空分别加减。空头没有假设可以融券，也未计借券费或任何期权成本，所以仅是统一条件下的方向质量比较。毛回报、毛胜率和逐笔记录另表完整保存。", "",
        "普通技术确认的主要问题并不只是次数或手续费。MACD早期转折182笔中84笔失效，只有36笔达到预定目标；EMA20穿越101笔只有12笔达到目标。20日区间突破115笔中73笔最终是持有到期退出，只有13笔达到目标；其毛均值约+0.37%，扣参考摩擦仅余+0.07%，均值区间仍跨零。不能通过把计划盈亏比写为2，声称实际盈亏比已经很高。", "",
        "## 3. 结构点位：用双向机会增加次数", "",
        "多头沿用已有周支撑首次接近、日线转强规则。空头采用本轮首次测试的对称表达：接近已经确认的周阻力后，收盘回到阻力之下、跌破前日最低且为阴线；失效线是准备至确认的最高价加0.5个准备ATR，目标为准备时已经知道的最近下方周低点。周枢轴需要右侧2周确认，并到下一周才可使用。", "",
        "单独多头19笔、单独空头21笔；多空共用一个活跃点位后为37笔，不能直接相加为40笔。原多头的入场日、退出日及毛回报已逐笔核对一致，未因新空头结果调整旧参数。原旧主方案P1的失败及所有旧记录保留。", "",
        *year_table, "",
        "2026只统计到9月16日，不作为完整年平均分母。双向完整年平均3次，2022年为零，包含样本边缘的最长空档328个交易日。增加方向使机会增加，但仍有长时间无合格点位；本次没有放宽锚点或重复触发来凑次数。", "",
        *era_table, "",
        f"双向样本净均值的20日区块95%区间为{ci.mean_lower95:.2%}至{ci.mean_upper95:.2%}，跨零；胜率描述性区间约{ci.win_lower95:.1%}至{ci.win_upper95:.1%}。2024年至截止日只有8笔、4赢4亏，平均参考净回报约-0.61%。两个最大盈利点位合计占所有点位参考收益率加总的{contribution:.1%}，这是集中度诊断，不是账户收益归因。", "",
        "因此64.86%是历史样本胜率，尚不是可用于未来的稳定胜率估计。这条路线体现的是较高胜率、平均盈利小于平均亏损；按新要求p×B>1也不合格。大实际盈亏比路线本轮没有找到，不能因频率软化就将它改称合格策略。", "",
        "![空头成功和失败](空头点位成功与失败.png)", "",
        "![多头成功和失败](多头点位成功与失败.png)", "",
        "## 4. 反推规则目前落在哪里", "",
        "当前证据支持把研究问题收敛为：已确认周支撑/阻力附近的日线拒绝，为什么容易形成较多小盈利，但平均盈利没有覆盖平均亏损；以及更早或更晚的普通技术确认为什么也没有形成足够优势。已有结构点位只是诊断材料，未通过用户的乘积门槛、近期稳定性或均值不确定性检查。", "",
        "后续优先分析这37笔成功与失败在入场前的差别，以及目标距离、收盘确认与次日开盘之间的损耗；必须同时保留未发生交易的锚点失败过程。在没有额外证据前，不从最近8笔的负结果反向交易，不选最有利年代，不用日内高低点假定止损成交，也不改变固定等待期或目标做救援。", "",
        "## 5. 文件与复核", "",
        "- 六组质量/频率比较：`六组点位质量与频率.csv`。", "- 37笔双向点位，包括原价、参考成交、已知失效/目标、日期、实际R：`37笔双向点位明细.csv`。",
        "- 逐年次数：`周线多空逐年次数.csv`；上下行确认时差：`上涨下跌指标时差.csv`。",
        "- 完整上涨图谱、三个确认规则、周锚点双向实验的协议、代码、全量信号及拒绝过程保存在各自独立研究目录；`sources.json`记录路径和哈希。", "",
        "必要复核包含7项边界单元测试、历史前缀不变性、当前周禁用、原多头一致性与方向/股息损益重算。没有新采集分钟线，没有生成订单，也没有把这些检查称为独立预测验证。数据截止2026-09-16，本报告不提供2026-10-01的即时多空点位。", "",
    ]
    (OUT / "研究结论.md").write_text("\n".join(report), encoding="utf-8")
    source_paths = [a.OUT / "protocol.json", a.OUT / "result.json", a.OUT / "verification.json",
                    b.OUT / "protocol.json", b.OUT / "result.json", b.OUT / "verification.json",
                    c.OUT / "protocol.json", c.OUT / "result.json", c.OUT / "verification.json",
                    b.CONTEXT / "active_goal_effective_requirements.json",
                    a.OUT / "results/阶段快照.parquet", b.OUT / "results/点位逐笔与拒绝原因.parquet",
                    c.OUT / "results/点位逐笔与拒绝原因.parquet", Path(__file__)]
    a.save_json(OUT / "sources.json", {"at": a.now(), "files": {str(p.relative_to(ROOT)): a.digest(p) for p in source_paths},
                 "historical_data_reused": True, "independent_validation": False, "scope": "POINTS_ONLY_FREQUENCY_SOFT"})
    shutil.copyfile(b.CONTEXT / "active_goal_effective_requirements.json", OUT / "当前用户要求.json")
    a.save_json(OUT / "summary.json", {"at": a.now(), "status": "POINT_RESEARCH_PROGRESS_NOT_VALIDATED_EDGE",
                 "scope": "标的多空点位，频率软目标", "top_two_point_return_contribution": contribution,
                 "new_complete_up_episodes": 49, "new_complete_down_episodes": 49,
                 "comparison_rows": rows, "user_strict_product_gate_pass_count": int(compare.user_strict_product_gate_pass.sum()),
                 "goal_achieved": False, "next_question": "周锚点成功与失败在事前信息和价格路径上的差别；解释高胜率但p×B不足，保持旧规则结果，不作参数救援。"})
    state_path = b.CONTEXT / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(updated_at=a.now(), status="active", goal_achieved=False,
                 latest_completed_study="510300_DIRECTIONAL_RESEARCH_20261001", latest_result=str((OUT / "summary.json").relative_to(ROOT)),
                 latest_report=str((OUT / "研究结论.md").relative_to(ROOT)),
                 latest_progress="完成49上涨49下跌图谱、3类398条完整点位记录及周锚点双向37笔；六组p×实际盈亏比均<=1。双向年均3次、参考胜率64.86%、乘积0.550，目标未实现。",
                 minimum_cycles_each_full_year=None, frequency_objective="提高有效次数，非逐年五次硬门槛",
                 current_phase="POINTS_ONLY", latest_research_status="POINT_RESEARCH_PROGRESS_NOT_VALIDATED_EDGE",
                 new_accounts_in_current_phase=0, next_research_question="诊断周锚点成功与失败的事前差异，以及计划目标与实际退出的损耗；不选择期权，不改失败规则参数。")
    a.save_json(state_path, state)
    print("多空点位报告、逐笔表与三张可导出研究图已生成。", flush=True)


if __name__ == "__main__":
    main()
