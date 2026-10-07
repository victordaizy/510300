"""从已保存的固定研究结果生成解释、图表与案例，不重选参数或重跑策略。"""
from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fixed_smc", ROOT / "research/smc_sweep_fvg_historical_v1.py")
STUDY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STUDY)
OUT = STUDY.OUT
RESULTS = OUT / "results"
FIGURES = OUT / "figures"
LABELS = ["P0 破低收回", "P1 再突破试探区间", "P2 再要求FVG", "P3 再等回踩收回", "P4 FVG＋三项背景"]
HORIZON_NAMES = {"M5": "入场后5分钟", "M30": "入场后30分钟", "T0_CLOSE": "当日收盘", "T1_OPEN": "次日开盘", "T1_CLOSE": "次日收盘", "T2_CLOSE": "第2日收盘", "T5_CLOSE": "第5日收盘"}
FEATURE_NAMES = {"macd_rising": "MACD柱回升", "low_volatility": "相对低波动", "relative_volume_high": "相对放量"}


def read(name):
    return pd.read_parquet(RESULTS / (name + ".parquet"))


def check_saved_inputs_for_reporting():
    """报告只读取已存结果；共享进度配置后续变化单独留痕，不改冻结清单。"""
    frozen = json.loads((OUT / "freeze.json").read_text(encoding="utf-8"))
    assert STUDY.sha(Path(STUDY.__file__)) == frozen["code_sha256"]
    assert STUDY.sha(OUT / "protocol.json") == frozen["protocol_sha256"]
    changes = []
    for key, item in frozen["inputs"].items():
        source = Path(item["path"])
        current = STUDY.sha(source)
        if current != item["sha256"]:
            changes.append({"input": key, "path": str(source), "frozen_sha256": item["sha256"], "current_sha256": current, "mtime_epoch_seconds": source.stat().st_mtime})
    assert all(x["input"] == "mandate" for x in changes), "实际价格或分红输入改变，不能使用本报告流程。"
    if changes:
        verified = json.loads((OUT / "saved_verification.json").read_text(encoding="utf-8"))
        assert all(x["mtime_epoch_seconds"] > datetime.fromisoformat(verified["verified_at"]).timestamp() for x in changes), "配置变化早于保存结果复算，必须先确认计算来源。"
        STUDY.save_json(OUT / "project_mandate_observed_during_reporting.json", json.loads(STUDY.FILES["mandate"].read_text(encoding="utf-8")))
    STUDY.save_json(OUT / "report_only_input_receipt.json", {"checked_at": STUDY.now(), "frozen_code_protocol_and_price_inputs_unchanged": True, "changed_noncalculation_inputs": changes, "original_strict_check": "原研究脚本的完整冻结检查会因共享mandate字节变化而失败；不修改冻结清单或冻结实现。" if changes else "全部相同", "report_action": "只读先前保存且已复算的事件结果；额外损益分解不改变入场、退出、参数、费用或筛选。", "backtest_rerun": False})


def pct(value, signed=True):
    if pd.isna(value):
        return "未计算"
    return f"{value:+.3%}" if signed else f"{value:.1%}"


def md_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] + ["| " + " | ".join(map(str, row)) + " |" for row in rows])


def link(path, title):
    return f"[{title}](<{Path(path).resolve().as_posix()}>)"


def diagnostics(lab, daily):
    """事后解释固定输出中的损益来源，不将拆分结果形成新交易规则。"""
    z = lab[lab.horizon.eq("T1_OPEN") & lab.cost.eq("STRESS") & lab.label_status.eq("AVAILABLE")].copy()
    z["entry_day_close"] = z.date.map(daily.close)
    z["intraday_price_return"] = z.entry_day_close / z.entry_reference_price - 1
    z["overnight_and_dividend_return"] = (z.exit_reference_price + z.dividend_per_share - z.entry_day_close) / z.entry_reference_price
    z["allocated_intraday_return"] = z.quantity * (z.entry_day_close - z.entry_reference_price) / STUDY.ALLOCATION
    z["allocated_overnight_dividend_return"] = z.quantity * (z.exit_reference_price + z.dividend_per_share - z.entry_day_close) / STUDY.ALLOCATION
    z["allocated_slippage_rounding_cost"] = -z.slippage_rounding_cny / STUDY.ALLOCATION
    z["allocated_commission_cost"] = -z.commission_cny / STUDY.ALLOCATION
    columns = ["allocated_intraday_return", "allocated_overnight_dividend_return", "allocated_slippage_rounding_cost", "allocated_commission_cost"]
    error = float((z[columns].sum(axis=1) - z.net_return).abs().max())
    assert error < 1e-12
    summary = z.groupby("policy").agg(events=("event_id", "size"), intraday_price_return=("intraday_price_return", "mean"), overnight_and_dividend_return=("overnight_and_dividend_return", "mean"), gross_return=("gross_return", "mean"), net_return=("net_return", "mean"), allocated_intraday_return=("allocated_intraday_return", "mean"), allocated_overnight_dividend_return=("allocated_overnight_dividend_return", "mean"), allocated_slippage_rounding_cost=("allocated_slippage_rounding_cost", "mean"), allocated_commission_cost=("allocated_commission_cost", "mean")).reset_index()
    STUDY.csv(RESULTS / "固定结果损益分解.csv", summary)
    STUDY.csv(RESULTS / "逐事件损益分解.csv", z)
    joint = z[z.policy.eq("P4_CONTEXT")].sort_values("date")
    STUDY.csv(RESULTS / "主组合全部19次事件.csv", joint)
    return z, summary, joint, error


def style():
    font_manager.fontManager.addfont(r"C:\Windows\Fonts\msyh.ttc")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10, "axes.titlesize": 12, "axes.labelcolor": "#34485c", "text.color": "#1a2e43", "axes.edgecolor": "#bdc8d3", "xtick.color": "#516171", "ytick.color": "#516171", "figure.facecolor": "#f7f9fb", "axes.facecolor": "#ffffff", "savefig.facecolor": "#f7f9fb"})


def save_figure(fig, name):
    fig.savefig(FIGURES / (name + ".png"), dpi=170)
    fig.savefig(FIGURES / (name + ".svg"))
    plt.close(fig)


def overview(summary, cells, decomposition):
    fig, ax = plt.subplots(2, 2, figsize=(14.6, 10.4), gridspec_kw={"width_ratios": [1.12, 1.]})
    fig.subplots_adjust(left=.20, right=.97, top=.88, bottom=.145, wspace=.38, hspace=.47)
    fig.text(.055, .96, "510300｜形态、MACD、波动率与量的固定检验", fontsize=22, weight="bold")
    fig.text(.055, .917, "2021-08-12—2026-08-12 · 1,211个交易日 · 固定参数历史检验 · MACD 12/26/9", fontsize=11, color="#5b6d7e")
    main = summary[summary.horizon.eq("T1_OPEN")].set_index(["policy", "cost"])
    y = np.arange(5)
    for off, cost, col, title in [(-.24, None, "#7795ae", "毛收益"), (0., "BASE", "#3b697f", "基准成本后"), (.24, "STRESS", "#c86b50", "压力成本后")]:
        values = [float(main.loc[(p, cost or "STRESS"), "mean_gross" if cost is None else "mean_net"]) * 100 for p in STUDY.POLICIES]
        ax[0, 0].barh(y + off, values, .22, color=col, label=title)
        if cost == "STRESS":
            for k, v in enumerate(values):
                n = int(main.loc[(STUDY.POLICIES[k], cost), "events"])
                ax[0, 0].text(v - .018, k + off, f"{v:.3f}% · n={n}", ha="right", va="center", fontsize=9)
    ax[0, 0].set(yticks=y, yticklabels=LABELS, xlim=(-.83, .055), xlabel="平均单次收益（%）", title="A  持有至次日开盘：五层定义均为负")
    ax[0, 0].invert_yaxis()
    ax[0, 0].axvline(0, color="#a5b2bf", lw=.8)
    ax[0, 0].legend(loc="lower left", frameon=False, ncol=3, bbox_to_anchor=(-.42, -.35), fontsize=9)
    hs = ["T0_CLOSE", "T1_OPEN", "T1_CLOSE", "T2_CLOSE", "T5_CLOSE"]
    joint = summary[summary.policy.eq("P4_CONTEXT") & summary.cost.eq("STRESS")].set_index("horizon")
    ax[0, 1].plot(range(5), [joint.loc[h, "mean_gross"] * 100 for h in hs], "o-", color="#7795ae", label="毛收益")
    ax[0, 1].plot(range(5), [joint.loc[h, "mean_net"] * 100 for h in hs], "o-", color="#c86b50", label="压力成本后")
    ax[0, 1].axvspan(-.35, .35, color="#e7ecf1", alpha=.8)
    ax[0, 1].axhline(0, color="#a5b2bf", lw=.8)
    ax[0, 1].set(xticks=range(5), xticklabels=["当日收盘*", "次日开盘", "次日收盘", "第2日收盘", "第5日收盘"], ylabel="平均单次收益（%）", title="B  主组合19次：损失已在当日发生")
    ax[0, 1].tick_params(axis="x", labelsize=9)
    ax[0, 1].legend(frameon=False, loc="upper right", fontsize=9)
    ax[0, 1].text(.01, -.27, "* 当日退出仅解释路径，新买份额受T+1约束。", transform=ax[0, 1].transAxes, fontsize=9)
    c = cells[cells.policy.eq("P2_FVG") & cells.macd_rising].reset_index(drop=True)
    names = ["波动较高／量低于基准", "波动较高／相对放量", "相对低波动／量低于基准", "相对低波动／相对放量"]
    for i, r in c.iterrows():
        if pd.notna(r.lower95):
            ax[1, 0].plot([r.lower95 * 100, r.upper95 * 100], [i, i], color="#8ea4b6", linewidth=3)
        ax[1, 0].plot(r.mean_net * 100, i, "o", color="#c86b50", markersize=7)
        ax[1, 0].text(.035, i, f"n={int(r.events)}", va="center", fontsize=9)
    ax[1, 0].set(yticks=range(4), yticklabels=names, xlabel="次日开盘压力净收益（%）", xlim=(-.92, .19), title="C  49次FVG都已满足MACD回升")
    ax[1, 0].invert_yaxis()
    ax[1, 0].axvline(0, color="#a5b2bf", lw=.8)
    fig.text(.055, .073, "C图横线为描述性95%块抽样区间；n<10不计算区间。MACD未回升的四格均为0个事件，不能估计差异。", fontsize=9)
    d = decomposition.set_index("policy").loc["P4_CONTEXT"]
    values = [d.allocated_intraday_return, d.allocated_overnight_dividend_return, d.allocated_slippage_rounding_cost, d.allocated_commission_cost, d.net_return]
    baseline = 0.
    for i, v in enumerate(values):
        bottom = baseline if i < 4 else 0.
        col = "#53857d" if v >= 0 else ("#c86b50" if i == 4 else "#7795ae")
        ax[1, 1].bar(i, v * 100, bottom=bottom * 100, color=col, width=.64)
        endpoint = bottom + v
        ax[1, 1].text(i, endpoint * 100 - .026 if v < 0 else endpoint * 100 + .016, f"{v * 100:+.3f}", ha="center", va="top" if v < 0 else "bottom", fontsize=9)
        if i < 4:
            baseline += v
            ax[1, 1].plot([i + .32, i + .68], [baseline * 100] * 2, color="#b9c5cf", lw=.8)
    ax[1, 1].axhline(0, color="#a5b2bf", lw=.8)
    ax[1, 1].set(xticks=range(5), xticklabels=["日内价格", "隔夜及分红", "滑点／价位", "佣金", "净收益"], ylim=(-.65, .025), ylabel="每次10万元预算的平均损益（%）", title="D  主组合损益分解：费用前已亏损")
    ax[1, 1].tick_params(axis="x", labelsize=9)
    for a in ax.flat:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="x" if a in [ax[0, 0], ax[1, 0]] else "y", alpha=.12)
    fig.text(.055, .036, "口径：孤立事件，不是20万元连续账户；新买份额仅次日起可卖。置信区间未校正项目历次试验，不能视为独立盈利验证。", fontsize=10, color="#5b6d7e")
    save_figure(fig, "收益与条件")


def case_charts(joint, bars, events):
    """按日期选主组合最早一笔盈利及最早一笔亏损；选择规则明确为事后展示。"""
    selection = [joint[joint.net_return.lt(0)].iloc[0], joint[joint.net_return.gt(0)].iloc[0]]
    fig, axes = plt.subplots(3, 2, figsize=(14.8, 9.6), gridspec_kw={"height_ratios": [3.4, 1.3, 1.3]}, sharex="col")
    fig.subplots_adjust(left=.075, right=.97, top=.85, bottom=.13, hspace=.13, wspace=.19)
    fig.text(.055, .965, "同样的确认条件，仍会出现不同结果", fontsize=21, weight="bold")
    fig.text(.055, .92, "主组合中按时间最早的一笔亏损与一笔盈利；案例用于说明，统计结论来自全部19次。", fontsize=11, color="#5b6d7e")
    receipts = []
    for col, row in enumerate(selection):
        day = pd.Timestamp(row.date)
        b = bars[bars.date.eq(day)].sort_values("time").reset_index(drop=True)
        ev = events[events.event_id.eq(row.event_id)].iloc[0]
        idx = np.arange(len(b))
        top, macd, volume = axes[:, col]
        for i, v in b.iterrows():
            color = "#bc5f53" if v.close >= v.open else "#4f7f7a"
            top.vlines(i, v.low, v.high, color=color, lw=.9)
            top.add_patch(Rectangle((i - .32, min(v.open, v.close)), .64, max(abs(v.close - v.open), .00015), facecolor=color, edgecolor=color, linewidth=.8))
        top.axhline(row.reference, color="#49677e", ls="--", lw=1., label="前日低点")
        top.axhline(row.structure_low, color="#9b779c", ls=":", lw=1., label="固定结构低点")
        ci = int(b.index[b.time.eq(row.signal_time)][0])
        si = int(b.index[b.time.eq(ev.sweep_time)][0])
        ei = ci + .6
        top.add_patch(Rectangle((ci, row.gap_lower), 48 - ci, row.gap_upper - row.gap_lower, facecolor="#e2ba69", edgecolor="none", alpha=.22))
        top.scatter(si, b.low.iloc[si], marker="v", s=48, color="#93789d", zorder=5)
        top.scatter(ci, b.close.iloc[ci], marker="o", s=38, color="#344f72", zorder=5)
        top.scatter(ei, row.entry_reference_price, marker="*", s=125, color="#cd823c", edgecolors="white", linewidths=.5, zorder=6)
        top.scatter(50, row.exit_reference_price, marker="D", s=46, color="#344f72", zorder=6)
        top.plot([47, 50], [float(b.close.iloc[-1]), row.exit_reference_price], ls=":", color="#9aa9b8")
        top.set_title(f"{day:%Y-%m-%d}  {'亏损' if row.net_return < 0 else '盈利'}例\n次日开盘压力净收益 {pct(row.net_return)}", loc="left", pad=12)
        top.set_ylabel("价格（元）")
        top.text(.025, .96, f"确认 {row.signal_time:%H:%M} · 入场代理 {row.entry_time:%H:%M}\n波动比 {row.rv20_relative_prior:.2f} · 相对量 {row.relative_volume:.2f}", transform=top.transAxes, va="top", fontsize=9, bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88})
        macd.bar(idx, b.macd_hist_bps, color=np.where(b.macd_hist_bps.ge(0), "#c18477", "#7ca39e"), width=.72)
        macd.axhline(0, color="#9aa9b8", lw=.7)
        macd.set_ylabel("MACD柱\n（基点）")
        volume.bar(idx, b.relative_volume, color="#8aa2b5", width=.72)
        volume.axhline(1, color="#bf8056", ls="--", lw=.9)
        volume.set_ylabel("同槽位\n相对量")
        volume.set(xticks=[0, 12, 23, 35, 47, 50], xticklabels=["09:35", "10:35", "11:30", "14:00", "15:00", "\n次日开盘"])
        volume.tick_params(axis="x", labelsize=9)
        for a in axes[:, col]:
            a.axvline(ci, color="#354f73", ls="--", alpha=.4, lw=.8)
            a.axvline(23.5, color="#b6c1cb", alpha=.45, lw=.8)
            a.set_xlim(-1, 52)
            a.spines[["top", "right"]].set_visible(False)
            a.grid(axis="y", alpha=.12)
        receipts.append({"selection_rule": "主组合压力净收益正负两组分别取时间最早事件；事后展示，不参与策略筛选。", "event_id": row.event_id, "date": day, "signal_time": row.signal_time, "entry_time": row.entry_time, "entry_reference_price": row.entry_reference_price, "exit_time": row.exit_time, "exit_reference_price": row.exit_reference_price, "gross_return": row.gross_return, "net_return": row.net_return, "macd_hist_change_bps": row.macd_hist_change_bps, "rv20_relative_prior": row.rv20_relative_prior, "relative_volume": row.relative_volume})
    fig.text(.055, .062, "▼ 首次跌破前日低点   ● 形态与背景确认   ★ 延迟入场代理   ◆ 次日开盘   淡黄色：确认时已形成的FVG区间", fontsize=10)
    fig.text(.055, .029, "显示5分钟OHLC及背景，不含逐笔主动买卖数据。图中的入场价格是原始分钟open；收益另计不利滑点、价位取整与佣金。", fontsize=9, color="#5b6d7e")
    STUDY.save_json(OUT / "case_selection.json", receipts)
    save_figure(fig, "成功与失败案例")
    return receipts


def report(result, summary, cells, marginal, decomposition, joint, cases, verification):
    main = summary[summary.horizon.eq("T1_OPEN")].set_index(["policy", "cost"])
    primary = main.loc[("P4_CONTEXT", "STRESS")]
    p2m = marginal[marginal.policy.eq("P2_FVG")].set_index("feature")
    selected = decomposition.set_index("policy").loc["P4_CONTEXT"]
    rows = []
    for p, label in zip(STUDY.POLICIES, LABELS):
        base, stress = main.loc[(p, "BASE")], main.loc[(p, "STRESS")]
        rows.append([label, int(stress.events), pct(stress.mean_gross), pct(base.mean_net), pct(stress.mean_net), pct(stress.positive_fraction, False), f"[{pct(stress.lower95)}, {pct(stress.upper95)}]"])
    feature_rows = [[FEATURE_NAMES[f], f"{int(r.kept_events)}/{int(r.parent_events)}", pct(r.kept_mean_net), pct(r.excluded_mean_net), "未增加筛选" if r.excluded_events == 0 else "未保留正的单次期望"] for f, r in p2m.iterrows()]
    cell_rows = []
    for _, r in cells[cells.policy.eq("P2_FVG")].iterrows():
        cell_rows.append(["回升" if r.macd_rising else "未回升", "相对低" if r.low_volatility else "高于历史中位数", "相对放量" if r.relative_volume_high else "低于历史基准", int(r.events), pct(r.mean_net)])
    horizon_rows = []
    for h in STUDY.HORIZONS:
        r = summary[summary.policy.eq("P4_CONTEXT") & summary.cost.eq("STRESS") & summary.horizon.eq(h)].iloc[0]
        horizon_rows.append([HORIZON_NAMES[h], int(r.events), pct(r.mean_gross), pct(r.mean_net), "可卖时点的价格模型" if r.legal_for_new_shares else "仅路径诊断，不能日内卖新买份额"])
    yearly = read("逐年统计")
    year_rows = []
    for year in range(2021, 2027):
        r = yearly[yearly.policy.eq("P4_CONTEXT") & yearly.horizon.eq("T1_OPEN") & yearly.cost.eq("STRESS") & yearly.year.eq(year)]
        year_rows.append([str(year) + ("（不完整年）" if year in [2021, 2026] else ""), int(r.events.iloc[0]) if len(r) else 0, pct(r.mean_net.iloc[0]) if len(r) else "未计算，无事件"])
    case_rows = [[pd.Timestamp(r["date"]).strftime("%Y-%m-%d"), pd.Timestamp(r["signal_time"]).strftime("%H:%M"), f"{r['entry_reference_price']:.3f}", f"{r['exit_reference_price']:.3f}", pct(r["gross_return"]), pct(r["net_return"])] for r in cases]
    all_joint_rows = [[f"{r.date:%Y-%m-%d}", f"{r.signal_time:%H:%M}", f"{r.macd_hist_change_bps:+.2f}", f"{r.rv20_relative_prior:.2f}", f"{r.relative_volume:.2f}", pct(r.gross_return), pct(r.net_return)] for r in joint.itertuples()]
    positive_increment = next(r for r in result["filter_comparisons"] if r["comparison"] == "P4_CONTEXT_MINUS_P2_FVG")
    nav = "\n".join([f"- {link(OUT / 'protocol.json', '计算收益前固定的定义')}、{link(OUT / 'freeze.json', '冻结时间与输入指纹')}、{link(OUT / 'input_quality.json', '输入质量结果')}。", f"- {link(RESULTS / '全部交易日与事件.csv', '全部1211个交易日与失败状态')}、{link(RESULTS / '全部信号.csv', '全部622条分层信号')}、{link(RESULTS / '入场及未入场.csv', '全部入场与未入场原因')}。", f"- {link(RESULTS / '全部事件收益.csv', '全部事件、期限及成本收益')}、{link(RESULTS / 'MACD波动率量八格.csv', '三个母样本的24个背景单元格')}、{link(RESULTS / '单项背景增量.csv', '9个单项背景比较')}。", f"- {link(RESULTS / '主组合全部19次事件.csv', '主组合19次完整事件')}、{link(RESULTS / '固定结果损益分解.csv', '固定输出的损益分解')}、{link(OUT / 'saved_verification.json', '保存结果复算')}。", f"- {link(ROOT / 'research/smc_sweep_fvg_historical_v1.py', '完整研究实现')}、{link(ROOT / 'tests/test_smc_sweep_fvg_historical_v1.py', '有针对性的时序与收益测试')}、{link(Path(__file__), '报告与图表生成实现')}。"])
    text = f"""# 510300：前日低点试探、FVG 与 MACD／波动率／成交量的历史研究

本轮已经完成历史实证。**固定组合没有发现可用的正收益优势**：19次“破低收回→区间再突破→已形成FVG→MACD回升、相对低波动、相对放量”事件，持有至次日开盘，毛收益平均{pct(primary.mean_gross)}、基准成本后{pct(main.loc[('P4_CONTEXT', 'BASE'), 'mean_net'])}、压力成本后{pct(primary.mean_net)}；19次中仅2次压力净收益为正。主方案的描述性95%区间为[{pct(primary.lower95)}, {pct(primary.upper95)}]。

研究成果是一个可复核的否定结论，以及三个具体发现：本定义下MACD回升与FVG确认高度重叠；低波动、放量没有筛出更好的延续；减少亏损交易可以改善总贡献，但不能据此说被保留交易有优势。本轮状态为`REJECTED_FROZEN_NO_PARAMETER_RESCUE`。目标净夏普1.2、年化10%、最大回撤10%的完整账户目标**尚未达成**。

研究日期：2026-10-01。使用历史区间2021-08-12—2026-08-12，1,211个交易日、291,851条原始分钟记录。所有规则在本轮首次收益计算前固定；这些价格早已用于项目历史研究，不能称独立留出样本。

## 油管方法与本次研究的关系

| 类别 | 实际观察什么 | 本轮能验证到哪一层 |
| --- | --- | --- |
| DOM、Footprint、CVD等盘口订单流 | 逐笔主动买卖、价格响应和订单簿变化 | 当前OHLCV输入没有这些字段，真实订单流`NOT_COMPUTED` |
| ICT/SMC价格结构 | 关键位置、跌破收回、结构变化、三柱FVG | 将其中一组价格过程转成事前可判断的明确规则 |
| 我们原有研究 | 统计关系、日线条件、成本和账户限制 | 增加日内发生顺序，保留成本、T+1和全部失败样本 |

Axia配套原文展示了大周期背景、低点测试以及在价格梯中观察卖方强度、买方响应和吸收的思路。本研究借用“位置→测试→响应”的提问方式，**分钟成交量无法验证主动卖单是否被吸收**。[Axia作者配套原文](https://axiafutures.com/blog/elite-trader-trades-false-break-setup-with-200-lots/)

FVG按三柱价格范围定义：看涨FVG要求第三根最低价高于第一根最高价。这不等于已识别机构订单或真实公允价值。本次加上前日低点事件与入场时钟，是我们为510300编写的受限实验，不能称完整复现ICT体系。[FVG工具原始说明](https://docs.luxalgo.com/platform/algos/price-action-concepts/imbalances)

视频参考为[ICT 2022 Mentorship Episode 6](https://www.youtube.com/watch?v=Bkt8B3kLATQ)及Axia页面关联视频。读取范围为可访问的原始简介、元数据与作者配套文章；没有取得完整字幕，不声称完整观看并验证了整套课程。项目既有15分钟无条件FVG、日线RECLAIM与失败锚定网格研究均保留，本轮属于相关价格形态家族的新条件检验。

## 计算前固定的定义

五分钟柱分别在上午和下午由连续五条分钟记录聚合，不跨午休；09:30集合竞价独立保留。参考价是前一个完整交易日的最低价，在除息日作现金分红平移。开盘已经低于参考价的152个交易日单独保存，不重命名为日内从上方跌破。

1. P0：每日只取首次跌破前日低点至少0.001元的事件，在同一半日内、其后最多6根五分钟柱中，首次收盘收回参考价。
2. P1：P0之后最多6根柱内，收盘超过首次试探柱高点至少0.001元；若先收盘跌破收复时已知的结构低点则失败。
3. P2：P1确认当时已经形成看涨三柱FVG，中间柱上涨、第一柱不早于试探；缺口不能事后回填到先前的确认时间。
4. P3：在P2确认之后最多6根柱内，等待FVG上沿回踩收回；先收盘跌破缺口下沿则作废。
5. P4（唯一主方案）：在P2的同一确认时刻，叠加MACD回升、相对低波动、相对放量。P4不要求P3回踩，是P2的另一个子集。

| 背景变量 | 固定口径 | 时点约束 |
| --- | --- | --- |
| MACD | 五分钟12/26/9；柱值2×(DIF−DEA)，较上一根增加 | 用已完成柱；260根预热；回升可以发生在零轴下方 |
| 波动率 | 前20个完整交易日含息对数收益标准差×√242，与此前252个可得RV20中位数比较；基准至少126日 | 不读取当日最终日线收益；不高于中位数定义相对低波动 |
| 成交量 | 当前完成五分钟量／此前20个交易日同半日、同槽位成交量中位数 | 分母不含当日；比值≥1定义相对放量，不是任意“大单”阈值 |

另完整列出P0/P1/P2的24个八格单元和9个单独背景比较。这些是描述性分解，不能挑出最好单元替换主方案。MACD按已发生分红向前累计平移，避免未来分红反向修改过去指标；背景未知保留为不可评价，没有当作“不满足条件”来填零。

## 交易时点、费用与研究范围

供应商没有承诺分钟标签是起点还是终点。信号标签t保守记作t+1分钟可得，以标签t+3分钟的open作为延迟入场代理。拒绝跨午休、14:57及以后、该分钟无成交、入场价已破结构低点及涨停买入。这只是成交模型，没有逐笔成交回报或排队证明。

主退出为下一交易日开盘；次日收盘、第2／5日收盘为固定次要期限。股票ETF实施T+1，因此5／30分钟和当日收盘只用于解释价格路径，不能算新买份额的可执行日内收益。[上海证券交易所说明](https://www.sse.com.cn/assortment/fund/etf/question/c/c_20240118_5734755.shtml)

每个孤立事件分配10万元预算，100份整数手、最小价位0.001元，保留不足一手及费用后的现金。基准成本每边佣金2bp、滑点5bp；压力每边佣金4bp、滑点10bp，最低佣金5元，并作不利价位取整。费用是固定情景假设，未宣称真实券商报价。现金分红按登记日权益计入应收，没有在到账前充当可投资现金。

“单次净收益”的分母是该事件的10万元预算，**不是20万元完整账户日收益**。没有把事件收益年化成账户夏普；多个期限的持仓也没有拼接为可用资金重复投资。

## 主结果：更多确认没有变成更高净期望

{md_table(['条件', '可评价事件', '平均毛收益', '基准净收益', '压力净收益', '压力胜率', '压力均值95%区间'], rows)}

以上统一为次日开盘退出。毛收益不扣费用，净收益还受整数手和现金余量影响，因此毛净差不完全等于抽象费率。每个策略最多每日一个母事件，各策略之间重叠，不能将样本数相加作为独立证据。

从全体交易日看，625日没有日内首次跌破，152日开盘已低于参考价，余下434日构成首破母事件；349次收回、178次区间再突破、52次确认时已有FVG、24次完成后续回踩，主背景组合19次。模型入场后主期限样本数分别为310／173／50／24／19；另有34条分层信号入场前结构已坏、12条没有连续入场窗口，均保存原因。

![收益与条件](<{(FIGURES / '收益与条件.png').as_posix()}>)

## MACD、波动和量究竟增加了什么

背景全可用的FVG父样本为49次；其余1次背景不足，未放入背景比较。

{md_table(['单项条件', '保留/共同父样本', '保留组压力净收益', '排除组压力净收益', '观察'], feature_rows)}

**MACD在这个定义和样本里是重复确认。** 49／49次FVG都已出现MACD柱回升；在更宽的P1父样本里也是167／172次。不能据此证明MACD普遍无效，但这里的“形态+MACD”没有提供第二份独立支持。低波动、相对放量各自保留组的平均亏损更大；组间差异尚不能解释为因果作用，也没有用这些差异倒过来调方向。

FVG的完整八格如下；四个MACD未回升格没有样本，不能记作零收益。

{md_table(['MACD', '波动状态', '量状态', '事件数', '次日开盘压力净收益'], cell_rows)}

**少交易为什么会有正“增量”？** 在共同49个父机会中，P2全部交易的平均贡献为{pct(positive_increment['parent_mean'])}；P4只交易19次，把其余已知不满足条件的机会记作现金零贡献后，平均贡献为{pct(positive_increment['child_mean_contribution'])}，差值{pct(positive_increment['mean_difference'])}。这是削减负期望交易带来的少亏。P4保留交易的单次均值仍为{pct(primary.mean_net)}，在该机会口径中全部不交易的零贡献甚至更高。因此“筛选增量为正”不能单独证明买入优势。

## 亏损发生在什么时候

主组合19次，原始入场价到当日收盘平均{pct(selected.intraday_price_return)}；当日收盘到次日开盘加上分红权益，以原入场价为共同分母，平均{pct(selected.overnight_and_dividend_return)}。二者之和为毛收益{pct(selected.gross_return)}。**这组样本的主要价格亏损已经发生在入场当日，隔夜略有补回，不能把失败全部归因于T+1。**

按实际整数手与10万元预算作精确损益分解：日内价格{pct(selected.allocated_intraday_return)}，隔夜及分红{pct(selected.allocated_overnight_dividend_return)}，滑点及价位取整{pct(selected.allocated_slippage_rounding_cost)}，佣金{pct(selected.allocated_commission_cost)}，合计{pct(selected.net_return)}。零费用也无法把这组已实现的平均毛收益改成正数。

{md_table(['固定退出期限', '可评价数', '毛收益', '压力净收益', '解释'], horizon_rows)}

日内5／30分钟因午休或收盘窗口会少于19条，没有补成零。上述固定期限中，主组合的压力净均值全部为负；没有事后挑一个最优退出替代主期限。原始价格MAE/MFE单独标记除息与右端路径不完整，不能把除息跌幅当真实经济亏损，亦不能把日内跌破结构线当作当日可成交止损。

## 时间分段与两类案例

固定早段（2021-08—2023-12）13次，主组合压力净均值{pct(primary.early_mean_net)}；后段（2024-01—2026-08）6次，{pct(primary.late_mean_net)}。这是同一已使用历史的分段描述，不是未见样本验证。去掉最好的一笔，主组合净均值为{pct(primary.mean_without_best)}。

{md_table(['年度', '主组合事件数', '次日开盘压力净均值'], year_rows)}

选择规则为主组合中按日期最早的一笔净亏损与最早的一笔净盈利。此规则在看完固定结果后用于展示，不参与策略筛选，也不以两个案例替代全体统计。

{md_table(['事件日', '确认时刻', '入场原价', '次日开盘原价', '毛收益', '压力净收益'], case_rows)}

2021-09-16已有形态、MACD回升、低波动和相对放量，确认后价格继续下降；2021-10-21同样条件随后上涨。共同点是入场时都只能看到条件满足，不能在当时预知哪一类结局。

![成功与失败案例](<{(FIGURES / '成功与失败案例.png').as_posix()}>)

## 主组合全部事件

MACD变化单位为基点；波动比为先前RV20／历史中位数，相对量为当前完成柱／先前20日同槽位量中位数。2026-01-16持有跨现金除息，毛净收益已计登记权益；图表的原始价格路径不应直接当作经济损失。

{md_table(['事件日', '确认', 'MACD柱变化', '波动比', '相对量', '毛收益', '压力净收益'], all_joint_rows)}

## 结论、边界与停止条件

本轮可保留的方法成果是把“位置→试探→响应→确认”转成只依赖当时信息的事件账本，并检验背景变量是否真的增加区分能力。当前证据不支持把该固定多头组合纳入510300买入依据，也不支持反向做空或反过来挑波动、量的方向。

唯一主门要求压力次日开盘净均值与95%下界为正、对P2的贡献增量下界为正、两个固定半段都为正。实际仅“减少交易后的贡献增量”一项通过，其余失败。按计算收益前的停止条件，本轮完整账户`NOT_RUN_PRIMARY_EVENT_GATE_FAILED`，账户夏普／年化／最大回撤`NOT_COMPUTED`；没有触发新的采集、前瞻任务或交易指令。

局限集中在三点：分钟OHLCV不包含盘口机制，无法判断卖单是否被真实吸收；历史数据在2026年取得，不构成逐时实时接收认证，分钟时间标签也存在起止语义不确定；19个主事件和已反复使用的全项目历史不足以作独立效果验证，所有块抽样区间均未调整全项目试验次数。以上限定本研究能回答的问题，不推导“所有订单流方法无效”或“510300永远没有机会”。

后续如果研究真盘口方法，需要能够区分成交方向、订单与撤单的历史证据，回答“在关键位置出现大量卖出后，价格为什么没有继续下降”。继续叠加同一价格序列衍生指标，目前没有新增证据支持。本轮不自动重开失败参数，也不把结果后构想当成已经完成的下一项策略。

## 复核与文件入口

冻结时间见记录；分钟聚合OHLC与独立日线完全一致，量额误差低于1e-8。8项有针对性的测试通过，包括午休边界、确认后才形成的FVG不能回填、当前日线不能影响先前波动率、退出开盘不能偷看之后分钟极值。对保存的{verification['saved_return_rows_recomputed']:,}条完整收益逐条复算，净收益误差为0；36个真实事件前缀和8个指标前缀截断检查通过，70行汇总与账本一致。这些检查确认计算口径，没有将其称为独立外部评审或真实成交。

生成报告时，共享研究进度配置`mandate`在原计算与复算完成后发生更新，导致原脚本严格检查整份配置字节时失败。行情、分红、分钟元数据、冻结协议和研究代码均保持相同。报告只读取已存且已复算的事件结果，保留{link(OUT / 'report_only_input_receipt.json', '配置后续变化记录')}，没有更新旧指纹或重跑收益。

{nav}
"""
    path = OUT / "研究结论.md"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    check_saved_inputs_for_reporting()
    FIGURES.mkdir(exist_ok=True)
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    verification = json.loads((OUT / "saved_verification.json").read_text(encoding="utf-8"))
    lab, summary, cells, marginal = [read(name) for name in ["全部事件收益", "分组统计", "MACD波动率量八格", "单项背景增量"]]
    daily = pd.read_parquet(STUDY.FILES["daily"])
    daily["date"] = pd.to_datetime(daily.date)
    daily = daily.set_index("date").sort_index()
    _, decomposition, joint, error = diagnostics(lab, daily)
    style()
    overview(summary, cells, decomposition)
    cases = case_charts(joint, read("五分钟价格柱"), read("全部交易日与事件"))
    path = report(result, summary, cells, marginal, decomposition, joint, cases, verification)
    STUDY.save_json(OUT / "report_receipt.json", {"created_at": STUDY.now(), "research_status": result["status"], "report": str(path), "source_result_sha256": STUDY.sha(OUT / "result.json"), "report_sha256": STUDY.sha(path), "report_code_sha256": STUDY.sha(Path(__file__)), "decomposition_max_identity_error": error, "case_selection": "事后展示：主组合最早一笔盈利与最早一笔亏损", "new_strategy_rules": 0, "new_parameter_searches": 0, "new_backtests": 0, "full_account_metrics": "NOT_COMPUTED", "real_order_flow": "NOT_COMPUTED", "goal_strategy_achieved": False})
    print(json.dumps({"研究报告": str(path), "主组合事件": len(joint), "损益分解最大误差": error, "图表": [str(FIGURES / "收益与条件.png"), str(FIGURES / "成功与失败案例.png")]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
