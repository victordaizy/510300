"""整理MSCI纳入比例提升的五事件证据，不继续拟合或改选窗口。"""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import historical_msci_inclusion_demand_v1 as study


ROOT, OUT = study.ROOT, study.OUT
REPORT = "MSCI纳入提升_资金需求与指数历史结果.md"
FIGURE = "五次MSCI实施_北向净买入与指数收益.png"


def check_saved(d):
    prices = pd.read_parquet(ROOT / study.SOURCES["price_state"], columns=["date", "open", "close"])
    prices["date"] = pd.to_datetime(prices.date).dt.tz_localize(None)
    prices = prices.set_index("date")
    div = pd.read_csv(ROOT / study.SOURCES["dividends"])
    div["record_date"] = pd.to_datetime(div.record_date)
    rate, slip, tick, shares = Decimal("0.0004"), Decimal("0.001"), Decimal("0.001"), Decimal("10000")
    checks = []
    for row in d.to_dict("records"):
        for prefix in ["primary", "post"]:
            first, last = pd.Timestamp(row[prefix + "_entry_date"]), pd.Timestamp(row[prefix + "_exit_date"])
            begin, end = Decimal(str(prices.at[first, "open"])), Decimal(str(prices.at[last, "open"]))
            buy = ((begin * (1 + slip) / tick).to_integral_value(rounding=ROUND_CEILING)) * tick
            sell = ((end * (1 - slip) / tick).to_integral_value(rounding=ROUND_FLOOR)) * tick
            entitlement = sum((Decimal(str(x)) for x in div.loc[div.record_date.ge(first) & div.record_date.lt(last), "cash_dividend_per_share"]), Decimal(0))
            paid = shares * buy + max(Decimal(5), shares * buy * rate)
            received = shares * (sell + entitlement) - max(Decimal(5), shares * sell * rate)
            net = float(received / paid - 1)
            error = abs(net - row[prefix + "_net_return"])
            intervals = len(prices.loc[first:last]) - 1
            if error > 1e-12 or intervals != 5:
                raise ValueError("固定窗口或净收益重算不一致。")
            checks.append({"id": row["id"], "window": prefix, "decimal_net_return": net, "saved_error": error, "open_intervals": intervals})
    result = {"checked_at": study.now(), "windows": 10, "max_net_return_error": max(x["saved_error"] for x in checks),
              "fees_slippage_dividend_recomputed": True, "new_candidate_runs": 0, "new_fits": 0, "new_accounts": 0,
              "meaning": "仅核对十个标签的取价、股息、费用和窗口长度，不代表独立策略验证。"}
    pd.DataFrame(checks).to_csv(OUT / "保存标签取价及费用复核.csv", index=False, encoding="utf-8-sig")
    study.save("verification.json", result)
    return result


def figure(d):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [1.12, 1]})
    fig.patch.set_facecolor("#f7f8fa")
    x = np.arange(len(d))
    colors = ["#197184" if r > 0 else "#bc4939" for r in d.primary_net_return]
    bars = axes[0].bar(x, d.primary_net_return * 100, width=.56, color=colors)
    axes[0].axhline(0, color="#66717f", linewidth=.8)
    axes[0].bar_label(bars, labels=[f"{v:+.2f}%" for v in d.primary_net_return * 100], padding=5, fontsize=12)
    axes[0].set_ylim(-2.6, 1.1)
    axes[0].set_ylabel("510300固定五日净收益")
    axes[0].set_title("5次实施日均有北向净买入，固定五日窗口只有1次净收益为正", loc="left", fontsize=17, pad=20, weight="bold")
    axes[0].text(.99, .97, "平均净收益 −0.78%\n未扣费平均 −0.48%", ha="right", va="top", transform=axes[0].transAxes,
                 fontsize=12, bbox={"boxstyle": "round,pad=.5", "facecolor": "#eef1f5", "edgecolor": "none"})
    first = axes[1].bar(x - .17, d.realized_northbound_5d_yi, width=.32, label="五个交易日合计", color="#5e7898")
    second = axes[1].bar(x + .17, d.realized_northbound_implementation_yi, width=.32, label="其中：实施日", color="#c0ccd8")
    axes[1].bar_label(first, labels=[f"{v:.1f}" for v in d.realized_northbound_5d_yi], padding=3, fontsize=10)
    axes[1].bar_label(second, labels=[f"{v:.1f}" for v in d.realized_northbound_implementation_yi], padding=3, fontsize=10)
    axes[1].axhline(0, color="#66717f", linewidth=.8)
    axes[1].set_ylim(-95, 345)
    axes[1].set_ylabel("全部北向净买入 / 亿元")
    axes[1].legend(frameon=False, loc="upper left", ncol=2)
    axes[1].set_xticks(x, [s + "\n" + f"{a:.1%} → {b:.1%}" for s, a, b in zip(d.implementation, d.iif_before, d.iif_after)])
    for ax in axes:
        ax.set_facecolor("#f7f8fa")
        ax.grid(axis="y", alpha=.13)
        ax.set_axisbelow(True)
    fig.text(.085, .018, "窗口：实施日前第4个交易日开盘至实施后次开盘；固定10000份，佣金万四/边、最低5元，滑点千一/边。\n北向流量为事后全部净买入，不能分离MSCI被动资金；图中收益为单次投入收益，非20万元账户收益。", fontsize=9, color="#536170")
    fig.subplots_adjust(left=.085, right=.965, top=.89, bottom=.13, hspace=.18)
    fig.savefig(OUT / FIGURE, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def report(d, result):
    rows, states = [], []
    for r in d.to_dict("records"):
        rows.append(f"| {r['implementation']} | {r['iif_before']:.1%}→{r['iif_after']:.1%} | {r['primary_entry_date']}—{r['primary_exit_date']} | {r['realized_northbound_implementation_yi']:.2f} | {r['realized_northbound_5d_yi']:.2f} | {r['primary_gross_return']:+.2%} | **{r['primary_net_return']:+.2%}** | {r['post_net_return']:+.2%} |")
        states.append(f"| {r['implementation']} | {r['decision_at'][:10]} | {r['pre20_total_return']:+.2%} | {r['funding_gap_pp'] * 100:+.2f} | {r['funding_gap_change5_pp'] * 100:+.2f} | {r['orders_index']:.1f} | {r['primary_net_return']:+.2%} |")
    cells = []
    for c in result["joint_cells"]:
        mean = "无观察，不填0" if c["mean"] is None else f"{c['mean']:+.2%}"
        cells.append(f"| {c['trend']} | {c['funding']} | {c['n']} | {mean} | {c['positive']} |")
    source_list = []
    for e in study.EVENTS:
        url = next(x["url"] for x in study.DOCUMENTS if x["id"] == e["id"])
        source_list.append(f"- [{e['review_date']}审议公告]({url})：实施收盘日{e['implementation']}，纳入因子{e['iif_before']:.1%}→{e['iif_after']:.1%}；公告预估A股在新兴市场指数的合计权重为{e['weight_em_planned']:.2%}。两个百分比口径不同。")
    text = f"""# MSCI纳入提升：资金需求与510300历史结果

本轮找到一个有明确上游原因的需求事件：海外基准提高A股配置比例。但五次实施的固定五日窗口，仅1次510300成本后收益为正，平均 **{result['primary_net']['mean']:+.2%}**；不计成本平均 **{result['primary_gross']['mean']:+.2%}**。不能据此给指数方向加高分，该固定五日表达拒绝封存。

本轮是2018—2019年五个案例的历史发现。没有拟合新树、搜索阈值或建立完整账户；夏普为`NOT_COMPUTED`，总体夏普1.2及收益、频次目标仍未达到。此前贷款消息的失败表达和主要期账户结果保持原判，收益不拼接。

## 全部五次事件

固定实施日为E，在E-4交易日开盘进入，E+1开盘退出；例如首次事件为2018-05-25开盘至2018-06-01开盘。入场前一交易日16:00决定，当期公告均已可见。

使用固定10000份510300、双边万四佣金（每边最低5元）、双边千一滑点、0.001元价格步长及登记日股息权益。下表是每笔投入的收益率，不是20万元完整账户收益。实施后五日只说明走势是否延续，不作为另一个择时方案。

| 实施收盘日 | 大盘A股纳入因子 | 主窗口开盘至开盘 | 实施日北向净买入/亿元 | 五日北向净买入/亿元 | 五日未扣费 | 五日扣费后 | 实施后五日扣费后 |
|---|---:|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

![北向净买入与510300收益](<{str(OUT / FIGURE)}>)

五次实施日的全部北向净买入均为正，五个主窗口有四个净买入为正；其中四个主窗口净收益为负。2019年11月实施日净买入214.30亿元，窗口合计净买入288.30亿元，510300仍亏损2.05%。2019年5月则是实施日净买入55.67亿元、五日合计净流出65.64亿元，说明只看实施日会遗漏此前的反向流动。

同一天看，五次中四次510300当日总回报为正，2018年8月31日为负。这个当日现象不等于提前五日买入可获利，也不能用日线数据声称已证明收盘竞价价格冲击。没有据此新增当日或分钟策略。

## 原因、需求、预期与成交是四个环节

1. **上游原因是可投资性改善和指数规则变化。** [2017年官方决定]({study.DOCUMENTS[-2]['url']})把互联互通接入、停牌及产品审批等约束放松列为纳入背景，并提前列出2018年两步安排。不是因为临近实施的几天宏观数据突然改善。
2. **提高纳入比例改变跟踪基金的目标配置。** 在其余市值和规模不变的示意条件下，若A为合格A股的自由流通调整市值、B为其余成分市值、q为纳入因子，A股权重是 `qA/(B+qA)`。这本身就不等于对q简单线性加分。真实审议还会改变成分股及其他国家权重，不能把这个示意式当作实际流入额。
3. **目标增加不等于全部尚未执行。** [2019年2月官方决定]({study.DOCUMENTS[-1]['url']})已公布5%→10%→15%→20%的安排，也说明分三步实施是为了分散执行压力。临近实施的剩余买入取决于目标持仓减去已有持仓，还受跟踪资产规模、现金流、提前交易和替代工具影响。本轮没有这些逐基金数据，因此剩余被动需求保留`NOT_COMPUTED`。
4. **实现的资金与价格不是同一变量。** 北向净买入同时包含主动与被动资金，沪深300与MSCI A股部分也不是相同篮子。只有结合交集权重、其他交易者供给、已经反映的预期和同期冲击，才可能判断价格结果；不能把观察到净买入解释为没有其他卖压。

以上机制解释为何这五个事件适合检验“已知需求是否足以支持指数方向”。它们不能单独识别“若没有MSCI纳入，指数会怎样”，也没有证据证明亏损必然由提前计价或对手盘抢跑造成。

## 事前多维联合状态

所有值取自E-5日决策时刻。资金差为已知DR007减当时已生效七天政策利率，保留原滞后时钟；一基点=0.01个百分点。此前20日涨跌是状态代理，不把此前上涨直接称为已经充分计价。

| 实施日 | 状态截至日 | 此前20日总回报 | 已知资金差/基点 | 资金差五日变化/基点 | 已公布订单指数 | 五日净收益 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(states)}

主窗口收益计算前已固定两条条件的四格交互：此前20日正/非正，资金差收窄/不收窄。纳入提升为共同背景，其他融资、波动和订单数值保存在事件明细中，没有根据结果再筛选。

| 价格状态 | 资金状态 | 事件数 | 平均五日净收益 | 正收益次数 |
|---|---|---:|---:|---:|
{chr(10).join(cells)}

有观察的三格均值都为负，第四格没有观察。2019年8月和11月同处“此前上涨、资金差未收窄”，仍然一正一负。五次实施还只来自两项路线决定，不能在这里学出可靠的分段阈值、权重或上涨概率。联合评分应保留这项证据的弱度，本轮没有把空格补零或给已知需求人为加分。

## 本轮结论及边界

本轮新增七份官方归档的事实核对、五次联合状态、十个固定窗口标签和25个事后北向日值。标签的原始开盘、费用、股息与五日长度已用独立十进制定价重算。

固定五日表达为`{result['candidate_status']}`。不改变入场提前天数、不反向、不增加事后阈值过滤、不用实施后区间营救。本次无新增合格策略，全年频次与完整账户夏普未计算，也不据五个案例宣称所有指数纳入机制永远无效。

官网原始文件直接下载返回403，网页阅读工具可读取同一官方归档；本地保存了逐项事实和核对记录，**没有下载到七份原文副本**。原始输入价格、资金和北向数据沿用本地固定文件及哈希。北向是东方财富经AKShare取得的旧口径历史数据，25条均有值；不能称为官方被动专项流量。官网归档也不等于当年实时接收回执或完整初始版本。

## 来源与文件

{chr(10).join(source_list)}

- `protocol.json`：在本轮收益计算前固定的范围、窗口、费用和联合状态。
- `source_web_evidence.json`、`source_result.json`、`收益计算前的来源访问说明.json`：原文地址、事实核对和403后的读取方式。
- `五次纳入的事前状态与历史结果.csv`：全部事件和条件值；`十个固定窗口日线.csv`：取价。
- `实施窗口全部北向净买入.csv`：25条事后观察；`事前资金状态取值.csv`：资金时钟。
- `四格联合状态.json`、`result.json`：全部分组与裁决；`保存标签取价及费用复核.csv`：必要的计算核对。
- 源码：`research/historical_msci_inclusion_demand_v1.py`与`research/finalize_historical_msci_inclusion_demand_v1.py`。
"""
    (OUT / REPORT).write_text(text, encoding="utf-8")


def finalize():
    if (OUT / "delivery_receipt.json").exists():
        raise RuntimeError("本轮结果已交付，不覆盖。")
    d = pd.read_csv(OUT / "五次纳入的事前状态与历史结果.csv", dtype={"id": str})
    result = study.read(OUT / "result.json")
    verification = check_saved(d)
    figure(d)
    report(d, result)
    summary = "新增核对七份MSCI官方归档，固定五次纳入提升与事前价格资金四格。实施日北向均净买入，但五日净收益1正4负、均值-0.783%，未扣费均值-0.485%；当前五日表达拒绝，未拟合或开新账户。"
    next_question = "MSCI纳入固定五日表达封存；后续选择新的指数级机制，须补上相对已知预期和实际剩余约束的证据。不要把份额、目标权重或已实现净买入直接赋成方向高分，也不要改变本轮窗口或筛选条件营救。"
    result_path = str((OUT / "result.json").relative_to(ROOT)).replace("\\", "/")
    report_path = str((OUT / REPORT).relative_to(ROOT)).replace("\\", "/")
    config_paths = [ROOT / "config/510300_historical_cause_discovery_v1.json", ROOT / "config/510300_existing_data_training_mandate_v1.json"]
    snapshots = OUT / "prior_status"
    snapshots.mkdir(exist_ok=True)
    for path in config_paths:
        (snapshots / path.name).write_bytes(path.read_bytes())
        j = study.read(path)
        if path.name.startswith("510300_historical_cause"):
            j.update(latest_completed_study=result_path, latest_report=report_path, updated_at=study.now(),
                     current_study=str((OUT / "protocol.json").relative_to(ROOT)).replace("\\", "/"), latest_result_summary=summary,
                     next_historical_question=next_question, latest_completed_branch_boundary="五次MSCI纳入的固定五日表达拒绝封存；没有新账户、分数或前瞻信号。")
        else:
            j.update(current_round=study.STUDY, latest_progress_receipt=result_path, last_research_result=summary,
                     latest_continuation_report=report_path, latest_continuation_classification="PROGRESS_MSCI_INCLUSION_JOINT_STATE_DISCOVERY",
                     current_driver_continuation_classification="PROGRESS_MSCI_INCLUSION_JOINT_STATE_DISCOVERY",
                     current_driver_consecutive_blocked_goal_turns=0, next_research_question=next_question,
                     latest_historical_diagnostic_at=study.now(), latest_historical_report=report_path,
                     latest_historical_msci_inclusion_demand=result_path,
                     last_source_result="官网阅读工具核对七份官方归档的日期和比例；直接下载403，无本地原文二进制；北向、资金和价格复用固定历史数据。",
                     goal_status="active", goal_achieved=False)
        path.write_text(json.dumps(j, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    study.save("delivery_receipt.json", {"completed_at": study.now(), "result": result_path, "report": report_path,
                                        "figure": str((OUT / FIGURE).relative_to(ROOT)).replace("\\", "/"),
                                        "source_documents_consulted": 7, "source_original_binaries_downloaded": 0,
                                        "events": 5, "new_label_windows": 10, "new_model_fits": 0, "new_accounts": 0,
                                        "verification": verification, "goal_achieved": False,
                                        "files": {p.name: study.sha(p) for p in OUT.iterdir() if p.is_file() and p.suffix in [".json", ".csv", ".parquet", ".md", ".png"]}})
    print("已保存五事件报告、图表、必要取价复核与继续研究状态。", flush=True)


if __name__ == "__main__":
    finalize()
