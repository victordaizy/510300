"""保存完整金融研究报告及四场景图，不重复账户或改变失败配置。"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from research import industry_leader_failure_diagnosis_v1 as diagnosis
from research import industry_leader_failure_study_v1 as study
from research.write_industry_structure_description_report_v1 import markdown, value


def plot():
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False, "font.size": 9})
    colors = {study.CONTROL_A: "#247b91", study.CONTROL_STAGE: "#725c89", study.candidate.POLICIES[0]: "#b7722d",
        study.candidate.POLICIES[1]: "#7f8b72", study.candidate.POLICIES[2]: "#92999f"}
    figures = []
    for period in study.parent.PERIODS:
        for cost in study.parent.COSTS:
            fig, axes = plt.subplots(3, 1, figsize=(12.8, 10.2), sharex=True)
            for policy in (study.CONTROL_A, study.CONTROL_STAGE, *study.candidate.POLICIES):
                d = diagnosis.saved(period, cost, policy)["daily"]
                opacity = .58 if policy in study.candidate.POLICIES[1:] else 1
                width = 1.1 if policy in study.candidate.POLICIES[1:] else 1.6
                label = study.NAMES[policy]
                axes[0].plot(d.date, d.equity / 200000, label=label, color=colors[policy], alpha=opacity, linewidth=width)
                axes[1].plot(d.date, 100 * d.drawdown, label=label, color=colors[policy], alpha=opacity, linewidth=width)
                if policy in (study.CONTROL_STAGE, study.candidate.POLICIES[0]):
                    axes[2].plot(d.date, 100 * d.exposure, label=label, color=colors[policy], linewidth=1.)
            axes[1].axhline(10, color="#a9a9a9", linewidth=.8, linestyle="--", label="原10%回撤门")
            for axis, label in zip(axes, ["完整账户净值／20万元", "完整账户回撤／%", "持仓市值／净值 %"]):
                axis.set_ylabel(label)
                axis.grid(axis="y", alpha=.2)
                axis.legend(loc="best", fontsize=8, ncol=2)
            fig.suptitle(f"{period} / {cost}：固定行业失效与全部账户对照\n原资金、费用、风险和现金日；本配置拒绝，历史开发结果", fontsize=13)
            fig.tight_layout(rect=(0, 0, 1, .94))
            path = study.OUT / "figures" / f"{period}_{cost}_完整净值回撤与暴露.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=140)
            plt.close(fig)
            figures.append({"period": period, "cost": cost, "path": study.relative(path), "sha256": study.digest(path)})
    return figures


def main():
    out, parent = study.OUT, study.parent
    path = out / "实际进入行业失效_完整结果与失败归因.md"
    if path.exists() or (out / "delivery_receipt.json").exists():
        raise RuntimeError("行业失效金融报告已经保存，不覆盖。")
    summary = study.read(out / "summary.json")
    diag = study.read(out / "post_run_diagnosis.json")
    metrics = pd.DataFrame(summary["metrics"])
    full = markdown(["时期／费用", "政策", "净CAGR", "净夏普", "最大回撤", "完成／赢／输", "胜率", "实际净盈亏比", "p×B", "标准净期望", "完整年次数"], [
        [f"{r.period}／{r.cost}", study.NAMES[r.policy], value(r.net_cagr, True, 3), value(r.net_sharpe, digits=6),
         value(r.max_drawdown, True, 3), f"{r.completed_cycles}／{r.wins}／{r.losses}", value(r.win_rate, True, 2),
         value(r.payoff), value(r.p_times_b), value(r.standard_expectancy_loss_units), value(r.average_full_year_cycles, digits=3)] for r in metrics.itertuples()])
    timing = markdown(["时期／费用", "政策", "实际第五日检查", "领先可知／未知", "额外请求／成交"], [
        [f'{r["period"]}／{r["cost"]}', study.NAMES[r["policy"]], r["checks"],
         f'{r["known_leader_checks"]}／{r["unknown_leader_checks"]}', f'{r["extra_requests"]}／{r["actual_extra_sells"]}'] for r in summary["timing_checks"]])
    extra = pd.read_parquet(out / "results/全部实际额外退出_原周期盈亏与失效信息.parquet")
    pressure = extra.loc[extra.cost.eq("STRESS")]
    actual_exit = markdown(["时期／实际进入", "观察／实际卖出", "原领先行业累计", "ETF累计", "新净损益元", "阶段原净损益元", "阶段原卖出"], [
        [f"{r.period}／{r.entry_date:%Y-%m-%d}", f"{r.decision_date:%m-%d}／{r.exit_date:%m-%d}",
         value(r.leader_mean_cumulative_return, True, 3), value(r.etf_lagged_five_return, True, 3),
         value(r.net_pnl, digits=2), value(r.stage_net_pnl, digits=2), f"{r.stage_exit_date:%Y-%m-%d}"] for r in pressure.itertuples()])
    context = pd.read_parquet(out / "results/原八描述周期与实际进入五日检验_全部对应.parquet")
    clock = markdown(["原信号／说明观察", "真实进入后观察", "ETF累计", "领先行业累计", "实际状态"], [
        [f"{r.anchor_date:%Y-%m-%d}／{r.date:%m-%d}", str(r.actual_decision_date)[:10] if not pd.isna(r.actual_decision_date) else "未到检查",
         value(r.actual_etf_lagged_five_return, True, 3), value(r.actual_leader_mean_cumulative_return, True, 3), r.actual_check_status] for r in context.itertuples()])
    attribution = markdown(["时期／费用", "毛损益元", "全部摩擦元", "净损益元", "相对阶段完整财富差", "其中完成／开放差", "固定实际数量零费用CAGR／夏普"], [
        [f'{r["period"]}／{r["cost"]}', value(r["actual_gross_pnl"], digits=2), value(r["total_friction"], digits=2),
         value(r["actual_net_pnl"], digits=2), value(r["ending_equity_delta_vs_stage"], digits=2),
         f'{value(r["completed_pnl_delta_vs_stage"],digits=2)}／{value(r["open_pnl_delta_vs_stage"],digits=2)}',
         f'{value(r["zero_cost_fixed_actual_quantities_cagr"],True,3)}／{value(r["zero_cost_fixed_actual_quantities_sharpe"],digits=4)}'] for r in diag["attribution"]])
    intervals = pd.read_parquet(out / "results/全部四对照两尺度成本后增量区间.parquet")
    ci_table = markdown(["时期／费用", "对照", "块", "CAGR增量95%区间（百分点）", "夏普增量95%区间"], [
        [f"{r.period}／{r.cost}", study.NAMES[r.control], r.block,
         f"[{100*r.cagr_delta_lower:.3f}, {100*r.cagr_delta_upper:.3f}]",
         f"[{r.sharpe_delta_lower:.4f}, {r.sharpe_delta_upper:.4f}]"] for r in intervals.itertuples()])
    annual = pd.read_parquet(out / "results/全部逐年实际完成周期与账户收益.parquet")
    annual = annual.loc[annual.policy.eq(study.candidate.POLICIES[0])]
    year_table = markdown(["时期／费用", "年", "完整年", "完成周期", "完整账户年收益"], [
        [f"{r.period}／{r.cost}", r.year, "是" if r.full_year else "不足全年", r.completed_cycles,
         value(r.calendar_year_return, True, 3)] for r in annual.itertuples()])
    figures = plot()
    figure_text = "\n\n".join(f'![{r["period"]}／{r["cost"]}](figures/{r["period"]}_{r["cost"]}_完整净值回撤与暴露.png)' for r in figures)
    report = f"""# 实际进入后的固定领先行业失效：完整结果与失败归因

2026-10-06；TECH.R223登记，TECH.R224结果。实际结果：**拒绝该固定配置**。四场景经济门0/4，历史稳定门不通过；收益/夏普完整目标未达，过拟合去除和独立验证未建立。该结论约束本次五日固定行业退出用途，允许另立不同进入机制。

## 最重要的结果

近期2020—2026压力费用下，新主政策净年化−0.1242%、全日历净夏普−0.026868、最大回撤7.0748%，45完成周期15赢30输。胜率33.33%，实际平均净赢／平均净亏回报比1.9923，p×B=0.6641，标准净期望−0.002580亏损单位。相对原阶段−0.2400%／−0.066082改善，但相对原A3.9908%／1.216910显著落后；同观察覆盖仅ETF正退出的−0.0487%／0.001210也更好。

早期2015—2019压力费用下，新主政策2.0172%／0.493683，低于原阶段2.1871%／0.529186；28完成周期12赢16输，p×B=0.9157未过用户>1门。标准净期望+0.3443仍正，所以不能把“p×B未过1”误写成每个场景数学期望必负。用户p×B>1是额外的强要求，本研究分别报告两项。

## 本次实际做了什么

用R220实际分类/公布钟、R222固定行业方法和原成分含息收益，在实际进入前一源日固定前三行业。与R222信号锚点不同，唯一检查为实际进入索引加5收盘，使用进入日至进入后第4日五个已形成回报，下一合法开盘执行；它不是实际开盘成交收益。ETF仍正且原领先行业非正时额外退出，原结构/周线/宏观风险退出优先。行业只作观察，交易范围510300.SH/CASH_CNY。

退出观测对象仅三领先行业，各98%股票后锚回报完整及当日市场源允许/300成员才可使用；跟随完整性独立保存。该用途在新金融结果前明确，未改R222全组观察门，也未按结果调整覆盖。分类公布钟不得晚于实际买入09:30，初始锚仍用294/300、行业至少5成员/98%20报价和至少4行业的原条件。源龄、未知和原来源失败均保留。

同覆盖价格对照只要求相同三领先可知门及ETF累计正；全日历价格对照只要求ETF正。原A和阶段账户为基线。原资金20万元、风险和50%上限、T+1/整手/最小费用/股息权益/自然开放终点/完整现金日/两时期两费用保持。未补买、未拆成事后可执行组合。

9必要测试通过，原R212四场景五账表及终态精确复现后固定配置，一次完整运行12新账户；读取8保存基线，形成20账户比较和32个两尺度区间。12新账户现金/库存/毛净损益及信息时序核验通过。0新拟合、训练目标、采集、参数网格或金融重跑。

## 全部20账户

{full}

净CAGR、夏普含所有现金日和真实费用；B来自实际完成周期净回报而非计划止盈止损，标准期望=p×B−亏损概率。完整年次数2015—2019包括2019，2020—2026包括2020—2025，2026单列不当全年。近期45比原阶段44多一个完成周期，但新增发生在2026，完整年频率仍6.667，没有变成更高的每年频率。

## 为什么原8个说明性亏损没有成为8次可执行退出

原R222使用原信号作锚，本次必须用实际买入。原8周期中，3个已在实际第五日检查前或当日开盘由原规则退出，4个在真实第五日ETF同期已非正，因此不满足本次“ETF仍正”的规则；只有2023年3月周期实际触发。其余3次近期真实触发来自原8以外的进入路径。不能将原8亏损加总成策略增量。

{clock}

全实际检查及未知如下，未知不触发、不延后重试、不从评价分母删除：

{timing}

## 实际退出与资金再使用的影响

每种费用口径，早期只触发1次额外退出、近期4次，全部有下一合法开盘实际卖单。压力费用下近期4笔原亏周期都改善，对应净损益差合计+2808.29元；提前释放资金后新增2026-05-11信号、05-12实际进入的重新定价交易，净亏−1753.77元。其他持仓数量变化也影响收益；完整账户实际只比阶段改善+1488.24元，不能只报+2808.29元。

早期2017-03-22进入、03-30提前卖出：基础费用原小盈利20.00元变为−28.59元，压力费用原−89.81元变−132.54元；不是两种费用都在截断原赢家。释放资金后新增2017-04-05信号、04-06进入、04-19退出的重新定价交易，压力净亏−1181.26元；连同其他数量及开放损益差，完整账户反而少1780.55元。

{actual_exit}

旧恢复反例保持：2019-06-12进入仍至07-10退出，新压力净盈利1982.65元；2025-06-26进入仍至08-04退出、新净盈利1012.60元，未被本额外退出截断。2024-09-25至10-17保留，新压力净盈利3776.25元/净回报12.1422%。这些金额与原阶段的差别含先前账户变化产生的数量/费用影响，不声称改善了该进入日或把已知大上涨当独立验证。

## 信号、摩擦与账户约束

近期压力毛损益5867.90元，佣金2136.35元、滑点5338.70元，摩擦合计7475.05元，净亏1607.15元。固定实际数量和持有路径，把费用设为0的解释上界年化仅0.4464%、夏普0.1666，仍远低于原A。因此当前主要问题不只是交易费用，原进入与持有形成的毛优势也太弱。这个零费用计算不是可执行政策，没有重算风险、数量和资金行为。

四场景均无实际下单拒绝、未触发账户DD永久停买；压力早期34次、近期46次风险减仓存在，反映原风险预算影响。不能把“无拒单”写成“没有仓位风险约束”，亦不能为了通过直接放开原风险。

{attribution}

完成周期差加开放损益差严格等于完整账户财富差。早期新压力开放损益3746.88元，近期0；原A旧保存终态未提供开放损益字段，本报告保持未知，没有填0。敞口、换手、最坏日/笔、集中度、全部费用、零交易年和最长空仓在完整指标表保存。

## 全部历史区间与稳定判断

原20/252日循环块各2000次、固定种子510300154，四场景对四对照共32个区间完整报告。近期相对原A的收益和夏普增量，两个尺度区间均为负；相对阶段的小幅改善区间跨0，不能称为稳定。早期相对阶段点值变差。以下均为反复使用的开发历史描述，非独立证据，也未校正全部历史选择。

{ci_table}

## 完整逐年次数和收益

{year_table}

## 下一研究方向

当前不继续优化这个五日退出配置。更优先的问题是**进入证据有没有真正更新，以及价格突破是否对应新的资金/宏观信息与行业参与**。本次两个新增亏损都是重新定价路线，但不能因此按已知盈亏直接删除整个路线：2024大上涨也是该路线。

下一描述应使用全部143原进入事件，逐点保存PMI实际公布/参考期、政策利率已知钟、资金价格与融资变化、行业参与/源龄，以及价格量能事件的发生顺序；原四案例、全部假启动和本次两个新增亏损同时解释。区分新的信息更新与旧宏观状态持续、第一次价格确认与同一背景的重复触发，不能只筛原盈利段。先检验当时字段是否足以描述这些不同情形，再固定一个进入—持有完整机制及同覆盖价格对照，按原账户/成本/风险检验。

这项进入信息研究尚未登记/运行。原R212/R216/R224失败保留，历史首版未认证、独立验证NOT_ESTABLISHED；当前没有已准入待跑新金融候选。原13独立前瞻保持，收益夏普目标继续active，不将本配置拒绝等同全项目停止。

## 完整账户图与可复算文件

{figure_text}

依据：[用途卡](../../../docs/510300_INDUSTRY_LEADER_FAILURE_V1.md)、[固定协议](protocol.json)、[实际金融结果](summary.json)、[失败归因](post_run_diagnosis.json)、[20完整指标](results/二十完整账户_行业与价格同资金风险费用比较.csv)、[全部实际退出](results/全部实际额外退出_原周期盈亏与失效信息.csv)、[原8案例真实时钟](results/原八描述周期与实际进入五日检验_全部对应.csv)、[全部区间](results/全部四对照两尺度成本后增量区间.csv)。
"""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(report)
    study.write(out / "delivery_receipt.json", {"at": parent.original.now(), "report": study.relative(path),
        "report_sha256": study.digest(path), "summary_sha256": study.digest(out / "summary.json"), "figures": figures,
        "new_financial_runs_in_delivery": 0, "actual_status": summary["status"], "goal_achieved": False})
    print("金融完整报告和四场景净值/回撤/暴露图已保存，拒绝配置及全部未知保持。", flush=True)


if __name__ == "__main__":
    main()
