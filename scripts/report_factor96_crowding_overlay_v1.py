"""保存T06固定失败、T07来源门和因子库累计研究进度。"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.factor96_crowding_overlay_v1 import OUT, STUDY, now, save

PROGRAM = ROOT/"reports/research/510300_factor96_program_v1"
LABELS = {"PRICE_ALL": "价格基准：全部覆盖", "PRICE_COMMON": "价格基准：共同覆盖",
          "FINANCE_ONLY": "仅融资条件削减", "INTERNAL_ONLY": "仅内部走弱削减", "FULL": "主方案：联合条件削减"}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def figure():
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    for policy, color, style in [("PRICE_COMMON", "#969da9", "--"), ("FINANCE_ONLY", "#b48b37", ":"),
                                  ("INTERNAL_ONLY", "#bb6976", "-"), ("FULL", "#247995", "-")]:
        z = pd.read_parquet(OUT/f"accounts/MAIN/200000/STRESS/LAG1/{policy}/ledger.parquet")
        dates = pd.concat([pd.Series([z.date.iloc[0]-pd.Timedelta(days=1)]), z.date], ignore_index=True)
        nav = np.r_[1., z.equity/200000]
        axes[0].plot(dates, nav, label=LABELS[policy], color=color, ls=style, lw=1.8 if policy == "FULL" else 1.3)
        axes[1].plot(dates, (nav/np.maximum.accumulate(nav)-1)*100, color=color, ls=style, lw=1.8 if policy == "FULL" else 1.3)
    axes[0].set_title("T06 融资拥挤削减层：联合条件未改善收益，目标未达成", fontproperties=font, fontsize=17, loc="left", pad=58)
    axes[0].legend(prop=font, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
    axes[0].set_ylabel("完整账户净值", fontproperties=font)
    axes[1].set_ylabel("自账户峰值回撤（%）", fontproperties=font)
    axes[1].set_xlabel("2021—2025年；保留全部现金日、费用及分红", fontproperties=font)
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.08, .025, "20万元 · 压力费用 · 共同5日持有上限 · 风险预算随回撤收缩，末段变平不表示盈利恢复", fontproperties=font, fontsize=10, color="#555555")
    fig.tight_layout(rect=[.02, .055, .99, .99])
    fig.savefig(OUT/"主期完整账户净值与回撤.png", dpi=160)
    plt.close(fig)


def main():
    assert not (OUT/"研究结论.md").exists(), "本轮正式报告已存在，不重复覆盖进度"
    result, verified = read(OUT/"result.json"), read(OUT/"saved_verification_receipt.json")
    assert verified["ledgers"] == 56 and not result["historical_joint_point_pass"]
    m = pd.read_csv(OUT/"metrics.csv")
    selected = m[(m.period == "MAIN") & (m.cost == "STRESS") & (m.lag == 1)]
    full = selected[selected.policy.eq("FULL")].set_index("capital")
    primary, small = full.loc[200000], full.loc[20000]
    early = m[(m.period == "EARLY") & (m.capital == 200000) & (m.cost == "STRESS") & (m.lag == 1) & (m.policy == "FULL")].iloc[0]
    delayed = m[(m.period == "MAIN") & (m.capital == 200000) & (m.cost == "STRESS") & (m.lag == 2) & (m.policy == "FULL")].iloc[0]
    comparisons = []
    for row in selected[selected.capital.eq(200000)].itertuples():
        comparisons.append(f"|{LABELS[row.policy]}|{row.net_sharpe:.6f}|{row.cagr:.3%}|{row.max_drawdown:.3%}|{row.closed_cycles}|")
    increments = []
    for row in result["increments"]:
        other = row["comparison"].removeprefix("FULL_MINUS_")
        increments.append(f"|主方案减{LABELS[other]}|{row['annual_arithmetic_increment']*100:.6f}|[{row['ci95_low']*100:.6f}, {row['ci95_high']*100:.6f}]|")
    decisions = pd.read_parquet(OUT/"accounts/MAIN/200000/STRESS/LAG1/FULL/decisions.parquet")
    ledger = pd.read_parquet(OUT/"accounts/MAIN/200000/STRESS/LAG1/FULL/ledger.parquet")
    cut = decisions.overlay_active & ~decisions.overlay_before
    cuts = decisions.loc[cut].copy()
    cuts.to_csv(OUT/"主方案全部削减事件.csv", index=False, encoding="utf-8-sig")
    action_rows = []
    for row in cuts.itertuples():
        action_rows.append(f"|{row.date.date()}|{row.stat_date.date()}|{row.F5:.3%}|{row.r5:.3%}|{row.median20_change5:.3%}|{row.filled_quantity:,}|")
    report = f"""# T06融资拥挤持仓削减：固定检验结果

**只操作510300、完整账户成本后夏普至少1.2的目标仍未实现。** T06主方案在2021—2025年、20万元压力成本下净夏普{primary.net_sharpe:.6f}、净年化{primary.cagr:.3%}、最大回撤{primary.max_drawdown:.3%}，期末{primary.end_equity:,.2f}元，累计损益{primary.net_profit:,.2f}元。联合条件实际触发7次削减，相对无削减对照的年化收益差接近零，95%区间包含零。2万元主方案夏普{small.net_sharpe:.6f}，同样未达标。这是固定持仓削减层的失败记录，不是新独立入场策略。

本库累计完成T03、T05、T14、T02、T10、T06共6个固定主问题，包含5个入场候选和1个覆盖层，合格项0。正式账户情景累计264个，另保留T02原实现40个已否定情景，实际执行累计304个。各情景包含时期、本金、费用、对照和滞后版本，不能当作264个独立策略。其余12项尚未完成原定义绩效检验；96个因子也没有逐个获得独立收益验证。

## 固定问题与旧研究的区别

旧融资级联研究使用多项全A特征和分类器，固定失败结果保留。本轮只比较固定20日价格基准下的持仓管理：无削减、单独融资条件、单独成分走弱、两个条件共同触发。单条件均采用相同数据覆盖门，PRICE_ALL另用于显示缺失覆盖影响。没有从旧研究中更换阈值、ETF池或择取盈利年份。

20日基准只充当覆盖层的观察对象，不声称它已验证有效。T06草案明确max_hold_days=5；在计算新收益前，将其解释为所有对照臂共同最多持有5个开盘间隔。旧PRICE20_ONLY研究的60日持有、6%止损、8%追踪退出和重入等待未混入本轮。本次失败也不构成更换基准后继续试削减参数的许可。

## 规则、缺失与交易时钟

统计日s的融资余额五日变化F5须高于严格前252交易日的80分位，且510300含分红五日收益不为正，融资条件才成立。F5窗口连续六日必须完整，分位至少有120个有效历史观测，当前F5不进入自身阈值。该阈值并不另要求F5>0，不能把规则名称当成绝对融资增长限制；本轮主期7个实际触发的F5均为正。F05中的余额/匹配自由流通市值没有使用，仍属未检验部分。

内部走弱取s与s−5均为当时有效沪深300成员、且两端20日总收益都完整的同一批股票，至少294只，两端均须通过原四态数据门。比较这批成员在两端的20日总收益中位数，当前低于五日前才为真。官方确认停牌可保留零收益；未确认缺失、来源冲突或未解决公司行为不能填零。ETF20日收益减中位数的分歧另存描述列，未增设筛选阈值，也不用等权中位数冒充权重贡献。

主版本在下一A股交易日收盘判断s的融资、价格停滞和成分状态，再于随后开盘模拟执行；整个经济统计日一起滞后。额外一日延迟只作敏感性，不选出新主方案。源历史首版尚未认证，延迟是保守历史研究假设，不是已证明的实时可交易时钟。

价格财富指数高于20日均线，且共同数据门已知，才能下一开盘进入新基准周期。削减只管理已经存在的周期：先按最新ES与回撤预算收缩允许份额，再将其减半，向下取整100份。两个条件都明确为假且连续两日，才可恢复风险允许的原份额；未知重置清除计数并保留削减。单条件对照等待自身条件消失两日。原周期份额上限只能下调，不因盈利或价格上涨向上扩张。

价格基准跌回均线、五日到期或10%回撤停机，退出优先并持续请求至合法成交。因半仓整手取整成为0份，仍保持虚拟基准周期和削减状态，不能下一日误当新机会恢复原仓。所有交易均限制510300.SH和CASH_CNY，采用100份、0.001元、T+1、方向涨跌停、现金约束及分红权益/应收/付款。

共同风险沿用前两自然年、五日结果已成熟的相近状态ES95；每天更新，最高50%目标、2.5%ES预算、10%跳空下5%损失预算、剩余回撤空间折半。回撤达到10%后下一合法开盘退出且不恢复。风险预算不保证真实跳空、涨跌停或延迟退出的损失上限。

## 完整账户结果

以下为2021—2025年、20万元、压力成本，完整保留1,212个交易日。年化242日，现金及无风险基准按零计；252日诊断另存，不据此换口径。

|固定情景|净夏普|净年化|最大回撤|完整持仓周期|
|---|---:|---:|---:|---:|
{chr(10).join(comparisons)}

2万元主方案期末{small.end_equity:,.2f}元，损益{small.net_profit:,.2f}元，净年化{small.cagr:.3%}、最大回撤{small.max_drawdown:.3%}。早期2017—2020年20万元压力主方案净夏普{early.net_sharpe:.6f}、净年化{early.cagr:.3%}，同样未达标。额外一日延迟主期净夏普{delayed.net_sharpe:.6f}、净年化{delayed.cagr:.3%}，保持敏感性身份。

主期共同数据可用1,137天，融资条件61天、内部走弱595天、联合条件36天；其中价格基准仍为真17天，结合真实持仓和退出优先级只触发7次削减。主期7次削减后均在满足两日恢复前先结束原基准周期，没有实际恢复。单条件和早期延迟情景确有恢复记录；冻结前测试也专门覆盖了恢复路径。覆盖状态保留到退出日产生的零份额记录，不全是“减半100份变零”的实例。

|削减执行日|对应统计日|F5|ETF五日收益|同成员中位收益变化|实际股数变化|
|---|---|---:|---:|---:|---:|
{chr(10).join(action_rows)}

主期主方案共{int(primary.fills)}次成交、{int(primary.closed_cycles)}个完整持仓周期；佣金{primary.commission:,.2f}元、滑点{primary.slippage:,.2f}元。平均账户风险暴露{primary.mean_exposure:.3%}，{int(ledger.shares.eq(0).sum())}天收盘零份额。接近回撤限制后，风险预算持续缩小，净值末段变平不能解读为盈利能力恢复。BASE每边万2、最低5元、5bp滑点；STRESS每边万4、最低5元、10bp滑点，价格向不利方向取整，期末持仓另提压力退出成本准备。

同一主期逐日配对收益，使用冻结的20日循环区块和4,000次已保存抽样：

|比较|年化算术收益差（百分点）|95%区间（百分点）|
|---|---:|---:|
{chr(10).join(increments)}

全部区间包含0，未证明联合条件具有稳定增量；没有做全历史多轮选择偏差校正。各臂账户风险预算会随自身净值变化，配对差是整个动态削减政策的差别，不是固定相同份额下的局部因果效应。

![完整账户净值与回撤](主期完整账户净值与回撤.png)

## T07来源门

现有本地同指数份额源来自旧固定10只上海ETF的周度研究，其中沪深300子池仅3只。旧研究五条固定二元规则已经失败，不能改池、改阈值、改方向或改时钟复活。本库G03却要求当时动态同指数产品池，以前日NAV乘日份额增量、除以前日资产，再汇总五日；G04还要比较510300与整个体系的迁移差。这些定义不能由原固定池周度收盘价乘份额变化替代。

本轮仍缺当时全体系产品上市/退出、日频份额及拆分、匹配前日NAV和对应数值的历史发布时间依据。因此T07保留NOT_RUN_DAILY_SYSTEM_FLOW_SOURCE_GATE，收益为未计算，不计作第7个完成策略。既有份额发布时点闭环检查已保存2,220个历史响应；它们仅有统计日和数值等字段，下载时间不能充当首次公开时间，不无差别重扫这批响应。

交易所材料说明ETF申赎清单在开市前公布，但这不证明另一个规模查询表TOT_VOL的每个历史数值已在同一时点公开。[上交所ETF申赎说明](https://etf.sse.com.cn/fund/learning/download/c/10055613/files/32901abf70d54e999f46975b52ba46de.pdf) 以上为本地已找到来源的准入判断，不声称外部不存在可补齐的官方资料。

## 复算与停止条件

冻结前11项测试通过；独立只读脚本从分类源价格和公司行为复算1,330,832条交易收益，核对14,771条官方停牌零收益，以逐窗直接相乘交叉核对研究代码的滚动对数和，复算同成员比较、融资分位和滞后。56份账户共61,208行，逐行重放减半/恢复/到期、风险份额、费用、T+1及方向涨跌停；2,990份成熟风险记录的候选窗口、相近状态排序、已成熟标签和ES均一致。保存抽样只读复算，没有新随机搜索或新账户。

这证明固定输入到保存结果的算术一致，不认证原始历史首版、数据修订可得性或策略有效性。外部GPT审阅未进行，独立前向样本0；暂停的采集任务没有重启，没有真实交易。分类源上游证据和此前失败结果随旧完整交付包保存；本轮直接输入、冻结代码与结果在当前包内。

T06按固定定义结束，不改80分位、5日、20日、两日恢复或价格基准救援，也不将单条件对照晋升为赢家。下一轮优先核对T12实际回购新增披露与T13已知供给窗口，先判断来源时点和旧解禁研究的实质差别，再决定能否冻结；不能把计划回购当实际回购，不能把可解禁当实际卖出。T04、T09、T07继续保留各自数据门，T16等待已接受的基准，T17不能拼接失败分支，T18需额外解决成熟校准和重复搜索边界。
"""
    (OUT/"研究结论.md").write_text(report, encoding="utf-8")
    figure()
    strategies, factors = read(PROGRAM/"strategy_progress.json"), read(PROGRAM/"factor_progress.json")
    for row in strategies:
        if row["id"] == "T06":
            row.update(current_status="COMPLETE_FIXED_OVERLAY_TARGET_FAILED",
                       current_evidence="固定5日价格观察基准上的联合减半层，56账户；主期20万元压力夏普-0.639362，7次削减、0次恢复，增量区间含0。不是新增独立入场。",
                       result_path="reports/research/510300_factor96_crowding_overlay_v1/result.json")
        elif row["id"] == "T07":
            row.update(current_status="NOT_RUN_DAILY_SYSTEM_FLOW_SOURCE_GATE",
                       current_evidence="现有固定上海ETF池周度份额不能代替历史动态同指数体系日频NAV申赎；缺份额拆分及历史发布时钟，不救援旧五条失败规则。",
                       source_gate_path="reports/research/510300_factor96_crowding_overlay_v1/source_evidence/overlap_and_source_decisions.json")
    for row in factors:
        if row["id"] == "F05":
            row.update(current_status="PARTIAL_COMPONENT_TESTED", current_note="T06中融资增长高分位且价格停滞的削减条件未达标；余额/自由流通市值未额外测试，不赋独立因子夏普。")
        elif row["id"] == "E05":
            row.update(current_status="PARTIAL_COMPONENT_TESTED", current_note="T06采用同成员20日中位收益五日下降，ETF减中位数仅描述；联合削减及单条件对照均未达标。")
        elif row["id"] in ["G03", "G04"]:
            row.update(current_status="NOT_RUN_DAILY_SYSTEM_FLOW_SOURCE_GATE", current_note="缺动态同指数日频份额/NAV和历史发布时钟；未以旧固定池周频代理替代。")
    save(PROGRAM/"strategy_progress.json", strategies)
    save(PROGRAM/"factor_progress.json", factors)
    pd.DataFrame(strategies).to_csv(PROGRAM/"18策略当前进度.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(factors).to_csv(PROGRAM/"96因子当前进度.csv", index=False, encoding="utf-8-sig")
    previous = read(OUT/"program_before/status.json")
    assert previous["cumulative_admitted_account_scenarios"] == 208
    status = {**previous, "at": now(), "completed_fixed_candidates": [*previous["completed_fixed_candidates"], "T06"],
              "completed_entry_candidates": ["T03", "T05", "T14", "T02", "T10"], "completed_overlays": ["T06"],
              "not_run_candidates": 12, "admitted_account_scenarios_this_round": 56,
              "cumulative_admitted_account_scenarios": 264, "cumulative_executed_account_scenarios": 304,
              "latest_round": STUDY, "latest_result": "reports/research/510300_factor96_crowding_overlay_v1/result.json",
              "next_candidates": ["T12_DISCLOSURE_SOURCE_AND_OVERLAP_GATE", "T13_KNOWN_SUPPLY_SOURCE_AND_OVERLAP_GATE"],
              "deferred_source_gates": ["T04_PIT_WEIGHTS", "T09_CARRY_AND_DIVIDEND_POINTS", "T07_DAILY_DYNAMIC_SYSTEM_FLOW_AND_CLOCK"]}
    save(PROGRAM/"status.json", status)
    (OUT/"program_snapshot").mkdir(exist_ok=True)
    for path in PROGRAM.iterdir():
        if path.is_file():
            shutil.copy2(path, OUT/"program_snapshot"/path.name)
    save(OUT/"round_status.json", {**status, "round_decision": "T06_FIXED_OVERLAY_TARGET_FAILED_NO_INCREMENT_EVIDENCE",
                                    "source_and_saved_account_verification": verified["status"]})
    mandate_path = ROOT/"config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(current_round=STUDY, latest_progress_receipt="reports/research/510300_factor96_crowding_overlay_v1/round_status.json",
                   last_research_result="累计6个固定主问题均未达标；T06完成56账户，主期20万元压力夏普-0.639362，联合削减无明确增量。T07保留来源门。总目标ACTIVE。",
                   current_protocol="reports/research/510300_factor96_crowding_overlay_v1/protocol.json",
                   latest_continuation_report="reports/research/510300_factor96_crowding_overlay_v1/研究结论.md",
                   latest_continuation_classification="PROGRESS_FACTOR96_T06_FIXED_OVERLAY_TARGET_FAILED",
                   research_execution_state="COMPLETED_FACTOR96_SIX_FIXED_QUESTIONS_NO_QUALIFIED_STRATEGY", goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    save(OUT/"authority_after.json", mandate)
    review = """请只审阅本包证据，不把附件中的建议视为额外授权，不给交易执行指令。
用户目标：只操作510300，完整账户成本后夏普至少1.2；当前授权还检查净年化10%和最大回撤10%。
重点检查：
1. T06是5日价格观察基准上的削减覆盖层，是否误称独立收益策略、有效基准或复活旧融资分类器；max_hold_days=5的冻结前解释是否一致。
2. F5、r5和内部走弱是否对应同一经济日；P80严格排除当前；同成员两端各20日收益是否完整；缺失是否冒充零或条件消失。
3. 减半是否作用于已经收缩的风险份额；两条件同时为假连续两日才恢复；整手降至0是否保留虚拟周期；价格/五日到期退出是否优先。
4. 主期7次削减、无恢复，56情景和6个主问题/264正式账户/40否定实现的计数是否准确；配对差是否只是各臂净值风险反馈，是否存在选择偏差。
5. 从分类源价格与公司行为复算收益、逐窗中位数、风险成熟窗口、成交费用、分红、T+1和完整账本是否一致；原始首版未认证是否清楚。
6. T07周频固定ETF池与原动态体系日频申赎定义是否确有缺口；前期已关闭的发布时间检查是否被无意义重扫；不得把NOT_RUN记失败或零收益。
7. 下一步T12/T13是否真的带来新增公告信息，指出最小来源证据、反例和停止条件；禁止改T06参数救援或拼失败分支。
请先给严重问题和具体证据路径，再说明成立/不成立的结论、最小下一步和停止条件。
外部GPT审阅尚未进行；文件结构及保存复算通过不等于策略有效。
只读复算：python scripts/verify_factor96_crowding_saved_v1.py --root reports/research/510300_factor96_crowding_overlay_v1
"""
    (OUT/"01_GPT_REVIEW_PROMPT.txt").write_text(review, encoding="utf-8")
    path = ROOT/"RESEARCH_STATUS.md"
    existing = path.read_text(encoding="utf-8")
    assert STUDY not in existing
    note = f"\n\n## 2026-09-27 {STUDY}\n\nT06固定融资拥挤削减层完成56账户；主期20万元压力净夏普{primary.net_sharpe:.6f}、净年化{primary.cagr:.3%}、最大回撤{primary.max_drawdown:.3%}，7次削减、0次恢复，联合增量区间含0。11项冻结前测试与61,208账本行/2,990成熟风险记录复算通过。T07缺历史动态全体系日频份额、NAV和发布时间，维持NOT_RUN。累计6个固定主问题（5入场+1覆盖层）、0合格、264正式情景+40否定实现，总目标ACTIVE。详见reports/research/510300_factor96_crowding_overlay_v1/研究结论.md。\n"
    path.write_text(existing+note, encoding="utf-8")
    print("T06报告、图表和累计进度已保存：6个固定问题，0项合格，总目标继续。", flush=True)


if __name__ == "__main__":
    main()
