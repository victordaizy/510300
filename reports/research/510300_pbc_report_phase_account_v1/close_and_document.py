"""一次归档本轮真实新来源与固定金融结果，更新长期事实文档。"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

ROOT = Path(__file__).absolute().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research import all_factor_macro_earnings_joint_v1 as io
from research import all_factor_macro_earnings_account_v1 as previous
from research import pbc_report_phase_account_v1 as study

OUT = study.OUT
SOURCE = study.SOURCE
STATE = ROOT / "reports/research/510300_daily_weekly_goal_continuation_20261001/state.json"
FORWARD = ["forward_protocol", "forward_registry", "new_prospective_observations", "earliest_future_exchange_session",
           "registered_candidate_intents", "new_prospective_completed_points", "next_new_close_eligible_at", "current_validated_candidates",
           "forward_account_comparison_protocol", "latest_forward_account_check", "new_prospective_sessions_this_continuation",
           "new_prospective_cycles_this_continuation", "next_experiment", "independent_official_share_observer",
           "current_absolute_goal_targets", "current_absolute_goal_contract"]
FINANCE = ["latest_actual_financial_decision", "latest_actual_financial_result", "latest_actual_financial_status",
           "current_new_strategy_return_sharpe", "latest_actual_financial_primary_four_scene_metrics"]


def rel(path):
    return path.absolute().relative_to(ROOT).as_posix()


def git_state():
    return subprocess.run(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, check=True,
                          capture_output=True, encoding="utf-8").stdout


def pretty(value, percent=False):
    if pd.isna(value):
        return "未定义"
    return f"{value:.4%}" if percent else f"{value:.6f}"


def verify_saved_accounts(metrics):
    rows = []
    for row in metrics.to_dict("records"):
        if row["policy"] == "A_SAVED_WEIGHT":
            daily = previous.baseline(row["period"], row["cost"])["daily"]
        else:
            daily = pd.read_parquet(OUT / f"results/accounts/{row['period']}/{row['cost']}/{row['policy']}/daily.parquet")
        returns = daily.net_return.to_numpy(float)
        rebuilt_cagr = (float(daily.equity.iloc[-1]) / 200000) ** (252 / len(daily)) - 1
        rebuilt_sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)) if np.std(returns, ddof=1) > 0 else np.nan
        peaks = np.maximum.accumulate(np.r_[200000., daily.equity.to_numpy(float)])[1:]
        rebuilt_drawdown = float((1 - daily.equity.to_numpy(float) / peaks).max())
        errors = [abs(rebuilt_cagr - row["net_cagr"]), abs(rebuilt_sharpe - row["net_sharpe"]), abs(rebuilt_drawdown - row["max_drawdown"])]
        io.require(np.isfinite(errors).all() and max(errors) < 1e-10, "账户指标从保存日账重算不一致。")
        rows.append({"period": row["period"], "cost": row["cost"], "policy": row["policy"],
                     "calendar_days": len(daily), "maximum_metric_recomputation_error": max(errors)})
    return pd.DataFrame(rows)


def main():
    summary = io.read(OUT / "summary.json")
    io.require(io.read(OUT / "run_completed.json")["terminal"], "实验未完成不能归档。")
    io.require(io.read(OUT / "goal_service_active_at_close.json")["goal"]["status"] == "active", "未有当前实际active服务证据。")
    io.write(OUT / "close_started.json", {"at": io.now(), "state_updates_planned": 1}, exclusive=True)
    old = io.read(STATE)
    before_git = git_state()
    backup = OUT / "state_before_R268"
    backup.mkdir()
    shutil.copy2(STATE, backup / "state.json")
    documents = [ROOT / "docs" / name for name in ("PROJECT_STATE.md", "RESEARCH_DECISIONS.md", "PROJECT_STATE_TECHNICAL_LINE.md", "RESEARCH_DECISIONS_TECHNICAL_LINE.md")]
    for path in documents:
        shutil.copy2(path, backup / path.name)
    (backup / "tracked_git_status.txt").write_text(before_git, encoding="utf-8")
    metrics = pd.read_parquet(OUT / "results/全部20账户_四场景同口径比较.parquet")
    verified = verify_saved_accounts(metrics)
    verified.to_parquet(OUT / "全部20账户_保存日账指标重算.parquet", index=False)
    source_summary = io.read(SOURCE / "summary.json")
    original_cases = pd.read_parquet(SOURCE / "原全部30案例_当时报告与原解释等级.parquet")
    predictions = pd.read_parquet(OUT / "results/全部四模型_阶段评分与未知.parquet")
    selected = predictions[predictions.policy.eq(study.PRIMARY)].drop(columns="policy")
    selected = selected.rename(columns={name: "fixed_model_" + name for name in selected.columns if name != "date"})
    cases = original_cases.merge(selected, on="date", how="left", validate="one_to_one")
    pd.testing.assert_frame_equal(original_cases, cases[original_cases.columns], check_exact=True)
    study.table("原全部30案例_冻结主模型预测及原解释等级", cases)
    checks = pd.read_parquet(OUT / "results/全部16账户_资金库存与时钟.parquet")
    recent = metrics[metrics.period.eq("2020_2026") & metrics.cost.eq("STRESS") & metrics.policy.eq(study.PRIMARY)].iloc[0]
    report = ["# 央行双文本与量价阶段：完整账户研究结论", "",
              "固定用途拒绝。新的官方报告原件与案例对应已经取得，但唯一主候选在四个完整账户场景都未满足用户10%净年化、1.5净夏普、10%回撤及实际净pB/期望门。未实现目标，未建立独立验证。", "",
              "## 新信息及来源边界", "",
              "官方央行目录固定2011Q4基准及2012Q1—2026Q2，共59季度；一次请求取得58原件。分开第四部分的经济描述与第五部分的未来政策子章节，分别与立即前季比较中文二元频数余弦变化。它衡量文字变化，不定义利好/利空、超预期、真实资金流或胜率。",
              "2025Q4发布页TLS失败原样保留，未重试。该季文本及下一季前季比较未知；2026/2/11—5/11共54日、5/12—8/12共66日保留无新进入，共120日。完整范围的3368日具有双通道，未删除缺数据时期。历史报告第一原始版本未认证，不能声称独立PIT成功。",
              "首版格式识别将25份已有内容误判未知：旧封面的汉字年份、经济章名宏观经济形势/分析及日历起点前多报告对应已纯本地修正。另保留pandas时间精度连接失败；V2.1仅统一ns。首版与V2源码、登记、原响应和失败均保存，未重采集或改科学度量。",
              "全部30原案例的原值及解释等级逐列保持，全部当时文本可见，但仅使用11个不同季度；30案例/3368日不能充当独立报告数。", "",
              "文献依据：金融研究2021年第6期姜富伟、胡逸驰、黄楠的期刊摘要区分政策指引与经济叙述的市场关系，支持分通道讨论；完整论文尚未取得，未复现情绪字典。2019年Emerging Markets Review原摘要的即时反应不持久是反证，不能推为日周线延续收益。这些是机制线索，不是510300扣费账户业绩证据。",
              "- 官方目录：https://www.pbc.gov.cn/zhengcehuobisi/125207/125227/125957/index.html",
              "- 原期刊摘要：https://www.jryj.org.cn/CN/abstract/abstract897.shtml",
              "- 反证论文：https://doi.org/10.1016/j.ememar.2019.05.002", "",
              "## 已冻结的唯一检验", "",
              "先共同讨论原案例，再登记TECH.R267并运行TECH.R268：主候选为8个日周线量价字段、报告年龄、经济变化及政策变化的阶段条件树；同池去组为量价+报告年龄、再加经济通道、再加政策通道。全因素12类83项讨论及原宏观/盈利结果保留，未把本子模型冒称为全部83项已校准胜率。",
              "各月只用成熟标签、原756日窗口/252行/三类各10；至少8报告季度，先去标签重叠再令各季度总训练权重相同。树深2、叶至少60日、实际叶至少4不同季度，四模型同训练池/同权。预定低复杂度一次552个拟合调用/142月记录，无参数网格或按新结果选择深度。预测可用天数分别2678/2460/2542/2409，差异来自各自预定叶支持条件，未知不补概率。",
              "原20万元、最高50%股票请求及ES/跳空预算、10%回撤停止进入、次开盘、T+1、整手、最少5元手续费、分红、2ATR/1ATR/20收盘退出全保留。两费用、两时期的完整交易日日历包含现金与自然期末持仓；不拼接旧CORE，不增仓位。16新账户与4保存A共同报告，不择优换主候选。", "",
              "|时期|费用|账户|净年化|净夏普|回撤|实际净pB|完成周期|期末财富元|", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in metrics.to_dict("records"):
        report.append(f"|{row['period']}|{row['cost']}|{row['policy']}|{pretty(row['net_cagr'], True)}|{pretty(row['net_sharpe'])}|{pretty(row['max_drawdown'], True)}|{pretty(row['p_times_b'])}|{row['completed_cycles']}|{row['ending_equity']:.2f}|")
    report += ["", "四场景全部门通过False；所有新模型和原A在新绝对目标下均未满足。旧R256三源联合拒绝继续保留，最新实际金融事实更新为本R268，不把描述性接入写成收益提高。", "",
               "近期压力主模型毛损益2531.80元，佣金852.48元、滑点2254.70元，合计摩擦3107.18元，净损益-575.38元；费用消耗了较薄毛收益。早期压力毛损益-10050.10元，本身已负，不能把所有失败归为费用。压力成本不下调。",
               f"近期14个完成周期、7盈7亏；实际胜率50%、净盈亏比{pretty(recent['payoff'])}，pB={pretty(recent['p_times_b'])}，标准净期望{pretty(recent['standard_expectancy_loss_units'])}、周期平均净收益{pretty(recent['mean_cycle_net_return'], True)}。按周期百分比计算的微弱正均值没有变成完整资金净收益，不能代替账户财富、费用和实际分配。逐年完成次数软目标也未改善：较早平均2.6次/完整年，近期约2.1667次，原A相应4.4/5.3333次。",
               "## 原全部30案例：当时文本及既有等级", "",
               "|日期|原解释等级|当时报告|经济文字变化|政策文字变化|固定模型状态|固定模型净pB估计|", "|---|---:|---|---:|---:|---|---:|"]
    for row in cases.to_dict("records"):
        report.append(f"|{pd.Timestamp(row['date']).date()}|{row['core_explanatory_score']}|{row['pbc_quarter']}|{pretty(row['pbc_economic_change'])}|{pretty(row['pbc_guidance_change'])}|{row['fixed_model_status']}|{pretty(row.get('fixed_model_predicted_p_times_b', np.nan))}|")
    report += ["", "2015/6/24—30反弹失败使用2015Q1，政策文字变化约0.013592，不能因文字稳定而推定价格修复；这段日MACD/ATR仍为负。2019/1/8—18使用2018Q3，同一季度文字不能解释日线修复的每一天。2020/3/27—4/23使用2019Q4，报告发布于2/19，不能回填尚未发布的2020Q1。",
               "2024/9/23—30以及10/8都使用2024Q2，经济变化约0.155527、政策变化约0.084545。9月启动与10月拥挤共享宏观文字，阶段区别来自当时价量/波动变化；11/8公布的2024Q3不属于9月事前信息。2025/5/12可见5/9公布的2025Q1；不同于旧5/6模型决定，不能把后信息提前。这里是描述对应，不是新因果断言或独立成功。", "",
               "## 接受、拒绝与重新验证", "",
               "接受完整固定来源范围、原文两个角色、真实公布上界和原30案例对应；修正三个确定实现错误。拒绝这一个已登记的双文本阶段树及其账户用途，四场景经济门失败。没有拒绝所有央行信息或所有未来策略，但不基于本结果改变窗口、叶支持、深度、文本符号、仓位、费用或时期来挽救配置。真正语义、数量预期和因果传导未经本实验验证，保持未知；不因标题重新打开旧催化、盈利/宏观树失败。",
               "下一步优先研究真实新增需求信息与上涨启动/延续/拥挤阶段的关系。已登记官方ETF份额观察是具体入口：原最早首版本窗10/8 23:05、第二相邻版本10/9实际到达后，才可能从10/12观察净份额变化；实际取得钟决定可用时点，缺失不补，份额变化不冒称订单流/人民币净流入。当前这两版本尚未取得，未登记新的可运行收益模型。原1008日机制前瞻合同和两个冻结候选不变，不以本失败历史反选候选。",
               "必要检查已完成：3个阶段语义测试；16新账户资金库存/时钟核对最大误差约4.14e-11元；全部20账户从保存日账重新计算CAGR、Sharpe、回撤吻合。原30字段逐列相等，120未知日保存，独立样本0，overfitting_removed=False，未建立采集自动化或交易。目标服务本轮实际恢复active，旧blocked三轮为历史档案；本轮是取得新来源并实际检验不同用途的PROGRESS，失败结果不等于模型改善。", "",
               "证据入口：同目录protocol.json、summary.json、results/全部20账户_四场景同口径比较.parquet、results/原全部30案例_冻结主模型预测及原解释等级.parquet；上游央行source_v1和source_reconciliation_v2_1。"]
    report_path = OUT / "央行双文本与量价阶段_完整研究结论.md"
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    font = FontProperties(fname="C:/Windows/Fonts/msyh.ttc")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for axis, period in zip(axes, ("2015_2019", "2020_2026")):
        for policy, name, color in (("A_SAVED_WEIGHT", "原A", "#2c6eaf"), ("TECH_AGE_COMMON", "同池量价＋报告年龄", "#c08a34"), (study.PRIMARY, "主双文本阶段模型", "#b34848")):
            daily = previous.baseline(period, "STRESS")["daily"] if policy == "A_SAVED_WEIGHT" else pd.read_parquet(OUT / f"results/accounts/{period}/STRESS/{policy}/daily.parquet")
            axis.plot(pd.to_datetime(daily.date), daily.equity / 200000, color=color, label=name, linewidth=1.5)
        axis.axhline(1, color="#666666", linewidth=.6)
        axis.set_title(period.replace("_", "—") + "｜压力费用完整账户", fontproperties=font, fontsize=12)
        axis.set_ylabel("净值（初始20万元）", fontproperties=font)
        axis.legend(prop=font, fontsize=9)
        axis.grid(alpha=.18)
    fig.suptitle("510300固定文本阶段用途：完整账户结果未通过目标", fontproperties=font, fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "完整压力账户净值比较.png", dpi=150)
    plt.close(fig)
    source_block = """### TECH.R263—R266：央行经济叙述与政策指引完整季度来源（2026-10-06）

假设→经济描述与未来政策指引是不同信息角色；文字变化可用于阶段讨论，但本身无看多方向。
验证→先冻结完整2011Q4基准及2012Q1—2026Q2的59季度，一次官方请求，正文两章节立即前季二元余弦；全部3488原点/原30案例按原15:05公布上界对应，未知不补零、不回退旧报告。
结果→58原件、58双章节、56前季双通道；3368已知/120未知原点；30案例全部对应，仅11不同季度。2025Q4 TLS失败及下一季比较缺口保持；历史第一版未认证。首版25份格式误判、日历起点冲突、V2时间精度失败分别保留；独立V2.1纯本地修正，0新HTTP、未改科学方法或收益规则。
接受/拒绝→接受固定范围与正文/公布钟/案例对应为新的描述输入，不解释为胜率、超预期、订单流或独立验证。首版接入不产生金融收益；数值用途由下一R267独立登记，四场景完整日历保留全部未知。
重新验证→确定格式错误已修正，不重试原失败或删除缺件；真正新原件/第一版本证据或不同机制需另登记。证据在`reports/research/510300_pbc_report_text_source_v1`及`510300_pbc_report_text_source_reconciliation_v2_1`。

"""
    main_block = f"""### TECH.R267—R268：央行双文本×量价阶段固定完整账户拒绝（2026-10-06）

**正式目标仍为20万元完整扣费账户：净CAGR≥10%、净Sharpe≥1.5、实际最大回撤≤10%，另保留实际净pB>1/净期望及独立验证。目标未完成。最新实际金融现在是R268，原R256拒绝保留。**

假设→新增政策指引变化与经济描述变化，可能条件于日周线量价阶段提供不同预测信息；同一季度不会单独区分启动与拥挤。
验证方法→原30案例先对应后冻结一个主候选与三同池去组：原8价量/报告年龄/两文本通道；原月度成熟756日窗口、252行/三类各10、至少8季度，原重叠权重后季度平衡；深度2/叶60且评分叶至少4不同季度。552实际拟合调用、16新账户+4原A，两时期两费用/原资金风险与完整日历，120未知保留无新进入，无参数网格、0新标签。
结果→四场景均拒绝。早期压力主净年化-1.2850%、净Sharpe-0.527740、回撤6.5991%、净pB0.328403、13周期；近期压力净年化-0.0444%、净Sharpe-0.014740、回撤5.7667%、净pB0.526318、14周期。原A近期3.9908%/1.216910、较早1.8371%/0.438043，原A也未达10%/1.5。近期毛2531.80元扣佣金852.48/滑点2254.70后净-575.38元；较早毛已负。双文本主模型较早等于经济去组、近期等于政策去组，未带来两通道共同增量。频率软目标亦未改善。
接受/拒绝理由→接受新增原件和真实不同用途的完整检验；拒绝该已登记配置，提高收益/夏普未实现。回撤低不替代收益/夏普/实际pB；周期微弱正百分比均值不代替资金净财富。全因素12类83项讨论保留，这个11输入子模型不冒称全83项胜率，未知仍保留。历史第一版/独立验证未建立，overfitting_removed=False。
是否重新验证→此配置terminal，不调深度/窗口/文本符号/叶支持/仓位/费用/时期救援。更明确的语义/真实数量预期未经检验，不能自动重开旧族。下一优先为真实新增需求源×量价阶段；官方份额相邻新版本须真实取得，原10/8首窗、10/9第二版及其可用钟尚待输入，未准入新收益模型。原1008日机制前瞻合同与两个候选保持。
目标服务→当前工具实际active已保存；旧R260 blocked三轮作为历史档案。本轮是取得新官方原件并运行不同用途的PROGRESS，研究失败不冒称模型改善；新的无进展审核计数0，不继承旧blocked计数。独立合格0，目标不设complete。
必要验证→3个关键阶段测试；16新账户资金库存/时钟闭合（最大约4.14e-11元），全部20保存日账CAGR/Sharpe/DD重算吻合；全部30原案例/解释等级逐列保持。保留旧文档正文、原tracked Git21处、前瞻十三字段、份额合同及新绝对目标合同；金融五字段按真实新R268更新，旧R256完整五字段先归档。
证据→`{rel(OUT)}/protocol.json`、`summary.json`、`results/全部20账户_四场景同口径比较.parquet`、`央行双文本与量价阶段_完整研究结论.md`、`完整压力账户净值比较.png`、`project_state_update_receipt.json`。

"""
    for path in documents:
        old_text = path.read_text(encoding="utf-8-sig")
        title, separator, rest = old_text.partition("\n")
        path.write_text(title + separator + "\n" + main_block + source_block + rest, encoding="utf-8")
        new_text = path.read_text(encoding="utf-8")
        io.require(new_text.endswith(rest), "旧长期事实正文被改动。")
    new = dict(old)
    new["previous_financial_state_before_TECH_R268"] = {key: old.get(key) for key in FINANCE}
    new["previous_blocked_cycle_before_TECH_R263"] = {key: old.get(key) for key in ("status", "goal_tool_status_confirmed", "blocked_audit_count", "goal_blocked_audit_count", "current_goal_turn_blocker_id", "latest_continuation_audit", "latest_goal_tool_status_receipt", "current_blocked_reentry_condition")}
    new.update(updated_at=io.now(), status="research_active", goal_achieved=False, latest_completed_study=rel(OUT), latest_result=rel(OUT / "summary.json"),
               latest_report=rel(report_path), latest_research_status=summary["status"],
               latest_progress="取得58官方央行报告原件及全部案例对应，实际完成不同双文本阶段16新账户并拒绝；未提高模型收益/Sharpe",
               current_study="510300_PBC_REPORT_PHASE_ACCOUNT_V1", current_phase="FIXED_PBC_TEXT_PHASE_FULL_ACCOUNT_REJECTED",
               current_direction="保留所有因素讨论和真实新输入的阶段研究；双文本变化固定树用途终止，不调失败配置",
               previous_goal_turn_classification="PROGRESS_NEW_INFORMATION_AND_COMPLETE_FIXED_EXPERIMENT",
               latest_goal_turn_classification="PROGRESS_NEW_INFORMATION_AND_COMPLETE_FIXED_EXPERIMENT", blocked_audit_count=0, goal_blocked_audit_count=0,
               goal_tool_status_confirmed="active", current_goal_turn_blocker_id=None, current_blocked_scope=None, current_blocked_reentry_condition=None,
               goal_status="active", latest_goal_service_status="active", latest_goal_tool_status_receipt=rel(OUT / "goal_service_active_at_close.json"),
               goal_status_reconciliation="本轮服务实际active；新58原件、552拟合和16完整账户是实质新用途检验，该固定用途拒绝；目标未实现。旧blocked周期归档，fresh审核0。",
               latest_actual_financial_decision="TECH.R268", latest_actual_financial_result=rel(OUT / "summary.json"),
               latest_actual_financial_status=summary["status"], current_new_strategy_return_sharpe="COMPUTED_R268_FIXED_TEXT_PHASE_ACCOUNTS_REJECTED_NOT_INDEPENDENTLY_VALIDATED",
               latest_actual_financial_primary_four_scene_metrics=summary["primary_four_scene_metrics"],
               new_accounts_in_current_phase=16, latest_pbc_report_text_source=rel(SOURCE / "summary.json"), latest_pbc_text_stage_financial=rel(OUT / "summary.json"),
               latest_pbc_text_stage_signal_status="DEVELOPMENT_ONLY_NOT_LIVE_NO_MODEL_PROMOTION",
               next_research_question="真实新增需求信息（已登记官方份额未来相邻版本）与上涨启动/延续/拥挤阶段能否提供不同机制？当前没有已准入未运行的新收益模型。",
               latest_pbc_text_stage_research_disposition="TERMINAL_FIXED_CONFIGURATION_NOT_TARGET_ACHIEVED")
    for key in FORWARD:
        io.require(new.get(key) == old.get(key), "原目标/前瞻/份额合同字段改变：" + key)
    io.write(STATE, new)
    io.require(git_state() == before_git, "原tracked Git状态改变。")
    io.write(OUT / "project_state_update_receipt.json", {"at": io.now(), "state_updates": 1, "documents": [rel(p) for p in documents],
             "old_bodies_preserved": True, "previous_financial_five_fields_archived": True, "latest_actual_financial_updated_to": "TECH.R268",
             "forward_fields_share_and_absolute_targets_preserved": True, "tracked_git_status_preserved": True,
             "goal_service_actual_status": "active", "fresh_blocked_audit_count": 0, "goal_achieved": False}, exclusive=True)
    io.write(OUT / "verification_receipt.json", {"at": io.now(), "saved_accounts_recomputed": len(verified),
             "maximum_metric_recomputation_error": float(verified.maximum_metric_recomputation_error.max()),
             "new_account_maximum_accounting_error_cny": float(checks.maximum_accounting_error.max()),
             "original_case_values_and_grades_preserved": True, "unknown_text_origins_kept": 120,
             "tests": "tests_receipt.json", "primary_terminal_status": summary["status"], "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False}, exclusive=True)
    print("来源、全部30案例、四场景20账户及失败已归档；长期事实一次更新，目标未实现。", flush=True)


if __name__ == "__main__":
    main()
