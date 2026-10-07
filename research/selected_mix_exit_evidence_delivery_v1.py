"""整理学习退出证据、固定关闭对照和早期反例，保留不同区间的完整结果。"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.selected_mix_support_envelope_daily_v1 as support
import research.selected_mix_exit_usage_factorial_v1 as usage
import research.selected_mix_support_earlier_diagnostic_v1 as earlier
from research.selected_mix_reappraisal_v1 import read, save, digest, now

SUMMARY = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
NAMES = {"W1_R1": "两种学习退出都保留", "W0_R1": "只保留普通岭模型退出", "W1_R0": "只保留周期内模型退出",
         "W0_R0": "关闭两种学习退出", "D60_SUPPORT_ENVELOPE_RISK_BAND10": "限制单项外推",
         "UNCHANGED_DAILY": "原两年日更规则", "SUPPORT_ENVELOPE": "限制单项外推"}


def metrics_text(metric):
    return f"压力夏普{metric['sharpe']:.3f}、复合年化{metric['annual_return']:.2%}、最大回撤{abs(metric['max_drawdown']):.2%}"


def pair_text(row):
    return f"算术年化收益差{row['annual_arithmetic_difference']*100:+.3f}个百分点，95%区间[{row['lower_95']*100:+.3f}, {row['upper_95']*100:+.3f}]个百分点"


def reports(s, u, e):
    common = ["", "完整账户按20万元、242个交易日年化，现金收益0；含全部空仓日、费用、滑点、分红和T+1。目标仓位上限50%，五日ES95预算2.5%，假设标的10%跳空损失预算至多权益5%且不超过距90%峰值余量的一半。风险投影后使用10个百分点调仓带，明确退出和风险超限优先。", "",
              "尾部预算不是最大回撤保证。期末采用正常收盘估值，保留实际未平仓份额，不伪造终点卖出。全部历史均曾被研究，新增独立观察为0；所有抽样区间未作完整研究家族选择校正。", "",
              "本轮仅本地资料和510300/现金研究，当前市场判断NO_VIEW，没有新行情采集或订单。固定协议、来源、控制器决定、账本和结果分别保存在同目录。"]
    rows = [f"限制单项外推的实验已完成，{metrics_text(s['primary'])}。此前父规则为压力夏普0.954、年化3.62%、回撤5.06%，但本轮仍未达到全部目标。", "",
            "原八项特征、系数和最近两年训练成员保持。每个决策日分别取同类D60成熟样本的最小值和最大值，实际状态任一项越界就不使用当天学习退出，并清零连续负值计数；原价格、止损及最长持有规则继续执行。全部边界每天随当日训练成员更新。", "",
            "各单项范围内不等于联合状态可靠，更不等于置信区间。这是固定外推限制，不是识别真假跌破或证明市场心理的办法。", "",
            f"在2020年1月2日至2026年9月16日，{pair_text(next(row for row in s['comparisons'] if row['cost']=='STRESS'))}。正的点估计不足以确认稳定增量。", "",
            f"同一规则在预先固定的2015—2019年反例检查中只有{metrics_text(e['primary'])}；{pair_text(e['early_period_increment'])}。两个时期的增量点估计方向相反，两边区间均跨零，故不能晋升为稳定改进。", "",
            "实际保存2,634个每日支持范围，复用已保存回归系数，新增2个末端账户和22个内部依赖账户；没有重新挑选特征、分位阈值或系数。", *common]
    (support.OUT / "研究结论.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
    rows = ["完整系统内分别关闭两种学习退出的固定比较已完成。直接关闭学习退出未改善这次完整账户，全部其他入场、价格退出、训练支持路由、风险估计与组合模块保持。", "",
            "2020年1月2日至2026年9月16日的四个压力结果：", ""]
    for row in u["all_accounts"]:
        if row["cost"] == "STRESS":
            rows.append(f"- {NAMES[row['policy']]}：{metrics_text(row)}。")
    rows += ["", "关闭学习退出不等于删掉全部模型：原成熟资格仍作为支持路由输入，原每日风险分配也保留。这样只检验学习退出的请求，不把路由变化混入作用。对照中也没有应用单项外推限制。", "",
             "固定压力收益分解：", ""]
    contrast_names = {"WITHIN_ON_NO_RIDGE": "不使用普通岭退出时，增加周期内退出",
                      "RIDGE_ON_NO_WITHIN": "不使用周期内退出时，增加普通岭退出",
                      "INTERACTION": "两者的交互项", "BOTH_LEARNING_VS_NEITHER": "两种都用相对两种都不用"}
    for row in u["contrasts"]:
        if row["cost"] == "STRESS":
            rows.append(f"- {contrast_names[row['contrast']]}：{pair_text(row)}。")
    rows += ["", "交互区间的下界为正，但它来自未经多重选择校正的开发比较。退出会改变后续参考风险和真实账户路径，不能把交互项解释成两项独立收益简单相加，也不能据此认定稳定的未来优势。", "",
             "新增6个末端账户和66个内部依赖账户，复用两种都保留的2个父账户，无新增系数拟合。预先指定的主对照是两种都关闭，没有从四格中事后重选主候选。", *common]
    (usage.OUT / "研究结论.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
    baseline = next(row for row in e["all_accounts"] if row["policy"] == "UNCHANGED_DAILY" and row["cost"] == "STRESS")
    rows = [f"固定2015年1月5日至2019年12月31日反例检查已完成。原规则{metrics_text(baseline)}；限制单项外推后{metrics_text(e['primary'])}。", "",
            f"本段{pair_text(e['early_period_increment'])}，与2020年后{pair_text(e['main_period_increment'])}分开保留。增量点估计方向相反，没有确认跨时期稳定改善。", "",
            "每个末端账户在2015年1月5日以20万元现金开始，原2013参考节点保留原起点，其余主账户节点改到本诊断起点。两年模型、支持范围和五日风险分布只读取当日成熟历史，不能借用后来年份的估计。没有修改系数或外推规则。", "",
            "该段历史已用于旧研究，不是独立留出。它的作用是主动寻找反例，不能与后段拼接，也不能挑选其中有利年份。期末改用与新主研究一致的正常收盘计价，因此不与旧人工开盘平仓的旧诊断直接混排。", "",
            f"新增4个末端账户和44个内部依赖账户，核对{e['verification']['model_records_checked_in_fixed_period']}条该区间合格模型记录及{e['verification']['risk_forecasts_checked']}个历史风险原点，无新增系数拟合。", *common]
    (earlier.OUT / "研究结论.md").write_text("\n".join(rows)+"\n", encoding="utf-8")


def figure(s, u, e):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei"], "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.8), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    source_rows = []
    groups = [
        [(usage.OUT, "W1_R1", "原两年日更", "#6B7A8B"),
         (support.OUT, support.PRIMARY, "限制单项外推", "#187C86"),
         (usage.OUT, "W0_R0", "关闭两种学习退出", "#B28250")],
        [(earlier.OUT, "UNCHANGED_DAILY", "原两年日更", "#6B7A8B"),
         (earlier.OUT, "SUPPORT_ENVELOPE", "限制单项外推", "#187C86")]]
    for col, group in enumerate(groups):
        for folder, policy, label, color in group:
            result = {support.OUT: s, usage.OUT: u, earlier.OUT: e}[folder]
            metric = next(row for row in result["all_accounts"] if row["policy"] == policy and row["cost"] == "STRESS")
            path = folder / "accounts/STRESS" / policy / "ledger.parquet"
            ledger = pd.read_parquet(path)
            nav = np.r_[200000., ledger.equity.to_numpy(float)]
            dd = (nav / np.maximum.accumulate(nav)-1)[1:]
            axes[0, col].plot(ledger.date, ledger.equity/10000, color=color, lw=1.5,
                label=f"{label}｜夏普{metric['sharpe']:.3f}，年化{metric['annual_return']:.2%}")
            axes[1, col].plot(ledger.date, dd, color=color, lw=1.2)
            source_rows.append({"path": path.relative_to(ROOT).as_posix(), "sha256": digest(path)})
        axes[0, col].set_title(["2020—2026：主研究区间", "2015—2019：固定反例区间"][col], loc="left", fontsize=12, pad=12)
        axes[0, col].legend(frameon=False, fontsize=8.8, loc="upper left")
        axes[0, col].margins(y=.35)
        axes[0, col].set_ylabel("账户权益（万元）")
        axes[1, col].set_ylabel("历史峰值回撤")
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        axes[1, col].xaxis.set_major_locator(mdates.YearLocator(2 if col == 0 else 1))
        axes[1, col].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.2)
    fig.suptitle("学习退出是否有足够证据：保留改善，也保留反例", x=.075, y=.97, ha="left", fontsize=17, weight="bold")
    fig.text(.075, .923, "固定规则跨两个既定时期分别运行；左、右账户各自从20万元开始，没有拼接或挑选年份。", fontsize=11, color="#56616C")
    fig.text(.075, .052, "压力成本，242日年化，包含空仓、T+1、费用和分红；末日正常收盘计价。", fontsize=9)
    fig.text(.075, .029, "两个区间均曾被研究；结果不构成独立前向验证。", fontsize=9, color="#56616C")
    fig.subplots_adjust(left=.075, right=.975, top=.855, bottom=.125, wspace=.2, hspace=.13)
    path = earlier.OUT / "学习退出证据与早期反例.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    save(earlier.OUT / "figure_sources.json", {"figure": path.name, "sources": source_rows})


def verify_saved():
    records = []
    for module in [support, usage, earlier]:
        module.verify_sources(module.OUT)
        result = read(module.OUT / "result.json")
        for metric in result["all_accounts"]:
            path = module.OUT / "accounts" / metric["cost"] / metric["policy"] / "ledger.parquet"
            ledger = pd.read_parquet(path)
            nav = np.r_[200000., ledger.equity.to_numpy(float)]
            returns = nav[1:] / nav[:-1] - 1
            np.testing.assert_allclose(returns, ledger.net_return, atol=1e-12, rtol=0)
            np.testing.assert_allclose([returns.mean() / returns.std(ddof=1) * np.sqrt(242),
                (nav[-1]/nav[0])**(242/len(returns))-1, (nav/np.maximum.accumulate(nav)-1).min()],
                [metric["sharpe"], metric["annual_return"], metric["max_drawdown"]], atol=1e-10, rtol=0)
            np.testing.assert_allclose(ledger.cash+ledger.shares*ledger.mark+ledger.dividend_receivable, ledger.equity, atol=1e-7, rtol=0)
            records.append({"study": module.STUDY, "policy": metric["policy"], "cost": metric["cost"],
                            "rows": len(ledger), "ledger_sha256": digest(path)})
    save(SUMMARY / "exit_evidence_saved_check_20260925.json", {"at": now(), "status": "PASS_SAVED_ACCOUNT_METRICS",
        "accounts": len(records), "daily_rows": sum(row["rows"] for row in records), "checks": records,
        "goal_achieved": False})


def main():
    s, u, e = [read(module.OUT / "result.json") for module in [support, usage, earlier]]
    verify_saved()
    reports(s, u, e)
    figure(s, u, e)
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    study_ids = [support.STUDY, usage.STUDY, earlier.STUDY]
    if mandate.get("current_round") not in study_ids:
        print("本轮历史结果已整理，已有后续研究，未回退最新进度。", flush=True)
        return
    baseline_early = next(row for row in e["all_accounts"] if row["policy"] == "UNCHANGED_DAILY" and row["cost"] == "STRESS")
    message = (f"学习退出支持范围、两种退出使用分解、2015-2019反例检查已完成。限制外推在主区间{metrics_text(s['primary'])}，"
               f"在早期区间夏普由{baseline_early['sharpe']:.3f}降到{e['primary']['sharpe']:.3f}，未确认跨时期稳定增量。高夏普目标仍未完成，保持进行中。")
    progress = {"at": now(), "study_ids": study_ids, "economic_comparisons": 3,
        "new_full_accounts": 12, "reused_full_accounts": 2, "new_internal_accounts": 132,
        "new_coefficient_fits": 0, "new_reference_accounts": 0, "daily_support_envelopes": 2634,
        "main_support_primary": s["primary"], "exit_usage_primary": u["primary"], "earlier_support_primary": e["primary"],
        "cross_period_increment_consistency_established": False, "goal_workflow": "ACTIVE_RESEARCH", "goal_achieved": False,
        "new_independent_observations": 0, "current_market_view": "NO_VIEW", "orders_authorized": False}
    progress_path = SUMMARY / "exit_evidence_progress_20260925.json"
    save(progress_path, progress)
    rows = [message, "",
            "本轮把三个问题分开：预测越界时应否使用、学习退出整体有无作用、同一限制能否经受另一段历史的反例。结果支持继续保留完整对照，不支持把近期改善直接升级为已验证策略。", "",
            f"- 主区间限制单项外推：{metrics_text(s['primary'])}；父规则为0.954 / 3.62% / 5.06%。",
            "- 退出使用分解：压力夏普依次为都保留0.954、只保留普通岭0.557、只保留周期内0.708、都关闭0.544。直接关闭学习退出未改善。",
            f"- 固定早期区间：原规则{metrics_text(baseline_early)}；限制外推后{metrics_text(e['primary'])}。", "",
            "学习退出限制的主区间为2020年1月2日至2026年9月16日；反例区间为2015年1月5日至2019年12月31日，分别按20万元和242日年化结算。两个时期不拼接、不混排，也不冒称独立留出。", "",
            f"主区间{pair_text(e['main_period_increment'])}；早期区间{pair_text(e['early_period_increment'])}。增量方向相反，两边区间均跨零。", "",
            "两种学习退出的交互项在本次未校正的开发抽样中下界为正，但两种一起使用相对都不用的总体增量区间跨零。需要避免把交互和整体优势混为一谈。", "",
            "三个实验新增12个末端账户、132个内部依赖账户，复用2个父账户；没有新拟合回归系数。各项范围边界有2,634个逐日记录，它们不是2,634次独立机会。已核对14条保存账户的净值身份和绩效。", "",
            "原版历史1.219 / 10.29% / 6.51%继续保留，其原规则例外、历史选择影响和独立验证不足不变。此前宏观持续状态和具体策略价值分配的全部结果亦保留，本轮不把宏观失败信息强行加回账户。", "",
            "完整来源：", "",
            "- reports/research/510300_selected_mix_support_envelope_daily_v1/研究结论.md",
            "- reports/research/510300_selected_mix_exit_usage_factorial_v1/研究结论.md",
            "- reports/research/510300_selected_mix_support_earlier_diagnostic_v1/研究结论.md", "",
            "运行与查询入口：scripts/run_510300_exit_evidence_research.ps1。已完成步骤复用保存结果。", "",
            "验收仍为净夏普至少1.2、年化至少10%、最大回撤不超过10%，按最近两年训练、每天更新及当前尾部预算。当前没有同时达到这些要求且获得独立验证的策略。已有数据研究继续，未恢复行情采集或交易。"]
    (SUMMARY / "最新研究结论.md").write_text("\n".join(rows)+"\n", encoding="utf-8")
    current_path = SUMMARY / "current_status.json"
    current = read(current_path)
    completed = current.get("completed_followup_studies", [])
    for study_id in study_ids:
        if study_id not in completed:
            completed.append(study_id)
    current.update(updated_at=now(), goal_status="active", goal_achieved=False, completed_followup_studies=completed,
        latest_user_requested_study=earlier.OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=e["status"],
        latest_goal_turn_classification="PROGRESS_SUPPORT_EXIT_USAGE_AND_EARLIER_COUNTEREVIDENCE_COMPLETED",
        latest_exit_evidence_progress=progress, same_condition_consecutive_no_progress_goal_turns=0,
        active_blocker_id=None, active_blocker_description=None, report="最新研究结论.md",
        remaining_research_question="单项支持限制的收益增量跨时期方向不一致；尚需未用于选择的信息证明可执行收益来源，不能靠继续调整旧失败阈值替代。")
    save(current_path, current)
    mandate.update(last_research_result=message, goal_achieved=False, latest_progress_receipt=progress_path.relative_to(ROOT).as_posix())
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    old = status_path.read_text(encoding="utf-8")
    start_marker, end_marker = "<!-- EXIT_EVIDENCE_PROGRESS_20260925 -->", "<!-- END_EXIT_EVIDENCE_PROGRESS_20260925 -->"
    if start_marker in old and end_marker in old:
        first, last = old.index(start_marker), old.index(end_marker)+len(end_marker)
        old = old[:first]+old[last:].lstrip("\n")
    status_path.write_text(start_marker+"\n\n> 2026-09-25 "+message+" 详见[最新研究结论](reports/research/510300_integrated_research_continuation_20260924/最新研究结论.md)。\n\n"+end_marker+"\n\n"+old, encoding="utf-8")
    print("三项学习退出证据已保存，反例保留，目标仍进行中且未完成。", flush=True)


if __name__ == "__main__":
    main()
