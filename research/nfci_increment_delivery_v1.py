"""从保存结果生成本轮NFCI研究结论及图，不重跑模型或账户。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.selected_mix_reappraisal_v1 import read, save, now

STUDY = "510300_SELECTED_MIX_NFCI_INCREMENT_DAILY_V1"
OUT = ROOT / "reports/research/510300_selected_mix_nfci_increment_daily_v1"
SOURCE = ROOT / "reports/research/510300_nfci_graph_vintages_source_v1_3"
INTEGRATED = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
NAMES = {"BASELINE": "原两年日更对照", "NFCI_AGGREGATE": "增加总体金融条件", "NFCI_COMPOSITION": "再增加信用与风险差异"}


def main():
    result, source = read(OUT / "result.json"), read(SOURCE / "result.json")
    primary = result["primary"]
    headline = (f"已取得NFCI三序列的{source['requested_vintages']}个历史版本日，并完成固定策略增量比较。"
                f"预指定主方案压力夏普{primary['sharpe']:.3f}、年化{primary['annual_return']:.2%}、"
                f"最大回撤{abs(primary['max_drawdown']):.2%}；高夏普目标仍未完成。")
    paragraphs = [headline, "",
        "本次按用户新授权恢复公开资料取得。总指标、信用与风险三个序列均来自圣路易斯联储ALFRED官方档案，"
        "使用每个版本中当时可见的最新周值及同版四周前值。三者为美国金融条件，跨市场作用属于待检验假设。", "",
        "经济假设是：整体压力相近时，信用收紧与市场风险上升可能代表不同的后续持有价值。"
        "因此先比较总指标，再只加一项信用与风险变化差，不根据结果更改方向、窗口或阈值。", "",
        "各版本日按芝加哥当地午夜加一自然日后视为可用，合并到中国市场15:05原点，最早次日开盘执行。"
        "保留观察日、版本日和可用时刻。官方历史版本并不证明过去本站HTTP请求何时到达。", "",
        "输入检查发现2023年1月3日至5日共三个交易日的最新官方档案观察超过21天。直接查询对应日期仍返回旧观察，"
        "没有使用后来版本回填。实验在拟合和账户计算前固定：这些原点回退至同一天原价格模型，所有交易日照常计入账户；"
        "所有原训练样本的宏观信息仍完整，没有删训练样本或放宽新鲜度。", "",
        "策略沿用原价格入场与失效规则、三类过程共享训练、最近两年每日更新、两日负预测退出确认、"
        "风险目标先行与10个百分点调仓带。仅在相同成熟样本上增加变量，未改变资本、成本、T+1、现金、分红和尾部预算。", ""]
    for cost in ["BASE", "STRESS"]:
        paragraphs.append("基础费用结果：" if cost == "BASE" else "压力费用结果：")
        paragraphs.append("")
        for row in result["all_accounts"]:
            if row["cost"] == cost:
                paragraphs.append(f"- {NAMES[row['policy']]}：夏普{row['sharpe']:.3f}，年化{row['annual_return']:.2%}，"
                                  f"回撤{abs(row['max_drawdown']):.2%}，期末权益{row['end_equity']:,.2f}元。")
        paragraphs.append("")
    paragraphs.append("预指定主方案的压力费用配对比较：")
    paragraphs.append("")
    for row in result["comparisons"]:
        if row["cost"] == "STRESS" and row["left"] == "NFCI_COMPOSITION":
            paragraphs.append(f"- 相对{NAMES[row['right']]}，算术年化收益差{100*row['annual_arithmetic_difference']:+.3f}个百分点，"
                              f"20日区块95%区间[{100*row['lower_95']:+.3f}, {100*row['upper_95']:+.3f}]个百分点；"
                              f"正增量完整年为{row['positive_complete_years']}。")
    paragraphs.extend(["", "继续资格检验：" + ("本轮固定增量门槛通过，仍需独立验证。" if result['continuation_gate'] else "本轮固定增量门槛未通过，当前用途冻结。"), "",
        f"主方案滚动两年共同数值门槛通过窗口数为{result['rolling_two_year_joint_passes']}。"
        "滚动窗口彼此重叠，不能当成独立验证次数。配对区间尚未校正整个项目反复选择，新的历史信息也不能消除旧价格样本被研究过的影响。", "",
        f"完成{result['training_checks']['new_fits']}次新增回归拟合、四个新增完整账户、两个原对照复现及66个内部依赖账户。"
        "原对照按全部保存账本字段重现，账户现金、分红、权益与事前尾部预算已核对。", "",
        "批量POST下载超时、图表请求标识超时与大批返回列截断的原始档案保留。"
        "最终使用默认Requests客户端标识和每批九列；只有逐列符合请求的版本才准入。网络失败批次按同一请求补取，未改变经济定义。", "",
        "本项NFCI增量实验没有新增独立市场观察，账户行情仍止于2026-09-16。原固定策略的后续日期补充研究另行记录。"
        "当前市场观点为NO_VIEW；没有订单、期权或自动恢复采集任务。"
        "本轮目标仍为20万元全账户、净夏普至少1.2、年化至少10%、回撤不超过10%，以原账户尾部预算约束错误判断。", "",
        "官方说明：[NFCI构造](https://www.chicagofed.org/research/data/nfci/about)、"
        "[ALFRED历史版本](https://alfred.stlouisfed.org/help/downloaddata)。", "",
        "代码：research/selected_mix_nfci_increment_daily_v1.py；固定协议：protocol.json；"
        "完整结果：result.json；逐年结果：results/yearly_metrics.json；逐日账本：accounts。"])
    (OUT / "研究结论.md").write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    colors = {"BASELINE": "#687789", "NFCI_AGGREGATE": "#c18736", "NFCI_COMPOSITION": "#126b78"}
    ledger_check = []
    for policy in NAMES:
        ledger = pd.read_parquet(OUT / "accounts/STRESS" / policy / "ledger.parquet")
        saved = next(x for x in result["all_accounts"] if x["policy"] == policy and x["cost"] == "STRESS")
        np.testing.assert_allclose(ledger.equity.iloc[-1], saved["end_equity"], atol=1e-7, rtol=0)
        np.testing.assert_allclose(ledger.equity, 200000 * (1 + ledger.net_return).cumprod(), atol=1e-6, rtol=0)
        equity = ledger.equity.to_numpy(float)
        peaks = np.maximum.accumulate(np.r_[200000., equity])[1:]
        axes[0].plot(ledger.date, equity / 10000, label=f"{NAMES[policy]}  夏普 {saved['sharpe']:.3f}", color=colors[policy], linewidth=1.6)
        axes[1].plot(ledger.date, 100 * (equity / peaks - 1), color=colors[policy], linewidth=1.2)
        ledger_check.append({"policy": policy, "rows": len(ledger), "saved_equity_recomputed": True})
    axes[0].axhline(20, color="#999999", linewidth=.7, linestyle="--")
    axes[0].set_ylabel("完整账户权益（万元）")
    axes[0].legend(loc="upper left", fontsize=10)
    axes[0].set_title("NFCI新历史版本：同一两年日更策略的固定增量比较", loc="left", fontsize=16, pad=14)
    axes[1].set_ylabel("历史峰值回撤（%）")
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.08, .025, "压力费用｜20万元起始｜2020-01-02 至 2026-09-16｜历史开发比较，不是独立前向验证", fontsize=10, color="#555555")
    fig.tight_layout(rect=[.02, .055, .99, 1])
    fig.savefig(OUT / "NFCI新证据与完整账户.png", dpi=160)
    plt.close(fig)
    progress = {"at": now(), "study_id": STUDY, "source": source, "primary": primary,
                "new_model_fits": result["training_checks"]["new_fits"], "new_final_accounts": 4,
                "reproduced_baseline_accounts": 2, "new_independent_market_observations": 0,
                "goal_achieved": False, "research_state": "USER_RESUMED_RESEARCH_WITH_NEW_PUBLIC_EVIDENCE"}
    save(INTEGRATED / "nfci_new_evidence_progress_20260925.json", progress, True)
    save(OUT / "delivery_checks.json", {"at": now(), "saved_account_recomputation": ledger_check,
        "image": "NFCI新证据与完整账户.png", "image_visual_check": "PENDING", "goal_achieved": False}, True)
    s = read(INTEGRATED / "current_status.json")
    s.update(updated_at=now(), scope="已有本地数据与用户本次授权的新公开证据研究",
             research_execution_state="USER_RESUMED_RESEARCH_NEW_EVIDENCE_EXPERIMENT_COMPLETED",
             latest_goal_turn_classification="PROGRESS_NEW_NFCI_ARCHIVAL_DATA_AND_FIXED_INCREMENT",
             same_condition_consecutive_no_progress_goal_turns=0, active_blocker_id=None, active_blocker_description=None,
             latest_nfci_progress=progress, latest_user_requested_study=OUT.relative_to(ROOT).as_posix(),
             latest_user_requested_study_status=result["status"], goal_achieved=False,
             goal_metadata_note="用户已新授权资料取得，本轮已有实际新证据与实验进展；目标工具上次状态为blocked，未伪造工具恢复或完成。")
    if STUDY not in s["completed_followup_studies"]:
        s["completed_followup_studies"].append(STUDY)
    save(INTEGRATED / "current_status.json", s)
    m = read(ROOT / "config/510300_existing_data_training_mandate_v1.json")
    m.update(current_round=STUDY, latest_integrated_experiment=STUDY,
             current_protocol=(OUT / "protocol.json").relative_to(ROOT).as_posix(),
             latest_progress_receipt=(INTEGRATED / "nfci_new_evidence_progress_20260925.json").relative_to(ROOT).as_posix(),
             last_research_result=headline, research_execution_state=s["research_execution_state"], goal_achieved=False)
    save(ROOT / "config/510300_existing_data_training_mandate_v1.json", m)
    (INTEGRATED / "最新研究结论.md").write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    root_status = ROOT / "RESEARCH_STATUS.md"
    prefix = "<!-- NFCI_NEW_EVIDENCE_20260925 -->\n\n> " + headline + " 本次按用户新授权恢复新资料研究，以下旧受阻记录保留为历史。详见[本轮结论](reports/research/510300_selected_mix_nfci_increment_daily_v1/研究结论.md)。\n\n<!-- END_NFCI_NEW_EVIDENCE_20260925 -->\n\n"
    root_status.write_text(prefix + root_status.read_text(encoding="utf-8"), encoding="utf-8")
    print(headline)


if __name__ == "__main__":
    main()
