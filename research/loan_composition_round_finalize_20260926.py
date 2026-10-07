"""完成贷款构成研究结论与主线状态，保留所有未通过结果。"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.loan_maturity_composition_daily_v1 as experiment
import research.rmb_loan_composition_completion_v1 as source
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.strategy_review_diagnostics_v1 import metrics

MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"


def run():
    out = experiment.OUT
    if (out / "completed_round.json").exists():
        raise RuntimeError("贷款构成研究已完成，不覆盖完成记录。")
    result, origins = read(out / "result.json"), read(source.OUT / "result.json")
    primary = result["primary"]
    ledger_path = out / "accounts/STRESS" / primary["policy"] / "ledger.parquet"
    verified = metrics(pd.read_parquet(ledger_path))
    for key in ["end_equity", "sharpe", "annual_return", "max_drawdown"]:
        np.testing.assert_allclose(verified[key], primary[key], atol=1e-12, rtol=0)
    passed = sum(bool(r["historical_point_targets_met"]) for r in result["all_accounts"])
    increment = next(r for r in result["comparisons"] if r["cost"] == "STRESS" and r["purpose"] == "COMPOSITION_INCREMENT")
    total = next(r for r in result["comparisons"] if r["cost"] == "STRESS" and r["purpose"] == "TOTAL_AND_COMPOSITION")
    control = next(r for r in result["all_accounts"] if r["cost"] == "STRESS" and r["policy"] == experiment.TOTAL + "__FIXED5_NO_ADD")
    pnl = result["primary_cash_attribution"]
    report = f"""104个月央行原文中的贷款期限构成已提取完成，并完成12个新的510300完整模拟账户。主方案压力成本净夏普{primary['sharpe']:.6f}，复合年化{primary['annual_return']:.4%}，最大回撤{abs(primary['max_drawdown']):.4%}；三项目标同时通过{passed}/12。高夏普总目标保持active，尚未达成。

本轮新增的是企业及住户中长期贷款的部门与期限构成，未重跑旧M1/M2、信用利差或社融政府债规则。[央行金融统计原文示例](https://www.pbc.gov.cn/diaochatongjisi/116219/116225/2972dbe4d53540e5a9d0b5781c71480d/index.html)同时提供贷款总量与部门分项，并说明自2023年起的统计机构范围扩展。研究保留统计月、原文公布时间、保守可用时间和统计版本，不能把统计月末当作可交易时点。

104个月中，有48个月按当月披露、56个月按年初累计披露；同一份原文还可能并列当月与累计数字。先采用原文直接累计；只有当月值时，加此前已经公布的同年累计。每年1月重新起算。2026年的报告同时有社融实体人民币贷款和总人民币贷款，余额只取总贷款章节。已有首版解析的94个成功月份逐字段保持不变，补全代码另存；首版失败记录保留。

2022年4月原文没有住户短期和中长期分项，不能用住房贷款替代中长期贷款。因此4—5月的合并期限比率保留未知，6月直接公布累计后才恢复。最终102个月比率可用。分项净增保留负数，子项和也不强行凑成总量；这类净增比率理论上不保证处于0—100%。

唯一新策略变量为（企业中长期贷款累计净增＋住户中长期贷款累计净增）／人民币贷款累计净增。它可能反映融资期限偏好，但不能单独识别民营投资、居民信心或直接股票买盘。所有信息模型共用统计月份sin/cos，基础对照包含价格与银行资金；第二对照再加贷款余额官方同比；主方案最后加入期限比率。

每日仅用最近两日历年的已成熟五日原点。累计指标所依赖的每份原始月报也必须在当前两年下界以内。2023年统计范围变化前后不混合训练；新范围样本不足时保持NO_VIEW。最终{result['unique_prediction_days']:,}个预测日、{result['new_conditional_distributions']:,}个分布，单日训练{result['training_rows_min']}—{result['training_rows_max']}个成熟原点，但仅有{result['distinct_training_release_months_min']}—{result['distinct_training_release_months_max']}份不同月份报告；反复日更不能增加独立宏观事实数量。

主方案为固定第五个后续开盘到期、持仓期间不加仓。20万元账户从2020-01-02完整记账至2026-09-24，期末{primary['end_equity']:,.2f}元；毛损益{pnl['gross_pnl']:,.2f}元，费用与末端储备{pnl['cost_and_reserve']:,.2f}元，净损益{pnl['net_profit']:,.2f}元。平均权益敞口{primary['mean_exposure']:.2%}，完成63个持仓周期。低回撤主要说明风险暴露有限，不能替代足够的收益。

只含贷款总量的同期限压力对照夏普{control['sharpe']:.6f}，年化{control['annual_return']:.4%}。期限构成的增量年化算术收益差为{increment['daily_blocks']['annual_arithmetic_difference']:.4%}；20日区块95%区间[{increment['daily_blocks']['lower_95']:.4%}, {increment['daily_blocks']['upper_95']:.4%}]，整个月报公布间隔区块区间[{increment['release_blocks']['lower_95']:.4%}, {increment['release_blocks']['upper_95']:.4%}]。2020—2023年的增量为{increment['fixed_periods'][0]['annual_arithmetic_difference']:.4%}，2024年至末端为{increment['fixed_periods'][1]['annual_arithmetic_difference']:.4%}。期限构成自身的稳定增量没有得到支持。

贷款总量与期限合并相对基础价格资金对照的年化算术差为{total['daily_blocks']['annual_arithmetic_difference']:.4%}，20日区块区间[{total['daily_blocks']['lower_95']:.4%}, {total['daily_blocks']['upper_95']:.4%}]。这一开发比较为正，并不等于单独的期限构成有效，更不代表完整账户已达到夏普1.2、年化10%的目标。所有区间未调整大量历史研究造成的选择偏差；逐月报区块也不能保留相邻月的全部持续性。

压力滚动持有对照夏普0.385718、年化1.2018%，同样未达标，不替换预先指定的主方案。主方案全部滚动两年共同目标通过0次。253个固定五日相位的成熟尾部观测中，有194个可计算监控窗口，尾部报警0次；没有报警不证明收益规律有效。

保存分布重算、未来及过期标签扰动、共同训练池、统计范围和累计原文依赖检查完成。12个完整账户保留现金、T+1、100份、压力成本、股息和尾部预算，主保存账本的净值及指标重算一致。固定协议之后、股票收益计算之前的一次索引实现修正另有execution_freeze记录，未改变模型条件、样本门槛或费用。原冻结文件不覆盖。

七天政策利率、操作规模以及IF基差用途已定位到既有研究，本轮不把重复线索重新计作实验。下一项处理同一批原文中的住户存款与贷款净增加差额，作为居民银行存贷行为的单一新候选；它不是居民净储蓄或流向股市的直接统计量。当前未完成独立前向验证，current_market_view仍为NO_VIEW，没有实际交易资格。未制作审核包或用户表格。
"""
    path = out / "本轮研究结论.md"
    path.write_text(report, encoding="utf-8")
    receipt = {"at": now(), "status": "COMPLETED_12_LOAN_COMPOSITION_ACCOUNTS_NO_QUALIFIED_STRATEGY",
               "classification": "PROGRESS_104_MONTHLY_SOURCES_AND_12_LOAN_COMPOSITION_ACCOUNTS",
               "new_full_accounts": 12, "joint_target_pass_accounts": passed,
               "source_months": origins["source_months"], "joint_ratio_months": origins["joint_ratio_months"],
               "new_conditional_distributions": result["new_conditional_distributions"], "primary": primary,
               "composition_increment": increment, "loan_total_and_composition_increment": total,
               "saved_primary_metrics_recomputed": {"path": ledger_path.relative_to(ROOT).as_posix(), "sha256": digest(ledger_path)},
               "result_sha256": digest(out / "result.json"), "report_sha256": digest(path),
               "goal_status": "active", "goal_achieved": False, "current_market_view": "NO_VIEW",
               "new_independent_forward_observations": 0, "orders_authorized": False, "review_package_created": False}
    save(out / "completed_round.json", receipt, True)
    state = read(MAIN / "current_status.json")
    state.update(updated_at=now(), goal_status="active", goal_achieved=False,
                 latest_goal_turn_classification=receipt["classification"], same_condition_consecutive_no_progress_goal_turns=0,
                 active_blocker_id=None, active_blocker_description=None, latest_loan_composition_round=receipt,
                 latest_user_requested_study=out.relative_to(ROOT).as_posix(), latest_user_requested_study_status=receipt["status"],
                 remaining_research_question="贷款期限构成12账户已完成且未达标；继续提取住户存贷款净增加差额并检验相对贷款和存款总量的增量。",
                 goal_metadata_note="本轮完成104个月新来源构成和12个账户，是实质研究进展；高夏普总目标未完成。")
    for study in ["510300_RMB_LOAN_COMPOSITION_SOURCE_V1", source.STUDY, experiment.STUDY]:
        if study not in state["completed_followup_studies"]:
            state["completed_followup_studies"].append(study)
    save(MAIN / "current_status.json", state)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(as_of_date=now()[:10], current_round=experiment.STUDY, latest_integrated_experiment=experiment.STUDY,
                   current_protocol=(out / "protocol.json").relative_to(ROOT).as_posix(),
                   latest_progress_receipt=(out / "completed_round.json").relative_to(ROOT).as_posix(),
                   latest_continuation_report=path.relative_to(ROOT).as_posix(), latest_continuation_classification=receipt["classification"],
                   research_execution_state=receipt["status"], last_research_result="104个月贷款构成、12个账户已完成；共同目标通过0个，目标active。",
                   goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    marker = "<!-- LOAN_COMPOSITION_ROUND_20260926 -->"
    previous = status_path.read_text(encoding="utf-8")
    assert marker not in previous
    status_path.write_text(f"{marker}\n\n> 2026-09-26 贷款构成104个月来源、12个新完整账户已完成，共同目标通过0个。主压力夏普{primary['sharpe']:.6f}，年化{primary['annual_return']:.4%}；总目标ACTIVE且未完成。见[本轮结论]({path.relative_to(ROOT).as_posix()})。\n\n" + previous, encoding="utf-8")
    print("贷款构成12个账户已正式完成，主线仍为active；没有达标策略。", flush=True)


if __name__ == "__main__":
    run()
