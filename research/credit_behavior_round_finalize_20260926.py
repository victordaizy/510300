"""汇总贷款、住户银行行为和银行家审批三组结果，明确重复对照和未达标状态。"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.loan_maturity_composition_daily_v1 as loan
import research.household_bank_balance_daily_v1 as household
import research.banker_loan_approval_daily_v1 as banker
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.strategy_review_diagnostics_v1 import metrics

MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"


def comparison(result, purpose):
    return next(r for r in result["comparisons"] if r["cost"] == "STRESS" and r["purpose"] == purpose)


def run():
    if (banker.OUT / "completed_round.json").exists():
        raise RuntimeError("三组信贷行为研究已完成，不覆盖完成记录。")
    modules = [loan, household, banker]
    results = {m.STUDY: read(m.OUT / "result.json") for m in modules}
    l, h, b = [results[m.STUDY] for m in modules]
    source = read(banker.source.OUT / "result.json")
    seen, duplicates, recomputations = {}, [], []
    total, passed = 0, 0
    for module in modules:
        result = results[module.STUDY]
        for item in result["all_accounts"]:
            total += 1
            passed += bool(item["historical_point_targets_met"])
            path = module.OUT / "accounts" / item["cost"] / item["policy"] / "ledger.parquet"
            frame = pd.read_parquet(path)
            cols = ["idx", "equity", "shares", "cash", "commission", "slippage_cost", "filled_quantity"]
            values = frame[cols]
            key = tuple(pd.util.hash_pandas_object(values, index=False).tolist())
            if key in seen:
                old_path, old_values = seen[key]
                pd.testing.assert_frame_equal(values, old_values, check_exact=True)
                duplicates.append({"original": old_path.relative_to(ROOT).as_posix(), "recomputed": path.relative_to(ROOT).as_posix()})
            else:
                seen[key] = path, values
            if item["cost"] == "STRESS" and item["policy"] == result["primary"]["policy"]:
                verified = metrics(frame)
                for name in ["end_equity", "sharpe", "annual_return", "max_drawdown"]:
                    np.testing.assert_allclose(verified[name], item[name], atol=1e-12, rtol=0)
                gross = float((frame.price_pnl + frame.dividend_recognized).sum())
                costs = float((frame.commission + frame.slippage_cost).sum() + frame.terminal_exit_reserve.iloc[-1])
                np.testing.assert_allclose(gross - costs, item["profit"], atol=1e-6, rtol=0)
                recomputations.append({"study": module.STUDY, "ledger": path.relative_to(ROOT).as_posix(),
                                       "sha256": digest(path), "gross_pnl": gross, "costs": costs, "net_profit": item["profit"]})
    assert total == 36 and len(seen) == 32 and len(duplicates) == 4
    li, hi, bi = comparison(l, "COMPOSITION_INCREMENT"), comparison(h, "HOUSEHOLD_INCREMENT"), comparison(b, "APPROVAL_INCREMENT")
    hp, bp, lp = h["primary"], b["primary"], l["primary"]
    hcontrol = next(r for r in h["all_accounts"] if r["cost"] == "STRESS" and r["policy"] == household.TOTAL + "__FIXED5_NO_ADD")
    bcontrol = next(r for r in b["all_accounts"] if r["cost"] == "STRESS" and r["policy"] == banker.TOTAL + "__FIXED5_NO_ADD")
    household_report = f"""住户银行存贷差额已形成104个月来源，并完成12次完整账户计算；其中4个贷款总量对照与前一组路径完全一致，其余8个为不同账户路径。本组共同目标通过0个，主压力净夏普{hp['sharpe']:.6f}、复合年化{hp['annual_return']:.4%}、最大回撤{abs(hp['max_drawdown']):.4%}。高夏普目标尚未达成。

主变量为100乘（住户存款年初累计净增减住户贷款年初累计净增）除以当期人民币存款余额。直接累计优先，当月值只加此前已公布累计，分字段记录依赖。2022年4月另报的住户存款累计采用直接值；住户贷款总量已知，原期限实验的细分缺失没有被错误继承。统计月和公布时间分开，2023年扩展机构范围前后不混训练。

该差额只是银行存贷行为，不是居民国民账户净储蓄、可用投资现金或股票净流入。[央行2023年一季度金融统计发布会](https://xining.pbc.gov.cn/goutongjiaoliu/113456/113469/2025092212553125171/index.html)也讨论了存款与理财收益及资金配置的关系；把存款变化简化为股市意愿会丢失其他来源。

全部模型共享价格、资金、统计月份及贷款总量；第二对照加入存款总量同比，主方案最后加入住户差额。{h['unique_prediction_days']:,}个日预测原点每天使用{h['training_rows_min']}—{h['training_rows_max']}个成熟训练行，但只有{h['distinct_training_release_months_min']}—{h['distinct_training_release_months_max']}个不同月报。累计依赖也在两年窗口内，固定126近邻，不搜索窗口、方向或部门。

20万元主账户期末{hp['end_equity']:,.2f}元，毛损益{h['primary_cash_attribution']['gross_pnl']:,.2f}元，费用及储备{h['primary_cash_attribution']['cost_and_reserve']:,.2f}元，净损益{hp['profit']:,.2f}元。毛损益已经为负，不能只将失败归因于费用。加入住户指标前，同期限存贷款总量对照夏普{hcontrol['sharpe']:.6f}。

住户指标相对该对照的年化算术差为{hi['daily_blocks']['annual_arithmetic_difference']:.4%}。20日区块95%区间[{hi['daily_blocks']['lower_95']:.4%}, {hi['daily_blocks']['upper_95']:.4%}]；整月报公布间隔区间[{hi['release_blocks']['lower_95']:.4%}, {hi['release_blocks']['upper_95']:.4%}]。2020—2023年为{hi['fixed_periods'][0]['annual_arithmetic_difference']:.4%}，2024年至末端为{hi['fixed_periods'][1]['annual_arithmetic_difference']:.4%}。本用途的历史增量不受支持，不能翻转指标方向或删去时期重新包装。

所有12个账户、源缺失和训练不足时点、年度及滚动两年结果保留。主账户完成66个周期，滚动两年共同通过0次。保存分布、未来及过期标签、现金/T+1/费用/股息和尾部预算检查完成；没有独立前向结果，也没有当前交易资格。
"""
    household_report_path = household.OUT / "本轮研究结论.md"
    household_report_path.write_text(household_report, encoding="utf-8")
    save(household.OUT / "completed_round.json", {"at": now(), "status": h["status"],
        "account_recomputations": 12, "new_distinct_paths_vs_prior_loan_round": 8, "repeated_control_paths": 4,
        "primary": hp, "increment": hi, "result_sha256": digest(household.OUT / "result.json"),
        "report_sha256": digest(household_report_path), "goal_status": "active", "goal_achieved": False}, True)
    old = read(ROOT / "reports/research/510300_selected_mix_reappraisal_v1/result.json")
    old_main = next(r for r in old["historical_metrics"] if r["period"] == "main" and r["cost"] == "STRESS")
    report = f"""本轮已把三个新的宏观与资金行为假设落实为代码和完整账户：贷款期限构成、住户存贷净增加差额、银行家贷款需求之外的审批条件。共完成{total}次账户计算，其中{len(duplicates)}个为重复的匹配对照，实际{len(seen)}条不同账户路径。共同目标通过{passed}个。当前20万元账户的目标仍是压力成本净夏普不低于1.2、复合年化不低于10%、最大回撤不超过10%；高夏普总目标ACTIVE，尚未完成。

先看三个预先指定主账户，全部从2020-01-02连续记账至2026-09-24，包含空仓、信息缺失和训练不足的日子。

贷款期限构成主账户：期末{lp['end_equity']:,.2f}元，净夏普{lp['sharpe']:.6f}，年化{lp['annual_return']:.4%}，最大回撤{abs(lp['max_drawdown']):.4%}。相对仅增加贷款总量的同期限对照，构成信息的年化算术增量仅{li['daily_blocks']['annual_arithmetic_difference']:.4%}；月报区块95%区间[{li['release_blocks']['lower_95']:.4%}, {li['release_blocks']['upper_95']:.4%}]，且2024年至末端增量为负。贷款总量与构成的合并效果比价格资金基础对照好，不能据此把功劳归于期限构成本身。详细来源、102个月可用比率和12账户结论保留在前一阶段文件。

住户存贷差额主账户：期末{hp['end_equity']:,.2f}元，净夏普{hp['sharpe']:.6f}，年化{hp['annual_return']:.4%}，最大回撤{abs(hp['max_drawdown']):.4%}。相对存贷款总量对照的年化算术差为{hi['daily_blocks']['annual_arithmetic_difference']:.4%}，月报区块区间[{hi['release_blocks']['lower_95']:.4%}, {hi['release_blocks']['upper_95']:.4%}]，两个固定时期均负。它在扣费前已亏损，不能靠费用解释制造正期望。

银行家需求与审批主账户：期末{bp['end_equity']:,.2f}元，净夏普{bp['sharpe']:.6f}，年化{bp['annual_return']:.4%}，最大回撤{abs(bp['max_drawdown']):.4%}。毛损益{b['primary_cash_attribution']['gross_pnl']:,.2f}元，费用及储备{b['primary_cash_attribution']['cost_and_reserve']:,.2f}元，净收益{bp['profit']:,.2f}元。仅含需求判断的同期限对照夏普{bcontrol['sharpe']:.6f}。审批条件的年化算术增量{bi['daily_blocks']['annual_arithmetic_difference']:.4%}；20日区块区间[{bi['daily_blocks']['lower_95']:.4%}, {bi['daily_blocks']['upper_95']:.4%}]，季度公布间隔区块区间[{bi['quarter_release_blocks']['lower_95']:.4%}, {bi['quarter_release_blocks']['upper_95']:.4%}]。2020—2023年与2024年至末端的增量分别为{bi['fixed_periods'][0]['annual_arithmetic_difference']:.4%}、{bi['fixed_periods'][1]['annual_arithmetic_difference']:.4%}；两段点估计为正，但不确定区间跨零，远不足以宣布稳定高夏普。

三项信息的含义分别保留，不能串成一个已经得到证明的宏观故事。新增中长期贷款不等于真实民营投资，住户存贷差额不等于资金流入股市，银行审批指数也不是实际批准率。[央行银行家问卷编制说明](https://www.pbc.gov.cn/diaochatongjisi/fileDir/resource/cms/2025/03/2025032117151243875.pdf)将贷款需求和审批条件分别定义为扩散判断，审批反映放松或不变，无法单独识别因果供给冲击。本轮只在需求判断之外增加审批一项，未继续把政策感受或其他行业指数叠进入模。

季度来源由五个官方目录定位2018Q1—2025Q4共32份原报告。首次30份完成；一次SSL连接中断按相同公开地址普通重试成功，一份PDF存在重叠字形及标题文本顺序问题，经原页渲染和原表单元格核对补全。原始PDF和首次问题记录保留，没有修改原文数值。11列和12列附表按真正表头映射；某些年份编制说明少列行业，编号不能当作表格列号。源字段准入完成后，日历连接的微秒/纳秒类型差异以另存补全处理，未重新提取或改写32季数值。

按报告页面所载时间，32季只形成29个不同保守公布日。其中2023Q3/Q4、2024Q3/Q4、2025Q1/Q2各同日发布。[2025年第一季度报告页面](https://www.pbc.gov.cn/diaochatongjisi/116219/116227/5791313/index.html)所载日期为2025-07-25；本研究不把该值回填到3月末。每份报告只取其当季行，附表中后来重列的旧值不回写历史。页面时间与历史最早送达时间仍不是同一认证，全部结果保持开发证据属性。

季度模型共{b['unique_prediction_days']:,}个日预测原点、{b['new_conditional_distributions']:,}个分布。每个训练池有{b['training_rows_min']}—{b['training_rows_max']}个成熟日原点，却只有{b['distinct_training_source_quarters_min']}—{b['distinct_training_source_quarters_max']}个不同季度状态。每日更新可以改变价格输入和成熟样本，不能凭空增加宏观独立样本。季度对照按22个完整公布间隔重采样；该方法仍未保留相邻季度全部持续性，更未消除既有大量试验的选择影响。

本轮模型都在每天重建最近两日历年的共同成熟训练池，相关累计原文或季度公布时钟也在窗口内；固定126近邻、100份、T+1、压力费用、现金股息分账、最多50%目标仓位、五日ES95预算2.5%、标的10%跳空和账户回撤余量仍执行。主固定第五个后续开盘到期，持仓不加仓，风险可提前缩量。现有三条主账户所有滚动两年共同目标通过数均为0。尾部监控没有报警不等于收益优势存在。

{total}个账本的约束检查及三个主账户保存指标、毛损益减费用的核算完成。全部模型分布、训练时点、失败路径、年度及滚动两年结果保存。三组总计新增{sum(r['new_conditional_distributions'] for r in [l, h, b]):,}个条件分布；这不是同等数量的独立样本。重复的4个贷款总量对照已逐值确认并单列，不能算4条新增策略。

关于旧版夏普1.2：现场保存的复现仍为主期压力净夏普{old_main['sharpe']:.6f}、年化{old_main['annual_return']:.4%}、最大回撤{abs(old_main['max_drawdown']):.4%}。原复核承认历史优势可能真实存在；同时308条历史路径筛选及后续选择没有完整校正，最大盈利周期约占42%、前五约85%，且不兼容新的两年日更合同。因此既不能断言它只是过拟合，也不能拿它冒充当前目标已完成。本轮没有恢复旧版实际执行或再调其参数。

在最后市场日2026-09-24，已取得季度来源最新仍为2026-01-23公布的2025Q4，距当日约243天，超出预定120天上限。官方目录及定向检索未取得2026Q1/Q2报告；这是本次未取得，不是证明它们在任何渠道均不存在。current_market_view保持NO_VIEW；独立前向观察仍为0。

本轮三组实验均已结束，没有遗留计算进程。下一待查线索为企业家问卷中的资金周转与销货款回笼定义及原文，先判断它与已失败的PMI新订单用途是否不同，再固定单一可检验假设；当前并未将此线索算成策略结果。不得用翻转指标、修改本轮期限或删除失败时期营救结果。目标继续保持active，没有制作审核包或用户表格，也没有交易授权。
"""
    report_path = banker.OUT / "本轮研究结论.md"
    report_path.write_text(report, encoding="utf-8")
    classification = "PROGRESS_36_ACCOUNT_RECOMPUTATIONS_32_DISTINCT_PATHS_AND_THREE_NEW_MEASUREMENTS"
    status = "COMPLETED_36_ACCOUNTS_32_DISTINCT_PATHS_NO_QUALIFIED_STRATEGY" if not passed else "COMPLETED_36_ACCOUNTS_INDEPENDENT_VALIDATION_REQUIRED"
    receipt = {"at": now(), "status": status, "classification": classification,
        "account_recomputations": total, "distinct_account_paths": len(seen), "repeated_control_paths": duplicates,
        "joint_target_pass_accounts": passed, "monthly_originals_reused": 104, "new_quarterly_originals": 32,
        "effective_quarter_release_days": source["unique_effective_release_days"],
        "new_conditional_distributions": sum(r["new_conditional_distributions"] for r in [l, h, b]),
        "primary_loan": lp, "primary_household": hp, "primary_banker": bp,
        "loan_composition_increment": li, "household_increment": hi, "banker_approval_increment": bi,
        "saved_primary_recomputations": recomputations,
        "result_hashes": {str(m.OUT.relative_to(ROOT) / "result.json"): digest(m.OUT / "result.json") for m in modules},
        "source_result_hash": digest(banker.source.OUT / "result.json"), "report_sha256": digest(report_path),
        "goal_status": "active", "goal_achieved": False, "current_market_view": "NO_VIEW",
        "new_independent_forward_observations": 0, "orders_authorized": False, "review_package_created": False}
    save(banker.OUT / "completed_round.json", receipt, True)
    state = read(MAIN / "current_status.json")
    state.update(updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification=classification, same_condition_consecutive_no_progress_goal_turns=0,
        active_blocker_id=None, active_blocker_description=None,
        latest_user_requested_study=banker.OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=status,
        latest_credit_behavior_round=receipt,
        remaining_research_question="三项信贷行为用途36次账户计算/32条不同路径已完成且未达标。下一项辨认企业家问卷中资金周转与销货款回笼原文、时钟及与旧PMI用途的差异；不翻转或改窗救本轮失败。",
        goal_metadata_note="本轮取得32份新季度原报告，完成三项信息及36次完整账户计算，是实质进展；高夏普目标未完成。")
    for study in [loan.STUDY, household.source.STUDY, household.STUDY,
                  "510300_BANKER_SURVEY_SOURCE_PROBE_V1", "510300_BANKER_SURVEY_SOURCE_COMPLETION_V1",
                  banker.source.STUDY, banker.STUDY]:
        if study not in state["completed_followup_studies"]:
            state["completed_followup_studies"].append(study)
    state["banker_field_admission_completion"] = {"original_stage": "510300_BANKER_SURVEY_FIELD_ADMISSION_V1",
        "status": "COMPLETED_VIA_SAVED_FIELD_AVAILABILITY_COMPLETION", "completion": banker.source.STUDY}
    save(MAIN / "current_status.json", state)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(as_of_date=now()[:10], current_round=banker.STUDY, latest_integrated_experiment=banker.STUDY,
        current_protocol=(banker.OUT / "protocol.json").relative_to(ROOT).as_posix(),
        latest_progress_receipt=(banker.OUT / "completed_round.json").relative_to(ROOT).as_posix(),
        latest_continuation_report=report_path.relative_to(ROOT).as_posix(), latest_continuation_classification=classification,
        research_execution_state=status, last_research_result="三个新用途36次账户计算、32条不同路径，三项目标通过0个；独立前向0，总目标active。",
        goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    marker = "<!-- CREDIT_BEHAVIOR_ROUND_20260926 -->"
    previous = status_path.read_text(encoding="utf-8")
    assert marker not in previous
    status_path.write_text(f"{marker}\n\n> 2026-09-26 贷款期限、住户存贷和银行家审批三个新用途完成36次账户计算，32条不同路径；共同目标通过{passed}个。新增32份季度原报告，目标ACTIVE且未完成。见[本轮结论]({report_path.relative_to(ROOT).as_posix()})。\n\n" + previous, encoding="utf-8")
    assert (MAIN / "最新研究结论.md").read_bytes() == report_path.read_bytes()
    print(f"三组信贷行为实验已完成：{total}次账户计算、{len(seen)}条不同路径、共同目标通过{passed}个。主线保持active。", flush=True)


if __name__ == "__main__":
    run()
