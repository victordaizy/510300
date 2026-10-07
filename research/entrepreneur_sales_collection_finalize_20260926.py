"""登记企业收款增量的实际结果，区分新增账户与已匹配的复用对照。"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.entrepreneur_sales_collection_daily_v1 as study
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.strategy_review_diagnostics_v1 import metrics

MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"


def run():
    if (study.OUT / "completed_round.json").exists():
        raise RuntimeError("企业收款研究已完成，不能覆盖完成记录。")
    result = read(study.OUT / "result.json")
    source = read(study.source.OUT / "result.json")
    main = result["primary"]
    increment = next(r for r in result["comparisons"] if r["cost"] == "STRESS" and r["purpose"] == "SALES_COLLECTION_INCREMENT")
    combined = next(r for r in result["comparisons"] if r["cost"] == "STRESS" and r["purpose"] == "BANK_AND_COLLECTION_INCREMENT")
    control = next(r for r in result["all_accounts"] if r["cost"] == "STRESS" and r["policy"] == study.BANK + "__FIXED5_NO_ADD")
    ledger_path = study.OUT / "accounts/STRESS" / study.PRIMARY / "ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    recomputed = metrics(ledger)
    for key in ["sharpe", "annual_return", "max_drawdown", "end_equity", "profit"]:
        np.testing.assert_allclose(recomputed[key], main[key], atol=1e-12, rtol=0)
    original_hashes = {}
    path_columns = ["idx", "equity", "shares", "cash", "commission", "slippage_cost", "filled_quantity"]
    for account in result["reused_control_accounts"]:
        original = ROOT / account["source"] / "ledger.parquet"
        original_hashes[tuple(pd.util.hash_pandas_object(pd.read_parquet(original)[path_columns], index=False))] = original
    new_distinct, duplicate_new = {}, []
    for account in result["all_accounts"]:
        if not account["policy"].startswith(study.SIGNAL):
            continue
        path = study.OUT / "accounts" / account["cost"] / account["policy"] / "ledger.parquet"
        values = pd.read_parquet(path)[path_columns]
        key = tuple(pd.util.hash_pandas_object(values, index=False))
        if key in original_hashes or key in new_distinct:
            prior_path = original_hashes.get(key, new_distinct.get(key))
            pd.testing.assert_frame_equal(values, pd.read_parquet(prior_path)[path_columns], check_exact=True)
            duplicate_new.append({"path": path.relative_to(ROOT).as_posix(), "same_as": prior_path.relative_to(ROOT).as_posix()})
        else:
            new_distinct[key] = path
    assert len(result["reused_control_accounts"]) == 8 and len(new_distinct) == 4 and not duplicate_new
    primary_cycles = next(r for r in result["cycle_counts"] if r["policy"] == study.PRIMARY and r["cost"] == "STRESS")
    full_error = next(r for r in result["forecast_evaluation"] if r["subset"] == "ALL")
    decline_error = next(r for r in result["forecast_evaluation"] if r["subset"] == "PRIOR_FIVE_DAY_DECLINE")
    attribution = result["primary_cash_attribution"]
    report = f"""企业销货款回笼这一项新信息已经完成真实回测：新增4个完整账户，另有8个对照在逐值确认输入及预测一致后复用。全部12个比较账户均未达到净夏普1.2、年化收益10%、最大回撤10%的共同目标。主压力账户夏普{main['sharpe']:.6f}、复合年化{main['annual_return']:.4%}、最大回撤{abs(main['max_drawdown']):.4%}。总目标仍为active，尚未找到经过验证的稳定高夏普策略。

本次取得2018Q1—2025Q4共32份企业家原报告，64次定向公开请求全部保存，形成29个不同保守公布日。2018Q2与2020Q4的标题因PDF文本顺序问题未连续提取，经原页渲染确认；指标按真实表头定位，正文、当季附表行和同名单元格三处一致。早期10列、后期9列的变化来自删除企业家信心列，收款与周转字段没有错位。后续附表重列的历史数值没有回填。

来源是全国5000多户工业企业调查。[央行2025年第四季度企业家问卷原报告](https://www.pbc.gov.cn/diaochatongjisi/116219/116227/2026012320114154002/2026012320111834526.pdf)把销货款回笼指数定义为状况良好的比例加0.5乘一般比例。它反映企业的收款判断，不是经营现金流金额、应收账款周转天数，也不是沪深300成分股资金净流量。资金周转指标仅用于原文解释，本次没有加入模型或用于事后替换。

待检验逻辑是：银行审批宽松并不自动意味着企业回款顺畅，收款状况可能帮助区别一次价格压力是暂时现金需求还是更持久的经营问题。这只是机制假设。模型在价格、资金、统计季度之外先纳入银行贷款需求与审批，再增加唯一的企业收款指标；同池价格资金对照仍保留，不能把银行信息当作已经证明有效的基础。

银行与企业两个来源按各自页面公布日末入模，同一统计季度、均在120自然日内才可用。本次32季的保守公布日实际完全一致，因此8个对照无需重新运行账户。3,312条控制分布逐值复算一致后，直接复用原账本；只有企业收款产生1,104条新的条件分布和4条新的账户路径。

每天重新使用最近两日历年的成熟原点，相关季度原文的公布时钟也在窗口内。共有{result['unique_prediction_days']:,}个预测日，每个训练池{result['training_rows_min']}—{result['training_rows_max']}个日原点，但只有{result['distinct_training_source_quarters_min']}—{result['distinct_training_source_quarters_max']}个不同季度事实。保留{result['schedule_statuses']['NO_VIEW_SOURCE']}个源信息未知日和{result['schedule_statuses']['NO_VIEW_TRAINING']}个训练不足日。没有搜索近邻数量、期限、方向或窗口。

20万元主账户从2020-01-02连续记账至2026-09-24，期末{main['end_equity']:,.2f}元，净盈利{main['profit']:,.2f}元；毛损益{attribution['gross_pnl']:,.2f}元，费用与退出储备{attribution['cost_and_reserve']:,.2f}元。共{primary_cycles['closed']}个完成周期，平均敞口{main['mean_exposure']:.4%}。T+1、100份、现金股息、最多50%目标仓位、五日ES95预算2.5%、标的10%跳空与回撤余量约束均执行。主方案固定第五个后续开盘到期，持仓不加仓；同信息的滚动期限对照也全部保留。

相对只含银行需求与审批的同期限压力对照，夏普从{control['sharpe']:.6f}增加到{main['sharpe']:.6f}。收款信息的全期年化算术收益增量为{increment['daily_blocks']['annual_arithmetic_difference']:.4%}，20日区块95%区间[{increment['daily_blocks']['lower_95']:.4%}, {increment['daily_blocks']['upper_95']:.4%}]；季度公布间隔区块为[{increment['quarter_release_blocks']['lower_95']:.4%}, {increment['quarter_release_blocks']['upper_95']:.4%}]。两个区间都跨零。2020—2023年增量{increment['fixed_periods'][0]['annual_arithmetic_difference']:.4%}，2024年至末端为{increment['fixed_periods'][1]['annual_arithmetic_difference']:.4%}，后段并未继续改善。

银行与收款合并信息相对价格资金对照的年化算术增量为{combined['daily_blocks']['annual_arithmetic_difference']:.4%}。20日区块下界虽为正，整季公布间隔区块下界仍为{combined['quarter_release_blocks']['lower_95']:.4%}，不能只挑较窄的区间宣称稳定。季度重采样仅22个公布间隔，未保留全部相邻季度持续性，也未解决大量历史研究后的选择偏差。

全样本预测均方误差相对银行对照改善{full_error['primary_mse_improvement'][study.BANK]:.4%}，在实际允许新增的五日下跌子样本仅改善{decline_error['primary_mse_improvement'][study.BANK]:.4%}；相对同池无条件历史分布反而更差。因此账户点值变好，不等于预测整体可靠性已有实质提升。主账户滚动两年共同目标通过0次；尾部监控218个成熟非重叠原点中159个可评估、0个警报，也不能代替收益优势。

这轮保留了一条有机制但尚不稳定的历史增量。没有将0.410包装成1.2，没有通过增加仓位、删去近期负增量或更换未检验指标救结果。所有4个新账户和8个复用对照、年度/滚动结果及相同样本比较均已保存。

最后市场日已取得来源仍为2026-01-23公布的2025Q4，距当时约243天，超出120天上限。current_market_view为NO_VIEW，新增独立前向交易为0。此处没有当前可执行信号，也没有恢复任何实际下单。

账户计算已结束。下一条正在核查的证据是国家统计局工业企业应收账款平均回收期：它来自财务汇总，与主观问卷不同，但2018年前后收入分母口径及同比可比范围必须先厘清，不能直接拼接。该线索尚未计入本轮策略收益。总目标保持active，没有制作审核包或用户表格。
"""
    report_path = study.OUT / "本轮研究结论.md"
    report_path.write_text(report, encoding="utf-8")
    classification = "PROGRESS_32_NEW_CASH_REPORTS_FOUR_NEW_ACCOUNTS_EIGHT_REUSED_CONTROLS"
    execution = "COMPLETED_FOUR_NEW_SALES_COLLECTION_ACCOUNTS_NO_QUALIFIED_STRATEGY"
    receipt = {"at": now(), "study_id": study.STUDY, "status": execution, "classification": classification,
        "new_source_reports": 32, "effective_source_days": 29, "new_accounts": 4, "new_distinct_account_paths": 4,
        "reused_matched_accounts": 8, "new_conditional_distributions": result["new_conditional_distributions"],
        "recomputed_control_distributions": result["recomputed_control_distributions"],
        "joint_target_pass_accounts": 0, "primary": main, "sales_collection_increment": increment,
        "source_result_sha256": digest(study.source.OUT / "result.json"), "result_sha256": digest(study.OUT / "result.json"),
        "primary_ledger_sha256": digest(ledger_path), "report_sha256": digest(report_path),
        "goal_status": "active", "goal_achieved": False, "current_market_view": "NO_VIEW",
        "new_independent_forward_observations": 0, "orders_authorized": False, "review_package_created": False}
    save(study.OUT / "completed_round.json", receipt, True)
    state = read(MAIN / "current_status.json")
    state.update(updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification=classification, same_condition_consecutive_no_progress_goal_turns=0,
        active_blocker_id=None, active_blocker_description=None,
        latest_user_requested_study=study.OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=execution,
        latest_entrepreneur_cash_round=receipt,
        remaining_research_question="企业收款4个新账户与8个同池对照已完成；全期增量正、近段略负，目标未达。下一条核查工业企业应收账款回收期财务指标的原文、收入分母和同比口径，不修改本轮失败规则。",
        goal_metadata_note="新增32份企业家原报告并运行4个不同账户，属于实质进展；账户改善尚未通过跨期增量和高夏普共同目标。")
    for name in [study.source.source.STUDY, study.source.STUDY, study.STUDY]:
        if name not in state["completed_followup_studies"]:
            state["completed_followup_studies"].append(name)
    save(MAIN / "current_status.json", state)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(as_of_date=now()[:10], current_round=study.STUDY, latest_integrated_experiment=study.STUDY,
        current_protocol=(study.OUT / "protocol.json").relative_to(ROOT).as_posix(),
        latest_progress_receipt=(study.OUT / "completed_round.json").relative_to(ROOT).as_posix(),
        latest_continuation_report=report_path.relative_to(ROOT).as_posix(), latest_continuation_classification=classification,
        research_execution_state=execution,
        last_research_result="企业收款完成4个新账户并复用8个匹配对照；主压力夏普0.410254、年化1.0306%、回撤4.3000%，目标未达，独立前向0。",
        goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    marker = "<!-- ENTREPRENEUR_CASH_ROUND_20260926 -->"
    previous = status_path.read_text(encoding="utf-8")
    assert marker not in previous
    status_path.write_text(f"{marker}\n\n> 2026-09-26 企业收款完成32份原报告、4个新账户及8个复用对照；主压力净夏普0.410254、年化1.0306%、回撤4.3000%，共同目标通过0个，总目标ACTIVE。见[本轮结论]({report_path.relative_to(ROOT).as_posix()})。\n\n" + previous, encoding="utf-8")
    assert (MAIN / "最新研究结论.md").read_bytes() == report_path.read_bytes()
    print("企业收款结果已登记：4个新账户、8个复用对照，主夏普0.410254，目标仍active。", flush=True)


if __name__ == "__main__":
    run()
