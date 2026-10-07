"""记录财务收款期限研究与上一项问卷证据，保留未达标事实。"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.industrial_collection_delay_daily_v1 as study
import research.entrepreneur_sales_collection_daily_v1 as survey
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.strategy_review_diagnostics_v1 import metrics

MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"


def run():
    if (study.OUT / "completed_round.json").exists():
        raise RuntimeError("财务收款期限研究已完成，不覆盖。")
    result = read(study.OUT / "result.json")
    source = read(study.source.OUT / "result.json")
    old = read(survey.OUT / "result.json")
    p = result["primary"]
    q = old["primary"]
    increment = next(r for r in result["comparisons"] if r["cost"] == "STRESS" and r["purpose"] == "RECEIVABLE_DELAY_INCREMENT")
    control = next(r for r in result["all_accounts"] if r["cost"] == "STRESS" and r["policy"] == study.CONTROL + "__FIXED5_NO_ADD")
    keycols = ["idx", "equity", "shares", "cash", "commission", "slippage_cost", "filled_quantity"]
    seen, duplicates, counts = {}, [], {"survey_new_accounts": 0, "industrial_new_accounts": 0}
    for module, items in [(survey, [a for a in old["all_accounts"] if a["policy"].startswith(survey.SIGNAL)]), (study, result["all_accounts"])]:
        for item in items:
            path = module.OUT / "accounts" / item["cost"] / item["policy"] / "ledger.parquet"
            ledger = pd.read_parquet(path)
            key = tuple(pd.util.hash_pandas_object(ledger[keycols], index=False))
            if key in seen:
                prior_path = seen[key]
                pd.testing.assert_frame_equal(ledger[keycols], pd.read_parquet(prior_path)[keycols], check_exact=True)
                duplicates.append({"original": prior_path.relative_to(ROOT).as_posix(), "repeat": path.relative_to(ROOT).as_posix()})
            else:
                seen[key] = path
            counts["survey_new_accounts" if module is survey else "industrial_new_accounts"] += 1
            if item["policy"] == module.PRIMARY and item["cost"] == "STRESS":
                m = metrics(ledger)
                for name in ["end_equity", "annual_return", "sharpe", "max_drawdown", "profit"]:
                    np.testing.assert_allclose(m[name], item[name], atol=1e-12, rtol=0)
                gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
                fees = float((ledger.commission + ledger.slippage_cost).sum() + ledger.terminal_exit_reserve.iloc[-1])
                np.testing.assert_allclose(gross - fees, item["profit"], atol=1e-6, rtol=0)
    assert counts == {"survey_new_accounts": 4, "industrial_new_accounts": 8}
    passes = result["joint_target_pass_accounts"] + old["joint_target_pass_new_accounts"]
    attribution = result["primary_cash_attribution"]
    monitor = next(r for r in result["mature_monitor"] if r["model"] == study.SIGNAL)
    status = "COMPLETED_TWELVE_NEW_CASH_COLLECTION_ACCOUNTS_NO_QUALIFIED_STRATEGY" if not passes else "COMPLETED_CASH_COLLECTION_ACCOUNTS_INDEPENDENT_VALIDATION_REQUIRED"
    report = f"""本轮围绕企业能否及时收回销售款，完成两类不同证据的研究：企业家主观回款判断，以及国家统计局财务口径的应收账款回收期。累计新增12个完整账户，实际{len(seen)}条不同路径；季度实验另外8个原对照经逐值匹配后复用。12个新增账户共同目标通过{passes}个。最近完成的财务回收期主压力账户净夏普{p['sharpe']:.6f}、年化收益{p['annual_return']:.4%}、最大回撤{abs(p['max_drawdown']):.4%}。独立前向结果仍为0，高夏普总目标保持active，尚未完成。

此前的企业家问卷取得32份原季度报告、29个公布日，主压力账户期末{q['end_equity']:,.2f}元、夏普{q['sharpe']:.6f}、年化{q['annual_return']:.4%}、回撤{abs(q['max_drawdown']):.4%}。相对同池银行需求与审批对照，全期点估计改善，近期增量略负，按季度公布间隔估计的区间跨零。每个两年窗口只有4—7个不同季度状态，不能把每日重复使用看成大量宏观证据。该实验已封存原结果，没有改参数营救。

这次财务指标覆盖2020年2月至2026年7月的72份原月报，1月按制度免报。先通过官方最新发布67页目录定位54个月，再定向取得目录覆盖以外的18份旧档案，正文、经济效益表总计与定义均核对。18份旧档案仅有原页时钟支撑，部分只有日级成文日期，明确按当日日末保守可用；没有把网页迁移后的2023年URL日期当原报告日期，也没有伪造09:30时间。历史不可变首版仍未认证。

[国家统计局2019年原报告](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900265.html)说明，当年用营业收入替代主营业务收入，且应收统计包含票据；[2020年原报告](https://www.stats.gov.cn/xxgk/sjfb/zxfb2020/202003/t20200327_1767778.html)改回应收账款。本研究只纳入2020年后同名口径，公式为360乘平均应收账款、除以营业收入，再乘累计月数/12。国家统计局还说明调查范围和比较基数会调整，不能从不同年份已公布水平直接推算官方同比。因此唯一新增字段采用原文当时直接公布的同比增减天数。

这一指标属于工业企业财务汇总，不是全部企业、全部宏观部门或沪深300企业的加权现金流，也不是逐笔发票实际回款时间。回收期拉长既可能来自债权余额增加，也可能来自收入分母减少；样本变化也有影响。机制假设是它可能帮助区分暂时资金压力与持续经营压力，不能直接被解释为纯流动性冲击或违约上升。

8个新账户都在同一个可观察样本上比较。基础模型使用价格压力、趋势、波动结构、银行资金日历残差和统计月份sin/cos；主模型只加原文回收期同比变化。每天使用最近两日历年的成熟五日原点，月报公布时间本身也在两年内；252个原点才开始训练，126近邻固定，训练内标准化和截断，未搜索方向、阈值、窗口或宏观字段组合。月报有效期固定70自然日，用于容纳1月免报的间隔。

共有{result['unique_prediction_days']:,}个预测日、{result['new_conditional_distributions']:,}个条件分布。每个训练池{result['training_rows_min']}—{result['training_rows_max']}个日原点，来自{result['distinct_training_release_months_min']}—{result['distinct_training_release_months_max']}个不同月报。完整账本保留源信息未知和训练不足的日子；前期现金日不能被当作经历了市场冲击仍稳定获利。

20万元主压力账户从2020-01-02连续记账至2026-09-24，期末{p['end_equity']:,.2f}元，净损益{p['profit']:,.2f}元；毛损益{attribution['gross_pnl']:,.2f}元，费用与退出储备{attribution['cost_and_reserve']:,.2f}元。平均敞口{p['mean_exposure']:.4%}。T+1、100份、现金股息、压力费用、最多50%目标仓位和账户尾部预算均执行。主期限固定第五个后续开盘退出，风险可提前减仓，持仓不加仓；滚动期限结果也完整保留。

同样本、同固定期限的价格资金对照净夏普{control['sharpe']:.6f}、年化{control['annual_return']:.4%}、最大回撤{abs(control['max_drawdown']):.4%}。新增回收期信息的年化算术收益差为{increment['daily_blocks']['annual_arithmetic_difference']:.4%}；20日区块95%区间[{increment['daily_blocks']['lower_95']:.4%}, {increment['daily_blocks']['upper_95']:.4%}]，月报公布间隔区间[{increment['release_blocks']['lower_95']:.4%}, {increment['release_blocks']['upper_95']:.4%}]。两个预先固定时期的增量分别为{increment['fixed_periods'][0]['annual_arithmetic_difference']:.4%}和{increment['fixed_periods'][1]['annual_arithmetic_difference']:.4%}。预定信息增量检验通过状态为{result['development_increment_gate']}，主账户滚动两年共同目标通过{result['primary_rolling_two_year_joint_passes']}次。

尾部监控有{monitor['mature_nonoverlap_rows']}个成熟非重叠原点、{monitor['assessable_rows']}个可评估时点和{monitor['alerts']}个警报。这个监控检查风险估计，没有报警不表示存在高夏普优势。各月报重复使用和相邻月度状态的持续性仍限制统计把握；区块结果也未校正全部历史反复选择。

两类证据的样本覆盖不同，不能把它们的收益相加，或根据各自较好的时期拼接账户。保存分布重算、未来/过期标签扰动、来源时钟、现金/T+1/费用/股息和预算检查已完成。所有失败路径保留，没有因本轮结果变化目标、扩大资产范围或恢复旧版策略执行。

最新财务原文为{source['latest_source']['stat_month']}报告、公布时间{source['latest_source']['published_at']}。来源新鲜不等于策略已经验证：current_market_view仍为NO_VIEW，没有当前交易建议或订单授权。两个账户实验及本次来源取得均已结束，没有遗留计算进程。总目标仍继续，后续需要新的可区分证据或更有信息量的行为测量，不能依靠修改这两组失败规则制造达标结果。没有制作审核包或用户表格。
"""
    report_path = study.OUT / "本轮研究结论.md"
    report_path.write_text(report, encoding="utf-8")
    classification = "PROGRESS_QUARTERLY_AND_FINANCIAL_CASH_COLLECTION_TWELVE_NEW_ACCOUNTS"
    receipt = {"at": now(), "study_id": study.STUDY, "status": status, "classification": classification,
        **counts, "new_accounts_total": 12, "distinct_new_paths": len(seen), "duplicates": duplicates,
        "reused_quarterly_controls": 8, "source_quarters": 32, "source_financial_months": 72,
        "joint_target_pass_new_accounts": passes, "primary_survey": q, "primary_industrial": p,
        "industrial_increment": increment, "industrial_result_sha256": digest(study.OUT / "result.json"),
        "survey_result_sha256": digest(survey.OUT / "result.json"), "source_result_sha256": digest(study.source.OUT / "result.json"),
        "report_sha256": digest(report_path), "goal_status": "active", "goal_achieved": False,
        "new_independent_forward_observations": 0, "current_market_view": "NO_VIEW", "orders_authorized": False}
    save(study.OUT / "completed_round.json", receipt, True)
    state = read(MAIN / "current_status.json")
    state.update(updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification=classification, same_condition_consecutive_no_progress_goal_turns=0,
        active_blocker_id=None, active_blocker_description=None,
        latest_user_requested_study=study.OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=status,
        latest_cash_collection_round=receipt,
        remaining_research_question="两种企业收款证据已分别形成完整账户，不同样本不可拼接收益；继续寻找新的机制证据，不改本轮冻结规则，独立前向仍不足。",
        goal_metadata_note="本轮取得32份季度和72份财务月报，新增12个账户；具备实质进展但总目标尚未完成。")
    for name in [study.source.prior.probe.STUDY, study.source.prior.STUDY, study.source.STUDY, study.STUDY]:
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
        research_execution_state=status,
        last_research_result=f"两种收款证据完成12个新账户、{len(seen)}条不同路径；最新主压力夏普{p['sharpe']:.6f}、年化{p['annual_return']:.4%}、回撤{abs(p['max_drawdown']):.4%}，独立前向0，总目标active。",
        goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    marker = "<!-- INDUSTRIAL_COLLECTION_DELAY_ROUND_20260926 -->"
    previous = status_path.read_text(encoding="utf-8")
    assert marker not in previous
    status_path.write_text(f"{marker}\n\n> 2026-09-26 两种企业收款证据完成12个新账户、{len(seen)}条不同路径；共同目标通过{passes}个，独立前向0，总目标ACTIVE。见[本轮结论]({report_path.relative_to(ROOT).as_posix()})。\n\n" + previous, encoding="utf-8")
    assert (MAIN / "最新研究结论.md").read_bytes() == report_path.read_bytes()
    print(f"两种收款证据完成12个新账户、{len(seen)}条不同路径；共同目标通过{passes}个，总目标active。", flush=True)


if __name__ == "__main__":
    run()
