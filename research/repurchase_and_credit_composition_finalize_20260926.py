"""保存本轮回购、期限与融资构成结果，更新仍未达成的高夏普主线。"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.funding_repurchase_demand_daily_v1 as repurchase
import research.repurchase_fixed_maturity_account_v1 as expiry
import research.repurchase_horizon_loss_diagnostic_v1 as diagnostic
import research.tsf_government_composition_source_v1 as source
import research.tsf_composition_funding_daily_v1 as macro
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.strategy_review_diagnostics_v1 import metrics

MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"


def percent(value):
    return f"{value * 100:.4f}%"


def run():
    r, e, d, s, m = [read(module.OUT / "result.json") for module in [repurchase, expiry, diagnostic, source, macro]]
    rows = [*r["all_accounts"], *[v for v in e["all_accounts"] if not v["policy"].endswith("__ORIGINAL")], *m["all_accounts"]]
    assert len(rows) == 36 and r["new_full_accounts"] + e["new_full_accounts"] + m["new_full_accounts"] == 36
    assert len(r["account_checks"]) + len(e["account_checks"]) + len(m["account_checks"]) == 36
    recomputed = []
    for module, result in [(repurchase, r), (expiry, e), (macro, m)]:
        primary = result["primary"]
        ledger_path = module.OUT / "accounts/STRESS" / primary["policy"] / "ledger.parquet"
        recomputed_metrics = metrics(pd.read_parquet(ledger_path))
        for key in ["sharpe", "annual_return", "max_drawdown", "end_equity"]:
            np.testing.assert_allclose(recomputed_metrics[key], primary[key], atol=1e-12, rtol=0)
        recomputed.append({"study_id": module.STUDY, "ledger": ledger_path.relative_to(ROOT).as_posix(), "sha256": digest(ledger_path)})
    assert not any(v["historical_point_targets_met"] for v in rows)
    assert s["selected_months"] == s["extracted_months"] == 80
    historical_plan_folder = ROOT / "reports/research/510300_corporate_repurchase_original_plan_completion_v1/results"
    historical_documents = read(historical_plan_folder / "supplement_documents.json") + read(historical_plan_folder / "gateway_supplement_documents.json")
    assert len(historical_documents) == 79
    main_age = next(v for v in d["account_attribution"] if v["model"] == repurchase.PRIMARY)
    gross_first_five = sum(main_age.get(k, 0.) for k in ["ENTRY_INTRADAY", "FIRST_NIGHT", "INTRADAY_BEFORE_5_EXIT", "NIGHTS_2_TO_5", "DIVIDEND_FIRST_NIGHT", "DIVIDEND_NIGHTS_2_TO_5"])
    gross_after_five = sum(main_age.get(k, 0.) for k in ["INTRADAY_AFTER_5_EXIT", "NIGHTS_AFTER_5", "DIVIDEND_NIGHTS_AFTER_5"])
    macro_control = next(v for v in m["all_accounts"] if v["policy"] == macro.FUNDING + "__FIXED5_NO_ADD" and v["cost"] == "STRESS")
    macro_roll = next(v for v in m["all_accounts"] if v["policy"] == macro.COMPOSITION + "__ROLL_NO_ADD" and v["cost"] == "STRESS")
    paired = next(v for v in m["comparisons"] if v["cost"] == "STRESS" and v["purpose"] == "COMPOSITION_INCREMENT")
    price_weakness = next(v for v in m["forecast_evaluation"] if v["subset"] == "PRIOR_FIVE_DAY_DECLINE")
    information = next(v for v in m["forecast_evaluation"] if v["subset"] == "ALL")
    pbc_url = "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/35ec0aa27604417888826e7ff128cc4a/index.html"
    report = f"""本轮完成36个新的完整模拟账户：8个回购披露增量账户、16个持仓期限/禁止加仓对照、12个社融融资构成账户。全部账户都未同时达到压力成本后夏普1.2、年化10%、最大回撤不超过10%的目标。高夏普总目标仍为active，尚未找到可确认稳定的策略。代码、预测、完整账本及失败结果均已保存；本轮未制作审核包或用户表格。

回购披露真正接入了账户。

在先前2,407份选定原文基础上，进一步补充79份历史原方案文件，把执行披露连接到当时已经存在的原方案。最终948条执行记录有直接身份依据，其中496条可按当时指数权重和此前成交额标准化。这个变量表示过去20个决策日系统能确认的新增披露信息，既不是全部实际回购，也不是每日真实买盘；未识别记录仍未知，重复累计额没有再次当成新增需求。

四种模型分别使用价格、价格加资金、价格加回购、三类信息合并，采用完全相同训练池、成本和风险预算。回购共同信息覆盖453日，满足252个成熟训练原点后，只剩{r['unique_prediction_days']}个预测日，从{r['first_prediction_date'][:10]}至{r['last_prediction_date'][:10]}。不能把训练之前的现金日说成策略长期稳定。

合并信息的主压力账户期末权益{r['primary']['end_equity']:,.2f}元，夏普{r['primary']['sharpe']:.6f}，年化{percent(r['primary']['annual_return'])}，最大回撤{percent(abs(r['primary']['max_drawdown']))}。在固定可用期2025-10-31至2026-08-27单独计算，夏普为-0.658679；缩短统计区间也没有得到优势。相比价格资金对照，回购信息没有通过增量检验。

不是把亏损归咎于披露延迟就结束。

逐批次重建原账本发现，合并模型形成3个完整持仓周期，持续43至59个交易日；其预测的却是五个交易日收益。实际批次前五日毛损益合计{gross_first_five:,.2f}元，第五日以后续持毛损益{gross_after_five:,.2f}元，另有费用{main_age['cost']:,.2f}元。这是原持仓路径的损益归属，不能直接当作第五日退出可以取得的利润。

因此继续完成16个完整账户反事实模拟：所有四类预测都比较“只禁止加仓”与“禁止加仓且固定第五个后续交易日开盘退出”，期间风险预算仍可要求提前减仓。主固定期限压力账户期末权益{e['primary']['end_equity']:,.2f}元，夏普{e['primary']['sharpe']:.6f}，年化{percent(e['primary']['annual_return'])}，最大回撤{percent(abs(e['primary']['max_drawdown']))}。其毛损益为{e['primary_cash_attribution']['gross_pnl']:,.2f}元，费用为{e['primary_cash_attribution']['cost_and_reserve']:,.2f}元。提前退出改变了后续资金、入场和换手，原先的归因利润不能完整保留。16个新账户也都没有达标，旧失败账户保留。

来源诊断显示，完成公告占已确认披露强度约{d['source_diagnostics']['completion_share_of_strength']:.1%}，并非多数信息都来自已完成方案。397条有明确报告截止日，截止日至可用日中位间隔4个自然日；这不是实际买入距披露的时长，因为期间购买通常无法逐日定位。茅台占该强度总和约{d['source_diagnostics']['largest_company_strength_fraction']:.1%}，前五家公司合计约{d['source_diagnostics']['top_five_strength_fraction']:.1%}，也不能把这项观测当成均匀代表全部成分股的买盘。

随后检验社融总量背后的融资构成。

从既有央行当期原始月报提取2019年12月至2026年7月共80个月的“政府债券余额占社融比重同比变化”。每份原文哈希与旧记录相同，总社融同比也逐月对应；这80个数值是新增提取字段，不是80个月新发生的独立前向样本。例如2025年8月月报同时披露总社融同比增长8.8%、政府债券占比21.1%及该比重同比提高2.2个百分点，说明总量和构成提供不同描述。[央行原文]({pbc_url})。

新假设是：总量增长相近时，融资部门的变化可能对应不同的财政支持、实体融资需求和资金环境。政府债券占比不是私人信贷、政府买股票或中性利率的替代；没有预设“占比上升就买”或相反方向。主比较只增加这一项构成信息，保留价格、资金意外和总社融已公布同比变化作为对照。

月报按保守披露日末处理，最早下一交易日使用，31自然日后过期。模型每天只使用此前两个日历年、五日结果已经成熟的原点，固定126近邻，未搜索窗口或方向。这轮产生{m['unique_prediction_days']}个预测日，时间为{m['first_prediction_date'][:10]}至{m['last_prediction_date'][:10]}；每次训练有{m['training_rows_min']}至{m['training_rows_max']}个成熟原点，但只有{m['distinct_training_reference_months_min']}至{m['distinct_training_reference_months_max']}个不同统计月份，不能把日频重复值当成同等数量的独立宏观消息。

融资构成主压力账户期末权益{m['primary']['end_equity']:,.2f}元，夏普{m['primary']['sharpe']:.6f}，年化{percent(m['primary']['annual_return'])}，最大回撤{percent(abs(m['primary']['max_drawdown']))}。毛损益仅{m['primary_cash_attribution']['gross_pnl']:,.2f}元，佣金、滑点及退出储备合计{m['primary_cash_attribution']['cost_and_reserve']:,.2f}元，净亏损{abs(m['primary']['profit']):,.2f}元。它不能覆盖交易成本。

同信息每日续持对照夏普{macro_roll['sharpe']:.6f}、年化{percent(macro_roll['annual_return'])}；较简单的价格资金固定期限对照夏普{macro_control['sharpe']:.6f}、年化{percent(macro_control['annual_return'])}。这些小幅正结果也远未达到目标，不能把其中某条重新指定为主策略。新增构成变量的均值预测MSE相对总社融对照恶化约{-information['primary_mse_improvement'][macro.AGGREGATE]:.2%}；在事前价格弱势子集也没有改善。

构成信息在不同期间的结果并不一致。相对同期限总社融对照，2020至2023年的年化算术收益差为{percent(paired['fixed_periods'][0]['annual_arithmetic_difference'])}，2024至末端为{percent(paired['fixed_periods'][1]['annual_arithmetic_difference'])}；全期配对区块区间跨过零，不能声称稳定增量。

这也提供了规律监控的一条具体反证：融资构成模型的尾部分位监控已有258个固定相位成熟观察，199次足够形成60条滚动监控，未触发预设尾部报警；但账户仍因优势不足、费用消耗而亏损。没有尾部警报只说明这项警报没有触发，不能说明收益规律仍然有效。回购研究则只有38个成熟非重叠观察，连这项监控的样本门槛也未达到。

统计口径说明另行保留。2019年12月社融新增国债及地方一般债的统计范围，原文还说明历史可比值回溯；研究没有把后来的回溯年表提前用于过去。总社融的三月差是当时已公布同比值之差，跨口径变化时不能解释为统一口径的真实信用加速度。原提取器未匹配到该说明开头的写法，已在methodology_note_supplement.json补存原文上下文，80个月数值和已冻结试验没有改动。

本轮完成的验证用于确认这些数字确实来自既定规则：36个账户均检查现金、分红、T+1及风险预算；五日期限入口在原模式下完整还原旧主账户；所有保存分布按选中历史样本重算，并验证未成熟及过期标签不影响当时预测；三个主账户指标重新从保存账本计算一致。这些是实现验证，不是外部投资审查或策略有效性的证明。

当前可保留的结论是：实际回购披露、融资构成与执行期限都已被具体检验；它们尚未产生足以覆盖成本的稳定优势。后续继续寻找独立的信息增量，并把预测偏差、费用后的优势和尾部风险分别检查，不能通过放宽本轮门槛、反选有利窗口或给失败变量换方向来宣布达标。高夏普目标保持active，当前市场视图NO_VIEW，独立前向验证仍为0。全部工作属于研究模拟，没有交易授权。
"""
    report_path = macro.OUT / "本轮研究结论.md"
    with report_path.open("x", encoding="utf-8") as handle:
        handle.write(report)
    paths = [module.OUT / "result.json" for module in [repurchase, expiry, diagnostic, source, macro]]
    completed = {"at": now(), "study_id": macro.STUDY,
        "status": "COMPLETED_36_ACCOUNTS_NO_QUALIFIED_STRATEGY",
        "new_full_accounts": 36, "new_conditional_distributions": r["new_conditional_distributions"] + m["new_conditional_distributions"],
        "new_funding_regression_fits": 0, "new_source_composition_months": 80,
        "joint_target_pass_accounts": 0, "primary_repurchase": r["primary"], "primary_maturity": e["primary"], "primary_composition": m["primary"],
        "saved_primary_metrics_recomputed": recomputed,
        "result_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}, "report_sha256": digest(report_path),
        "goal_status": "active", "goal_achieved": False, "current_market_view": "NO_VIEW",
        "new_independent_forward_observations": 0, "orders_authorized": False, "review_package_created": False}
    save(macro.OUT / "completed_round.json", completed, True)
    state_path = MAIN / "current_status.json"
    state = read(state_path)
    state.update(updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification="PROGRESS_36_NEW_ACCOUNTS_AND_80_RELEASED_MACRO_COMPOSITION_MONTHS",
        same_condition_consecutive_no_progress_goal_turns=0, active_blocker_id=None, active_blocker_description=None,
        latest_user_requested_study=macro.OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=completed["status"],
        latest_repurchase_and_credit_composition_round=completed,
        remaining_research_question="已确认回购披露及政府债券社融构成的36个新账户未达标。继续寻找与已失败用途不同的可检验信息增量，分别核对成本后优势、预测偏差及尾部风险；禁止反选本轮有利窗口。",
        goal_metadata_note="本轮完成原方案身份与披露强度、36个新完整账户、损失批次归因和80个月社融融资构成字段，属于实质进展；高夏普目标未完成。")
    studies = ["510300_CORPORATE_REPURCHASE_PLAN_IDENTITY_V1", "510300_CORPORATE_REPURCHASE_ORIGINAL_PLAN_COMPLETION_V1",
               "510300_CORPORATE_REPURCHASE_DISCLOSED_DEMAND_V1", repurchase.STUDY, diagnostic.STUDY, expiry.STUDY, source.STUDY, macro.STUDY]
    for study in studies:
        if study not in state["completed_followup_studies"]:
            state["completed_followup_studies"].append(study)
    save(state_path, state)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(as_of_date=now()[:10], current_round=macro.STUDY, latest_integrated_experiment=macro.STUDY,
        current_protocol=(macro.OUT / "protocol.json").relative_to(ROOT).as_posix(),
        latest_progress_receipt=(macro.OUT / "completed_round.json").relative_to(ROOT).as_posix(),
        latest_continuation_report=report_path.relative_to(ROOT).as_posix(),
        latest_continuation_classification=state["latest_goal_turn_classification"], research_execution_state=completed["status"],
        last_research_result="36个新完整账户均未同时达到1.2/10%/10%；完成回购批次损益归因和80个月政府债券社融构成提取；目标active。",
        goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    previous = status_path.read_text(encoding="utf-8")
    marker = "<!-- REPURCHASE_AND_CREDIT_COMPOSITION_20260926 -->"
    if marker not in previous:
        status_path.write_text(f"{marker}\n\n> 2026-09-26 完成回购、期限及融资构成36个新完整账户，全部未达到共同目标；宏观主压力夏普{m['primary']['sharpe']:.6f}、年化{percent(m['primary']['annual_return'])}。目标ACTIVE且未完成。见[本轮结论]({report_path.relative_to(ROOT).as_posix()})。\n\n" + previous, encoding="utf-8")
    print("36个新账户、损益归因、宏观构成及中文结论已保存；主线保持active。", flush=True)


if __name__ == "__main__":
    run()
