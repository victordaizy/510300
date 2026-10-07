"""记录资金传导断点与五日期限账户结果，不增加回测或改动旧研究。"""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.selected_mix_reappraisal_v1 import read, save, now
import research.funding_forecast_maturity_account_v1 as study

OUT = study.OUT
MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"
ECB_URL = "https://www.ecb.europa.eu/press/financial-stability-publications/fsr/special/html/ecb.fsrart202305_01~830184261b.en.html"
NAMES = {"PRICE": "基础价格", study.prior.FUNDING: "价格加资金意外",
         study.prior.RESPONSE: "价格加下午表现", study.prior.PRIMARY: "资金与下午价格联合"}


def run():
    result = read(OUT / "result.json")
    diagnostic = read(study.DIAGNOSTIC / "result.json")
    primary = result["primary"]
    original_model = study.prior.PRIMARY
    gate = next(row for row in diagnostic["gate_summary"] if row["policy"] == original_model and row["period"] == "2020_2023")
    attr = next(row for row in diagnostic["fixed_share_path_attribution"] if row["policy"] == original_model)
    support = next(row for row in diagnostic["neighbor_support"] if row["policy"] == original_model)
    outcome = next(row for row in diagnostic["five_day_outcomes"] if row["policy"] == original_model and row["subset"] == "ACTUAL_BUY")
    measures = {(row["policy"], row["cost"]): row for row in result["all_accounts"]}
    pairs = [row for row in result["comparisons"] if row["policy"] == study.PRIMARY and row["cost"] == "STRESS"]
    account_lines = []
    for model, name in NAMES.items():
        roll = measures[model + "__ROLL_NO_ADD", "STRESS"]
        fixed = measures[model + "__FIXED5_NO_ADD", "STRESS"]
        account_lines.append(f"- {name}：不加仓、逐日判断的夏普{roll['sharpe']:.6f}、年化{roll['annual_return']:.4%}；五日到期的夏普{fixed['sharpe']:.6f}、年化{fixed['annual_return']:.4%}、最大回撤{abs(fixed['max_drawdown']):.4%}、期末{fixed['end_equity']:.2f}元。")
    comparison_lines = []
    for row in pairs:
        comparison_lines.append(f"- 相对{row['control']}：算术年化增量{row['annual_arithmetic_difference']*100:.4f}个百分点，95%区间[{row['lower_95']*100:.4f}, {row['upper_95']*100:.4f}]个百分点；两个固定时期增量为{row['fixed_periods'][0]['annual_arithmetic_difference']*100:.4f}、{row['fixed_periods'][1]['annual_arithmetic_difference']*100:.4f}个百分点。")
    ledger = pd.read_parquet(OUT / "accounts/STRESS" / study.PRIMARY / "ledger.parquet")
    closed = pd.read_parquet(OUT / "accounts/STRESS" / study.PRIMARY / "cycles.parquet")
    mean_intervals = float((closed.end_index - closed.start_index).mean()) if len(closed) else None
    maximum_intervals = int((closed.end_index - closed.start_index).max()) if len(closed) else None
    account_cost = float((ledger.commission + ledger.slippage_cost).sum())
    gross = float((ledger.price_pnl + ledger.dividend_recognized).sum())
    reserve = float(ledger.terminal_exit_reserve.iloc[-1])
    assert abs(gross - account_cost - reserve - primary["profit"]) < 1e-6
    new_summary = {"same_share_path_gross_pnl": gross, "execution_cost": account_cost,
                   "terminal_reserve": reserve, "net_profit": primary["profit"],
                   "closed_cycles": len(closed), "mean_holding_intervals": mean_intervals,
                   "maximum_holding_intervals": maximum_intervals}
    save(OUT / "results/primary_interpretation.json", new_summary)
    passed = sum(row["historical_point_targets_met"] for row in result["all_accounts"])
    candidate = result["status"] == "DEVELOPMENT_CANDIDATE_REQUIRES_INDEPENDENT_VALIDATION"
    decision = "符合本轮开发候选门槛，仍需独立验证" if candidate else "固定方案不予晋升；没有合格高夏普策略"
    report = f"""本轮实际完成四个已保存压力账户的6532个决策日诊断，并新增16个期限对照账户。预指定联合五日到期方案的压力夏普为{primary['sharpe']:.6f}、复合年化{primary['annual_return']:.4%}、最大回撤{abs(primary['max_drawdown']):.4%}，20万元期末为{primary['end_equity']:.2f}元。{decision}。高夏普总目标仍为active、未完成。

先查明资金到收益的断点。

原联合方案在2020—2023共970个完整账户日，其中663日无预测。具备预测的307日中，287日因预测均值非正而未入场，12日不满足已知五日弱势条件，8日扣除费用与方差惩罚后不值得进入。这一时期没有因为连100份最低单位的预算也不足而被拒绝的入场日。第一笔交易迟到，不能一概解释为风险预算过紧。

全期920个预测原点中，原联合方案实际发生买入或增仓的27日，原模型平均预测五日毛收益{outcome['mean_forecast']:.4%}，实际对应五日标的含息收益均值{outcome['mean_return5']:.4%}，区块区间[{outcome['lower95']:.4%}, {outcome['upper95']:.4%}]。这27日按固定顺序只能留下{outcome['greedy_nonoverlapping_labels']}个不重叠五日标签；它们并不是27笔独立、满仓、可直接相加的交易。

原联合账户实际份额的隔夜价格损益为{attr['overnight_price_pnl']:.2f}元，股息为{attr['dividend_pnl']:.2f}元，两者合计{attr['overnight_price_pnl']+attr['dividend_pnl']:.2f}元；日内损益为{attr['intraday_price_pnl']:.2f}元，费用为{attr['costs']:.2f}元，净损益{attr['net_pnl']:.2f}元。除息价格损失与股息必须合看。该分解不意味着可以删除隔夜亏损，因为新买入份额仍受T+1限制。

联合模型的126个近邻平均占可用训练池{support['mean_selected_fraction']:.2%}，平均年龄{support['mean_neighbor_age_calendar_days']:.1f}个自然日；可按贪心顺序保留的不重叠训练标签中位数为{support['median_nonoverlapping_training_labels']:.0f}，这不是估计有效样本量。预测均值与同池HISTORY均值相关为{support['correlation_with_history_mean']:.3f}。这些是样本支持诊断，不据此搜索更短训练期或更小邻居数。

为什么做本次期限比较。

融资是否容易，与市场能否低成本吸收交易是不同对象。欧洲央行的机制分析支持把两者分开观察，但不提供510300的收益证明。[来源：欧洲央行2023年5月研究]({ECB_URL})。

本轮进一步检验的是实际持有期限：原模型预测未来五天，但每日重新判断可能让同一笔仓位持续更久。五日到期方案只在空仓时沿用原入场计划；实际买入后的第五个后续交易日开盘退出，中途仅按每天可用的尾部预测和回撤约束减仓。到期卖出当天不重新入场。期间禁止加仓，避免新旧预测期限混在同一周期。

为分清期限和加仓的影响，另设不加仓但仍按原五日预测逐日判断持有/退出的对照；它与原可加仓账户也单独比较。四种信息组合都执行两种持有规则及BASE/STRESS两档成本，共16个新账户；原8个账户仅只读复用。本轮没有搜索最佳持有天数、重新训练预测、增加指标或更换主候选。

完整压力账户结果如下。

{chr(10).join(account_lines)}

所有账户保留2020-01-02至2026-09-24的1633个交易日，20万元连续本金；资料与训练不足日留在全账户中。现金和无风险收益按零，242日年化。现行压力目标仍为夏普至少1.2、复合年化至少10%、最大回撤不超过10%。16个新账户中三目标点值同时通过{passed}个；主压力账户滚动两年联合通过{result['primary_rolling_two_year_joint_passes']}次。

联合五日到期方案的增量如下。这里的差值是日收益算术年化增量，不是两个复合年化的相减。

{chr(10).join(comparison_lines)}

区间使用预先固定的20日循环区块与2000次抽样，不是全项目多重试验校正。2020—2023没有实际交易的比较增量为零，不能称为跨时期稳定性。部分点值变好，也不能替代增量区间和账户目标。

主五日到期压力账户完成{len(closed)}个闭合周期，最长{maximum_intervals}个开盘间隔；按原实际份额计毛损益{gross:.2f}元、费用{account_cost:.2f}元、末端储备{reserve:.2f}元，净损益{primary['profit']:.2f}元。该分解仍不是一条重新运行的零成本策略。

工程结果与边界。

期限账户在冻结前完成原模拟器等价性、第五个后续交易日到期、缺预测仍到期、持仓不加仓及现金/T+1/尾部预算的定向检查。16个实际账户均完成会计恒等式与入场风险检查；既有预测、原账户与源文件保持不变。50%目标仓位、五日ES95预算2.5%、10%标的跳空预算5%及峰值余量一半、10%回撤触发退出全部沿用。到期与止损指令不保证遇到不可成交行情仍按价成交。

前置诊断曾因无成交日shares_after为空、idx同时作为列与索引而失败；两份原失败代码及记录保留，独立V1_2仅修正字段和索引后完成6532日分类与损益复算，没有改策略或原账户。期限账户本身完整运行成功。

新增独立前向观察仍为0。分钟资料仍有第三方来源、bar标签语义及历史首次送达未认证的原限制，本轮没有把这些历史资料升级为实时认证证据。当前市场视图为NO_VIEW。没有制作审核包或用户表格。{decision}，目标没有标记完成。
"""
    (OUT / "本轮研究结论.md").write_text(report, encoding="utf-8")
    status = "FUNDING_MATURITY_SIXTEEN_ACCOUNTS_COMPLETED_TARGET_UNMET" if not candidate else "FUNDING_MATURITY_REQUIRES_INDEPENDENT_VALIDATION"
    completed = {"at": now(), "study_id": study.STUDY, "diagnostic_study_id": diagnostic["study_id"],
        "status": status, "diagnosed_saved_decision_rows": diagnostic["decision_rows"],
        "new_unique_full_accounts": 16, "reused_original_accounts": 8, "new_fits": 0,
        "new_market_downloads": 0, "primary": primary, "primary_interpretation": new_summary,
        "all_new_account_joint_point_passes": passed, "development_candidate": candidate,
        "goal_status": "active", "goal_achieved": False, "independent_forward_observations": 0,
        "orders_authorized": False, "review_package_created": False}
    save(OUT / "completed_round.json", completed)
    status_path = MAIN / "current_status.json"
    state = read(status_path)
    state.update(updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification="PROGRESS_FUNDING_TRANSMISSION_DIAGNOSIS_AND_SIXTEEN_ACCOUNTS",
        same_condition_consecutive_no_progress_goal_turns=0, active_blocker_id=None, active_blocker_description=None,
        latest_user_requested_study=OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=status,
        latest_funding_maturity_round=completed,
        remaining_research_question="资金信息传导和预测期限已比较，仍缺压力成本后可重复的正收益优势及独立验证。",
        goal_metadata_note="本轮6532个保存决策日诊断和16个新账户为实际进展；目标未完成，旧失败保留。")
    for identifier in [diagnostic["study_id"], study.STUDY]:
        if identifier not in state["completed_followup_studies"]:
            state["completed_followup_studies"].append(identifier)
    save(status_path, state)
    (MAIN / "最新研究结论.md").write_text(report, encoding="utf-8")
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = read(mandate_path)
    mandate.update(as_of_date=now()[:10], latest_user_instruction="请继续啊", current_round=study.STUDY,
        latest_integrated_experiment=study.STUDY, current_protocol=(OUT / "protocol.json").relative_to(ROOT).as_posix(),
        latest_progress_receipt=(OUT / "completed_round.json").relative_to(ROOT).as_posix(),
        latest_continuation_report=(OUT / "本轮研究结论.md").relative_to(ROOT).as_posix(),
        latest_continuation_classification=state["latest_goal_turn_classification"], research_execution_state=status,
        last_research_result=f"资金传导诊断和16个期限账户完成；主压力夏普{primary['sharpe']:.3f}、年化{primary['annual_return']:.3%}，总目标active且未完成。",
        goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    root_status = ROOT / "RESEARCH_STATUS.md"
    marker = "<!-- FUNDING_MATURITY_20260926 -->"
    text = root_status.read_text(encoding="utf-8")
    if marker not in text:
        lead = f"{marker}\n\n> 2026-09-26 资金传导断点与五日期限研究完成：6532个保存决策日、16个新账户，主压力夏普{primary['sharpe']:.3f}、年化{primary['annual_return']:.3%}、回撤{abs(primary['max_drawdown']):.2%}。目标ACTIVE、尚未找到合格策略；见[本轮结论]({(OUT / '本轮研究结论.md').relative_to(ROOT).as_posix()})。\n\n"
        root_status.write_text(lead + text, encoding="utf-8")
    print("本轮结论及主线状态已更新；高夏普目标保持active、未完成。", flush=True)


if __name__ == "__main__":
    run()
