"""只读保存结果，补充经济解释并更新本任务研究状态；不重训、不搜索参数。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/research/510300_daily_liquidity_insurance_tail_v1"
INTEGRATED = ROOT / "reports/research/510300_integrated_research_continuation_20260924"


def normalize(value):
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save(path, value):
    path.write_text(json.dumps(normalize(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def cycles(ledger):
    rows, opened, accumulated = [], None, 0.
    previous_shares = 0
    for row in ledger.itertuples():
        if opened is None and previous_shares == 0 and row.shares > 0:
            opened = row.date
            accumulated = 0.
        if opened is not None:
            accumulated += row.pnl
            if row.shares == 0:
                rows.append({"entry_date": opened, "exit_date": row.date, "net_pnl": accumulated})
                opened = None
        previous_shares = row.shares
    assert opened is None
    frame = pd.DataFrame(rows)
    assert abs(frame.net_pnl.sum() - ledger.pnl.sum()) < 1e-6
    return frame


def main():
    if (OUT / "saved_interpretation.json").exists():
        raise RuntimeError("保存结果解释已完成，禁止重复追加综合报告。")
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    verification = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    admission = json.loads((OUT / "data_admission.json").read_text(encoding="utf-8"))
    ledger = pd.read_parquet(OUT / "accounts/STRESS/MACRO_ledger.parquet")
    decisions = pd.read_parquet(OUT / "accounts/STRESS/MACRO_decisions.parquet")
    price_decisions = pd.read_parquet(OUT / "accounts/STRESS/PRICE_decisions.parquet")
    x = pd.read_parquet(OUT / "results/decision_information.parquet")
    scores = pd.read_parquet(OUT / "results/prediction_evaluation.parquet")
    completed = cycles(ledger)
    completed.to_parquet(OUT / "results/complete_inventory_cycles.parquet", index=False)
    wins = completed[completed.net_pnl > 0]
    losses = completed[completed.net_pnl < 0]
    cycle_summary = {"completed": len(completed), "winners": len(wins), "losers": len(losses),
                     "win_rate": len(wins) / len(completed) if len(completed) else None,
                     "mean_win_cny": wins.net_pnl.mean(), "mean_loss_cny": losses.net_pnl.mean(),
                     "realized_payoff_ratio": wins.net_pnl.mean() / -losses.net_pnl.mean() if len(losses) and len(wins) else None,
                     "worst_cycle": completed.loc[completed.net_pnl.idxmin()].to_dict()}
    scored = scores[scores.model.eq("MACRO") & scores.gross_return5.notna()].copy()
    p = result["primary"]
    comparison = result["comparison"]
    common = ledger.merge(x[["date", "global_shock", "preholiday3"]], on="date", validate="one_to_one")
    strata = []
    for name in ["global_shock", "preholiday3"]:
        for state in [False, True]:
            group = common[common[name].fillna(False).eq(state)]
            strata.append({"field": name, "state": state, "calendar_days": len(group),
                           "held_days": int(group.shares.gt(0).sum()), "account_pnl_cny": group.pnl.sum(),
                           "mean_exposure": group.exposure.mean(),
                           "interpretation": "保存账户实际损益分层，不是移除这些日期后的可执行账户收益。"})
    worst = ledger.nsmallest(3, "net_return").merge(
        decisions[["date", "mu5", "es95", "planned_weight", "pre_open_requested_quantity"]], on="date", validate="one_to_one").merge(
        x[["date", "credit_acceleration3_pp", "funding_gap_pp", "option_source_date", "global_shock"]], on="date", validate="one_to_one")
    worst_columns = ["date", "net_return", "pnl", "mu5", "es95", "planned_weight", "pre_open_requested_quantity", "credit_acceleration3_pp", "funding_gap_pp", "option_source_date", "global_shock"]
    net_pnl = float(ledger.pnl.sum())
    gross_same_shares = net_pnl + p["cost_cny"]
    tail_frequency = float(scored.q05_breach.astype(float).mean())
    interpretation = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "result_status": result["status"], "net_pnl_cny": net_pnl,
        "gross_pnl_same_saved_shares_before_costs": gross_same_shares,
        "gross_addback_is_not_new_optimized_account": True,
        "complete_cycles": cycle_summary,
        "mature_predictions": len(scored), "q05_breach_rate": tail_frequency,
        "median_nonoverlap_intervals_among126_neighbors": scored.nonoverlap_intervals_selected.median(),
        "tail_observation_alert_days": result["tail_observation_alerts"],
        "tail_alert_days_are_not_independent_events": True,
        "days_with_different_target_share_counts_vs_price": int(decisions.target_shares.ne(price_decisions.target_shares).sum()),
        "worst_days_with_predecision_information": worst[worst_columns].to_dict("records"),
        "descriptive_account_strata": strata,
        "formal_short_volatility_exposure_proven": False,
        "interpretation": "负偏和下跌非线性暴露已出现，未观察到足以补偿这些风险的净收益；本地三项宏观保险代理没有改善本轮完整账户。",
        "limitations": ["历史反复研究，无新增前向样本", "NFCI与中国r*未进入实验", "期权首次公布时钟未认证，额外滞后只是保守回放假设",
                        "最近两年只有约24个月度信用更新，五日标签重叠", "观察报警未作为新增启停策略使用", "现货回撤承接不等同收取期权权利金"],
        "new_model_fits": 0, "new_accounts": 0, "parameter_changes": 0,
    }
    save(OUT / "saved_interpretation.json", interpretation)
    references = [
        {"title": "Chicago Fed About the NFCI", "url": "https://www.chicagofed.org/research/data/nfci/about", "use": "美国综合金融条件、风险信用杠杆分项与ANFCI区别"},
        {"title": "Chicago Fed NFCI FAQ", "url": "https://www.chicagofed.org/-/media/publications/nfci/nfci-faqs-pdf.pdf", "use": "周频公布与历史修订，不把最新版本回填过去"},
        {"title": "New York Fed Measuring the Natural Rate of Interest", "url": "https://www.newyorkfed.org/research/policy/rstar", "use": "实际中性利率定义，实时与事后估计分离"},
        {"title": "Drechsler Moreira Savov Liquidity and Volatility", "url": "https://www.nber.org/papers/w27959", "use": "承接交易可能暴露于意外信息和波动冲击，不构成510300有效性证明"},
    ]
    save(OUT / "external_references.json", {"methodology_only": True, "new_market_data_downloads": 0, "sources": references})
    price = next(a for a in result["all_account_metrics"] if a["cost"] == "STRESS" and a["policy"] == "PRICE")
    hist = next(a for a in result["all_account_metrics"] if a["cost"] == "STRESS" and a["policy"] == "HISTORY")
    rolling = result["all_rolling_two_years"]
    report = f"""本轮已完成两年滚动、逐日更新的流动性承接研究及8个完整账户。没有找到满足净夏普1.2、年化10%、最大回撤10%三项要求的策略。主宏观账户压力净夏普{p['net_sharpe']:.3f}、年化{p['annualized_return']:.2%}、最大回撤{abs(p['max_drawdown']):.2%}；20万元变为{p['ending_equity']:,.2f}元。结果只对应本轮冻结的方法与数据，不把它推广为所有风险补偿策略的否定。

用户已明确将本轮从单笔事前3:1改为账户尾部风险约束，并要求只使用最近两年训练、每天滚动更新。两年按日历计算，不把504个交易日冒称精确两年。历史规则和失败结果保持原状。当前仍只模拟510300.SH与现金，采集仍暂停，没有生成真实订单。

这次问题是：别人急于退出时，承接库存获得的随后价格修复，是否足以补偿坏消息持续、波动上升及无法立刻平仓的风险。现货买入没有直接收到期权权利金，隐含波动率也不是实际收到的保险费。只有完整账户实际收益为正，并覆盖尾部损失与成本，才能把这种机制转化为有效策略。相关机制依据见[NBER原论文](https://www.nber.org/papers/w27959)。

宏观信息分成三项，避免把所有利率与信用概念混用。社融存量同比的三个月变化描述信用扩张速度变化，它不是新增信用占GDP的信用脉冲，也不等于银行贷款增速。DR007减七日政策利率描述短端资金压力，不是实际利率相对中性利率。平值隐含波动率相对过去20日已实现波动率描述保险定价代理；它没有扣除对未来实现方差的条件预测，因此不宣称已识别真正方差风险溢价。

NFCI覆盖美国金融条件，包含风险、信用、杠杆的相关信息；其历史会随后续资料修订。必须使用当时的发布版本并考虑到中国市场的传导，不能把当前历史曲线直接回填到过去的中国买卖点。本轮没有合格NFCI历史发布版本，未使用它。[Chicago Fed说明](https://www.chicagofed.org/research/data/nfci/about)、[发布和修订规则](https://www.chicagofed.org/-/media/publications/nfci/nfci-faqs-pdf.pdf)。

当r*表示实际中性利率时，政策立场近似应写为i−预期通胀−r*；若采用名义中性利率，才可直接比较名义口径。r*不可直接观测，估计有不确定性，实时估计不能由全样本事后估计替代。本轮未找到合格的中国r*和预期通胀点时资料，因此没有用DR007缺口冒充这一变量。[纽约联储定义与版本说明](https://www.newyorkfed.org/research/policy/rstar)。

每个交易日09:00只使用当时最近两个日历年的成熟五日结果。当天09:30才结束的标签不参与09:00训练。所有对照共享同一信息覆盖，保留缺资料和训练不足期间的现金及旧库存。原点至少252条才更新分布；价格与宏观模型均取固定126个历史近邻，价格包括五日回撤、二十日趋势、短长波动比，宏观模型只再加入上述三项信息。不搜索近邻数，不选择最近两年的历史最优策略。共在{result['unique_updated_days']}个交易日生成{result['daily_model_updates']}组条件分布，首个可用日为{result['first_prediction_date'][:10]}。

账户只在过去五日收益为负时允许新增库存，每天根据新的完整收益分布重新评估持有数量。目标数量按预期收益、方差、调整费用与预留平仓费共同确定。条件五日ES95预算为权益2.5%；ETF下跌10%的单次压力情景预算为权益5%；持仓上限50%；随着距10%回撤线的空间缩小，风险额度同步下降。上述为预先固定的研究假设，不是市场保证。停止交易并不保证能在回撤线成交。T+1、整手、分红、最低佣金和开盘跳价均进入账户。

压力结果中，共同历史分布对照净夏普{hist['net_sharpe']:.3f}、年化{hist['annualized_return']:.2%}；仅价格对照净夏普{price['net_sharpe']:.3f}、年化{price['annualized_return']:.2%}；加入宏观和隐含波动率后净夏普{p['net_sharpe']:.3f}、年化{p['annualized_return']:.2%}。主模型相对价格对照的年化算术收益增量为{comparison['primary_annualized_arithmetic_increment']:.2%}，开发区块95%区间为[{comparison['increment_ci95'][0]:.2%}, {comparison['increment_ci95'][1]:.2%}]。前后两个固定子期方向不一致。成熟预测误差MSE恶化{-comparison['prediction_MSE_improvement']:.2%}，没有支持宏观增量。

主账户净亏损{abs(net_pnl):,.2f}元，其中费用和假设滑点{p['cost_cny']:,.2f}元。沿用完全相同的份额路径加回成本后，损益仍为{gross_same_shares:,.2f}元。这项加回仅用于归因，没有重新优化零成本账户。平均持仓{p['mean_exposure']:.2%}，{p['cash_days']}天收盘无持仓，44次买入调整、81次卖出调整；调整次数不能冒充独立完整交易。完整库存周期{cycle_summary['completed']}个，胜率{cycle_summary['win_rate']:.2%}，实际平均盈利/平均亏损之比{cycle_summary['realized_payoff_ratio']:.3f}。

负偏确实出现：日收益偏度{p['daily_skew']:.3f}，最差一天{p['worst_day']:.2%}。这说明承担了不对称风险，没有证明获得了相应正风险溢价。2022-04-25是最差日，当时社融同比三个月变化约+0.3个百分点、DR007低于政策率，但账户仍亏约2.01%。2025-04-07模型继续增加库存，当天亏约1.22%。这两天既有海外冲击记录均已为真；本轮将其留作背景，没有把它作为新增过滤。因此不能把这些亏损全部解释成当时无信息。旧“遇海外冲击一律避险”用途的失败记录仍保留，不能凭看到这两天就改过滤并重跑。

{len(scored)}个成熟五日预测的下侧5%分位突破率为{tail_frequency:.2%}，但总体覆盖接近5%不足以证明交易选择有优势。选出正预期并承担库存的子集仍会亏损。126个近邻的非重叠区间数中位数为{interpretation['median_nonoverlap_intervals_among126_neighbors']:.0f}，月频信用在两年内也只有约24次更新；不能把日频复制后的数据量当独立宏观样本。单独记录的尾部观察报警有{result['tail_observation_alerts']}个日期，它们相互重叠，不是同样数量的独立规律失效事件。本轮没有把这些观察报警再改成一套启停交易来追逐结果。

完整记录了{rolling['windows']}个高度重叠的滚动两年窗口。主账户夏普中位数{rolling['median_sharpe']:.3f}，最高{rolling['max_sharpe']:.3f}；最好的年化也只有{rolling['max_cagr']:.2%}，同时满足三项目标的窗口为{rolling['simultaneously_met_point_targets']}个。两年训练允许遗忘旧关系，但不能通过挑选两年结果使本轮达标。

验证已复算{verification['saved_distributions_recomputed']}组保存条件分布，核对训练窗口和标签成熟时钟，并验证8个账户的现金、份额、分红应收及权益恒等式。两个历史截断原点保持一致。数据拼接最初遇到pandas纳秒/微秒日期类型不一致，在新标签计算之前做了兼容修复；原冻结代码与参数保留，修复代码另存。期权历史首次发布时间尚未认证，额外滞后一天是保守开发假设，未建立严格实盘时点证明。

本轮结论是：目前可复用的是每日更新、风险预算和失效观察的可执行研究流程；尚无可晋升的高夏普策略。高夏普不是靠选择负偏得到，下一条收益来源必须证明“可获得补偿”超过“条件尾部损失与成本”。本轮不调整两年窗口、近邻数量、宏观方向或风险预算挽救结果，不扩大仓位。未来若补充NFCI或r*，需要独立的历史版本与传导假设；当前采集暂停保持。
"""
    (OUT / "研究结论.md").write_text(report, encoding="utf-8")
    state_path = INTEGRATED / "current_status.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.setdefault("historical_blocker_before_new_user_directions", {"id": state.get("active_blocker_id"), "description": state.get("active_blocker_description")})
    state.update(latest_goal_turn_classification="PROGRESS_USER_REQUESTED_DAILY_TAIL_INSURANCE_COMPLETED",
                 same_condition_consecutive_no_progress_goal_turns=0, goal_achieved=False,
                 latest_user_requested_study=str(OUT.relative_to(ROOT)), latest_user_requested_study_status=result["status"],
                 latest_research_workflow="USER_REQUESTED_EIGHT_LOCAL_ACCOUNTS_COMPLETED",
                 active_blocker_id=None, active_blocker_description=None,
                 latest_daily_tail_insurance_primary=p, latest_daily_tail_insurance_model_updates=result["daily_model_updates"],
                 latest_daily_tail_insurance_rr_requirement=None, training_window_calendar_years=2,
                 remaining_research_question="本轮保险定价代理未改善收益分布或完整账户；需独立新机制或更合格信息证明正补偿，当前未达高夏普目标。",
                 goal_metadata_note="目标工具仍保留此前blocked状态；本轮用户新授权已实际执行，不能据旧阻断计数判定新阻断。")
    save(state_path, state)
    mandate_path = ROOT / "config/510300_existing_data_training_mandate_v1.json"
    mandate = json.loads(mandate_path.read_text(encoding="utf-8"))
    mandate.update(current_round=result["study_id"], current_protocol=str((OUT / "protocol.json").relative_to(ROOT)),
                   latest_progress_receipt=str((OUT / "result.json").relative_to(ROOT)),
                   last_research_result="两年每日更新8账户完成：主压力夏普-0.591、年化-1.20%、回撤7.04%；宏观未形成增量，目标未达。",
                   goal_achieved=False)
    save(mandate_path, mandate)
    integrated_report = INTEGRATED / "最新研究结论.md"
    previous = integrated_report.read_text(encoding="utf-8")
    addition = f"最新进展：用户将本轮单笔3:1改为账户尾部风险，并指定最近两年每日滚动更新。新研究已完成{result['daily_model_updates']}组条件分布和8个完整账户。主压力净夏普{p['net_sharpe']:.3f}、年化{p['annualized_return']:.2%}、回撤{abs(p['max_drawdown']):.2%}；未达标，宏观未提供可靠增量。全部875个滚动两年窗口无达标窗口。目标未完成；旧冻结研究和采集暂停保持。\n\n详见[本轮研究结论](../510300_daily_liquidity_insurance_tail_v1/研究结论.md)。\n\n---\n\n"
    integrated_report.write_text(addition + previous, encoding="utf-8")
    print(json.dumps(normalize({"已完成": "研究结论与综合状态更新", "完整周期": cycle_summary,
                              "最差日期机制": worst[worst_columns].to_dict("records"), "已有背景分层": strata}), ensure_ascii=False))


if __name__ == "__main__":
    main()
