"""记录新增价格传导信息实际怎样改变账户，保留失败与不确定性。"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import research.manufacturing_price_transmission_daily_v1 as study
from research.selected_mix_reappraisal_v1 import read, save, now, digest
from research.strategy_review_diagnostics_v1 import metrics

MAIN = ROOT / "reports/research/510300_integrated_research_continuation_20260924"


def decision_effects(result):
    frames, signatures, duplicates, differences = {}, {}, [], []
    for item in result["all_accounts"]:
        path = study.OUT / "accounts" / item["cost"] / item["policy"]
        ledger = pd.read_parquet(path / "ledger.parquet")
        fresh = metrics(ledger)
        for key in ["end_equity", "profit", "annual_return", "sharpe", "max_drawdown", "mean_exposure"]:
            if fresh[key] is None or item[key] is None:
                assert fresh[key] is None and item[key] is None
            else:
                np.testing.assert_allclose(fresh[key], item[key], atol=1e-10, rtol=0)
        key = (item["policy"], item["cost"])
        signature = hashlib.sha256(ledger[["equity", "shares", "cash", "net_return"]].to_numpy(float).tobytes()).hexdigest()
        if signature in signatures:
            duplicates.append({"path": key, "same_as": signatures[signature]})
        else:
            signatures[signature] = key
        frames[key] = ledger
    for cost in study.distribution.COSTS:
        base = frames["BASE__FIXED20_NO_ADD", cost]
        for arm in ["MEAN_ONLY", "TAIL_ONLY", "JOINT"]:
            left = frames[f"{arm}__FIXED20_NO_ADD", cost]
            assert left.date.equals(base.date)
            differences.append({"arm": arm, "cost": cost, "changed_position_days": int(left.shares.ne(base.shares).sum()),
                "changed_fill_days": int(left.filled_quantity.ne(base.filled_quantity).sum()),
                "max_abs_equity_difference": float((left.equity - base.equity).abs().max())})
    boundaries = []
    cost = study.distribution.COSTS["STRESS"]
    for arm in ["BASE", "TAIL_ONLY", "JOINT"]:
        decisions = pd.read_parquet(study.OUT / "accounts/STRESS" / f"{arm}__FIXED20_NO_ADD" / "decisions.parquet")
        rows = decisions[decisions.shares_before_decision.eq(0) & decisions.pressure5.gt(0) & decisions.reference_price.notna()]
        counts = {"eligible_flat_decisions": len(rows), "positive_target_decisions": int(rows.target_shares.gt(0).sum()),
                  "next_lot_tail_blocked": 0, "next_lot_gap_blocked": 0, "next_lot_position_blocked": 0,
                  "next_lot_tail_only_blocked": 0, "next_lot_all_risk_feasible": 0}
        for r in rows.to_dict("records"):
            quantity, ref = int(r["target_shares"]) + 100, float(r["reference_price"])
            limits, _ = study.parent.adjusted_limits(study.engine, quantity, ref, r, cost)
            price = study.engine.fill_price(ref, -1, cost, .001)
            exit_cost = quantity * (ref - price) + study.engine.commission(quantity, price, cost)
            notional = quantity * ref
            tail = notional * limits["es95"] + exit_cost > limits["es_budget_cny"] + 1e-8
            gap = .1 * notional + exit_cost > limits["gap_budget_cny"] + 1e-8
            position = notional > limits["position_budget_cny"] + 1e-8
            counts["next_lot_tail_blocked"] += int(tail)
            counts["next_lot_gap_blocked"] += int(gap)
            counts["next_lot_position_blocked"] += int(position)
            counts["next_lot_tail_only_blocked"] += int(tail and not gap and not position)
            counts["next_lot_all_risk_feasible"] += int(not (tail or gap or position))
        boundaries.append({"arm": arm, **counts})
    raw = pd.read_parquet(study.OUT / "results/predictions.parquet")
    q = raw[raw.scope.eq(study.MAIN_SCOPE)].pivot(index="idx", columns="model", values="es95_5")
    risk_changed_days = int((~np.isclose(q[study.SIGNAL], q[study.CONTROL], atol=1e-14, rtol=0)).sum())
    return {"at": now(), "distinct_paths": len(signatures), "duplicates": duplicates, "account_differences": differences,
        "risk_predictions_differ": risk_changed_days > 0, "risk_prediction_changed_days": risk_changed_days,
        "entry_next_lot_budget_diagnostics": boundaries,
        "meaning": "相同实际账户状态下检查预定目标再加100份的风险约束；不是放宽预算后的新策略，不证明所有未交易日都由风险限制导致。",
        "new_accounts": 0, "new_parameter_searches": 0}


def run():
    if (study.OUT / "completed_round.json").exists():
        raise RuntimeError("成本传导研究已经汇总，不覆盖完成记录。")
    result = read(study.OUT / "result.json")
    source = read(study.source.OUT / "result.json")
    effect = decision_effects(result)
    save(study.OUT / "results/decision_effect_diagnostic.json", effect, True)
    p = result["primary"]
    accounts = {r["policy"].split("__")[0]: r for r in result["all_accounts"] if r["cost"] == "STRESS"}
    increments = {r["arm"]: r for r in result["comparisons"] if r["purpose"] == "INCREMENT_VS_BASE" and r["cost"] == "STRESS"}
    inc = increments["JOINT"]
    schedule = next(r for r in result["schedules"] if r["scope"] == study.MAIN_SCOPE)
    sensitivity = next(r for r in result["schedules"] if r["scope"] == study.METHOD_SCOPE)
    evaluation = {(r["scope"], r["model"], r["horizon"], r["subset"]): r for r in result["forecast_evaluation"]}
    mse_improvement = 1 - evaluation[study.MAIN_SCOPE, study.SIGNAL, 20, "ALL"]["MSE"] / evaluation[study.MAIN_SCOPE, study.CONTROL, 20, "ALL"]["MSE"]
    tail_improvement = 1 - evaluation[study.MAIN_SCOPE, study.SIGNAL, 5, "ALL"]["quantile_loss"] / evaluation[study.MAIN_SCOPE, study.CONTROL, 5, "ALL"]["quantile_loss"]
    tail_effect = next(r for r in effect["account_differences"] if r["arm"] == "TAIL_ONLY" and r["cost"] == "STRESS")
    attribution = result["primary_cash_attribution"]
    monitors = {r["horizon"]: r for r in result["mature_monitors"] if r["scope"] == study.MAIN_SCOPE and r["model"] == study.SIGNAL}
    latest = source["latest_source"]
    paragraphs = []
    labels = {"BASE": "价格资金基准", "MEAN_ONLY": "只更新收益均值与方差", "TAIL_ONLY": "只更新五日尾部预算", "JOINT": "两者同时更新（主方案）"}
    for arm in study.ARMS:
        row = accounts[arm]
        text = f"{labels[arm]}：压力净夏普{study.mechanics.sharpe_text(row['sharpe'])}，年化收益{row['annual_return']:.4%}，最大回撤{abs(row['max_drawdown']):.4%}，期末权益{row['end_equity']:,.2f}元。"
        if arm != "BASE":
            diff = increments[arm]
            text += f" 相对基准年化算术收益差{diff['daily_blocks']['annual_arithmetic_difference']:.4%}，月报区块95%区间[{diff['release_blocks']['lower_95']:.4%}, {diff['release_blocks']['upper_95']:.4%}]。"
        paragraphs.append(text)
    account_text = "\n\n".join(paragraphs)
    report = f"""本轮完成制造业采购成本与出厂价格传导环境的一项新信息检验：116个月原始报告、8个完整账户，实际{effect['distinct_paths']}条不同净值路径。主压力账户净夏普{study.mechanics.sharpe_text(p['sharpe'])}、年化收益{p['annual_return']:.4%}、最大回撤{abs(p['max_drawdown']):.4%}。共同目标通过{result['joint_target_pass_accounts']}个；独立前向结果仍为0，总目标保持active，尚未完成。

上一轮住房资料与12个账户已完成，属于实质进展。本轮先核对旧PMI新订单、LPR变化和股债联合响应的失败终态，没有重启其参数搜索。新的唯一宏观信息是当次制造业出厂价格扩散指数减主要原材料购进价格扩散指数，研究它能否帮助区分价格压力下的后续修复与持续恶化。它属于成本传导环境的候选测量，不是市场共识意外，也不是实际利润率。

[国家统计局编制方法](https://www.stats.gov.cn/zs/tjws/zytjzbqs/cgzlzs/202501/t20250121_1958396.html)把这两项归为扩散指数。差值衡量两类涨价普遍程度的差异；不等于售价减成本的金额、企业毛利率或PPI涨跌幅。调查针对制造业，不能代表沪深300全部行业。经济假设是成本上行而销售端难以传导时，经营压力可能更持久；价格下降也可能伴随政策缓解或需求修复，模型没有预先锁定正负交易方向。

复用115份已保存当月原报告，只新取得2026年8月报告和官方方法页，共116个月，覆盖2017-01至2026-08，字段缺口0。2017年原报告明确出厂价格指数从当年1月起发布，2015—2016保持缺失，没有从后报旧月份行回填。每份只提取该次统计月对应行，桌面/移动重复表逐值一致。重新核对原页公布时钟，统一当日日末可用；仅日级日期的原页没有伪造日内时刻。原始网页快照仍未认证为不可变首版，全部结果属于开发证据。

原文显示，2019年1月切换行业分类版本，2022年4月制造业样本数从3000变为3200。主研究使用各期实际公布的同名季调指数，并保留样本变化限制；另按相同分类版本和样本规模形成严格历史池，只评估预测敏感性，没有另外运行账户。季调序列可能修订，所以始终不拿未来报告中的历史行覆盖旧值。

四种账户的区别在于新增信息进入哪个决策模块：收益模块包括20日均值和方差，尾部模块使用每天更新的五日ES95；主方案同时更新两者。所有账户共用相同信息日和成熟训练池，每天只训练最近两个日历年，来源公布也在两年内；20日退出开盘必须严格早于当前决策。最低252行、固定126近邻、训练内标准化clip5。20日持有期在查看收益前固定，持仓不加仓，尾部预算可提前减仓，缺预测满5日请求退出。

共{result['unique_prediction_days']:,}个预测日，主训练池{schedule['training_rows_min']}—{schedule['training_rows_max']}个日原点，但每池只有{schedule['release_months_min']}—{schedule['release_months_max']}个不同月报。首次主预测为{schedule['first_prediction'][:10]}，完整账户从2020-01-02到2026-09-24保留全部现金日和交易日。重叠日原点与重复月报不能被当作独立样本。

{account_text}

主方案相对基准的20日区块95%收益差区间为[{inc['daily_blocks']['lower_95']:.4%}, {inc['daily_blocks']['upper_95']:.4%}]，完整月报公布间隔区间为[{inc['release_blocks']['lower_95']:.4%}, {inc['release_blocks']['upper_95']:.4%}]。两个事前固定时期2020—2023和2024—末端的增量分别为{inc['fixed_periods'][0]['annual_arithmetic_difference']:.4%}、{inc['fixed_periods'][1]['annual_arithmetic_difference']:.4%}。区块没有校正全部历史反复选择；月报间隔抽样也不能保留相邻月份的全部持续性。

新增信息的20日均值MSE相对共同基准改善{mse_improvement:.4%}，五日5%分位损失改善{tail_improvement:.4%}，负数表示变差。严格同方法的预测敏感性条件通过状态为{result['method_prediction_sensitivity_pass']}，最后状态为{sensitivity['last_status']['status']}。预测变化、风险预算变化和实际账户收益是三个不同层面，不能互相替代。完整预定开发增量条件为{result['development_increment_gate']}，主账户滚动两年共同目标通过{result['primary_rolling_two_year_joint_passes']}次。

尾部预测相对基准在{effect['risk_prediction_changed_days']}个日子发生变化；只更新尾部预算的压力账户，实际仓位有{tail_effect['changed_position_days']}天不同、成交数量有{tail_effect['changed_fill_days']}天不同。这区分了“预测不同但账户不变”和“实际交易改变后是否改善收益”。另保存同一账户状态下目标再加100份会触及哪项预算的诊断，没有放宽预算或生成新策略。四账户的交互收益差仅作机制拆分，不能当作可交易组合或把四条净值相加。

主20万元账户毛损益{attribution['gross_pnl']:,.2f}元，费用与退出储备{attribution['cost_and_reserve']:,.2f}元，净损益{p['profit']:,.2f}元，平均敞口{p['mean_exposure']:.4%}。T+1、100份、现金分红分账、压力费用、最多50%目标仓位、五日ES和跳空/回撤预算均执行。保存分布重算、未来及过期标签扰动、四种输入组合和完整账户核算均通过。尾部估计不保证实际损失上限。

五日监控有{monitors[5]['nonoverlap_rows']}个成熟非重叠原点、{monitors[5]['assessable_rows']}个可评估时点、{monitors[5]['alerts']}个警报；20日分别为{monitors[20]['nonoverlap_rows']}、{monitors[20]['assessable_rows']}、{monitors[20]['alerts']}，状态{monitors[20]['status']}。未报警不能证明收益优势。

最新原报告是{latest['stat_month']}，公布时间{latest['published_at']}，购进价格扩散指数{latest['input_price_diffusion']:.1f}、出厂价格{latest['output_price_diffusion']:.1f}，差值{latest['output_minus_input_diffusion_pp']:.1f}个百分点。这是已公布的宏观事实，不是当前交易指令。本轮来源、训练、8个账户和结果登记均已结束，current_market_view仍为NO_VIEW。没有真实订单、审核包或用户表格。后续研究保留本轮结果与目标门槛。
"""
    report_path = study.OUT / "本轮研究结论.md"
    report_path.write_text(report, encoding="utf-8")
    status = "COMPLETED_MANUFACTURING_PRICE_TRANSMISSION_EIGHT_ACCOUNTS_NO_QUALIFIED_STRATEGY" if not result["joint_target_pass_accounts"] else "COMPLETED_MANUFACTURING_PRICE_TRANSMISSION_ACCOUNTS_VALIDATION_PENDING"
    classification = "PROGRESS_MANUFACTURING_COST_TRANSMISSION_EIGHT_ACCOUNT_FACTORIAL"
    receipt = {"at": now(), "study_id": study.STUDY, "status": status, "classification": classification,
        "source_months": 116, "new_accounts": 8, "distinct_paths": effect["distinct_paths"], "duplicate_paths": effect["duplicates"],
        "primary": p, "joint_target_pass_accounts": result["joint_target_pass_accounts"], "primary_increment": inc,
        "tail_account_effect": tail_effect, "result_sha256": digest(study.OUT / "result.json"), "report_sha256": digest(report_path),
        "decision_effect_sha256": digest(study.OUT / "results/decision_effect_diagnostic.json"),
        "goal_status": "active", "goal_achieved": False, "new_independent_forward_observations": 0,
        "current_market_view": "NO_VIEW", "orders_authorized": False}
    save(study.OUT / "completed_round.json", receipt, True)
    state = read(MAIN / "current_status.json")
    state.update(updated_at=now(), goal_status="active", goal_achieved=False,
        latest_goal_turn_classification=classification, previous_goal_turn_classification="PROGRESS_HOUSING_BREADTH_TWO_HORIZONS_AND_REBASE_SENSITIVITY",
        same_condition_consecutive_no_progress_goal_turns=0, active_blocker_id=None, active_blocker_description=None,
        latest_user_requested_study=study.OUT.relative_to(ROOT).as_posix(), latest_user_requested_study_status=status,
        latest_manufacturing_price_transmission_round=receipt,
        next_evidence_lead={"study": study.STUDY, "status": "SOURCES_AND_EIGHT_ACCOUNTS_COMPLETED"},
        remaining_research_question="成本传导信息已按收益模块/尾部预算拆分8个账户；下一步根据实际决策变化寻找新的可检验优势，不能把预测改善当成账户优势，也不调整失败规则营救结果。",
        goal_metadata_note="本轮完成116个月原字段和8账户作用拆分，具备实质进展；高夏普与独立验证仍未建立。")
    for name in [study.source.STUDY, study.STUDY]:
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
        last_research_result=f"成本传导116个月、8账户/{effect['distinct_paths']}条不同路径完成；主压力夏普{study.mechanics.sharpe_text(p['sharpe'])}、年化{p['annual_return']:.4%}、回撤{abs(p['max_drawdown']):.4%}；总目标active。",
        goal_status="active", goal_achieved=False)
    save(mandate_path, mandate)
    status_path = ROOT / "RESEARCH_STATUS.md"
    marker = "<!-- MANUFACTURING_PRICE_TRANSMISSION_ROUND_20260926 -->"
    previous = status_path.read_text(encoding="utf-8")
    assert marker not in previous
    status_path.write_text(f"{marker}\n\n> 2026-09-26 成本传导116个月、8账户/{effect['distinct_paths']}条不同路径完成；共同目标通过{result['joint_target_pass_accounts']}个，独立前向0，总目标ACTIVE。见[本轮结论]({report_path.relative_to(ROOT).as_posix()})。\n\n" + previous, encoding="utf-8")
    assert (MAIN / "最新研究结论.md").read_bytes() == report_path.read_bytes()
    print(f"成本传导研究已登记，8账户/{effect['distinct_paths']}条不同路径，总目标保持active。", flush=True)


if __name__ == "__main__":
    run()
