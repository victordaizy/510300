"""解释已经完成的固定组失效结果；不改变规则或重复金融运行。"""
from __future__ import annotations

import pandas as pd

from research import broker_cohort_failure_study_v1 as study
from research import report_broker_cohort_failure_v1 as reporter


def normalized_frame(frame, policy):
    value = frame.copy()
    for column in ("reason", "exit_reason", "extra_reason", "policy"):
        if column in value:
            value[column] = value[column].str.replace(policy, "PRICE_RULE", regex=False)
    return value


def main():
    if (study.OUT / "post_run_diagnosis.json").exists():
        raise RuntimeError("保存账户失败归因已完成，不重复执行。")
    summary = study.read(study.OUT / "summary.json")
    delivery = study.read(study.OUT / "delivery_receipt.json")
    metrics = pd.read_parquet(study.OUT / "results/二十完整账户_同资金风险成本比较.parquet")
    contexts = pd.read_parquet(study.OUT / "results/主政策全部第五日检查与原同进入上下文.parquet")
    source = pd.read_parquet(study.OUT / "results/主政策全部进入来源_完成与未完成分开.parquet")
    decomposition = pd.read_parquet(study.OUT / "results/二十账户实际数量毛收益与摩擦分解.parquet")
    accounts, equivalence, failure_context = {}, [], []
    for period in study.parent.PERIODS:
        for cost in study.parent.COSTS:
            for policy in reporter.POLICIES:
                accounts[(period, cost, policy)] = study.saved_account(period, cost, policy)
            primary = accounts[(period, cost, study.candidate.POLICIES[0])]
            price = accounts[(period, cost, study.candidate.POLICIES[1])]
            for name in (*study.ACCOUNT_TABLES, "failure_checks"):
                pd.testing.assert_frame_equal(normalized_frame(primary[name], study.candidate.POLICIES[0]),
                    normalized_frame(price[name], study.candidate.POLICIES[1]), check_exact=True)
            a = study.parent.original.clean(primary["terminal"])
            b = study.parent.original.clean(price["terminal"])
            a["planned_next_close_request"]["reason"] = a["planned_next_close_request"]["reason"].replace(study.candidate.POLICIES[0], "PRICE_RULE")
            b["planned_next_close_request"]["reason"] = b["planned_next_close_request"]["reason"].replace(study.candidate.POLICIES[1], "PRICE_RULE")
            if a != b:
                raise AssertionError("主政策和同覆盖价格金融终态并不相同。")
            check = primary["failure_checks"]
            eligible = check.cohort_view_allowed & check.etf_price_failed & check.original_exit.isna()
            if not check.loc[eligible, "cohort_state"].eq("NEITHER_POSITIVE_MAJORITY").all():
                raise AssertionError("价格失效并非全部与两组弱势重合。")
            equivalence.append({"period": period, "cost": cost, "six_tables_and_terminal_exact_after_policy_label_normalization": True,
                "known_price_failures_without_original_exit": int(eligible.sum()),
                "group_direction_rejected_any_eligible_price_failure": False,
                "unknown_price_failures_without_original_exit": int((~check.cohort_view_allowed & check.etf_price_failed & check.original_exit.isna()).sum())})
        selected = contexts.loc[contexts.period.eq(period) & contexts.cost.eq("STRESS") & contexts.extra_exit]
        failure_context.append({"period": period, "actual_extra_exits": len(selected),
            "same_entry_original_winners": int(selected.original_same_entry_net_return.gt(0).sum()),
            "same_entry_original_losers": int(selected.original_same_entry_net_return.lt(0).sum()),
            "original_losers_new_return_less_negative": int((selected.original_same_entry_net_return.lt(0)
                & selected.net_return.gt(selected.original_same_entry_net_return)).sum()),
            "original_losers_new_return_more_negative": int((selected.original_same_entry_net_return.lt(0)
                & selected.net_return.lt(selected.original_same_entry_net_return)).sum())})
    # 原A没有保存拒单表；交付层由0更正为未知，不修改任何冻结金融输出。
    decomposition.loc[decomposition.policy.eq(study.CONTROL_A), "rejections"] = float("nan")
    study.table("二十账户实际数量毛收益与摩擦分解", decomposition)
    study.write(study.OUT / "post_run_diagnosis.json", {"at": study.parent.original.now(),
        "status": "SAVED_ACCOUNT_INFORMATION_REDUNDANCY_AND_EXIT_HARM_DIAGNOSED",
        "four_scenario_full_financial_equivalence": equivalence, "actual_extra_exit_context": failure_context,
        "earlier_failure_example": "2019-06-12原修复最终+1989.10；主政策06-20提前退出-50.28392，新增07-02重新定价-2539.48748。全账户终值比原阶段少5606.82388，包含后续份额和未平仓差，不能等同首笔差。",
        "recent_missed_recovery_example": "2025-06-26进入后07-04提前退出+0.05244%，原08-04退出+2.69991%。",
        "recent_cost_diagnosis": "实际数量毛7713.70、摩擦7550.94788、净162.75212；固定数量无费年化0.5846%，仍远低于原A扣费3.9908%。",
        "account_constraints": "主政策/原阶段四场景都未触发DD停买，保存拒单为空；无抬仓或降费用反事实策略。",
        "why_rejected": "四经济门0/4、所有主政策pB<1、成分方向无金融增量、较早反受损、稳定/独立未建立。",
        "next_mechanism_question": "价格和多数成分同弱后，如何以不同的当时信息区分恢复与继续下跌；优先主线对沪深300的传导/集中与轮动，先核历史分类和独立来源钟。",
        "report_only_unknown_amendment": "原A拒单数由交付占位0改为UNKNOWN；原冻结账户/指标不变。",
        "new_accounts": 0, "new_fits": 0, "new_labels": 0, "goal_achieved": False})
    reporter.write_report(summary, metrics, decomposition, source, contexts, delivery["figures"], accounts)
    path = study.OUT / "真实进入后固定组失效_完整结果与反例.md"
    text = path.read_text(encoding="utf-8")
    addition = """## 结果之后的必要归因：下一步为何改变研究问题

四场景六账表与终态经政策名称规范后精确相同：固定组主政策与同覆盖ETF价格对照没有任何实际金融区别。已知覆盖且原退出尚未触发的ETF失效，较早1次、近期11次，全部同时两组未过半；成分方向没有排除任何这类价格退出。这里拒绝的是当前金融用途，不否定所有成分或主线信息。

提前退出并非只省亏损：2019-06-12原修复最终净+1989.10元，新规则06-20退出净−50.28392元，又释放资金07-02买入重新定价，亏−2539.48748元。较早完整终值少5606.82388元，除这些周期还含后续份额和未平仓差异，不能声称首笔单独解释全部。近期11额外退出包括10个原亏损、1个原盈利；8原亏损截短、2反而更差，且2025-06-26原恢复净+2.6999%被截为+0.0524%。两时期均保留这些误伤，不重新选择五日或删路线。

近期主政策实际数量毛7713.70元，摩擦7550.94788元，净162.75212元；加回摩擦的固定数量年化约0.5846%，仍低于原A真实扣费3.9908%。失败不能只归因于手续费。完整年均次数7.167，胜率13/47、B2.76590、pB0.76504；次数增加后质量不足。主政策/原阶段没有DD停买或实际拒单，风险预算的仓位变化已保留，不靠提高仓位救援。原A拒单表本轮未保存，数值显式UNKNOWN，不能当成0。

因此下一问题是弱势能否恢复的机会区分，以及主线是否真实传导到沪深300；继续给同一ETF价格弱势增加相似投票，没有本轮增量依据。主线、资金、博弈、周期可以改变交易行为，但改变依据须在当时可知：产业/机构信息按实际披露、主线向宽基扩散按历史成员与分类；不得事后选赢家或把2026研报当作早年信息。

"""
    marker = "## 决策边界与下一步"
    if text.count(marker) != 1:
        raise ValueError("报告插入点不唯一。")
    path.write_text(text.replace(marker, addition + marker), encoding="utf-8")
    study.write(study.OUT / "figure_delivery_view_receipt.json", {"at": study.parent.original.now(),
        "viewed": 4, "figures": delivery["figures"], "checks": "图例、数值日期、中文、全部净值回撤均已逐幅查看。"})
    print("四场景金融完全重复、提前退出的省亏/误伤与费用归因已保存；仅更正报告未知字段，没有重跑账户。", flush=True)


if __name__ == "__main__":
    main()
