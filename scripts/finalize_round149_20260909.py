"""交付月度最弱增长预算结果，并登记一次比较两种单向预算的下一步。"""
import json
from research.robust_block_growth_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第149轮结束，不替换143。主历史基础／压力夏普为0.600／0.538，年化2.37%／2.12%，最大回撤7.26%／7.93%；"
        "较早历史夏普0.961／0.983，年化4.85%／5.05%，最大回撤3.95%／4.03%。四项夏普均低于143，主历史年化收益也低于买入持有，完整目标未实现。")
    detail = ("六项必要测试4.24秒；八十一加六十一次月度检查中，完成六十八加四十八次预算优化，共116次，两档费用共享而不重复计数。"
        "四新账户核心计算2.085329秒，零新收益预测模型、参考账户和外部下载；核心时间不含开发、测试、保存核对和文档。"
        "独立有界优化器与逐日标量对数求和核对116次最优性，最大目标值差为零；同时核对142个时点、完整242日窗口、81／81／80分段、初始143、平局状态、月内保持及下一开盘。"
        "账户核对覆盖5646个判断、四条资金账户和64个完整实际周期。主各22周期329个持仓收盘74次成交；较早各10周期、294／300个持仓收盘、各37次成交。"
        "主131预算均值20.79%、583个判断为正；较早均值14.60%、178个判断为正，最大都到100%。这次确实改变了预算和实际路径，未出现缺失目标或未完成终点退出。"
        "相对143，主基础价格利润少34393.60元、分红多2053.90元、费用多430.31元，净少32770.01元；压力净少32934.96元。"
        "较早基础／压力净少4962.26／4999.59元。较早回撤略减，仍不足以补偿收益下降。以上为全账户差额，含重新取整和自身净值变化。"
        "结果说明该固定历史窗口下的最弱阶段增长预算没有带来后续增益；不调整分段、窗口或目标函数救回。下一轮一次比较两种信号段内单向预算，缩短重复搭建流程。")
    next_path = ROOT / "docs/510300_MONOTONE_EPISODE_BUDGET_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300最弱阶段增长预算_第149轮_20260909/最弱阶段增长_结果及全部中文规则.md",
        "每月优化近期三个子阶段的最弱增长", "CLOSED_FOUR_SHARPES_BELOW143_MAIN_EXCESS_NEGATIVE",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "MONOTONE_EPISODE_BUDGET_BATCH_IMPLEMENTATION", "共享数据与账户流程，一次比较143正信号段内预算只减不增和只增不减两个固定机制")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=150, registered=False, planned_settings=2, planned_new_accounts=8)
    index["current_best_four_scenario_comparison_candidate"].update(last_compared_completed_round=149,
        comparison_basis="SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO149")
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
