"""保留固定各半在主历史的互补，并准备使用历史信息的策略选择。"""
import json
from research.equal_reference_pair_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第134轮主历史有互补改善，完整目标仍未实现。主基础／压力净夏普1.114／1.053，年化4.31%／4.06%，最大回撤4.22%／4.60%；"
        "较早净夏普0.795／0.788，年化4.82%／4.80%，最大回撤8.09%／8.21%。四整段年化收益都高于买入持有，"
        "超额分别0.68、0.46、0.93、0.93个百分点，但仍未证明逐年稳定超额。"
        "主净夏普比132提高0.272／0.264，较早比132低0.135／0.165；四情景最低0.788430，略低于132的0.788877，"
        "不能因都四舍五入到0.789便宣称新的最优。保留主历史互补的比较，不改变本轮各半比例、参考或按事后年份选择来挽救。")
    detail = ("六项必要合成测试2.66秒，四条新账户核心1.0443572秒，零新模型、零新参考；核心时间不含开发、测试、核对和文档。"
        "已独立核对固定平均目标、5646个实际判断、四条账户及72个完整周期，共用核对模块没有重跑旧账户。"
        "主两费用各20周期、341个持股收盘、69次成交；较早各16周期、339个持股收盘、49次成交。"
        "主有352个正目标原点，实际341个持股收盘；正目标不必然达到可购买的整手，且终点另有开盘退出，不能把来源意向计数当成交计数。"
        "相对132，主基础净盈利少13607.14元，但绝对佣金滑点减少2003.26元；新组合降低收益同时更大幅度降低波动，因此夏普提高。"
        "主基础年化算术收益4.294%，波动3.856%，132对应5.162%与6.132%。"
        "较早基础净盈利少19536.97元、压力少23559.81元，构成早期表现的代价。"
        "相对91，主基础净盈利多15591.53元、早基础多22199.58元；各半提高91的收益机会，同时承受132的更多波动。"
        "下一项复用既有月度历史排名控制器，用当时已经发生的基础费用收益判断132、91或现金，检验阶段差异能否提前利用。")
    next_path = ROOT / "docs/510300_RECENT_REFERENCE_SELECTION_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300两参考固定各半组合_第134轮_20260909/固定各半组合_结果及全部中文规则.md",
        "下行风险与连续两策略目标各半组合", "COMPLETED_EQUAL_REFERENCE_PAIR_MAIN_GAIN_EARLY_WEAKER_NOT_TARGET",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "RECENT_PERFORMANCE_REFERENCE_SELECTION_PLANNED", "复用月首242日历史排名控制器，在132、91和现金间选择，检验可知的阶段适应")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=135, registered=False, planned_settings=1, planned_new_accounts=4)
    index["additional_equal_reference_main_comparison_candidate"] = {
        "study": "510300_EQUAL_REFERENCE_PAIR_V1", "model": "EQUAL_REFERENCE_PAIR",
        "status": "MAIN_LOCAL_GAIN_VS132_EARLY_WEAKER_NOT_INDEPENDENTLY_VALIDATED",
        "source_result": str((OUT / "result.json").relative_to(ROOT)),
        "base_main_sharpe": 1.1136205013855784, "stress_main_sharpe": 1.0532395592480046,
        "base_earlier_sharpe": .7953223204246888, "stress_earlier_sharpe": .7884304862205913,
        "minimum_four_scenario_sharpe": .7884304862205913, "four_full_period_excess_returns_positive": True,
        "goal_achieved": False}
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
