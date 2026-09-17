"""交付单成分退出、全部因素数值及概率与盈亏幅度分离的下一项。"""
import json
import shutil
from research.single_component_exit_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    decision = ("第157轮单成分退出完成。主基础／压力净夏普0.312／0.243，较早0.709／0.685；"
        "年化收益主2.03%／1.52%、较早7.94%／7.64%。四项夏普均低于1.2、128及143，关闭这套固定单成分退出规则，完整目标未实现。")
    detail = ("八项必要测试7.35秒，25组实际输入各拟合一次，89个月复用，保留141个月度时钟、114个可用月份。"
        "训练与四条新账户核心计算7.712011秒，不含实现、测试、核对和交付；全部25组均成功。"
        "已独立用协动矩阵的奇异值分解重建组合方向，逐组核对单分数回归、周期权重、均值尺度及截距，数值误差均在事先设定的容差以内。"
        "66个真实周期、1128个持仓状态、720个预测、5646个收盘判断和132次实际开盘成交，以及全部费用分红都已核对。"
        "\n\n25组模型均保留七项非零因素，恒定进入类别系数为零。持仓年龄、含成本浮盈亏、高点回撤、5日及20日涨跌、120日均线偏离的系数在全部版本均为负，"
        "20日波动的系数有正有负。一个预测成分仍对应八项完整输入、八个方向权重和最终系数，本文后部列出25组全部数值及截距，不能只给一个模型名称。"
        "\n\n主各24个周期、309个持仓收盘、48次成交；较早各9个周期、255个持仓收盘、18次成交，其中51个状态可预测、204个入场时没有成熟模型。"
        "主各17周期触发学习退出、较早各3周期触发学习退出。四账户均完整清仓，没有受阻未成交请求。"
        "主最大回撤12.67%／13.32%，较早13.79%／13.92%。"
        "\n\n主基础净利润28487.21元、压力21083.24元；较早基础93833.75元、压力89725.85元。"
        "相对128，主终值少84066.07／82884.04元，较早少11233.42／18530.80元。"
        "主基础价格利润加分红少83689.00元，费用多377.07元；压力对应价格加分红少82175.00元、费用多709.04元。"
        "较早虽然节省42.38／85.50元费用，但价格利润少11275.80／18616.30元。差距主要来自持仓路径，不能通过忽略费用或单看交易次数解释。"
        "\n\n相对128保存了36条进出日期变化记录，这些包含缺少或新增的进入，并非36笔互相独立的盈利交易。"
        "例如基础账户2020年7月1日进入，退出从7月7日提前到7月6日；2024年9月25日进入，从9月30日提前到9月27日，均错过部分后续上涨。"
        "也出现原策略没有的2026年6月23日进入、7月14日止损退出，实际亏16633.78元。"
        "较早2017年5月18日进入，从原7月21日退出提前到6月27日。单笔利润还受之前复利资金规模影响，不能直接将利润差当纯时机因果效应。"
        "\n\n主两档年化收益低于买入持有1.60／2.08个百分点，较早高4.04／3.77个百分点。"
        "四项夏普和四项年化收益都低于156；相对143四项夏普都低，主年化收益较低、较早年化收益较高。"
        "下一项把占优概率和两类平均盈亏幅度分开估计，避免仅比较胜率或单一线性分数；仍只做一套固定方法，不加数据或参数网格。")
    write_json(OUT / "candidate_outcomes.json", {"recorded_at": now(), "goal_achieved": False,
        "SINGLE_COMPONENT_EXIT": "CLOSED_FOUR_SHARPE_BELOW12_128_143_AND_FOUR_RETURNS_BELOW156"}, exclusive=True)
    next_path = ROOT / "docs/510300_PROBABILITY_PAYOFF_EXIT_NEXT_20260909.md"
    document = ROOT / "deliverables/510300单成分退出_第157轮_20260909/单成分退出_结果及全部中文规则.md"
    delivery = deliver_round(ROOT, OUT, CONFIG, document, "原八因素合成单一分数与固定版本退出",
        "CLOSED_SINGLE_COMPONENT_EXIT_FULL_GOAL_NOT_MET", decision, detail, next_path,
        next_path.read_text(encoding="utf-8").splitlines(), "PROBABILITY_PAYOFF_EXIT_FINITE_CANDIDATE_PREPARED",
        "周期等权估计两类高斯状态概率，分别乘类内平均净增量，固定每笔模型")
    coefficient_path = OUT / "全部已拟合模型系数.md"
    shutil.copy2(coefficient_path, document.parent / coefficient_path.name)
    with document.open("a", encoding="utf-8") as stream:
        stream.write("\n" + coefficient_path.read_text(encoding="utf-8"))
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=158, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=25, planned_new_reference_accounts=0, external_data_required=False)
    index["current_best_four_scenario_comparison_candidate"].update(last_compared_completed_round=157,
        comparison_basis="SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO157")
    write_json(path, index)
    require(document.read_text(encoding="utf-8").count("## 首次拟合：") == 25, "交付未包含25组全部因素参数")
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
