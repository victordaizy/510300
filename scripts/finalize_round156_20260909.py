"""交付稀疏退出结果、所有模型系数及有限的单成分下一项方案。"""
import json
import shutil
from research.sparse_vintage_exit_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    decision = ("第156轮稀疏退出完成。主基础／压力净夏普0.517／0.647，较早0.817／0.794；"
        "年化收益主4.29%／5.26%、较早9.79%／9.48%。四项夏普均低于1.2及143，完整目标未实现，关闭这套固定强度的稀疏退出规则。")
    detail = ("七项必要测试15.09秒；25组实际输入各拟合一次，后来89个月复用已存在模型，保留141个月度时钟和114个可用月份。"
        "训练与四账户核心28.338204秒，不包含实现、测试、核对和交付。全部25组均成功，未使用求解失败替代。"
        "独立复算完整八系数及每周期截距的凸最优条件，最大残差约万亿分之一；56个实际周期、1255个持仓状态、847个预测、5646个收盘判断和112次开盘成交及费用分红均已核对。"
        "\n\n每组模型实际留下1至3项因素：5组留下1项、14组2项、6组3项。含成本浮盈亏在25组均非零，20日涨跌在20组、5日涨跌在6组非零；"
        "持仓年龄、持仓高点回撤、进入类别、120日均线偏离、20日波动在本次拟合中系数均为零。全部原八因素、标准化规则及25组完整系数都在本文后部，未隐藏零系数。"
        "这些是本次训练结果，不能据此认定零系数因素在其他设计中没有价值。"
        "\n\n主基础20周期329个持仓收盘，压力18周期330个；较早各9周期298个持仓收盘，其中94个可预测、204个入场时没有成熟模型。"
        "四账户均完整清仓，没有受阻未成交请求。主最大回撤17.02%／17.23%，较早13.79%／13.92%。"
        "相对128，主终值减少48276.34／23089.55元、净夏普减少0.366／0.184；较早终值增加15011.61／7335.89元、净夏普增加0.051／0.014。"
        "四段年化收益均高于买入持有，但四项夏普均低于143。相对143，只有主基础年化收益较低，其余三项年化收益较高，因此不能把年化收益与夏普的比较混用。"
        "\n\n压力费用下主区间收益反而更高，原因已经从保存结果对上：本轮浮盈亏的25组系数全部为负，较高费用降低本账户浮盈亏输入，会提高同一版本给出的继续持有预测。"
        "386对同日进入、同版本的可用预测差，均完全来自这一因素；其中4对预测在零的两侧。主区间3笔共同进入的退出日期因此不同，后续另有2次进入仅发生在基础账户。"
        "例如2026年5月6日进入，基础5月13日退出，压力6月23日退出；压力账户还持有上一笔，所以没有基础账户6月5日进入、7月20日退出并亏16468.51元的那一笔。"
        "较早区间两档费用的进出日期一致。该现象来自费用对预测和后续交易路径的反馈，不能解释为同一固定成交路径增加费用能增加利润。"
        "周期损益包含当时复利后的实际资金规模，也不能将单笔利润差直接当作纯退出时机因果效应。"
        "\n\n下一项只检验一个加权单成分预测方法：把原八因素与继续收益的周期内共同变化形成一个分数，减少独立拟合方向。"
        "原成熟样本、月度时钟、固定每笔模型和真实账户继续复用；不补外部资料、不试惩罚强度网格。")
    require((OUT / "saved_cost_feedback_receipt.json").is_file(), "费用反馈说明未保存")
    write_json(OUT / "candidate_outcomes.json", {"recorded_at": now(), "goal_achieved": False,
        "SPARSE_VINTAGE_EXIT": "CLOSED_FOUR_SHARPE_BELOW12_AND143_MAIN_WORSE_EARLY_BETTER_THAN128"}, exclusive=True)
    next_path = ROOT / "docs/510300_SINGLE_COMPONENT_EXIT_NEXT_20260909.md"
    document = ROOT / "deliverables/510300稀疏退出_第156轮_20260909/稀疏退出_结果及全部中文规则.md"
    delivery = deliver_round(ROOT, OUT, CONFIG, document, "原八因素稀疏学习与固定版本退出",
        "CLOSED_SPARSE_VINTAGE_EXIT_FULL_GOAL_NOT_MET", decision, detail, next_path,
        next_path.read_text(encoding="utf-8").splitlines(), "SINGLE_COMPONENT_EXIT_FINITE_CANDIDATE_PREPARED",
        "固定一个由周期内因素与目标协动形成的预测成分，复用成熟样本和真实进出场")
    coefficient_path = OUT / "全部已拟合模型系数.md"
    shutil.copy2(coefficient_path, document.parent / coefficient_path.name)
    with document.open("a", encoding="utf-8") as stream:
        stream.write("\n" + coefficient_path.read_text(encoding="utf-8"))
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=157, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=25, planned_new_reference_accounts=0, external_data_required=False)
    index["current_best_four_scenario_comparison_candidate"].update(last_compared_completed_round=156,
        comparison_basis="SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO156")
    write_json(path, index)
    require(document.read_text(encoding="utf-8").count("## 首次拟合：") == 25, "交付未包含25组完整因素系数")
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
