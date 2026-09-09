"""交付继续预测预算的较早改善与主历史退步，结束此固定公式。"""
import json
from research.continuation_strength_blend_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第144轮未改善四情景的均衡结果，不替换143，结束这项继续持有预测幅度公式。主基础／压力夏普1.043／0.983，"
        "年化4.00%／3.78%，最大回撤4.58%／4.67%；较早夏普1.085／1.106，年化5.70%／5.94%，最大回撤4.19%／4.34%。"
        "较早两项夏普均提高约0.063，但主两项均降低约0.249，四情景最低值从143的1.022降至0.983。"
        "四整段年化收益高于买入持有，不代表逐年稳定超额或达到完整夏普1.2目标。")
    detail = ("六项必要测试7.96秒，四账户核心2.370361秒；核心时间不含开发、测试、核对及文档。零新增收益预测模型、参考账户或下载。"
        "直接连接原参考当日实际收盘份额，以对数持仓天数独立还原年龄、直接20日收益窗口还原风险幅度，核对5646个判断和四账户60个实际周期。"
        "主每档20周期、339个持仓收盘、84次成交；较早每档10周期、302／308持仓收盘、39次成交。所有目标已知、无受阻请求，终点清仓。"
        "主每档313个持仓预测可用，其中284个预测为正；较早72／78个可用，其中65／70个为正，另外各204个持仓收盘明确没有成熟模型。"
        "这些无模型状态使用143目标，同时保留预测缺失，没有补零预测。"
        "新增131预算的主平均5.79%／5.85%、较早2.16%／2.21%，只是策略预算；实际平均股票敞口为主8.83%／8.94%、较早12.05%／12.46%。"
        "相对143，主基础价格损益少7384.50元、分红多948.70元、费用多619.23元，净少7055.03元；主压力净少6946.46元。"
        "较早基础价格收益多5778.70元、费用多126.33元，净多5652.37元；压力净多6090.64元。主风险和交易成本同时增加，较早改善不足以弥补主退步。"
        "两费用的持仓预算和取整路径分别计算，压力费用下收益更高不能解释为加费有效。"
        "因固定公式未改善均衡目标，不继续对其计算额外抽样或调参。下一项只测120日上升趋势的线性连贯程度，使用现有价格路径，不补EPS。")
    next_path = ROOT / "docs/510300_TREND_COHERENCE_BLEND_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300继续预测幅度预算_第144轮_20260909/继续预测预算_结果及全部中文规则.md",
        "按保存继续持有预测幅度分配预算", "CLOSED_EARLY_LOCAL_GAIN_MAIN_AND_MINIMUM_FOUR_SHARPE_WORSE_THAN143",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "TREND_COHERENCE_BLEND_IMPLEMENTATION", "用120日上升对数财富直线的拟合优度，在131与143之间形成新预算")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=145, registered=False, planned_settings=1, planned_new_accounts=4)
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
