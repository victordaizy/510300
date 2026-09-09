"""交付线性趋势连贯预算的完整结果，保留143为当前比较候选。"""
import json
from research.trend_coherence_blend_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第145轮四项夏普及年化收益均低于143，结束本固定趋势连贯预算公式，不替换当前候选。主基础／压力夏普1.177／1.114，"
        "年化4.38%／4.13%，最大回撤3.81%／4.04%；较早夏普1.017／1.036，年化5.19%／5.36%，最大回撤3.95%／4.03%。"
        "较早回撤较小，但四情景最低夏普从143的1.022降至1.017，主回撤增加。四整段年化超额仍为正，完整夏普1.2、稳定超额和独立证据目标尚未实现。")
    detail = ("五项必要测试3.49秒，四账户核心1.485234秒；核心秒数不含开发、测试、核对和文档。没有新增收益预测模型训练、新参考账户或下载，"
        "但逐日计算了滚动线性趋势统计。独立使用带常数项最小二乘和残差平方和还原斜率与拟合优度，再核对5646个判断、四账户64个完整实际周期。"
        "主每档22周期、313个持仓收盘、70次成交；较早各10周期、302／308持仓收盘、40次成交。无未知目标或受阻请求，终点清仓。"
        "131平均预算主22.80%、较早34.51%，最高95.49%／94.80%；实际平均股票敞口为主7.26%、较早11.21%／11.54%，二者不同。"
        "相对143，主基础价格损益少3008.40元、分红多2890.50元、费用多521.73元，净少639.63元；压力净少1209.76元。"
        "较早基础价格少902.00元、费用省96.72元，净少805.28元；压力净少1242.84元。主波动上升而收益下降，因此主夏普降幅大于净收益降幅。"
        "高拟合优度仅描述历史路径的直线程度，不能当成上涨概率或模型有效性证明，统计背景见正文NIST说明。"
        "固定公式失败后不额外做收益抽样或调参。下一项使用现有日内高低区间加隔夜波动估计风险，在唯一143父目标上设置仓位乘数。")
    next_path = ROOT / "docs/510300_RANGE_OVERNIGHT_RISK_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300趋势连贯预算_第145轮_20260909/趋势连贯预算_结果及全部中文规则.md",
        "按上升趋势连贯程度分配来源", "CLOSED_FOUR_SHARPE_AND_CAGR_BELOW143_EARLY_DRAWDOWN_REDUCTION_INSUFFICIENT",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "RANGE_OVERNIGHT_RISK_INPUT_REVIEW", "以已有日内高低区间加分红修正隔夜风险，在143目标上设置仓位乘数")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=146, registered=False, planned_settings=1, planned_new_accounts=4)
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
