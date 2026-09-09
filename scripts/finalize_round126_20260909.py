"""交付单一自适应比较结果，接续现有强弱因子的持仓内增量检验。"""
import json
from research.adaptive_regret_experts_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第126轮不同起点自适应策略组合没有达到目标。主基础／压力净夏普0.887／0.821，年化2.76%／2.55%，最大回撤4.20%／4.27%；"
        "较早净夏普0.588／0.558，年化3.05%／2.89%，最大回撤8.32%／8.36%。四个年化收益均低于同口径买入持有。"
        "相对第91轮，主历史夏普下降、较早历史提高；主基础略高于第114轮0.884，但主压力及较早两项都低于114，未形成一致改善。"
        "关闭本次不同起点自适应组合，不改损失变换、出生频率、先验、基础候选或交易带宽挽救。")
    detail = ("十一项必要测试4.13秒，3209次逐日更新和四个实际账户的核心计算3.55秒，不含开发、测试、文档和核对。"
        "160组各三条内部比较记录，最多480条活跃记录，共775710次内部更新；这些不算480个策略。"
        "已用直接势函数和按出生起点的累计差核对776190条保存权重记录，核对5646个实际判断、四账户和82个完整持仓周期。"
        "源缺口、数值失败、零权重备用处理和实际未成交均为零。主每账户262个持仓收盘、52次成交，较早339个持仓收盘、32次成交。"
        "2013年5月31日起完整学习前缀中，急跌、原学习退出、现金的平均预算分别约31.48%、37.25%、31.27%；"
        "这不是实际股票占比，原参考空仓时其预算也不要求买股票。主真实平均股票暴露约6.43%，较早约10.90%。"
        "主基础保存成交路径的价格与分红贡献43100.00元，佣金滑点3552.60元，净利39547.40元；较早基础分别34629.10、1938.02、32691.08元。"
        "这些是原保存路径分解，不是另跑零费用账户。核对中重复解压已经改为一次缓存，并修正了周期汇总函数的参数传递，最终核对通过；没有重跑策略或改写账户。"
        "下一项只拟在114退出模型中加入现有当前收盘日内隔夜强弱值，1461条原状态、141个模型时点和114个成熟月份输入均已核对；尚无新拟合、预测或账户。"
        "历史已经反复观察，完整目标与独立证据仍未达到。")
    next_path = ROOT / "docs/510300_SESSION_STRENGTH_WITHIN_NEXT_20260909.md"
    result = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300不同起点自适应策略组合_第126轮_20260909/不同起点自适应策略组合_结果及全部中文规则.md",
        "不同起点自适应策略组合", "COMPLETED_ADAPTIVE_REGRET_EXPERTS_NO_CONSISTENT_INCREMENT_NOT_TARGET", decision, detail,
        next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "EXISTING_D60_WITHIN_CYCLE_INPUT_PREFLIGHT_COMPLETE_IMPLEMENTATION_PENDING",
        "原114八因子持仓内退出加入当前六十日日内相对隔夜强弱，直接复用已保存因素")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=127, registered=False,
        preflight_receipt="reports/research/510300_session_strength_within_preflight_20260909/result.json", new_models_or_accounts=0)
    write_json(path, index)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
