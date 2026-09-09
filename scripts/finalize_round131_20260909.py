"""交付普通波动组合的分阶段收益代价，并登记下一项有限比较。"""
import json
from research.vintage_reference_risk_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第131轮有较早历史的局部改善，但未达到完整目标。主基础／压力净夏普0.746／0.687，年化3.75%／3.44%，"
        "最大回撤7.30%／7.49%；较早净夏普1.045／1.061，年化5.53%／5.66%，最大回撤3.95%／4.03%。"
        "主压力年化比买入持有低0.17个百分点，四种情景仍均低于夏普1.2。保留较早历史的风险收益改善作比较，"
        "结束这套无条件普通波动组合，不改原风险水平、窗口、带宽、费用配对或另挑参考挽救。")
    detail = ("七项合成测试5.14秒，四条新账户核心2.5627538秒，零新拟合、零新参考；核心时间不含开发、测试、核对和文档。"
        "已独立核对3456日旧风险乘数、5646个实际判断与自身净值请求、四条账户及58个完整周期。"
        "所有持股日期和58笔进入退出日期与128相同，改变的是自己的规模；主每档20周期、313个持股收盘，早每档9周期、276／282个持股收盘。"
        "主每档62次成交，早每档33次；主12个周期、早6个周期有多次增减仓。"
        "主基础年化算术平均收益保留128的54.06%，波动保留64.08%；压力对应52.97%和64.10%。"
        "较早基础对应60.68%和44.46%，压力60.68%和44.61%。因此较早的风险降低足以抵偿收益缩减，主历史则相反。"
        "主基础净收益比128少57280.28元，主压力少53712.53元，早基础／压力少42829.86／44322.59元。"
        "虽然成交次数增加，绝对佣金与滑点合计反而下降，不能把主历史变弱简单归因于费用增加。"
        "四情景最低夏普0.687，低于128的0.766；已有最均衡比较候选仍是128，它也没有完成1.2目标。"
        "本次是事后选择组合的历史比较，没有独立验证或新的交易权限。下一项比较下行平方风险与全部平方幅度，分开识别风险定义的影响。")
    next_path = ROOT / "docs/510300_DOWNSIDE_REFERENCE_RISK_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300固定参考与普通波动组合_第131轮_20260909/普通波动组合_结果及全部中文规则.md",
        "固定入场模型参考与普通波动仓位组合", "COMPLETED_RISK_COMBINATION_EARLY_GAIN_MAIN_WEAKER_NOT_TARGET",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "DOWNSIDE_AND_SECOND_MOMENT_REFERENCE_COMPARISON_PLANNED", "同一参考下检验下行风险与全部平方幅度，保留完整资金账户")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=132, registered=False, planned_settings=2, planned_new_accounts=8)
    index["additional_vintage_risk_early_comparison_candidate"] = {
        "study": "510300_VINTAGE_REFERENCE_RISK_V1", "model": PRIMARY,
        "status": "EARLY_RISK_ADJUSTED_GAIN_MAIN_WEAKER_NOT_INDEPENDENTLY_VALIDATED",
        "source_result": str((OUT / "result.json").relative_to(ROOT)),
        "base_main_sharpe": .7456255590756491, "stress_main_sharpe": .6870928761585842,
        "base_earlier_sharpe": 1.0453995527844013, "stress_earlier_sharpe": 1.0606321875522593,
        "goal_achieved": False}
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
