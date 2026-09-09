"""关闭缩量累计趋势并交付中文规则，接续固定九次价格比较。"""
import json
from research.negative_volume_trend_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第129轮缩量日累计涨跌趋势未改善。主基础／压力净夏普0.074／0.048，年化0.22%／−0.06%，最大回撤19.80%／20.09%；"
        "较早净夏普−0.093／−0.112，年化−4.08%／−4.47%，最大回撤52.93%／53.40%。四账户收益均低于买入持有，未达夏普1.2。"
        "关闭这套固定加法缩量累计与255日均线策略，不改累计口径、均线周期、方向或加止损挽救。")
    detail = ("七项必要合成测试4.27秒，零拟合、零新参考，四账户及完整因子核心运行1.5560386秒，不含开发、测试、核对与文档。"
        "保留3456日原日历，1746个缩量日计入涨跌，3202日有完整均线；两个实际评价起点均已有足够历史。"
        "主每档14周期、576个持股收盘；较早每档13周期、530个持股收盘，无输入缺失或未成交。"
        "已用独立加法求和及指数权重展开核对3456个指标与3202个均线值，最大绝对差分别约3.41和3.87乘十的负十二次方，全部方向一致。"
        "共享保存核对模块检查5646个实际判断、四账户及54完整周期，没有重跑模拟。"
        "主基础价格与分红合计7591.00元，费用4621.27元后盈利2969.73元；压力相应7512.40元和8289.56元，净亏777.16元。"
        "较早基础价格与分红已经亏34605.20元，再付3230.83元，净亏37836.03元；压力净亏41135.21元。"
        "较早基础2015年1月5日到8月25日一笔净亏26001.54元，2015年8月至2016年1月及2016年1月至2月两笔继续亏损；"
        "固定累计趋势未及时避开这些下跌，不能以少交手续费解决，也不按事后年份追加规则。"
        "下一项仅完成公开九次准备阶段方法查重与进出场设计，尚未计算新九次指标或账户。")
    next_path = ROOT / "docs/510300_SETUP_NINE_REVERSAL_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300缩量累计趋势_第129轮_20260909/缩量累计趋势_结果及全部中文规则.md",
        "缩量日累计涨跌趋势", "COMPLETED_ADDITIVE_NEGATIVE_VOLUME_TREND_REJECTED_NOT_TARGET", decision, detail,
        next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "SETUP_NINE_FIXED_REVERSAL_METHOD_REVIEWED_IMPLEMENTATION_PENDING",
        "连续九次相对四日前弱势后进入、连续九次强势后退出，零拟合单一反弹假设")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=130, registered=False, new_models_or_accounts=0)
    index["fixed_signal_fast_workflow"] = "docs/510300_FAST_FIXED_SIGNAL_WORKFLOW_20260909.md"
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
