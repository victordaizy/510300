"""交付固定入场模型的局部改善，衔接零拟合缩量趋势检验。"""
import json
from research.entry_vintage_exit_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第128轮固定每笔实际入场首次收盘的原114模型版本，仅改善较早历史。主基础／压力净夏普0.884／0.831，年化6.97%／6.52%，最大回撤11.44%／11.68%；"
        "两个完整账户与原114逐列相同。较早净夏普0.766／0.780，年化8.74%／8.97%，最大回撤13.79%／13.92%，原114较早净夏普0.752／0.728。"
        "四项均未达1.2，保留为较早历史局部改善的比较候选，不能称稳定超额或独立验证通过。不改锁定时点、部分更新、模型混合或阈值挽救。")
    detail = ("六项必要测试5.00秒，四个新实际账户核心运行2.2248379秒，不含开发、测试、文档和保存结果核对。"
        "复用141个旧模型时点，其中114个已有模型、27个支持不足；零新拟合、零新参考。"
        "已核58笔实际周期的固定版本、1184个持仓状态、776个预测、四账户财富和24条分年增量；主两档逐列等同114。"
        "较早基础终值增加3136.53元，其中价格收益增加3150.70元，佣金和滑点增加14.17元；较早压力终值增加10570.47元，其中价格增加10601.80元、费用增加31.33元。"
        "两档2017年5月18日进入的周期，退出从7月17日延至7月21日；压力另有2019年5月31日进入的一笔从6月25日延至7月3日。"
        "压力费用表现高于基础有明确账户路径原因：2019年6月21日，两档用同一2019年5月6日模型，基础预测−0.0003661125、压力+0.0001126569，"
        "差异全部来自实际周期浮盈亏输入。下一交易日基础累计两次负预测触发退出，压力只有一次，于是随后退出日期不同。"
        "这是含费用状态反馈导致的少数交易差异，不是费用改善收益的普遍规律，也不是新增零成本回测。"
        "本轮历史改善集中于这些退出变化，缺少独立新样本。下一项采用已有日线的缩量累计涨跌趋势，不补EPS或其他慢来源。")
    next_path = ROOT / "docs/510300_NEGATIVE_VOLUME_TREND_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300入场模型版本固定退出_第128轮_20260909/入场模型版本固定退出_结果及全部中文规则.md",
        "每笔入场固定退出模型版本", "COMPLETED_EARLY_ONLY_GAIN_MAIN_IDENTICAL_NOT_TARGET", decision, detail,
        next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "NEGATIVE_VOLUME_ADDITIVE_TREND_METHOD_REVIEWED_IMPLEMENTATION_PENDING",
        "单一缩量日累计涨跌与255日指数均线，现有日线零拟合明确进出场")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=129, registered=False, new_models_or_accounts=0)
    index["additional_entry_vintage_comparison_candidate"] = {
        "study": "510300_ENTRY_VINTAGE_EXIT_V1", "model": "ENTRY_VINTAGE_EXIT",
        "status": "EARLY_ONLY_LOCAL_GAIN_MAIN_IDENTICAL_NOT_INDEPENDENTLY_VALIDATED",
        "source_result": "reports/research/510300_entry_vintage_exit_v1/result.json",
        "base_main_sharpe": .883877592355659, "stress_main_sharpe": .8314812504844292,
        "base_earlier_sharpe": .7660394266347471, "stress_earlier_sharpe": .7797482588132696,
        "changed_cycle_timing_rows": 3, "goal_achieved": False}
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
