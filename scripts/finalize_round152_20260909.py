"""交付相关误差失败结果、完整中文规则，并保存下一项有限方向。"""
import json
from research.cycle_serial_error_exit_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, write_json


def main():
    decision = ("第152轮相关误差退出策略结束，完整目标未实现。主基础／压力净夏普0.859／0.823，较早0.751／0.727；"
                "四项均低于原128的0.884／0.831、0.766／0.780。原143继续作为均衡比较对象，不将本轮较高年化收益误报成高夏普达标。")
    detail = ("七项必要测试5.28秒；25种不同输入只拟合25次，89个成熟月份复用已经完成的同一输入参数。"
        "保留141个月度记录、114个成熟模型记录和27个不足月份，零求解失败、零新参考账户及外部行情下载。"
        "正式训练和四账户核心计算7.117759秒，不包含实现、测试、保存核对和文档时间。"
        "以完整相关矩阵和包含全部周期截距的联合方程独立复算25组参数，最大系数差5.551115123125783e-16；"
        "核对141个月度时钟、114个成熟缓存记录、58个实际完整周期、1171个持仓状态、763个可用预测、5646个全部收盘判断与116次真实开盘成交。"
        "四账户净值、分红权利、佣金、滑点、终点开盘及进入和全部退出条件已还原。"
        "主各20周期，持仓309／312个收盘，全有成熟预测；较早各9周期、275个持仓收盘，其中71个有预测、204个没有成熟模型，"
        "不能把不足月份当成零预测或后来补模型。主各14周期、较早各3周期最终由学习条件参与请求退出。"
        "相对128，主基础／压力终值少5565.50／1408.49元，较早少2797.72／10237.26元。"
        "对应价格利润少5436.40／1395.20元和2802.10／10249.00元；主分红再少249.70／63.30元，较早分红无差。"
        "费用各省120.60／50.01元和4.38／11.74元，无法补回价格和分红损失。"
        "四账户进入日期没有改变，六条账户周期记录的退出都提前：涉及2021年1月5日、2021年10月29日、2019年2月28日、2019年5月31日进入的四个日期。"
        "这些六条记录分属不同费用情景，不能叫六次独立证据；六条自身周期净利润均降低。"
        "新路径四段年化收益均超过买入持有，但净夏普均低于1.2且低于原128，说明本次统计误差修正没有转化为交易改善。"
        "关闭本条规则，不调相关边界、阶数、窗口或确认天数。下一步准备无需训练的相邻高低价方向与普通波动预算。")
    write_json(OUT / "candidate_outcomes.json", {"recorded_at": now(), "goal_achieved": False,
        "CYCLE_SERIAL_ERROR_EXIT": "CLOSED_FOUR_SHARPE_AND_TERMINAL_VALUE_LOSSES_VS128"}, exclusive=True)
    next_path = ROOT / "docs/510300_VORTEX_RISK_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300周期相邻误差退出_第152轮_20260909/相邻误差退出_结果及全部中文规则.md",
        "同一周期相邻误差修正后的固定版本退出", "CLOSED_SERIAL_ERROR_EXIT_FOUR_SCENARIO_LOSSES_VS128",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "VORTEX_RISK_FINITE_CANDIDATE_PREPARED", "用现有连续经济高低价方向和普通波动预算，准备一个无需模型训练的完整进出场设置")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=153, registered=False, planned_settings=1, planned_new_accounts=4,
                              planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    index["current_best_four_scenario_comparison_candidate"].update(last_compared_completed_round=152,
        comparison_basis="SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO152")
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
