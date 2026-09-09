"""关闭每日趋势选择，并交付收益与风险的实际取舍。"""
import json
from research.trend_reference_router_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第140轮每日趋势选择未满足目标，结束此固定设置，139继续保留为四情景均衡候选。"
        "本轮主基础／压力夏普0.959／0.866，年化4.28%／3.84%，最大回撤6.10%／6.53%；"
        "较早夏普1.128／1.133，年化6.03%／6.11%，最大回撤3.95%／4.03%。"
        "较早两项夏普比139提高0.1485／0.1219，但主下降0.3228／0.3605。四整段年化收益仍高于买入持有，"
        "四情景最低夏普却从139的0.979降到0.866，没有达到完整夏普、稳定超额或独立证据要求。")
    detail = ("六项必要测试4.16秒，四条新账户核心1.714047秒，零新模型、零新参考和下载；核心时间不含开发、测试、核对及文档。"
        "独立以逐日120个历史财富值的平均核对5646个判断、四账户和76个完整周期，账户价格损益、分红、佣金、滑点与最终净值对应。"
        "主各26周期264个持股收盘76次成交；较早各12周期、281／287个持股收盘、40次成交。所有目标明确，无未成交请求，终点清仓。"
        "主虽然与139同为264个持股收盘，具体日历不同，不能称进出完全相同。主发生75次来源选择变化，较早38次；它们不等于实际成交次数。"
        "主基础相对139净多4746.01元，其中价格损益少1857.90元、分红多10735.20元、费用多4131.29元；压力净仅多612.60元、费用多7863.10元。"
        "主年化算术收益由139的3.964%／3.784%升至4.290%／3.872%，年化波动却由3.092%／3.085%升至4.471%／4.470%。"
        "这是收益增加小于风险增加，不能全部归因于手续费。较早基础／压力净多13001.05／10053.43元；2015、2016账户经济差额为零，"
        "2017增加15972.30／15653.08元，2018小幅增加，2019减少4213.21／6370.59元。不能按这些年份选择使用哪套策略。"
        "本轮已有明确的主历史退化，因此不再重复进行耗时的不确定性抽样；较早点值改善未获独立验证。"
        "指标文件的展示名称沿用了模板的退出模型措辞，正确规则为本文件完整说明的120日趋势选择；模型标识、因素、冻结设置及实际目标正确。"
        "下一项固定信号周期内的来源选择，直接检验每日更换来源的代价，不调整均线窗口或方向。")
    next_path = ROOT / "docs/510300_EPISODE_TREND_REFERENCE_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300趋势条件选择_第140轮_20260909/趋势选择_结果及全部中文规则.md",
        "用已有120日趋势选择两套保存策略", "COMPLETED_EARLY_LOCAL_GAIN_MAIN_WEAKER_TREND_SELECTION_CLOSED",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "EPISODE_TREND_REFERENCE_SELECTION_PLANNED", "将趋势选择限制为信号周期开始一次，期间保留所选策略至明确零目标")
    document = ROOT / "deliverables/510300趋势条件选择_第140轮_20260909/趋势选择_结果及全部中文规则.md"
    document.write_text(document.read_text(encoding="utf-8").replace("按退出模型的趋势选择策略", "按120日趋势选择已有策略"), encoding="utf-8")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=141, registered=False, planned_settings=1, planned_new_accounts=4)
    index["additional_trend_router_early_local_comparison"] = {
        "study": "510300_TREND_REFERENCE_ROUTER_V1", "model": "TREND_REFERENCE_ROUTER",
        "source_result": str((OUT / "result.json").relative_to(ROOT)),
        "base_main_sharpe": .9594439950106831, "stress_main_sharpe": .8660770263641994,
        "base_earlier_sharpe": 1.1276685909638446, "stress_earlier_sharpe": 1.1329304614649898,
        "minimum_four_scenario_sharpe": .8660770263641994, "four_full_period_excess_returns_positive": True,
        "status": "EARLY_POINT_GAIN_MAIN_WEAKER_CLOSED_NOT_INDEPENDENTLY_VALIDATED", "goal_achieved": False}
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
