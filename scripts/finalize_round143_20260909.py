"""交付连续预算的四项局部增益并保留不确定性和历史选择限制。"""
import json
from research.trend_noise_reference_blend_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第143轮成为当前四情景均衡比较的新候选，完整目标仍未实现。主基础／压力夏普1.291／1.232，"
        "年化4.42%／4.20%，最大回撤2.68%／2.82%；较早夏普1.022／1.044，年化5.25%／5.46%，最大回撤4.20%／4.33%。"
        "四项夏普、年化收益均超过139，四整段年化收益均高于买入持有，最低夏普由0.979提高到1.022。"
        "主回撤较139有所增大，较早夏普仍不足1.2，不能宣布稳定超额或独立验证通过。")
    detail = ("五项必要测试7.18秒，四账户核心1.720817秒，零模型拟合、零新参考和下载；核心时间不含开发、测试、核对及文档。"
        "用直接历史窗口和等价幅度比例独立核对5646个判断、四账户66个完整实际周期。主各23周期、301持股收盘、66次成交；"
        "较早各10周期、302／308持股收盘、37次成交。所有目标已知、无受阻、终点全部清仓。"
        "131平均预算主16.19%、较早21.84%，最高分别62.01%和71.97%；这是策略预算，不是实际总持仓。主实际平均股票敞口约6.15%，较早约11.34%／11.71%。"
        "相对139，主基础价格与分红多7699.90元、费用多583.95元、净多7115.95元；压力净多6545.31元。"
        "较早基础净多3207.84元、压力净多1762.85元。2015、2016关键经济账本与139逐项完全相同；"
        "较早改善主要来自2017多4365.79／4261.69元，2018小幅减少，2019少1109.20／2432.24元。因此不是所有阶段都改善。"
        "主23周期中9个亏损，前三盈利周期占总净利59.38%／61.49%；较早10周期中3个亏损，前三占92.22%／94.99%。"
        "主7个统计年度只有3个超额为正，2026为不足一年的片段；较早5年只有3个超额为正。"
        "沿用20／60日循环区块、2000次抽样与原种子，仅分析保存收益。20日区块的主基础／压力夏普95%区间约0.483至1.902／0.421至1.844，"
        "较早约0.144至1.985／0.145至2.006。两种区块长度下，对139的夏普增量与相对买入持有的年化超额区间全部跨零。"
        "这些敏感性不修复多次历史研究的选择偏差，局部点值增益未获独立证实。"
        "下一项只复用128已保存的继续持有预测幅度，先核对时钟、模型身份及无模型状态，再作一个新的预算组合。")
    next_path = ROOT / "docs/510300_CONTINUATION_STRENGTH_BLEND_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300趋势波动连续预算_第143轮_20260909/连续预算_结果及全部中文规则.md",
        "按正趋势与波动幅度连续分配来源", "COMPLETED_FOUR_SHARPE_AND_RETURN_LOCAL_GAINS_EARLY_BELOW12_NO_INDEPENDENT_PROOF",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "CONTINUATION_STRENGTH_BLEND_INPUT_REVIEW", "复用128同费用已保存的继续持有预测幅度，在131与143之间形成新的连续预算")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    primary = {m["cost"]: m for m in result["primary"]}
    early = {m["cost"]: m for m in result["earlier_diagnostics"] if m["model"] == "TREND_NOISE_REFERENCE_BLEND"}
    index["previous_balanced_candidate_before_round143"] = index["current_best_four_scenario_comparison_candidate"]
    index["current_best_four_scenario_comparison_candidate"] = {
        "study": result["study_id"], "model": "TREND_NOISE_REFERENCE_BLEND",
        "status": "POST_SELECTED_FOUR_SHARPE_RETURN_GAIN_EARLY_BELOW12_NOT_INDEPENDENTLY_VALIDATED",
        "source_result": str((OUT / "result.json").relative_to(ROOT)),
        "comparison_basis": "SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO143",
        "base_main_sharpe": primary["BASE"]["net_sharpe"], "stress_main_sharpe": primary["STRESS"]["net_sharpe"],
        "base_earlier_sharpe": early["BASE"]["net_sharpe"], "stress_earlier_sharpe": early["STRESS"]["net_sharpe"],
        "minimum_four_scenario_sharpe": min(m["net_sharpe"] for m in [*primary.values(), *early.values()]),
        "four_full_period_excess_returns_positive": all(m["annualized_return_excess_vs_buy_hold"] > 0 for m in [*primary.values(), *early.values()]),
        "every_year_stable_excess_established": False, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "four_sharpe_and_cagr_improve_vs139": True, "main_drawdown_larger_than139": True,
        "saved_sensitivity": str((OUT / "saved_diagnostic_receipt.json").relative_to(ROOT))}
    index["next_work"].update(candidate_round=144, registered=False, planned_settings=1, planned_new_accounts=4)
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
