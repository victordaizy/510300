"""交付训练支持选择，更新四情景比较并保留原历史判断。"""
import json
from research.model_support_reference_router_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第139轮成为当前四情景均衡比较的新候选，但完整目标尚未实现。主历史基础／压力净夏普为1.282／1.227，"
        "年化收益3.99%／3.81%，最大回撤2.37%／2.47%；较早历史夏普提高到0.979／1.011，"
        "年化收益4.99%／5.31%，最大回撤4.23%／4.37%。四整段年化收益均高于买入持有，"
        "最低夏普从原132的0.789提高到0.979，但较早两情景仍低于1.2。主历史完整账户与137逐列相同，"
        "不能将其算作新的独立业绩证据。")
    detail = ("六项必要测试5.48秒，四条新账户核心2.353063秒，零新模型、零新参考及下载；核心时间不含开发、测试、核对、文档。"
        "按15:05前已形成的最近训练记录独立核对5646个判断、四账户和64个完整持仓周期。"
        "主各22周期264个持股收盘56次成交，较早各10周期、302／308个持股收盘、36次成交；没有未知目标或未成交请求，终点均清仓。"
        "较早历史只在2017年3月1日15:05切换一次：此前525个原点选131，此后694个原点选137；主历史1604个原点全部选137。"
        "因此本样本只是一次从训练准备进入成熟阶段的检验，不能声称已经验证反复识别牛熊环境。"
        "较早基础／压力夏普较137增加0.2025／0.2104，净利润增加4413.08／5351.97元；"
        "相对131夏普仍低0.0662／0.0496、净利润少6630.68／4337.05元。2017年相对131少赚16708.70／16371.85元，"
        "2018和2019年部分补回；不按这些年份更换策略。"
        "较早各10周期中3个亏损，前三盈利周期占总净利92.48%／95.48%，样本集中。"
        "复用主历史137的敏感性，只对新的较早账户作20／60日循环区块、2000次抽样。较早基础夏普95%区间分别约0.077至1.971、"
        "0.004至1.928，压力约0.077至1.980、0.011至1.936；超额收益及对137、131的夏普增量区间均跨零。"
        "这些是多次研究后的敏感性，未建立稳定超额或独立效力。较早每档费用仅3／5个年度收益高于买入持有。"
        "下一项固定使用已存在的120日趋势因素选择131或139，只生成四条新增实际账户。")
    next_path = ROOT / "docs/510300_TREND_REFERENCE_ROUTER_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300训练支持条件选择_第139轮_20260909/训练支持选择_结果及全部中文规则.md",
        "按当时模型训练支持选择已有策略", "COMPLETED_FOUR_SCENARIO_MINIMUM_IMPROVED_EARLY_BELOW12_NO_INDEPENDENT_PROOF",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "FINITE_TREND_REFERENCE_SELECTION_PLANNED", "用已存在120日趋势因素在131与139收盘目标之间选择")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    primary = {m["cost"]: m for m in result["primary"]}
    early = {m["cost"]: m for m in result["earlier_diagnostics"] if m["model"] == "MODEL_SUPPORT_REFERENCE_ROUTER"}
    index["previous_balanced_candidate_before_round139"] = index["current_best_four_scenario_comparison_candidate"]
    index["current_best_four_scenario_comparison_candidate"] = {
        "study": result["study_id"], "model": "MODEL_SUPPORT_REFERENCE_ROUTER",
        "status": "POST_SELECTED_FOUR_SCENARIO_MINIMUM_GAIN_EARLY_BELOW12_NOT_INDEPENDENTLY_VALIDATED",
        "source_result": str((OUT / "result.json").relative_to(ROOT)),
        "comparison_basis": "SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO139",
        "base_main_sharpe": primary["BASE"]["net_sharpe"], "stress_main_sharpe": primary["STRESS"]["net_sharpe"],
        "base_earlier_sharpe": early["BASE"]["net_sharpe"], "stress_earlier_sharpe": early["STRESS"]["net_sharpe"],
        "minimum_four_scenario_sharpe": min(m["net_sharpe"] for m in [*primary.values(), *early.values()]),
        "four_full_period_excess_returns_positive": all(m["annualized_return_excess_vs_buy_hold"] > 0 for m in [*primary.values(), *early.values()]),
        "every_year_stable_excess_established": False, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "main_accounts_exactly_same_as137": True, "earlier_support_transitions": 1,
        "saved_sensitivity": str((OUT / "saved_diagnostic_receipt.json").relative_to(ROOT))}
    index["next_work"].update(candidate_round=140, registered=False, planned_settings=1, planned_new_accounts=4)
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
