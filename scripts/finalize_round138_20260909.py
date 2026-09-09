"""结束最终目标上限，区分仅夏普最低值与同时需要超额的候选。"""
import json
from research.final_target_volatility_cap_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第138轮最终仓位上限未满足目标，结束此设定。主基础／压力夏普1.093／1.040，年化2.96%／2.81%，"
        "最大回撤2.37%／2.41%；较早夏普0.789／0.802，年化3.86%／3.97%，最大回撤5.68%／5.76%。"
        "较早回撤缩小，夏普较137只提高0.0123／0.0012；主夏普下降0.1894／0.1862，四项年化收益都低于137。"
        "四项中三项整段年化收益低于买入持有，不能为了微小的最低夏普改善忽略超额目标。"
        "四情景最低0.789084略高132的0.788877，仅高0.000207；它是仅按最低夏普的比较，不是同时满足收益目标的新最优。")
    detail = ("六项必要测试3.91秒，四条新账户核心1.5890421秒，零新模型、零新参考和下载；核心时间不含开发、测试、核对、文档。"
        "独立按超过上限才替换目标核对5646个判断、四账户和76个完整周期。四条实际持股日历及周期起止与137完全相同，"
        "主各22周期264持股收盘55次成交，较早各16周期、328／334持股收盘、46／47次成交。"
        "主各29个原点触及上限，实际股数却在245／233行不同，初期仓位及资金变化会影响后续整手，不能只按29天解释总收益。"
        "较早169／175个原点触及上限，实际全部328／334个持股收盘的股数不同。"
        "主基础价格分红少17327.00元、费用少689.45元、净少16637.55元；压力净少15928.23元。"
        "较早基础价格分红少9994.90元、费用少852.54元、净少9142.36元；压力净少10860.88元。"
        "这次改变的是规模，既有进入退出时点没有改善。主基础算术年化收益由3.964%降至2.953%，波动由3.092%降至2.702%，收益降幅超过风险降幅。"
        "旧109、131的普通波动设定及本轮结果保持，不改20日、10%、来源或年份挽救。"
        "下一项根据当时已经存在的退出模型训练支持选择131或137，明确检查模型15:05时点；不按事后年份或实际未来表现决定选择。")
    next_path = ROOT / "docs/510300_MODEL_SUPPORT_REFERENCE_ROUTER_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300最终仓位波动上限_第138轮_20260909/最终上限_结果及全部中文规则.md",
        "只在合成最终目标超过普通波动上限时缩小", "COMPLETED_FINAL_CAP_MAIN_WEAKER_THREE_EXCESS_NEGATIVE_CLOSED",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "MODEL_SUPPORT_REFERENCE_ROUTER_PLANNED", "使用当时已形成的模型训练支持状态，在131与137原收盘目标之间选择")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=139, registered=False, planned_settings=1, planned_new_accounts=4)
    index["minimum_four_sharpe_only_cap_comparison"] = {
        "study": "510300_FINAL_TARGET_VOLATILITY_CAP_V1", "model": "FINAL_TARGET_VOLATILITY_CAP", "minimum_four_scenario_sharpe": .7890842964980687,
        "base_main_sharpe": 1.0928391779627953, "stress_main_sharpe": 1.040285232201544,
        "base_earlier_sharpe": .7890842964980687, "stress_earlier_sharpe": .8017834992397782,
        "status": "ONLY_MINIMUM_SHARPE_MARGINALLY_HIGHER_THAN132_THREE_EXCESS_NEGATIVE_NOT_ECONOMIC_PROMOTION",
        "four_full_period_excess_returns_positive": False, "full_period_negative_excess_count": 3, "goal_achieved": False,
        "source_result": str((OUT / "result.json").relative_to(ROOT))}
    index["current_best_four_scenario_comparison_candidate"]["comparison_scope_note"] = (
        "132保留为四整段年化超额均正的均衡比较；138仅按最低夏普更高0.000207，但三项超额负，另列minimum_four_sharpe_only_cap_comparison。")
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
