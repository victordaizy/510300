"""保留下行风险的薄弱情景改善，交付两设置及下一项进入退出条件。"""
import json
from research.downside_reference_risk_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第132轮下行风险主要候选改善了最弱情景，仍未达到夏普1.2。主基础／压力净夏普0.842／0.789，"
        "年化5.10%／4.76%，最大回撤7.71%／7.92%；较早净夏普0.931／0.954，年化6.38%／6.66%，最大回撤8.02%／8.24%。"
        "四情景年化收益均高于买入持有，超额分别为1.48、1.15、2.48、2.79个百分点，但四个整段超额为正不等于逐年稳定超额。"
        "四情景最低夏普0.789，高于截至130轮标准下一开盘候选最佳128的0.766，也高于131和本轮构造对照，成为新的均衡比较候选。"
        "全部平方幅度对照主0.733／0.673、早1.022／1.043，未通过目标；其主要情景收益低于买入持有，结束这一固定对照设置。"
        "主要候选保留局部改善，不改本轮参数，也没有取得独立验证。")
    detail = ("两个设置共八条新账户，六项必要合成测试3.83秒，完整账户核心1.9594593秒，零新模型、零新参考、零新下载。"
        "核心时间不含开发、测试、保存核对及文档。独立逐窗口求和核对两套各3456日风险，核对11292个实际判断、八条账户和116个完整周期；"
        "全部份额、费用、净值和绩效可由保存结果还原。八条持股日期与128一致，改变规模但没有新增退出日期。"
        "主要候选主每档20周期、313个持股收盘、59次成交；早每档9周期、276／282个持股收盘、38次成交。"
        "主要候选相对131，主基础／压力净盈利多22885.35／21956.91元，净夏普提高0.096／0.102；"
        "较早净盈利多10870.92／12872.52元，但净夏普下降0.115／0.107，说明更高收益同时带来更高风险。"
        "相对同样不减收益均值的全部平方幅度对照，主要候选主基础净夏普提高0.109、压力提高0.116，"
        "较早则下降0.091／0.090。因此方向性风险定义在主历史有局部作用，并非所有阶段都更好。"
        "下一项将利用这两个已有风险量提出明确的参与和退出条件，检验能否改善进入退出时机，而不重复计算现有来源和模型。")
    next_path = ROOT / "docs/510300_DOWNSIDE_BALANCE_GATE_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300下行风险与平方幅度比较_第132轮_20260909/下行风险比较_结果及全部中文规则.md",
        "下行风险与全部平方幅度比较", "COMPLETED_DOWNSIDE_WORST_SCENARIO_GAIN_NOT_TARGET",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "DOWNSIDE_BALANCE_ENTRY_EXIT_GATE_PLANNED", "保持132风险规模，检验下跌平方幅度占优时退出的单一条件组合")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=133, registered=False, planned_settings=1, planned_new_accounts=4)
    index["current_best_four_scenario_comparison_candidate"] = {
        "study": "510300_DOWNSIDE_REFERENCE_RISK_V1", "model": PRIMARY,
        "status": "POST_SELECTED_WORST_SCENARIO_GAIN_NOT_INDEPENDENTLY_VALIDATED",
        "source_result": str((OUT / "result.json").relative_to(ROOT)),
        "comparison_basis": "SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_ROUNDS131_AND132",
        "base_main_sharpe": .8418517897015924, "stress_main_sharpe": .7888771896511613,
        "base_earlier_sharpe": .9305789082229796, "stress_earlier_sharpe": .9538729798857059,
        "minimum_four_scenario_sharpe": .7888771896511613, "four_full_period_excess_returns_positive": True,
        "every_year_stable_excess_established": False, "goal_achieved": False}
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
