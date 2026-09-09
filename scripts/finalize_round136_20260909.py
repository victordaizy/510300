"""保留主历史两档费用超过1.2的候选，完整披露早期和稳定性限制。"""
import json
from research.covariance_reference_pair_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第136轮主历史两档费用净夏普均超过1.2，完整目标仍未实现。主基础／压力夏普1.275／1.219，"
        "年化4.02%／3.82%，最大回撤2.51%／2.66%，整段年化超额0.39／0.22个百分点。"
        "较早夏普0.731／0.755，年化4.46%／4.71%，最大回撤8.09%／8.21%，整段超额0.56／0.84个百分点。"
        "四整段超额均正，但较早两夏普均不足1.2，且低于固定各半134；四情景最低0.730529低于均衡比较132的0.788877。"
        "保留主历史风险收益改善，不把选择后的历史点值当作独立证明。")
    detail = ("六项必要测试4.39秒；四条新账户核心3.6289418秒，零新模型、零新参考及下载；核心时间不含开发、测试、核对、文档。"
        "独立从基础费用账本复算月度方差、协方差、收益差方差和预算，核对5646个判断、四账户及77个完整周期。"
        "主各80次月首尝试、22次预算改变、22个实际周期、273个持股收盘和58次成交；较早各60次尝试、22次预算改变，"
        "基础17周期331持股收盘52成交，压力16周期336持股收盘52成交。主304个正目标原点不等于273个实际持股收盘，整手和终点会影响实际参与。"
        "主基础比91多赚10695.07元、夏普提高0.042；比134少赚4896.46元，却因波动进一步降低，夏普提高0.162。"
        "主基础算术年化3.987%、波动3.126%，134对应4.294%和3.856%。较早基础比134少赚4414.22元、压力少1114.35元。"
        "压力较早优于基础源于对应费用参考退出状态和实际资金路径不同，不能解释为增加费用改善同一成交路径。"
        "\n\n为核实局部改善，复用既有20日和60日循环区块、每档2000次及原随机种子，另外登记并只抽样保存收益。"
        "主基础夏普95%敏感性区间分别0.469—1.898和0.616—1.835，压力分别0.404—1.848和0.552—1.778；都包含1.2以下水平。"
        "相对固定各半及91的夏普差区间都跨零，整段年化超额区间也跨零。此选择后抽样不修复多次搜索，不构成独立验证。"
        "主最大的三个盈利周期占总净利61.13%／63.15%，较早114.87%／118.16%，意味着其他周期合计抵消部分利润。"
        "主统计的7个年度区间仅3个正超额，2026只统计至8月14日；较早5年仅3年正超额，不能称逐年稳定。"
        "较早2015账户和134相同；2016与2017基础净利差分别−2544.67、−9590.48元，2018与2019分别改善2333.48、5387.45元，说明已有风险权重的阶段得失，不能按这些事后年份选赢家。")
    next_path = ROOT / "docs/510300_JOINT_DOWNSIDE_REFERENCE_PAIR_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300共同波动月度预算_第136轮_20260909/共同波动预算_结果及全部中文规则.md",
        "两套参考组合的月度共同风险预算", "COMPLETED_MAIN_BOTH_COSTS_SHARPE12_WITH_EXCESS_EARLY_FAIL_NO_INDEPENDENT_PROOF",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "JOINT_DOWNSIDE_REFERENCE_PAIR_PLANNED", "直接最小化两参考合成后的负收益平方，保持既有日历和实际进出规则")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=137, registered=False, planned_settings=1, planned_new_accounts=4)
    index["additional_covariance_main_sharpe_target_candidate"] = {
        "study": "510300_COVARIANCE_REFERENCE_PAIR_V1", "model": "COVARIANCE_REFERENCE_PAIR",
        "status": "MAIN_BOTH_COSTS_POINT_TARGET_AND_EXCESS_EARLY_FAIL_NOT_INDEPENDENTLY_VALIDATED",
        "source_result": str((OUT / "result.json").relative_to(ROOT)),
        "base_main_sharpe": 1.2753029879737618, "stress_main_sharpe": 1.2187097534828781,
        "base_earlier_sharpe": .730529053707908, "stress_earlier_sharpe": .7546560595274266,
        "minimum_four_scenario_sharpe": .730529053707908, "four_full_period_excess_returns_positive": True,
        "every_year_stable_excess_established": False, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "saved_sensitivity": str((OUT / "saved_diagnostic_receipt.json").relative_to(ROOT))}
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
