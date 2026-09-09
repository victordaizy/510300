"""交付直接共同下行的小幅改善，并保存完整目标未完成的状态。"""
import json
from research.joint_downside_reference_pair_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第137轮四项净夏普均比136小幅改善，完整目标仍未实现。主基础／压力夏普1.282／1.227，年化3.99%／3.81%，"
        "最大回撤2.37%／2.47%，整段年化超额0.37／0.20个百分点；较早夏普0.777／0.801，年化4.63%／4.88%，"
        "最大回撤8.09%／8.21%，整段超额0.73／1.01个百分点。"
        "主两费用点值超过1.2，较早仍不足；四情景最低0.776745仍低于均衡比较132的0.788877，不能称四时期均衡最优。"
        "保留对136的局部改善，历史及费用差异照实展示，不把反复观察的区间或选择后敏感性认作独立验证。")
    detail = ("六项必要合成测试3.74秒，四条新账户核心1.8251784秒，零新模型、零新参考和下载；核心秒数不含开发、测试、核对、文档。"
        "独立核对采用按合成日收益过零点分段求二次函数最小值，与运行时的凸函数导数二分相互核对，确认真正先合成收益再取下行。"
        "核对5646个判断、四账户和76个完整周期。主两费用各80次月首尝试、22次预算改变、22个实际周期、264个持股收盘、56次成交；"
        "较早各60次尝试、20次预算改变、16个实际周期、基础328与压力334个持股收盘、各47次成交。"
        "主295个正目标原点不等于264个实际持股收盘，整手和终点仍影响参与。"
        "相对136，主基础净利润少368.63元、压力少289.39元，夏普分别增加0.00697和0.00782，属于收益略降、波动降幅更大的改善。"
        "较早基础净利多2036.51元，其中价格收益多1781.30元、费用少255.21元；压力净利多2112.63元，其中价格收益多1665.50元、费用少447.13元。"
        "\n\n保存收益敏感性沿用20日和60日循环区块、每档2000次及固定种子。主基础夏普95%区间分别0.498—1.897、0.632—1.845，"
        "主压力0.426—1.853、0.562—1.791，仍包含1.2以下水平。主相对136的夏普增量区间跨零，"
        "基础20日区块为−0.110至0.122、压力−0.110至0.122；整段超额区间也跨零。它是选择后敏感性，不修复多次搜索。"
        "最大三个盈利周期占主净利61.34%／63.31%，较早110.49%／113.95%；较早其余周期合计抵消部分利润。"
        "较早2015与136账户相同，当年仍使用预设各半预算，新增改善发生于其他年份；不能据此把预算初始化认定为失败原因，也不能事后跳过2015。"
        "最新研究状态现在可以通过scripts/read_510300_fast_status.py直接读取必要字段，避免再次展开全部历史索引。")
    next_path = ROOT / "docs/510300_FINAL_TARGET_VOLATILITY_CAP_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300合成收益下行预算_第137轮_20260909/合成下行预算_结果及全部中文规则.md",
        "直接最小化合成收益的下行平方", "COMPLETED_FOUR_SHARPE_LOCAL_GAIN_MAIN_BOTH_ABOVE12_EARLY_NOT_TARGET",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "FINAL_TARGET_VOLATILITY_CAP_PLANNED", "只在最终合成目标超过既有普通波动上限时缩小，复用风险序列及完整资金账户")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=138, registered=False, planned_settings=1, planned_new_accounts=4)
    index["additional_joint_downside_main_sharpe_target_candidate"] = {
        "study": "510300_JOINT_DOWNSIDE_REFERENCE_PAIR_V1", "model": "JOINT_DOWNSIDE_REFERENCE_PAIR",
        "status": "FOUR_LOCAL_SHARPE_GAINS_VS136_MAIN_BOTH_POINT_TARGET_EARLY_FAIL_NOT_INDEPENDENT",
        "source_result": str((OUT / "result.json").relative_to(ROOT)),
        "base_main_sharpe": 1.2822686297888573, "stress_main_sharpe": 1.2265293512278383,
        "base_earlier_sharpe": .7767449527427255, "stress_earlier_sharpe": .8005969225243525,
        "minimum_four_scenario_sharpe": .7767449527427255, "four_full_period_excess_returns_positive": True,
        "every_year_stable_excess_established": False, "independent_validation": "NOT_ESTABLISHED", "goal_achieved": False,
        "saved_sensitivity": str((OUT / "saved_diagnostic_receipt.json").relative_to(ROOT))}
    index["fast_status_entry"] = "scripts/read_510300_fast_status.py"
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
