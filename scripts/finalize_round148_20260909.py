"""交付同状态模型更新否决的零主增益和较早损失。"""
import json
from research.model_update_veto_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第148轮结束，不替换143：主历史没有触发新增退出，基础／压力夏普仍为1.291／1.232，"
        "所核对的经济账本与143逐项完全一致，不能算新的独立成功。较早夏普降至0.997／0.977，年化5.04%／4.94%，"
        "最大回撤4.19%／4.33%。四整段年化收益仍高于买入持有，但较早两项夏普和收益均低于143，完整目标未实现。")
    detail = ("六项必要测试7.55秒，四账户核心3.426424秒，零模型拟合、新参考和下载；核心时间不含开发、测试、核对和文档。"
        "本轮实际生成648个最新版预测，逐项用同一原128八因素的标量标准化、裁剪和乘积求和独立核对；又核对5646个判断、四账户66个完整实际周期。"
        "主每档生成261个最新预测，其中只有7个参数身份与固定版不同，均未触发指定负面分歧。主仍为23周期301持仓收盘66成交。"
        "较早基础生成62个预测、43个身份不同，触发一次否决；压力64个预测、45个身份不同，触发两次否决。"
        "原204个无固定模型收盘中，只有父目标正且未否决的199个需要处理，均保留无学习状态而没有补零预测。"
        "基础在2017年7月11日收盘触发，下一开盘7月12日请求退出，锁定至8月11日父目标归零，共23个正父目标收盘被否决。"
        "压力在2017年7月13日触发、7月14日请求退出，锁定21个正父目标收盘；另于2019年6月21日触发、6月24日请求退出，锁定7个收盘至7月2日父归零。"
        "触发前分别比较同一持仓状态，没有借用另一条实际参考路径的预测。所有否决后来已按父零解除，无未知目标或受阻，终点全退。"
        "较早仍各10实际周期，持仓收盘减少至279／280，成交各36次。相对143，基础价格利润少2554.40元，仅省9.66元费用，净少2544.74元；"
        "压力价格利润少6365.80元，仅省12.57元费用，净少6353.23元。分红差额均为零。结果没有支持这些预测分歧具有可利用的退出优势。"
        "不增加确认天数或分歧幅度门槛救回本公式。下一项按当时近期三段最弱增长表现估计月度来源预算，直接检验事前条件分配。")
    next_path = ROOT / "docs/510300_ROBUST_BLOCK_GROWTH_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300模型更新分歧退出_第148轮_20260909/模型更新否决_结果及全部中文规则.md",
        "同一持仓状态的固定版与最新版模型分歧退出", "CLOSED_MAIN_ECONOMIC_PATH_IDENTICAL143_EARLY_VETO_LOSS",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "ROBUST_BLOCK_GROWTH_IMPLEMENTATION", "每月用已有242日来源净收益的三个连续子阶段，选择最弱对数增长更好的131与143预算")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=149, registered=False, planned_settings=1, planned_new_accounts=4)
    index["current_best_four_scenario_comparison_candidate"].update(last_compared_completed_round=148,
        comparison_basis="SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO148")
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
