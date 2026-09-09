"""关闭无效的新增方向退出条件，继续既有策略机会的固定组合。"""
import json
from research.downside_balance_gate_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第133轮未改善，关闭这个固定下行占优退出条件。主基础／压力净夏普0.776／0.693，年化4.17%／3.69%，最大回撤6.65%／7.18%；"
        "较早净夏普0.854／0.864，年化5.44%／5.61%，最大回撤9.43%／9.79%。四情景净夏普及年化收益全部低于132，"
        "较早回撤也更深。四段年化收益仍略高于买入持有，但未达到1.2，不能认为增加一个风险条件就更准确。"
        "保留132为已有均衡比较候选，不改本轮20日、相等边界、参考或重试规则救回。")
    detail = ("六项必要合成测试4.00秒，四条新账户核心1.0592329秒，零新模型、零新参考；核心时间不含开发、测试、保存核对和文档。"
        "独立用保存数值的精确分数比较上下两侧，核对5646个实际判断、四条完整账户及90个周期，自己净值请求、份额、费用和终点均匹配。"
        "主每档拒绝168个原正目标收盘，实际持股收盘从313减为146，完整周期从20增为29，成交从59增为66；"
        "较早拒绝65个正目标收盘，持股从276／282减为211／217，周期从9增为16，成交从38增为47。"
        "相对132，主基础价格与分红收益少13165.00元，佣金滑点多2810.37元，净少15975.37元；压力净少17881.15元。"
        "较早基础价格分红收益少10991.60元，费用多1008.64元，净少12000.24元；压力净少13470.86元。"
        "减少参与时间没有选择性地排除损失，新增进出又提高费用，这是本次四情景变差的直接账本证据。"
        "核对部分已抽成支持部分买卖的共用模块，后续复用，减少重复开发。下一项仅比较132与91的固定各半目标组合，仍重新计算自己的实际账户。")
    next_path = ROOT / "docs/510300_EQUAL_REFERENCE_PAIR_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300下行占优退出条件_第133轮_20260909/退出条件_结果及全部中文规则.md",
        "下行平方幅度占优时退出", "COMPLETED_DOWNSIDE_BALANCE_GATE_REJECTED_NOT_TARGET",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "EQUAL_SAVED_REFERENCE_PAIR_PLANNED", "固定各半组合132与91已保存收盘目标，检验不同仓位路径的互补")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=134, registered=False, planned_settings=1, planned_new_accounts=4)
    index["continuous_target_saved_checks"] = "research/saved_target_account_checks_v1.py"
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
