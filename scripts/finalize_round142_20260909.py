"""关闭共同持仓约束，说明缩小回撤与保留收益的区别。"""
import json
from research.consensus_reference_target_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第142轮两套目标共同确认未达到目标，结束该固定交集设置。主基础／压力夏普1.005／0.957，"
        "年化2.57%／2.44%，最大回撤2.06%／2.23%；较早夏普0.900／0.923，年化4.37%／4.53%，最大回撤3.95%／4.03%。"
        "虽然四项回撤都比139小，四项夏普和年化收益却都更低，主两档费用的年化收益低于买入持有。"
        "139继续是当前四情景均衡候选；不能靠收缩风险宣布达到完整目标。")
    detail = ("四项必要测试3.55秒，四条新增账户核心1.466048秒；零拟合、零新参考、零下载，核心时间不含开发、测试、核对及文档。"
        "独立以两个明确目标的条件比较还原5646个判断，核对四账户60个完整实际周期。主各21周期、226个持股收盘、50次成交；"
        "较早各9周期、276／282个持股收盘、32／33次成交。无未知目标、无受阻请求，终点清仓。"
        "主257个原点两套都正，57个仅131正、38个仅139正、1252个都零；较早276／282个都正、26个仅139正，没有仅131正。"
        "这些是目标原点，不能与成交持股日数混用。交集同时删除单方机会及压低部分共同持仓规模，总损益差不能全部解释为被删除日期。"
        "相对139，主基础毛价格分红少23761.50元、费用只省1077.91元、净少22683.59元；压力净少21574.50元。"
        "较早基础毛少8097.60元、费用省540.50元、净少7557.10元；压力净少9536.11元。"
        "因此共同同意没有带来更好的风险收益比；两来源本来也不独立。明确失败后不再重复bootstrap。"
        "下一项检验按正趋势幅度相对已有波动幅度连续分配131与139预算，保留两套原目标，不再作全部替换或全部排除。")
    next_path = ROOT / "docs/510300_TREND_NOISE_REFERENCE_BLEND_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300共同目标确认_第142轮_20260909/共同目标_结果及全部中文规则.md",
        "两套明确正目标共同确认持仓", "COMPLETED_SMALLER_DRAWDOWNS_BUT_FOUR_RETURNS_AND_SHARPE_WEAKER_CLOSED",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "TREND_NOISE_REFERENCE_BLEND_PLANNED", "按正趋势幅度相对波动幅度连续分配131和139预算，合成最终目标")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=143, registered=False, planned_settings=1, planned_new_accounts=4)
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
