"""交付信号周期来源固定的真实成本取舍，结束此设置。"""
import json
from research.episode_trend_reference_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第141轮信号周期固定来源没有达到目标，结束此固定设置，139继续保留为均衡候选。"
        "主基础／压力夏普0.937／0.873，年化4.44%／4.13%，最大回撤6.95%／7.16%；"
        "较早夏普0.918／0.935，年化4.58%／4.72%，最大回撤4.15%／4.26%。"
        "四整段年化超额均正，但四项夏普都低于139。与140相比只在主压力夏普上增加0.0073，其他三项下降；"
        "不能把交易更少视为策略已改善。")
    detail = ("六项必要测试3.52秒，四条新增账户核心1.810392秒；零新模型、零新参考和下载，核心时间不含规则开发、测试、核对及交付。"
        "独立按所选来源下一次明确零目标分段，核对5646个判断、四账户及68个实际完整周期。"
        "主各24周期295持股收盘66次成交，较早各10周期、302／308持股收盘、34次成交。"
        "主信号周期各开始24个、由零目标结束23个，最后一个信号在研究终点仍为正，但实际账户已按统一终点开盘全部清仓；"
        "较早10个信号周期均明确结束。信号周期状态与实际账户持股严格区分。目标无未知、无未成交请求。"
        "与140相比主每档少10次成交、较早少6次。主基础费用少1971.87元、毛价格分红多675.50元、净多2647.37元；"
        "压力费用少3844.72元、毛多792.10元、净多4636.82元。主年化波动却约4.75%，高于140约4.47%，所以基础夏普仍下降。"
        "较早基础毛收益少18875.10元、费用只省863.57元、净少18011.53元；压力毛少18995.50元、费用省1589.76元、净少17405.74元。"
        "相对139，较早净少5010.48／7352.31元。结论是来源固定并未普遍保住有效机会，费用减少不足以弥补收益损失。"
        "明确失败后不重复bootstrap；下一项只检验131与139正目标的共同持仓约束，保留全部旧判断。")
    next_path = ROOT / "docs/510300_CONSENSUS_REFERENCE_TARGET_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300信号周期固定来源_第141轮_20260909/周期固定来源_结果及全部中文规则.md",
        "信号周期内固定趋势选择的来源", "COMPLETED_LOWER_TRADING_COSTS_NOT_ENOUGH_FOUR_SHARPE_BELOW139_CLOSED",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "CONSENSUS_REFERENCE_TARGET_PLANNED", "两套保存目标均为正才持仓，最终采用较小目标；任一明确退出则退出")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=142, registered=False, planned_settings=1, planned_new_accounts=4)
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
