"""关闭涡旋方向规则并交付，准备固定宽度的边界再平衡机制。"""
import json
from research.vortex_risk_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, write_json


def main():
    decision = ("第153轮相邻高低价方向与普通波动预算结束。主基础／压力净夏普0.166／0.017，较早0.405／0.248；"
                "年化收益主0.99%／−0.16%、较早3.08%／1.74%，四项均低于同区间买入持有，完整目标未实现。143继续作为均衡比较候选。")
    detail = ("六项必要测试4.95秒；一套规则四账户核心1.761345秒，零模型拟合、零新参考账户和外部行情下载。"
        "核心时间不包含开发、测试、保存核对与文档。已用当前财富等价比例独立还原经济开高低收，逐个14日窗口累加两种移动和真实波幅，"
        "逐个20日窗口按原始含分红收益复算普通波动；两档费用目标完全相同，实际账户独立。"
        "5646个收盘目标、四个完整资金账户、290个实际周期与765次真实开盘成交已核对。"
        "主各79周期806个持仓收盘，基础212次成交、压力211次；较早各66周期699个持仓收盘、171次成交。"
        "主807个正目标、797个零目标，较早700个正目标、519个零目标；完整研究判断区间没有未知方向或波动，也没有方向恰好相等。"
        "主基础价格利润加分红31386.00元，佣金滑点17879.64元，净13506.36元；压力对应29949.30元与32091.35元，净亏2142.05元。"
        "较早基础价格加分红47682.50元、费用14625.85元、净33056.65元；压力对应45815.60元、27628.97元、净18186.63元。"
        "以上是保存路径分解，不能据此改变费用后重新验收。主回撤25.81%／28.80%，较早16.70%／17.65%，均高于143。"
        "相对143终值主少52895.71／64891.89元，较早少25757.82／43173.23元。频繁切换与费用侵蚀同时存在，"
        "本轮没有证据支持它能取代或确认原策略，不改窗口或方向继续救回。"
        "下一项回到当前143已持仓的调仓数量，保持原10个百分点宽度与进入退出信号，只检验越界后调至最近边界的独立执行机制。")
    write_json(OUT / "candidate_outcomes.json", {"recorded_at": now(), "goal_achieved": False,
        "VORTEX_RISK": "CLOSED_FOUR_LOW_SHARPE_NEGATIVE_CAGR_EXCESS_AND_HIGHER_DRAWDOWN_VS143"}, exclusive=True)
    next_path = ROOT / "docs/510300_BOUNDARY_REBALANCE_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300相邻高低价方向_第153轮_20260909/相邻高低价方向_结果及全部中文规则.md",
        "相邻高低价方向与普通波动预算", "CLOSED_VORTEX_RISK_FULL_GOAL_NOT_MET",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "BOUNDARY_REBALANCE_FINITE_CANDIDATE_PREPARED", "保持143信号和10个百分点宽度，检验已有持仓越界后仅调整至最近边界")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=154, registered=False, planned_settings=1, planned_new_accounts=4,
                              planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    index["current_best_four_scenario_comparison_candidate"].update(last_compared_completed_round=153,
        comparison_basis="SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO153")
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
