"""关闭两种平均K线规则，交付全部中文结果并转向保存状态归因。"""
import json
from research.heikin_price_state_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, write_json


def main():
    decision = ("第151轮两个候选均结束，继续保留143作为均衡比较对象。独立平均K线主历史基础／压力夏普0.169／−0.180，"
        "较早−0.237／−0.508；平均K线确认143的主夏普1.146／1.033，较早0.753／0.614。八项年化收益均低于同区间买入持有，完整目标未实现。")
    detail = ("六项必要测试5.75秒，两候选八新账户核心2.648623秒；零新模型、参考账户和外部行情下载。核心时间不含开发、测试、核对及文档。"
        "完整3456日输入已检查，经济收盘与原含分红财富一致。独立用当日财富等价比例和线性滤波核对平均K线，核对11292个目标、八条账户、887个完整实际周期，"
        "并逐项核对1818次成交使用真实开盘、0.001元价位、原滑点和佣金，没有用平均K线价格成交。"
        "独立策略主各210周期、814个持仓收盘、420次成交；较早各151周期、674个持仓收盘、302次成交。"
        "主基础价格利润加分红94311.30元，被75801.26元佣金滑点消耗，净18510.04元；压力对应85249.20元和121806.26元，净亏36557.06元。"
        "较早基础在保存实际路径上的价格加分红已经亏9392.30元，再付37699.60元费用，净亏47091.90元；压力净亏78208.12元。"
        "以上只是保存路径的损益分解，不是假设无费用后重新建账的收益，因此不能借此改变原费用进行验收。"
        "确认143使主各131个正父目标收盘被否决，较早109／111个；主实际周期从143的23增至44，较早从10增至38／39。"
        "确认账户主各100次成交、较早86／88次。四账户价格利润都低于143，费用都更高；主净少19964.36／21528.91元，较早少26631.20／34703.23元。"
        "全部八账户目标已知、终点清仓；独立满仓请求可能按现金和整手部分成交，已按真实成交记账，不将目标1误称成100%实际仓位。"
        "这组结果没有支持平均K线能识别更有利的进入退出时点，不再增加确认日数或平滑参数救回。下一步仅利用保存账本做事前市场状态的失败归因。")
    write_json(OUT / "candidate_outcomes.json", {"recorded_at": now(), "goal_achieved": False,
        "HEIKIN_PRICE_STATE": "CLOSED_WEAK_PRICE_PATH_AND_FREQUENT_TRADING_COST",
        "HEIKIN_CONFIRMED_REFERENCE": "CLOSED_FOUR_SHARPE_LOSSES_AND_HIGHER_COST_VS143"}, exclusive=True)
    next_path = ROOT / "docs/510300_SAVED_STATE_FAILURE_DIAGNOSTIC_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300平均K线状态_第151轮_20260909/平均K线两种策略_结果及全部中文规则.md",
        "平均K线独立交易与确认原策略的共同检验", "CLOSED_BOTH_HEIKIN_RULES_FULL_GOAL_NOT_MET",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "SAVED_STATE_FAILURE_DIAGNOSTIC", "先检验事前趋势与波动状态是否在两个时期一致解释当前策略与改动的损益差异，不新增账户")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(registered=False, diagnostic_only=True, planned_new_accounts=0, planned_new_models=0)
    index["current_best_four_scenario_comparison_candidate"].update(last_compared_completed_round=151,
        comparison_basis="SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO151")
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
