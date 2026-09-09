"""交付双候选批量结果，保留全部方向和完整目标未达状态。"""
import json
from research.monotone_episode_budget_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json, now


def main():
    decision = ("第150轮两个固定方向均结束，现有均衡候选仍是143。预算只减不增的主历史基础／压力夏普提高到1.328／1.276，"
        "年化4.43%／4.25%，最大回撤2.39%／2.50%；但较早夏普降到0.790／0.798，年化3.12%／3.17%，低于买入持有。"
        "预算只增不减的主夏普0.980／0.929、较早0.984／1.009，四项均低于143。不能仅凭主历史点估计提高宣布完成。")
    detail = ("本轮共用一次数据读取和每费用一组对照读取，两个候选八条新账户核心计算2.129732秒；五项必要测试3.53秒。"
        "零新增模型、参考账户和外部行情下载，核心秒数不含开发、测试、核对及文档。新通用批量流程可供后续有限候选复用，旧冻结运行器不修改。"
        "独立按父明确零分组，再用组内累计最小和最大值核对11292个判断、八条资金账户、132个完整实际周期，完整公布两方向。"
        "两方向主各23个父信号段，较早各10段；目标为正、零和未知的日历均与143一致。只减方案主各301个持仓收盘，成交57／58次；较早302／308个持仓收盘、各23次成交。"
        "只增方案主各311个持仓收盘、各52次成交；较早302／308个收盘、各29次成交。目标日历相同仍可能因净值、带宽和整手规则产生不同实际路径。"
        "主只减基础相对143，价格利润增加296.70元、分红减少695.20元、费用节省531.43元，净增仅132.93元；压力净增746.37元。"
        "因此主夏普的改善同时包含风险降低和费用变化，不是大量新增收益。较早只减却净少25319.10／27365.53元，降低预算错过的价格利润远大于省下的费用。"
        "只增在较早净多2717.26／3082.32元，但波动同步增加，夏普仍降低；主净少10842.87／10620.54元。"
        "所有八账户无未知目标或受阻请求，终点清仓。已保存完整逐日账本、分年指标、账户损益差额及下面全部中文因子和进出规则。"
        "这两种单向机制均不继续改参数救回；下一轮改为比较同一平均K线状态的独立交易与143确认，继续使用批量流程。")
    next_path = ROOT / "docs/510300_HEIKIN_PRICE_STATE_NEXT_20260909.md"
    write_json(OUT / "candidate_outcomes.json", {"recorded_at": now(), "goal_achieved": False,
        "EPISODE_BUDGET_NONINCREASING": "CLOSED_MAIN_GAIN_EARLY_SHARPE_AND_EXCESS_FAILURE",
        "EPISODE_BUDGET_NONDECREASING": "CLOSED_FOUR_SHARPES_BELOW143"}, exclusive=True)
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300单向持仓预算比较_第150轮_20260909/两种单向预算_结果及全部中文规则.md",
        "两种信号段内单向预算一次批量比较", "CLOSED_BOTH_MONOTONE_DIRECTIONS_FULL_GOAL_NOT_MET",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "HEIKIN_PRICE_STATE_INPUT_PREPARATION", "用真实日线构造含分红平均K线，下一批比较独立进出场和143状态确认")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=151, registered=False, planned_settings=2, planned_new_accounts=8)
    index["current_best_four_scenario_comparison_candidate"].update(last_compared_completed_round=150,
        comparison_basis="SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO150")
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
