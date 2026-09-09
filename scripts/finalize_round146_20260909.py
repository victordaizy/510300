"""交付高低区间和隔夜风险的减仓结果，保留收益不足结论。"""
import json
from research.range_overnight_risk_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import write_json


def main():
    decision = ("第146轮四项回撤都缩小，但四项夏普低于143、四整段年化收益低于买入持有，结束这一固定风险乘数。"
        "主基础／压力夏普1.128／1.068，年化2.51%／2.37%，最大回撤1.95%／2.17%；"
        "较早夏普1.008／1.035，年化2.78%／2.92%，最大回撤2.51%／2.60%。"
        "四情景最低夏普从143的1.022降至1.008；风险下降没有形成更好的风险调整收益，不能完成稳定超额和夏普1.2目标。")
    detail = ("五项必要测试5.02秒，四账户核心1.463927秒，零模型拟合、新参考及下载；核心时间不含开发、测试、核对和文档。"
        "使用直接逐窗口样本方差及高低区间平方还原风险，核对5646个判断和四账户66个实际完整周期。"
        "主各23周期306持仓收盘65成交；较早各10周期302／308持仓收盘39成交；无未知或受阻，终点清仓。"
        "四条正目标日历与143完全相同，但实际份额和带宽路径不同，主实际持仓收盘从301增至306，因此不能称实际买卖时点完全不变。"
        "主风险乘数平均66.45%、最小17.18%，1604次判断中1519次小于一；较早平均62.21%、最小11.09%，1219次中1051次小于一。"
        "实际股票平均敞口主约4.09%、较早6.54%／6.76%，明显低于143的主约6.15%及较早11.34%／11.71%。"
        "主基础相对143少价格及分红毛收益32260.70元，节省佣金和滑点1622.78元，净少30637.92元；压力净少29095.68元。"
        "较早基础毛收益少30145.90元，仅省919.80元费用，净少29226.10元；压力净少30152.72元。"
        "原143的父目标已包含风险预算，本轮继续减仓后盈利也同步明显减少；这说明本次额外风险缩减无增益，不等于高低价数据无价值。"
        "不再对失败的固定公式做额外抽样或调参，下一项改变新持仓机会的接受条件，保留原退出规则。")
    next_path = ROOT / "docs/510300_EPISODE_TREND_ADMISSION_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300区间隔夜风险_第146轮_20260909/区间隔夜风险_结果及全部中文规则.md",
        "日内高低区间和隔夜风险约束仓位", "CLOSED_FOUR_SHARPE_BELOW143_FOUR_CAGR_BELOW_BUY_HOLD",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "EPISODE_TREND_ADMISSION_IMPLEMENTATION", "用现有200日财富均线，在143每段新正目标开始时决定整段进入资格")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=147, registered=False, planned_settings=1, planned_new_accounts=4)
    index["current_best_four_scenario_comparison_candidate"].update(last_compared_completed_round=146,
        comparison_basis="SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO146")
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
