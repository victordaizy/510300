"""交付第155轮完整失败结果，接续有限的稀疏退出学习方案。"""
import json
from research.median_slope_risk_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, write_json


def main():
    decision = ("第155轮六十日成对斜率中位数结束。主基础／压力净夏普−0.230／−0.270，较早0.539／0.502；"
        "年化收益主−2.29%／−2.63%、较早4.40%／4.07%。四项夏普和年化收益都低于143，主区间净亏，完整目标未实现，关闭这套固定趋势规则。")
    detail = ("六项必要测试17.75秒，一套规则四账户核心7.145318秒；核心时间不含开发、测试、核对和交付。"
        "3397个完整60日窗口的斜率已通过SciPy 1.18.0独立统计函数核对，只计算一次完整历史，较早前缀和两档费用复用。"
        "已由原始未复权收盘和分红重建财富，并逐个20日窗口复算普通波动；5646个收盘判断、52个实际周期、340次真实开盘成交及全部权益费用已核对。"
        "两档费用共用完全相同的因素和目标，实际资金账户各自成交。"
        "\n\n主各17个周期、841个持仓收盘、102次成交；较早各9个周期、737个持仓收盘、68次成交。"
        "主841个正目标、763个零目标，其中1个斜率恰好为零；较早738个正目标、481个零目标，其中2个斜率恰好为零。"
        "两个研究区间没有未知目标。早期正目标数比实际持仓收盘多一，终点开盘清仓优先，不能把正目标数量直接当作实际持仓日数。"
        "\n\n主基础价格利润加分红已经亏23395.60元，佣金滑点再付5101.43元，净亏28497.03元；"
        "压力对应价格加分红亏23162.80元、费用9177.53元、净亏32340.33元。费用并非主区间失败的唯一原因。"
        "较早基础价格加分红51843.30元、费用3363.00元、净48480.30元；压力对应50913.30元、6436.26元、44477.04元。"
        "主最大回撤31.31%／32.24%，较早15.16%／15.32%，均高于143。"
        "\n\n主年化收益低于买入持有5.92／6.23个百分点；较早仅高0.51／0.20个百分点。"
        "相对143终值主少94899.10／95090.16元，较早少10334.17／16882.83元。"
        "趋势估计对单个异常价格较稳健，不意味着相应交易方向有净收益优势；较早和主区间差异也不能用于事后按年份切换。"
        "下一项保留已有完整持仓样本，检验固定成本尺度的绝对值惩罚能否自动减少无用退出因素，避免继续添加独立趋势指标。")
    write_json(OUT / "candidate_outcomes.json", {"recorded_at": now(), "goal_achieved": False,
        "MEDIAN_SLOPE_RISK": "CLOSED_MAIN_GROSS_AND_NET_LOSS_FOUR_SHARPE_AND_CAGR_BELOW143"}, exclusive=True)
    next_path = ROOT / "docs/510300_SPARSE_VINTAGE_EXIT_NEXT_20260909.md"
    delivery = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300成对斜率中位数_第155轮_20260909/成对斜率中位数_结果及全部中文规则.md",
        "六十日成对斜率中位数与普通波动预算", "CLOSED_MEDIAN_SLOPE_RISK_FULL_GOAL_NOT_MET",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "SPARSE_VINTAGE_EXIT_FINITE_CANDIDATE_PREPARED", "保留原八因素与完整成熟周期，固定绝对值惩罚使弱因素系数可为零")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=156, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=25, planned_new_reference_accounts=0, external_data_required=False)
    index["current_best_four_scenario_comparison_candidate"].update(last_compared_completed_round=155,
        comparison_basis="SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO155")
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
