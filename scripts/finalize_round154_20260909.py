"""关闭边界再平衡规则，交付完整中文结果并准备中位数斜率规则。"""
import json
from research.boundary_rebalance_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    require((OUT / "saved_timing_explanation.json").is_file(), "实际退出时点差异还未写入解释")
    decision = ("第154轮边界再平衡结束。主基础／压力净夏普1.264／1.209，较早0.967／0.995；"
        "年化收益主4.36%／4.15%、较早4.64%／4.88%。四项净夏普和年化收益都低于143，完整目标未实现，关闭这一固定边界机制。143仍为比较候选。")
    detail = ("六项必要测试4.90秒；一套规则四账户核心1.636844秒，零监督训练、零新参考和外部行情下载。"
        "核心时间不含实现、测试、保存核对或交付。原143每段每费用的收盘目标逐行相同，"
        "新账户仅在已有持仓越界时调至最近边界；完整真实成交、权益、分红、费用及终点仍按自身资金记账。"
        "已独立核对5646个自身净值请求、66个实际周期和274次开盘成交。另有152个请求与同一当前账户状态下的中心算法不同，"
        "这是动作对照，不能算152个新策略或额外中心回测。"
        "\n\n主基础／压力均为23个周期、311个持仓收盘，分别72／73次成交；原143各301个持仓收盘、66次成交。"
        "较早各10个周期、302／308个持仓收盘，分别64／65次成交；原143为37／37次。"
        "小额交易更多，但成交量和总费用较低。主较143分别少付费用342.02／666.25元，较早少付125.46／413.92元；"
        "价格利润加分红却分别少1426.00／1519.10元及7552.10／7502.30元，节约抵不过利润损失。"
        "终值相对143主少1083.98／852.85元，较早少7426.64／7088.38元。"
        "\n\n进场日期全部相同，实际退出日期并非全部相同。2021年1月5日和2月3日进入的两笔，"
        "原143分别1月14日、2月19日退出，新规则分别1月21日、2月26日退出，每笔晚五个交易日。"
        "原因是当时父目标虽仍为极小正数，原中心规则按整百份取整得到零；边界规则仍留下约10%仓位，直到父目标明确为零才卖完。"
        "两档费用共四条周期变化只涉及两笔日期，不是四份独立证据。周期净利差还包含此前调仓差异，不能全部归为延迟退出的因果影响。"
        "较早两档费用的所有进入退出日期相同。"
        "\n\n主净利65318.09／61896.98元，较早51387.83／54271.48元；四整段年化收益均高于买入持有，"
        "但四项都落后143，较早夏普也未达到1.2。该结果不触发稳定或独立有效的认定。"
        "下一项只用现有价格计算六十日全部成对斜率的中位数，检验独立方向和明确进出场，不再调整失败边界比例。")
    write_json(OUT / "candidate_outcomes.json", {"recorded_at": now(), "goal_achieved": False,
        "BOUNDARY_REBALANCE": "CLOSED_FOUR_SHARPE_AND_CAGR_BELOW143_COST_SAVING_SMALLER_THAN_GROSS_PROFIT_LOSS"}, exclusive=True)
    next_path = ROOT / "docs/510300_MEDIAN_SLOPE_RISK_NEXT_20260909.md"
    delivered = deliver_round(ROOT, OUT, CONFIG,
        ROOT / "deliverables/510300边界再平衡_第154轮_20260909/边界再平衡_结果及全部中文规则.md",
        "同一目标区间外调整至最近边界", "CLOSED_BOUNDARY_REBALANCE_FULL_GOAL_NOT_MET",
        decision, detail, next_path, next_path.read_text(encoding="utf-8").splitlines(),
        "MEDIAN_SLOPE_RISK_FINITE_CANDIDATE_PREPARED", "六十日全部成对斜率中位数决定方向，普通波动决定规模")
    path = ROOT / "reports/research/510300_sharpe_1_2_latest_research.json"
    index = json.loads(path.read_text(encoding="utf-8"))
    index["next_work"].update(candidate_round=155, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    index["current_best_four_scenario_comparison_candidate"].update(last_compared_completed_round=154,
        comparison_basis="SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO154")
    write_json(path, index)
    print(json.dumps(delivered, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
