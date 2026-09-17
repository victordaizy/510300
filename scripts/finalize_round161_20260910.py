"""交付持续回撤双上限的四场景失败结果，接续明确的新排序信号。"""
import json
import pandas as pd
from research.drawdown_depth_risk_v1 import ROOT, OUT, CONFIG
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, write_json


def main():
    records = []
    cycles = pd.read_csv(OUT/'saved_actual_cycles.csv')
    for period in ['evaluation', 'earlier_diagnostic']:
        for cost in ['BASE', 'STRESS']:
            own = cycles[cycles.period.eq(period)&cycles.cost.eq(cost)]
            records.append({'period': period, 'cost': cost, 'complete_cycles': len(own),
                'at_most_two_close_cycles': int(own.held_closes.le(2).sum())})
    write_json(OUT/'saved_short_cycle_summary.json', {'recorded_at': now(), 'saved_only': True,
        'new_models': 0, 'new_accounts': 0, 'records': records}, exclusive=True)
    decision = ('第161轮趋势准入及普通波动、持续回撤双风险上限完成。主基础／压力净夏普0.029／−0.052，较早0.912／0.852；'
        '年化收益主−0.11%／−0.78%、较早7.68%／7.13%。四项夏普低于1.2和143，主账户净亏，完整目标未实现，关闭这套固定结构。')
    detail = ('六项必要测试5.79秒，零模型训练，一套规则四账户核心计算1.167185秒，不含开发、测试、结果核对和交付。'
        '独立由原始收盘和分红重建财富与普通波动，按逐窗口逐日循环复算3397个完整六十日回撤窗口；'
        '持续回撤最大数值差2.78乘十的负十七次方，趋势最大差4.45乘十的负十六次方。'
        '四账户5646个收盘决定、116段完整持仓、454次真实开盘成交以及权益、费用、分红均已核对。\n\n'
        '主每账户38段持仓、862个持仓收盘、135次成交；较早每账户20段、730个持仓收盘、92次成交。'
        '主有15段、较早有9段只持有至多两个收盘，说明完整规则仍频繁在市场中进出。'
        '四账户均没有未知目标或受阻未成交请求，终点已经清仓。\n\n'
        '持续回撤上限在主54个判断日、较早58个判断日严格低于普通波动上限，确实改变了一部分目标。'
        '这些天数是目标受到限制的天数，不是对应新增成交次数，也不是独立获利证据。'
        '本轮没有另建删除回撤限制的完整账户，因此不能单独把全部损益差归因于这个因素。\n\n'
        '主基础价格损益−4155.30元、分红11331.90元、费用8630.87元，净亏1454.27元；压力分别−5276.80、10980.80、15811.63元，净亏10107.63元。'
        '主基础算术平均日收益略正，净夏普也略正，但复利累计收益为负，两者并不矛盾。'
        '较早基础价格利润88935.70元、分红7447.20元、费用6098.05元，净赚90284.85元；压力净赚82991.16元。'
        '主最大回撤27.13%／29.01%，较早10.16%／10.32%。\n\n'
        '相对143，主终值少67856.34／72857.47元，较早多31470.38／21631.30元，但四项夏普仍更低。'
        '主年化落后买入持有3.74／4.39个百分点，较早领先3.78／3.26个百分点。'
        '较早收益改善不能补偿主区间失败，也不能事后按年份分配新旧策略。下一项改检验日内相对隔夜差值的带符号排序，'
        '同时定义进入退出，保持现有免费来源和无需训练的快速流程。')
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        'DRAWDOWN_DEPTH_RISK': 'CLOSED_MAIN_NET_LOSS_FOUR_SHARPE_BELOW12_AND143'}, exclusive=True)
    next_path = ROOT/'docs/510300_SESSION_SIGNED_RANK_NEXT_20260910.md'
    delivery = deliver_round(ROOT, OUT, CONFIG,
        ROOT/'deliverables/510300持续回撤双上限_第161轮_20260910/持续回撤双上限_结果及全部中文规则.md',
        '一百二十日趋势准入及普通波动与持续回撤双上限', 'CLOSED_DRAWDOWN_DEPTH_RISK_FULL_GOAL_NOT_MET',
        decision, detail, next_path, next_path.read_text(encoding='utf-8').splitlines(),
        'SESSION_SIGNED_RANK_FINITE_CANDIDATE_PREPARED', '日内相对隔夜差值的带符号排序、连续进出场确认和普通波动仓位')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=162, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=161,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO161')
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
