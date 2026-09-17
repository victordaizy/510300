"""关闭未改善结果的正均值预算，交付完整规则并准备成熟周期方案。"""
import json
from research.runs_net_profit_budget_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    main_rows = [r for r in result['all_metrics'] if r['model'] == PRIMARY]
    early_rows = [r for r in result['earlier_diagnostics'] if r['model'] == PRIMARY]
    require(all(r['net_sharpe'] >= 1.2 for r in main_rows) and all(r['net_sharpe'] < 1.2 for r in early_rows), '正均值预算夏普结论不同')
    require(all(r['annualized_return_excess_vs_buy_hold'] < 0 for r in main_rows+early_rows), '正均值预算超额结论不同')
    decision = ('第169轮过去平均净收益正值的月度预算完成。主基础／压力净夏普1.275／1.211，较早0.846／0.806；'
        '年化收益主2.89%／2.74%、较早3.64%／3.45%，四场景均低于买入持有。主夏普数值通过1.2，'
        '但较早更弱且没有整段正超额，未改善143、166、167及168的比较结果，关闭这一固定方法，完整目标未完成。')
    detail = ('六项必要测试3.94秒通过，四个新账户核心计算2.184997秒，不含开发、测试、核对及交付。'
        '零新模型训练、零新增参考账户，六套保存对照两段两费用共二十四条，加四个新账户，共二十八条指标记录。\n\n'
        '独立从父净资产重算每日净收益，用独立均值计算核对142条预算和初始化记录，其中116条使用完整二百四十二日窗口。'
        '两项预算最大误差为一点一一乘十的负十六次方，11292条父目标、5646个新决定、73段完整持仓和210次开盘成交通过核对。'
        '全部实际资金、费用、分红和终点退出一致；这属于计算实现核对，不是收益目标的独立样本验证。\n\n'
        '主80次月首尝试中，12次窗口不足保留预算、68次有正均值并分配预算，没有双零现金预算。'
        '较早60次中，12次窗口不足、41次正均值分配、7次双非正均值转现金，共138个判断日为双零预算。'
        '主平均143／165预算约50.26%／49.74%，较早约65.11%／23.57%，较早其余为现金预算。\n\n'
        '主每账户26段持仓、384个持仓收盘、68次成交。较早基础10段、370个持仓收盘、36次成交；'
        '压力11段、371个持仓收盘、38次成交。四账户均没有未知目标或受阻未成交请求，终点完整清仓。'
        '不同费用造成实际净资产、整手和后续持仓不同，因此较早周期数允许不同。\n\n'
        '主基础价格加分红利润44335.00元、费用2696.50元、净赚41638.50元；压力分别44115.90、4916.24、39199.66元。'
        '较早基础分别40987.70、1572.90、39414.80元；压力分别40811.90、3540.10、37271.80元。'
        '主最大回撤2.13%／2.22%，较早3.98%／4.33%；平均实际股票市值占净资产主约5.89%，较早10.40%／10.47%。\n\n'
        '相对买入持有，主年化落后0.731／0.871个百分点，较早落后0.261／0.419个百分点。'
        '相对143，四场景夏普均更低；主终值少24763.57／23550.18元，较早少19399.67／24088.06元。'
        '过去日历窗口盈利并没有在本试验中转化为更好的下一阶段分配，不能将后验盈利选择称为已证明有效的策略切换。\n\n'
        '保留143作为原均衡比较基准、167的较高主段夏普记录及168相加封顶的四场景正超额记录。'
        '下一项更换观察单位，仅用已经完成并且分红收益已经确认的持仓周期，检验其盈亏效率是否有助于分配。'
        '事后保存的周期分红不能在成熟前读取，具体时点边界已写入下一项事前规则。')
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        PRIMARY: 'CLOSED_EARLIER_SHARPE_BELOW12_ALL_FOUR_EXCESSES_NEGATIVE_AND_BELOW_REFERENCE'}, exclusive=True)
    next_path = ROOT/'docs/510300_RUNS_CLOSED_CYCLE_BUDGET_NEXT_20260910.md'
    document = ROOT/'deliverables/510300月度净收益预算_第169轮_20260910/月度净收益预算_结果及全部中文规则.md'
    delivery = deliver_round(ROOT, OUT, CONFIG, document, '每月按过去完整净收益正值分配预算',
        'CLOSED_RUNS_NET_PROFIT_BUDGET_FULL_GOAL_NOT_MET', decision, detail, next_path,
        next_path.read_text(encoding='utf-8').splitlines(), 'RUNS_CLOSED_CYCLE_BUDGET_FINITE_CANDIDATE_PREPARED',
        '每月按最近二十个已成熟实际持仓周期的盈亏效率分配两套预算，分别至少五个')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=170, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        planned_saved_control_models=['RUNS_NET_PROFIT_BUDGET', 'RUNS_OPPORTUNITY_CAPPED_SUM', 'RUNS_COVARIANCE_BUDGET',
            'RUNS_REFERENCE_BLEND', 'TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE', 'BUY_HOLD'],
        planned_main_metric_rows=16, planned_earlier_metric_rows=16)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=169,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO169')
    index['latest_main_sharpe_pass_candidate'].update(last_compared_completed_round=169,
        selection_note='继续保留167作为主历史两费用较高夏普比较记录；169主段虽通过数值线但四场景超额均为负')
    index['latest_all_four_scenario_excess_candidate']['last_compared_completed_round'] = 169
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
