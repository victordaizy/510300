"""关闭成熟周期预算并交付中文规则，下一项改用独立每日市场状态。"""
import json
import pandas as pd
from research.runs_closed_cycle_budget_v1 import ROOT, OUT, CONFIG, PRIMARY
from research.fast_round_delivery_v1 import deliver_round
from research.intraday_overnight_increment_v1 import now, require, write_json


def main():
    result = json.loads((OUT/'result.json').read_text(encoding='utf-8'))
    main_rows = [r for r in result['all_metrics'] if r['model'] == PRIMARY]
    early_rows = [r for r in result['earlier_diagnostics'] if r['model'] == PRIMARY]
    require(all(r['net_sharpe'] >= 1.2 for r in main_rows) and all(r['net_sharpe'] < 1.2 for r in early_rows), '成熟周期预算夏普结论不同')
    require(sum(r['annualized_return_excess_vs_buy_hold'] > 0 for r in main_rows+early_rows) == 1, '成熟周期预算超额场景数不同')
    differences = pd.read_csv(OUT/'saved_comparison_differences.csv')
    require(differences[differences.comparison.eq('RUNS_REFERENCE_BLEND')].sharpe_difference.lt(0).all(), '相对固定各半夏普结论不同')
    decision = ('第170轮成熟持仓周期盈亏效率的月度预算完成。主基础／压力净夏普1.565／1.500，较早0.990／0.989；'
        '年化主3.67%／3.51%、较早3.82%／3.84%。主夏普达到1.2，但较早仍不足，只有主基础年化略超过买入持有。'
        '四场景夏普均低于第166轮固定各半组合，完整目标未完成，关闭这套固定成熟周期方法。')
    detail = ('七项必要测试7.80秒通过，四个新账户核心计算4.632899秒，不含开发、测试、核对与交付。'
        '零新增模型训练、零新增参考账户。七套保存对照两段两费用共二十八条，加四新账户共三十二条指标记录。\n\n'
        '独立从父账户实际买卖重算50个周期的买入支出、费用、归属分红和净收益，核对每个成熟日期。'
        '这50个基础费用父周期中，主143有23个、165有8个；较早143有10个、165有9个。'
        '实际资料中没有卖出之后才除息的归属分红周期，延迟成熟边界已经在必要测试中验证，不把潜在问题说成本次实际发生的错误。\n\n'
        '142条月度预算和初始化记录、1637条周期入选记录通过核对，预算最大误差为四点四四乘十的负十六次方。'
        '另核对11292条父收盘目标、5646个新决定、92段新账户实际持仓和241次开盘成交，全部资金、费用、分红及终点退出一致。'
        '核对脚本首次读取时仍带有上一方法的日历窗口列名，已改为本轮实际字段后完成核对；策略规则、冻结源码和账户结果没有修改或重跑。'
        '这些独立计算核对不等于独立历史样本已证明收益目标。\n\n'
        '主80次月首中，53次不足两父各五个成熟周期而保留预算，27次有效更新，第一次更新为2024年6月3日。'
        '较早60次中，37次样本不足、23次有效更新，第一次更新为2018年2月1日。两段没有双零现金预算。'
        '这说明周期样本积累速度较慢，策略长期仍接近初始各半；它没有实现原先希望的及时阶段切换。\n\n'
        '主平均143／165预算约51.10%／48.90%，较早约46.01%／53.99%。主各30段实际持仓、443个持仓收盘和71次成交；'
        '较早各16段，基础448个持仓收盘、49次成交，压力454个持仓收盘、50次成交。'
        '四账户没有未知目标或受阻未成交请求，终点全部清仓。\n\n'
        '主基础价格加分红利润56863.90元、费用2855.80元、净赚54008.10元；压力分别56556.90、5204.26、51352.64元。'
        '较早基础分别43894.00、2331.37、41562.63元；压力分别46282.60、4528.98、41753.62元。'
        '较早压力净利润略高，来自对应费用父目标、实际净资产和整手路径不同，不能解释为提高费用本身创造了收益。\n\n'
        '主最大回撤1.48%／1.56%，较早3.55%／3.72%；平均实际股票市值占净资产主约6.37%，较早11.10%／11.27%。'
        '相对买入持有，主基础年化领先0.047个百分点、主压力落后0.100个百分点，较早落后0.077／0.034个百分点。'
        '相对固定各半，主终值少1219.41／1163.66元、较早少3163.84／2715.60元，四场景夏普均降低。\n\n'
        '143的原均衡比较记录、167的较高主段夏普记录及168相加封顶的正超额记录保留。'
        '下一轮转为独立每日价格状态，直接检验相邻收益的相关方向与累计方向，不等待父策略完成持仓周期。')
    write_json(OUT/'candidate_outcomes.json', {'recorded_at': now(), 'goal_achieved': False,
        PRIMARY: 'CLOSED_EARLIER_SHARPE_BELOW12_THREE_EXCESSES_NONPOSITIVE_AND_BELOW_EQUAL_BLEND'}, exclusive=True)
    next_path = ROOT/'docs/510300_RETURN_LAG_STATE_NEXT_20260910.md'
    document = ROOT/'deliverables/510300成熟周期预算_第170轮_20260910/成熟周期预算_结果及全部中文规则.md'
    delivery = deliver_round(ROOT, OUT, CONFIG, document, '按已经成熟的实际持仓周期盈亏效率分配预算',
        'CLOSED_RUNS_CLOSED_CYCLE_BUDGET_FULL_GOAL_NOT_MET', decision, detail, next_path,
        next_path.read_text(encoding='utf-8').splitlines(), 'RETURN_LAG_STATE_FINITE_CANDIDATE_PREPARED',
        '六十日相邻收益配对相关及累计方向，两日进入和分别两日退出的独立市场状态')
    path = ROOT/'reports/research/510300_sharpe_1_2_latest_research.json'
    index = json.loads(path.read_text(encoding='utf-8'))
    index['next_work'].update(candidate_round=171, registered=False, planned_settings=1, planned_new_accounts=4,
        planned_new_model_fits=0, planned_new_reference_accounts=0, external_data_required=False,
        planned_saved_control_models=['TREND_NOISE_REFERENCE_BLEND', 'RETURN_RUNS_STATE', 'BUY_HOLD'],
        planned_main_metric_rows=8, planned_earlier_metric_rows=8)
    index['current_best_four_scenario_comparison_candidate'].update(last_compared_completed_round=170,
        comparison_basis='SAVED_STANDARD_NEXT_OPEN_FRONTIER_THROUGH130_PLUS_COMPLETED_ROUNDS131_TO170')
    index['latest_main_sharpe_pass_candidate'].update(last_compared_completed_round=170,
        selection_note='保留167作为主历史两费用较高夏普比较记录；170主段通过但较早不足，四场景夏普低于166')
    index['latest_all_four_scenario_excess_candidate']['last_compared_completed_round'] = 170
    write_json(path, index)
    print(json.dumps(delivery, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
